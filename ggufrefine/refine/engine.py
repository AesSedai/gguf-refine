"""Batched GPU Hessian codeword refinement for stock IQ block formats.

FastRefiner.run(W, Hf, mult, code, nmat): coordinate descent over the codewords of E stacked problems [E, Rt, K]
against per-problem Hessians Hf [E, K, K], minimising tr((Q - W) Hf (Q - W)^T) with the scales fixed at their stock
values. Supported: IQ2_S, IQ3_S, the parity-sign types IQ2_XXS / IQ2_XS / IQ3_XXS, IQ1_S / IQ1_M. Model-independent:
callers supply W, H and the stock codes. The fp64 reference implementations are in tests/gpu/reference.py.

Execution
  * Matrices that share an input (e.g. gate and up of one expert) are one [nmat * R, K] problem against the shared
    Hf: rows are independent under the objective.
  * Visit order: one np.random.default_rng(seed).permutation of the items per sweep, cut into blocks of NB
    consecutive items. One Triton launch walks a block's items sequentially for every row (quant/kernels), keeping
    the gradient exact inside the block; the update of the other columns is deferred and applied as one batched GEMM
    per block (g -= D_blk @ Hf[C, :]). Every item sees the exact gradient, so blocking does not change the result.
  * One device->host sync per sweep (move counts + objective); a matrix stops early when a sweep accepts no move.

Items and moves
  IQ2_S / IQ3_S   one grid group; exhaustive over the signed codewords (IQ2_S: magnitude step, then sign step).
  parity types    one 8-weight sign chunk (7 stored sign bits + implied even parity): magnitude step per grid group,
                  then a sign step over the 128 legal patterns.
  IQ1_S / IQ1_M   one 8-weight group, values m * (grid[idx] + delta). IQ1_M searches grid x delta jointly; IQ1_S keeps
                  each 32-weight sub-block's shared delta and searches the grid. flip=True adds a per-sweep pass over
                  the sub-blocks that negates the shared delta and re-picks its 4 grid indices.

Acceptance: a move is taken when its fp32 score gain exceeds tol * |cur|. The accept gate (default on for the parity
and IQ1 types) rechecks each accepted move in the direct form delta = d A d - 2 d.g with a rounding bound and drops
it unless delta < -(margin + bound) * scale. ShadowCheck replays the moves of a few rows in float64 against an exactly
maintained gradient (tests). OzakiRefresh is an optional near-exact gradient refresh between sweeps.
"""
import numpy as np
import torch

from ..quant import codes as iq
from ..quant.kernels import iq as kn
from ..quant.kernels import parity as kp
from ..quant.kernels import iq1 as k1

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
KCFG = {"IQ3_S": dict(BR=64, BC=128, CH=32, warps=4, BSZ=128),
        "IQ2_S": dict(BR=32, BC=64, CH=32, warps=4, BSZ=128, prec="ieee"),
        "IQ2_XXS": dict(BR=32, BC=32, BCS=32, CH=32, warps=4, BSZ=128, prec="ieee"),   # logs/refine-fast-parity-tune.log
        "IQ2_XS": dict(BR=32, BC=32, BCS=32, CH=32, warps=4, BSZ=128, prec="ieee"),
        "IQ3_XXS": dict(BR=32, BC=32, BCS=32, CH=32, warps=4, BSZ=128, prec="ieee"),
        "IQ1_S": dict(BR=32, BC=32, CH=32, warps=4, BSZ=128, prec="ieee"),
        "IQ1_M": dict(BR=32, BC=32, CH=32, warps=4, BSZ=128, prec="ieee")}
GATED = iq.PARITY + iq.IQ1                                                    # types with the accept gate on by default


def fields_to_dev(t, f, R, K, dev=None):
    """iq.unpack fields of one [R, K] matrix -> (mult [R, NG] f32, code [R, NG] i32), on dev (numpy if dev is None)"""
    s = iq.SPEC[t]
    if t in iq.IQ1:                                                           # code = delta_neg << 11 | idx, one per 8-weight group
        mult = (f["d"][:, None] * iq.scale_mult(t, f["scales"])).astype(np.float32)
        mult = np.ascontiguousarray(np.repeat(mult, s["sub"] // 8, axis=1).reshape(R, K // 8))
        code = (f["idx"].astype(np.int32) | (f["delta"].astype(np.int32) << 11)).reshape(R, K // 8)
        if dev is None:
            return mult, code
        return (torch.from_numpy(mult).to(dev, non_blocking=True), torch.from_numpy(code).to(dev, non_blocking=True))
    gs = s["gsize"]
    sub = iq.scale_mult(t, f["scales"])
    mult = (f["d"][:, None] * sub).astype(np.float32)                         # [nb, nsub]
    mult = np.repeat(mult, s["sub"] // gs, axis=1).reshape(R, K // gs)
    sg = f["signs"].reshape(-1, gs).astype(np.int32)
    spat = (sg << np.arange(gs, dtype=np.int32)).sum(1).astype(np.int32)
    code = (f["idx"].astype(np.int32).reshape(-1) << gs) | spat
    if dev is None:
        return np.ascontiguousarray(mult), code.reshape(R, K // gs)
    return (torch.from_numpy(np.ascontiguousarray(mult)).to(dev, non_blocking=True),
            torch.from_numpy(code.reshape(R, K // gs)).to(dev, non_blocking=True))


def code_to_fields(t, code, f0):
    """device code [R, NG] -> fields dict (d/scales from the baseline f0)"""
    gs = iq.SPEC[t]["gsize"]
    c = (code.cpu().numpy() if torch.is_tensor(code) else np.asarray(code)).reshape(-1)
    out = dict(f0)
    if t in iq.IQ1:
        out["idx"] = (c & 2047).astype(np.uint16).reshape(f0["idx"].shape)
        out["delta"] = (c >> 11).astype(bool).reshape(f0["delta"].shape)
        return out
    out["idx"] = (c >> gs).astype(np.uint16).reshape(f0["idx"].shape)
    out["signs"] = (((c & ((1 << gs) - 1))[:, None] >> np.arange(gs)) & 1).astype(bool).reshape(f0["signs"].shape)
    return out


def _int8_slices(X, s):
    """X with |X| < 1 (fp32) -> s int8 tensors A_p with X = sum_p 2^(-7p) A_p + O(2^(-7s)), every step exact"""
    out, r = [], X
    for _ in range(s):
        r = r * 128.0
        a = torch.trunc(r)
        out.append(a.to(torch.int8))
        r = r - a
    return out


def _pow2_above(m):
    """smallest power of two strictly greater than m (1 where m == 0)"""
    return torch.where(m > 0, torch.exp2(torch.floor(torch.log2(m.clamp_min(1e-38))) + 1), torch.ones_like(m))


class OzakiRefresh:
    """g = R @ Hf with int8 tensor-core GEMMs (exact int32 accumulation) on 7-bit slices of the row-scaled R
    and column-scaled Hf (Ozaki scheme): every product A_p B_q is exact, products of one level p+q are summed
    exactly in int32 (|A_p B_q| <= K*127^2 = 5.8e7 for K <= 3584, <= s terms per level), levels are combined in
    fp64 and the result is rounded once to fp32. Error: truncation of the slices (<= 2^(-7s) * K * rowmax|R| *
    colmax|Hf| per entry, cross terms p+q > s+1 dropped) plus the final fp32 rounding (2^-24 |g|), against
    ~u * sum_k |r_k||Hf_kj| for the fp32 GEMM (refined experts have |g| << sum |r||Hf|).
    Hf slices are built once per run (s * E * K^2 bytes)."""

    def __init__(self, Hf, s=5):
        E, K, _ = Hf.shape
        assert K * 127 * 127 * s < 2 ** 31, K
        self.s = s
        self.tau = _pow2_above(Hf.abs().amax(1))                              # [E, K] per column
        Bs = _int8_slices(Hf / self.tau[:, None, :], s)
        self.B = [[b[e].t().contiguous().t() for b in Bs] for e in range(E)]   # column-major for torch._int_mm
        del Bs

    def __call__(self, R, out):
        s = self.s
        for e in range(R.shape[0]):
            sig = _pow2_above(R[e].abs().amax(1))                             # [Rt] per row
            A = _int8_slices(R[e] / sig[:, None], s)
            acc = None
            for lev in range(2, s + 2):
                S = None
                for p in range(1, lev):
                    q = lev - p
                    if p > s or q > s:
                        continue
                    t = torch._int_mm(A[p - 1], self.B[e][q - 1])
                    S = t if S is None else S.add_(t)
                term = S.double() * 2.0 ** (-7 * lev)
                acc = term if acc is None else acc.add_(term)
                del S, term
            out[e] = (acc * sig.double()[:, None] * self.tau[e].double()[None, :]).float()
            del A, acc
        return out


class ShadowCheck:
    """float64 replay of the fast refiner's moves on a subset of rows (tests / --shadow only).

    Keeps g64 = (W - Q) Hf64 exact (Hf64 = the fp32 Hf the kernel uses, widened, i.e. the objective
    the refiner minimises evaluated exactly) for the chosen rows. Around every block launch it reads
    the chunk values before the launch, after the magnitude step(s) (DV1, written by the DBG kernel)
    and after the sign step (Q), reconstructs each step's move d and computes its exact objective
    change d A d - 2 d.g64, then applies it to g64. Counts moves and uphill moves (delta > 0), and
    the fp32 gradient error the kernel saw (GB vs g64). At the end of each sweep the tracked
    objective must match a from-scratch float64 recompute (checks the replay itself)."""

    def __init__(self, W, Hf, rows, gpc):
        self.rows, self.gpc = rows, gpc
        self.W = W[:, rows].double()
        self.H = Hf.double()
        self.st = dict(steps=0, moved=0, uphill=0, uphill_max_rel=0.0, uphill_sum_rel=0.0, downhill_min_rel=0.0,
                       gerr_max_rel=0.0, replay_rel_err=[], gate_violations=0, margin_need=[],
                       gain_rel=[])

    def begin_sweep(self, Q):
        R = self.W - Q[:, self.rows].double()
        self.g = torch.bmm(R, self.H)
        new_obj = (R * self.g).sum(-1)
        if hasattr(self, "obj"):
            rel = float(((self.obj - new_obj).abs() / new_obj.abs().clamp_min(1e-300)).max())
            self.st["replay_rel_err"].append(rel)
        self.obj = new_obj                                                    # [E, nd]

    def before_block(self, Q, c, GB):
        self.v0 = Q[:, self.rows][:, :, c].double()
        gb = GB[:, self.rows].double()
        ge = self.g[:, :, c]
        rel = float(((gb - ge).abs().amax(-1) / ge.abs().amax(-1).clamp_min(1e-300)).max())
        self.st["gerr_max_rel"] = max(self.st["gerr_max_rel"], rel)

    def after_block(self, Q, DV1, c, GB, margin=None):
        """GB after the launch holds, at each chunk's columns, the fp32 gradient the kernel used at that
        chunk's start. margin_need per moved step = |2 d.(g_kernel - g_exact)| / scale (scale as in
        kernels.iq._gate): the gate margin that would absorb the gradient error of that step."""
        gk = GB[:, self.rows].double()
        v2 = Q[:, self.rows][:, :, c].double()
        v1 = DV1[:, self.rows].double()
        lo = torch.zeros(8, dtype=torch.bool, device=Q.device); lo[:4] = True
        for jl in range(len(c) // 8):
            sl = slice(jl * 8, jl * 8 + 8)
            cj = c[sl]
            s0, s1, s2 = self.v0[..., sl], v1[..., sl], v2[..., sl]
            states = [s0, s1, s2] if self.gpc == 1 else [s0, torch.where(lo, s1, s0), s1, s2]
            HB = self.H.index_select(1, cj)                                    # [E, 8, K]
            A = HB.index_select(2, cj)                                         # [E, 8, 8]
            for a, b in zip(states, states[1:]):
                d = b - a                                                      # [E, nd, 8]
                moved = (d != 0).any(-1)
                dAd = (torch.bmm(d, A) * d).sum(-1)
                delta = dAd - 2 * (d * self.g[:, :, cj]).sum(-1)
                gkb = gk[..., sl]
                gkc = gkb - torch.bmm(a - s0, A)                               # kernel's g_cur, fp64 from its fp32 gb
                dk = dAd - 2 * (d * gkc).sum(-1)
                Aa, da = A.abs(), d.abs()
                scale = (torch.bmm(da, Aa) * da).sum(-1) + 2 * (da * (gkb.abs() + torch.bmm((a - s0).abs(), Aa))).sum(-1)
                need = (2 * (d * (gkc - self.g[:, :, cj])).sum(-1)).abs() / scale.clamp_min(1e-300)
                if moved.any():
                    self.st["margin_need"].append(need[moved].float().cpu())
                    self.st["gain_rel"].append((-dk / scale.clamp_min(1e-300))[moved].float().cpu())
                    if margin is not None:
                        self.st["gate_violations"] += int((moved & ~(dk < -margin * scale)).sum())
                rel = delta / self.obj.abs().clamp_min(1e-300)
                up = moved & (delta > 0)
                st = self.st
                st["steps"] += int(d.shape[0] * d.shape[1]); st["moved"] += int(moved.sum()); st["uphill"] += int(up.sum())
                if up.any():
                    st["uphill_max_rel"] = max(st["uphill_max_rel"], float(rel[up].max()))
                    st["uphill_sum_rel"] += float(rel[up].sum())
                if moved.any():
                    st["downhill_min_rel"] = min(st["downhill_min_rel"], float(rel[moved].min()))
                self.obj = self.obj + torch.where(moved, delta, torch.zeros_like(delta))
                self.g -= torch.bmm(d, HB)


def _shadow_after_flip(self, Q, c):
    """ShadowCheck replay of an IQ1_S flip-pass block (32-column items, one move per item)"""
    v2 = Q[:, self.rows][:, :, c].double()
    for jl in range(len(c) // 32):
        sl = slice(jl * 32, jl * 32 + 32)
        cj = c[sl]
        d = v2[..., sl] - self.v0[..., sl]
        moved = (d != 0).any(-1)
        HB = self.H.index_select(1, cj)
        A = HB.index_select(2, cj)
        delta = (torch.bmm(d, A) * d).sum(-1) - 2 * (d * self.g[:, :, cj]).sum(-1)
        rel = delta / self.obj.abs().clamp_min(1e-300)
        up = moved & (delta > 0)
        st = self.st
        st["steps"] += int(d.shape[0] * d.shape[1]); st["moved"] += int(moved.sum()); st["uphill"] += int(up.sum())
        st["flip_moved"] = st.get("flip_moved", 0) + int(moved.sum()); st["flip_uphill"] = st.get("flip_uphill", 0) + int(up.sum())
        if up.any():
            st["uphill_max_rel"] = max(st["uphill_max_rel"], float(rel[up].max()))
            st["uphill_sum_rel"] += float(rel[up].sum())
        self.obj = self.obj + torch.where(moved, delta, torch.zeros_like(delta))
        self.g -= torch.bmm(d, HB)


ShadowCheck.after_flip_block = _shadow_after_flip


class FastRefiner:
    def __init__(self, t, dev, cfg=None):
        self.t, self.dev = t, dev
        self.gs = iq.SPEC[t]["gsize"]
        self.parity = t in iq.PARITY
        self.iq1 = t in iq.IQ1
        self.cfg = dict(KCFG[t], **(cfg or {}))
        self.G = torch.from_numpy(iq.grid(t)).to(dev)
        if self.iq1:
            self.tabs = k1.iq1_tables(iq.grid(t), dev)
        elif self.parity:
            assert iq.SPEC[t]["sub"] % 8 == 0                                 # a sign chunk never straddles two scales
            self.tabs = kp.parity_tables(iq.grid(t), iq.KSIGNS, dev)
        elif t == "IQ3_S":
            self.tabs = dict(CS=kn.iq3_tables(iq.grid(t), dev))
        else:
            self.tabs = kn.iq2_tables(iq.grid(t), dev)
        self.timing = {}
        self.gate_rejects = None

    def recon(self, mult, code):
        """exact dequant of device codes: [E, Rt, NG] -> [E, Rt, K]"""
        gs = self.gs
        E, Rt, NG = code.shape
        if self.iq1:                                                          # m * (grid + delta), exact in fp32
            return (self.tabs["GFULL"][code.long()] * mult[..., None]).reshape(E, Rt, NG * 8)
        mag = self.G[(code >> gs).long()]                                     # [E, Rt, NG, gs]
        sgn = ((code[..., None] >> torch.arange(gs, device=self.dev)) & 1).bool()
        return ((mag * mult[..., None]) * torch.where(sgn, -1.0, 1.0)).reshape(E, Rt, NG * gs)

    def run(self, W, Hf, mult, code, nmat, sweeps=8, tol=1e-6, seed=0, gate=None, margin=0.0, shadow=None, refresh="fp32", flip=False,
            schedule=None):
        """W [E, Rt, K], Hf [E, K, K] (symmetric), mult/code [E, Rt, NG] (code updated in place).
        Rows are nmat stacked matrices of Rt/nmat rows each. Returns per-(expert, mat) moves and
        obj_log lists (ridged objective after each sweep, entry 0 = start).
        gate: accept gate (None = on for parity types, off for IQ2_S / IQ3_S); per-sweep
        rejections in self.gate_rejects[e][mat]. shadow: ShadowCheck (parity types only).
        refresh: "fp32" (cuBLAS bmm, default) or "ozaki" (OzakiRefresh: near-exact gradient refresh).
        flip: IQ1_S only, add the per-sweep delta-flip pass (kernels.iq1._iq1s_flip_kernel); per-sweep counts in
        self.flip_moves[e][mat] (also included in moves).
        schedule: optional callable sw -> (W_sw or None, Hf_sw or None, noisy) giving the target and Hessian that sweep sw
        descends on (None = W / Hf); the gradient refresh at the end of sweep sw-1 is done against sweep sw's target, so a
        whole noisy-then-clean schedule runs in one call. Early stop applies only to sweeps with noisy=False. The last
        obj_log entry is always the clean objective under (W, Hf). schedule=None: every sweep on (W, Hf)."""
        dev, gs, cfg = self.dev, self.gs, self.cfg
        gate = (1 if (self.parity or self.iq1) else 0) if gate is None else int(gate)    # 0 off, 1 fp32 gate, 2 fp64 gate
        E, Rt, K = W.shape
        Rm = Rt // nmat
        NG = K // gs
        CW = 8 if self.parity else gs                                         # columns per visited item
        NI = K // CW                                                          # items per sweep
        BSZ = cfg["BSZ"]
        NB = BSZ // CW
        assert NI % NB == 0, (NI, NB)
        nblk = NI // NB
        rng = np.random.default_rng(seed)
        perms = np.stack([rng.permutation(NI) for _ in range(sweeps)])      # reference visit order
        grp = torch.from_numpy(perms.astype(np.int32)).to(dev)                # [sweeps, NI]
        cols = torch.from_numpy((perms[:, :, None] * CW + np.arange(CW)).reshape(sweeps, nblk, BSZ)).to(dev)
        if flip:
            assert self.t == "IQ1_S" and BSZ % 32 == 0 and (K // 32) % (BSZ // 32) == 0, (self.t, BSZ, K)
            rng_f = np.random.default_rng(seed + 7919)                         # flip-pass order
            perms_f = np.stack([rng_f.permutation(K // 32) for _ in range(sweeps)])
            NBF = BSZ // 32
            nblk_f = (K // 32) // NBF
            grp_f = torch.from_numpy(perms_f.astype(np.int32)).to(dev)
            cols_f = torch.from_numpy((perms_f[:, :, None] * 32 + np.arange(32)).reshape(sweeps, nblk_f, BSZ)).to(dev)
            accf = torch.zeros(E, Rt, dtype=torch.int32, device=dev)
            self.flip_moves = [[[] for _ in range(nmat)] for _ in range(E)]
        W0, Hf0 = W, Hf

        def tgt(sw):                                                          # (target, Hessian, noisy) for sweep sw
            if schedule is None or sw >= sweeps:
                return W0, Hf0, False
            Ws, Hs, noisy = schedule(sw)
            return (W0 if Ws is None else Ws), (Hf0 if Hs is None else Hs), bool(noisy)
        W, Hf, noisy = tgt(0)
        if schedule is not None:
            assert refresh == "fp32" and shadow is None, "schedule: fp32 refresh, no shadow"
        Q = self.recon(mult, code)
        Rr = W - Q
        if refresh == "ozaki":
            oz = OzakiRefresh(Hf)
            g = oz(Rr, torch.empty_like(Rr))
        else:
            assert refresh == "fp32", refresh
            g = torch.bmm(Rr, Hf)
        objd = lambda: (Rr * g).reshape(E, nmat, -1).double().sum(-1)         # noqa: E731
        obj0 = objd()
        del Rr
        acc = torch.zeros(E, Rt, dtype=torch.int32, device=dev)
        rej = torch.zeros(E, Rt, dtype=torch.int32, device=dev)
        act = torch.ones(E, nmat, dtype=torch.int32, device=dev)
        D = torch.empty(E, Rt, BSZ, device=dev)
        DV1 = None
        if shadow is not None:
            assert self.parity or self.iq1
            DV1 = torch.empty(E, Rt, BSZ, device=dev)
        moves = [[[] for _ in range(nmat)] for _ in range(E)]
        self.gate_rejects = [[[] for _ in range(nmat)] for _ in range(E)]
        self.gate_rejects_real = [[[] for _ in range(nmat)] for _ in range(E)]   # rejected moves that were not no-ops
        obj_log = [[[v] for v in row] for row in obj0.cpu().tolist()]
        alive = np.ones((E, nmat), bool)
        for sw in range(sweeps):
            acc.zero_()
            rej.zero_()
            if shadow is not None:
                shadow.begin_sweep(Q)
            for b in range(nblk):
                c = cols[sw, b]
                HC = Hf.index_select(1, c)                                    # [E, BSZ, K] = Hf[C, :]
                HCC = HC.index_select(2, c)                                   # [E, BSZ, BSZ]
                GB = g.index_select(2, c)                                     # [E, Rt, BSZ]
                if self.iq1:
                    if shadow is not None:
                        shadow.before_block(Q, c, GB)
                    k1.launch_iq1_block(self.t, Q, GB, D, HCC, mult, code, acc, rej, act, grp[sw, b * NB:(b + 1) * NB],
                                        self.tabs, Rm, tol, cfg, gate, margin, DV1)
                    if shadow is not None:
                        shadow.after_block(Q, DV1, c, GB, margin if gate else None)
                elif self.parity:
                    if shadow is not None:
                        shadow.before_block(Q, c, GB)
                    kp.launch_parity_block(Q, GB, D, HCC, mult, code, acc, rej, act, grp[sw, b * NB:(b + 1) * NB],
                                           self.tabs, Rm, tol, cfg, gate, margin, DV1)
                    if shadow is not None:
                        shadow.after_block(Q, DV1, c, GB, margin if gate else None)
                elif gate:
                    kn.launch_block(self.t, Q, GB, D, HCC, mult, code, acc, act, grp[sw, b * NB:(b + 1) * NB],
                                    self.tabs, Rm, tol, cfg, REJ=rej, gate=gate, margin=margin)
                else:
                    kn.launch_block(self.t, Q, GB, D, HCC, mult, code, acc, act, grp[sw, b * NB:(b + 1) * NB],
                                    self.tabs, Rm, tol, cfg)
                g.baddbmm_(D, HC, beta=1.0, alpha=-1.0)                       # deferred update, all columns
            if flip:
                accf.zero_()
                for b in range(nblk_f):
                    c = cols_f[sw, b]
                    HC = Hf.index_select(1, c)
                    HCC = HC.index_select(2, c)
                    GB = g.index_select(2, c)
                    if shadow is not None:
                        shadow.before_block(Q, c, GB)
                    k1.launch_iq1s_flip_block(Q, GB, D, HCC, mult, code, accf, rej, act, grp_f[sw, b * NBF:(b + 1) * NBF],
                                              self.tabs, Rm, tol, cfg, gate, margin)
                    if shadow is not None:
                        shadow.after_flip_block(Q, c)
                    g.baddbmm_(D, HC, beta=1.0, alpha=-1.0)
                acc += accf
                fm = accf.reshape(E, nmat, Rm).sum(-1).cpu().numpy()
                for e in range(E):
                    for k in range(nmat):
                        if alive[e, k]:
                            self.flip_moves[e][k].append(int(fm[e, k]))
            if schedule is not None:
                cur_noisy = noisy
                W, Hf, noisy = tgt(sw + 1)                                    # refresh against the next sweep's target
            Rr = W - Q
            if refresh == "ozaki":
                oz(Rr, g)
            else:
                torch.bmm(Rr, Hf, out=g)                                      # exact refresh once per sweep
            o = objd().cpu().numpy()
            mr = torch.stack([acc.reshape(E, nmat, Rm).sum(-1), (rej & 0xFFFF).reshape(E, nmat, Rm).sum(-1),
                              (rej >> 16).reshape(E, nmat, Rm).sum(-1)]).cpu().numpy()
            mv, rj, rjr = mr[0], mr[1], mr[2]                                 # the one sync per sweep
            del Rr
            for e in range(E):
                for k in range(nmat):
                    if alive[e, k]:
                        obj_log[e][k].append(float(o[e, k]))
                        moves[e][k].append(int(mv[e, k]))
                        self.gate_rejects[e][k].append(int(rj[e, k]))
                        self.gate_rejects_real[e][k].append(int(rjr[e, k]))
                        if mv[e, k] == 0 and not (schedule is not None and cur_noisy):
                            alive[e, k] = False
            if not alive.any():
                break
            act.copy_(torch.from_numpy(alive.astype(np.int32)))
        if shadow is not None:
            shadow.begin_sweep(Q)                                             # closes the last sweep's replay check
        return Q, moves, obj_log


def hessian(chunks, dev, maxrows=32768):
    """chunks: (x fp16 [n, k], w f32 [n]) -> sum (w x)(w x)^T, fp32 on dev"""
    x, w = chunks
    k = x.shape[1]
    H = torch.zeros(k, k, device=dev)
    for i in range(0, max(len(w), 1), maxrows):
        if torch.is_tensor(x):
            xs = x[i:i + maxrows].to(dev).float() * w[i:i + maxrows].to(dev)[:, None]
        else:
            xs = torch.from_numpy(x[i:i + maxrows]).to(dev).float() * torch.from_numpy(w[i:i + maxrows]).to(dev)[:, None]
        if i == 0 and len(w) <= maxrows:
            H = xs.T @ xs
        else:
            H += xs.T @ xs
    return H


def quad(R, H, fp64):
    """per-row-block objective sum_i r_i H r_i^T for R [E, Rt, K], H [E, K, K] -> [E, Rt] (double)"""
    if fp64:
        Rd = R.double()
        return (torch.bmm(Rd, H.double()) * Rd).sum(-1)
    return (torch.bmm(R, H) * R).double().sum(-1)
