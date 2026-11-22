"""Float64 / float32 reference implementations of Hessian codeword refinement, one problem at a time, for checking
ggufrefine.refine.engine.FastRefiner (tests/gpu/test_engine.py).

All three minimise tr((W - Q) Hf (W - Q)^T) over the codes of one [R, K] matrix with the block scales fixed, by
coordinate descent in the same visit order as the engine (one np.random.default_rng(seed).permutation per sweep),
keeping the gradient g = (W - Q) Hf exact with rank-k updates after every accepted move and refreshing it once per
sweep. A candidate is accepted iff its gain exceeds tol * |current item objective|.

  Refiner        IQ3_S (exhaustive over 512 x 16 signed codewords per 4-weight group) and IQ2_S (per 8-weight group,
                 magnitude step over 1024 codewords with signs fixed, then sign step over 256 patterns with
                 magnitudes fixed); float32 search.
  ParityRefiner  IQ2_XXS / IQ2_XS / IQ3_XXS: every 8 weights share a 7-bit sign field with an implied even-parity 8th
                 sign, so only 128 sign patterns exist. Per 8-weight chunk: magnitude step per grid group (signs
                 fixed), then sign step over the 128 legal patterns (IQ3_XXS: cross terms between its two 4-dim
                 groups included). float32 search, gradient in float64, and a float64 gate: the chosen move is
                 applied only if its exact objective change is negative (gate_rejects counts refusals).
  IQ1Refiner     IQ1_S / IQ1_M: u = m * (grid[idx] + delta), delta = +-0.125. IQ1_M searches 2048 grid rows x 2 deltas
                 jointly per 8-weight group; IQ1_S keeps the delta shared by a 32-weight sub-block, with an optional
                 flip pass (negate delta, greedy re-pick of the 4 grid indices, keep only if the exact change is
                 negative). Float64 throughout; debug=True asserts no row objective increases and the tracked
                 objective matches a recompute.

Each run(W, f, Hf, ...) takes the target W [R, K] (float32 numpy), the codec fields f of the stock codes
(ggufrefine.quant.codes.unpack) and Hf [K, K] (torch, on the device), and returns (refined fields, accepted moves per
sweep); obj_log holds the objective at the start and after every sweep.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ggufrefine.quant import codes as iq  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")


def objective(R, H):
    """sum_i r_i H r_i^T and the matching signal for NMSE"""
    return float(((R @ H) * R).sum())


class Refiner:
    def __init__(self, t, dev):
        self.t, self.dev = t, dev
        s = iq.SPEC[t]
        self.gs = s["gsize"]
        self.G = torch.from_numpy(iq.grid(t)).to(dev)                          # [ncode, gs]
        nsign = 1 << self.gs
        bits = ((torch.arange(nsign, device=dev)[:, None] >> torch.arange(self.gs, device=dev)) & 1)
        self.S = (1 - 2 * bits).float()                                        # [nsign, gs] +1 / -1
        if t == "IQ3_S":                                                       # all signed codewords
            self.C = (self.G[:, None, :] * self.S[None, :, :]).reshape(-1, self.gs)    # [ncode*16, 4]
            self.CC = (self.C[:, :, None] * self.C[:, None, :]).reshape(len(self.C), -1)
        else:
            self.GG = (self.G[:, :, None] * self.G[:, None, :]).reshape(len(self.G), -1)   # [1024, 64]
            self.SS = (self.S[:, :, None] * self.S[:, None, :]).reshape(nsign, -1)          # [256, 64]

    def run(self, W, f, Hf, sweeps=8, tol=1e-6, seed=0):
        """W [R, K] target, f fields (rows*K/256 blocks), Hf [K, K]. Returns refined fields + stats."""
        dev, gs, t = self.dev, self.gs, self.t
        R, K = W.shape
        nb_row = K // iq.QK
        idx = torch.from_numpy(f["idx"].astype(np.int64)).to(dev).reshape(R, -1)          # [R, K/gs]
        sgn = torch.from_numpy(f["signs"]).to(dev).reshape(R, K)                          # True = negative
        sub = iq.scale_mult(t, f["scales"])                                                # [nb, nsub]
        mult = torch.from_numpy((f["d"][:, None] * sub).astype(np.float32)).to(dev)       # [nb, nsub]
        mult = mult.reshape(R, nb_row, -1).repeat_interleave(iq.SPEC[t]["sub"] // gs, dim=2).reshape(R, -1)  # per group
        Wt = torch.from_numpy(W).to(dev)
        Hf = Hf.float()

        def recon():
            return (self.G[idx] * mult[:, :, None]).reshape(R, K) * torch.where(sgn, -1.0, 1.0)

        Qr = recon()
        Rr = Wt - Qr
        g = Rr @ Hf
        rng = np.random.default_rng(seed)
        moves = []
        self.obj_log = [float((Rr * g).sum())]                                   # ridged objective, sweep 0 = start
        for sw in range(sweeps):
            n_acc = 0
            for j in rng.permutation(K // gs):
                B = slice(j * gs, (j + 1) * gs)
                A = Hf[B, B]
                v = Qr[:, B]
                q = v @ A + g[:, B]
                m = mult[:, j]
                cur = (v @ A * v).sum(1) - 2 * (v * q).sum(1)
                if t == "IQ3_S":
                    T = self.CC @ A.reshape(-1)                                           # [8192]
                    S = m[:, None] ** 2 * T[None] - 2 * m[:, None] * (q @ self.C.T)
                    best = S.argmin(1)
                    new = self.C[best] * m[:, None]
                    gain = cur - S.gather(1, best[:, None]).squeeze(1)
                    ok = gain > tol * cur.abs().clamp_min(1e-30)
                    if ok.any():
                        d_ = torch.where(ok[:, None], new - v, torch.zeros_like(v))
                        Qr[:, B] += d_; g -= d_ @ Hf[B, :]
                        code, sp_ = best // len(self.S), best % len(self.S)
                        idx[ok, j] = code[ok]
                        sgn[ok, B] = (self.S[sp_[ok]] < 0)
                        n_acc += int(ok.sum())
                else:
                    # magnitudes with signs fixed: metric (sigma sigma^T) * A
                    sig = torch.where(sgn[:, B], -1.0, 1.0)
                    T = ((sig[:, :, None] * sig[:, None, :]) * A[None]).reshape(R, -1) @ self.GG.T     # [R, 1024]
                    S = m[:, None] ** 2 * T - 2 * m[:, None] * ((sig * q) @ self.G.T)
                    best = S.argmin(1)
                    gain = cur - S.gather(1, best[:, None]).squeeze(1)
                    ok = gain > tol * cur.abs().clamp_min(1e-30)
                    if ok.any():
                        new = self.G[best] * sig * m[:, None]
                        d_ = torch.where(ok[:, None], new - v, torch.zeros_like(v))
                        Qr[:, B] += d_; g -= d_ @ Hf[B, :]
                        idx[ok, j] = best[ok]; n_acc += int(ok.sum())
                    # signs with magnitudes fixed: metric (mu mu^T) * A
                    v = Qr[:, B]; q = v @ A + g[:, B]
                    cur = (v @ A * v).sum(1) - 2 * (v * q).sum(1)
                    mu = self.G[idx[:, j]] * m[:, None]
                    T = ((mu[:, :, None] * mu[:, None, :]) * A[None]).reshape(R, -1) @ self.SS.T       # [R, 256]
                    S = T - 2 * ((mu * q) @ self.S.T)
                    best = S.argmin(1)
                    gain = cur - S.gather(1, best[:, None]).squeeze(1)
                    ok = gain > tol * cur.abs().clamp_min(1e-30)
                    if ok.any():
                        new = mu * self.S[best]
                        d_ = torch.where(ok[:, None], new - v, torch.zeros_like(v))
                        Qr[:, B] += d_; g -= d_ @ Hf[B, :]
                        sgn[ok, B] = (self.S[best[ok]] < 0); n_acc += int(ok.sum())
            Rr = Wt - Qr
            g = Rr @ Hf                                                           # refresh once per sweep
            self.obj_log.append(float((Rr * g).sum()))
            moves.append(n_acc)
            if n_acc == 0:
                break
        out = dict(f)
        out["idx"] = idx.reshape(f["idx"].shape).cpu().numpy().astype(np.uint16)
        out["signs"] = sgn.reshape(f["signs"].shape).cpu().numpy()
        return out, moves


class ParityRefiner:
    def __init__(self, t, dev):
        assert t in iq.PARITY, t
        self.t, self.dev = t, dev
        s = iq.SPEC[t]
        self.gs = s["gsize"]
        self.gpc = 8 // self.gs                                                # grid groups per sign chunk
        self.G = torch.from_numpy(iq.grid(t)).to(dev)                          # [ncode, gs]
        self.GG = (self.G[:, :, None] * self.G[:, None, :]).reshape(len(self.G), -1)
        self.S = torch.from_numpy(np.where(iq.legal_signs(), -1.0, 1.0).astype(np.float32)).to(dev)  # [128, 8]
        self.SS = (self.S[:, :, None] * self.S[:, None, :]).reshape(len(self.S), -1)                  # [128, 64]

    def run(self, W, f, Hf, sweeps=8, tol=1e-6, seed=0, debug=False):
        """W [R, K] target, f fields, Hf [K, K] (float64). Returns refined fields + accepted moves per sweep."""
        dev, gs, t, gpc = self.dev, self.gs, self.t, self.gpc
        R, K = W.shape
        nb_row = K // iq.QK
        idx = torch.from_numpy(f["idx"].astype(np.int64)).to(dev).reshape(R, -1)          # [R, K/gs]
        sgn = torch.from_numpy(f["signs"]).to(dev).reshape(R, K)                          # True = negative
        sub = iq.scale_mult(t, f["scales"])
        mult = torch.from_numpy((f["d"][:, None] * sub).astype(np.float32)).to(dev)       # [nb, nsub]
        mult = mult.reshape(R, nb_row, -1).repeat_interleave(iq.SPEC[t]["sub"] // gs, dim=2).reshape(R, -1)
        Wt = torch.from_numpy(W).to(dev)
        Hf64 = Hf.double()
        Hf = Hf.float()

        def recon():
            return (self.G[idx] * mult[:, :, None]).reshape(R, K) * torch.where(sgn, -1.0, 1.0)

        def exact_rows(Q):
            r = Wt.double() - Q.double()
            return ((r @ Hf64) * r).sum(1)

        Qr = recon()
        g = (Wt.double() - Qr.double()) @ Hf64                                # gradient kept in float64
        if debug:
            f64 = exact_rows(Qr)                                              # all rows, tracked by accepted deltas
            dbg = torch.from_numpy(np.random.default_rng(seed + 1).choice(R, min(R, 64), replace=False)).to(dev)
            Wd64, Hd = Wt.double()[dbg], Hf64

            def sub_exact():                                                  # from scratch, independent of g
                r = Wd64 - Qr[dbg].double()
                return ((r @ Hd) * r).sum(1)
            prev = sub_exact()
            self.debug_stats = dict(steps=0, max_rel_increase=0.0, violations=0, recompute_rel_err=[])

        def apply(B, ok, new, v):
            """accept the float32-chosen candidate only where its exact float64 objective change is < 0"""
            nonlocal g
            d_ = new - v
            d64 = d_.double()
            dfx = ((d64 @ Hf64[B, B]) * d64).sum(1) - 2 * (d64 * g[:, B]).sum(1)
            self.gate_rejects += int((ok & ~(dfx < 0)).sum())
            ok = ok & (dfx < 0)
            nv = torch.where(ok[:, None], new, v)                              # write codes' values exactly (no drift)
            d_ = nv - v
            Qr[:, B] = nv
            g -= d_.double() @ Hf64[B, :]
            if debug:
                nonlocal f64, prev
                f64 = f64 + torch.where(ok, dfx, torch.zeros_like(dfx))
                now = sub_exact()
                rel = (now - prev) / prev.abs().clamp_min(1e-300)
                bad = rel > 1e-11
                st = self.debug_stats
                st["steps"] += 1
                st["max_rel_increase"] = max(st["max_rel_increase"], float(rel.max()))
                st["violations"] += int(bad.sum())
                assert not bad.any(), f"objective increased on {int(bad.sum())} rows, max rel {float(rel.max()):.3e}"
                prev = now
            return ok

        def score_step(B, v, q, cand_T, cand_lin):
            """returns best index, gain, cur for S = cand_T - 2 cand_lin"""
            A = Hf[B, B]
            cur = (v @ A * v).sum(1) - 2 * (v * q).sum(1)
            S = cand_T - 2 * cand_lin
            best = S.argmin(1)
            gain = cur - S.gather(1, best[:, None]).squeeze(1)
            ok = gain > tol * cur.abs().clamp_min(1e-30)
            return best, ok

        rng = np.random.default_rng(seed)
        self.gate_rejects = 0                                                  # float32 winners the float64 gate refused
        moves = []
        self.obj_log = [float(exact_rows(Qr).sum())]                           # float64 exact, sweep 0 = start
        for sw in range(sweeps):
            n_acc = 0
            for c in rng.permutation(K // 8):
                # magnitude steps: each grid group in the chunk, signs fixed
                for k in range(gpc):
                    j = c * gpc + k
                    B = slice(j * gs, (j + 1) * gs)
                    A = Hf[B, B]
                    v = Qr[:, B]
                    q = v @ A + g[:, B].float()
                    m = mult[:, j]
                    sig = torch.where(sgn[:, B], -1.0, 1.0)
                    T = ((sig[:, :, None] * sig[:, None, :]) * A[None]).reshape(R, -1) @ self.GG.T   # [R, ncode]
                    best, ok = score_step(B, v, q, m[:, None] ** 2 * T, m[:, None] * ((sig * q) @ self.G.T))
                    if ok.any():
                        ok = apply(B, ok, self.G[best] * sig * m[:, None], v)
                        idx[ok, j] = best[ok]; n_acc += int(ok.sum())
                # sign step: whole 8-weight chunk, magnitudes fixed, 128 legal patterns
                B = slice(c * 8, c * 8 + 8)
                A = Hf[B, B]
                v = Qr[:, B]
                q = v @ A + g[:, B].float()
                js = slice(c * gpc, (c + 1) * gpc)
                mu = (self.G[idx[:, js]] * mult[:, js, None]).reshape(R, 8)
                T = ((mu[:, :, None] * mu[:, None, :]) * A[None]).reshape(R, -1) @ self.SS.T          # [R, 128]
                best, ok = score_step(B, v, q, T, (mu * q) @ self.S.T)
                if ok.any():
                    ok = apply(B, ok, mu * self.S[best], v)
                    sgn[ok, B] = self.S[best[ok]] < 0; n_acc += int(ok.sum())
            g = (Wt.double() - Qr.double()) @ Hf64                            # refresh once per sweep
            ex = exact_rows(Qr)
            self.obj_log.append(float(ex.sum()))
            if debug:
                assert torch.equal(Qr, recon()), "Qr drifted from the codes"
                par = sgn.reshape(R, -1, 8).sum(-1) & 1
                assert not par.any(), "illegal sign pattern"
                rel = float(((f64 - ex).abs() / ex.abs().clamp_min(1e-300)).max())
                self.debug_stats["recompute_rel_err"].append(rel)
                assert rel < 1e-8, rel
                f64 = ex
            moves.append(n_acc)
            if n_acc == 0:
                break
        out = dict(f)
        out["idx"] = idx.reshape(f["idx"].shape).cpu().numpy().astype(np.uint16)
        out["signs"] = sgn.reshape(f["signs"].shape).cpu().numpy()
        return out, moves


class IQ1Refiner:
    def __init__(self, t, dev):
        assert t in iq.IQ1, t
        self.t, self.dev = t, dev
        G = torch.from_numpy(iq.grid(t)).double().to(dev)                     # [2048, 8]
        self.C = torch.cat([G + 0.125, G - 0.125])                            # [4096, 8], index = delta_neg << 11 | idx

    def run(self, W, f, Hf, sweeps=8, tol=1e-6, seed=0, flip=False, debug=False):
        """W [R, K] float32 target, f iq.unpack fields of the [R, K] matrix, Hf [K, K] (any float dtype).
        Returns (refined fields, moves per sweep); obj_log (exact float64 objective, entry 0 = start),
        flip_moves per sweep (IQ1_S, flip=True)."""
        t, dev, C = self.t, self.dev, self.C
        assert not flip or t == "IQ1_S", "flip item is for IQ1_S only"
        R, K = W.shape
        NG = K // 8
        spec = iq.SPEC[t]
        code = torch.from_numpy((f["idx"].astype(np.int64) | (f["delta"].astype(np.int64) << 11)).reshape(R, NG)).to(dev)
        dl = (f["d"][:, None] * iq.scale_mult(t, f["scales"])).astype(np.float32)             # [nb, nsub], exact
        mult = torch.from_numpy(np.repeat(dl, spec["sub"] // 8, axis=1).reshape(R, NG)).double().to(dev)
        Wd = torch.from_numpy(W).double().to(dev)
        H = Hf.double().to(dev)

        def recon():
            return (C[code] * mult[:, :, None]).reshape(R, K)

        Qr = recon()
        g = (Wd - Qr) @ H
        exact_rows = lambda Q: (((Wd - Q) @ H) * (Wd - Q)).sum(1)              # noqa: E731
        if debug:
            dbg = torch.from_numpy(np.random.default_rng(seed + 1).choice(R, min(R, 64), replace=False)).to(dev)
            prev = exact_rows(Qr)[dbg]
            self.debug_stats = dict(steps=0, max_rel_increase=0.0, recompute_rel_err=[])

        def check_step():
            nonlocal prev
            r = Wd[dbg] - Qr[dbg]
            now = ((r @ H) * r).sum(1)
            rel = (now - prev) / prev.abs().clamp_min(1e-300)
            st = self.debug_stats
            st["steps"] += 1
            st["max_rel_increase"] = max(st["max_rel_increase"], float(rel.max()))
            assert not (rel > 1e-11).any(), f"objective increased: max rel {float(rel.max()):.3e}"
            prev = now

        def best_in(j, q, A, allowed=None):
            """argmin over candidates of S(u) = u A u - 2 u.q, u = m C_c, for group j; allowed [R, 4096] bool or None"""
            m = mult[:, j]
            cAc = ((C @ A) * C).sum(1)                                         # [4096]
            S = m[:, None] ** 2 * cAc[None, :] - 2 * m[:, None] * (q @ C.T)    # [R, 4096]
            if allowed is not None:
                S = torch.where(allowed, S, torch.full_like(S, float("inf")))
            b = S.argmin(1)
            return b, S.gather(1, b[:, None]).squeeze(1)

        half = torch.arange(4096, device=dev) >> 11                            # delta bit of each candidate

        def group_step(j):
            B = slice(j * 8, j * 8 + 8)
            A = H[B, B]
            v = Qr[:, B]
            q = v @ A + g[:, B]
            cur = (v @ A * v).sum(1) - 2 * (v * q).sum(1)
            allowed = (half[None, :] == (code[:, j] >> 11)[:, None]) if t == "IQ1_S" else None
            b, sb = best_in(j, q, A, allowed)
            u = C[b] * mult[:, j, None]
            d = u - v
            delta = ((d @ A) * d).sum(1) - 2 * (d * g[:, B]).sum(1)
            ok = ((cur - sb) > tol * cur.abs().clamp_min(1e-30)) & (delta < 0)
            nv = torch.where(ok[:, None], u, v)
            d = nv - v
            Qr[:, B] = nv
            code[:, j] = torch.where(ok, b, code[:, j])
            g.sub_(d @ H[B, :])
            self.track += float(torch.where(ok, delta, torch.zeros_like(delta)).sum())
            return int(ok.sum())

        def flip_step(sb_):
            """IQ1_S sub-block sb_ (groups 4sb_..4sb_+3): flip delta, greedy re-pick of the 4 indices"""
            B = slice(sb_ * 32, sb_ * 32 + 32)
            A32 = H[B, B]
            v0 = Qr[:, B].clone()
            g0 = g[:, B]
            cur = (v0 @ A32 * v0).sum(1) - 2 * (v0 * (v0 @ A32 + g0)).sum(1)
            nd = 1 - (code[:, 4 * sb_] >> 11)                                  # flipped delta bit [R]
            c_new = code[:, 4 * sb_:4 * sb_ + 4] & 2047 | (nd[:, None] << 11)
            vt = (C[c_new] * mult[:, 4 * sb_:4 * sb_ + 4, None]).reshape(R, 32)
            allowed = half[None, :] == nd[:, None]
            for k in range(4):
                j = 4 * sb_ + k
                Bk = slice(8 * k, 8 * k + 8)
                gk = g0 - (vt - v0) @ A32                                     # exact gradient of the tentative state
                A = A32[Bk, Bk]
                v = vt[:, Bk]
                q = v @ A + gk[:, Bk]
                b, _ = best_in(j, q, A, allowed)                               # greedy: best given the new delta
                c_new[:, k] = b
                vt[:, Bk] = C[b] * mult[:, j, None]
            d = vt - v0
            delta = ((d @ A32) * d).sum(1) - 2 * (d * g0).sum(1)
            ok = delta < -tol * cur.abs().clamp_min(1e-30)
            nv = torch.where(ok[:, None], vt, v0)
            d = nv - v0
            Qr[:, B] = nv
            self.track += float(torch.where(ok, delta, torch.zeros_like(delta)).sum())
            code[:, 4 * sb_:4 * sb_ + 4] = torch.where(ok[:, None], c_new, code[:, 4 * sb_:4 * sb_ + 4])
            g.sub_(d @ H[B, :])
            return int(ok.sum())

        rng = np.random.default_rng(seed)
        rng_f = np.random.default_rng(seed + 7919)
        moves, self.flip_moves = [], []
        self.obj_log = [float(exact_rows(Qr).sum())]
        self.track = self.obj_log[0]                                           # objective tracked by accepted deltas
        for sw in range(sweeps):
            n_acc = 0
            for j in rng.permutation(NG):
                n_acc += group_step(int(j))
                if debug:
                    check_step()
            nf = 0
            if flip:
                for b in rng_f.permutation(K // 32):
                    nf += flip_step(int(b))
                    if debug:
                        check_step()
                self.flip_moves.append(nf)
            if debug:
                assert torch.equal(Qr, recon()), "Q drifted from the codes"
            Qr = recon()
            g = (Wd - Qr) @ H
            ex = float(exact_rows(Qr).sum())
            if debug:
                rel = abs(self.track - ex) / abs(ex)
                self.debug_stats["recompute_rel_err"].append(rel)
                assert rel < 1e-8, rel
            self.track = ex
            self.obj_log.append(ex)
            moves.append(n_acc + nf)
            if n_acc + nf == 0:
                break
        out = dict(f)
        c = code.cpu().numpy()
        out["idx"] = (c & 2047).astype(np.uint16).reshape(f["idx"].shape)
        out["delta"] = (c >> 11).astype(bool).reshape(f["delta"].shape)
        return out, moves
