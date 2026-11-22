"""Triton kernel for batched Hessian codeword refinement of IQ1_S / IQ1_M (ggufrefine.refine.engine).
Launch structure as kernels.parity_sign / kernels.free_sign: one launch = one block of NB items of the
per-sweep permutation, all rows of all E experts; each program owns BR rows of one expert and walks
the block's items sequentially (exact in-block gradient); the caller applies the deferred update
of the other columns as one batched GEMM per block.

Item = one 8-weight group (a permutation of the K/8 groups per sweep). Values
u = m * (grid[idx] + delta), m = d * (2s+1) fixed per group, delta = +-0.125.
  IQ1_M  (FIXDELTA=0) joint search over the 4096 codewords grid[idx] + delta (code = dneg << 11 | idx)
         against CBF; S(u) = m^2 c A c - 2 m c.q with q = v A + g_B, cur = S(v).
  IQ1_S  (FIXDELTA=1) delta is shared by the 4 groups of a 32-weight sub-block and stays fixed here:
         with vo = m delta 1 (the fixed part) and p = q - vo A, the grid part G = (u - vo) / m is
         searched over the 2048 grid rows (CBG): S'(G) = m^2 G A G - 2 m G.p, cur' = S'(G_cur), and
         S'(best) - cur' is the exact objective change (same form as the parity magnitude step).
Both are one [BR, 64] x [64, ncand] tl.dot per candidate tile (_feat_rows / _feat_codebook with
U = 1: 36 quadratic + 8 linear feature slots), argmin with lowest-index ties.
Acceptance as the engine's other types: the fp32 test gain = cur - S_best > tol * |cur|, then with
GATE != 0 kernels.common._gate (direct-form recheck of the move against the fp32 gradient).
Codes written are always legal (11-bit index, delta bit; IQ1_S keeps the sub-block's delta).
All search arithmetic FP32 (tl.dot input_precision="ieee": no TF32).

IQ1_S flip item (_iq1s_flip_kernel, FastRefiner.run(flip=True)): item = one 32-weight sub-block (its own
per-sweep permutation of the K/32 sub-blocks, run as a second pass after the group pass). Negate the
shared delta, re-pick the 4 grid indices greedily in order 0..3 under the new delta (each exhaustive
over 2048, exact in-item gradient g_B - (v_t - v0) A32), then keep the whole 32-weight move only if
its direct objective change delta = d A32 d - 2 d.g_B is < -tol * |S(v0)| and, with GATE != 0,
< -(margin + FLIP_ARITH) * scale (scale as kernels.common._gate, over the 32 columns; GATE=2 is
treated as GATE=1 here: the flip recheck is fp32 with the rounding bound). Same as
the fp64 reference IQ1Refiner(flip=True).
"""
import torch
import triton
import triton.language as tl

FLIP_ARITH = tl.constexpr(1e-5)   # rounding bound of the fp32 32-wide delta relative to `scale` (~170 u_fp32; two
                      # 32-term dot products + a 32-term sum per chain)

from .common import _argmin_dot, _feat_codebook, _feat_rows, _gate, _quad8


def iq1_tables(grid, dev):
    """grid [2048, 8] in {-1, 0, 1} -> CBF [64, 4096] features of grid + delta (index = dneg << 11 | idx),
    CBG [64, 2048] features of the bare grid, GF [8, 4096] the codeword values (exact in fp32)"""
    g = torch.as_tensor(grid, dtype=torch.float64)
    full = torch.cat([g + 0.125, g - 0.125])
    return dict(CBF=_feat_codebook(full).to(dev), CBG=_feat_codebook(g).to(dev),
                GF=full.T.contiguous().float().to(dev), GFULL=full.float().to(dev))


@triton.jit
def _iq1_block_kernel(Q, GB, D, HCC, MULT, CODE, ACC, REJ, ACT, GRP, CB, GF, DV1,
                      Rt, K, NG, Rm, NMAT, tol, margin,
                      NB: tl.constexpr, BSZ: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr,
                      CH: tl.constexpr, NCAND: tl.constexpr, FIXDELTA: tl.constexpr, PREC: tl.constexpr,
                      GATE: tl.constexpr, DBG: tl.constexpr):
    pid = tl.program_id(0)
    e = tl.program_id(1).to(tl.int64)
    rows = pid * BR + tl.arange(0, BR)
    rm = rows < Rt
    rows64 = rows.to(tl.int64)
    a8 = tl.arange(0, 8)
    act = tl.load(ACT + e * NMAT + rows // Rm, mask=rm, other=0)
    live = rm & (act != 0)
    qrow = Q + e * Rt * K + rows64 * K
    gbrow = GB + e * Rt * BSZ + rows64 * BSZ
    drow = D + e * Rt * BSZ + rows64 * BSZ
    crow = CODE + e * Rt * NG + rows64 * NG
    mrow = MULT + e * Rt * NG + rows64 * NG
    hbase = HCC + e * BSZ * BSZ
    ones = tl.full([BR, 8], 1.0, tl.float32)
    acc = tl.zeros([BR], dtype=tl.int32)
    rej = tl.zeros([BR], dtype=tl.int32)
    for jl in range(NB):
        j = tl.load(GRP + jl)                                                  # group id
        col0 = j * 8
        l0 = jl * 8
        v0 = tl.load(qrow[:, None] + col0 + a8[None, :], mask=rm[:, None], other=0.0)      # [BR, 8]
        gb = tl.load(gbrow[:, None] + l0 + a8[None, :], mask=rm[:, None], other=0.0)
        m = tl.load(mrow + j, mask=rm, other=0.0)
        c0 = tl.load(crow + j, mask=rm, other=0)
        A = tl.load(hbase + (l0 + a8[:, None]) * BSZ + l0 + a8[None, :])                   # [8, 8]
        q = tl.sum(v0[:, :, None] * A[None, :, :], axis=1) + gb
        if FIXDELTA:
            dneg = c0 >> 11
            vo = ones * (tl.where(dneg != 0, -0.125, 0.125) * m)[:, None]                  # exact
            vk = v0 - vo                                                                   # = m * grid row, exact
            p = q - tl.sum(vo[:, :, None] * A[None, :, :], axis=1)
            cur = _quad8(vk, A) - 2.0 * tl.sum(vk * p, axis=1)
            X = _feat_rows(ones, -2.0 * m[:, None] * p, A, m * m, a8, a8)
            best_v, best_i = _argmin_dot(X, CB, NCAND, BC, BR, PREC)
            best_c = (dneg << 11) | best_i
        else:
            cur = _quad8(v0, A) - 2.0 * tl.sum(v0 * q, axis=1)
            X = _feat_rows(ones, -2.0 * m[:, None] * q, A, m * m, a8, a8)
            best_v, best_i = _argmin_dot(X, CB, NCAND, BC, BR, PREC)
            best_c = best_i
        ok = live & ((cur - best_v) > tol * tl.maximum(tl.abs(cur), 1e-30))
        u = tl.load(GF + a8[None, :] * 4096 + best_c[:, None]) * m[:, None]
        if GATE != 0:
            ok, rj = _gate(ok, u, v0, v0, gb, A, margin, GATE)
            rej += rj.to(tl.int32)
        v = tl.where(ok[:, None], u, v0)
        if DBG:
            tl.store(DV1 + e * Rt * BSZ + rows64[:, None] * BSZ + l0 + a8[None, :], v, mask=rm[:, None])
        d = v - v0
        tl.store(qrow[:, None] + col0 + a8[None, :], v, mask=ok[:, None])
        tl.store(drow[:, None] + l0 + a8[None, :], d, mask=rm[:, None])
        tl.store(crow + j, best_c, mask=ok)
        acc += ok.to(tl.int32)
        # in-block gradient for the groups still to come
        lo = l0 + 8
        for cc0 in range(0, BSZ, CH):
            if cc0 + CH > lo:
                cc = cc0 + tl.arange(0, CH)
                hb = tl.load(hbase + (l0 + a8[:, None]) * BSZ + cc[None, :])                # [8, CH]
                gp = gbrow[:, None] + cc[None, :]
                gt = tl.load(gp, mask=rm[:, None], other=0.0)
                gt = gt - tl.sum(d[:, :, None] * hb[None, :, :], axis=1)
                tl.store(gp, gt, mask=rm[:, None] & (cc[None, :] >= lo))
        tl.debug_barrier()
    old = tl.load(ACC + e * Rt + rows64, mask=rm, other=0)
    tl.store(ACC + e * Rt + rows64, old + acc, mask=rm)
    if GATE != 0:
        old = tl.load(REJ + e * Rt + rows64, mask=rm, other=0)
        tl.store(REJ + e * Rt + rows64, old + rej, mask=rm)


def launch_iq1_block(t, Q, GB, D, HCC, MULT, CODE, ACC, REJ, ACT, GRP, tabs, Rm, tol, cfg, gate, margin, DV1=None):
    E, Rt, K = Q.shape
    NG = MULT.shape[2]
    fix = t == "IQ1_S"
    ncand = 2048 if fix else 4096
    grid = (triton.cdiv(Rt, cfg["BR"]), E)
    _iq1_block_kernel[grid](Q, GB, D, HCC, MULT, CODE, ACC, REJ, ACT, GRP, tabs["CBG"] if fix else tabs["CBF"], tabs["GF"],
                            D if DV1 is None else DV1, Rt, K, NG, Rm, ACT.shape[1], tol, float(margin),
                            NB=GRP.numel(), BSZ=GB.shape[2], BR=cfg["BR"], BC=min(cfg["BC"], ncand), CH=cfg["CH"],
                            NCAND=ncand, FIXDELTA=fix, PREC=cfg.get("prec", "ieee"), GATE=int(gate), DBG=DV1 is not None,
                            num_warps=cfg["warps"], num_stages=cfg.get("stages", 1))


@triton.jit
def _iq1s_flip_kernel(Q, GB, D, HCC, MULT, CODE, ACC, REJ, ACT, GRP, CBG, GF,
                      Rt, K, NG, Rm, NMAT, tol, margin,
                      NB: tl.constexpr, BSZ: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr,
                      CH: tl.constexpr, PREC: tl.constexpr, GATE: tl.constexpr):
    pid = tl.program_id(0)
    e = tl.program_id(1).to(tl.int64)
    rows = pid * BR + tl.arange(0, BR)
    rm = rows < Rt
    rows64 = rows.to(tl.int64)
    a8 = tl.arange(0, 8)
    a4 = tl.arange(0, 4)
    a32 = tl.arange(0, 32)
    grp_of = a32 // 8
    pos_of = a32 % 8
    act = tl.load(ACT + e * NMAT + rows // Rm, mask=rm, other=0)
    live = rm & (act != 0)
    qrow = Q + e * Rt * K + rows64 * K
    gbrow = GB + e * Rt * BSZ + rows64 * BSZ
    drow = D + e * Rt * BSZ + rows64 * BSZ
    crow = CODE + e * Rt * NG + rows64 * NG
    mrow = MULT + e * Rt * NG + rows64 * NG
    hbase = HCC + e * BSZ * BSZ
    ones = tl.full([BR, 8], 1.0, tl.float32)
    acc = tl.zeros([BR], dtype=tl.int32)
    rej = tl.zeros([BR], dtype=tl.int32)
    for jl in range(NB):
        sb = tl.load(GRP + jl)                                                 # sub-block id
        col0 = sb * 32
        l0 = jl * 32
        v0 = tl.load(qrow[:, None] + col0 + a32[None, :], mask=rm[:, None], other=0.0)     # [BR, 32]
        gb = tl.load(gbrow[:, None] + l0 + a32[None, :], mask=rm[:, None], other=0.0)
        A32 = tl.load(hbase + (l0 + a32[:, None]) * BSZ + l0 + a32[None, :])               # [32, 32]
        m = tl.load(mrow + sb * 4, mask=rm, other=0.0)                         # one scale per 32 weights (IQ1_S)
        c4 = tl.load(crow[:, None] + sb * 4 + a4[None, :], mask=rm[:, None], other=0)     # [BR, 4]
        dneg = tl.sum(tl.where(a4[None, :] == 0, c4, 0), axis=1) >> 11
        nd = 1 - dneg
        c4 = (c4 & 2047) | (nd << 11)[:, None]
        dlt = tl.where(nd != 0, -0.125, 0.125) * m
        vo = ones * dlt[:, None]                                               # m * delta', exact
        for k in tl.static_range(4):
            cc32 = tl.sum(tl.where(grp_of[None, :, None] == a4[None, None, :], c4[:, None, :], 0), axis=2)    # [BR, 32]
            vt = tl.load(GF + pos_of[None, :] * 4096 + cc32) * m[:, None]
            dv = vt - v0
            ck = tl.sum(tl.where(a4[None, :] == k, c4, 0), axis=1)
            v = tl.load(GF + a8[None, :] * 4096 + ck[:, None]) * m[:, None]                # [BR, 8]
            Ak = tl.load(hbase + (l0 + a32[:, None]) * BSZ + l0 + 8 * k + a8[None, :])      # [32, 8]
            gbk = tl.load(gbrow[:, None] + l0 + 8 * k + a8[None, :], mask=rm[:, None], other=0.0)
            gk = gbk - tl.sum(dv[:, :, None] * Ak[None, :, :], axis=1)
            A = tl.load(hbase + (l0 + 8 * k + a8[:, None]) * BSZ + l0 + 8 * k + a8[None, :])  # [8, 8]
            q = tl.sum(v[:, :, None] * A[None, :, :], axis=1) + gk
            p = q - tl.sum(vo[:, :, None] * A[None, :, :], axis=1)
            X = _feat_rows(ones, -2.0 * m[:, None] * p, A, m * m, a8, a8)
            best_v, best_i = _argmin_dot(X, CBG, 2048, BC, BR, PREC)
            c4 = tl.where(a4[None, :] == k, ((nd << 11) | best_i)[:, None], c4)
        cc32 = tl.sum(tl.where(grp_of[None, :, None] == a4[None, None, :], c4[:, None, :], 0), axis=2)
        vt = tl.load(GF + pos_of[None, :] * 4096 + cc32) * m[:, None]
        dv = vt - v0
        dA = tl.dot(dv, A32, input_precision="ieee")
        delta = tl.sum(dv * dA, axis=1) - 2.0 * tl.sum(dv * gb, axis=1)
        v0A = tl.dot(v0, A32, input_precision="ieee")
        cur = -tl.sum(v0A * v0, axis=1) - 2.0 * tl.sum(v0 * gb, axis=1)
        ok = live & (delta < -tol * tl.maximum(tl.abs(cur), 1e-30))
        if GATE != 0:
            da = tl.abs(dv)
            scale = tl.sum(da * tl.dot(da, tl.abs(A32), input_precision="ieee"), axis=1) + 2.0 * tl.sum(da * tl.abs(gb), axis=1)
            okg = delta < -((margin + FLIP_ARITH) * scale)
            rj = ok & (okg == 0)
            real = tl.sum(da, axis=1) > 0
            rej += rj.to(tl.int32) + 65536 * (rj & real).to(tl.int32)
            ok = ok & okg
        d = tl.where(ok[:, None], dv, 0.0)
        tl.store(qrow[:, None] + col0 + a32[None, :], vt, mask=ok[:, None])
        tl.store(drow[:, None] + l0 + a32[None, :], d, mask=rm[:, None])
        tl.store(crow[:, None] + sb * 4 + a4[None, :], c4, mask=ok[:, None])
        acc += ok.to(tl.int32)
        lo = l0 + 32
        for cc0 in range(0, BSZ, CH):
            if cc0 + CH > lo:
                cc = cc0 + tl.arange(0, CH)
                hb = tl.load(hbase + (l0 + a32[:, None]) * BSZ + cc[None, :])               # [32, CH]
                gp = gbrow[:, None] + cc[None, :]
                gt = tl.load(gp, mask=rm[:, None], other=0.0)
                gt = gt - tl.dot(d, hb, input_precision="ieee")
                tl.store(gp, gt, mask=rm[:, None] & (cc[None, :] >= lo))
        tl.debug_barrier()
    old = tl.load(ACC + e * Rt + rows64, mask=rm, other=0)
    tl.store(ACC + e * Rt + rows64, old + acc, mask=rm)
    if GATE != 0:
        old = tl.load(REJ + e * Rt + rows64, mask=rm, other=0)
        tl.store(REJ + e * Rt + rows64, old + rej, mask=rm)


def launch_iq1s_flip_block(Q, GB, D, HCC, MULT, CODE, ACC, REJ, ACT, GRP, tabs, Rm, tol, cfg, gate, margin):
    E, Rt, K = Q.shape
    assert GB.shape[2] % 32 == 0 and cfg.get("CHF", 32) >= 16
    grid = (triton.cdiv(Rt, cfg["BR"]), E)
    _iq1s_flip_kernel[grid](Q, GB, D, HCC, MULT, CODE, ACC, REJ, ACT, GRP, tabs["CBG"], tabs["GF"],
                            Rt, K, MULT.shape[2], Rm, ACT.shape[1], tol, float(margin),
                            NB=GRP.numel(), BSZ=GB.shape[2], BR=cfg["BR"], BC=cfg["BC"], CH=cfg.get("CHF", 32),
                            PREC=cfg.get("prec", "ieee"), GATE=int(gate),
                            num_warps=cfg.get("warpsF", cfg["warps"]), num_stages=cfg.get("stages", 1))
