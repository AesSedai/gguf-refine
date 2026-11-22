"""Triton kernels for batched Hessian codeword refinement (ggufrefine.refine.engine).

One launch processes ONE BLOCK of the visiting order (NB consecutive groups of the per-sweep
permutation, BSZ = NB * GS gathered columns) for all rows of all E experts. The objective
sum_i r_i Hf r_i^T separates over rows, so each program owns BR rows of one expert and walks the
NB groups sequentially with no inter-program communication:

  per group j (global group id, columns col0 = j*GS; local columns l0 = jl*GS in the gathered block)
    v   = Q[row, col0:col0+GS]                        current reconstruction
    q   = v A + g_blk[row, l0:l0+GS]                  A = Hf[B, B]  (q is invariant to moves in B)
    cur = v A v - 2 v.q                               score of the current codeword
    exhaustive search  S(u) = u A u - 2 u.q  over the candidate set, argmin with lowest-index ties
    accept iff cur - S_best > tol * |cur|             (same test as the fp64 reference)
    Q <- new, D_blk[row, l0:] <- new - v, code <- best
    g_blk[row, later columns of the block] -= d A_{B, block}      (exact in-block gradient)

The caller applies the deferred update for all K columns once per block as one batched GEMM
(g -= D_blk @ Hf[C, :]); see engine.FastRefiner.

IQ3_S: 8192 signed codewords (code = grid_idx*16 + sign4), S = m^2 T_c - 2m q.C_c with
       T_c = C_c A C_c computed per program, 5 FMA per (row, codeword), registers only.
IQ2_S: magnitude step (1024 codewords, signs fixed) then sign step (256 patterns, magnitudes fixed),
       each as one [BR, 64] x [64, ncand] dot (default input_precision="ieee" = FP32 FMA; "tf32x3" is
       ~20% faster and then the chosen candidate is re-scored in FP32 for the acceptance test): the 8x8 quadratic form uses the upper triangle
       (coefficient 2 off-diagonal) and the 8 linear terms live in 8 lower-triangle slots
       ((a, a-1) for a=1..7 and (7, 0) for a=0), so a single K=64 dot gives the exact score.
All arithmetic FP32 (tl.dot input_precision="ieee": no TF32).
"""
import torch
import triton
import triton.language as tl

from .common import _feat_codebook, _feat_rows, _gate


# ---------------------------------------------------------------------------------------- tables
def iq3_tables(grid, dev):
    """grid [512, 4] magnitudes -> CS [4, 8192] signed codewords, index = code*16 + sign_pattern"""
    g = torch.as_tensor(grid, dtype=torch.float32)
    sp = torch.arange(16)
    s = 1.0 - 2.0 * ((sp[:, None] >> torch.arange(4)) & 1).float()          # [16, 4]
    C = (g[:, None, :] * s[None]).reshape(-1, 4)                             # [8192, 4]
    return C.T.contiguous().to(dev)


def iq2_tables(grid, dev):
    g = torch.as_tensor(grid, dtype=torch.float64)                            # [1024, 8]
    sp = torch.arange(256)
    s = 1.0 - 2.0 * ((sp[:, None] >> torch.arange(8)) & 1).double()          # [256, 8]
    return dict(CBm=_feat_codebook(g).to(dev), CBs=_feat_codebook(s).to(dev),
                Gt=g.T.contiguous().float().to(dev))                          # Gt [8, 1024]


# ---------------------------------------------------------------------------------------- IQ3_S
@triton.jit
def _iq3_block_kernel(Q, GB, D, HCC, MULT, CODE, ACC, ACT, GRP, CS, REJ,
                      Rt, K, NG, Rm, NMAT, tol, margin,
                      NB: tl.constexpr, BSZ: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr,
                      CH: tl.constexpr, NCODE: tl.constexpr, GATE: tl.constexpr):
    pid = tl.program_id(0)
    e = tl.program_id(1).to(tl.int64)
    rows = pid * BR + tl.arange(0, BR)
    rm = rows < Rt
    rows64 = rows.to(tl.int64)
    a4 = tl.arange(0, 4)
    act = tl.load(ACT + e * NMAT + rows // Rm, mask=rm, other=0)
    live = rm & (act != 0)
    qrow = Q + e * Rt * K + rows64 * K                                        # [BR] row pointers
    gbrow = GB + e * Rt * BSZ + rows64 * BSZ
    drow = D + e * Rt * BSZ + rows64 * BSZ
    hbase = HCC + e * BSZ * BSZ
    acc = tl.zeros([BR], dtype=tl.int32)
    rej = tl.zeros([BR], dtype=tl.int32)
    for jl in range(NB):
        j = tl.load(GRP + jl)
        col0 = j * 4
        l0 = jl * 4
        v = tl.load(qrow[:, None] + col0 + a4[None, :], mask=rm[:, None], other=0.0)      # [BR, 4]
        gb = tl.load(gbrow[:, None] + l0 + a4[None, :], mask=rm[:, None], other=0.0)
        m = tl.load(MULT + e * Rt * NG + rows64 * NG + j, mask=rm, other=0.0)
        hp = hbase + (l0 + a4[:, None]) * BSZ + l0 + a4[None, :]
        A = tl.load(hp)                                                                    # [4, 4]
        vA = tl.sum(v[:, :, None] * A[None, :, :], axis=1)                                 # [BR, 4]
        q = vA + gb
        cur = tl.sum(vA * v, axis=1) - 2.0 * tl.sum(v * q, axis=1)
        # scalars of A (symmetric)
        a00 = tl.load(hbase + (l0 + 0) * BSZ + l0 + 0)
        a11 = tl.load(hbase + (l0 + 1) * BSZ + l0 + 1)
        a22 = tl.load(hbase + (l0 + 2) * BSZ + l0 + 2)
        a33 = tl.load(hbase + (l0 + 3) * BSZ + l0 + 3)
        a01 = tl.load(hbase + (l0 + 0) * BSZ + l0 + 1)
        a02 = tl.load(hbase + (l0 + 0) * BSZ + l0 + 2)
        a03 = tl.load(hbase + (l0 + 0) * BSZ + l0 + 3)
        a12 = tl.load(hbase + (l0 + 1) * BSZ + l0 + 2)
        a13 = tl.load(hbase + (l0 + 1) * BSZ + l0 + 3)
        a23 = tl.load(hbase + (l0 + 2) * BSZ + l0 + 3)
        m2 = m * m
        l_0 = -2.0 * m * tl.sum(tl.where(a4[None, :] == 0, q, 0.0), axis=1)
        l_1 = -2.0 * m * tl.sum(tl.where(a4[None, :] == 1, q, 0.0), axis=1)
        l_2 = -2.0 * m * tl.sum(tl.where(a4[None, :] == 2, q, 0.0), axis=1)
        l_3 = -2.0 * m * tl.sum(tl.where(a4[None, :] == 3, q, 0.0), axis=1)
        best_v = tl.full([BR], float("inf"), tl.float32)
        best_i = tl.zeros([BR], dtype=tl.int32)
        for c0 in range(0, NCODE, BC):
            ci = c0 + tl.arange(0, BC)
            c_0 = tl.load(CS + ci)
            c_1 = tl.load(CS + NCODE + ci)
            c_2 = tl.load(CS + 2 * NCODE + ci)
            c_3 = tl.load(CS + 3 * NCODE + ci)
            T = (a00 * c_0 * c_0 + a11 * c_1 * c_1 + a22 * c_2 * c_2 + a33 * c_3 * c_3
                 + 2.0 * (a01 * c_0 * c_1 + a02 * c_0 * c_2 + a03 * c_0 * c_3
                          + a12 * c_1 * c_2 + a13 * c_1 * c_3 + a23 * c_2 * c_3))
            S = (m2[:, None] * T[None, :] + l_0[:, None] * c_0[None, :] + l_1[:, None] * c_1[None, :]
                 + l_2[:, None] * c_2[None, :] + l_3[:, None] * c_3[None, :])
            tmin = tl.min(S, axis=1)
            targ = tl.min(tl.where(S == tmin[:, None], ci[None, :], NCODE), axis=1)
            upd = tmin < best_v
            best_v = tl.where(upd, tmin, best_v)
            best_i = tl.where(upd, targ, best_i)
        gain = cur - best_v
        ok = live & (gain > tol * tl.maximum(tl.abs(cur), 1e-30))
        new = tl.load(CS + a4[None, :] * NCODE + best_i[:, None]) * m[:, None]
        if GATE != 0:
            ok, rj = _gate(ok, new, v, v, gb, A, margin, GATE)
            rej += rj.to(tl.int32)
        d = tl.where(ok[:, None], new - v, 0.0)
        tl.store(qrow[:, None] + col0 + a4[None, :], new, mask=ok[:, None])
        tl.store(drow[:, None] + l0 + a4[None, :], d, mask=rm[:, None])
        tl.store(CODE + e * Rt * NG + rows64 * NG + j, best_i, mask=ok)
        acc += ok.to(tl.int32)
        # in-block gradient for the groups still to come: g_blk[:, c] -= sum_a d_a Hf[B_a, c]
        d_0 = tl.sum(tl.where(a4[None, :] == 0, d, 0.0), axis=1)
        d_1 = tl.sum(tl.where(a4[None, :] == 1, d, 0.0), axis=1)
        d_2 = tl.sum(tl.where(a4[None, :] == 2, d, 0.0), axis=1)
        d_3 = tl.sum(tl.where(a4[None, :] == 3, d, 0.0), axis=1)
        lo = l0 + 4
        for c0 in range(0, BSZ, CH):
            if c0 + CH > lo:
                cc = c0 + tl.arange(0, CH)
                h0 = tl.load(hbase + (l0 + 0) * BSZ + cc)
                h1 = tl.load(hbase + (l0 + 1) * BSZ + cc)
                h2 = tl.load(hbase + (l0 + 2) * BSZ + cc)
                h3 = tl.load(hbase + (l0 + 3) * BSZ + cc)
                gp = gbrow[:, None] + cc[None, :]
                gt = tl.load(gp, mask=rm[:, None], other=0.0)
                gt = gt - (d_0[:, None] * h0[None, :] + d_1[:, None] * h1[None, :]
                           + d_2[:, None] * h2[None, :] + d_3[:, None] * h3[None, :])
                tl.store(gp, gt, mask=rm[:, None] & (cc[None, :] >= lo))
        tl.debug_barrier()
    old = tl.load(ACC + e * Rt + rows64, mask=rm, other=0)
    tl.store(ACC + e * Rt + rows64, old + acc, mask=rm)
    if GATE != 0:
        old = tl.load(REJ + e * Rt + rows64, mask=rm, other=0)
        tl.store(REJ + e * Rt + rows64, old + rej, mask=rm)


# ---------------------------------------------------------------------------------------- IQ2_S
@triton.jit
def _iq2_block_kernel(Q, GB, D, HCC, MULT, CODE, ACC, ACT, GRP, CBM, CBS, GT, REJ,
                      Rt, K, NG, Rm, NMAT, tol, margin,
                      NB: tl.constexpr, BSZ: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr,
                      CH: tl.constexpr, PREC: tl.constexpr, RESCORE: tl.constexpr, GATE: tl.constexpr):
    pid = tl.program_id(0)
    e = tl.program_id(1).to(tl.int64)
    rows = pid * BR + tl.arange(0, BR)
    rm = rows < Rt
    rows64 = rows.to(tl.int64)
    a8 = tl.arange(0, 8)
    f64 = tl.arange(0, 64)
    act = tl.load(ACT + e * NMAT + rows // Rm, mask=rm, other=0)
    live = rm & (act != 0)
    qrow = Q + e * Rt * K + rows64 * K
    gbrow = GB + e * Rt * BSZ + rows64 * BSZ
    drow = D + e * Rt * BSZ + rows64 * BSZ
    hbase = HCC + e * BSZ * BSZ
    acc = tl.zeros([BR], dtype=tl.int32)
    rej = tl.zeros([BR], dtype=tl.int32)
    for jl in range(NB):
        j = tl.load(GRP + jl)
        col0 = j * 8
        l0 = jl * 8
        v = tl.load(qrow[:, None] + col0 + a8[None, :], mask=rm[:, None], other=0.0)       # [BR, 8]
        gb = tl.load(gbrow[:, None] + l0 + a8[None, :], mask=rm[:, None], other=0.0)
        m = tl.load(MULT + e * Rt * NG + rows64 * NG + j, mask=rm, other=0.0)
        code = tl.load(CODE + e * Rt * NG + rows64 * NG + j, mask=rm, other=0)
        A = tl.load(hbase + (l0 + a8[:, None]) * BSZ + l0 + a8[None, :])                   # [8, 8]
        vA = tl.sum(v[:, :, None] * A[None, :, :], axis=1)
        q = vA + gb
        cur = tl.sum(vA * v, axis=1) - 2.0 * tl.sum(v * q, axis=1)
        sp = code & 255
        gi = code >> 8
        sig = 1.0 - 2.0 * ((sp[:, None] >> a8[None, :]) & 1).to(tl.float32)                # [BR, 8]
        # ---- magnitude step (signs fixed): u = m * sig * G_c
        X = _feat_rows(sig, -2.0 * m[:, None] * sig * q, A, m * m, a8, a8)
        best_v = tl.full([BR], float("inf"), tl.float32)
        best_i = tl.zeros([BR], dtype=tl.int32)
        for c0 in range(0, 1024, BC):
            ci = c0 + tl.arange(0, BC)
            cb = tl.load(CBM + f64[:, None] * 1024 + ci[None, :])                          # [64, BC]
            S = tl.dot(X, cb, input_precision=PREC)
            tmin = tl.min(S, axis=1)
            targ = tl.min(tl.where(S == tmin[:, None], ci[None, :], 1024), axis=1)
            upd = tmin < best_v
            best_v = tl.where(upd, tmin, best_v)
            best_i = tl.where(upd, targ, best_i)
        if RESCORE:                                            # exact FP32 score of the chosen candidate
            u = tl.load(GT + a8[None, :] * 1024 + best_i[:, None]) * sig * m[:, None]
            best_v = tl.sum(tl.sum(u[:, :, None] * A[None, :, :], axis=1) * u, axis=1) - 2.0 * tl.sum(u * q, axis=1)
        ok1 = live & ((cur - best_v) > tol * tl.maximum(tl.abs(cur), 1e-30))
        if GATE != 0:
            u1 = (tl.load(GT + a8[None, :] * 1024 + best_i[:, None]) * m[:, None]) * sig
            ok1, rj = _gate(ok1, u1, v, v, gb, A, margin, GATE)
            rej += rj.to(tl.int32)
        gi = tl.where(ok1, best_i, gi)
        mu = tl.load(GT + a8[None, :] * 1024 + gi[:, None]) * m[:, None]                   # [BR, 8]
        v1 = tl.where(ok1[:, None], mu * sig, v)
        cur1 = tl.sum(tl.sum(v1[:, :, None] * A[None, :, :], axis=1) * v1, axis=1) - 2.0 * tl.sum(v1 * q, axis=1)
        # ---- sign step (magnitudes fixed): u = mu * s_p
        X = _feat_rows(mu, -2.0 * mu * q, A, tl.full([BR], 1.0, tl.float32), a8, a8)
        best_v = tl.full([BR], float("inf"), tl.float32)
        best_i = tl.zeros([BR], dtype=tl.int32)
        for c0 in range(0, 256, BC):
            ci = c0 + tl.arange(0, BC)
            cb = tl.load(CBS + f64[:, None] * 256 + ci[None, :])
            S = tl.dot(X, cb, input_precision=PREC)
            tmin = tl.min(S, axis=1)
            targ = tl.min(tl.where(S == tmin[:, None], ci[None, :], 256), axis=1)
            upd = tmin < best_v
            best_v = tl.where(upd, tmin, best_v)
            best_i = tl.where(upd, targ, best_i)
        if RESCORE:
            u = mu * (1.0 - 2.0 * ((best_i[:, None] >> a8[None, :]) & 1).to(tl.float32))
            best_v = tl.sum(tl.sum(u[:, :, None] * A[None, :, :], axis=1) * u, axis=1) - 2.0 * tl.sum(u * q, axis=1)
        ok2 = live & ((cur1 - best_v) > tol * tl.maximum(tl.abs(cur1), 1e-30))
        if GATE != 0:
            u2 = mu * (1.0 - 2.0 * ((best_i[:, None] >> a8[None, :]) & 1).to(tl.float32))
            ok2, rj = _gate(ok2, u2, v1, v, gb, A, margin, GATE)
            rej += rj.to(tl.int32)
        sp = tl.where(ok2, best_i, sp)
        sig2 = 1.0 - 2.0 * ((sp[:, None] >> a8[None, :]) & 1).to(tl.float32)
        v2 = tl.where(ok2[:, None], mu * sig2, v1)
        chg = ok1 | ok2
        d = tl.where(chg[:, None], v2 - v, 0.0)
        tl.store(qrow[:, None] + col0 + a8[None, :], v2, mask=chg[:, None])
        tl.store(drow[:, None] + l0 + a8[None, :], d, mask=rm[:, None])
        tl.store(CODE + e * Rt * NG + rows64 * NG + j, (gi << 8) | sp, mask=chg)
        acc += ok1.to(tl.int32) + ok2.to(tl.int32)
        lo = l0 + 8
        for c0 in range(0, BSZ, CH):
            if c0 + CH > lo:
                cc = c0 + tl.arange(0, CH)
                hb = tl.load(hbase + (l0 + a8[:, None]) * BSZ + cc[None, :])               # [8, CH]
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


def launch_block(t, Q, GB, D, HCC, MULT, CODE, ACC, ACT, GRP, tabs, Rm, tol, cfg, REJ=None, gate=False, margin=0.0):
    """gate: 0/False (default) compiles the kernels without the accept gate (the verified path), 1 fp32 gate, 2 fp64 gate"""
    E, Rt, K = Q.shape
    NG = MULT.shape[2]
    NB = GRP.numel()
    BSZ = GB.shape[2]
    BR = cfg["BR"]
    grid = (triton.cdiv(Rt, BR), E)
    if t == "IQ3_S":
        _iq3_block_kernel[grid](Q, GB, D, HCC, MULT, CODE, ACC, ACT, GRP, tabs["CS"], ACC if REJ is None else REJ,
                                Rt, K, NG, Rm, ACT.shape[1], tol, float(margin),
                                NB=NB, BSZ=BSZ, BR=BR, BC=cfg["BC"], CH=cfg["CH"], NCODE=8192, GATE=int(gate),
                                num_warps=cfg["warps"], num_stages=cfg.get("stages", 1))
    else:
        _iq2_block_kernel[grid](Q, GB, D, HCC, MULT, CODE, ACC, ACT, GRP, tabs["CBm"], tabs["CBs"], tabs["Gt"],
                                ACC if REJ is None else REJ, Rt, K, NG, Rm, ACT.shape[1], tol, float(margin),
                                NB=NB, BSZ=BSZ, BR=BR, BC=cfg["BC"], CH=cfg["CH"], PREC=cfg.get("prec", "ieee"),
                                RESCORE=bool(cfg.get("rescore", cfg.get("prec", "ieee") != "ieee")), GATE=int(gate),
                                num_warps=cfg["warps"], num_stages=cfg.get("stages", 1))
