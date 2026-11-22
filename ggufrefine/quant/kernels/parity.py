"""Triton kernel for batched Hessian codeword refinement of the parity-sign IQ types
(IQ2_XXS, IQ2_XS, IQ3_XXS) (ggufrefine.refine.engine). Same launch structure as kernels.iq: one
launch = one block of the per-sweep visiting order, all rows of all E experts, each program owns
BR rows of one expert and walks the block's NB items sequentially; the caller applies the deferred
update of the other columns as one batched GEMM per block.

The item is an 8-weight SIGN CHUNK (one 7-bit sign field + implied even-parity 8th sign), visited
in a random order (one permutation of the K/8 chunks per sweep). Per chunk:
  magnitude step  per grid group k of the chunk (IQ2_*: one 8-dim group, IQ3_XXS: two 4-dim groups,
                  in order), signs fixed, exhaustive over the grid (256 / 512 / 256 entries)
  sign step       whole chunk, magnitudes fixed, exhaustive over the 128 legal patterns, full 8x8
                  block of Hf (IQ3_XXS: cross terms between its two groups included)
All scores use one 8-wide formulation. With v the current chunk values, A = Hf[B, B] (8x8) and
q = v A + g_B (invariant to moves inside B), a candidate u that differs from v only on the
positions M of the step scores
      S(u) = u_M A_MM u_M - 2 u_M . p_M,      p = q - (v masked to ~M) A
and S(u) - S(v) is the exact objective change. Magnitude step: features of U = sig masked to M
against a codebook of grid vectors placed at M (zeros elsewhere); sign step: U = mu (magnitudes)
against the 128 legal +-1 patterns. Both are one [BR, 64] x [64, ncand] tl.dot per candidate tile
(_feat_rows / _feat_codebook from kernels.iq: 36 quadratic + 8 linear feature slots).

Acceptance: (1) the fp32 test of the reference refiners, gain = cur - S_best > tol * |cur|;
(2) with GATE != 0, kernels.iq._gate: the chosen move's objective change is recomputed in the
direct form delta = d A d - 2 d . g_cur (d = u - v, g_cur = g_B - (v - v_chunk_start) A), which has
no cancellation against the large v A terms that make the fp32 S values close, and the move is
kept only if delta < -(margin + arithmetic bound) * scale. GATE=1 evaluates it in fp32 with a
rigorous rounding bound (GATE_ARITH), GATE=2 in fp64. Either way the input gradient g_B is the
fp32-maintained one, so the gate rules out moves that are uphill w.r.t. that gradient; the
residual vs the exact objective is measured by engine.ShadowCheck. Rejections are counted in
REJ (low 16 bits: all, high bits: those that were not no-ops).

Codes: IQ2_*: code = grid_idx << 8 | sign8 (one per chunk); IQ3_XXS: per 4-dim group,
code = grid_idx << 4 | sign4 (chunk c = groups 2c, 2c+1; its sign8 = lo nibble of 2c | hi of 2c+1).
Sign patterns written are always legal (taken from LEG = ksigns_iq2xs).
All search arithmetic FP32 (tl.dot input_precision="ieee": no TF32).
"""
import numpy as np
import torch
import triton
import triton.language as tl

from .iq import _feat_codebook, _feat_rows, _gate


def parity_tables(grid, ksigns, dev):
    """grid [ncode, gs] magnitudes, ksigns [128] uint8 (bit a = sign of weight a, 1 = negative) ->
    CBM [gpc, 64, ncode] magnitude codebooks (grid placed at group k's positions), CBS [64, 128] legal
    sign-pattern codebook (row order = 7-bit field value), GT8 [gpc, 8, ncode] zero-padded grids,
    LEG [128] int32 8-bit patterns"""
    g = torch.as_tensor(grid, dtype=torch.float64)
    n, gs = g.shape
    gpc = 8 // gs
    G8 = torch.zeros(gpc, n, 8, dtype=torch.float64)
    for k in range(gpc):
        G8[k, :, k * gs:(k + 1) * gs] = g
    bits = (torch.as_tensor(ksigns.astype(np.int64))[:, None] >> torch.arange(8)) & 1
    S = (1.0 - 2.0 * bits).double()                                            # [128, 8]
    return dict(CBM=torch.stack([_feat_codebook(G8[k]) for k in range(gpc)]).contiguous().to(dev),
                CBS=_feat_codebook(S).to(dev),
                GT8=G8.transpose(1, 2).contiguous().float().to(dev),
                LEG=torch.as_tensor(ksigns.astype(np.int32)).to(dev))


@triton.jit
def _argmin_dot(X, CB, NCAND: tl.constexpr, BC: tl.constexpr, BR: tl.constexpr, PREC: tl.constexpr):
    """best (value, lowest index) of X [BR, 64] . CB[:, c] over c < NCAND (CB row-major [64, NCAND])"""
    f64 = tl.arange(0, 64)
    best_v = tl.full([BR], float("inf"), tl.float32)
    best_i = tl.zeros([BR], dtype=tl.int32)
    for c0 in range(0, NCAND, BC):
        ci = c0 + tl.arange(0, BC)
        cb = tl.load(CB + f64[:, None] * NCAND + ci[None, :])                 # [64, BC]
        S = tl.dot(X, cb, input_precision=PREC)
        tmin = tl.min(S, axis=1)
        targ = tl.min(tl.where(S == tmin[:, None], ci[None, :], NCAND), axis=1)
        upd = tmin < best_v
        best_v = tl.where(upd, tmin, best_v)
        best_i = tl.where(upd, targ, best_i)
    return best_v, best_i


@triton.jit
def _quad8(v, A):
    """v [BR, 8] -> v A v [BR]"""
    return tl.sum(tl.sum(v[:, :, None] * A[None, :, :], axis=1) * v, axis=1)


@triton.jit
def _parity_block_kernel(Q, GB, D, HCC, MULT, CODE, ACC, REJ, ACT, GRP, CBM, CBS, GT8, LEG, DV1,
                         Rt, K, NG, Rm, NMAT, tol, margin,
                         NB: tl.constexpr, BSZ: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr,
                         BCS: tl.constexpr, CH: tl.constexpr, NCODE: tl.constexpr, GS: tl.constexpr,
                         GPC: tl.constexpr, PREC: tl.constexpr, GATE: tl.constexpr, DBG: tl.constexpr):
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
    acc = tl.zeros([BR], dtype=tl.int32)
    rej = tl.zeros([BR], dtype=tl.int32)
    for jl in range(NB):
        j = tl.load(GRP + jl)                                                  # chunk id
        col0 = j * 8
        l0 = jl * 8
        v0 = tl.load(qrow[:, None] + col0 + a8[None, :], mask=rm[:, None], other=0.0)      # [BR, 8]
        gb = tl.load(gbrow[:, None] + l0 + a8[None, :], mask=rm[:, None], other=0.0)
        m = tl.load(mrow + j * GPC, mask=rm, other=0.0)                       # chunk lies in one sub-block
        A = tl.load(hbase + (l0 + a8[:, None]) * BSZ + l0 + a8[None, :])                   # [8, 8]
        q = tl.sum(v0[:, :, None] * A[None, :, :], axis=1) + gb
        if GPC == 1:
            c0 = tl.load(crow + j, mask=rm, other=0)
            sp = c0 & 255
            gi0 = c0 >> 8
            gi1 = gi0
        else:
            c0 = tl.load(crow + 2 * j, mask=rm, other=0)
            c1 = tl.load(crow + 2 * j + 1, mask=rm, other=0)
            sp = (c0 & 15) | ((c1 & 15) << 4)
            gi0 = c0 >> 4
            gi1 = c1 >> 4
        sig = 1.0 - 2.0 * ((sp[:, None] >> a8[None, :]) & 1).to(tl.float32)                # [BR, 8]
        v = v0
        nacc = tl.zeros([BR], dtype=tl.int32)
        # ---- magnitude step(s), signs fixed
        for k in tl.static_range(GPC):
            mk = ((a8 >= k * GS) & (a8 < (k + 1) * GS))[None, :]
            vk = tl.where(mk, v, 0.0)
            vo = tl.where(mk, 0.0, v)
            p = q - tl.sum(vo[:, :, None] * A[None, :, :], axis=1)
            cur = _quad8(vk, A) - 2.0 * tl.sum(vk * p, axis=1)
            U = tl.where(mk, sig, 0.0)
            X = _feat_rows(U, -2.0 * m[:, None] * U * p, A, m * m, a8, a8)
            best_v, best_i = _argmin_dot(X, CBM + k * 64 * NCODE, NCODE, BC, BR, PREC)
            ok = live & ((cur - best_v) > tol * tl.maximum(tl.abs(cur), 1e-30))
            mag = tl.load(GT8 + k * 8 * NCODE + a8[None, :] * NCODE + best_i[:, None]) * m[:, None]
            u = tl.where(mk, mag * sig, v)
            if GATE != 0:
                ok, rj = _gate(ok, u, v, v0, gb, A, margin, GATE)
                rej += rj.to(tl.int32)
            if k == 0:
                gi0 = tl.where(ok, best_i, gi0)
            else:
                gi1 = tl.where(ok, best_i, gi1)
            v = tl.where(ok[:, None], u, v)
            nacc += ok.to(tl.int32)
        if DBG:
            tl.store(DV1 + e * Rt * BSZ + rows64[:, None] * BSZ + l0 + a8[None, :], v, mask=rm[:, None])
        # ---- sign step over the 128 legal patterns, magnitudes fixed
        mu = tl.load(GT8 + a8[None, :] * NCODE + gi0[:, None]) * m[:, None]
        if GPC == 2:
            mu += tl.load(GT8 + 8 * NCODE + a8[None, :] * NCODE + gi1[:, None]) * m[:, None]
        cur1 = _quad8(v, A) - 2.0 * tl.sum(v * q, axis=1)
        X = _feat_rows(mu, -2.0 * mu * q, A, tl.full([BR], 1.0, tl.float32), a8, a8)
        best_v, best_i = _argmin_dot(X, CBS, 128, BCS, BR, PREC)
        pat = tl.load(LEG + best_i)
        u = mu * (1.0 - 2.0 * ((pat[:, None] >> a8[None, :]) & 1).to(tl.float32))
        ok = live & ((cur1 - best_v) > tol * tl.maximum(tl.abs(cur1), 1e-30))
        if GATE != 0:
            ok, rj = _gate(ok, u, v, v0, gb, A, margin, GATE)
            rej += rj.to(tl.int32)
        sp = tl.where(ok, pat, sp)
        v = tl.where(ok[:, None], u, v)
        nacc += ok.to(tl.int32)
        # ---- write back
        chg = nacc > 0
        d = tl.where(chg[:, None], v - v0, 0.0)
        tl.store(qrow[:, None] + col0 + a8[None, :], v, mask=chg[:, None])
        tl.store(drow[:, None] + l0 + a8[None, :], d, mask=rm[:, None])
        if GPC == 1:
            tl.store(crow + j, (gi0 << 8) | sp, mask=chg)
        else:
            tl.store(crow + 2 * j, (gi0 << 4) | (sp & 15), mask=chg)
            tl.store(crow + 2 * j + 1, (gi1 << 4) | (sp >> 4), mask=chg)
        acc += nacc
        # in-block gradient for the chunks still to come
        lo = l0 + 8
        for c0 in range(0, BSZ, CH):
            if c0 + CH > lo:
                cc = c0 + tl.arange(0, CH)
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


def launch_parity_block(Q, GB, D, HCC, MULT, CODE, ACC, REJ, ACT, GRP, tabs, Rm, tol, cfg, gate, margin, DV1=None):
    E, Rt, K = Q.shape
    NG = MULT.shape[2]
    NB = GRP.numel()
    BSZ = GB.shape[2]
    BR = cfg["BR"]
    gpc, _, ncode = tabs["GT8"].shape
    grid = (triton.cdiv(Rt, BR), E)
    _parity_block_kernel[grid](Q, GB, D, HCC, MULT, CODE, ACC, REJ, ACT, GRP,
                               tabs["CBM"], tabs["CBS"], tabs["GT8"], tabs["LEG"], D if DV1 is None else DV1,
                               Rt, K, NG, Rm, ACT.shape[1], tol, margin,
                               NB=NB, BSZ=BSZ, BR=BR, BC=min(cfg["BC"], ncode), BCS=min(cfg.get("BCS", cfg["BC"]), 128),
                               CH=cfg["CH"], NCODE=ncode, GS=8 // gpc, GPC=gpc, PREC=cfg.get("prec", "ieee"),
                               GATE=int(gate), DBG=DV1 is not None,
                               num_warps=cfg["warps"], num_stages=cfg.get("stages", 1))
