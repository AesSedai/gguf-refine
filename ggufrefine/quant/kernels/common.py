"""Triton helpers shared by the three kernel families: candidate features for the 8-wide quadratic score
(_feat_codebook on the host, _feat_rows per program), the fp32 acceptance gate, and the 8-wide argmin / quadratic
form used by the parity and ternary searches."""
import torch
import triton
import triton.language as tl


def _feat_codebook(U):
    """U [n, 8] -> [64, n]: feature a*8+b = U_a U_b (a<=b), U_a at (a, a-1), U_0 at (7, 0), else 0"""
    n = U.shape[0]
    F = torch.zeros(n, 8, 8, dtype=torch.float64)
    a = torch.arange(8)
    up = a[:, None] <= a[None, :]
    F[:, up] = (U[:, :, None] * U[:, None, :])[:, up]
    for k in range(1, 8):
        F[:, k, k - 1] = U[:, k]
    F[:, 7, 0] = U[:, 0]
    return F.reshape(n, 64).T.contiguous().float()


GATE_ARITH = tl.constexpr(4e-6)   # fp32 gate: bound on the rounding error of delta relative to `scale` (~64 u_fp32; the
                      # computation below has <= ~20 roundings per term chain)


@triton.jit
def _gate(ok, u, v, v0, gb, A, margin, MODE: tl.constexpr):
    """accept gate for one move v -> u of a group/chunk (GATE constexpr: 0 off, 1 fp32, 2 fp64).
    v0, gb = values / gradient at the start of the item; the current in-kernel gradient is
    g_cur = gb - (v - v0) A, and with d = u - v the exact objective change given that gradient is
        delta = d A d - 2 d . g_cur
    (direct form: no cancellation against the large v A terms that make the search's fp32 S values
    close). The move is kept only if
        delta < -thr,   thr = (margin [+ GATE_ARITH in fp32 mode]) * scale,
        scale = sum_ab |d_a| |A_ab| |d_b| + 2 sum_a |d_a| (|gb_a| + sum_b |A_ab| |v_b - v0_b|)
    scale bounds every term that enters delta, so in fp32 mode GATE_ARITH covers the arithmetic
    error of delta and margin is left for the error of the maintained fp32 gradient gb itself.
    Returns (accept, rejection code): code = 1 per rejection + 65536 if the rejected move was not a
    no-op (the fp32 search can "accept" u == v when the feature-form score of the current codeword
    rounds below cur; such moves change nothing)."""
    if MODE == 2:
        Ax = A.to(tl.float64)
        dl = u.to(tl.float64) - v.to(tl.float64)
        dv = v.to(tl.float64) - v0.to(tl.float64)
        gbx = gb.to(tl.float64)
        thr_rel = margin
    else:
        Ax = A
        dl = u - v
        dv = v - v0
        gbx = gb
        thr_rel = margin + GATE_ARITH
    # delta = d A d - 2 d.(gb - dv A) = d . A (d + 2 dv) - 2 d.gb   (A symmetric): one matvec
    da = tl.abs(dl)
    delta = tl.sum(tl.sum((dl + 2.0 * dv)[:, :, None] * Ax[None, :, :], axis=1) * dl, axis=1) - 2.0 * tl.sum(dl * gbx, axis=1)
    scale = (tl.sum(tl.sum((da + 2.0 * tl.abs(dv))[:, :, None] * tl.abs(Ax)[None, :, :], axis=1) * da, axis=1)
             + 2.0 * tl.sum(da * tl.abs(gbx), axis=1))
    okg = delta < -(thr_rel * scale)
    rj = ok & (okg == 0)
    real = tl.sum(da, axis=1) > 0                                             # rejected move changes the codes
    return ok & okg, rj.to(tl.int32) + 65536 * (rj & real).to(tl.int32)


@triton.jit
def _feat_rows(U, L, A, m2, a8r, a8c):
    """row-side features [BR, 64] matching _feat_codebook: coef * m2 * U_a U_b A_ab on the upper
    triangle (2 off-diagonal), L_a at (a, a-1), L_0 at (7, 0)"""
    ar = a8r[None, :, None]
    ac = a8c[None, None, :]
    quad = (U[:, :, None] * U[:, None, :]) * A[None, :, :] * m2[:, None, None]
    coef = tl.where(ar < ac, 2.0, tl.where(ar == ac, 1.0, 0.0))
    lin_a = tl.where(ac == ar - 1, 1.0, 0.0) * L[:, :, None]
    lin_0 = tl.where((ar == 7) & (ac == 0), 1.0, 0.0) * L[:, None, :]
    X = coef * quad + lin_a + lin_0
    return tl.reshape(X, [X.shape[0], 64])


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
