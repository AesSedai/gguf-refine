"""Field-level unpack/pack for ggml IQ2_S and IQ3_S blocks (QK_K = 256), vectorized.

Layouts (ggml-common.h / gguf-py quants.py):
  IQ3_S (110 B): d f16 | qs[64]  idx low 8 bits, 64 groups of 4 | qh[8]  idx bit 8, bit i of byte k -> group 8k+i
                 | signs[32] bit j of byte b -> weight 8b+j | scales[4] nibble per 32 weights (lo = even sub-block)
                 w = d * (1 + 2*s) * grid[idx][pos] * sign
  IQ2_S  (82 B): d f16 | qs[32]  idx low 8 bits, 32 groups of 8 | signs[32] | qh[8] 2 bits per group,
                 byte k holds groups 4k..4k+3 at shifts 0,2,4,6 | scales[8] nibble per 16 weights
                 w = d * (0.5 + s) * 0.25 * grid[idx][pos] * sign
Fields returned per block: d [n] f32, scales [n, n_sub] u8, idx [n, n_groups] u16, signs [n, 256] bool (True = negative).

Parity-sign types (7 stored sign bits per 8 weights; ksigns_iq2xs[s7] = s7 | parity(s7) << 7, so every
8-weight chunk has an EVEN number of negatives; pack() asserts this):
  IQ2_XXS (66 B): d f16 | per 32 weights (8 B): 4 grid idx bytes (8-dim groups), then u32 LE with
                  sign7 of group l at bits 7l..7l+6 and the scale nibble at bits 28..31
                  w = d * (0.5 + s) * 0.25 * grid[idx][pos] * sign          (scale per 32 weights)
  IQ2_XS  (74 B): d f16 | qs u16[32] = idx (9 bits) | sign7 << 9, one per 8-dim group | scales[8] nibble
                  per 16 weights (lo = even)    w = d * (0.5 + s) * 0.25 * grid[idx][pos] * sign
  IQ3_XXS (98 B): d f16 | qs[64] grid idx byte per 4-dim group | u32 LE[8], one per 32 weights: sign7 of
                  8-weight chunk l (= 4-dim groups 2l, 2l+1) at bits 7l.., scale nibble at bits 28..31
                  w = d * (0.5 + s) * 0.5 * grid[idx][pos] * sign

IQ1 types (no sign bits; grid values in {-1, 0, 1}, 2048 8-dim rows, plus a +-0.125 delta): see the
IQ1_S / IQ1_M section at the end. Fields: d, d_raw, scales, idx (11 bits per 8-weight group),
delta [n, 32] bool per group (True = -0.125); w = d * (2s+1) * (grid[idx][pos] + delta).
"""
import numpy as np

from .. import paths  # noqa: F401  (puts the native gguf-py first on sys.path)
from gguf.quants import IQ1_S, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_S, IQ3_XXS  # noqa: E402

QK = 256
SPEC = {
    "IQ3_S": dict(nbytes=110, gsize=4, ngroups=64, nsub=8, sub=32, grid_cls=IQ3_S),
    "IQ2_S": dict(nbytes=82, gsize=8, ngroups=32, nsub=16, sub=16, grid_cls=IQ2_S),
    "IQ2_XXS": dict(nbytes=66, gsize=8, ngroups=32, nsub=8, sub=32, grid_cls=IQ2_XXS, parity=True),
    "IQ2_XS": dict(nbytes=74, gsize=8, ngroups=32, nsub=16, sub=16, grid_cls=IQ2_XS, parity=True),
    "IQ3_XXS": dict(nbytes=98, gsize=4, ngroups=64, nsub=8, sub=32, grid_cls=IQ3_XXS, parity=True),
    "IQ1_S": dict(nbytes=50, gsize=8, ngroups=32, nsub=8, sub=32, grid_cls=IQ1_S, dsub=32),
    "IQ1_M": dict(nbytes=56, gsize=8, ngroups=32, nsub=16, sub=16, grid_cls=IQ1_S, dsub=8),
}
PARITY = ("IQ2_XXS", "IQ2_XS", "IQ3_XXS")
IQ1 = ("IQ1_S", "IQ1_M")
IQ1_DELTA = np.float32(0.125)                                                 # IQ1S_DELTA == IQ1M_DELTA (ggml-common.h)
KSIGNS = np.frombuffer(IQ2_XXS.ksigns, np.uint8)                              # == ksigns_iq2xs (ggml-common.h)


def legal_signs():
    """[128, 8] bool: row s7 = the 8 signs (True = negative) that sign field s7 decodes to"""
    return ((KSIGNS[:, None] >> np.arange(8, dtype=np.uint8)) & 1).astype(bool)


def grid(t):
    c = SPEC[t]["grid_cls"]
    if c.grid is None:
        c.init_grid()
    return c.grid.reshape(-1, SPEC[t]["gsize"]).astype(np.float32)          # [n_codewords, gsize] magnitudes


def scale_mult(t, s):
    s = s.astype(np.float32)
    if t in IQ1:
        return 2 * s + 1
    if t == "IQ3_XXS":
        return (0.5 + s) * 0.5
    return (1 + 2 * s) if t == "IQ3_S" else (0.5 + s) * 0.25


def _bits(b, nbits_per, count):
    """little-endian bit fields: b [n, m] uint8 -> [n, m * (8 // nbits_per)]"""
    shifts = np.arange(0, 8, nbits_per, dtype=np.uint8)
    return ((b[..., None] >> shifts) & ((1 << nbits_per) - 1)).reshape(b.shape[0], -1)[:, :count]


def _pack_bits(v, nbits_per):
    """inverse of _bits: v [n, k] small ints -> [n, k * nbits_per / 8] uint8"""
    per = 8 // nbits_per
    v = v.astype(np.uint8).reshape(v.shape[0], -1, per)
    shifts = np.arange(0, 8, nbits_per, dtype=np.uint8)
    return np.bitwise_or.reduce(v << shifts, axis=-1).astype(np.uint8)


def unpack(t, blocks):
    """blocks [n, nbytes] uint8 -> dict of fields"""
    s = SPEC[t]
    assert blocks.shape[1] == s["nbytes"], blocks.shape
    if t in IQ1:
        return _unpack_iq1(t, blocks)
    if t in PARITY:
        return _unpack_parity(t, blocks)
    d = blocks[:, :2].copy().view(np.float16).astype(np.float32).ravel()
    if t == "IQ3_S":
        qs, qh, sg, sc = blocks[:, 2:66], blocks[:, 66:74], blocks[:, 74:106], blocks[:, 106:110]
        idx = qs.astype(np.uint16) | (_bits(qh, 1, 64).astype(np.uint16) << 8)
    else:
        qs, sg, qh, sc = blocks[:, 2:34], blocks[:, 34:66], blocks[:, 66:74], blocks[:, 74:82]
        idx = qs.astype(np.uint16) | (_bits(qh, 2, 32).astype(np.uint16) << 8)
    return dict(d=d, d_raw=blocks[:, :2].copy(), scales=_bits(sc, 4, s["nsub"]), idx=idx, signs=_bits(sg, 1, QK).astype(bool))


def pack(t, f):
    s = SPEC[t]
    if t in IQ1:
        return _pack_iq1(t, f)
    if t in PARITY:
        return _pack_parity(t, f)
    n = f["idx"].shape[0]
    out = np.empty((n, s["nbytes"]), np.uint8)
    out[:, :2] = f["d_raw"]
    qs = (f["idx"] & 0xFF).astype(np.uint8)
    sg = _pack_bits(f["signs"].astype(np.uint8), 1)
    sc = _pack_bits(f["scales"], 4)
    if t == "IQ3_S":
        out[:, 2:66], out[:, 66:74], out[:, 74:106], out[:, 106:110] = qs, _pack_bits(f["idx"] >> 8, 1), sg, sc
    else:
        out[:, 2:34], out[:, 34:66], out[:, 66:74], out[:, 74:82] = qs, sg, _pack_bits(f["idx"] >> 8, 2), sc
    return out


def dequant(t, f):
    s = SPEC[t]
    if t in IQ1:                                                              # dl * (grid + delta), dl = d * (2s+1) in f32
        dl = np.repeat(f["d"][:, None] * scale_mult(t, f["scales"]), s["sub"], axis=1)    # [n, 256] f32
        val = grid(t)[f["idx"].astype(np.int64)] + np.where(f["delta"], -IQ1_DELTA, IQ1_DELTA)[..., None]
        return dl * val.reshape(-1, QK)
    g = grid(t)[f["idx"].astype(np.int64)]                                   # [n, ngroups, gsize]
    mag = g.reshape(-1, QK)
    sub = np.repeat(scale_mult(t, f["scales"]), s["sub"], axis=1)            # [n, 256]
    return f["d"][:, None] * sub * mag * np.where(f["signs"], -1.0, 1.0)


def _u32(b):
    """b [n, 4m] uint8 -> [n, m] little-endian uint32"""
    return np.ascontiguousarray(b).view("<u4").astype(np.uint32)


def _signs_from7(s7):
    """s7 [n, k] 7-bit fields -> [n, 8k] bool"""
    return legal_signs()[s7.astype(np.int64)].reshape(s7.shape[0], -1)


def _signs_to7(signs):
    """signs [n, 8k] bool -> [n, k] 7-bit fields; asserts each 8-chunk has even parity (is legal)"""
    sg = signs.reshape(signs.shape[0], -1, 8)
    assert not (sg.sum(-1) & 1).any(), "illegal sign pattern: odd number of negatives in an 8-chunk"
    return (sg[..., :7].astype(np.uint32) << np.arange(7, dtype=np.uint32)).sum(-1).astype(np.uint32)


def _unpack_parity(t, blocks):
    n = blocks.shape[0]
    d_raw = blocks[:, :2].copy()
    d = d_raw.view(np.float16).astype(np.float32).ravel()
    sh7 = np.arange(4, dtype=np.uint32) * 7
    if t == "IQ2_XXS":
        q = blocks[:, 2:].reshape(n, 8, 8)                                    # [n, ib32, 8 bytes]
        idx = q[:, :, :4].reshape(n, 32).astype(np.uint16)
        aux = _u32(q[:, :, 4:].reshape(n, 32))                                # [n, 8]
        s7 = (aux[:, :, None] >> sh7) & 127                                   # [n, 8, 4]
        scales = (aux >> 28).astype(np.uint8)
    elif t == "IQ2_XS":
        qs = np.ascontiguousarray(blocks[:, 2:66]).view("<u2").astype(np.uint16)   # [n, 32]
        idx = qs & 511
        s7 = qs >> 9
        scales = _bits(blocks[:, 66:74], 4, 16)
    else:
        idx = blocks[:, 2:66].astype(np.uint16)
        aux = _u32(blocks[:, 66:98])                                          # [n, 8]
        s7 = (aux[:, :, None] >> sh7) & 127
        scales = (aux >> 28).astype(np.uint8)
    return dict(d=d, d_raw=d_raw, scales=scales, idx=idx, signs=_signs_from7(s7.reshape(n, -1)))


def _pack_parity(t, f):
    s = SPEC[t]
    n = f["idx"].shape[0]
    out = np.empty((n, s["nbytes"]), np.uint8)
    out[:, :2] = f["d_raw"]
    s7 = _signs_to7(f["signs"])                                               # [n, 32]
    idx = f["idx"].astype(np.uint32)
    sc = f["scales"].astype(np.uint32)
    assert idx.max(initial=0) < len(grid(t)) and sc.max(initial=0) < 16
    sh7 = np.arange(4, dtype=np.uint32) * 7
    if t == "IQ2_XXS":
        aux = ((s7.reshape(n, 8, 4) << sh7).sum(-1) | (sc << 28)).astype("<u4")    # [n, 8]
        q = np.empty((n, 8, 8), np.uint8)
        q[:, :, :4] = idx.reshape(n, 8, 4).astype(np.uint8)
        q[:, :, 4:] = aux.view(np.uint8).reshape(n, 8, 4)
        out[:, 2:] = q.reshape(n, 64)
    elif t == "IQ2_XS":
        out[:, 2:66] = (idx | (s7 << 9)).astype("<u2").view(np.uint8).reshape(n, 64)
        out[:, 66:74] = _pack_bits(f["scales"], 4)
    else:
        out[:, 2:66] = idx.astype(np.uint8)
        aux = ((s7.reshape(n, 8, 4) << sh7).sum(-1) | (sc << 28)).astype("<u4")
        out[:, 66:98] = aux.view(np.uint8).reshape(n, 32)
    return out


# ------------------------------------------------------------------------------ IQ1_S / IQ1_M
# IQ1_S (50 B): d f16 | qs[32] idx low 8 bits, one per 8-weight group | qh u16[8], one per 32 weights:
#               bits 3l..3l+2 = idx bits 8..10 of group 4ib+l, bits 12..14 = scale s, bit 15 = delta sign
#               w = d * (2s+1) * (grid[idx][pos] + delta), delta = -0.125 if bit 15 else +0.125
#               (scale AND delta shared by the 4 groups of a 32-weight sub-block)
# IQ1_M (56 B): qs[32] idx low 8 bits | qh[16]: byte k holds groups 2k (lo nibble) and 2k+1 (hi nibble),
#               nibble = idx bits 8..10 | delta sign << 3 | scales u16[4]: s of 16-weight sub-block k at
#               bits 3*(k%4).. of u16 k//4; the f16 super-block scale d = top nibbles (u16 i -> bits 4i..4i+3)
#               w = d * (2s+1) * (grid[idx][pos] + delta), delta per 8-weight group
# Fields: d [n] f32, d_raw [n, 2] u8 (the f16 bytes), scales [n, nsub] u8, idx [n, 32] u16 (11 bits),
#         delta [n, 32] bool per group (True = -0.125; IQ1_S: equal within each 4-group sub-block).
def _unpack_iq1(t, blocks):
    n = blocks.shape[0]
    if t == "IQ1_S":
        d_raw = blocks[:, :2].copy()
        qs = blocks[:, 2:34].astype(np.uint16)
        qh = np.ascontiguousarray(blocks[:, 34:50]).view("<u2").astype(np.uint16)          # [n, 8]
        hi = (qh[:, :, None] >> (3 * np.arange(4, dtype=np.uint16))) & 7                  # [n, 8, 4]
        idx = qs | (hi.reshape(n, 32) << 8)
        scales = ((qh >> 12) & 7).astype(np.uint8)
        delta = np.repeat((qh >> 15).astype(bool), 4, axis=1)
    else:
        qs = blocks[:, :32].astype(np.uint16)
        qh = blocks[:, 32:48]
        nib = ((qh[:, :, None] >> np.array([0, 4], np.uint8)) & 15).reshape(n, 32)        # group g -> qh[g//2] >> 4(g%2)
        idx = qs | ((nib & 7).astype(np.uint16) << 8)
        delta = (nib >> 3).astype(bool)
        sc = np.ascontiguousarray(blocks[:, 48:56]).view("<u2").astype(np.uint16)          # [n, 4]
        scales = ((sc[:, :, None] >> (3 * np.arange(4, dtype=np.uint16))) & 7).reshape(n, 16).astype(np.uint8)
        du = ((sc >> 12) << (4 * np.arange(4, dtype=np.uint16))).sum(1).astype("<u2")      # [n]
        d_raw = du.view(np.uint8).reshape(n, 2).copy()
    d = d_raw.view(np.float16).astype(np.float32).ravel()
    return dict(d=d, d_raw=d_raw, scales=scales, idx=idx, delta=delta)


def _pack_iq1(t, f):
    s = SPEC[t]
    n = f["idx"].shape[0]
    idx = f["idx"].astype(np.uint32)
    sc = f["scales"].astype(np.uint32)
    dl = f["delta"].astype(np.uint32)
    assert idx.max(initial=0) < 2048 and sc.max(initial=0) < 8, "IQ1: idx >= 2048 or scale >= 8"
    out = np.empty((n, s["nbytes"]), np.uint8)
    if t == "IQ1_S":
        d4 = dl.reshape(n, 8, 4)
        assert (d4 == d4[:, :, :1]).all(), "IQ1_S: delta sign differs inside a 32-weight sub-block"
        hi = ((idx >> 8).reshape(n, 8, 4) << (3 * np.arange(4, dtype=np.uint32))).sum(-1)
        qh = (hi | (sc << 12) | (d4[:, :, 0] << 15)).astype("<u2")
        out[:, :2] = f["d_raw"]
        out[:, 2:34] = (idx & 255).astype(np.uint8)
        out[:, 34:50] = qh.view(np.uint8).reshape(n, 16)
    else:
        nib = ((idx >> 8) | (dl << 3)).reshape(n, 16, 2)
        out[:, :32] = (idx & 255).astype(np.uint8)
        out[:, 32:48] = (nib[:, :, 0] | (nib[:, :, 1] << 4)).astype(np.uint8)
        du = f["d_raw"].copy().view("<u2").astype(np.uint32).ravel()                     # [n]
        w = (sc.reshape(n, 4, 4) << (3 * np.arange(4, dtype=np.uint32))).sum(-1)          # [n, 4]
        w |= ((du[:, None] >> (4 * np.arange(4, dtype=np.uint32))) & 15) << 12
        out[:, 48:56] = w.astype("<u2").view(np.uint8).reshape(n, 8)
    return out
