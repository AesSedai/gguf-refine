"""Calibration exposure of a KLD base, and paired KLD deltas split by it.

A scored position is "exposed" when it lies in an aligned width-token window of the base that also occurs,
at any offset, in a reference corpus (the Hessian or imatrix calibration tokens). Paired per-token KLD of two
evals of the same base is then compared on exposed vs unexposed positions, with a chunk bootstrap.

    python -m ggufrefine.eval.overlap --base X.kld --a logs/kld/stock/combined --b logs/kld/refined/combined \
        --ref hessian=/path/tokens.npy[:290] --ref imatrix=/path/calib.npy --width 64 --width 128
"""
import argparse
import struct
import sys
from collections import defaultdict

import numpy as np

from .kldswap import load_tokens

_P = np.array([pow(1000003, i, 1 << 64) for i in range(8)], dtype=np.uint64)


def base_tokens(path):
    """token matrix [chunks, ctx] from the header of a llama-perplexity --kl-divergence-base file"""
    with open(path, "rb") as f:
        assert f.read(8) == b"_logits_", path
        ctx, _, chunks = struct.unpack("<Iii", f.read(12))
        return np.fromfile(f, dtype="<i4", count=chunks * ctx).reshape(chunks, ctx)


def window_hits(query, reference, width):
    """bool [rows, ctx // width]: aligned window j of query row i occurs somewhere in a reference row"""
    anchors = defaultdict(list)
    for row, tokens in enumerate(query):
        for start in range(0, len(tokens) - width + 1, width):
            anchors[tokens[start:start + 8].tobytes()].append((row, start))
    keys = np.array([np.frombuffer(k, dtype=np.int32) for k in anchors], dtype=np.uint64)
    hashes = keys @ _P                                                        # 8-token prefilter, full window verified
    hits = np.zeros((len(query), query.shape[1] // width), dtype=bool)
    for tokens in reference:
        digests = np.lib.stride_tricks.sliding_window_view(tokens, 8).astype(np.uint64) @ _P
        for pos in np.flatnonzero(np.isin(digests, hashes)):
            for row, start in anchors.get(tokens[pos:pos + 8].tobytes(), []):
                if np.array_equal(query[row, start:start + width], tokens[pos:pos + width]):
                    hits[row, start // width] = True
    return hits


def scored_mask(hits, width, ctx, first_scored):
    """window hits -> per scored position mask [rows, ctx - 1 - first_scored] (positions first_scored..ctx-2 predict
    the next token; llama-perplexity scores the second half: first_scored = ctx // 2)"""
    return np.repeat(hits, width, axis=1)[:, first_scored:ctx - 1]


def split_delta(a, b, mask, n_boot=10000, seed=0):
    """paired KLD (b - a) on masked / unmasked positions; a, b: eval prefixes with tokens.bin"""
    A, B = load_tokens(a), load_tokens(b)
    ch = sorted(A)
    ka, kb = np.stack([A[c][0] for c in ch]), np.stack([B[c][0] for c in ch])
    assert ka.shape == mask.shape, (ka.shape, mask.shape)
    rng = np.random.default_rng(seed)
    out = {}
    for lab, m in (("exposed", mask), ("unexposed", ~mask)):
        d = np.where(m, kb - ka, np.nan)
        s, n = np.nansum(d, 1), m.sum(1)
        idx = rng.integers(0, len(ch), (n_boot, len(ch)))
        boot = s[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
        ma, mb = float(ka[m].mean()), float(kb[m].mean())
        out[lab] = dict(n=int(m.sum()), a=ma, b=mb, rel=(mb - ma) / ma, ci95=[float(x) for x in np.percentile(boot, [2.5, 97.5])])
    return out


def _ref(spec):
    name, path = spec.split("=", 1)
    rows = None
    if path.endswith("]") and "[:" in path:
        path, rows = path[:-1].split("[:")
    arr = np.load(path)
    return name, arr[:int(rows)] if rows else arr


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="the .kld base both evals scored")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--ref", action="append", required=True, help="NAME=tokens.npy, optionally [:ROWS]")
    ap.add_argument("--width", type=int, action="append", default=None)
    a = ap.parse_args(argv)
    q = base_tokens(a.base)
    ctx = q.shape[1]
    for w in a.width or [64]:
        for name, ref in map(_ref, a.ref):
            mask = scored_mask(window_hits(q, ref, w), w, ctx, ctx // 2)
            for lab, r in split_delta(a.a, a.b, mask).items():
                print(f"w{w} {name:12s} {lab:9s} n={r['n']:7d}  A {r['a']:.5f}  B {r['b']:.5f}  rel {r['rel']:+.1%}  "
                      f"diff CI95 [{r['ci95'][0]:+.5f},{r['ci95'][1]:+.5f}]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
