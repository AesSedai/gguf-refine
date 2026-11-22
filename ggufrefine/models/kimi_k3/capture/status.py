#!/usr/bin/env python3
"""Status of a multi-part capture, read from the artifacts themselves (no listing of the virtiofs
part directories: names are constructed).

  python -m ggufrefine.models.kimi_k3.capture.status --out O --grams G [--parts 0-30] [--check K]
per part: expert-dump layers complete (meta.json with n == the part's token count), tokens.npy present,
Gram layers holding the part in the merged split. --check K exits 0 iff part K is complete
(all dumps, tokens.npy, Grams of every layer + output merged)."""
import argparse
import json
import sys
from pathlib import Path


N_SEQ, PER, N_CTX = 305, 10, 8192


def part_status(out, grams, k, moe_layers, gram_layers, test_parts):
    p = out / f"part{k:02d}"
    lo, hi = k * PER, min(N_SEQ, (k + 1) * PER)
    n = (hi - lo) * N_CTX
    dumps = 0
    for il in moe_layers:
        m = p / "x2" / f"layer-{il}" / "meta.json"
        if m.exists() and json.load(open(m))["n"] == n:
            dumps += 1
    split = "test" if k in test_parts else "train"
    merged = 0
    for g in gram_layers:
        m = grams / split / f"layer-{g}" / "meta.json"
        if m.exists() and k in json.load(open(m))["parts"]:
            merged += 1
    return dict(part=k, seqs=[lo, hi], dumps=dumps, dumps_want=len(moe_layers), tokens=(p / "tokens.npy").exists(),
                ckpt=(p / "ckpt.pt").exists(), grams_merged=merged, grams_want=len(gram_layers),
                complete=dumps == len(moe_layers) and (p / "tokens.npy").exists() and merged == len(gram_layers))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--grams", type=Path, required=True)
    ap.add_argument("--parts", default="0-30")
    ap.add_argument("--test-parts", default="29,30")
    ap.add_argument("--n-layer", type=int, default=93)
    ap.add_argument("--dense-lead", type=int, default=1)
    ap.add_argument("--check", type=int, default=None)
    a = ap.parse_args()
    lo, hi = map(int, a.parts.split("-"))
    moe = list(range(a.dense_lead, a.n_layer))
    gl = [str(i) for i in range(a.n_layer)] + ["output"]
    tp = {int(x) for x in a.test_parts.split(",")}
    if a.check is not None:
        s = part_status(a.out, a.grams, a.check, moe, gl, tp)
        print(json.dumps(s))
        return 0 if s["complete"] else 1
    for k in range(lo, hi + 1):
        print(json.dumps(part_status(a.out, a.grams, k, moe, gl, tp)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
