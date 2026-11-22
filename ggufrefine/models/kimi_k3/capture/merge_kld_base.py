#!/usr/bin/env python3
"""Join forward.py part directories into one llama-perplexity --kl-divergence-base file.

Layout (tools/perplexity/perplexity.cpp): "_logits_", uint32 n_ctx, int32 n_vocab,
int32 n_chunk, int32 tokens[n_chunk*n_ctx], then per chunk (n_ctx - 1 - n_ctx/2)
records of nv = 2*((n_vocab+1)/2)+4 uint16.

usage: python -m ggufrefine.models.kimi_k3.capture.merge_kld_base --vocab 163840 --out base.kld part_dir0 part_dir1 ...
"""
import argparse
import shutil
import struct
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("parts", nargs="+", type=Path)
    a = ap.parse_args()
    toks = [np.load(p / "tokens.npy").astype(np.int32) for p in a.parts]
    T = toks[0].shape[1]
    assert all(t.shape[1] == T for t in toks)
    n_chunk = sum(t.shape[0] for t in toks)
    nv = 2 * ((a.vocab + 1) // 2) + 4
    rec_bytes = (T - 1 - T // 2) * nv * 2
    for p, t in zip(a.parts, toks):
        sz = (p / "kld_part.bin").stat().st_size
        assert sz == t.shape[0] * rec_bytes, f"{p}: {sz} bytes, expected {t.shape[0] * rec_bytes} (incomplete?)"
    tmp = a.out.with_suffix(a.out.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(b"_logits_")
        f.write(struct.pack("<Iii", T, a.vocab, n_chunk))
        for t in toks:
            f.write(t.tobytes())
        for p in a.parts:
            with open(p / "kld_part.bin", "rb") as src:
                shutil.copyfileobj(src, f, 64 << 20)
    tmp.replace(a.out)
    print(f"{a.out}: n_ctx={T} n_vocab={a.vocab} n_chunk={n_chunk} size={a.out.stat().st_size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
