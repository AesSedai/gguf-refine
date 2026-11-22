#!/usr/bin/env python3
"""Sum forward.py --imatrix partials over processes and write a GGUF imatrix.

Layout matches tools/imatrix/imatrix.cpp save_imatrix(): per weight
  <name>.in_sum2  f32 ggml [n_per_row, n_mat]   (numpy (n_mat, n_per_row))
  <name>.in_sum   f32 ggml [n_per_row, n_mat]
  <name>.counts   f32 ggml [1, n_mat]
n_mat is the expert count for *_exps, else 1 (per-head 3D tensors are stored flattened
with one count, as llama-imatrix does). The derived <name>.stats tensors are added by
re-saving the output with `llama-imatrix --in-file <out> -o <final>` (no -f needed).

usage: python -m ggufrefine.models.kimi_k3.capture.write_imatrix --out x.gguf --dataset calibration_datav5.txt --chunk-size 8192 part_dir...
"""
import argparse
import glob
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from .... import paths  # noqa: E402,F401
import gguf  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--chunk-size", type=int, required=True)
    ap.add_argument("parts", nargs="+", type=Path)
    a = ap.parse_args()

    files = defaultdict(list)
    for p in a.parts:
        for f in glob.glob(str(p / "imatrix" / "*.npz")):
            files[Path(f).name].append(f)
    n_parts = len(a.parts)
    incomplete = [k for k, v in files.items() if len(v) != n_parts]
    assert not incomplete, f"files missing from some parts: {sorted(incomplete)[:5]}"
    n_tokens = sum(np.load(p / "tokens.npy").size for p in a.parts)
    n_chunks = n_tokens // a.chunk_size

    sums = {}
    for fname in sorted(files):
        for f in files[fname]:
            z = np.load(f)
            for k in z.files:
                sums[k] = sums[k] + z[k] if k in sums else z[k].astype(np.float64)

    w = gguf.GGUFWriter(str(a.out), arch="imatrix")
    w.kv_data[0].pop(gguf.Keys.General.ARCHITECTURE, None)   # llama-imatrix files carry no architecture
    w.add_type("imatrix")
    w.add_array("imatrix.datasets", [a.dataset])
    w.add_uint32("imatrix.chunk_count", n_chunks)
    w.add_uint32("imatrix.chunk_size", a.chunk_size)
    names = sorted({k.rsplit(".", 1)[0] for k in sums})
    for n in names:
        s2, s1, c = sums[n + ".in_sum2"], sums[n + ".in_sum"], sums[n + ".counts"]
        if s2.ndim == 1:
            s2, s1 = s2[None, :], s1[None, :]
        assert s2.shape == s1.shape and s2.shape[0] == c.size, (n, s2.shape, c.shape)
        w.add_tensor(n + ".in_sum2", s2.astype(np.float32))
        w.add_tensor(n + ".in_sum", s1.astype(np.float32))
        w.add_tensor(n + ".counts", c.reshape(-1, 1).astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"{a.out}: {len(names)} weights, {n_chunks} chunks of {a.chunk_size} from {n_parts} parts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
