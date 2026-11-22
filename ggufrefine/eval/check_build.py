"""Check a rebuilt GGUF against another of the same recipe before a matched comparison.

Requires: same tensor names, types, shapes and data offsets; byte-identical tensors outside --differ (sampled:
all small ones + every 50th large one, sha256); tensors matching --differ differ from the other file (sampled);
and, with --reference/--imatrix, the first file's codes of sampled (tensor, expert) pairs equal a fresh stock
quantization of the reference weights (quant.ggml, imatrix row). Prints CHECK PASSED / CHECK FAILED; exit 1 on failure.

    python -m ggufrefine.eval.check_build --a STOCK.gguf --b REFINED.gguf --differ _exps. \
        --reference BF16-MXFP4.gguf --imatrix X-imatrix.gguf --sample blk.1.ffn_gate_exps.weight:0,blk.46.ffn_down_exps.weight:346
"""
import argparse
import hashlib
import os
import sys

import numpy as np

from .. import ggufio
from ..quant import ggml as gq


def sha(path, off, n, cap=None):
    h = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY)
    try:
        pos, n = 0, n if cap is None else min(n, cap)
        while pos < n:
            b = os.pread(fd, min(256 << 20, n - pos), off + pos)
            h.update(b)
            pos += len(b)
    finally:
        os.close(fd)
    return h.hexdigest()


def check(a, b, differ, reference=None, imatrix=None, sample=()):
    A, B = ggufio.tensor_table(a), ggufio.tensor_table(b)
    fail = 0
    if A.keys() != B.keys():
        print("FAIL tensor name sets differ")
        return False
    key = lambda t: (t.offset, t.nbytes, t.type, t.shape)  # noqa: E731
    bad = [n for n in A if key(A[n]) != key(B[n])]
    print(f"layout: {len(A)} tensors, {len(bad)} with different offset/size/type/shape {bad[:5]}")
    fail += bool(bad)
    same_set = sorted(n for n in A if differ not in n)
    big = [n for n in same_set if A[n].nbytes > (64 << 20)]
    smp = [n for n in same_set if A[n].nbytes <= (64 << 20)] + big[::50]
    diff = [n for n in smp if sha(a, A[n].offset, A[n].nbytes) != sha(b, B[n].offset, B[n].nbytes)]
    print(f"unchanged set: {len(smp)} of {len(same_set)} tensors hashed, {len(diff)} differ {diff[:5]}")
    fail += bool(diff)
    dset = sorted(n for n in A if differ in n)
    ident = [n for n in dset[::23] if sha(a, A[n].offset, A[n].nbytes, 64 << 20) == sha(b, B[n].offset, B[n].nbytes, 64 << 20)]
    print(f"changed set: {len(dset[::23])} sampled, {len(ident)} identical in their first 64 MiB (expected 0)")
    fail += bool(ident)
    if reference is not None and sample:
        ref, ga, im = ggufio.Gguf(reference), ggufio.Gguf(a), ggufio.Imatrix(imatrix) if imatrix else None
        ok = 0
        for name, e in sample:
            W = ref.weights(name, e)
            q = gq.quantize(W, A[name].type, im.row(name, e) if im is not None and name in im else None)
            ok += np.array_equal(np.asarray(q, np.uint8).reshape(-1), ga.expert(name, e).reshape(-1))
        print(f"codes vs fresh stock quantization: {ok}/{len(sample)} identical")
        fail += ok != len(sample)
    print("CHECK PASSED" if not fail else "CHECK FAILED")
    return not fail


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="the build being checked (e.g. the stock rebuild)")
    ap.add_argument("--b", required=True, help="the file it is compared with")
    ap.add_argument("--differ", default="_exps.", help="substring of the tensors expected to differ")
    ap.add_argument("--reference", default=None, help="unquantized reference GGUF for the fresh-quantization check")
    ap.add_argument("--imatrix", default=None)
    ap.add_argument("--sample", default="", help="comma list of tensor:expert")
    a = ap.parse_args(argv)
    sample = [(s.rsplit(":", 1)[0], int(s.rsplit(":", 1)[1])) for s in a.sample.split(",") if s]
    return 0 if check(a.a, a.b, a.differ, a.reference, a.imatrix, sample) else 1


if __name__ == "__main__":
    sys.exit(main())
