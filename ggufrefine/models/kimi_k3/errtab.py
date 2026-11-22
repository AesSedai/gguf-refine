"""K3.1 error tables for the type plan (output keyed by GGUF tensor name).

    python -m ggufrefine.models.kimi_k3.errtab --layers 4,46 --out logs/errtab/errtab-g0.jsonl \
        --model HF_DIR --dump CAPTURE_ROOT --imatrix X-imatrix.gguf
"""
import argparse
import sys
from pathlib import Path

from ... import ggufio
from ...alloc.errtab import errtab_layer
from . import DumpHessians, SafetensorsWeights, groups


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", required=True, help="comma list")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--imatrix", type=Path, required=True)
    ap.add_argument("--test-parts", default="29,30")
    ap.add_argument("--strata", type=int, default=8)
    ap.add_argument("--per-stratum", type=int, default=6)
    ap.add_argument("--types", default="IQ2_XXS,IQ2_XS,IQ2_S,IQ3_XXS,IQ3_S,IQ1_S,IQ1_M,IQ4_XS")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--sweeps", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--quant-threads", type=int, default=6)
    ap.add_argument("--experts", default="", help="testing: comma list overriding the sample (stratum -1)")
    a = ap.parse_args(argv)
    w = SafetensorsWeights(a.model)
    hs = DumpHessians(a.dump, a.model, a.test_parts.split(","))
    im = ggufio.Imatrix(a.imatrix)
    for L in [int(x) for x in a.layers.split(",")]:
        errtab_layer(L, groups(L), w, hs, im, a.types.split(","), a.out, counts=hs.counts(L), n_strata=a.strata,
                     per_stratum=a.per_stratum, lam=a.lam, sweeps=a.sweeps, batch=a.batch, quant_threads=a.quant_threads,
                     experts=[int(x) for x in a.experts.split(",")] if a.experts else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
