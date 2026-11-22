"""Refine one Kimi-K3.1 MoE layer (generic driver + this adapter).

    python -m ggufrefine.models.kimi_k3.refine --layer 46 --model HF_DIR \
        --dump CAPTURE_ROOT --test-part 29,30 --requantize --imatrix X-imatrix.gguf \
        --types gate_up=IQ2_XXS,down=IQ2_XXS --results logs/L46.jsonl --write-gguf SUBSTRATE.gguf

Stock codes come from --stock-gguf (tensors already of the requested type; --overrides for a backup dir of
tensors patched in place) or, with --requantize, from the quantizer + imatrix. --write-gguf writes refined
codes straight into that GGUF (requires --requantize, since the file's own codes stop being stock);
--out writes <tensor>-e<E>.npy instead.
"""
import argparse
import sys
from pathlib import Path

from ... import ggufio
from ...refine import driver
from ...refine.problem import GgufCodes, RequantizedCodes
from . import DumpHessians, SafetensorsWeights, groups


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True, help="HF dir with the MXFP4 safetensors")
    ap.add_argument("--dump", type=Path, required=True, help="capture root holding the expert dump (part*/x2/layer-L)")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--groups", default="ffn_gate_up_exps,ffn_down_exps")
    ap.add_argument("--types", default="gate_up=IQ2_XXS,down=IQ2_XXS", help="gate_up=T,down=T")
    ap.add_argument("--experts", default="", help="comma list; default all")
    ap.add_argument("--test-part", default="29,30")
    ap.add_argument("--train-parts", default="")
    ap.add_argument("--stock-gguf", type=Path, default=None)
    ap.add_argument("--overrides", type=Path, default=None)
    ap.add_argument("--requantize", action="store_true")
    ap.add_argument("--imatrix", type=Path, default=None)
    ap.add_argument("--quant-threads", type=int, default=16)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--write-gguf", type=Path, default=None)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--hessian", choices=["full", "diag"], default="full")
    ap.add_argument("--sweeps", type=int, default=8)
    ap.add_argument("--noise", default="", help="SCALE:NOISY:CLEAN, e.g. 0.25:24:8")
    ap.add_argument("--noise-groups", default="ffn_gate_up_exps")
    ap.add_argument("--gate", choices=["auto", "on", "exact", "off"], default="auto")
    ap.add_argument("--gate-margin", type=float, default=0.0)
    ap.add_argument("--refresh", choices=["fp32", "ozaki"], default="fp32")
    ap.add_argument("--iq1s-flip", choices=["on", "off"], default="on")
    ap.add_argument("--shadow-rows", type=int, default=0)
    ap.add_argument("--eval-fp64", action="store_true")
    ap.add_argument("--bsz", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--loaders", type=int, default=4)
    ap.add_argument("--prefetch", type=int, default=1)
    ap.add_argument("--verify-codes", action="store_true")
    a = ap.parse_args(argv)
    role = {"ffn_gate_up_exps": "gate_up", "ffn_down_exps": "down"}
    tmap = dict(kv.split("=") for kv in a.types.split(","))
    want = a.groups.split(",")
    gs = [g for g in groups(a.layer) if g.key in want]
    types = {g.key: tmap[role[g.key]] for g in gs}
    weights = SafetensorsWeights(a.model)
    if a.requantize:
        stock = RequantizedCodes(weights, ggufio.Imatrix(a.imatrix) if a.imatrix else None, a.quant_threads)
    else:
        if a.stock_gguf is None:
            ap.error("--stock-gguf or --requantize is required")
        if a.write_gguf is not None and a.write_gguf.resolve() == a.stock_gguf.resolve():
            ap.error("--write-gguf into the --stock-gguf file would read refined codes as stock on resume; use --requantize")
        stock = GgufCodes(ggufio.Gguf(a.stock_gguf, a.overrides))
    if a.write_gguf is not None and not a.requantize and a.stock_gguf is None:
        ap.error("--write-gguf needs a stock source")
    sink = driver.GgufSink(a.write_gguf) if a.write_gguf else (driver.NpySink(a.out) if a.out else None)
    hs = DumpHessians(a.dump, a.model, a.test_part.split(","), a.train_parts.split(",") if a.train_parts else None)
    s = driver.Settings(lam=a.lam, hessian=a.hessian, sweeps=a.sweeps, noise=a.noise,
                        noise_groups=tuple(a.noise_groups.split(",")) if a.noise else (), gate=a.gate,
                        gate_margin=a.gate_margin, refresh=a.refresh, iq1s_flip=a.iq1s_flip == "on",
                        shadow_rows=a.shadow_rows, eval_fp64=a.eval_fp64, bsz=a.bsz, batch=a.batch, loaders=a.loaders,
                        prefetch=a.prefetch, verify_codes=a.verify_codes,
                        tag={"split": "part", "test_part": a.test_part, "train_parts": a.train_parts})
    experts = [int(x) for x in a.experts.split(",")] if a.experts else None
    driver.refine_layer(a.layer, gs, types, weights, hs, stock, s, a.results, sink, experts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
