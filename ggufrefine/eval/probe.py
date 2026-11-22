"""KLD sensitivity probes with llama-kld-swap.

One load of a base GGUF. For each block of consecutive layers and each role, the role's tensors in that block
are replaced by the same tensors of a donor GGUF (retype-gguf: any type, the donor need not share the layout),
evaluated, and reverted; every eval is paired against the unpatched eval on the same tokens. A last probe swaps
every block of every role (additivity check). Examples: base = stock build, donor = refined build (KLD gain of
refinement per block / role); base = refined IQ2_XXS, donor = an IQ2_S build (KLD per byte of an upgrade).
Resumable: an eval whose <out>.json exists is skipped; summary.jsonl gets one line per probe.

    python -m ggufrefine.eval.probe --gguf STOCK.gguf --donor REFINED.gguf --base combined.kld --out logs/probes \
        --model kimi_k3 --layers 1-92 --blocks 8
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

from .kld import DEFAULT_ARGS, DEFAULT_ENV
from .kldswap import KldSwap, compare, format_compare


def run(gguf, donor, base, out, roles, layers, n_blocks, extra=(), env=None):
    """roles: callable L -> {role: [tensor names]}"""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n = len(layers)
    blocks = [layers[i * n // n_blocks:(i + 1) * n // n_blocks] for i in range(n_blocks)]
    role_names = list(roles(layers[0]))
    probes = [(f"b{bi}-{role}", [(L, role) for L in blk]) for bi, blk in enumerate(blocks) for role in role_names]
    probes.append(("all", [(L, role) for L in layers for role in role_names]))
    with KldSwap(["-m", str(gguf), *DEFAULT_ARGS, *extra], log=str(out / "kldswap.log"), env={**DEFAULT_ENV, **(env or {})}) as ks:
        ks.base(str(base))
        ref = str(out / "unpatched")
        if not Path(ref + ".json").exists():
            ks.eval(out=ref, tokens=True)
        for name, cells in probes:
            o = str(out / name)
            if Path(o + ".json").exists():
                continue
            for L, role in cells:
                for tn in roles(L)[role]:
                    ks.retype_gguf(tn, donor)
            ks.eval(out=o, tokens=True)
            ks.revert("all")
            r = compare(ref, o)
            print(f"== {name}: layers {cells[0][0]}-{cells[-1][0]}\n" + format_compare(r), flush=True)
            with open(out / "summary.jsonl", "a") as fh:
                fh.write(json.dumps({"probe": name, "cells": cells, "kld_unpatched": r["kld"]["A"], "kld": r["kld"]["B"],
                                     "diff": r["kld"]["diff"], "boot_se": r["kld"]["boot_se"], "rel": r["kld"]["rel"],
                                     "ci95": r["kld"]["boot_ci95"]}) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="loaded once")
    ap.add_argument("--donor", required=True, help="GGUF the probed tensors are taken from")
    ap.add_argument("--base", required=True, help="KLD base file")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", required=True, help="adapter module under ggufrefine.models with ALLOC['roles']")
    ap.add_argument("--layers", required=True, help="e.g. 1-92")
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--extra", default="")
    a = ap.parse_args(argv)
    roles = importlib.import_module(f"ggufrefine.models.{a.model}").ALLOC["roles"]
    lo, hi = map(int, a.layers.split("-"))
    run(a.gguf, a.donor, a.base, a.out, roles, list(range(lo, hi + 1)), a.blocks, a.extra.split())
    return 0


if __name__ == "__main__":
    sys.exit(main())
