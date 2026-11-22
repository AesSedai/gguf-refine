"""Type plan from error tables: stratified layer-total error estimates per (layer, role, type), then the exact
knapsack.

A role is a set of tensors that must share a type (e.g. gate+up, refined jointly). The model config (ALLOC in the
adapter module) gives:
  roles(L)   -> {role: [tensor names]}
  conv       -> {tensor-name suffix: factor} converting each tensor's output NMSE to the common unit (K3.1: measured
                propagation into the expert output, gate 0.58 / up 0.47 / down 1.0); default 1
  scale(L)   -> tensor whose test_sig turns the summed NMSE into absolute error (K3.1: ffn_down_exps, the expert
                output), or None to use each tensor's own test_sig
Layer total for (L, role, type) = sum over strata of N_h * mean_h(v), v = scale_sig * sum_m conv[m] * nmse[m], with
the finite-population variance. Option sizes are exact bytes from the GGUF tensor shapes.

    python -m ggufrefine.alloc.plan --model kimi_k3 --gguf REF.gguf --errtab 'logs/errtab/*.jsonl' \
        --counts-dir logs/errtab --avg-bpw 2.0625 --layers 1-92 --types IQ2_XXS,IQ2_XS,IQ2_S,IQ3_XXS,IQ3_S
"""
import argparse
import glob
import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from .. import ggufio
from .knapsack import bpw, price, solve


def load(pattern):
    """records from all files matching the comma-separated globs; one per (layer, expert, type), a refined record
    replacing a stock-only one for the same key"""
    by = {}
    for pat in pattern.split(","):
        for f in sorted(glob.glob(pat)):
            with open(f) as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    k = (r["layer"], r["expert"], r["type"])
                    refined = any(isinstance(v, dict) and "test_nmse_ref" in v for v in r.values())
                    if k not in by or refined:
                        by[k] = r
    return list(by.values())


def _conv(conv, name):
    for suffix, f in (conv or {}).items():
        if name.split(".")[-2] == suffix or name.endswith(suffix):
            return f
    return 1.0


def estimate(recs, counts_dir, cfg):
    """-> ({(L, role, type, tag): (total, se)}, sorted layers with a complete sample); tag base / ref"""
    by = defaultdict(list)
    for r in recs:
        if r.get("stratum", -1) >= 0:
            by[(r["layer"], r["type"])].append(r)
    est, layers = {}, set()
    for (L, t), rs in by.items():
        cpath = Path(counts_dir) / f"counts-L{L}.json"
        if not cpath.exists():
            continue
        with open(cpath) as fh:
            strata = json.load(fh)["strata"]
        got = defaultdict(list)
        for r in rs:
            got[r["stratum"]].append(r)
        if any(len(got[si]) < rs[0]["n_h"] for si in range(len(strata))):
            continue                                                          # layer x type incomplete
        layers.add(L)
        scale = cfg["scale"](L) if cfg.get("scale") else None
        for role, mats in cfg["roles"](L).items():
            for tag in ("base", "ref"):
                key = f"test_nmse_{tag}"
                if key not in rs[0][mats[0]]:
                    continue
                tot, var = 0.0, 0.0
                for si, g in got.items():
                    if scale is not None:
                        v = np.array([sum(_conv(cfg.get("conv"), m) * r[m][key] for m in mats) * r[scale]["test_sig"] for r in g])
                    else:
                        v = np.array([sum(_conv(cfg.get("conv"), m) * r[m][key] * r[m]["test_sig"] for m in mats) for r in g])
                    Nh, nh = g[0]["N_h"], len(g)
                    tot += Nh * v.mean()
                    if nh > 1:
                        var += Nh ** 2 * v.var(ddof=1) / nh * (1 - nh / Nh)
                est[(L, role, t, tag)] = (tot, var ** 0.5)
    return est, sorted(layers)


def plan(est, layers, cfg, info, types, budget, grid, stock_only=False, weights=None):
    """-> (assign {"L.role": type}, used bytes, cost)"""
    items, keys = [], []
    for L in layers:
        for role, mats in cfg["roles"](L).items():
            n = sum(int(np.prod(info[m].shape)) for m in mats)
            opts = []
            for t in types:
                e = est.get((L, role, t, "base" if stock_only else "ref")) or est.get((L, role, t, "base"))
                if e is None:
                    continue
                w = (weights or {}).get(str(L), {}).get(role, 1.0)
                opts.append((price(n, t), w * e[0], t))
            items.append(opts)
            keys.append((L, role))
    pick, cost = solve(items, budget, grid)
    used = sum(items[i][k][0] for i, k in enumerate(pick))
    return {f"{L}.{role}": items[i][pick[i]][2] for i, (L, role) in enumerate(keys)}, used, cost


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="adapter module under ggufrefine.models with an ALLOC config")
    ap.add_argument("--gguf", required=True, help="any GGUF of the model (tensor shapes)")
    ap.add_argument("--errtab", required=True)
    ap.add_argument("--counts-dir", required=True)
    ap.add_argument("--budget-gb", type=float, default=None, help="bytes for the planned tensors (1e9)")
    ap.add_argument("--avg-bpw", type=float, default=None, help="alternative budget: mean bpw over the planned tensors")
    ap.add_argument("--layers", default="", help="e.g. 1-48")
    ap.add_argument("--types", default="IQ2_XXS,IQ2_XS,IQ2_S,IQ3_XXS,IQ3_S")
    ap.add_argument("--stock-only", action="store_true")
    ap.add_argument("--weights", type=Path, default=None, help="json {layer: {role: w}} multiplying the cost")
    ap.add_argument("--grid-gb", type=float, default=0.01)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    cfg = importlib.import_module(f"ggufrefine.models.{a.model}").ALLOC
    info = ggufio.tensor_table(a.gguf)
    est, layers = estimate(load(a.errtab), a.counts_dir, cfg)
    if a.layers:
        lo, hi = map(int, a.layers.split("-"))
        layers = [L for L in layers if lo <= L <= hi]
    n_w = sum(int(np.prod(info[m].shape)) for L in layers for mats in cfg["roles"](L).values() for m in mats)
    budget = int(a.budget_gb * 1e9) if a.budget_gb is not None else int(n_w * a.avg_bpw / 8)
    weights = json.load(open(a.weights)) if a.weights else None
    types = a.types.split(",")
    assign, used, cost = plan(est, layers, cfg, info, types, budget, int(a.grid_gb * 1e9), a.stock_only, weights)
    print(f"layers {len(layers)}  budget {budget / 1e9:.2f} GB  used {used / 1e9:.2f} GB  mean {used * 8 / n_w:.4f} bpw  cost {cost:.4g}")
    hist = defaultdict(int)
    for k, t in assign.items():
        hist[(k.split(".", 1)[1], t)] += 1
    for role in sorted({k.split(".", 1)[1] for k in assign}):
        print(f"  {role:18s} " + "  ".join(f"{t} x{hist[(role, t)]} ({bpw(t):.4g} bpw)" for t in types if hist[(role, t)]))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"budget_bytes": budget, "used_bytes": used, "cost": cost, "stock_only": a.stock_only,
                       "weights": str(a.weights), "assign": assign}, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
