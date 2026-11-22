"""Adapter for any model captured with ggufrefine.capture.ggml (or .hf): groups and Hessians from merged Gram splits,
weights from a GGUF of the reference precision, stock codes from the quantizer + imatrix.

Groups: each Gram input group (tensors that read one input) becomes one refine group when its tensors have equal row
counts and types (gate + up), else one group per tensor sharing that Hessian (q / k / v).
Types: without --type / --types, each tensor keeps the type the stock --write-gguf file has (a llama-quantize preset
mixes types); tensors whose type is not refinable (Q4_K, Q6_K, ...) are left as they are. Hessians: H = sum_t x_t x_t^T of the
train and test splits (<grams>/{train,test}/layer-<L>/). A layer is refined in one driver call.

    python -m ggufrefine.models.dense --gguf BF16.gguf --grams GRAMS --imatrix X.imatrix.gguf \\
        --write-gguf COPY-OF-STOCK.gguf --results logs/refine.jsonl [--layers 0-47] [--only attn_q,ffn_down]
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .. import ggufio
from ..capture.grams import tri_len, unpack_tri
from ..refine import driver
from ..refine import engine
from ..refine.problem import Group, RequantizedCodes


def role(name):
    return re.sub(r"^blk\.\d+\.", "", name).removesuffix(".weight")


class GramSplits:
    """merged Gram splits: meta per layer, and Hessian loads (one layer cached on the device)"""

    def __init__(self, root, dev="cuda:0", train="train", test="test"):
        self.root, self.dev, self.split = Path(root), torch.device(dev), (train, test)
        self.layer, self.cache = None, {}

    def meta(self, layer):
        with open(self.root / self.split[0] / f"layer-{layer}" / "meta.json") as fh:
            return json.load(fh)

    def layers(self):
        return sorted(int(p.name.split("-")[1]) for p in (self.root / self.split[0]).glob("layer-*") if p.name[6:].isdigit())

    def _read(self, split, layer, g, K):
        d = self.root / split / f"layer-{layer}"
        tri = np.fromfile(d / f"{g}.tri.f32", np.float32)
        assert tri.size == tri_len(K), (d, g, tri.size)
        return torch.from_numpy(unpack_tri(tri.astype(np.float64), K).astype(np.float32)).to(self.dev)

    def hessian(self, layer, g, K):
        if self.layer != layer:
            self.cache, self.layer = {}, layer
            torch.cuda.empty_cache()
        if g not in self.cache:
            self.cache[g] = tuple(self._read(s, layer, g, K) for s in self.split)
        return self.cache[g]


REFINABLE = ("IQ2_S", "IQ3_S") + tuple(engine.GATED)


def groups(layer, splits, info, only=None, type_of=None):
    """-> ([Group], {group key: Gram group name}); type_of(name) -> type, or None to leave the tensor out"""
    out, src = [], {}
    for g, m in splits.meta(layer)["groups"].items():
        names = [n for n in m["weights"] if (only is None or role(n) in only) and (type_of is None or type_of(n))]
        if not names:
            continue
        rows = [info[n].rows for n in names]
        same = len(set(rows)) == 1 and (type_of is None or len({type_of(n) for n in names}) == 1)
        parts = [names] if same else [[n] for n in names]
        for p in parts:
            key = "+".join(role(n) for n in p)
            out.append(Group(layer, key, tuple(p), 1, m["K"], tuple(info[n].rows for n in p), len(out)))
            src[key] = (g, m["K"], m["n"])
    return out, src


class Hessians:
    def __init__(self, splits, src):
        self.splits, self.src = splits, src

    def load(self, layer, groups_, e):
        out = {}
        for g in groups_:
            name, K, n = self.src[g.key]
            out[g.key] = self.splits.hessian(layer, name, K)
            out["rows"] = n
        return out


class GgufWeights:
    def __init__(self, path):
        self.g = ggufio.Gguf(path)

    def weights(self, name, e):
        return self.g.weights(name, e)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", type=Path, required=True, help="reference-precision GGUF (BF16 / F16 / F32)")
    ap.add_argument("--grams", type=Path, required=True, help="merge_grams output (train / test splits)")
    ap.add_argument("--imatrix", type=Path, required=True, help="the imatrix the stock file was quantized with")
    ap.add_argument("--type", default=None, help="one type for every refined tensor (default: each tensor's stock type)")
    ap.add_argument("--types", default="", help="per role, e.g. attn_q=IQ2_XS,ffn_down=IQ3_XXS (overrides --type)")
    ap.add_argument("--only", default="", help="comma list of roles to refine (default: all captured)")
    ap.add_argument("--layers", default="", help="e.g. 0-47 (default: every captured layer)")
    ap.add_argument("--write-gguf", type=Path, default=None, help="stock GGUF of the same recipe, refined in place")
    ap.add_argument("--out", type=Path, default=None, help="or write <tensor>-e0.npy here")
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--sweeps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0, help="visit-order seed")
    ap.add_argument("--quant-threads", type=int, default=16)
    a = ap.parse_args(argv)
    info = ggufio.tensor_table(a.gguf)
    splits = GramSplits(a.grams)
    only = set(a.only.split(",")) if a.only else None
    per_role = dict(kv.split("=") for kv in a.types.split(",")) if a.types else {}
    if a.type or per_role:
        def type_of(n):
            return per_role.get(role(n), a.type)
    else:
        if not a.write_gguf:
            ap.error("without --type / --types the types come from the stock --write-gguf file")
        stock_types = {n: ti.type for n, ti in ggufio.tensor_table(a.write_gguf).items()}

        def type_of(n):
            t = stock_types.get(n)
            return t if t in REFINABLE else None
    weights = GgufWeights(a.gguf)
    stock = RequantizedCodes(weights, ggufio.Imatrix(a.imatrix), a.quant_threads)
    sink = driver.GgufSink(a.write_gguf) if a.write_gguf else (driver.NpySink(a.out) if a.out else None)
    s = driver.Settings(lam=a.lam, sweeps=a.sweeps, seed=a.seed, batch=1, loaders=1, prefetch=1, tag={"split": "grams", **({"seed": a.seed} if a.seed else {})})   # one loader: GramSplits caches one layer
    layers = splits.layers()
    if a.layers:
        lo, hi = map(int, a.layers.split("-"))
        layers = [L for L in layers if lo <= L <= hi]
    t0 = time.time()
    for i, L in enumerate(layers):
        gs, src = groups(L, splits, info, only, type_of)
        if not gs:
            print(f"layer {L}: nothing to refine", flush=True)
            continue
        types = {}
        for g in gs:
            t = {type_of(n) for n in g.tensors}
            if len(t) != 1 or None in t:
                ap.error(f"{g.key}: no single type ({t}); give --type or --types for every role")
            types[g.key] = t.pop()
        driver.refine_layer(L, gs, types, weights, Hessians(splits, src), stock, s, a.results, sink)
        el = time.time() - t0
        print(f"[{time.strftime('%H:%M:%S')}] layer {L} done ({i + 1}/{len(layers)}, {el / 60:.1f} min, "
              f"eta {el / (i + 1) * (len(layers) - i - 1) / 60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
