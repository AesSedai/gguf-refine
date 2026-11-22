"""Kimi-K3 / K3.1 adapter: routed-expert groups, native MXFP4 weights from the HF safetensors, and Hessians from
the routed-expert dump of the capture forward (capture/forward.py --expert-dump-layers, reader expert_dump).

Groups per MoE layer L (1..92): ffn_gate_up_exps = (ffn_gate_exps, ffn_up_exps) on the 3584-wide latent, and
ffn_down_exps on the 3072-wide situ activation, 896 experts each. Hessians are routing-weighted,
H = sum_t (w_t x_t)(w_t x_t)^T over the rows routed to the expert, split into train / held-out by dump part.
"""
import json
import os
import threading
from pathlib import Path

import numpy as np
import torch

from ...refine.problem import Group
from ...ggufio import pread_all
from . import model as km
from .expert_dump import GpuExpertDump

HF = {"ffn_gate_exps": "w1", "ffn_up_exps": "w3", "ffn_down_exps": "w2"}
N_EXPERTS, LATENT, INTER = 896, 3584, 3072


def groups(layer):
    p = f"blk.{layer}."
    return [Group(layer, "ffn_gate_up_exps", (p + "ffn_gate_exps.weight", p + "ffn_up_exps.weight"), N_EXPERTS, LATENT, (INTER, INTER), 0),
            Group(layer, "ffn_down_exps", (p + "ffn_down_exps.weight",), N_EXPERTS, INTER, (LATENT,), 1)]


def roles(layer):
    """tensors that must share a type: gate and up are refined as one problem"""
    p = f"blk.{layer}."
    return {"gate_up": [p + "ffn_gate_exps.weight", p + "ffn_up_exps.weight"], "down": [p + "ffn_down_exps.weight"]}


# alloc.plan config: propagation of each matrix's output NMSE into the expert-output NMSE (measured on K3.1:
# gate 0.58, up 0.47, down 1.0), converted to absolute error with the down matrix's held-out signal
ALLOC = {"roles": roles, "conv": {"ffn_gate_exps": 0.58, "ffn_up_exps": 0.47, "ffn_down_exps": 1.0},
         "scale": lambda layer: f"blk.{layer}.ffn_down_exps.weight"}


def record_from_hf_names(r):
    """errtab / refine record keyed by the HF expert matrix names (w1 / w3 / w2) -> GGUF tensor-name keys"""
    out = dict(r)
    for g, m in HF.items():
        if m in out:
            out[f"blk.{r['layer']}.{g}.weight"] = out.pop(m)
    return out


def _split(name):
    """blk.L.ffn_gate_exps.weight -> (L, 'w1')"""
    _, L, g, _ = name.split(".")
    return int(L), HF[g]


class SafetensorsWeights:
    """weights(name, e) -> fp32 [R, K] from the native MXFP4 expert tensors (pread of the shard, exact dequant)"""

    def __init__(self, model_dir):
        self.dir = Path(model_dir)
        with open(self.dir / "model.safetensors.index.json") as fh:
            self.wm = json.load(fh)["weight_map"]
        self.hdr, self.lock = {}, threading.Lock()

    def _shard(self, shard):
        with self.lock:
            if shard not in self.hdr:
                fd = os.open(self.dir / shard, os.O_RDONLY)
                n = int.from_bytes(pread_all(fd, 8, 0), "little")
                self.hdr[shard] = (fd, 8 + n, json.loads(bytes(pread_all(fd, n, 8))))
            return self.hdr[shard]

    def raw(self, name):
        fd, base, h = self._shard(self.wm[name])
        m = h[name]
        assert m["dtype"] == "U8", m
        s, t = m["data_offsets"]
        return torch.frombuffer(pread_all(fd, t - s, base + s), dtype=torch.uint8).reshape(m["shape"])

    def weights(self, name, e):
        L, m = _split(name)
        p = f"{km.PFX}layers.{L}.block_sparse_moe.experts.{e}.{m}."
        return km.mxfp4_dequant(self.raw(p + "weight_packed"), self.raw(p + "weight_scale")).numpy().astype(np.float32)


class DumpHessians:
    """load(layer, groups, e) -> {group key: (H_train, H_test)} on dev, from a GPU-resident expert dump
    (GpuExpertDump; one layer held at a time). test_parts: held-out dump parts; train_parts: None = the rest."""

    def __init__(self, root, model_dir, test_parts, train_parts=None, dev="cuda:0"):
        self.root, self.model, self.dev = Path(root), Path(model_dir), torch.device(dev)
        self.test_parts = [int(x) for x in test_parts]
        self.train_parts = None if train_parts is None else [int(x) for x in train_parts]
        self.layer, self.dump, self.lock = None, None, threading.Lock()

    def _get(self, layer):
        with self.lock:
            if self.layer != layer:
                self.dump = None
                torch.cuda.empty_cache()
                self.dump = GpuExpertDump(self.root, layer, model=self.model, dev=str(self.dev))
                self.layer = layer
            return self.dump

    def counts(self, layer):
        """routing count per expert in the dump (all parts), for errtab's strata"""
        d = self._get(layer)
        return np.bincount(d.ids.reshape(-1).astype(np.int32), minlength=d.E)

    def load(self, layer, groups_, e):
        from ...refine.engine import hessian                                   # Triton only when Hessians are built
        d = self._get(layer)
        hx, ax, wt = d.rows(e)
        pid = d.part_ids(e)
        test = np.isin(pid, self.test_parts)
        train = (np.isin(pid, self.train_parts) & ~test) if self.train_parts else ~test
        itr = torch.from_numpy(np.nonzero(train)[0]).to(hx.device)
        ite = torch.from_numpy(np.nonzero(test)[0]).to(hx.device)
        out = {"rows": len(wt), "rows_train": int(train.sum()), "rows_test": int(test.sum())}
        src = {"ffn_gate_up_exps": hx, "ffn_down_exps": ax}
        for g in groups_:
            x = src[g.key]
            out[g.key] = tuple(hessian((x.index_select(0, i), wt.index_select(0, i)), self.dev) for i in (itr, ite))
        return out
