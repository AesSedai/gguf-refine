"""Reader for the routed-expert dump written by capture/forward.py --expert-dump-layers (writer and column spec:
forward.ExpertDumpWriter).

A capture of several parts holds one directory per part and layer, <root>/part*/x2/layer-<L>. ExpertDump joins the
parts of one layer: rows(e) -> (h fp16 [n, 3584], a [n, 3072], w f32 [n]) for every token routed to expert e, rows
in sorted part order and, within a part, in token order; part_ids(e) gives each row's part. h and w are read from
the dump; a = situ(h @ w1^T, h @ w3^T) is recomputed on the GPU from the native MXFP4 w1/w3 of expert e, in fp32
with TF32 off, as the forward pass computes it (the forward pass feeds the fp32 latent, the dump holds its fp16
rounding). meta(e) gives per-row tok / seq / pos / token / rms / slot.

GpuExpertDump holds the layer's h on the GPU (16.7 GiB per K3.1 layer) and returns CUDA tensors; same values.

    d = GpuExpertDump("/data/capture", 46, model="/models/Kimi-K3.1-Pro")
    h, a, w = d.rows(17)
"""
import concurrent.futures as cf
import json
import os
from pathlib import Path

import numpy as np
import torch

from . import model as km

FORMAT = "k31-xdump2"           # meta.json "format" of a dump layer directory
COLUMNS = (("h", "h.f16", "float16"), ("ids", "ids.i16", "int16"), ("w", "w.f32", "float32"),
           ("tok", "tok.i64", "int64"), ("seq", "seq.i32", "int32"), ("pos", "pos.i32", "int32"),
           ("token", "token.i32", "int32"), ("rms", "rms.f32", "float32"))


def _no_tf32(fn):
    def wrap(*args, **kw):
        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            return fn(*args, **kw)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32
    return wrap


class ExpertDump:
    def __init__(self, root, layer, model, dev="cuda:0", subdir="x2", a_dtype=np.float16, chunk=32768):
        """root: directory holding part*/<subdir>/layer-<layer>, or a single layer directory. model: the HF
        directory (config and expert weights). a_dtype: dtype of the returned a (float32 keeps the recomputed values
        unrounded)."""
        root = Path(root)
        dirs = [root] if (root / "meta.json").exists() else sorted(root.glob(f"part*/{subdir}/layer-{layer}"))
        if not dirs:
            raise FileNotFoundError(f"no expert dump of layer {layer} under {root}")
        self.layer, self.dev, self.a_dtype, self.chunk = layer, torch.device(dev), a_dtype, chunk
        self.model = Path(model)
        with open(self.model / "config.json") as fh:
            tc = json.load(fh)["text_config"]
        self.dirs, self.h, self.metas = dirs, [], []
        ids, w, cols = [], [], {k: [] for k in ("tok", "seq", "pos", "token", "rms")}
        for d in dirs:
            if not (d / "meta.json").exists():
                raise FileNotFoundError(f"{d}: no meta.json, dump incomplete")
            with open(d / "meta.json") as fh:
                m = json.load(fh)
            assert m["format"] == FORMAT and m["layer"] == layer, (d, m["format"], m["layer"])
            n, K, k = m["n"], m["k_gate_up"], m["topk"]

            def col(name, shape):
                c = m["columns"][name]
                f = d / c["file"]
                want = int(np.prod(shape)) * np.dtype(c["dtype"]).itemsize
                if f.stat().st_size != want:
                    raise ValueError(f"{f}: {f.stat().st_size} bytes, meta says {want}")
                return np.memmap(f, dtype=c["dtype"], mode="r", shape=tuple(shape))

            self.h.append(col("h", (n, K)))
            ids.append(np.asarray(col("ids", (n, k))))
            w.append(np.asarray(col("w", (n, k))))
            for c in cols:
                cols[c].append(np.asarray(col(c, (n,))))
            self.metas.append(m)
        self.K, self.topk = K, k
        self.n_part = np.array([m["n"] for m in self.metas], np.int64)
        self.part_off = np.concatenate([[0], np.cumsum(self.n_part)])          # token offset of each part
        self.ids = np.concatenate(ids)                                         # [N, k] int16
        self.w = np.concatenate(w)                                             # [N, k] f32
        self.cols = {c: np.concatenate(v) for c, v in cols.items()}
        flat = self.ids.reshape(-1).astype(np.int32)
        if flat.min() < 0:
            raise ValueError("negative expert id in dump")
        self.E = tc["num_experts"]
        assert flat.max() < self.E
        self.order = np.argsort(flat, kind="stable")                           # (part, token, slot) order per expert
        self.bounds = np.concatenate([[0], np.cumsum(np.bincount(flat, minlength=self.E))])
        self._W = None
        self.beta, self.lin_beta = tc["activation_situ_beta"], tc["activation_situ_linear_beta"]
        self.k_down = tc["moe_intermediate_size"]

    def count(self, e):
        return int(self.bounds[e + 1] - self.bounds[e])

    def _sel(self, e):
        s = self.order[self.bounds[e]:self.bounds[e + 1]]
        return s // self.topk, s % self.topk                                   # global token row, slot

    def _h(self, g):
        p = np.searchsorted(self.part_off, g, side="right") - 1
        out = np.empty((len(g), self.K), np.float16)
        for i in np.unique(p):                                                 # loop over parts only
            m = p == i
            out[m] = self.h[i][g[m] - self.part_off[i]]
        return out

    def weights(self, e):
        """native MXFP4 w1 (gate), w3 (up) of expert e, dequantised on the GPU to fp32 [3072, 3584]"""
        if self._W is None:
            self._W = km.Weights(self.model, torch.device("cpu"))
        p = f"{km.PFX}layers.{self.layer}.block_sparse_moe.experts.{e}."
        return [km.mxfp4_dequant(self._W.raw(p + f"{n}.weight_packed").to(self.dev),
                                 self._W.raw(p + f"{n}.weight_scale").to(self.dev)) for n in ("w1", "w3")]

    @_no_tf32
    def activations(self, e, h):
        """a = situ(h @ w1^T, h @ w3^T) in fp32 on the GPU, TF32 off; torch [n, 3072]"""
        w1, w3 = self.weights(e)
        ht = torch.as_tensor(h).to(self.dev).float() if not torch.is_tensor(h) else h.to(self.dev).float()
        out = torch.empty(ht.shape[0], w1.shape[0], dtype=torch.float32, device=self.dev)
        for r in range(0, ht.shape[0], self.chunk):
            x = ht[r:r + self.chunk]
            out[r:r + self.chunk] = km.situ(x @ w1.t(), x @ w3.t(), self.beta, self.lin_beta)
        return out

    def rows(self, e):
        g, slot = self._sel(e)
        h = self._h(g)
        w = self.w[g, slot]
        if len(g) == 0:
            return h, np.empty((0, self.k_down), self.a_dtype), w
        a = self.activations(e, h)
        a = (a.half() if self.a_dtype == np.float16 else a).cpu().numpy()
        return h, a, w

    def part_ids(self, e):
        """index of the dump part each row of rows(e) came from (parts hold disjoint sequences)"""
        g, _ = self._sel(e)
        return (np.searchsorted(self.part_off, g, side="right") - 1).astype(np.int16)

    def meta(self, e):
        """per row of rows(e): tok (global token index into the tokens file), seq, pos, token id,
        rms of the summed routed output of that token, and slot (rank of e in the token's top-k)"""
        g, slot = self._sel(e)
        out = {c: v[g] for c, v in self.cols.items()}
        out["slot"] = slot.astype(np.int16)
        return out


class GpuExpertDump(ExpertDump):
    """ExpertDump with the layer's h streamed from disk in chunks straight into one fp16 tensor on the device;
    rows(e) gathers and recomputes a on the GPU and returns torch CUDA tensors."""

    def __init__(self, root, layer, model, dev="cuda:0", threads=8, chunk=256 << 20):
        super().__init__(root, layer, model=model, dev=dev)
        N = int(self.n_part.sum())
        self.H = torch.empty((N, self.K), dtype=torch.float16, device=self.dev)
        Hb = self.H.view(torch.uint8).view(-1)
        jobs = []
        for i, mm in enumerate(self.h):
            base = int(self.part_off[i]) * self.K * 2
            nb = int(mm.shape[0]) * self.K * 2
            for off in range(0, nb, chunk):
                jobs.append((mm.filename, off, min(chunk, nb - off), base + off))

        def rd(j):
            fn, off, n, dst = j
            buf = torch.empty(n, dtype=torch.uint8).pin_memory()
            mv = memoryview(buf.numpy())
            fd = os.open(fn, os.O_RDONLY)
            try:
                got = 0
                while got < n:
                    k = os.preadv(fd, [mv[got:]], off + got)
                    if k <= 0:
                        raise IOError(f"{fn}: short read at {off + got}")
                    got += k
            finally:
                os.close(fd)
            Hb[dst:dst + n].copy_(buf)
            torch.cuda.current_stream(self.dev).synchronize()
            return n
        with cf.ThreadPoolExecutor(threads) as ex:
            self.bytes_read = sum(ex.map(rd, jobs))
        self.h = None                                                         # no host copy

    @_no_tf32
    def activations(self, e, h):
        """as ExpertDump.activations, with the fp32 conversion done per chunk (no full fp32 copy of h)"""
        w1, w3 = self.weights(e)
        out = torch.empty(h.shape[0], w1.shape[0], dtype=torch.float32, device=self.dev)
        for r in range(0, h.shape[0], self.chunk):
            x = h[r:r + self.chunk].float()
            out[r:r + self.chunk] = km.situ(x @ w1.t(), x @ w3.t(), self.beta, self.lin_beta)
        return out

    def rows(self, e):
        g, slot = self._sel(e)
        w = torch.from_numpy(np.ascontiguousarray(self.w[g, slot])).to(self.dev)
        if len(g) == 0:
            return (torch.empty((0, self.K), dtype=torch.float16, device=self.dev),
                    torch.empty((0, self.k_down), dtype=torch.float16, device=self.dev), w)
        h = self.H.index_select(0, torch.from_numpy(g.astype(np.int64)).to(self.dev))
        a = self.activations(e, h).half()                                     # same fp16 rounding as the host path
        return h, a, w
