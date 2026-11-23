#!/usr/bin/env python3
"""Input Gram matrices of the dense (non-expert) matmuls, accumulated during a capture forward pass.

One accumulator per distinct matmul INPUT (a "group"), shared by every GGUF weight that consumes it:
    G = sum_t x_t x_t^T  [K, K]     s = sum_t x_t  [K]     n = number of tokens
unweighted (dense tensors see every token). Per-head inputs (attn_k_b, attn_v_b: one [K_in, K_out]
matrix per head, each consuming only its own head's slice) keep one Gram per head, [H, K, K] -- the
diagonal blocks of the Gram of the flattened head-major [H*K] input that IMatrix.heads sums.

Precision: fp32 GEMM per chunk of <= 8192 tokens with TF32 off (also under --fast), each chunk
added to an fp64 accumulator on the GPU; s in fp64.

On-disk format "k31-grams" v1, one directory per layer (layer-<il>/, and layer-output/):
    <group>.tri.f32   float32, packed upper triangle in row-major order (row i holds G[i, i:]),
                      K(K+1)/2 values; per-head groups store H such triangles back to back
    <group>.sum.f64   float64 [K] or [H, K]
    meta.json         written last (tmp + rename): groups -> weights, K, heads, n, files
The fp64 -> fp32 rounding on write is relative <= 6e-8 per entry; the fp32 GEMM chunk error is ~1e-6.
unpack_tri() below restores the symmetric matrix.
"""
import contextlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

FORMAT, VERSION = "k31-grams", 1


@contextlib.contextmanager
def fp32_exact():
    """TF32 off for the enclosed matmuls, whatever the global setting is."""
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(prev)


def tri_len(K):
    return K * (K + 1) // 2


def unpack_tri(tri, K):
    """packed row-major upper triangle (numpy, [..., K(K+1)/2]) -> symmetric [..., K, K] float64"""
    lead = tri.shape[:-1]
    out = np.zeros(lead + (K, K), dtype=np.float64)
    iu = np.triu_indices(K)
    out[(Ellipsis,) + iu] = tri
    il = np.tril_indices(K, -1)
    out[(Ellipsis,) + il] = np.swapaxes(out, -1, -2)[(Ellipsis,) + il]
    return out


class Grams:
    def __init__(self, chunk=8192):
        self.on, self.chunk, self.acc = False, chunk, {}
        self.gemm_s = 0.0
        self._masks = {}

    def _ent(self, key, names, K, H, dev):
        e = self.acc.get(key)
        if e is None:
            shape = (K, K) if H is None else (H, K, K)
            e = self.acc[key] = dict(weights=list(names), K=K, heads=H, n=0,
                                     G=torch.zeros(shape, dtype=torch.float64, device=dev),
                                     s=torch.zeros(shape[:-1], dtype=torch.float64, device=dev))
        else:
            assert e["weights"] == list(names) and e["K"] == K and e["heads"] == H, (key, names, K, H)
        return e

    def dense(self, key, names, x):
        """x [N, K]"""
        if not self.on:
            return
        e = self._ent(key, names, x.shape[1], None, x.device)
        with fp32_exact():
            for i in range(0, x.shape[0], self.chunk):
                xc = x[i:i + self.chunk]
                e["G"].add_(torch.mm(xc.t(), xc))
                e["s"].add_(xc.sum(0, dtype=torch.float64))
        e["n"] += x.shape[0]

    def heads(self, key, names, x):
        """x [T, H, K] -> per-head Grams [H, K, K]"""
        if not self.on:
            return
        e = self._ent(key, names, x.shape[2], x.shape[1], x.device)
        with fp32_exact():
            for i in range(0, x.shape[0], self.chunk):
                xc = x[i:i + self.chunk].transpose(0, 1)               # [H, t, K]
                e["G"].add_(torch.bmm(xc.transpose(1, 2), xc))
                e["s"].add_(xc.sum(1, dtype=torch.float64))
        e["n"] += x.shape[0]

    def _mask(self, K, dev):
        if K not in self._masks:
            self._masks = {K: torch.ones(K, K, dtype=torch.bool, device=dev).triu_()}
        return self._masks[K]

    def flush(self):
        """-> {key: (info, tri float32 numpy, sum float64 numpy)}; resets the accumulators."""
        out = {}
        for key, e in self.acc.items():
            G = e["G"]
            m = self._mask(e["K"], G.device)
            tri = (G[m] if e["heads"] is None else G[:, m]).float().cpu().numpy()
            assert tri.shape[-1] == tri_len(e["K"])
            info = {k: e[k] for k in ("weights", "K", "heads", "n")}
            out[key] = (info, tri, e["s"].cpu().numpy())
            e["G"] = e["s"] = None
        self.acc = {}
        self._masks = {}
        return out


def wait_space(d, need, reserve_gb, what, wait=False):
    """block until `need` bytes fit in d with reserve_gb to spare. wait=False (nothing frees space concurrently):
    raise instead. wait=True only when a merger with --delete runs alongside and frees source dirs."""
    t0, last = time.time(), 0.0
    while True:
        free = shutil.disk_usage(d).free
        if free - need >= reserve_gb * 1e9:
            return time.time() - t0
        if not wait:
            raise OSError(f"{what}: {free / 1e9:.1f} GB free in {d}, the layer needs {need / 1e9:.1f} GB + {reserve_gb} GB "
                          f"reserve. Free space on that filesystem (or point the output elsewhere) and rerun")
        if time.time() - last > 60:
            print(f"[{time.strftime('%H:%M:%S')}] {what}: {free / 1e9:.1f} GB free in {d}, need {need / 1e9:.1f} + "
                  f"{reserve_gb} GB reserve; waiting for the merger", flush=True)
            last = time.time()
        time.sleep(2.0)


def write_layer(root, layer, flushed, meta_extra, reserve_gb=8.0, wait=False):
    """One process's Grams of one layer -> root/layer-<layer>/ (meta.json last, atomically)."""
    d = Path(root) / f"layer-{layer}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").unlink(missing_ok=True)
    need = sum(t.nbytes + s.nbytes for _, t, s in flushed.values())
    waited = wait_space(d, need, reserve_gb, f"grams layer {layer}", wait)
    groups = {}
    for key, (info, tri, s) in flushed.items():
        files = {"tri": f"{key}.tri.f32", "sum": f"{key}.sum.f64"}
        for arr, fn in ((tri, files["tri"]), (s, files["sum"])):
            with open(d / fn, "wb") as f:
                f.write(np.ascontiguousarray(arr).tobytes())
                f.flush()
                os.fsync(f.fileno())
        groups[key] = dict(info, files=files, tri_shape=list(tri.shape), sum_shape=list(s.shape))
    meta = dict(format=FORMAT, version=VERSION, layer=layer, groups=groups,
                layout="tri: float32 packed row-major upper triangle (row i = G[i, i:]), per head back to back; "
                       "sum: float64; G = sum_t x x^T, unweighted",
                **meta_extra)
    tmp = d / "meta.json.tmp"
    json.dump(meta, open(tmp, "w"), indent=1)
    os.replace(tmp, d / "meta.json")
    return need, waited
