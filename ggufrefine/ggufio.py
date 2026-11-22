"""GGUF access by tensor name: layout, per-expert byte slices, dequantized weights, imatrix rows, and
verified in-place writes.

Shapes follow ggml order (ne0 = K, ne1 = R, ne2 = experts). A 2-D tensor is a single "expert" (E = 1),
so dense and MoE matrices go through the same calls. Reads use explicit preads (no mmap page faults:
~10x faster cold on virtiofs).
"""
import concurrent.futures as cf
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import paths  # noqa: F401
from gguf import GGMLQuantizationType as QT, GGUFReader  # noqa: E402
from gguf.constants import GGML_QUANT_SIZES  # noqa: E402
from gguf.quants import dequantize  # noqa: E402


@dataclass(frozen=True)
class TensorInfo:
    name: str
    offset: int          # absolute file offset of the data
    nbytes: int
    type: str            # GGMLQuantizationType name
    shape: tuple         # ggml order (ne0, ne1, ...)

    @property
    def n_experts(self):
        return self.shape[2] if len(self.shape) >= 3 else 1

    @property
    def rows(self):      # R per expert
        return self.shape[1] if len(self.shape) >= 2 else 1

    @property
    def cols(self):      # K
        return self.shape[0]

    @property
    def expert_bytes(self):
        return self.nbytes // self.n_experts


def tensor_table(path):
    """name -> TensorInfo for every tensor in the GGUF (header parse only)"""
    r = GGUFReader(str(path))
    out = {t.name: TensorInfo(t.name, int(t.data_offset), int(t.n_bytes), t.tensor_type.name, tuple(int(x) for x in t.shape))
           for t in r.tensors}
    del r
    return out


def bytes_per_row(qtype, K):
    block, size = GGML_QUANT_SIZES[QT[qtype] if isinstance(qtype, str) else qtype]
    assert K % block == 0, (qtype, K, block)
    return K // block * size


def pread_all(fd, n, off):
    buf = bytearray(n)
    mv = memoryview(buf)
    got = 0
    while got < n:
        k = os.preadv(fd, [mv[got:]], off + got)
        if k <= 0:
            raise IOError(f"short read at {off + got}")
        got += k
    return buf


class Gguf:
    """read access to one GGUF; expert(name, e) -> raw bytes [R, bytes_per_row]; weights(name, e) -> fp32 [R, K].

    overrides: optional backup dir with manifest.json + <tensor>.orig (tensors patched in place whose original
    bytes should be read instead), as written by ggufrefine.patch."""

    def __init__(self, path, overrides=None):
        self.path = Path(path)
        self.info = tensor_table(path)
        self.fd = os.open(path, os.O_RDONLY)
        self.orig = {}
        if overrides is not None and (Path(overrides) / "manifest.json").exists():
            for name, m in json.load(open(Path(overrides) / "manifest.json")).items():
                p = Path(overrides) / f"{name}.orig"
                ti = self.info[name]
                assert p.stat().st_size == ti.nbytes == m["bytes"] and ti.type == m["type"] and ti.offset == m["offset"], (name, p)
                self.orig[name] = os.open(p, os.O_RDONLY)

    def expert(self, name, e=0):
        ti = self.info[name]
        assert 0 <= e < ti.n_experts, (name, e)
        per = ti.expert_bytes
        fd, base = (self.orig[name], 0) if name in self.orig else (self.fd, ti.offset)
        raw = np.frombuffer(pread_all(fd, per, base + e * per), np.uint8)
        return raw.reshape(ti.rows, -1)

    def weights(self, name, e=0):
        ti = self.info[name]
        return np.asarray(dequantize(self.expert(name, e), QT[ti.type]), np.float32).reshape(ti.rows, ti.cols)

    def close(self):
        for fd in [self.fd, *self.orig.values()]:
            os.close(fd)


class Imatrix:
    """llama-imatrix GGUF output: row(name, e) -> per-column importance [K], the vector llama-quantize passes to
    ggml_quantize_chunk (common/imatrix-loader.cpp: in_sum2 / count, all ones for an expert with count 0)."""

    def __init__(self, path):
        self.t = {t.name: np.asarray(t.data) for t in GGUFReader(str(path)).tensors}

    def __contains__(self, name):
        return name + ".in_sum2" in self.t

    def row(self, name, e=0):
        cnt = np.asarray(self.t[name + ".counts"]).ravel()
        s2 = np.asarray(self.t[name + ".in_sum2"]).reshape(cnt.size, -1)[e].astype(np.float32)   # row first, then convert
        c = np.float32(cnt[e])
        return s2 / c if c > 0 else np.ones_like(s2)


def write_verified(path, offset, src, nbytes=None, threads=8, chunk=256 << 20):
    """write src (bytes-like, or a path to a file of exactly nbytes) at offset in path, fsync, read every chunk
    back and compare; returns the sha256 of the written bytes. Chunks are written in parallel (single-stream
    virtiofs writes were ~25 MB/s, 8 streams ~130 MB/s)."""
    from_file = isinstance(src, (str, os.PathLike))
    if from_file:
        n = os.stat(src).st_size
        sfd = os.open(src, os.O_RDONLY)
    else:
        mv = memoryview(src).cast("B")
        n = len(mv)
    if nbytes is not None and n != nbytes:
        raise ValueError(f"source holds {n} bytes, target {nbytes}")
    fd = os.open(path, os.O_RDWR)
    try:
        def one(pos):
            k = min(chunk, n - pos)
            buf = os.pread(sfd, k, pos) if from_file else bytes(mv[pos:pos + k])
            assert len(buf) == k
            w = 0
            while w < k:
                w += os.pwrite(fd, buf[w:], offset + pos + w)
            return pos, buf
        with cf.ThreadPoolExecutor(threads) as ex:
            chunks = dict(ex.map(one, range(0, n, chunk)))
        os.fsync(fd)
        h = hashlib.sha256()
        for pos in sorted(chunks):
            h.update(chunks[pos])
        with cf.ThreadPoolExecutor(threads) as ex:
            ok = all(ex.map(lambda pos: os.pread(fd, len(chunks[pos]), offset + pos) == chunks[pos], sorted(chunks)))
    finally:
        os.close(fd)
        if from_file:
            os.close(sfd)
    if not ok:
        raise IOError(f"{path}@{offset}: read-back differs")
    return h.hexdigest()


def write_expert(path, info, e, raw, threads=4):
    """write one expert's bytes of tensor `info` in place (raw: uint8, info.expert_bytes long)"""
    raw = np.ascontiguousarray(raw, np.uint8).reshape(-1)
    if not 0 <= e < info.n_experts:
        raise ValueError(f"{info.name}: expert {e} outside 0..{info.n_experts - 1}")
    if raw.nbytes != info.expert_bytes:
        raise ValueError(f"{info.name}: {raw.nbytes} bytes, one expert holds {info.expert_bytes}")
    return write_verified(path, info.offset + e * info.expert_bytes, raw, threads=threads)
