"""Input Grams of every weight matmul, captured from llama.cpp's own graph (libllama-capture, native patch 0010).

llama.cpp evaluates the model (any architecture it supports, any placement: -ngl / -sm / --fit / -lm); its eval
callback hands each MUL_MAT input to this module as a device pointer, which is wrapped as a torch tensor without a
copy and accumulated as G += x^T x (fp32 GEMM per <= 8192 rows, fp64 accumulator on the GPU that holds the
activation, see grams.Grams). Weights that read the same input tensor (same graph name and data: q/k/v, gate/up)
share one Gram; the group is named after its first weight. Output: one part directory in the grams format (layer-<L>/ and layer-output/), merged
into train / test splits by merge_grams.

    python -m ggufrefine.capture.ggml --tokens T.npy --seqs 0:40 --part-id 0 --out grams-src/part00 \\
        -- -m model-BF16.gguf -c 65536 -b 8192 -ub 8192 -np 8 --kv-unified -fa on -ngl 99 -sm layer

The token file is int32 [n_seq, n_ctx] (python -m ggufrefine.capture.ggml tokenize ...); every call evaluates
-np sequences at once.
"""
import argparse
import ctypes
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .. import paths
from . import grams as gram_io

MUL_MAT, MUL_MAT_ID = 0, 1                              # op codes of the lc callback


class LcTensor(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("type", ctypes.c_int32), ("device", ctypes.c_int32),
                ("ne", ctypes.c_int64 * 4), ("nb", ctypes.c_uint64 * 4), ("name", ctypes.c_char * 64)]


CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32,
                      ctypes.POINTER(LcTensor), ctypes.POINTER(LcTensor))


class _Cuda:
    """__cuda_array_interface__ view of a ggml tensor (2-D rows of ne0 elements)"""

    def __init__(self, t):
        assert t.type == 0, f"ggml type {t.type}: only f32 activations are captured"
        rows = t.ne[1] * t.ne[2] * t.ne[3]
        assert t.nb[0] == 4 and (t.ne[2] * t.ne[3] == 1 or t.nb[2] == t.nb[1] * t.ne[1]), "non-contiguous rows"
        self.__cuda_array_interface__ = {"shape": (int(rows), int(t.ne[0])), "typestr": "<f4",
                                         "strides": (int(t.nb[1]), 4), "data": (int(t.data), False), "version": 3}


def layer_of(name):
    m = re.match(r"blk\.(\d+)\.", name)
    return int(m.group(1)) if m else "output"


class LlamaCapture:
    """lc = LlamaCapture(llama_args, weights); lc.decode(tokens [n_seq, n_tok]); lc.flush() -> {layer: flushed}

    weights: the GGUF tensor names to capture (default: every tensor llama.cpp multiplies with that is named
    blk.* / output / token_embd)."""

    def __init__(self, args, weights=None, min_rows=16):
        lib = ctypes.CDLL(str(paths.LLAMA_BIN / "libllama-capture.so"), mode=ctypes.RTLD_GLOBAL)
        lib.lc_open.restype = ctypes.c_void_p
        lib.lc_open.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), CB, ctypes.c_void_p]
        lib.lc_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]
        lib.lc_decode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.c_int32]
        lib.lc_close.argtypes = [ctypes.c_void_p]
        self.lib, self.weights, self.min_rows = lib, None if weights is None else set(weights), min_rows
        self.G = gram_io.Grams()
        self.G.on = True
        self.groups = {}           # group key -> [weights]
        self.cur, self.seen = {}, set()
        self.frozen = False        # group membership is fixed after the first ubatch
        self.gemm_s, self.n_cb, self.err = 0.0, 0, None
        self._cb = CB(self._on)
        argv = [b"llama-capture"] + [a.encode() for a in args]
        arr = (ctypes.c_char_p * len(argv))(*argv)
        self.h = lib.lc_open(len(argv), arr, self._cb, None)
        if not self.h:
            raise RuntimeError("lc_open failed (see the llama.cpp log above)")
        info = (ctypes.c_int32 * 6)()
        lib.lc_info(self.h, info)
        self.n_ctx, self.n_batch, self.n_ubatch, self.n_seq_max, self.n_vocab, self.add_bos = list(info)

    def _want(self, name, op, x):
        if op != MUL_MAT or x.type != 0 or x.ne[1] * x.ne[2] * x.ne[3] < self.min_rows:
            return False
        if self.weights is not None:
            return name in self.weights
        return name.startswith("blk.") or name.startswith("output") or name == "token_embd.weight"

    def _on(self, user, name, op, ask, x, ids):
        try:
            name = name.decode()
            if ask:
                return int(self._want(name, op, x.contents))
            t = x.contents
            if t.device < 0:
                raise RuntimeError(f"{name}: input is not in CUDA memory (device {t.device}); keep the layers on GPUs")
            if name in self.seen:                                          # every weight runs once per ubatch
                self.seen.clear()
                self.cur.clear()
                self.frozen = True
            self.seen.add(name)
            key = (t.name, t.data, tuple(t.ne), tuple(t.nb))                # graph allocator reuses addresses
            g = self.cur.get(key)
            if g is not None:                                              # same input as an earlier weight
                if name not in self.groups[g]:
                    if self.frozen:
                        raise RuntimeError(f"{name} joined group {g} after the first ubatch")
                    self.groups[g].append(name)
                return 1
            g = name
            if g not in self.groups:
                if self.frozen:
                    raise RuntimeError(f"new input group {g} after the first ubatch")
                self.groups[g] = [name]
            self.cur[key] = g
            dev = torch.device("cuda", t.device)
            xs = torch.as_tensor(_Cuda(t), device=dev)
            t0 = time.perf_counter()
            with torch.cuda.device(dev):
                self.G.dense(g, [g], xs)
                torch.cuda.synchronize(dev)                                # ggml reuses the buffer after we return
            self.gemm_s += time.perf_counter() - t0
            self.n_cb += 1
            return 1
        except Exception as e:                                             # an exception cannot cross the C frame
            if self.err is None:
                self.err = e
            return 0

    def decode(self, tokens):
        tokens = np.ascontiguousarray(tokens, np.int32)
        n_seq, n_tok = tokens.shape
        rc = self.lib.lc_decode(self.h, tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), n_seq, n_tok)
        if self.err is not None:
            raise self.err
        if rc != 0:
            raise RuntimeError(f"lc_decode returned {rc}")

    def flush(self):
        """-> {layer: {group: (info, tri f32, sum f64)}} with info["weights"] = the group's weights"""
        out = {}
        for g, (info, tri, s) in self.G.flush().items():
            info = dict(info, weights=list(self.groups[g]))
            out.setdefault(layer_of(g), {})[g] = (info, tri, s)
        self.G = gram_io.Grams()
        self.G.on = True
        return out

    def close(self):
        if self.h:
            self.lib.lc_close(self.h)
            self.h = None


def tokenize(text_path, model, n_ctx, out, n_seq=None):
    """text -> int32 [n_seq, n_ctx] with llama.cpp's tokenizer (llama-tokenize: vocab only, BOS per the vocab);
    whole n_ctx chunks of the token stream, as llama-imatrix / llama-perplexity cut them"""
    import subprocess
    r = subprocess.run([str(paths.tool("llama-tokenize")), "-m", str(model), "-f", str(text_path), "--ids", "--log-disable"],
                       check=True, capture_output=True, text=True)
    ids = np.array(json.loads(r.stdout.strip().splitlines()[-1]), np.int32)
    n = len(ids) // n_ctx if n_seq is None else n_seq
    if n * n_ctx > len(ids):
        raise ValueError(f"{len(ids)} tokens: fewer than {n} x {n_ctx}")
    toks = ids[:n * n_ctx].reshape(n, n_ctx)
    np.save(out, toks)
    return toks


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "tokenize":
        ap = argparse.ArgumentParser(prog="ggufrefine.capture.ggml tokenize")
        ap.add_argument("--text", required=True, type=Path)
        ap.add_argument("--model", required=True, type=Path, help="any GGUF of the model (vocab)")
        ap.add_argument("--n-ctx", type=int, required=True)
        ap.add_argument("--n-seq", type=int, default=None)
        ap.add_argument("--out", type=Path, required=True)
        a = ap.parse_args(argv[1:])
        t = tokenize(a.text, a.model, a.n_ctx, a.out, a.n_seq)
        print(f"{t.shape[0]} x {t.shape[1]} tokens -> {a.out}")
        return 0
    llama = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", required=True, type=Path)
    ap.add_argument("--seqs", default=None, help="slice of sequences, e.g. 0:40")
    ap.add_argument("--part-id", type=int, required=True, help="recorded in the Gram metadata (merge_grams splits on it)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--reserve-gb", type=float, default=8.0)
    a = ap.parse_args(argv)
    toks = np.load(a.tokens)
    s0, s1 = (0, toks.shape[0]) if a.seqs is None else map(int, a.seqs.split(":"))
    toks = toks[s0:s1]
    lc = LlamaCapture(llama)
    n_par = lc.n_seq_max
    print(f"[{time.strftime('%H:%M:%S')}] {toks.shape[0]} sequences x {toks.shape[1]}, {n_par} per call "
          f"(n_ctx {lc.n_ctx}, n_batch {lc.n_batch}, n_ubatch {lc.n_ubatch})", flush=True)
    t0 = time.time()
    for i in range(0, toks.shape[0], n_par):
        lc.decode(toks[i:i + n_par])
        done = min(i + n_par, toks.shape[0])
        el = time.time() - t0
        print(f"[{time.strftime('%H:%M:%S')}] {done}/{toks.shape[0]} sequences  {done * toks.shape[1] / el:.0f} tok/s  "
              f"gram {lc.gemm_s:.0f}s of {el:.0f}s  eta {el / done * (toks.shape[0] - done) / 60:.1f} min", flush=True)
    meta = {"part": a.part_id, "seqs": [s0, s1], "n_ctx": int(toks.shape[1]), "tokens_file": str(a.tokens.resolve()),
            "capture": "llama.cpp graph (ggufrefine.capture.ggml)", "llama_args": llama}
    for layer, fl in lc.flush().items():
        gram_io.write_layer(a.out, layer, fl, dict(meta, saved=time.strftime("%Y-%m-%d %H:%M:%S")), a.reserve_gb)
    lc.close()
    print(f"[{time.strftime('%H:%M:%S')}] done in {(time.time() - t0) / 60:.1f} min -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
