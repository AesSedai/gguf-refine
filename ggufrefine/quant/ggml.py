"""Stock ggml quantization of fp32 matrices, as llama-quantize would produce it.

quantize(W, type, imatrix_row) uses ggml_cuda_quantize_weights from the patched libggml-cuda (patch 0009,
byte-identical to ggml_quantize_chunk; test-quantize-cuda) when it supports the type, else
ggml_quantize_chunk from libggml-base on the CPU, multithreaded over row chunks. GGUF_REFINE_GPUQ=0 forces
the CPU path. Both libraries come from paths.LLAMA_BIN.
"""
import concurrent.futures as cf
import ctypes
import os
import threading

import numpy as np

from .. import paths
from gguf import GGMLQuantizationType as QT  # noqa: E402
from gguf.quants import quant_shape_to_byte_shape  # noqa: E402

_LOCK = threading.Lock()
_STATE = {"gpu": None, "gpu_tried": False, "cpu": None}


def cpu_lib():
    """libggml-base of the patched build (ctypes), with ggml_quantize_chunk typed"""
    if _STATE["cpu"] is None:
        lib = ctypes.CDLL(str(paths.LLAMA_BIN / "libggml-base.so"), mode=ctypes.RTLD_GLOBAL)
        lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        lib.ggml_quantize_chunk.restype = ctypes.c_size_t
        lib.ggml_quantize_requires_imatrix.argtypes = [ctypes.c_int]
        lib.ggml_quantize_requires_imatrix.restype = ctypes.c_bool
        _STATE["cpu"] = lib
    return _STATE["cpu"]


def _gpu_lib():
    """the CUDA quantizer, or None if unavailable (no patched build, no GPU, or GGUF_REFINE_GPUQ=0)"""
    if not _STATE["gpu_tried"]:
        _STATE["gpu_tried"] = True
        so = paths.LLAMA_BIN / "libggml-cuda.so"
        if os.environ.get("GGUF_REFINE_GPUQ", "1") != "0" and so.exists():
            try:
                for name in ("libggml-base.so", "libggml.so"):
                    ctypes.CDLL(str(paths.LLAMA_BIN / name), mode=ctypes.RTLD_GLOBAL)
                cu = ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
                cu.ggml_cuda_quantize_weights.restype = ctypes.c_bool
                cu.ggml_cuda_quantize_weights.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                                          ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
                cu.ggml_cuda_quantize_weights_supports.restype = ctypes.c_bool
                cu.ggml_cuda_quantize_weights_supports.argtypes = [ctypes.c_int]
                _STATE["gpu"] = cu
            except (OSError, AttributeError) as ex:
                print(f"GPU quantizer unavailable ({ex}); using ggml_quantize_chunk on the CPU", flush=True)
    return _STATE["gpu"]


def _qtype(t):
    return t if isinstance(t, QT) else getattr(QT, t)


def _imat(imat, K):
    if imat is None:
        return None
    im = np.ascontiguousarray(imat, dtype=np.float32)
    assert im.shape == (K,), (im.shape, K)
    return im


def quantize(W, qtype, imat=None, nthreads=16):
    """W fp32 [R, K] -> uint8 [R, K / block * type_size]; imat: per-column importance [K] or None"""
    qtype = _qtype(qtype)
    src = np.ascontiguousarray(W, dtype=np.float32)
    im = _imat(imat, src.shape[1])
    cu = _gpu_lib()
    if cu is not None and cu.ggml_cuda_quantize_weights_supports(int(qtype)):
        dst = np.zeros(quant_shape_to_byte_shape(src.shape, qtype), dtype=np.uint8)
        with _LOCK:
            ok = cu.ggml_cuda_quantize_weights(int(qtype), src.ctypes.data, dst.ctypes.data, src.shape[0], src.shape[1],
                                               None if im is None else im.ctypes.data)
        if ok:
            return dst
    return quantize_cpu(src, qtype, im, nthreads)


def quantize_cpu(W, qtype, imat=None, nthreads=16):
    """ggml_quantize_chunk over row chunks in threads (ctypes releases the GIL); every row is quantized
    independently, so the bytes do not depend on the chunking"""
    qtype = _qtype(qtype)
    lib = cpu_lib()
    src = np.ascontiguousarray(W, dtype=np.float32)
    R, K = src.shape
    im = _imat(imat, K)
    if im is None and lib.ggml_quantize_requires_imatrix(int(qtype)):
        raise ValueError(f"{qtype.name} requires an importance matrix")
    dst = np.zeros(quant_shape_to_byte_shape(src.shape, qtype), dtype=np.uint8)
    imp = None if im is None else im.ctypes.data_as(ctypes.c_void_p)
    lib.ggml_quantize_chunk(int(qtype), src.ctypes.data_as(ctypes.c_void_p), dst.ctypes.data_as(ctypes.c_void_p),
                            0, 1, K, imp)                                     # initialises the grids once, single-threaded
    chunk = (R + nthreads - 1) // nthreads

    def run(r0):
        return lib.ggml_quantize_chunk(int(qtype), src.ctypes.data_as(ctypes.c_void_p), dst.ctypes.data_as(ctypes.c_void_p),
                                       r0 * K, min(chunk, R - r0), K, imp)
    with cf.ThreadPoolExecutor(nthreads) as ex:
        written = sum(ex.map(run, range(0, R, chunk)))
    assert written == dst.nbytes, (written, dst.nbytes)
    return dst
