"""What the refine driver consumes: groups of matrices, their weights, Hessians and stock codes.

A Group is a set of weight tensors that read the same input (gate+up of an FFN, q/k/v of attention, the
three FFN matrices of nothing else...). They share one Hessian per expert and are refined as one stacked
[sum R, K] problem, so they must share a type. A dense tensor is a group member with one expert.

Sources are duck-typed:
  WeightSource.weights(name, e)                -> fp32 [R, K] (numpy or torch), the unquantized reference
  HessianSource.load(layer, groups, e)         -> {group.key: (H_train, H_test)} torch fp32 [K, K] on the
                                                  refine device, plus optional "rows"/"rows_train"/"rows_test"
  StockCodes.codes(name, type, e)              -> uint8 [R, bytes_per_row], stock llama-quantize bytes
Implementations: ggufio.Gguf (weights and codes from a GGUF), RequantizedCodes (codes from the quantizer +
imatrix), models.kimi_k3 (MXFP4 safetensors weights, dump-derived Hessians).
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Group:
    layer: int
    key: str                 # stable id within the layer, e.g. "ffn_gate_up_exps"
    tensors: tuple           # GGUF tensor names, stacked in this order
    n_experts: int
    K: int                   # shared input width
    rows: tuple              # R of each tensor
    index: int = 0           # position in the layer's full group list (seeds the noise RNG stream)


class GgufCodes:
    """stock codes read from a GGUF (whose tensors must already have the requested type)"""

    def __init__(self, gguf):
        self.g = gguf

    def codes(self, name, t, e):
        ti = self.g.info[name]
        if ti.type != t:
            raise ValueError(f"{name} is {ti.type} in {self.g.path}, requested {t}; use requantized codes")
        return self.g.expert(name, e)


class RequantizedCodes:
    """stock codes recomputed with the ggml quantizer from the reference weights and the imatrix
    (byte-identical to llama-quantize with the same imatrix)"""

    def __init__(self, weights, imatrix, nthreads=16):
        self.w, self.im, self.nthreads = weights, imatrix, nthreads

    def codes(self, name, t, e):
        W = self.w.weights(name, e)
        W = W.cpu().numpy() if hasattr(W, "cpu") else np.asarray(W)
        from ..quant import ggml as gq                                       # native library only when requantizing
        if self.im is not None and name not in self.im:
            raise KeyError(f"{name} has no imatrix entry (pass imatrix=None for an unweighted quantization)")
        imrow = self.im.row(name, e) if self.im is not None else None
        raw = gq.quantize(W.astype(np.float32, copy=False), t, imrow, self.nthreads)
        return np.asarray(raw, np.uint8).reshape(W.shape[0], -1)
