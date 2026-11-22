"""Kimi-K3.1 building blocks shared by the capture forward and the expert-dump reader, written from vLLM's
implementation (vllm/models/kimi_k3/nvidia/{model,kda,dsa}.py, ops/attn_res.py): safetensors weight lookup,
RMSNorm, the situ activation, compressed-tensors MXFP4 dequantisation and the attention-residual mix."""
import json
from pathlib import Path

import torch
from safetensors import safe_open

PFX = "language_model.model."
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


class Weights:
    """Tensor lookup over a set of safetensors shards via the index."""

    def __init__(self, model_dir: Path, dev):
        self.dir = Path(model_dir)
        self.dev = dev
        with open(self.dir / "model.safetensors.index.json") as fh:
            self.wm = json.load(fh)["weight_map"]
        self._open = {}

    def _h(self, shard):
        if shard not in self._open:
            self._open[shard] = safe_open(str(self.dir / shard), framework="pt", device="cpu")
        return self._open[shard]

    def raw(self, name):
        return self._h(self.wm[name]).get_tensor(name)

    def f32(self, name):
        return self.raw(name).to(self.dev, non_blocking=True).float()

    def has(self, name):
        return name in self.wm


def rms_norm(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def situ(g, u, beta, lin_beta):
    a = beta * torch.tanh(g / beta) * torch.sigmoid(g)
    if lin_beta is not None:
        u = lin_beta * torch.tanh(u / lin_beta)
    return a * u


def mxfp4_dequant(packed, scale):
    """compressed-tensors mxfp4-pack-quantized: low nibble = even element,
    E2M1 magnitude + sign bit 3, one E8M0 scale per 32 consecutive inputs."""
    lo = packed & 0x0F
    hi = packed >> 4
    codes = torch.stack((lo, hi), dim=-1).reshape(packed.shape[0], -1)  # [out, in]
    mag = E2M1.to(packed.device)[(codes & 0x07).long()]
    val = torch.where((codes & 0x08).bool(), -mag, mag)
    s = torch.exp2(scale.float() - 127.0)                                 # [out, in/32]
    return (val.view(val.shape[0], -1, 32) * s.unsqueeze(-1)).reshape(val.shape[0], -1)


def res_mix(cur, stack, norm_w, proj_w, eps):
    """AttnRes (vllm ops/attn_res.py): softmax over [stack..., cur] of
    qk . RMSNorm(src; norm_w); mix the RAW sources. Returns (mixed, probs)."""
    if not stack:
        return cur, None
    srcs = torch.stack(list(stack) + [cur], dim=1)                       # [T, S, D]
    qk = (norm_w * proj_w).view(1, 1, -1)
    rstd = torch.rsqrt(srcs.pow(2).mean(-1) + eps)                        # [T, S]
    logits = (srcs * qk).sum(-1) * rstd
    p = torch.softmax(logits, dim=-1)
    return (p.unsqueeze(-1) * srcs).sum(1), p
