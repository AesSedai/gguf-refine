"""Per-layer prior for the plan's routed-expert error weight (output feeds alloc.plan --weights after
combining roles).

errtab error is sum_t w_t^2 |dW x_t|^2 (routing-weighted, latent units, before routed_expert_norm). The norm divides
the summed routed output acc_t by rms_t, then diag(g) and routed_expert_up_proj U map it into the residual stream,
so the same error reaches the residual stream scaled by
  f_L = sum_{t,k} w_tk^2 / (rms_t^2 + eps) / sum_{t,k} w_tk^2   (routing-weighted mean of 1/(rms^2+eps))
  G_L = |U diag(g)|_F^2 / 3584                              (mean squared gain for an isotropic error)
Neither accounts for residual growth with depth or how later layers propagate the error; KLD probes calibrate that.

    python -m ggufrefine.models.kimi_k3.layer_scale --model HF_DIR --dump CAPTURE_ROOT \
        --out logs/errtab/layer_scale.json
"""
import argparse
import concurrent.futures as cf
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import model as km
from .expert_dump import ExpertDump


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--layers", default="1-92")
    a = ap.parse_args(argv)
    with open(a.model / "config.json") as fh:
        eps = json.load(fh)["text_config"]["rms_norm_eps"]
    W = km.Weights(a.model, torch.device("cpu"))
    P = km.PFX

    def one(L):
        d = ExpertDump(a.dump, L, model=a.model)
        w2 = d.w.astype(np.float64) ** 2                                     # [N, k]
        inv = 1.0 / (d.cols["rms"].astype(np.float64) ** 2 + eps)           # rms_norm: x * rsqrt(mean(x^2) + eps)
        f = float((w2.sum(1) * inv).sum() / w2.sum())
        p = f"{P}layers.{L}.block_sparse_moe."
        g = W.f32(p + "routed_expert_norm.weight")
        U = W.f32(p + "routed_expert_up_proj.weight")
        G = float(((U * g[None, :]) ** 2).sum() / U.shape[1])
        rms = d.cols["rms"]
        return L, dict(f=f, G=G, fG=f * G, rms_median=float(np.median(rms)), rms_p10=float(np.quantile(rms, 0.1)),
                       rms_p90=float(np.quantile(rms, 0.9)), g_mean=float(g.mean()))

    lo, hi = map(int, a.layers.split("-"))
    out, t0 = {}, time.time()
    with cf.ThreadPoolExecutor(a.threads) as ex:
        for L, r in ex.map(one, range(lo, hi + 1)):
            out[L] = r
            print(f"[{time.time() - t0:5.0f}s] L{L}: f {r['f']:.4g} G {r['G']:.4g} fG {r['fG']:.4g} rms med {r['rms_median']:.4g}", flush=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
