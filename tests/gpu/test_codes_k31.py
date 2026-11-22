"""Codec bit-exactness of ggufrefine.quant.codes on real Kimi-K3.1 expert weights (CPU; no GPU needed).

For 3 experts of layers 4, 46 and 88 (gate, up and down), the native MXFP4 weights are quantized by ggml with that
expert's imatrix row, for IQ2_XXS, IQ2_XS, IQ3_XXS, IQ2_S and IQ3_S. Required for every tensor:
pack(unpack(bytes)) == bytes block for block, codes.dequant == gguf-py dequantize weight for weight, and for the
parity-sign types every 8-weight sign chunk has even parity and re-packing the re-unpacked bytes is exact.

Skipped unless K31_MODEL (the HF safetensors directory) and K31_IMATRIX (a llama-imatrix GGUF of the model) are set:
    K31_MODEL=/models/Kimi-K3.1-Pro K31_IMATRIX=/models/K3.1-imatrix.gguf \\
        python -m unittest discover -s tests/gpu -p test_codes_k31.py -v
"""
import os
import sys
import time
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ggufrefine.quant import codes as iq  # noqa: E402
from ggufrefine.quant import ggml as gq  # noqa: E402
from gguf import GGMLQuantizationType as QT, GGUFReader  # noqa: E402
from gguf import quants as gquants  # noqa: E402

MODEL, IMATRIX = os.environ.get("K31_MODEL"), os.environ.get("K31_IMATRIX")
LAYERS, N_EXPERTS = (4, 46, 88), 3
TYPES = ("IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ2_S", "IQ3_S")
MATS = {"w1": "ffn_gate_exps", "w3": "ffn_up_exps", "w2": "ffn_down_exps"}     # HF expert matrix -> GGUF tensor


@unittest.skipUnless(MODEL and IMATRIX, "set K31_MODEL and K31_IMATRIX to run")
class K31Codes(unittest.TestCase):
    def test_codes_bit_exact(self):
        from ggufrefine.models.kimi_k3 import model as km
        W8 = km.Weights(Path(MODEL), torch.device("cpu"))
        IT = {t.name: t for t in GGUFReader(IMATRIX).tensors}
        stats = {t: dict(tensors=0, blocks=0, bytes_eq=0, deq_eq=0, weights=0, legal=0) for t in TYPES}
        fails = []
        for L in LAYERS:
            experts = sorted(np.random.default_rng(100 + L).choice(896, 6, replace=False))[:N_EXPERTS]
            for e in experts:
                t0 = time.time()
                for m, g in MATS.items():
                    p = f"{km.PFX}layers.{L}.block_sparse_moe.experts.{e}.{m}."
                    W = km.mxfp4_dequant(W8.raw(p + "weight_packed"), W8.raw(p + "weight_scale")).numpy().astype(np.float32)
                    name = f"blk.{L}.{g}.weight"
                    cnt = float(np.asarray(IT[name + ".counts"].data).ravel()[e])
                    imat = np.asarray(IT[name + ".in_sum2"].data, np.float32).reshape(896, -1)[e] / max(cnt, 1.0)
                    for t in TYPES:
                        raw = np.asarray(gq.quantize_cpu(W, getattr(QT, t), imat), np.uint8).reshape(-1, iq.SPEC[t]["nbytes"])
                        f = iq.unpack(t, raw)
                        re = iq.pack(t, f)
                        ref_deq = gquants.dequantize(raw.reshape(W.shape[0], -1), getattr(QT, t)).reshape(-1)
                        mine = iq.dequant(t, f).reshape(-1)
                        s = stats[t]
                        s["tensors"] += 1; s["blocks"] += len(raw); s["weights"] += mine.size
                        s["bytes_eq"] += int((re == raw).all(1).sum())
                        s["deq_eq"] += int((mine == ref_deq).sum())
                        if t in iq.PARITY:
                            s["legal"] += int(((f["signs"].reshape(len(raw), -1, 8).sum(-1) & 1) == 0).all(1).sum())
                            if not (iq.pack(t, iq.unpack(t, re)) == raw).all():
                                fails.append(f"L{L} e{e} {m} {t}: re-pack of re-unpacked bytes differs")
                        if not ((re == raw).all() and (mine == ref_deq).all()):
                            fails.append(f"L{L} e{e} {m} {t}: bytes {int((re == raw).all(1).sum())}/{len(raw)} "
                                         f"dequant {int((mine == ref_deq).sum())}/{mine.size}")
                print(f"L{L} e{e} {time.time() - t0:.1f}s", flush=True)
        for t, s in stats.items():
            print(f"{t:8s} tensors {s['tensors']:3d}  blocks bytes-equal {s['bytes_eq']}/{s['blocks']}  "
                  f"dequant-equal {s['deq_eq']}/{s['weights']}" + (f"  legal-blocks {s['legal']}/{s['blocks']}" if t in iq.PARITY else ""),
                  flush=True)
            if t in iq.PARITY:
                self.assertEqual(s["legal"], s["blocks"], f"{t}: illegal sign chunks")
        self.assertEqual(fails, [], "\n".join(fails))


if __name__ == "__main__":
    unittest.main()
