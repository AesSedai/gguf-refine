"""Data check (CPU, reads ~70 MB): the K3.1 expert weights from a BF16/MXFP4 GGUF via the generic ggufio path equal
the native safetensors MXFP4 dequant, so a GGUF can replace the safetensors as the weight source.
Runs when K31_MODEL (HF directory) and K31_GGUF (BF16/MXFP4 GGUF of the same model) are set, else skipped."""
import os
import unittest
from pathlib import Path

os.environ["GGUF_REFINE_GPUQ"] = "0"
import numpy as np  # noqa: E402

GGUF = Path(os.environ.get("K31_GGUF", "/nonexistent"))
HF = Path(os.environ.get("K31_MODEL", "/nonexistent"))


@unittest.skipUnless(GGUF.exists() and HF.exists(), "set K31_GGUF and K31_MODEL")
class K31Weights(unittest.TestCase):
    def test_gguf_equals_safetensors(self):
        from ggufrefine import ggufio
        from ggufrefine.models import kimi_k3
        g, st = ggufio.Gguf(GGUF), kimi_k3.SafetensorsWeights(HF)
        for L, e in ((1, 0), (46, 346), (92, 895)):
            for grp in kimi_k3.groups(L):
                for name in grp.tensors:
                    with self.subTest(name=name, e=e):
                        self.assertEqual(g.info[name].type, "MXFP4")
                        a, b = g.weights(name, e), st.weights(name, e)
                        self.assertEqual(a.shape, b.shape)
                        self.assertTrue(np.array_equal(a, b), f"max diff {np.abs(a - b).max()}")
        g.close()


if __name__ == "__main__":
    unittest.main()
