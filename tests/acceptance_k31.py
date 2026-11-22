"""GPU regression test: re-refine experts of one K3.1 layer with the package driver and require codes byte-identical
to a reference refined GGUF (an IQ2_XXS model refined with these settings: gate_up and down IQ2_XXS, held-out parts
29,30, lam 1, 8 sweeps, stock codes from the quantizer + imatrix). Needs one free GPU with ~40 GB (the layer's
expert dump is held on the GPU) and the environment in tests/requirements-k31-freeze.txt (other torch / Triton
versions can change fp32 reduction order and so individual codes).

    K31_REFINED=refined.gguf K31_MODEL=hf_dir K31_DUMP=capture_root K31_IMATRIX=imatrix.gguf \
        CUDA_VISIBLE_DEVICES=0 python tests/acceptance_k31.py --layer 46 --experts 0-15
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ggufrefine import ggufio  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int, default=46)
ap.add_argument("--experts", default="0-15")
for k in ("refined", "model", "dump", "imatrix"):
    ap.add_argument(f"--{k}", default=os.environ.get(f"K31_{k.upper()}"), help=f"default $K31_{k.upper()}")
a = ap.parse_args()
missing = [k for k in ("refined", "model", "dump", "imatrix") if not getattr(a, k)]
if missing:
    ap.error("missing " + ", ".join(f"--{k} / K31_{k.upper()}" for k in missing))
lo, hi = map(int, a.experts.split("-"))
ex = list(range(lo, hi + 1))
assert ex and 0 <= lo <= hi < 896, a.experts
tmp = Path(tempfile.mkdtemp(prefix="acc-k31-"))
cmd = [sys.executable, "-u", "-m", "ggufrefine.models.kimi_k3.refine", "--layer", str(a.layer), "--experts", ",".join(map(str, ex)),
       "--model", a.model, "--dump", a.dump, "--test-part", "29,30", "--requantize", "--imatrix", a.imatrix,
       "--types", "gate_up=IQ2_XXS,down=IQ2_XXS", "--results", str(tmp / "results.jsonl"), "--out", str(tmp / "npy"),
       "--batch", "16", "--loaders", "4", "--prefetch", "1"]
print(" ".join(cmd), flush=True)
subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])
g = ggufio.Gguf(a.refined)
same = n = 0
for name in (f"blk.{a.layer}.{t}.weight" for t in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")):
    for e in ex:
        got = np.load(tmp / "npy" / f"{name}-e{e}.npy").reshape(-1)
        n += 1
        same += np.array_equal(got, g.expert(name, e).reshape(-1))
print(f"{same}/{n} expert matrices byte-identical to the reference ({tmp})")
ok = n == 3 * len(ex) and same == n
if ok:
    shutil.rmtree(tmp)
print("ACCEPTANCE PASSED" if ok else "ACCEPTANCE FAILED")
sys.exit(0 if ok else 1)
