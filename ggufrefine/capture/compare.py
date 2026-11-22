"""Compare two Gram captures of the same tokens (e.g. ggml graph vs fp32 reference): per group, relative Frobenius error
|G_a - G_b| / |G_b| of the token-normalised Gram (so captures of different token counts compare), the same for its diagonal (the imatrix), and the cosine of the two
means; a summary per role (group name without the layer) over layers.

    python -m ggufrefine.capture.compare A_DIR B_DIR [--jsonl out.jsonl]      (B is the reference)
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from .grams import tri_len, unpack_tri


def read_group(d, g, info):
    K = info["K"]
    tri = np.fromfile(d / f"{g}.tri.f32", np.float32)
    s = np.fromfile(d / f"{g}.sum.f64", np.float64)
    assert tri.size == tri_len(K), (d, g)
    return unpack_tri(tri.astype(np.float64), K) / info["n"], s / info["n"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path, help="reference")
    ap.add_argument("--jsonl", type=Path, default=None)
    a = ap.parse_args(argv)
    rows, by_role = [], defaultdict(list)
    for mb in sorted(a.b.glob("layer-*/meta.json")):
        db, da = mb.parent, a.a / mb.parent.name
        ma = json.load(open(da / "meta.json"))
        mb = json.load(open(mb))
        for g, ib in mb["groups"].items():
            ia = ma["groups"].get(g)
            if ia is None or ia["weights"] != ib["weights"]:
                print(f"{db.name}/{g}: groups differ ({ia and ia['weights']} / {ib['weights']})")
                continue
            Ga, sa = read_group(da, g, ia)
            Gb, sb = read_group(db, g, ib)
            r = {"layer": db.name.split("-", 1)[1], "group": g, "K": ib["K"], "n": ib["n"],
                 "rel_fro": float(np.linalg.norm(Ga - Gb) / np.linalg.norm(Gb)),
                 "rel_diag": float(np.linalg.norm(np.diag(Ga) - np.diag(Gb)) / np.linalg.norm(np.diag(Gb))),
                 "max_rel_diag": float(np.max(np.abs(np.diag(Ga) - np.diag(Gb)) / np.maximum(np.diag(Gb), 1e-30))),
                 "mean_cos": float(sa @ sb / (np.linalg.norm(sa) * np.linalg.norm(sb) + 1e-300))}
            rows.append(r)
            by_role[re.sub(r"^blk\.\d+\.", "", g)].append(r)
    for role, rs in sorted(by_role.items()):
        f = np.array([r["rel_fro"] for r in rs])
        d = np.array([r["rel_diag"] for r in rs])
        worst = max(rs, key=lambda r: r["rel_fro"])
        print(f"{role:28s} n_layers {len(rs):3d}  rel_fro median {np.median(f):.2e} max {f.max():.2e} (layer {worst['layer']})  "
              f"rel_diag median {np.median(d):.2e} max {d.max():.2e}  mean_cos min {min(r['mean_cos'] for r in rs):.6f}")
    if a.jsonl:
        with open(a.jsonl, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
