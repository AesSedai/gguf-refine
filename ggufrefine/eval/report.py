"""KLD table for several quantizations of one model, overall and per corpus domain, with paired differences.

Reads <dir>/<base>.chunks.tsv as written by ggufrefine.eval.kld (one row per KLD-base chunk). With --manifest (the
ggufrefine.corpus manifest of the eval split the base was built from, one chunk per eval sequence) the chunks are
grouped by domain; `all_rp` pools every domain whose name ends in "rp", `--without D` adds the all-domain row without
D. PPLx uses the base file's log-probs, which llama-perplexity clamps to 16 nats below the top logit: on content with
many very improbable tokens it is biased upward (the BF16 null shows how much); KLD is not affected. Paired: each model minus the first, the mean over
chunks with a chunk bootstrap 95% interval (chunks are the independent unit; tokens within one are not).

    python -m ggufrefine.eval.report --base eval --manifest corpus/manifest.json \\
        stock=kld/stock refined-s0=kld/refined-s0 refined-s1=kld/refined-s1 [--markdown]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load(d, base):
    rows = [ln.rstrip("\n").split("\t") for ln in open(Path(d) / f"{base}.chunks.tsv")]
    head, rows = rows[0], rows[1:]
    cols = {h: np.array([float(r[i]) for r in rows]) for i, h in enumerate(head)}
    cols["chunk"] = cols["chunk"].astype(int)
    order = np.argsort(cols["chunk"])
    return {k: v[order] for k, v in cols.items()}


def summary(c, sel):
    n = c["n_tok"][sel].sum()
    return {"kld": c["kld_sum"][sel].sum() / n,
            "ppl_ratio": float(np.exp((c["nll_sum"][sel].sum() - c["nll_base_sum"][sel].sum()) / n)),
            "same_top": c["n_same_top"][sel].sum() / n, "chunks": int(sel.sum()), "tokens": int(n)}


def paired(a, b, sel, rng, n_boot=10000):
    """mean KLD of b minus a over the selected chunks (token-weighted), bootstrap over chunks"""
    ka, kb, n = a["kld_sum"][sel], b["kld_sum"][sel], a["n_tok"][sel]
    d = (kb.sum() - ka.sum()) / n.sum()
    idx = rng.integers(0, len(n), (n_boot, len(n)))
    boot = (kb[idx].sum(1) - ka[idx].sum(1)) / n[idx].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return d, lo, hi, d / (ka.sum() / n.sum())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="LABEL=DIR; the first is the reference for paired differences")
    ap.add_argument("--base", required=True, help="base name inside each DIR (<base>.chunks.tsv)")
    ap.add_argument("--manifest", type=Path, default=None, help="corpus manifest: per-domain rows")
    ap.add_argument("--without", action="append", default=[], help="also report all domains except this one")
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args(argv)
    runs = [r.split("=", 1) for r in a.runs]
    data = {lab: load(d, a.base) for lab, d in runs}
    ref = runs[0][0]
    nchunk = len(data[ref]["chunk"])
    for lab, c in data.items():
        assert (c["chunk"] == data[ref]["chunk"]).all() and (c["n_tok"] == data[ref]["n_tok"]).all(), f"{lab}: chunks differ"
    subsets = {"all": np.ones(nchunk, bool)}
    if a.manifest:
        dom = np.array(json.load(open(a.manifest))["splits"]["eval"]["domain_of_sequence"])
        assert len(dom) >= nchunk, (len(dom), nchunk)
        dom = dom[data[ref]["chunk"]]
        subsets["all_rp"] = np.array([d.endswith("rp") for d in dom])
        for w in a.without:
            subsets[f"all_without_{w}"] = dom != w
        for d in sorted(set(dom)):
            subsets[d] = dom == d
    rng = np.random.default_rng(0)
    if a.markdown:
        print("| subset | chunks | " + " | ".join(f"{lab} KLD" for lab, _ in runs) + " | "
              + " | ".join(f"{lab} vs {ref}" for lab, _ in runs[1:]) + " |")
        print("|---|---:|" + "---:|" * len(runs) + "---|" * (len(runs) - 1))
    for name, sel in subsets.items():
        s = {lab: summary(c, sel) for lab, c in data.items()}
        diffs = [paired(data[ref], data[lab], sel, rng) for lab, _ in runs[1:]]
        if a.markdown:
            print(f"| {name} | {int(sel.sum())} | " + " | ".join(f"{s[lab]['kld']:.4f}" for lab, _ in runs) + " | "
                  + " | ".join(f"{100 * r:+.1f}% [{lo:+.4f}, {hi:+.4f}]" for d, lo, hi, r in diffs) + " |")
        else:
            print(f"{name:18s} chunks {int(sel.sum()):3d}  " + "  ".join(
                f"{lab} KLD {s[lab]['kld']:.4f} PPLx {s[lab]['ppl_ratio']:.4f} top {100 * s[lab]['same_top']:.2f}%" for lab, _ in runs))
            for (lab, _), (d, lo, hi, r) in zip(runs[1:], diffs):
                print(f"{'':18s}   {lab} - {ref}: {d:+.5f} [{lo:+.5f}, {hi:+.5f}] ({100 * r:+.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
