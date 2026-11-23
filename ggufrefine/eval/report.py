"""KLD table for several quantizations of one model, overall and per corpus domain, with paired differences.

Reads <dir>/<base>.chunks.tsv as written by ggufrefine.eval.kld (one row per KLD-base chunk). With --manifest (the
ggufrefine.corpus manifest of the eval split the base was built from: one chunk per eval sequence, all of them) the
chunks are also reported per domain; `--pool NAME=PATTERN` adds a row pooling the domains matching a glob (e.g.
all_rp='*rp'), `--without D` the all-domain row without D. Unknown domains or patterns that match none are errors.

KLD is the reference-to-model divergence in nats per scored token (the second half of each chunk), summed over the
reference's tokens with log-prob above -16 (the base stores no smaller ones). PPLx uses the base file's log-probs,
which llama-perplexity clamps to 16 nats below the top logit: on content with many very improbable tokens it is
biased upward (the null shows how much). Paired: each model minus the first, the mean over chunks with a chunk
bootstrap 95% interval (chunks are the independent unit; tokens within one are not) and, in parentheses, relative
to the first.

--null LABEL names the run that scores the reference against its own base. The report exits 3 after printing when its
KLD exceeds --null-max (default 1e-3) on any row: the base and the evals then did not compute the same thing
(different ubatch, context, model file or build), and the differences between the other runs are not meaningful.

    python -m ggufrefine.eval.report --base eval --manifest corpus/manifest.json --null bf16 [--pool all_rp='*rp'] \\
        stock=kld/stock refined-s0=kld/refined-s0 refined-s1=kld/refined-s1 bf16=kld/bf16 [--markdown]
"""
import fnmatch
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
    ap.add_argument("--pool", action="append", default=[], help="NAME=GLOB: a row pooling the matching domains")
    ap.add_argument("--null", default=None, help="label of the reference-vs-its-own-base run")
    ap.add_argument("--null-max", type=float, default=1e-3, help="largest null KLD accepted on any row")
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args(argv)
    if (a.without or a.pool) and not a.manifest:
        ap.error("--without / --pool need --manifest")
    runs = [r.split("=", 1) for r in a.runs]
    if len({lab for lab, _ in runs}) != len(runs):
        ap.error("run labels must be distinct")
    data = {lab: load(d, a.base) for lab, d in runs}
    ref = runs[0][0]
    nchunk = len(data[ref]["chunk"])
    for lab, c in data.items():
        assert (c["chunk"] == data[ref]["chunk"]).all() and (c["n_tok"] == data[ref]["n_tok"]).all(), f"{lab}: chunks differ"
    if a.null and a.null not in data:
        ap.error(f"--null {a.null}: no run with that label")
    subsets = {"all": np.ones(nchunk, bool)}
    if a.manifest:
        dom = np.array(json.load(open(a.manifest))["splits"]["eval"]["domain_of_sequence"])
        if sorted(data[ref]["chunk"].tolist()) != list(range(len(dom))):
            raise SystemExit(f"{runs[0][1]}: chunks {data[ref]['chunk'].min()}..{data[ref]['chunk'].max()} ({nchunk}) do not "
                             f"match the manifest's {len(dom)} eval sequences (incomplete eval, or another base)")
        dom = dom[data[ref]["chunk"]]
        names = sorted(set(dom))
        if "all" in names:
            raise SystemExit("a domain is named 'all', which is the all-domain row")
        for p in a.pool:
            name, _, pat = p.partition("=")
            if name in subsets or name in names or not name:
                raise SystemExit(f"--pool {p}: the name {name!r} is taken (all, a domain, or another pool)")
            sel = np.array([fnmatch.fnmatch(d, pat) for d in dom])
            if not pat or not sel.any():
                raise SystemExit(f"--pool {p}: no domain matches (domains: {', '.join(names)})")
            subsets[name] = sel
        for w in a.without:
            if w not in names:
                raise SystemExit(f"--without {w}: no such domain (domains: {', '.join(names)})")
            subsets[f"all_without_{w}"] = dom != w
        for name, sel in subsets.items():
            if not sel.any():
                raise SystemExit(f"{name}: no chunks (e.g. --without the only domain)")
        for d in names:
            subsets[d] = dom == d
    rng = np.random.default_rng(0)
    null_bad = []
    if a.markdown:
        print("| subset | chunks | " + " | ".join(f"{lab} KLD" for lab, _ in runs) + " | "
              + " | ".join(f"{lab} vs {ref}" for lab, _ in runs[1:]) + " |")
        print("|---|---:|" + "---:|" * len(runs) + "---|" * (len(runs) - 1))
    for name, sel in subsets.items():
        s = {lab: summary(c, sel) for lab, c in data.items()}
        nonfinite = [lab for lab in s if not np.isfinite(s[lab]["kld"])]
        if nonfinite:
            raise SystemExit(f"{name}: KLD not finite for {', '.join(nonfinite)}")
        if a.null and not (abs(s[a.null]["kld"]) <= a.null_max):
            null_bad.append(f"{name}: {s[a.null]['kld']:.3g}")
        diffs = [paired(data[ref], data[lab], sel, rng) for lab, _ in runs[1:]]
        if a.markdown:
            print(f"| {name} | {int(sel.sum())} | " + " | ".join(f"{s[lab]['kld']:.4f}" for lab, _ in runs) + " | "
                  + " | ".join(f"{100 * r:+.1f}% [{lo:+.4f}, {hi:+.4f}]" for d, lo, hi, r in diffs) + " |")
        else:
            print(f"{name:18s} chunks {int(sel.sum()):3d}  " + "  ".join(
                f"{lab} KLD {s[lab]['kld']:.4f} PPLx {s[lab]['ppl_ratio']:.4f} top {100 * s[lab]['same_top']:.2f}%" for lab, _ in runs))
            for (lab, _), (d, lo, hi, r) in zip(runs[1:], diffs):
                print(f"{'':18s}   {lab} - {ref}: {d:+.5f} [{lo:+.5f}, {hi:+.5f}] ({100 * r:+.1f}%)")
    if null_bad:
        print(f"NULL CHECK FAILED: {a.null} KLD above {a.null_max:g} on " + ", ".join(null_bad) + ". The base and the "
              "evals did not compute the same thing (ubatch, context, model file or build differ); the other "
              "differences are not meaningful.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
