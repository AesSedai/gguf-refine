"""Score one GGUF against several KLD bases in a single llama-kld-swap load.

    python -m ggufrefine.eval.kld --gguf X.gguf --out logs/kld/X \
        --base combined-c8192=/path/combined.kld --base wiki-c512=/path/wiki-c512.kld [--extra "-ub 4096"]

Writes <out>/<name>.{json,chunks.tsv,tokens.bin} per base; a base whose json exists for the same gguf / base files
(path, size, mtime) and settings is skipped (resume); one that exists for different ones is an error.
-c and -b are the largest base n_ctx (at least 8192); -ub defaults to the same; --extra overrides any of them.
Defaults: -np 16 --kv-unified -fa on --fit on -lm dio, GGML_CUDA_MMQ_PREC=q8 (MXFP4 /
NVFP4 otherwise run with 4-bit activations on Blackwell). A c512 base packs 16 sequences per 8192 batch, which
is what -np 16 and the unified KV cache are for.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .kldswap import KldSwap

SHARED_ARGS = ["-np", "16", "--kv-unified", "-fa", "on", "--fit", "on", "-lm", "dio", "-t", "48"]
DEFAULT_ARGS = ["-c", "8192", "-b", "8192", "-ub", "8192", *SHARED_ARGS]   # for c8192 bases (eval.probe)
DEFAULT_ENV = {"GGML_CUDA_MMQ_PREC": "q8"}
OUTPUTS = (".json", ".chunks.tsv")


def base_n_ctx(path):
    with open(path, "rb") as fh:
        head = fh.read(20)
    if head[:8] != b"_logits_":
        raise SystemExit(f"{path} is not a llama-perplexity KLD base")
    return int(np.frombuffer(head[8:12], "<i4")[0])


def _ident(p):
    st = Path(p).stat()
    return [str(Path(p).resolve()), st.st_size, st.st_mtime_ns]


def run(gguf, out, bases, extra=(), env=None):
    """bases: {name: path}; returns {name: summary dict}. The model is loaded only if some base has no result."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    c = str(max([8192] + [base_n_ctx(p) for p in bases.values()]))
    args = ["-m", str(gguf), "-c", c, "-b", c, "-ub", c, *SHARED_ARGS, *extra]
    envs = {**DEFAULT_ENV, **(env or {})}
    todo = {}
    for name, path in bases.items():
        o = out / name
        meta = {"gguf": _ident(gguf), "base": _ident(path), "args": args, "env": envs}
        mp = Path(f"{o}.meta.json")
        have = [x for x in OUTPUTS if Path(f"{o}{x}").exists()]
        if have:
            old = json.load(open(mp)) if mp.exists() else None
            if old != meta:
                raise SystemExit(f"{o}.json exists for a different gguf / base file or settings (see {mp}); "
                                 f"remove {o}.* or use another --out")
            if len(have) != len(OUTPUTS):
                raise SystemExit(f"{o}: only {have} of {list(OUTPUTS)} present (interrupted or edited); remove {o}.* to rerun")
            print(f"{name}: exists for this gguf/base/settings, skip", flush=True)
            res[name] = json.load(open(f"{o}.json"))
        else:
            todo[name] = (path, o, mp, meta)
    if not todo:
        return res
    with KldSwap(args, log=str(out / "kldswap.log"), env=envs) as ks:
        for name, (path, o, mp, meta) in todo.items():
            with open(mp, "w") as fh:                   # before the eval: a killed eval leaves meta without results
                json.dump(meta, fh, indent=1)
            ks.base(str(path))
            ks.eval(out=str(o), tokens=True)
            res[name] = json.load(open(f"{o}.json"))
            print(f"== {name}: " + json.dumps({k: v for k, v in res[name].items() if k not in ("per_chunk", "patched")}), flush=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base", action="append", required=True, help="NAME=PATH, repeatable")
    ap.add_argument("--extra", default="", help="extra llama-kld-swap args")
    a = ap.parse_args(argv)
    run(a.gguf, a.out, dict(b.split("=", 1) for b in a.base), a.extra.split())
    return 0


if __name__ == "__main__":
    sys.exit(main())
