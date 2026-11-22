"""Score one GGUF against several KLD bases in a single llama-kld-swap load.

    python -m ggufrefine.eval.kld --gguf X.gguf --out logs/kld/X \
        --base combined-c8192=/path/combined.kld --base wiki-c512=/path/wiki-c512.kld [--extra "-ub 4096"]

Writes <out>/<name>.{json,chunks.tsv,tokens.bin} per base; a base whose json exists is skipped (resume).
Defaults: -c 8192 -b 8192 -ub 8192 -np 16 --kv-unified -fa on --fit on -lm dio, GGML_CUDA_MMQ_PREC=q8 (MXFP4 /
NVFP4 otherwise run with 4-bit activations on Blackwell). A c512 base packs 16 sequences per 8192 batch, which
is what -np 16 and the unified KV cache are for.
"""
import argparse
import json
import sys
from pathlib import Path

from .kldswap import KldSwap

DEFAULT_ARGS = ["-c", "8192", "-b", "8192", "-ub", "8192", "-np", "16", "--kv-unified", "-fa", "on",
                "--fit", "on", "-lm", "dio", "-t", "48"]
DEFAULT_ENV = {"GGML_CUDA_MMQ_PREC": "q8"}


def run(gguf, out, bases, extra=(), env=None):
    """bases: {name: path}; returns {name: summary dict}"""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    args = ["-m", str(gguf), *DEFAULT_ARGS, *extra]
    envs = {**DEFAULT_ENV, **(env or {})}
    with KldSwap(args, log=str(out / "kldswap.log"), env=envs) as ks:
        for name, path in bases.items():
            o = out / name
            meta = {"gguf": str(Path(gguf).resolve()), "base": str(Path(path).resolve()), "args": args, "env": envs}
            mp = Path(f"{o}.meta.json")
            if Path(f"{o}.json").exists():
                old = json.load(open(mp)) if mp.exists() else None
                if old == meta:
                    print(f"{name}: exists for this gguf/base/settings, skip", flush=True)
                    res[name] = json.load(open(f"{o}.json"))
                    continue
                raise SystemExit(f"{o}.json exists for a different gguf/base/settings ({mp}); use another --out")
            ks.base(str(path))
            ks.eval(out=str(o), tokens=True)
            with open(mp, "w") as fh:
                json.dump(meta, fh, indent=1)
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
