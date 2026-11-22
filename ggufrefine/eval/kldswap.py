"""Driver for llama-kld-swap: load a GGUF once, patch tensors in memory, re-run KLD, compare evals pairwise.

Library use:

    from ggufrefine.eval.kldswap import KldSwap, compare
    with KldSwap(["-m", GGUF, "-c", "8192", "-b", "8192", "-ub", "2048", "-fa", "on", "--fit", "on", "-lm", "dio"],
                 log="logs/kldswap.log", env={"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}) as ks:
        ks.base("base.kld")
        a = ks.eval(out="out/A", tokens=True)                 # unpatched
        ks.patch("blk.10.ffn_gate_exps.weight", "gate10.bin", type="iq2_s")
        b = ks.eval(out="out/B", tokens=True)
        ks.revert("all")
        ks.retype("blk.10.ffn_down_exps.weight", "down10.iq3_s.bin", type="iq3_s")   # another type, same shape
        c = ks.eval(out="out/C", tokens=True)
        ks.revert("all")                                       # type, placement and bytes back
    print(compare("out/A", "out/B"))

CLI:
    python -m ggufrefine.eval.kldswap run  [--bin PATH] [--log FILE] [--script CMDS] -- <llama-kld-swap args>
        start the tool, send the commands of CMDS (or stdin) one per line, print every reply
    python -m ggufrefine.eval.kldswap compare A B [--boot 10000] [--json]
        paired comparison of the outputs of two `eval out=A` / `eval out=B` runs (B - A)

Each eval with out=PREFIX writes PREFIX.chunks.tsv (one line per chunk, flushed as it completes), PREFIX.json
(the summary) and, with tokens=1, PREFIX.tokens.bin: "KLDTOK01", int32 T, then per chunk int32 chunk id and
float32 kld[T], p_diff[T], nll[T], nll_base[T], uint8 same_top[T].
"""
import argparse
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

from .. import paths

DEFAULT_BIN = paths.LLAMA_BIN / "llama-kld-swap"
TAG = "@@kldswap "


class KldSwapError(RuntimeError):
    pass


class KldSwap:
    def __init__(self, args, bin=DEFAULT_BIN, log=None, env=None, echo=True):
        """args: llama-kld-swap command-line arguments; log: file receiving the tool's stderr (tail-able);
        echo: also copy the tool's stderr to ours."""
        full_env = dict(os.environ)
        full_env.update(env or {})
        self.log_path = Path(log) if log else None
        self._log = open(self.log_path, "a", buffering=1) if self.log_path else None
        self._echo = echo
        self.proc = subprocess.Popen([str(bin), *map(str, args)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, bufsize=1, env=full_env)
        self._err_thread = threading.Thread(target=self._pump_stderr, daemon=True)
        self._err_thread.start()
        self.ready = self._reply("ready")

    def _pump_stderr(self):
        for line in self.proc.stderr:
            if self._log:
                self._log.write(line)
            if self._echo:
                sys.stderr.write(line)
                sys.stderr.flush()

    def _reply(self, cmd):
        while True:
            line = self.proc.stdout.readline()
            if not line:
                rc = self.proc.wait()
                raise KldSwapError(f"llama-kld-swap exited (rc={rc}) while waiting for the reply to '{cmd}'")
            if not line.startswith(TAG):
                if self._log:
                    self._log.write(line)
                continue
            r = json.loads(line[len(TAG):])
            if not r.get("ok"):
                raise KldSwapError(f"{r.get('cmd', cmd)}: {r.get('error')}")
            return r

    def send(self, cmd, **kw):
        """send one command (JSON form, so paths may contain spaces) and return its reply"""
        msg = {"cmd": cmd}
        msg.update({k: (v if isinstance(v, str) else json.dumps(v)) for k, v in kw.items() if v is not None})
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        return self._reply(cmd)

    def base(self, path):
        return self.send("base", file=str(path))

    def patch(self, name, file, type=None, offset=None):
        return self.send("patch", name=name, file=str(file), type=type, offset=None if offset is None else str(offset))

    def patch_gguf(self, name, gguf, src=None):
        return self.send("patch-gguf", name=name, file=str(gguf), src=src)

    def retype(self, name, file, type, offset=None):
        """replace tensor `name` with the raw bytes of `file` as ggml type `type` (same shape; the size follows from the type)"""
        return self.send("retype", name=name, file=str(file), type=type, offset=None if offset is None else str(offset))

    def retype_gguf(self, name, gguf, src=None):
        """replace tensor `name` with tensor `src` (default `name`) of another GGUF, whatever its type"""
        return self.send("retype-gguf", name=name, file=str(gguf), src=src)

    def revert(self, name="all"):
        return self.send("revert", name=name)

    def eval(self, chunks=None, start=0, out=None, tokens=False, quiet=False):
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
        return self.send("eval", chunks=None if chunks is None else str(chunks), start=str(start),
                         out=None if out is None else str(out), tokens="1" if tokens else "0", quiet="1" if quiet else "0")

    def info(self, name):
        return self.send("info", name=name)

    def placements(self):
        return self.send("placements")

    def verify(self, name="all"):
        return self.send("verify", name=name)

    def status(self):
        return self.send("status")

    def set(self, **kw):
        return self.send("set", **{k: str(v) for k, v in kw.items()})

    def close(self):
        if self.proc.poll() is None:
            try:
                self.send("quit")
            except (KldSwapError, BrokenPipeError):
                pass
            self.proc.wait()
        self._err_thread.join(timeout=5)
        if self._log:
            self._log.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


#
# reading eval outputs
#

def load_chunks(prefix):
    rows = np.genfromtxt(f"{prefix}.chunks.tsv", delimiter="\t", names=True, dtype=None, encoding=None)
    return np.atleast_1d(rows)


def load_tokens(prefix):
    """returns dict chunk -> (kld, p_diff, nll, nll_base, same_top), each [T]"""
    raw = Path(f"{prefix}.tokens.bin").read_bytes()
    if raw[:8] != b"KLDTOK01":
        raise ValueError(f"{prefix}.tokens.bin: bad magic")
    T = int(np.frombuffer(raw, np.int32, 1, 8)[0])
    rec = 4 + 4 * 4 * T + T
    body = raw[12:]
    out = {}
    for i in range(len(body) // rec):
        r = body[i * rec:(i + 1) * rec]
        c = int(np.frombuffer(r, np.int32, 1, 0)[0])
        f = np.frombuffer(r, np.float32, 4 * T, 4).reshape(4, T)
        top = np.frombuffer(r, np.uint8, T, 4 + 16 * T)
        out[c] = (f[0], f[1], f[2], f[3], top)
    return out


def compare(a, b, n_boot=10000, seed=0):
    """paired comparison B - A over the chunks both evals cover"""
    A, B = load_chunks(a), load_chunks(b)
    ia = {int(r["chunk"]): r for r in A}
    ib = {int(r["chunk"]): r for r in B}
    common = sorted(set(ia) & set(ib))
    if not common:
        raise ValueError("no common chunks")
    for c in common:
        if int(ia[c]["n_tok"]) != int(ib[c]["n_tok"]):
            raise ValueError(f"chunk {c}: token counts differ")
    n = len(common)
    ntok = np.array([ia[c]["n_tok"] for c in common], float)
    ka = np.array([ia[c]["kld_sum"] for c in common]) / ntok
    kb = np.array([ib[c]["kld_sum"] for c in common]) / ntok
    ta = np.array([ia[c]["n_same_top"] for c in common]) / ntok
    tb = np.array([ib[c]["n_same_top"] for c in common]) / ntok
    la = (np.array([ia[c]["nll_sum"] for c in common]) - np.array([ia[c]["nll_base_sum"] for c in common])) / ntok
    lb = (np.array([ib[c]["nll_sum"] for c in common]) - np.array([ib[c]["nll_base_sum"] for c in common])) / ntok
    w = ntok / ntok.sum()

    def paired(xa, xb):
        d = xb - xa
        mean = float((w * d).sum())
        se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        return {"A": float((w * xa).sum()), "B": float((w * xb).sum()), "diff": mean, "se_chunks": se,
                "rel": mean / float((w * xa).sum()) if (w * xa).sum() != 0 else float("nan"),
                "B_better_chunks": None}

    res = {"chunks": n, "kld": paired(ka, kb), "same_top": paired(ta, tb), "ln_ppl_ratio": paired(la, lb)}
    res["kld"]["B_better_chunks"] = int((kb < ka).sum())
    res["same_top"]["B_better_chunks"] = int((tb > ta).sum())
    res["ln_ppl_ratio"]["B_better_chunks"] = int((lb < la).sum())

    # per-chunk KLD sums; from the per-token values when both evals wrote them, since the tsv sums depend on the
    # order in which the worker threads were merged and can differ in the last bits between identical runs
    sums_a = np.array([ia[c]["kld_sum"] for c in common])
    sums_b = np.array([ib[c]["kld_sum"] for c in common])
    TA = TB = None
    if Path(f"{a}.tokens.bin").exists() and Path(f"{b}.tokens.bin").exists():
        TA, TB = load_tokens(a), load_tokens(b)
        if all(c in TA and c in TB for c in common):
            sums_a = np.array([TA[c][0].astype(np.float64).sum() for c in common])
            sums_b = np.array([TB[c][0].astype(np.float64).sum() for c in common])

    # bootstrap over chunks (chunks are the independent unit; tokens within a chunk are correlated)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = (sums_b[idx].sum(1) - sums_a[idx].sum(1)) / ntok[idx].sum(1)
    res["kld"]["boot_ci95"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    res["kld"]["boot_se"] = float(boot.std(ddof=1))
    res["kld"]["boot_p_B_lower"] = float((boot < 0).mean())
    res["kld"]["boot_p_B_higher"] = float((boot > 0).mean())

    # token-level pairing, when both evals wrote tokens
    if TA is not None:
        cc = [c for c in common if c in TA and c in TB]
        if cc:
            dk = np.concatenate([TB[c][0].astype(np.float64) - TA[c][0] for c in cc])
            dt = np.concatenate([TB[c][4].astype(np.float64) - TA[c][4].astype(np.float64) for c in cc])
            res["tokens"] = {
                "n": int(dk.size),
                "kld_diff": float(dk.mean()),
                "kld_diff_se_naive": float(dk.std(ddof=1) / np.sqrt(dk.size)),
                "kld_frac_B_lower": float((dk < 0).mean()),
                "kld_frac_equal": float((dk == 0).mean()),
                "same_top_flips_A_to_B": {"gained": int((dt > 0).sum()), "lost": int((dt < 0).sum())},
                "max_abs_kld_diff": float(np.abs(dk).max()),
            }
    return res


def format_compare(r):
    k = r["kld"]
    lines = [f"paired B - A over {r['chunks']} chunks"]
    for name in ("kld", "ln_ppl_ratio", "same_top"):
        x = r[name]
        lines.append(f"  {name:13s} A {x['A']:.6f}  B {x['B']:.6f}  B-A {x['diff']:+.6f} ± {x['se_chunks']:.6f} (chunk SE)"
                     f"  rel {100 * x['rel']:+.2f}%  B better in {x['B_better_chunks']}/{r['chunks']} chunks")
    lines.append(f"  kld bootstrap over chunks: SE {k['boot_se']:.6f}  95% CI [{k['boot_ci95'][0]:+.6f}, {k['boot_ci95'][1]:+.6f}]"
                 f"  P(B<A) {k['boot_p_B_lower']:.3f}  P(B>A) {k['boot_p_B_higher']:.3f}")
    if "tokens" in r:
        t = r["tokens"]
        lines.append(f"  tokens: n {t['n']}  mean dKLD {t['kld_diff']:+.6f} ± {t['kld_diff_se_naive']:.6f} (naive, ignores within-chunk correlation)"
                     f"  B lower on {100 * t['kld_frac_B_lower']:.1f}%  equal on {100 * t['kld_frac_equal']:.1f}%"
                     f"  same-top gained {t['same_top_flips_A_to_B']['gained']} lost {t['same_top_flips_A_to_B']['lost']}"
                     f"  max |dKLD| {t['max_abs_kld_diff']:.6f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("run")
    r.add_argument("--bin", default=str(DEFAULT_BIN))
    r.add_argument("--log")
    r.add_argument("--script")
    r.add_argument("args", nargs=argparse.REMAINDER)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--boot", type=int, default=10000)
    c.add_argument("--json", action="store_true")
    a = ap.parse_args()

    if a.mode == "compare":
        res = compare(a.a, a.b, a.boot)
        print(json.dumps(res, indent=1) if a.json else format_compare(res))
        return

    args = a.args[1:] if a.args[:1] == ["--"] else a.args
    with KldSwap(args, bin=a.bin, log=a.log) as ks:
        print(json.dumps(ks.ready), flush=True)
        src = open(a.script) if a.script else sys.stdin
        for line in src:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line in ("quit", "exit"):
                break
            ks.proc.stdin.write(line + "\n")
            ks.proc.stdin.flush()
            try:
                print(json.dumps(ks._reply(line.split()[0])), flush=True)
            except KldSwapError as e:
                print(json.dumps({"ok": False, "error": str(e)}), flush=True)


if __name__ == "__main__":
    main()
