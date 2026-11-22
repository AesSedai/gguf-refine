"""Calibration corpus with leak-free held-out splits, from a JSON spec of document sources.

Splits: `train` (the Hessians), `test` (held-out Grams: lambda selection, held-out error) and `eval` (the KLD base).
Leakage is prevented at three levels:
  1. Documents are assigned to splits by GROUP before packing (a group is whatever must not straddle the split: a
     role-play character, a shared system prompt / character card, by default the document itself), and each split
     is packed separately, so no document crosses a split.
  2. Every held-out token that lies in a `window`-token run (default 64) also present anywhere in train, or in an
     `exclude` text (e.g. the imatrix calibration file), is cut out of its document; the rest of the document stays.
  3. The manifest records per split and domain the groups, documents, tokens and cut fraction, and the token sha256.
Packing: documents of one domain are concatenated (truncated to `max_doc` tokens) and cut into n_ctx sequences; a
sequence never mixes domains, so per-chunk KLD can be reported by domain. With the tokenizer's add_bos, the first
token of every sequence is BOS (as llama-imatrix / llama-perplexity do per chunk).

Spec (paths may use $VARS):
  {"n_ctx": 8192, "max_doc": 3000, "seed": 1, "window": 64,
   "sequences": {"train": 305, "test": 24, "eval": 32},
   "exclude": ["$KLD/calibration_datav5.txt"],
   "chat_kwargs": {},
   "domains": {
     "<name>": {"weight": 0.12, "format": "<format>", "files": "<glob>", ...format fields...,
                "group": "<format string over the row>", "splits": ["train", "test", "eval"],
                "min_heldout": 2, "rows_per_file": 400, "max_doc": 3000, "window": "head" | "middle",
                "chat_kwargs": {"enable_thinking": true}, "replace": {"-": "<content for a turn that is exactly '-'>"}}}}
Formats (fields are str.format templates over a row; "{a[0][b]}" indexes nested values):
  text       "text": "{text}"
  chat       "turns": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "...",
             "reasoning_content": "..."}], "split_think": true (assistant "<think>R</think>A" -> reasoning R)
  sharegpt   "field": "conversations" (from/value or role/content), optional "append": [turn templates]
  messages   "field": "trajectory" (role/content dicts, JSON or Python literal), optional "tools": "<field>"
  sillytavern  SillyTavern chat exports (<character>/<chat>.jsonl); the group defaults to the character directory
             (trailing digits dropped: Dave, Dave1 and Dave2 are one group, so a chat's branches never straddle splits)
  text_file  one text file cut into n_ctx-token documents ("splits": ["train"] for an imatrix file reused in train)
Sequences per split are allocated over domains by weight (largest remainder); `min_heldout` raises a domain's
test / eval share. `splits` restricts a domain (e.g. a held-out-only source).

    python -m ggufrefine.corpus --spec corpus.json --tokenizer HF_DIR --out DIR
writes DIR/{train,test,eval}.tokens.npy, DIR/eval.txt (for llama-perplexity, see the runbook), DIR/manifest.json.
"""
import argparse
import ast
import glob
import hashlib
import json
import os
import random
import re
import string
import sys
import time
from pathlib import Path

import numpy as np

SPLITS = ("train", "test", "eval")


# ------------------------------------------------------------------------------------------------ rows
class _Row(dict):
    def __missing__(self, k):
        return ""


def _clean(v):
    if isinstance(v, dict):
        return _Row({k: _clean(x) for k, x in v.items()})
    if isinstance(v, list):
        return [_clean(x) for x in v]
    return "" if v is None else v


def fmt(template, row):
    return string.Formatter().vformat(template, (), _clean(row))


def _files(spec, rng):
    files = sorted(glob.glob(os.path.expandvars(spec["files"]), recursive=True))
    if not files:
        raise SystemExit(f"no files match {spec['files']!r}")
    rng.shuffle(files)
    return files


def rows(spec, rng):
    """(row id, row) in a reproducible random order: files shuffled, rows shuffled within a file (parquet: one random
    row group per file visit), at most rows_per_file per file"""
    import pyarrow.parquet as pq
    cap = spec.get("rows_per_file", 1 << 30)
    for f in _files(spec, rng):
        if f.endswith(".parquet"):
            pf = pq.ParquetFile(f)
            groups = list(range(pf.num_row_groups))
            rng.shuffle(groups)
            got = 0
            for g in groups:
                rs = pf.read_row_group(g, columns=spec.get("columns")).to_pylist()
                order = list(range(len(rs)))
                rng.shuffle(order)
                for i in order:
                    yield f"{f}:{g}:{i}", rs[i]
                    got += 1
                    if got >= cap:
                        break
                if got >= cap:
                    break
        else:
            with open(f, encoding="utf-8") as fh:
                lines = fh.readlines()
            order = list(range(len(lines)))
            rng.shuffle(order)
            for i in order[:cap]:
                if lines[i].strip():
                    yield f"{f}:{i}", json.loads(lines[i])


_ROLE = {"system": "system", "human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant",
         "model": "assistant", "function_call": "assistant", "observation": "user", "tool": "user"}


def _merge_roles(msgs):
    """consecutive messages of one role joined (templates that require alternation reject them otherwise)"""
    out = []
    for m in msgs:
        if out and out[-1]["role"] == m["role"] and m["role"] != "tool" and "tool_calls" not in m and "tool_calls" not in out[-1]:
            out[-1] = dict(out[-1], content=f"{out[-1]['content']}\n\n{m['content']}")
        else:
            out.append(dict(m))
    return out


def _split_think(text):
    text = text or ""
    if text.lstrip().startswith("<think>") and "</think>" in text:
        r, a = text.split("</think>", 1)
        return r.replace("<think>", "", 1).strip(), a.strip()
    return "", text


def _turns(templates, row, split_think=False, replace=None):
    out = []
    for t in templates:
        m = {k: fmt(v, row) for k, v in t.items()}
        if replace and m.get("content", "").strip() in replace:
            m["content"] = replace[m["content"].strip()]
        if split_think and m["role"] == "assistant" and not m.get("reasoning_content"):
            r, a = _split_think(m["content"])
            m["content"] = a
            if r:
                m["reasoning_content"] = r
        if not m.get("reasoning_content"):
            m.pop("reasoning_content", None)
        out.append(m)
    return out


def _literal(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return ast.literal_eval(v)
    return v


def _tool_args(m):
    """tool-call arguments stored as a JSON string -> mapping (chat templates iterate them as one)"""
    calls = m.get("tool_calls")
    if not calls:
        return m
    out = []
    for c in calls:
        f = c.get("function", c)
        if isinstance(f.get("arguments"), str):
            try:
                f = dict(f, arguments=json.loads(f["arguments"]))
            except ValueError:
                pass
        out.append(dict(c, function=f) if "function" in c else f)
    return dict(m, tool_calls=out)


def _st_character(name):
    return re.sub(r"[\s\d]+$", "", (name or "").strip()).lower()


def documents(name, spec, rng, tok):
    """(group, doc id, item) where item is ("ids", list), ("text", str) or ("chat", messages, tools, chat kwargs)"""
    for group, did, item in _documents(name, spec, rng, tok):
        if item[0] == "chat":
            item = (*item, spec.get("chat_kwargs"))
        yield group, did, item


def _documents(name, spec, rng, tok):
    f = spec["format"]
    if f == "text_file":
        path = os.path.expandvars(spec["files"])
        ids = tok.encode(open(path, encoding="utf-8").read())
        n = len(ids) // tok.n_ctx
        order = list(range(n))
        rng.shuffle(order)
        for i in order:
            yield f"{path}#{i}", f"{path}#{i}", ("ids", ids[i * tok.n_ctx:(i + 1) * tok.n_ctx])
        return
    if f == "sillytavern":
        for path in _files(spec, rng):
            with open(path, encoding="utf-8") as fh:
                lines = [json.loads(ln) for ln in fh if ln.strip()]
            if not lines:
                continue
            head = lines[0]
            msgs = [{"role": "user" if m.get("is_user") else "assistant", "content": m["mes"]}
                    for m in lines[1:] if not m.get("is_system") and m.get("mes")]
            if len(msgs) >= 2:
                group = fmt(spec["group"], head) if "group" in spec else _st_character(Path(path).parent.name)
                yield group, path, ("chat", _merge_roles(msgs), None)
        return
    for rid, row in rows(spec, rng):
        group = fmt(spec["group"], row) if "group" in spec else rid
        if f == "text":
            item = ("text", fmt(spec["text"], row))
        elif f == "chat":
            item = ("chat", _merge_roles(_turns(spec["turns"], row, spec.get("split_think", False), spec.get("replace"))), None)
        elif f == "sharegpt":
            conv = row[spec["field"]]
            msgs = [{"role": _ROLE.get(m.get("from") or m.get("role"), "user"),
                     "content": m.get("value") if "value" in m else m.get("content")} for m in conv]
            msgs = [m for m in msgs + _turns(spec.get("append", []), row) if m["content"]]
            item = ("chat", _merge_roles(msgs), None)
        elif f == "messages":
            msgs = [_tool_args({k: v for k, v in m.items() if v is not None}) for m in _literal(row[spec["field"]])]
            tools = _literal(row[spec["tools"]]) if spec.get("tools") else None
            item = ("chat", msgs, tools)
        else:
            raise SystemExit(f"{name}: unknown format {f!r}")
        yield group, rid, item


# ------------------------------------------------------------------------------------------------ tokens
class Tokenizer:
    """HF tokenizer + chat template of the model; bos = the BOS id when llama.cpp adds one (the GGUF's
    tokenizer.ggml.add_bos_token, read from `gguf`; HF configs disagree with it for some models)"""

    def __init__(self, path, n_ctx, chat_kwargs, gguf=None):
        from transformers import AutoTokenizer
        self.t = AutoTokenizer.from_pretrained(path)
        self.n_ctx, self.chat_kwargs = n_ctx, chat_kwargs
        add = getattr(self.t, "add_bos_token", False)
        if gguf is not None:
            from . import paths  # noqa: F401  (gguf-py on sys.path)
            from gguf import GGUFReader
            f = GGUFReader(str(gguf)).fields.get("tokenizer.ggml.add_bos_token")
            add = bool(f.parts[-1][0]) if f is not None else add
        self.bos = self.t.bos_token_id if add else None
        self.name = str(path)

    def encode(self, text):
        return self.t.encode(text, add_special_tokens=False)

    def __call__(self, item):
        kind = item[0]
        if kind == "ids":
            return list(item[1])
        if kind == "text":
            return self.encode(item[1])
        kw = {**self.chat_kwargs, **(item[3] if len(item) > 3 and item[3] else {})}
        out = self.t.apply_chat_template(item[1], tools=item[2], tokenize=True, add_generation_prompt=False, **kw)
        if isinstance(out, dict) or hasattr(out, "input_ids"):
            out = out["input_ids"]
        ids = list(out)
        if self.t.bos_token_id is not None and ids and ids[0] == self.t.bos_token_id:
            ids = ids[1:]                              # BOS is placed per sequence when packing
        return ids

    def decode(self, ids):
        return self.t.decode(ids)


# ------------------------------------------------------------------------------------------------ windows
_P = np.uint64(0x9E3779B97F4A7C15)


def window_hashes(ids, w):
    """uint64 hash of every w-token window of ids (len - w + 1 values), polynomial mod 2^64"""
    a = np.asarray(ids, dtype=np.uint64) + np.uint64(1)
    n = len(a)
    if n < w:
        return np.zeros(0, np.uint64)
    with np.errstate(over="ignore"):
        pw = np.cumprod(np.full(n, _P, dtype=np.uint64))                         # P^1 .. P^n
        S = np.concatenate([[np.uint64(0)], np.cumsum(a * pw, dtype=np.uint64)])  # S[i] = sum_{j<i} a_j P^(j+1)
        inv = pow(int(_P), -1, 1 << 64)
        ipw = np.cumprod(np.full(n - w + 1, np.uint64(inv), dtype=np.uint64))     # P^-1 .. P^-(n-w+1)
        return (S[w:] - S[:-w]) * ipw


class Exposure:
    def __init__(self, w):
        self.w, self.parts = w, []

    def add(self, ids):
        self.parts.append(window_hashes(ids, self.w))

    def freeze(self):
        self.ref = np.unique(np.concatenate(self.parts)) if self.parts else np.zeros(0, np.uint64)
        self.parts = None

    def mask(self, ids):
        """bool per token: inside some w-window that occurs in the reference"""
        h = window_hashes(ids, self.w)
        if len(h) == 0 or len(self.ref) == 0:
            return np.zeros(len(ids), bool)
        hit = self.ref[np.minimum(np.searchsorted(self.ref, h), len(self.ref) - 1)] == h     # ref is sorted
        m = np.zeros(len(ids) + 1, np.int32)
        idx = np.flatnonzero(hit)
        np.add.at(m, idx, 1)
        np.add.at(m, idx + self.w, -1)
        return np.cumsum(m[:-1]) > 0

    def cut(self, ids, min_len=32):
        """the unexposed runs of ids (>= min_len tokens), and the exposed token count"""
        if len(ids) < self.w:
            return [ids], 0
        m = self.mask(ids)
        out, i, n = [], 0, len(ids)
        while i < n:
            if m[i]:
                i += 1
                continue
            j = i
            while j < n and not m[j]:
                j += 1
            if j - i >= min_len:
                out.append(ids[i:j])
            i = j
        return out, int(m.sum())


# ------------------------------------------------------------------------------------------------ build
def allocate(weights, n, minimum=None):
    tot = sum(weights.values())
    raw = {k: v / tot * n for k, v in weights.items()}
    alloc = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - alloc[k], reverse=True)[: n - sum(alloc.values())]:
        alloc[k] += 1
    for k, m in (minimum or {}).items():
        while alloc.get(k, 0) < m:
            donor = max((d for d in alloc if d not in (minimum or {})), key=lambda d: alloc[d])
            alloc[donor] -= 1
            alloc[k] = alloc.get(k, 0) + 1
    return alloc


def truncate(ids, spec, n_ctx):
    if spec.get("window") == "middle" and len(ids) > n_ctx:
        mid = (len(ids) - n_ctx) // 2
        return ids[mid:mid + n_ctx]
    return ids[:spec.get("max_doc", n_ctx)]


def pack(docs, n_seq, n_ctx, bos):
    buf = [t for d in docs for t in d]
    need = n_seq * n_ctx
    n = min(n_seq, len(buf) // n_ctx)
    arr = np.array(buf[:n * n_ctx], dtype=np.int32).reshape(n, n_ctx)
    if bos is not None and n:
        arr[:, 0] = bos
    return arr, len(buf) >= need


def build(spec, tok, out, log=print):
    n_ctx, seed, w = spec.get("n_ctx", 8192), spec.get("seed", 1), spec.get("window", 64)
    doms = spec["domains"]
    for name, d in doms.items():
        d.setdefault("max_doc", spec.get("max_doc", 3000))
    weights = {k: d["weight"] for k, d in doms.items() if d.get("weight", 0) > 0}
    alloc = {}
    for s in SPLITS:
        ws = {k: v for k, v in weights.items() if s in doms[k].get("splits", SPLITS)}
        mins = {k: doms[k]["min_heldout"] for k in ws if s != "train" and doms[k].get("min_heldout")}
        alloc[s] = allocate(ws, spec["sequences"][s], mins) if spec["sequences"].get(s) else {}
    margin = spec.get("heldout_margin", 1.6)          # held-out tokens collected before cutting exposed runs
    collected = {s: {} for s in SPLITS}
    stats = {}
    for name, d in sorted(doms.items(), key=lambda x: -x[1].get("weight", 0)):
        need = {s: alloc[s].get(name, 0) * n_ctx * (margin if s != "train" else 1) for s in SPLITS}
        if not any(need.values()):
            continue
        rng = random.Random(f"{seed}:{name}")
        got = {s: 0 for s in SPLITS}
        docs = {s: [] for s in SPLITS}
        groups, gsplit = {s: set() for s in SPLITS}, {}
        failed, t0 = 0, time.time()
        for group, did, item in documents(name, d, rng, tok):
            s = gsplit.get(group)
            if s is None:                               # a new group goes where tokens are still needed, held-out first
                open_ = [x for x in ("test", "eval", "train") if got[x] < need[x]]
                if not open_:
                    break
                s = gsplit[group] = open_[0]
                groups[s].add(group)
            if got[s] >= need[s]:
                continue                                # a full held-out group's further docs are dropped, never moved
            try:
                ids = tok(item)
            except Exception as e:
                failed += 1
                if failed <= 2:
                    log(f"  {name}: render failed ({type(e).__name__}: {str(e)[:160]})")
                continue
            ids = truncate(ids, d, n_ctx) if item[0] != "ids" else ids
            if len(ids) < 32:
                continue
            docs[s].append((did, ids))
            got[s] += len(ids)
            if all(got[x] >= need[x] for x in SPLITS):
                break
        short = [s for s in SPLITS if got[s] < need[s] and (s == "train" or got[s] < need[s] / margin)]
        if short:
            raise SystemExit(f"{name}: source exhausted ({', '.join(f'{s} {got[s]}/{int(need[s])}' for s in short)} tokens)")
        for s in SPLITS:
            collected[s][name] = docs[s]
        stats[name] = {s: {"groups": len(groups[s]), "docs": len(docs[s]), "tokens_collected": got[s]} for s in SPLITS}
        stats[name]["render_failures"] = failed
        log(f"{name:18s} " + "  ".join(f"{s} {len(docs[s]):5d} docs/{len(groups[s]):4d} groups" for s in SPLITS)
            + f"  ({time.time() - t0:.0f} s)")

    # train first; its windows (and the exclude texts') are what held-out text must not contain
    out.mkdir(parents=True, exist_ok=True)
    seqs, owner = {}, {}
    tr = []
    for name, docs in collected["train"].items():
        if not alloc["train"].get(name):
            continue
        arr, ok = pack([ids for _, ids in docs], alloc["train"][name], n_ctx, tok.bos)
        assert ok, name
        tr.append(arr)
        owner.setdefault("train", []).extend([name] * len(arr))
    seqs["train"] = np.concatenate(tr)
    ex = Exposure(w)
    for row in seqs["train"]:
        ex.add(row[1:])                                 # not across the BOS slot
    for p in spec.get("exclude", []):
        ex.add(tok.encode(open(os.path.expandvars(p), encoding="utf-8").read()))
    ex.freeze()
    log(f"exposure reference: {len(ex.ref):,} distinct {w}-token windows (train + {len(spec.get('exclude', []))} exclude files)")
    for s in ("test", "eval"):
        part = []
        for name, docs in collected[s].items():
            if not alloc[s].get(name):
                continue
            kept, cut, tot = [], 0, 0
            for _, ids in docs:
                pieces, c = ex.cut(ids)
                kept.extend(pieces)
                cut += c
                tot += len(ids)
            n = alloc[s][name]
            arr, ok = pack(kept, n, n_ctx, tok.bos)
            if not ok:
                log(f"  {s}/{name}: {len(arr)} of {n} sequences after cutting exposed runs")
            part.append(arr)
            owner.setdefault(s, []).extend([name] * len(arr))
            stats[name][s].update(exposed_tokens=cut, exposed_frac=round(cut / max(tot, 1), 4), sequences=len(arr))
        seqs[s] = np.concatenate(part) if part else np.zeros((0, n_ctx), np.int32)
    for name in stats:
        stats[name]["train"]["sequences"] = alloc["train"].get(name, 0)

    # shuffle sequences within a split (domains interleave across capture parts / KLD chunks), keep the owner map
    manifest = {"n_ctx": n_ctx, "seed": seed, "window": w, "tokenizer": tok.name, "chat_kwargs": tok.chat_kwargs,
                "bos": tok.bos, "spec": spec, "domains": stats, "splits": {}}
    for s in SPLITS:
        perm = np.random.default_rng(seed).permutation(len(seqs[s]))
        arr = seqs[s][perm]
        dom = [owner[s][i] for i in perm]
        np.save(out / f"{s}.tokens.npy", arr)
        manifest["splits"][s] = {"sequences": int(len(arr)), "tokens": int(arr.size), "domain_of_sequence": dom,
                                 "sha256": hashlib.sha256(arr.tobytes()).hexdigest()}
        log(f"{s}: {arr.shape}")
    # the held-out check, recomputed on the packed arrays: no held-out window may occur in train / exclude
    for s in ("test", "eval"):
        hits = sum(int(ex.mask(row[1:].tolist()).sum()) for row in np.load(out / f"{s}.tokens.npy"))
        manifest["splits"][s]["exposed_tokens_after_cut"] = hits
        log(f"{s}: {hits} tokens in a {w}-window shared with train / exclude (packing junctions only)")
    with open(out / "eval.txt", "w", encoding="utf-8") as fh:
        for row in np.load(out / "eval.tokens.npy"):
            fh.write(tok.decode(row[1:].tolist() if tok.bos is not None else row.tolist()))
            fh.write("\n\n")
    with open(out / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=1)
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True, help="HF model directory (tokenizer + chat template)")
    ap.add_argument("--gguf", type=Path, default=None, help="a GGUF of the model: whether llama.cpp adds BOS")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    spec = json.load(open(a.spec))
    tok = Tokenizer(a.tokenizer, spec.get("n_ctx", 8192), spec.get("chat_kwargs", {}), a.gguf)
    build(spec, tok, a.out, log=lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
