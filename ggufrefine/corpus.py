"""Calibration corpus with leak-free held-out splits, from a JSON spec of document sources.

Splits: `train` (the Hessians), `test` (held-out Grams: the held-out reconstruction error in the refine logs) and
`eval` (the KLD base). Leakage is prevented at three levels:
  1. Documents are assigned to splits by GROUP before packing (a group is whatever must not straddle the split: a
     role-play character, a shared system prompt / character card, by default the document itself), and each split
     is packed separately, so no document crosses a split. Group keys are global: the same key in two domains is one
     group, in one split.
  2. Every held-out token that lies in a `window`-token run (default 64) also present anywhere in train, or in an
     `exclude` text (e.g. the imatrix calibration file), is cut out of its document; the rest of the document stays.
  3. The manifest records per split and domain the groups, documents, tokens and cut fraction, and the token sha256.
Packing: documents of one domain are concatenated (truncated to `max_doc` tokens) and cut into n_ctx sequences; a
sequence never mixes domains, so per-chunk KLD can be reported by domain. Domains are labels for allocation and
reporting; nothing requires particular names. When llama.cpp adds BOS (the GGUF's tokenizer.ggml.add_bos_token, or
the spec's "bos" when the GGUF has no such key) the first token of every sequence is BOS, as llama-imatrix /
llama-perplexity do per chunk.

Strictness: a row that lacks a field its templates use, has an unknown role, malformed tool-call arguments, or fails
to render with the chat template stops the build (domain, row id, reason), unless the domain sets
"skip_bad_rows": true; skipped rows are counted in the manifest. A held-out split that ends up with fewer sequences
than allocated (after cutting exposed runs) stops the build; raise "heldout_margin" (default 1.6: held-out tokens
collected per needed token) or add held-out data.

Spec (paths may use $VARS):
  {"n_ctx": 8192, "max_doc": 3000, "seed": 1, "window": 64, "bos": null,
   "sequences": {"train": 305, "test": 24, "eval": 32},
   "exclude": ["$KLD/calibration_datav5.txt"],
   "chat_kwargs": {},
   "domains": {
     "<name>": {"weight": 0.12, "format": "<format>", "files": "<glob>", ...format fields...,
                "group": "<format string over the row>", "splits": ["train", "test", "eval"],
                "min_heldout": 2, "rows_per_file": 400, "max_doc": 3000, "window": "head" | "middle",
                "chat_kwargs": {"enable_thinking": true}, "replace": {"-": "<content for a turn that is exactly '-'>"},
                "roles": {"observation": "user"}, "skip_bad_rows": false}}}
Formats (fields are str.format templates over a row; "{a[0][b]}" indexes nested values; a null field renders as ""):
  text       "text": "{text}"
  chat       "turns": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "...",
             "reasoning_content": "..."}], "split_think": true (assistant "<think>R</think>A" -> reasoning R)
  sharegpt   "field": "conversations" (from/value or role/content), optional "append": [turn templates]; roles
             system, human/user, gpt/assistant/model, function_call (-> assistant), observation/tool (-> tool);
             "roles" maps others, or remaps (e.g. tool -> user for a template without a tool role)
  messages   "field": "trajectory" (role/content dicts, JSON or Python literal), optional "tools": "<field>"
  sillytavern  SillyTavern chat exports (<character>/<chat>.jsonl); the group defaults to the character directory
             (trailing digits dropped: Dave, Dave1 and Dave2 are one group, so a chat's branches never straddle splits)
  text_file  one text file cut into n_ctx-token documents ("splits": ["train"] for an imatrix file reused in train)
Sequences per split are allocated over domains by weight (largest remainder); `min_heldout` sets a minimum of a
domain's test / eval sequences (taken from the largest other domains). `splits` restricts a domain (e.g. a
held-out-only source).

    python -m ggufrefine.corpus --spec corpus.json --check                      # spec, files and allocation only
    python -m ggufrefine.corpus --spec corpus.json --tokenizer HF_DIR --gguf MODEL.gguf --out DIR [--n-ctx N]
writes DIR/{train,test,eval}.tokens.npy, DIR/eval.txt, DIR/manifest.json (written last; its presence marks a
complete build).
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
class BadRow(ValueError):
    """a source row the spec cannot render as intended"""


class _Row(dict):
    def __missing__(self, k):
        raise BadRow(f"no field {k!r} (fields: {', '.join(sorted(self))})")


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
                    try:
                        row = json.loads(lines[i])
                    except ValueError as e:
                        raise SystemExit(f"{f}:{i + 1}: not a JSON line ({e})") from None
                    yield f"{f}:{i}", row


_ROLE = {"system": "system", "human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant",
         "model": "assistant", "function_call": "assistant", "observation": "tool", "tool": "tool"}


def _role(r, spec):
    roles = {**_ROLE, **spec.get("roles", {})}
    if r not in roles:
        raise BadRow(f"unknown role {r!r}; map it with the domain's \"roles\"")
    return roles[r]


def _roles_ok(msgs, spec):
    """messages / chat turns: every role mapped (the "roles" remap applies) to one a chat template knows"""
    out = []
    for m in msgs:
        r = _role(m.get("role"), spec)
        if r not in ("system", "user", "assistant", "tool"):
            raise BadRow(f"role {m.get('role')!r} maps to {r!r}, not system / user / assistant / tool")
        out.append(dict(m, role=r))
    return out


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
            except ValueError as e:
                raise BadRow(f"tool-call arguments are not JSON ({e})") from None
        if "arguments" in f and not isinstance(f["arguments"], dict):
            raise BadRow(f"tool-call arguments are a {type(f['arguments']).__name__}, not an object")
        out.append(dict(c, function=f) if "function" in c else f)
    return dict(m, tool_calls=out)


def _st_character(name):
    return re.sub(r"[\s\d]+$", "", (name or "").strip()).lower()


def documents(name, spec, rng, tok):
    """(group, doc id, item) where item is ("ids", list), ("text", str), ("chat", messages, tools, chat kwargs) or
    ("bad", reason) for a row that cannot be rendered as specified (group None)"""
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
            try:
                with open(path, encoding="utf-8") as fh:
                    lines = [json.loads(ln) for ln in fh if ln.strip()]
            except ValueError as e:
                raise SystemExit(f"{path}: not a SillyTavern JSONL export ({e})") from None
            if not lines:
                continue
            head = lines[0]
            msgs = [{"role": "user" if m.get("is_user") else "assistant", "content": m["mes"]}
                    for m in lines[1:] if not m.get("is_system") and m.get("mes")]
            if len(msgs) >= 2:
                group = fmt(spec["group"], head) if "group" in spec else _st_character(Path(path).parent.name)
                yield group, path, ("chat", _merge_roles(msgs), None)
        return
    if f not in FORMATS:
        raise SystemExit(f"{name}: unknown format {f!r}")
    for rid, row in rows(spec, rng):
        try:
            yield (*_row_item(f, spec, rid, row),)
        except (BadRow, KeyError, IndexError, TypeError, SyntaxError) as e:
            yield None, rid, ("bad", f"{type(e).__name__}: {e}")


def _row_item(f, spec, rid, row):
    group = fmt(spec["group"], row) if "group" in spec else rid
    if not group:
        raise BadRow(f"empty group key from {spec['group']!r}")
    if f == "text":
        return group, rid, ("text", fmt(spec["text"], row))
    if f == "chat":
        return group, rid, ("chat", _merge_roles(_roles_ok(_turns(spec["turns"], row, spec.get("split_think", False), spec.get("replace")), spec)), None)
    if f == "sharegpt":
        if spec["field"] not in row:
            raise BadRow(f"no field {spec['field']!r}")
        msgs = []
        for m in row[spec["field"]]:
            if "value" not in m and "content" not in m:
                raise BadRow(f"message without value / content: {str(m)[:120]}")
            msgs.append({"role": _role(m.get("from") or m.get("role"), spec),
                         "content": m.get("value") if "value" in m else m.get("content")})
        msgs = [m for m in msgs + _turns(spec.get("append", []), row) if m["content"]]
        return group, rid, ("chat", _merge_roles(msgs), None)
    if spec["field"] not in row:                         # messages
        raise BadRow(f"no field {spec['field']!r}")
    msgs = [_tool_args({k: v for k, v in m.items() if v is not None}) for m in _literal(row[spec["field"]])]
    tools = _literal(row[spec["tools"]]) if spec.get("tools") else None
    return group, rid, ("chat", _roles_ok(msgs, spec), tools)


FORMATS = ("text", "chat", "sharegpt", "messages", "sillytavern", "text_file")


# ------------------------------------------------------------------------------------------------ tokens
def gguf_bos(gguf, spec_bos=None):
    """the BOS id llama.cpp puts at the start of a sequence, or None. From the GGUF's tokenizer.ggml.add_bos_token and
    bos_token_id; spec_bos (true / false) is required when the GGUF has no add_bos_token (llama.cpp then falls back
    to a default per vocab type) and must agree with it otherwise."""
    from . import paths  # noqa: F401  (gguf-py on sys.path)
    from gguf import GGUFReader
    fields = GGUFReader(str(gguf)).fields

    def val(k):
        f = fields.get(k)
        return None if f is None else f.parts[f.data[0]][0].item()
    add = val("tokenizer.ggml.add_bos_token")
    if add is None and spec_bos is None:
        raise SystemExit(f"{gguf} has no tokenizer.ggml.add_bos_token (llama.cpp then uses a per-vocab default): "
                         "set \"bos\": true or false in the spec to say whether llama.cpp adds BOS for this model")
    if add is not None and spec_bos is not None and bool(add) != bool(spec_bos):
        raise SystemExit(f"spec \"bos\": {spec_bos} but {gguf} has add_bos_token={bool(add)}")
    add = bool(add) if add is not None else bool(spec_bos)
    if not add:
        return None
    bid = val("tokenizer.ggml.bos_token_id")
    if bid is None:
        raise SystemExit(f"{gguf}: add_bos_token is set but there is no tokenizer.ggml.bos_token_id")
    return int(bid)


class Tokenizer:
    """HF tokenizer + chat template of the model; bos = the BOS id llama.cpp adds per sequence (gguf_bos)"""

    def __init__(self, path, n_ctx, chat_kwargs, bos):
        from transformers import AutoTokenizer
        self.t = AutoTokenizer.from_pretrained(path)
        self.n_ctx, self.chat_kwargs = n_ctx, chat_kwargs
        if bos is not None and self.t.bos_token_id is not None and self.t.bos_token_id != bos:
            raise SystemExit(f"BOS id: GGUF {bos}, HF tokenizer {self.t.bos_token_id}")
        self.bos = bos
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
    minimum = minimum or {}
    if sum(minimum.values()) > n:
        raise SystemExit(f"min_heldout {minimum} needs {sum(minimum.values())} sequences, the split has {n}")
    for k, m in minimum.items():
        while alloc.get(k, 0) < m:
            donors = [d for d in alloc if d != k and alloc[d] > minimum.get(d, 0)]
            if not donors:
                raise SystemExit(f"min_heldout of {k}: no other domain has sequences left to give ({alloc})")
            donor = max(donors, key=lambda d: alloc[d])
            alloc[donor] -= 1
            alloc[k] = alloc.get(k, 0) + 1
    return alloc


def truncate(ids, spec, n_ctx):
    if spec.get("window") == "middle" and len(ids) > n_ctx:
        mid = (len(ids) - n_ctx) // 2
        return ids[mid:mid + n_ctx]
    return ids[:spec.get("max_doc", n_ctx)]


def pack(docs, n_seq, n_ctx, bos):
    """docs: [(group, ids)] -> ([n, n_ctx] int32 with n <= n_seq, groups with a token in the packed range)"""
    buf = [t for _, d in docs for t in d]
    n = min(n_seq, len(buf) // n_ctx)
    arr = np.array(buf[:n * n_ctx], dtype=np.int32).reshape(n, n_ctx)
    if bos is not None and n:
        arr[:, 0] = bos
    used, pos = set(), 0
    for g, d in docs:
        if pos >= n * n_ctx:
            break
        if len(d):
            used.add(g)
        pos += len(d)
    return arr, used


def _scored(row, bos):
    """the positions of a packed row that hold document tokens (all of them, unless slot 0 is BOS)"""
    return row[1:] if bos is not None else row


def plan(spec):
    """per split {domain: sequences}; raises SystemExit when the allocation or a min_heldout cannot be met"""
    doms = spec["domains"]
    weights = {k: d["weight"] for k, d in doms.items() if d.get("weight", 0) > 0}
    alloc = {}
    for s in SPLITS:
        ws = {k: v for k, v in weights.items() if s in doms[k].get("splits", SPLITS)}
        mins = {k: doms[k]["min_heldout"] for k in ws if s != "train" and doms[k].get("min_heldout")}
        n = spec["sequences"].get(s, 0)
        if not isinstance(n, int) or n < 0:
            raise SystemExit(f"sequences.{s}: {n!r} is not a count")
        alloc[s] = allocate(ws, n, mins) if n else {}
        if sum(alloc[s].values()) != n:
            raise SystemExit(f"{s}: {n} sequences requested but no domain with weight > 0 allows that split")
    if not spec["sequences"].get("train"):
        raise SystemExit("sequences.train must be > 0")
    return alloc


def check(spec, log=print):
    """the spec without a tokenizer: formats, $VARS, files, exclude files, switches, allocation"""
    bad = []
    if spec.get("bos") is not None and not isinstance(spec["bos"], bool):
        bad.append(f"bos: {spec['bos']!r} is not true / false / null")
    for name, d in spec["domains"].items():
        if d.get("weight", 0) <= 0:
            continue
        if d.get("format") not in FORMATS:
            bad.append(f"{name}: unknown format {d.get('format')!r}")
        if "skip_bad_rows" in d and not isinstance(d["skip_bad_rows"], bool):
            bad.append(f"{name}: skip_bad_rows {d['skip_bad_rows']!r} is not true / false")
        pat = os.path.expandvars(d.get("files", ""))
        if "$" in pat:
            bad.append(f"{name}: unset variable in {d.get('files')!r}")
        elif d.get("format") == "text_file" and not os.path.isfile(pat):
            bad.append(f"{name}: text_file needs one file, {pat!r} is not a file")
        elif not [p for p in glob.glob(pat, recursive=True) if os.path.isfile(p)]:
            bad.append(f"{name}: no files match {pat!r}")
        for s in d.get("splits", SPLITS):
            if s not in SPLITS:
                bad.append(f"{name}: unknown split {s!r}")
    for p in spec.get("exclude", []):
        q = os.path.expandvars(p)
        if "$" in q or not os.path.isfile(q):
            bad.append(f"exclude file {p!r} -> {q!r} not found")
    if bad:
        raise SystemExit("corpus spec:\n  " + "\n  ".join(bad))
    alloc = plan(spec)
    for s in SPLITS:
        log(f"{s:5s} {spec['sequences'].get(s, 0):4d} sequences: " + ", ".join(f"{k} {v}" for k, v in alloc[s].items() if v))
    return alloc


def build(spec, tok, out, log=print):
    n_ctx, seed, w = spec["n_ctx"], spec.get("seed", 1), spec.get("window", 64)
    doms = spec["domains"]
    for name, d in doms.items():
        d.setdefault("max_doc", spec.get("max_doc", 3000))
    alloc = plan(spec)
    margin = spec.get("heldout_margin", 1.6)          # held-out tokens collected before cutting exposed runs
    collected = {s: {} for s in SPLITS}
    stats = {}
    gsplit = {}                                        # group -> split, across all domains
    for name, d in sorted(doms.items(), key=lambda x: -x[1].get("weight", 0)):
        need = {s: alloc[s].get(name, 0) * n_ctx * (margin if s != "train" else 1) for s in SPLITS}
        if not any(need.values()):
            continue
        rng = random.Random(f"{seed}:{name}")
        got = {s: 0 for s in SPLITS}
        docs = {s: [] for s in SPLITS}
        groups = {s: set() for s in SPLITS}
        n_bad, n_short, n_other, n_full, t0 = 0, 0, 0, 0, time.time()

        def bad(did, why):
            nonlocal n_bad
            if not d.get("skip_bad_rows"):
                raise SystemExit(f"{name}: row {did}: {why}\n(fix the spec or the data, or set \"skip_bad_rows\": true "
                                 f"on {name} to skip and count such rows)")
            n_bad += 1
            if n_bad <= 3:
                log(f"  {name}: skipped row {did}: {why[:200]}")

        for group, did, item in documents(name, d, rng, tok):
            if item[0] == "bad":
                bad(did, item[1])
                continue
            s = gsplit.get(group)
            if s is None:                               # a new group goes where tokens are still needed, held-out first
                open_ = [x for x in ("test", "eval", "train") if got[x] < need[x]]
                if not open_:
                    break
                s = gsplit[group] = open_[0]
                groups[s].add(group)
            elif group not in groups[s]:
                if need[s] == 0:                        # the group is in a split this domain does not fill
                    n_other += 1
                    continue
                groups[s].add(group)
            if got[s] >= need[s]:
                n_full += 1
                continue                                # a full held-out group's further docs are dropped, never moved
            try:
                ids = tok(item)
            except Exception as e:                      # noqa: BLE001  (template errors are of many types)
                bad(did, f"chat template: {type(e).__name__}: {e}")
                continue
            ids = truncate(ids, d, n_ctx) if item[0] != "ids" else ids
            if len(ids) < 32:
                n_short += 1
                continue
            docs[s].append((group, did, ids))
            got[s] += len(ids)
            if all(got[x] >= need[x] for x in SPLITS):
                break
        short = [s for s in SPLITS if got[s] < need[s] and (s == "train" or got[s] < need[s] / margin)]
        if short:
            raise SystemExit(f"{name}: source exhausted ({', '.join(f'{s} {got[s]}/{int(need[s])}' for s in short)} tokens; "
                             f"{sum(len(g) for g in groups.values())} groups, {n_bad} bad rows, {n_short} docs under 32 "
                             f"tokens, {n_other} docs of groups held by another domain's split, {n_full} docs of groups in a full split)")
        for s in SPLITS:
            collected[s][name] = docs[s]
        stats[name] = {s: {"groups": len(groups[s]), "docs": len(docs[s]), "tokens_collected": got[s]} for s in SPLITS}
        stats[name].update(bad_rows_skipped=n_bad, short_docs=n_short, docs_of_other_split_groups=n_other,
                           docs_of_full_split_groups=n_full)
        log(f"{name:18s} " + "  ".join(f"{s} {len(docs[s]):5d} docs/{len(groups[s]):4d} groups" for s in SPLITS)
            + f"  ({time.time() - t0:.0f} s)" + (f"  {n_bad} bad rows skipped" if n_bad else ""))

    # train first; its windows (and the exclude texts') are what held-out text must not contain
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").unlink(missing_ok=True)
    seqs, owner = {}, {}
    tr = []
    for name, docs in collected["train"].items():
        if not alloc["train"].get(name):
            continue
        arr, used = pack([(g, ids) for g, _, ids in docs], alloc["train"][name], n_ctx, tok.bos)
        if len(arr) < alloc["train"][name]:
            raise SystemExit(f"train/{name}: {len(arr)} of {alloc['train'][name]} sequences")
        stats[name]["train"]["groups_packed"] = len(used)
        tr.append(arr)
        owner.setdefault("train", []).extend([name] * len(arr))
    seqs["train"] = np.concatenate(tr)
    ex = Exposure(w)
    for row in seqs["train"]:
        ex.add(_scored(row, tok.bos))                   # not across the BOS slot
    for p in spec.get("exclude", []):
        ex.add(tok.encode(open(os.path.expandvars(p), encoding="utf-8").read()))
    ex.freeze()
    log(f"exposure reference: {len(ex.ref):,} distinct {w}-token windows (train + {len(spec.get('exclude', []))} exclude files)")
    short = []
    for s in ("test", "eval"):
        part = []
        for name, docs in collected[s].items():
            if not alloc[s].get(name):
                continue
            kept, cut, tot = [], 0, 0
            for g, _, ids in docs:
                pieces, c = ex.cut(ids)
                kept.extend((g, p) for p in pieces)
                cut += c
                tot += len(ids)
            n = alloc[s][name]
            arr, used = pack(kept, n, n_ctx, tok.bos)
            if len(arr) < n:
                short.append(f"{s}/{name}: {len(arr)} of {n} sequences after cutting exposed runs "
                             f"({cut / max(tot, 1):.1%} of {tot} collected tokens cut)")
            part.append(arr)
            owner.setdefault(s, []).extend([name] * len(arr))
            stats[name][s].update(exposed_tokens=cut, exposed_frac=round(cut / max(tot, 1), 4), sequences=len(arr),
                                  groups_packed=len(used))
        seqs[s] = np.concatenate(part) if part else np.zeros((0, n_ctx), np.int32)
    if short:
        raise SystemExit("held-out shortfall:\n  " + "\n  ".join(short) + f"\nraise \"heldout_margin\" (now {margin}) "
                         "or give the domain more held-out data")
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
        hits = sum(int(ex.mask(_scored(row, tok.bos).tolist()).sum()) for row in np.load(out / f"{s}.tokens.npy"))
        manifest["splits"][s]["exposed_tokens_after_cut"] = hits
        log(f"{s}: {hits} tokens in a {w}-window shared with train / exclude (windows across packing junctions)")
    with open(out / "eval.txt", "w", encoding="utf-8") as fh:
        for row in np.load(out / "eval.tokens.npy"):
            fh.write(tok.decode(_scored(row, tok.bos).tolist()))
            fh.write("\n\n")
    with open(out / "manifest.json.tmp", "w") as fh:
        json.dump(manifest, fh, indent=1)
    os.replace(out / "manifest.json.tmp", out / "manifest.json")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--check", action="store_true", help="check the spec, its files and the allocation, then exit")
    ap.add_argument("--tokenizer", type=Path, help="HF model directory (tokenizer + chat template)")
    ap.add_argument("--gguf", type=Path, help="a GGUF of the model: whether llama.cpp adds BOS, and its id")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--n-ctx", type=int, default=None, help="sequence length (overrides the spec's n_ctx)")
    a = ap.parse_args(argv)
    spec = json.load(open(a.spec))
    if a.n_ctx:
        spec["n_ctx"] = a.n_ctx
    if not spec.get("n_ctx"):
        ap.error("the spec has no n_ctx; give --n-ctx")
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)  # noqa: E731
    check(spec, log)
    if a.check:
        return 0
    if not (a.tokenizer and a.gguf and a.out):
        ap.error("--tokenizer, --gguf and --out are required to build")
    tok = Tokenizer(a.tokenizer, spec["n_ctx"], spec.get("chat_kwargs", {}), gguf_bos(a.gguf, spec.get("bos")))
    build(spec, tok, a.out, log=log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
