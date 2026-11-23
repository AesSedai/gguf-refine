"""CPU-only checks (no GPU needed):  python -m unittest discover -s tests -v

- codecs: for every refinable IQ type, stock ggml codes -> unpack -> pack is the identity; the engine's
  (mult, code) fields reconstruct exactly the gguf-py dequantization; code_to_fields inverts fields_to_dev
- quantizer: the CPU path is deterministic and independent of the thread chunking
- ggufio / fold: per-expert reads, dequantized weights, in-place verified writes and folding on a synthetic GGUF
- imatrix rows, knapsack vs brute force, overlap window detection
"""
import contextlib
import itertools
import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ["GGUF_REFINE_GPUQ"] = "0"                                          # CPU quantizer only

import numpy as np  # noqa: E402

from ggufrefine import corpus, ggufio, fold  # noqa: E402
from ggufrefine.alloc import knapsack  # noqa: E402
from ggufrefine.eval import overlap  # noqa: E402
from ggufrefine.quant import codes as iq, ggml as gq  # noqa: E402
from gguf import GGMLQuantizationType as QT, GGUFWriter  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

TYPES = ("IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ1_S", "IQ1_M")
rng = np.random.default_rng(0)


def mat(R=8, K=512):
    return (rng.standard_normal((R, K)) * 0.02).astype(np.float32), rng.random(K).astype(np.float32) + 0.1


class Codecs(unittest.TestCase):
    def test_roundtrip_and_recon(self):
        import torch
        from ggufrefine.refine import engine as en
        for t in TYPES:
            with self.subTest(t=t):
                W, im = mat()
                raw = gq.quantize_cpu(W, t, im, nthreads=2).reshape(-1, iq.SPEC[t]["nbytes"])
                f = iq.unpack(t, raw)
                self.assertTrue(np.array_equal(iq.pack(t, f).reshape(-1), raw.reshape(-1)), "pack(unpack(x)) != x")
                deq = dequantize(raw.reshape(W.shape[0], -1), getattr(QT, t)).reshape(W.shape).astype(np.float32)
                mult, code = en.fields_to_dev(t, f, *W.shape)
                gs = iq.SPEC[t]["gsize"]
                if t in iq.IQ1:
                    g = iq.grid(t)                                            # m * (grid[idx] + delta)
                    idx, neg = code & 2047, (code >> 11).astype(bool)
                    rec = (g[idx] + np.where(neg, -iq.IQ1_DELTA, iq.IQ1_DELTA)[..., None]) * mult[..., None]
                else:
                    g = iq.grid(t)
                    mag = g[code >> gs]
                    sgn = ((code[..., None] >> np.arange(gs)) & 1).astype(bool)
                    rec = mag * mult[..., None] * np.where(sgn, -1.0, 1.0)
                self.assertTrue(np.array_equal(rec.reshape(W.shape).astype(np.float32), deq), "fields reconstruction != dequant")
                f2 = en.code_to_fields(t, torch.from_numpy(code), f)
                self.assertTrue(np.array_equal(iq.pack(t, f2).reshape(-1), raw.reshape(-1)), "code_to_fields(fields_to_dev) != id")

    def test_quantizer_chunking(self):
        W, im = mat(R=37, K=256)
        a = gq.quantize_cpu(W, "IQ2_XXS", im, nthreads=1)
        b = gq.quantize_cpu(W, "IQ2_XXS", im, nthreads=5)
        self.assertTrue(np.array_equal(a, b))


def write_gguf(path, tensors):
    w = GGUFWriter(str(path), "llama")
    for name, (arr, qt) in tensors.items():
        w.add_tensor(name, arr, raw_dtype=qt)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


class GgufIO(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.E, self.R, self.K = 3, 8, 512
        self.W = [mat(self.R, self.K)[0] for _ in range(self.E)]
        im = np.ones(self.K, np.float32)
        self.raw = np.stack([gq.quantize_cpu(w, "IQ2_XXS", im, 2) for w in self.W])        # [E, R, bytes]
        self.dense = gq.quantize_cpu(self.W[0], "IQ2_XXS", im, 2)
        write_gguf(self.tmp / "m.gguf", {"blk.0.ffn_gate_exps.weight": (self.raw, QT.IQ2_XXS),
                                         "blk.0.attn_q.weight": (self.dense, QT.IQ2_XXS)})

    def test_read(self):
        g = ggufio.Gguf(self.tmp / "m.gguf")
        ti = g.info["blk.0.ffn_gate_exps.weight"]
        self.assertEqual((ti.n_experts, ti.rows, ti.cols), (self.E, self.R, self.K))
        self.assertEqual(g.info["blk.0.attn_q.weight"].n_experts, 1)
        for e in range(self.E):
            self.assertTrue(np.array_equal(g.expert(ti.name, e), self.raw[e]))
            deq = dequantize(self.raw[e], QT.IQ2_XXS).reshape(self.R, self.K)
            self.assertTrue(np.array_equal(g.weights(ti.name, e), deq))
        self.assertTrue(np.array_equal(g.expert("blk.0.attn_q.weight"), self.dense))
        g.close()

    def test_write_expert_and_fold(self):
        p = self.tmp / "m.gguf"
        ti = ggufio.tensor_table(p)["blk.0.ffn_gate_exps.weight"]
        new = np.roll(self.raw, 1, axis=0)                                    # recognisably different bytes
        ggufio.write_expert(p, ti, 1, new[1])
        g = ggufio.Gguf(p)
        self.assertTrue(np.array_equal(g.expert(ti.name, 1), new[1]))
        self.assertTrue(np.array_equal(g.expert(ti.name, 0), self.raw[0]))  # neighbours untouched
        g.close()
        d = self.tmp / "npy"; d.mkdir()
        for e in range(self.E):
            np.save(d / f"{ti.name}-e{e}.npy", new[e])
        self.assertEqual(fold.fold(p, d, self.tmp / "manifest.jsonl"), 1)
        g = ggufio.Gguf(p)
        for e in range(self.E):
            self.assertTrue(np.array_equal(g.expert(ti.name, e), new[e]))
        g.close()
        self.assertEqual(list(d.glob("*.npy")), [])
        self.assertEqual(fold.fold(p, d, self.tmp / "manifest.jsonl"), 0)  # manifest: nothing left to do

    def test_imatrix_row(self):
        s2 = (rng.random((self.E, self.K)) * 10).astype(np.float32)
        cnt = np.array([[4.0, 0.0, 2.0]], np.float32)
        w = GGUFWriter(str(self.tmp / "im.gguf"), "imatrix")
        w.add_tensor("blk.0.ffn_gate_exps.weight.in_sum2", s2)
        w.add_tensor("blk.0.ffn_gate_exps.weight.counts", cnt)
        w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
        im = ggufio.Imatrix(self.tmp / "im.gguf")
        self.assertTrue(np.array_equal(im.row("blk.0.ffn_gate_exps.weight", 0), s2[0] / np.float32(4)))
        self.assertTrue(np.array_equal(im.row("blk.0.ffn_gate_exps.weight", 1), np.ones(self.K, np.float32)))  # count 0 -> ones
        self.assertTrue(np.array_equal(im.row("blk.0.ffn_gate_exps.weight", 2), s2[2] / np.float32(2)))

    def test_write_bounds(self):
        p = self.tmp / "m.gguf"
        ti = ggufio.tensor_table(p)["blk.0.ffn_gate_exps.weight"]
        before = p.read_bytes()
        with self.assertRaises(ValueError):
            ggufio.write_expert(p, ti, self.E, self.raw[0])                   # one past the last expert
        with self.assertRaises(ValueError):
            ggufio.write_expert(p, ti, 0, self.raw[0][:-1])                   # short payload
        self.assertEqual(p.read_bytes(), before)


class PatchSinkCheck(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.E, self.R, self.K = 5, 8, 256
        self.W = np.stack([mat(self.R, self.K)[0] for _ in range(self.E)])
        self.im = rng.random(self.K).astype(np.float32) + 0.1
        self.raw = np.stack([gq.quantize_cpu(w, "IQ2_XXS", self.im, 2) for w in self.W])
        write_gguf(self.tmp / "a.gguf", {"blk.0.ffn_down_exps.weight": (self.raw, QT.IQ2_XXS),
                                         "blk.0.attn_q.weight": (self.raw[0], QT.IQ2_XXS)})
        write_gguf(self.tmp / "ref.gguf", {"blk.0.ffn_down_exps.weight": (self.W, QT.F32),
                                           "blk.0.attn_q.weight": (self.W[0], QT.F32)})
        w = GGUFWriter(str(self.tmp / "im.gguf"), "imatrix")
        w.add_tensor("blk.0.ffn_down_exps.weight.in_sum2", np.tile(self.im, (self.E, 1)))
        w.add_tensor("blk.0.ffn_down_exps.weight.counts", np.ones((1, self.E), np.float32))
        w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()

    def test_patch_apply_revert(self):
        from ggufrefine import patch
        p, name = self.tmp / "a.gguf", "blk.0.ffn_down_exps.weight"
        orig = p.read_bytes()
        new = self.tmp / "new.bin"
        new.write_bytes(np.roll(self.raw, 2, axis=0).tobytes())
        patch.apply(p, self.tmp / "bk", [(name, new)])
        g = ggufio.Gguf(p)
        self.assertTrue(np.array_equal(g.expert(name, 2), self.raw[0]))
        g.close()
        g = ggufio.Gguf(p, overrides=self.tmp / "bk")                         # reads the original bytes via the backup
        self.assertTrue(np.array_equal(g.expert(name, 2), self.raw[2]))
        g.close()
        with self.assertRaises(ValueError):
            patch.apply(p, self.tmp / "bk", [(name, new)])                    # second apply refused
        patch.revert(p, self.tmp / "bk")
        self.assertEqual(p.read_bytes(), orig)

    def test_gguf_sink_batch(self):
        from ggufrefine.refine.driver import GgufSink
        p, name = self.tmp / "a.gguf", "blk.0.ffn_down_exps.weight"
        new = np.roll(self.raw, 1, axis=0)
        GgufSink(p).write_batch([(name, e, new[e]) for e in (4, 0, 1, 3)])    # runs 0-1, 3-4
        g = ggufio.Gguf(p)
        for e in range(self.E):
            self.assertTrue(np.array_equal(g.expert(name, e), new[e] if e != 2 else self.raw[2]), e)
        g.close()

    def test_check_build(self):
        import io
        from contextlib import redirect_stdout
        from ggufrefine.eval import check_build
        b = self.tmp / "b.gguf"
        write_gguf(b, {"blk.0.ffn_down_exps.weight": (np.roll(self.raw, 1, axis=0), QT.IQ2_XXS),
                       "blk.0.attn_q.weight": (self.raw[0], QT.IQ2_XXS)})
        sample = [("blk.0.ffn_down_exps.weight", e) for e in range(self.E)]
        with redirect_stdout(io.StringIO()):
            ok = check_build.check(self.tmp / "a.gguf", b, "_exps.", self.tmp / "ref.gguf", self.tmp / "im.gguf", sample)
        self.assertTrue(ok)
        with redirect_stdout(io.StringIO()):                                  # same file: the changed set is identical
            self.assertFalse(check_build.check(self.tmp / "a.gguf", self.tmp / "a.gguf", "_exps."))


class Resume(unittest.TestCase):
    def test_output_identity_and_torn_tail(self):
        from ggufrefine.refine import driver
        from ggufrefine.refine.problem import Group
        tmp = Path(tempfile.mkdtemp())
        g = Group(3, "ffn_down_exps", ("blk.3.ffn_down_exps.weight",), 4, 512, (8,), 0)
        s = driver.Settings()
        rec = {"layer": 3, "expert": 1, "lam": 1.0, "hessian": "full", "sweeps": 8, "output": None,
               "blk.3.ffn_down_exps.weight": {"type": "IQ2_XXS", "gate": 1, "gate_margin": 0.0}}
        res = tmp / "r.jsonl"
        res.write_text(json.dumps(rec) + "\n" + json.dumps(dict(rec, expert=2))[:-5])   # torn last line
        types = {"ffn_down_exps": "IQ2_XXS"}
        self.assertEqual(driver._done(res, 3, [g], types, s, None), {1})       # records-only run: expert 1
        self.assertEqual(driver._done(res, 3, [g], types, s, "npy:/x"), set())  # a sink run redoes it
        self.assertTrue(res.read_bytes().endswith(b"\n"))                     # tail terminated for the next append
        self.assertEqual(driver._done(res, 3, [g], {"ffn_down_exps": "IQ2_XS"}, s, None), set())

    def test_sink_preflight(self):
        from ggufrefine.refine import driver
        from ggufrefine.refine.problem import Group
        tmp = Path(tempfile.mkdtemp())
        raw = np.stack([gq.quantize_cpu(mat(8, 256)[0], "IQ2_XXS", np.ones(256, np.float32), 1) for _ in range(2)])
        write_gguf(tmp / "t.gguf", {"blk.0.ffn_down_exps.weight": (raw, QT.IQ2_XXS)})
        sink = driver.GgufSink(tmp / "t.gguf")
        ok = Group(0, "d", ("blk.0.ffn_down_exps.weight",), 2, 256, (8,), 0)
        sink.preflight([ok], {"d": "IQ2_XXS"})
        with self.assertRaises(ValueError):
            sink.preflight([ok], {"d": "IQ2_XS"})                             # wrong type in the target
        with self.assertRaises(ValueError):
            sink.preflight([Group(0, "d", ("blk.0.ffn_down_exps.weight",), 3, 256, (8,), 0)], {"d": "IQ2_XXS"})
        with self.assertRaises(ValueError):
            sink.write_batch([("blk.0.ffn_down_exps.weight", 2, raw[0])])     # expert out of range


class Knapsack(unittest.TestCase):
    def test_integer_budget(self):
        with self.assertRaises(ValueError):                                   # 1 byte over: infeasible, not rounded in
            knapsack.solve([[(1_000_000_001, 1.0, "a")]], 1_000_000_000, 1_000_000_000)
        pick, _ = knapsack.solve([[(1_000_000_001, 0.0, "a"), (999_999_999, 1.0, "b")]], 1_000_000_000, 1)
        self.assertEqual(pick, [1])

    def test_matches_brute_force(self):
        for trial in range(30):
            items = [[(int(rng.integers(1, 9)), float(rng.random()), str(k)) for k in range(int(rng.integers(1, 4)))]
                     for _ in range(4)]
            budget = int(rng.integers(4, 30))
            best = min((sum(o[1] for o in c), c) for c in itertools.product(*items) if sum(o[0] for o in c) <= budget) \
                if any(sum(o[0] for o in c) <= budget for c in itertools.product(*items)) else None
            if best is None:
                with self.assertRaises(ValueError):
                    knapsack.solve(items, budget, 1)
                continue
            pick, cost = knapsack.solve(items, budget, 1)
            self.assertAlmostEqual(cost, best[0])
            self.assertLessEqual(sum(items[i][k][0] for i, k in enumerate(pick)), budget)

    def test_price(self):
        self.assertEqual(knapsack.price(256, "IQ2_XXS"), 66)
        self.assertAlmostEqual(knapsack.bpw("IQ2_S"), 2.5625)


class Overlap(unittest.TestCase):
    def test_planted_window(self):
        q = rng.integers(0, 50000, (2, 512)).astype(np.int32)
        ref = rng.integers(0, 50000, (3, 700)).astype(np.int32)
        ref[1, 123:123 + 64] = q[1, 128:192]                                  # aligned window 2 of row 1, odd offset
        hits = overlap.window_hits(q, ref, 64)
        self.assertEqual(hits.sum(), 1)
        self.assertTrue(hits[1, 2])
        m = overlap.scored_mask(hits, 64, 512, 256)
        self.assertEqual(m.shape, (2, 255))
        self.assertEqual(m.sum(), 0)                                          # window 2 lies before the scored half


class Corpus(unittest.TestCase):
    """the leak checks of ggufrefine.corpus: window hashes, exposure cuts, grouping, allocation"""

    def test_window_hashes_match_brute_force(self):
        rng = np.random.default_rng(0)
        a = rng.integers(0, 50, 400).tolist()
        h = corpus.window_hashes(a, 8)
        self.assertEqual(len(h), len(a) - 7)
        seen = {}
        for i in range(len(a) - 7):
            w = tuple(a[i:i + 8])
            if w in seen:
                self.assertEqual(h[i], h[seen[w]])
            seen.setdefault(w, i)
        self.assertEqual(len(set(h.tolist())), len(seen))                    # distinct windows, distinct hashes

    def test_cut_removes_exactly_the_shared_run(self):
        rng = np.random.default_rng(1)
        train = rng.integers(0, 1000, 5000).tolist()
        doc = rng.integers(0, 1000, 600).tolist()
        doc[100:200] = train[2000:2100]
        doc[400:450] = train[3000:3050]                                        # shorter than the window: kept
        ex = corpus.Exposure(64)
        ex.add(train)
        ex.freeze()
        pieces, cut = ex.cut(doc)
        self.assertEqual(cut, 100)
        self.assertEqual([len(p) for p in pieces], [100, 400])
        for p in pieces:
            self.assertFalse(ex.mask(p).any())

    def test_allocation(self):
        a = corpus.allocate({"a": 0.9, "b": 0.08, "c": 0.02}, 10, {"c": 2})
        self.assertEqual(sum(a.values()), 10)
        self.assertGreaterEqual(a["c"], 2)
        with self.assertRaises(SystemExit):                                   # minima larger than the split
            corpus.allocate({"a": 0.5, "b": 0.5}, 4, {"a": 3, "b": 3})

    def test_rows_are_strict(self):
        with self.assertRaises(corpus.BadRow):                               # a field the template uses is missing
            corpus.fmt("{problem}", {"seed_topic": "x"})
        self.assertEqual(corpus.fmt("{a}{b}", {"a": "x", "b": None}), "x")  # present but null renders empty
        conv = {"c": [{"from": "human", "value": "q"}, {"from": "observation", "value": "r"}, {"from": "gpt", "value": "a"}]}
        _, _, item = corpus._row_item("tool_chat", {"messages": "c"}, "r0", conv)
        self.assertEqual([m["role"] for m in item[1]], ["user", "tool", "assistant"])
        with self.assertRaises(corpus.BadRow):                               # a tool message in a chat domain
            corpus._row_item("chat", {"messages": "c"}, "r0", conv)
        _, _, item = corpus._row_item("chat", {"messages": "c", "roles": {"observation": "user"}}, "r0", conv)
        self.assertEqual([m["role"] for m in item[1]], ["user", "assistant"])   # remapped, then merged with the user turn
        with self.assertRaises(corpus.BadRow):
            corpus._row_item("chat", {"messages": "c"}, "r0", {"c": [{"from": "speaker", "value": "x"}]})
        call = {"c": [{"role": "user", "content": "q"},
                      {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "f", "arguments": "{\"x\": 1}"}}]},
                      {"role": "tool", "content": "r"}, {"role": "assistant", "content": ""}], "t": [{"name": "f"}]}
        _, _, item = corpus._row_item("tool_chat", {"messages": "c", "tools": "t"}, "r0", call)
        self.assertEqual([m["role"] for m in item[1]], ["user", "assistant", "tool"])   # the empty last turn dropped
        self.assertEqual(item[1][1]["tool_calls"][0]["function"]["arguments"], {"x": 1})
        self.assertEqual(item[2], [{"name": "f"}])
        with self.assertRaises(corpus.BadRow):
            corpus._row_item("chat", {"messages": "c"}, "r0", call)
        with self.assertRaises(corpus.BadRow):
            corpus._tool_args({"tool_calls": [{"function": {"name": "f", "arguments": "{not json"}}]})
        _, _, item = corpus._row_item("chat", {"turns": [{"role": "user", "content": "{q}"},
                                                         {"role": "assistant", "content": "<think>r</think>a"}],
                                               "split_think": True}, "r0", {"q": "x"})
        self.assertEqual(item[1][1], {"role": "assistant", "content": "a", "reasoning_content": "r"})

    def test_check_formats(self):
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "a.jsonl")
            open(f, "w").close()
            base = {"n_ctx": 64, "sequences": {"train": 1}}
            for dom, msg in [({"format": "sillytavern"}, "unknown format"),
                             ({"format": "chat"}, "exactly one of"),
                             ({"format": "chat", "messages": "m", "tools": "t"}, "for format tool_chat"),
                             ({"format": "chat", "field": "m", "messages": "m"}, "now \"messages\""),
                             ({"format": "text"}, "needs \"text\"")]:
                with self.assertRaises(SystemExit) as e:
                    corpus.check({**base, "domains": {"a": {"weight": 1, "files": f, **dom}}}, log=lambda *_: None)
                self.assertIn(msg, str(e.exception))
            corpus.check({**base, "domains": {"a": {"weight": 1, "files": f, "format": "tool_chat", "messages": "m",
                                                    "tools": "t"}}}, log=lambda *_: None)

    def test_groups_are_global_and_counted_when_packed(self):
        """the same group key in two domains lands in one split; groups_packed counts groups with packed tokens"""
        r = np.random.default_rng(0)
        MARK = 10000

        def docs_for(name, d, rng, tok):
            for g in range(60):
                yield f"g{g}", f"{name}-{g}", ("ids", [MARK + g] + r.integers(0, 500, 99).tolist())

        class Tok:
            bos, name, chat_kwargs, n_ctx = None, "fake", {}, 100

            def __call__(self, item):
                return list(item[1])

            def encode(self, text):
                return []

            def decode(self, ids):
                return ""
        spec = {"n_ctx": 100, "window": 8, "seed": 1, "sequences": {"train": 8, "test": 2, "eval": 4},
                "domains": {"a": {"weight": 0.9, "format": "text"},
                            "b": {"weight": 0.1, "format": "text", "splits": ["eval"], "min_heldout": 2}}}
        orig = corpus.documents
        corpus.documents = docs_for
        try:
            with tempfile.TemporaryDirectory() as td:
                m = corpus.build(spec, Tok(), Path(td), log=lambda *_: None)
                arr = {s: np.load(Path(td) / f"{s}.tokens.npy") for s in corpus.SPLITS}
        finally:
            corpus.documents = orig
        gs = {s: set((a[a >= MARK] - MARK).tolist()) for s, a in arr.items()}
        self.assertFalse(gs["train"] & gs["eval"])
        self.assertFalse(gs["train"] & gs["test"])
        self.assertFalse(gs["test"] & gs["eval"])
        self.assertGreater(m["domains"]["b"]["docs_of_other_split_groups"], 0)   # b's docs of a's train / test groups
        self.assertEqual(m["domains"]["a"]["train"]["groups_packed"], 8)         # 8 x 100 tokens = 8 docs of 100

    def test_bos_from_gguf(self):
        with tempfile.TemporaryDirectory() as td:
            def gg(**kv):
                p = Path(td) / f"t{'-'.join(f'{k}{v}' for k, v in sorted(kv.items()))}.gguf"
                w = GGUFWriter(str(p), "llama")
                if "add" in kv:
                    w.add_bool("tokenizer.ggml.add_bos_token", kv["add"])
                if "bid" in kv:
                    w.add_uint32("tokenizer.ggml.bos_token_id", kv["bid"])
                w.write_header_to_file()
                w.write_kv_data_to_file()
                w.write_tensors_to_file()
                w.close()
                return p
            self.assertEqual(corpus.gguf_bos(gg(add=True, bid=7)), 7)
            self.assertIsNone(corpus.gguf_bos(gg(add=False)))
            with self.assertRaises(SystemExit):                              # no key: the spec must say
                corpus.gguf_bos(gg())
            self.assertIsNone(corpus.gguf_bos(gg(), spec_bos=False))
            with self.assertRaises(SystemExit):                              # spec contradicts the GGUF
                corpus.gguf_bos(gg(add=False), spec_bos=True)


class Report(unittest.TestCase):
    """ggufrefine.eval.report: the null check and the subset arguments fail loudly"""

    def _run(self, td, name, kld):
        d = Path(td) / name
        d.mkdir()
        with open(d / "eval.chunks.tsv", "w") as fh:
            fh.write("chunk\tn_tok\tkld_sum\tnll_sum\tnll_base_sum\tn_same_top\n")
            for i, k in enumerate(kld):
                fh.write(f"{i}\t100\t{k * 100}\t200\t190\t90\n")
        return f"{name}={d}"

    def test_null_and_subsets(self):
        from ggufrefine.eval import report
        with tempfile.TemporaryDirectory() as td:
            man = Path(td) / "manifest.json"
            man.write_text(json.dumps({"splits": {"eval": {"domain_of_sequence": ["x_rp", "y", "y"]}}}))
            runs = [self._run(td, "stock", [0.2, 0.1, 0.1]), self._run(td, "ok", [0.0, 0.0, 0.0])]
            bad = self._run(td, "bad", [0.0, 0.01, 0.0])
            q = ["--base", "eval", "--manifest", str(man)]
            with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn), contextlib.redirect_stderr(dn):
                self.assertEqual(report.main(q + ["--null", "ok", "--pool", "rp=*rp", *runs]), 0)
                self.assertEqual(report.main(q + ["--null", "bad", runs[0], bad]), 3)
                with self.assertRaises(SystemExit):
                    report.main(q + ["--without", "z", *runs])
                with self.assertRaises(SystemExit):
                    report.main(q + ["--pool", "rp=*zz", *runs])
                man.write_text(json.dumps({"splits": {"eval": {"domain_of_sequence": ["x_rp", "y", "y", "y"]}}}))
                with self.assertRaises(SystemExit):                          # 3 chunks scored, manifest has 4
                    report.main(q + runs)


if __name__ == "__main__":
    unittest.main()
