"""GPU checks of ggufrefine.refine.engine.FastRefiner against the references in reference.py, on synthetic problems.

Per type, E = 2 problems of [256, K] (K = 3072 for IQ3_S / IQ3_XXS, else 3584): W ~ N(0, 0.02^2), H = X^T X from
correlated random inputs, Hf = H + mean(diag H) I, stock codes from the ggml quantizer with a matching importance
vector. The reference refines one problem at a time, the engine both in one batch, 3 sweeps, same visit order.
Required for every type and problem:
  - the engine's start reconstruction equals the codec dequant of the stock codes, and its result equals the dequant
    of its refined codes;
  - the refined codes pack to legal blocks and gguf-py (IQ1: also ggml) dequantizes the packed bytes to exactly that
    result;
  - start objectives agree (relative 1e-5) and the objective decreases;
  - final objective fast / reference within 2e-4 (observed |1 - ratio| <= 4.9e-5: the engine searches in float32
    with a deferred gradient update, so near-ties resolve differently);
  - code agreement with the reference: indices >= 0.99, signs / IQ1 deltas >= 0.995 (observed >= 0.9952 / 0.9984).
Parity and IQ1 types also replay the run with the float64 ShadowCheck for each accept gate (fp32, fp64, off): the
replay reproduces the codes, no gate violations, uphill moves <= 0.1% of moved steps with relative size < 1e-6
(observed <= 2 of >= 33,000, max 1.7e-7), gradient error < 1e-4 (observed 7.4e-6), replay error < 1e-12.
IQ1_S also runs the delta-flip pass against the reference flip pass. The quantizer check requires the default
quantizer (GPU when available) to give the same bytes as single-threaded CPU ggml_quantize_chunk, on full-size
matrices of every refinable type.

The printed per-type lines are deterministic for a given torch / Triton / GPU, so two runs can be diffed (timings
aside). Needs a CUDA GPU and the native build (native/build.sh):
    CUDA_VISIBLE_DEVICES=0 python -m unittest discover -s tests/gpu -p test_engine.py -v
"""
import ctypes
import sys
import time
import unittest
from pathlib import Path

import numpy as np
import torch

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
sys.path.insert(0, str(here.parents[1]))
import reference  # noqa: E402
from ggufrefine.quant import codes as iq  # noqa: E402
from ggufrefine.quant import ggml as gq  # noqa: E402
from ggufrefine.refine import engine  # noqa: E402
from gguf import GGMLQuantizationType as QT  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

R_ROWS, E, SWEEPS = 256, 2, 3
RATIO_TOL, START_TOL = 2e-4, 1e-5
MIN_IDX_AGREE, MIN_SIGN_AGREE = 0.99, 0.995
GATES = (("gate-fp32", 1), ("gate-fp64", 2), ("nogate", 0))


def make_problem(t, R, K, dev, quantize, seed=1):
    rng = np.random.default_rng(seed)
    Ws, Hs, fs = [], [], []
    for _ in range(E):
        W = (rng.standard_normal((R, K)) * 0.02).astype(np.float32)
        X = rng.standard_normal((4 * K // 3, K)).astype(np.float32) * (1 + rng.random(K)).astype(np.float32)
        X[:, : K // 8] += 0.5 * X[:, K // 8: K // 4]                           # correlated columns
        Xt = torch.from_numpy(X).to(dev)
        H = (Xt.T @ Xt).double()
        H = 0.5 * (H + H.T)
        Hf = H + 1.0 * H.diagonal().mean() * torch.eye(K, device=dev, dtype=H.dtype)
        imat = (X ** 2).mean(0)
        raw = np.asarray(quantize(W, getattr(QT, t), imat), np.uint8).reshape(-1, iq.SPEC[t]["nbytes"])
        Ws.append(W); Hs.append(Hf); fs.append(iq.unpack(t, raw))
    return Ws, Hs, fs


def single_thread(W, qtype, imat):
    return gq.quantize_cpu(W, qtype, imat, nthreads=1)


def ggml_dequant(t, raw, shape):
    fn = getattr(gq.cpu_lib(), "dequantize_row_" + t.lower())
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    rr = np.ascontiguousarray(raw, dtype=np.uint8)
    out = np.zeros(int(np.prod(shape)), np.float32)
    fn(rr.ctypes.data_as(ctypes.c_void_p), out.ctypes.data_as(ctypes.c_void_p), out.size)
    return out.reshape(shape)


def batch(t, Ws, Hs, fs, R, K, dev):
    W = torch.stack([torch.from_numpy(w) for w in Ws]).to(dev)
    Hf = torch.stack(Hs).float()
    mc = [engine.fields_to_dev(t, f, R, K, dev) for f in fs]
    mult = torch.stack([m for m, _ in mc])
    code0 = torch.stack([c for _, c in mc])
    return W, Hf, mult, code0


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA GPU")
class Engine(unittest.TestCase):
    dev = torch.device("cuda:0")

    # ---- shared checks
    def check_codes(self, t, code_e, f0, Q1_e, R, K, ggml=False):
        """refined device codes -> fields -> bytes: Q1 == dequant(codes), legal blocks, gguf (and ggml) dequant == Q1"""
        f1 = engine.code_to_fields(t, code_e, f0)
        q = Q1_e.cpu().numpy()
        self.assertTrue(np.array_equal(q.astype(np.float64), iq.dequant(t, f1).reshape(R, K)), f"{t}: Q1 != dequant(codes)")
        raw = iq.pack(t, f1)                                                  # asserts legal parity / IQ1_S delta sharing
        f2 = iq.unpack(t, raw)
        self.assertTrue(np.array_equal(iq.pack(t, f2), raw), f"{t}: packed codes do not round-trip")
        for k in f1:
            self.assertTrue(np.array_equal(np.asarray(f2[k]), np.asarray(f1[k])), f"{t}: field {k} changed by packing")
        gg = dequantize(raw.reshape(R, -1), getattr(QT, t)).reshape(R, K).astype(np.float32)
        ngg = int((gg != q).sum())
        ngm = int((ggml_dequant(t, raw, (R, K)) != q).sum()) if ggml else 0
        self.assertEqual(ngg, 0, f"{t}: gguf dequant of the packed bytes != Q1")
        self.assertEqual(ngm, 0, f"{t}: ggml dequant of the packed bytes != Q1")
        return f1, ngg, ngm

    def check_objective(self, t, olr, olf):
        self.assertLess(abs(olf[0] / olr[0] - 1), START_TOL, f"{t}: start objectives differ")
        self.assertLess(olf[-1], olf[0], f"{t}: engine objective did not decrease")
        self.assertLess(olr[-1], olr[0], f"{t}: reference objective did not decrease")
        self.assertLess(abs(olf[-1] / olr[-1] - 1), RATIO_TOL, f"{t}: final objective fast/ref {olf[-1] / olr[-1]:.7f}")

    def check_agree(self, t, idx, sign):
        self.assertGreaterEqual(idx, MIN_IDX_AGREE, f"{t}: index agreement {idx:.5f}")
        self.assertGreaterEqual(sign, MIN_SIGN_AGREE, f"{t}: sign / delta agreement {sign:.5f}")

    def check_shadow(self, t, st, same):
        self.assertTrue(same, f"{t}: shadow replay did not reproduce the codes")
        self.assertEqual(st["gate_violations"], 0, f"{t}: gate violations")
        self.assertLessEqual(st["uphill"], 1e-3 * st["moved"], f"{t}: {st['uphill']} uphill of {st['moved']} moved")
        self.assertLess(st["uphill_max_rel"], 1e-6, f"{t}: uphill max rel {st['uphill_max_rel']:.2e}")
        self.assertLess(st["gerr_max_rel"], 1e-4, f"{t}: gradient error {st['gerr_max_rel']:.2e}")
        self.assertLess(max(st["replay_rel_err"]), 1e-12, f"{t}: replay error {max(st['replay_rel_err']):.2e}")

    def shadow_run(self, fr, W, Hf, mult, code0, R, gpc, **kw):
        rows = torch.from_numpy(np.sort(np.random.default_rng(7).choice(R, min(R, 64), replace=False))).to(self.dev)
        sh = engine.ShadowCheck(W, Hf, rows, gpc)
        code_s = code0.clone()
        fr.run(W, Hf, mult, code_s, 1, sweeps=SWEEPS, shadow=sh, **kw)
        return sh.st, code_s

    # ---- IQ3_S / IQ2_S (free signs)
    def free_sign(self, t):
        R, K, dev = R_ROWS, (3072 if t == "IQ3_S" else 3584), self.dev
        Ws, Hs, fs = make_problem(t, R, K, dev, single_thread)
        rr = reference.Refiner(t, dev)
        ref_out = []
        t0 = time.time()
        for e in range(E):
            f1, moves = rr.run(Ws[e], fs[e], Hs[e], sweeps=SWEEPS)
            ref_out.append((f1, moves, list(rr.obj_log)))
        torch.cuda.synchronize(); t_ref = time.time() - t0
        fr = engine.FastRefiner(t, dev)
        W, Hf, mult, code = batch(t, Ws, Hs, fs, R, K, dev)
        q0np = np.stack([iq.dequant(t, f).reshape(R, K) for f in fs])
        self.assertTrue(np.array_equal(fr.recon(mult, code).cpu().numpy().astype(np.float64), q0np), "recon != iq.dequant")
        torch.cuda.synchronize(); t0 = time.time()
        Q1, moves, obj_log = fr.run(W, Hf, mult, code, 1, sweeps=SWEEPS)
        torch.cuda.synchronize(); t_fast = time.time() - t0
        for e in range(E):
            f1r, mvr, olr = ref_out[e]
            f1f, _, _ = self.check_codes(t, code[e], fs[e], Q1[e], R, K)
            agree_idx = (f1r["idx"] == f1f["idx"]).mean()
            agree_sg = (f1r["signs"] == f1f["signs"]).mean()
            print(f"{t} e{e}: obj ref {['%.5g' % x for x in olr]}")
            print(f"{t} e{e}: obj fst {['%.5g' % x for x in obj_log[e][0]]}")
            print(f"   moves ref {mvr}\n   moves fst {moves[e][0]}\n   idx agree {agree_idx:.5f} sign agree {agree_sg:.5f} "
                  f"final ratio fast/ref {obj_log[e][0][-1] / olr[-1]:.6f}")
            self.check_objective(t, olr, obj_log[e][0])
            self.check_agree(t, agree_idx, agree_sg)
        print(f"{t}: ref {t_ref:.2f}s for {E} x [{R},{K}], fast {t_fast:.2f}s", flush=True)

    def test_iq3_s(self):
        self.free_sign("IQ3_S")

    def test_iq2_s(self):
        self.free_sign("IQ2_S")

    # ---- IQ2_XXS / IQ2_XS / IQ3_XXS (parity signs)
    def parity(self, t, margin=0.0):
        R, K, dev = R_ROWS, (3072 if t == "IQ3_XXS" else 3584), self.dev
        Ws, Hs, fs = make_problem(t, R, K, dev, single_thread)
        pr = reference.ParityRefiner(t, dev)
        ref_out = []
        t0 = time.time()
        for e in range(E):
            f1, moves = pr.run(Ws[e], fs[e], Hs[e], sweeps=SWEEPS)
            ref_out.append((f1, moves, list(pr.obj_log), pr.gate_rejects))
        torch.cuda.synchronize(); t_ref = time.time() - t0
        fr = engine.FastRefiner(t, dev)
        W, Hf, mult, code0 = batch(t, Ws, Hs, fs, R, K, dev)
        q0np = np.stack([iq.dequant(t, f).reshape(R, K) for f in fs])
        self.assertTrue(np.array_equal(fr.recon(mult, code0).cpu().numpy().astype(np.float64), q0np), "recon != iq.dequant")
        gpc = 8 // iq.SPEC[t]["gsize"]
        out = {}
        for tag, gate in GATES:
            code = code0.clone()
            fr.run(W[:1].contiguous(), Hf[:1].contiguous(), mult[:1].contiguous(), code[:1].clone(), 1, sweeps=1, gate=gate, margin=margin)
            torch.cuda.synchronize(); t0 = time.time()
            Q1, moves, obj_log = fr.run(W, Hf, mult, code, 1, sweeps=SWEEPS, gate=gate, margin=margin)
            torch.cuda.synchronize(); t_fast = time.time() - t0
            rej = [[[f"{a}/{b}" for a, b in zip(x, y)] for x, y in zip(rx, ry)] for rx, ry in zip(fr.gate_rejects, fr.gate_rejects_real)]
            st, code_s = self.shadow_run(fr, W, Hf, mult, code0, R, gpc, gate=gate, margin=margin)
            out[tag] = (Q1, code, moves, obj_log, rej, st, torch.equal(code_s, code), t_fast)
        for tag, _ in GATES:
            Q1, code, moves, obj_log, rej, st, same, t_fast = out[tag]
            print(f"--- {t} fast {tag} (margin {margin}): {t_fast:.2f}s; shadow run reproduces codes: {same}")
            print(f"    shadow: steps {st['steps']} moved {st['moved']} UPHILL {st['uphill']} (max rel {st['uphill_max_rel']:.3e}, "
                  f"sum rel {st['uphill_sum_rel']:.3e}) most-downhill rel {st['downhill_min_rel']:.3e} "
                  f"gerr_max_rel {st['gerr_max_rel']:.3e} replay_rel_err max {max(st['replay_rel_err']):.2e} "
                  f"gate_violations {st['gate_violations']}")
            mn = torch.cat(st["margin_need"]).double()
            print(f"    margin_need over {len(mn)} moved steps: median {mn.median():.2e} p99 {mn.quantile(0.99):.2e} "
                  f"p99.99 {mn.quantile(0.9999):.2e} max {mn.max():.2e}")
            gain = torch.cat(st["gain_rel"]).double()
            mx = float(mn.max())
            print("    moves with gain/scale below 1x/3x/10x max need: " + " ".join(f"{float((gain < f * mx).double().mean()):.2e}" for f in (1, 3, 10)))
            self.check_shadow(f"{t} {tag}", st, same)
            for e in range(E):
                f1r, mvr, olr, rjr = ref_out[e]
                f1f, _, _ = self.check_codes(t, code[e], fs[e], Q1[e], R, K)
                agree_idx = (f1r["idx"] == f1f["idx"]).mean()
                agree_sg = (f1r["signs"] == f1f["signs"]).mean()
                print(f"  e{e}: obj ref {['%.6g' % x for x in olr]}")
                print(f"  e{e}: obj fst {['%.6g' % x for x in obj_log[e][0]]}")
                print(f"     moves ref {mvr} (gate rejects {rjr})\n     moves fst {moves[e][0]} (gate rejects all/real {rej[e][0]})\n"
                      f"     idx agree {agree_idx:.5f} sign agree {agree_sg:.5f} final ratio fast/ref {obj_log[e][0][-1] / olr[-1]:.7f}")
                self.check_objective(f"{t} {tag}", olr, obj_log[e][0])
                self.check_agree(f"{t} {tag}", agree_idx, agree_sg)
        print(f"{t}: ref {t_ref:.2f}s for {E} x [{R},{K}]", flush=True)

    def test_iq2_xxs(self):
        self.parity("IQ2_XXS")

    def test_iq2_xs(self):
        self.parity("IQ2_XS")

    def test_iq3_xxs(self):
        self.parity("IQ3_XXS")

    # ---- IQ1_S / IQ1_M
    def iq1(self, t, margin=0.0):
        R, K, dev = R_ROWS, 3584, self.dev
        Ws, Hs, fs = make_problem(t, R, K, dev, lambda W, q, im: gq.quantize(W, q, im, 8))
        rr = reference.IQ1Refiner(t, dev)
        ref_out = []
        t0 = time.time()
        for e in range(E):
            f1, mv = rr.run(Ws[e], fs[e], Hs[e], sweeps=SWEEPS, debug=True)
            ref_out.append((f1, mv, list(rr.obj_log), dict(rr.debug_stats)))
        torch.cuda.synchronize(); t_ref = time.time() - t0
        fr = engine.FastRefiner(t, dev)
        W, Hf, mult, code0 = batch(t, Ws, Hs, fs, R, K, dev)
        q0np = np.stack([iq.dequant(t, f).reshape(R, K) for f in fs])
        self.assertTrue(np.array_equal(fr.recon(mult, code0).cpu().numpy(), q0np), "recon != iq.dequant")
        for tag, gate in GATES:
            code = code0.clone()
            fr.run(W[:1].contiguous(), Hf[:1].contiguous(), mult[:1].contiguous(), code[:1].clone(), 1, sweeps=1, gate=gate, margin=margin)
            torch.cuda.synchronize(); t0 = time.time()
            Q1, moves, obj_log = fr.run(W, Hf, mult, code, 1, sweeps=SWEEPS, gate=gate, margin=margin)
            torch.cuda.synchronize(); t_fast = time.time() - t0
            rej = [(sum(a), sum(b)) for a, b in zip([x[0] for x in fr.gate_rejects], [x[0] for x in fr.gate_rejects_real])]
            st, code_s = self.shadow_run(fr, W, Hf, mult, code0, R, 1, gate=gate, margin=margin)
            same = torch.equal(code_s, code)
            print(f"--- {t} fast {tag}: {t_fast:.2f}s; shadow run reproduces codes: {same}; shadow: steps {st['steps']} "
                  f"moved {st['moved']} UPHILL {st['uphill']} (max rel {st['uphill_max_rel']:.2e}) gerr_max_rel {st['gerr_max_rel']:.2e} "
                  f"replay_rel_err max {max(st['replay_rel_err']):.1e} gate_violations {st['gate_violations']}")
            self.check_shadow(f"{t} {tag}", st, same)
            for e in range(E):
                f1r, mvr, olr, dbg = ref_out[e]
                f1f, ngg, ngm = self.check_codes(t, code[e], fs[e], Q1[e], R, K, ggml=True)
                ratio = obj_log[e][0][-1] / olr[-1]
                print(f"  e{e}: obj ref {['%.6g' % x for x in olr]}  (debug: max rel increase {dbg['max_rel_increase']:.1e}, "
                      f"recompute err {max(dbg['recompute_rel_err']):.1e})")
                print(f"  e{e}: obj fst {['%.6g' % x for x in obj_log[e][0]]}")
                agree_idx, agree_d = (f1r['idx'] == f1f['idx']).mean(), (f1r['delta'] == f1f['delta']).mean()
                print(f"     moves ref {mvr}\n     moves fst {moves[e][0]} gate rejects all/real {rej[e]}\n"
                      f"     idx agree {agree_idx:.5f} delta agree {agree_d:.5f} "
                      f"final ratio fast/ref {ratio:.7f}  legal True gguf-mismatch {ngg} ggml-mismatch {ngm}")
                self.check_objective(f"{t} {tag}", olr, obj_log[e][0])
                self.check_agree(f"{t} {tag}", agree_idx, agree_d)
        print(f"{t}: ref {t_ref:.2f}s for {E} x [{R},{K}]", flush=True)
        if t == "IQ1_S":
            self.iq1s_flip(fr, rr, Ws, Hs, fs, ref_out, W, Hf, mult, code0, R, K)

    def iq1s_flip(self, fr, rr, Ws, Hs, fs, ref_out, W, Hf, mult, code0, R, K):
        t = "IQ1_S"
        code = code0.clone()
        fr.run(W[:1].contiguous(), Hf[:1].contiguous(), mult[:1].contiguous(), code[:1].clone(), 1, sweeps=1, flip=True)
        torch.cuda.synchronize(); t0 = time.time()
        Q1, moves, obj_log = fr.run(W, Hf, mult, code, 1, sweeps=SWEEPS, flip=True)
        torch.cuda.synchronize(); t_fast = time.time() - t0
        fmv = [x[0] for x in fr.flip_moves]
        rj = [(sum(a[0]), sum(b[0])) for a, b in zip(fr.gate_rejects, fr.gate_rejects_real)]
        st, code_s = self.shadow_run(fr, W, Hf, mult, code0, R, 1, flip=True)
        same = torch.equal(code_s, code)
        print(f"--- IQ1_S fast flip (gate fp32): {t_fast:.2f}s; shadow reproduces codes: {same}; shadow: moved {st['moved']} "
              f"UPHILL {st['uphill']} (flip moves {st.get('flip_moved', 0)}, flip uphill {st.get('flip_uphill', 0)}, max rel {st['uphill_max_rel']:.2e}) "
              f"replay_rel_err max {max(st['replay_rel_err']):.1e}")
        self.check_shadow("IQ1_S flip", st, same)
        for e in range(E):
            f1r, mvr = rr.run(Ws[e], fs[e], Hs[e], sweeps=SWEEPS, flip=True, debug=True)
            f1f, ngg, ngm = self.check_codes(t, code[e], fs[e], Q1[e], R, K, ggml=True)
            print(f"  e{e}: obj ref flip {['%.6g' % x for x in rr.obj_log]}  (vs fixed-delta ref {rr.obj_log[-1] / ref_out[e][2][-1]:.5f})")
            print(f"  e{e}: obj fst flip {['%.6g' % x for x in obj_log[e][0]]}")
            agree_idx, agree_d = (f1r['idx'] == f1f['idx']).mean(), (f1r['delta'] == f1f['delta']).mean()
            print(f"     moves ref {mvr} flip {rr.flip_moves}\n     moves fst {moves[e][0]} flip {fmv[e]} gate rejects all/real {rj[e]}\n"
                  f"     idx agree {agree_idx:.5f} delta agree {agree_d:.5f} "
                  f"final ratio fast/ref {obj_log[e][0][-1] / rr.obj_log[-1]:.7f}  legal True gguf-mismatch {ngg} ggml-mismatch {ngm}")
            self.check_objective("IQ1_S flip", list(rr.obj_log), obj_log[e][0])
            self.check_agree("IQ1_S flip", agree_idx, agree_d)

    def test_iq1_s(self):
        self.iq1("IQ1_S")

    def test_iq1_m(self):
        self.iq1("IQ1_M")

    # ---- quantizer
    def test_quantizer_threads(self):
        """default quantizer (GPU when available, else 16 CPU threads) == single-threaded CPU, full-size matrices"""
        for shapes in ((("IQ2_S", (3072, 3584)), ("IQ3_S", (3584, 3072))),
                       (("IQ2_XXS", (3072, 3584)), ("IQ2_XS", (3072, 3584)), ("IQ3_XXS", (3584, 3072))),
                       (("IQ1_S", (3072, 3584)), ("IQ1_M", (3072, 3584)))):
            rng = np.random.default_rng(2)
            for t, (R, K) in shapes:
                W = (rng.standard_normal((R, K)) * 0.02).astype(np.float32)
                imat = rng.random(K).astype(np.float32) + 0.1
                t0 = time.time(); a = single_thread(W, getattr(QT, t), imat); t1 = time.time()
                b = gq.quantize(W, getattr(QT, t), imat, 16); t2 = time.time()
                same = np.array_equal(a, b)
                print(f"{t}: single {t1 - t0:.2f}s, 16 threads {t2 - t1:.2f}s, identical={same}", flush=True)
                self.assertTrue(same, f"{t}: quantizer bytes differ from single-threaded CPU")


if __name__ == "__main__":
    unittest.main()
