"""Batched, resumable refinement of one layer's groups.

refine_layer(groups, types, sources, settings, results, sink) refines every expert of every group, `batch`
experts per GPU launch, with CPU loader threads `prefetch` batches ahead. Results are appended to a jsonl
(one record per expert, per-tensor entries keyed by GGUF tensor name) only after that expert's codes are
written, so a restart skips finished experts and redoes the rest. A record counts as done only for the same
settings and the same output (records carry the sink identity: a records-only run never satisfies a later run
that writes codes). Codes depend only on the inputs and settings, not on batching. Batches are numbered after
removing finished experts, so a resumed NOISY run draws different noise than an uninterrupted one (clean runs give
the same codes either way).
"""
import concurrent.futures as cf
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .. import ggufio
from ..quant import codes as iq
from . import engine as en
from gguf import GGMLQuantizationType as QT  # noqa: E402
from gguf.quants import dequantize  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@dataclass
class Settings:
    lam: float = 1.0
    hessian: str = "full"            # full | diag
    sweeps: int = 8
    seed: int = 0                    # visit-order seed of the refiner (record it in tag when not 0)
    noise: str = ""                  # "SCALE:NOISY:CLEAN" annealed target noise, for the groups in noise_groups
    noise_groups: tuple = ()
    gate: str = "auto"               # auto | on | exact | off
    gate_margin: float = 0.0
    refresh: str = "fp32"            # fp32 | ozaki
    iq1s_flip: bool = True
    shadow_rows: int = 0
    eval_fp64: bool = False
    bsz: int = 0
    batch: int = 16
    loaders: int = 12
    prefetch: int = 2
    verify_codes: bool = False
    tag: dict = field(default_factory=dict)   # extra fields recorded and part of the resume key (e.g. split)


class NpySink:
    """refined codes as <dir>/<tensor>-e<E>.npy (uint8 [R, bytes_per_row])"""

    def __init__(self, out):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.id = f"npy:{self.out.resolve()}"

    def write(self, name, e, arr):
        path = self.out / f"{name}-e{e}.npy"
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            np.save(fh, arr)
        os.replace(tmp, path)


class GgufSink:
    """refined codes written straight into a GGUF of the same layout, verified by read-back. The stock codes
    must then come from the quantizer (RequantizedCodes), not from this file."""

    def __init__(self, path):
        self.path = Path(path)
        self.info = ggufio.tensor_table(path)
        self.id = f"gguf:{self.path.resolve()}"

    def preflight(self, groups, types):
        """every target tensor exists with the requested type and the group's shape (before any write)"""
        for g in groups:
            for name, R in zip(g.tensors, g.rows):
                ti = self.info.get(name)
                if ti is None:
                    raise ValueError(f"{self.path}: no tensor {name}")
                if (ti.type, ti.rows, ti.cols, ti.n_experts) != (types[g.key], R, g.K, g.n_experts):
                    raise ValueError(f"{self.path}: {name} is {ti.type} [{ti.n_experts} x {ti.rows} x {ti.cols}], "
                                     f"group wants {types[g.key]} [{g.n_experts} x {R} x {g.K}]")

    def write(self, name, e, arr):
        ggufio.write_expert(self.path, self.info[name], e, arr)

    def write_batch(self, items):
        """[(name, e, arr)]: consecutive experts of one tensor are coalesced into one verified write (one fsync per
        run instead of per expert)"""
        by = {}
        for name, e, arr in items:
            by.setdefault(name, []).append((e, arr))
        for name, lst in by.items():
            ti = self.info[name]
            lst.sort(key=lambda x: x[0])
            i = 0
            while i < len(lst):
                j = i
                while j + 1 < len(lst) and lst[j + 1][0] == lst[j][0] + 1:
                    j += 1
                if not (0 <= lst[i][0] and lst[j][0] < ti.n_experts):
                    raise ValueError(f"{name}: experts {lst[i][0]}..{lst[j][0]} outside 0..{ti.n_experts - 1}")
                run = np.concatenate([np.ascontiguousarray(a, np.uint8).reshape(-1) for _, a in lst[i:j + 1]])
                if run.nbytes != (j - i + 1) * ti.expert_bytes:
                    raise ValueError(f"{name}: {run.nbytes} bytes for {j - i + 1} experts of {ti.expert_bytes}")
                ggufio.write_verified(self.path, ti.offset + lst[i][0] * ti.expert_bytes, run, threads=4)
                i = j + 1


def _gate_level(s, t):
    if s.gate == "auto":
        return int(t in en.GATED)
    return {"on": 1, "exact": 2, "off": 0}[s.gate]


def _records(results):
    """parsed records of a jsonl; a torn final line (no newline, e.g. after a crash) is ignored and the file is
    terminated so the next append starts on a fresh line"""
    p = Path(results)
    if not p.exists() or p.stat().st_size == 0:
        return []
    with open(p, "rb+") as fh:
        fh.seek(-1, os.SEEK_END)
        if fh.read(1) != b"\n":
            fh.write(b"\n")
    out = []
    with open(p) as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _done(results, layer, groups, types, s, output):
    keys = dict(layer=layer, lam=s.lam, hessian=s.hessian, sweeps=s.sweeps, output=output, **s.tag)
    done = set()
    for r in _records(results):
        if any(r.get(k) != v for k, v in keys.items()):
            continue
        ok = True
        for g in groups:
            t = types[g.key]
            gl = _gate_level(s, t)
            nz = s.noise if g.key in s.noise_groups else ""
            for name in g.tensors:
                m = r.get(name)
                if not (m and m.get("type") == t and m.get("gate", False) == gl and (not gl or m.get("gate_margin") == s.gate_margin)
                        and m.get("refresh", "fp32") == s.refresh and m.get("iq1s_flip", False) == (t == "IQ1_S" and s.iq1s_flip)
                        and m.get("noise", "") == nz):
                    ok = False
        if ok:
            done.add(r["expert"])
    return done


def refine_layer(layer, groups, types, weights, hessians, stock, settings, results, sink=None, experts=None, dev="cuda:0"):
    """groups: [Group] of one layer, all with the same n_experts; types: {group.key: type name}.
    weights / hessians / stock: see problem.py. sink: NpySink / GgufSink / None (records only)."""
    s = settings
    dev = torch.device(dev)
    E_all = groups[0].n_experts
    names = [n for g in groups for n in g.tensors]
    if len(names) != len(set(names)):
        raise ValueError(f"a tensor appears in more than one group: {names}")
    for g in groups:
        if g.n_experts != E_all or g.layer != layer:
            raise ValueError(f"group {g.key}: layer {g.layer} / {g.n_experts} experts, driver call is layer {layer} / {E_all}")
        if types[g.key] not in ("IQ2_S", "IQ3_S") + en.GATED:
            raise ValueError(f"group {g.key}: type {types[g.key]} is not refinable")
        if len(g.tensors) != len(g.rows) or len(set(g.rows)) != 1 or g.K % 256:
            raise ValueError(f"group {g.key}: stacked tensors need one row count each, all equal, and K % 256 == 0")
    experts = list(range(E_all)) if experts is None else list(experts)
    if any(not 0 <= e < E_all for e in experts):
        raise ValueError(f"experts outside 0..{E_all - 1}")
    output = getattr(sink, "id", None)
    if hasattr(sink, "preflight"):
        sink.preflight(groups, types)
    done = _done(results, layer, groups, types, s, output)
    todo = [e for e in experts if e not in done]
    log(f"L{layer}: {len(experts)} experts requested, {len(experts) - len(todo)} already in {results}, {len(todo)} to do; "
        f"groups {[g.key for g in groups]}, types {types}, lam {s.lam}, hessian {s.hessian}")
    if not todo:
        return 0
    refiners = {t: en.FastRefiner(t, dev, {"BSZ": s.bsz} if s.bsz else {}) for t in set(types.values())}

    def load(e):
        t0 = time.time()
        out = {"e": e, **hessians.load(layer, groups, e), "mats": {}}
        for g in groups:
            t = types[g.key]
            for name, R in zip(g.tensors, g.rows):
                W = weights.weights(name, e)
                if tuple(W.shape) != (R, g.K):
                    raise ValueError(f"{name} e{e}: weights {tuple(W.shape)}, group wants {(R, g.K)}")
                raw0 = stock.codes(name, t, e)
                if np.asarray(raw0).nbytes != R * g.K // 256 * iq.SPEC[t]["nbytes"]:
                    raise ValueError(f"{name} e{e}: {np.asarray(raw0).nbytes} stock bytes, {t} [{R} x {g.K}] needs "
                                     f"{R * g.K // 256 * iq.SPEC[t]['nbytes']}")
                f0 = iq.unpack(t, np.asarray(raw0, np.uint8).reshape(-1, iq.SPEC[t]["nbytes"]))
                mu, co = en.fields_to_dev(t, f0, R, g.K)
                out["mats"][name] = dict(W=W, f0=f0, mult=mu, code=co)
        out["load_sec"] = time.time() - t0
        return out

    pool = cf.ThreadPoolExecutor(s.loaders)
    writer = cf.ThreadPoolExecutor(1)
    packer = cf.ThreadPoolExecutor(8)
    res = open(results, "a")
    batches = [todo[i:i + s.batch] for i in range(0, len(todo), s.batch)]
    futq, submitted = {}, set()

    def prefetch(upto):
        for b in range(min(upto, len(batches))):
            if b not in submitted:
                submitted.add(b)
                futq[b] = [pool.submit(load, e) for e in batches[b]]
    prefetch(s.prefetch)
    T = {k: 0.0 for k in ("wait_load", "hessian", "setup", "refine", "eval", "finalize_wait")}
    n_done, t_start, t_last, pending = 0, time.time(), 0.0, []

    def save_one(item):
        name, e, t, c, f0, q1 = item
        arr = iq.pack(t, en.code_to_fields(t, c, f0))
        if q1 is not None:                                                    # verify_codes: the stored bytes ARE the refined Q
            f2 = iq.unpack(t, arr)
            assert np.array_equal(iq.pack(t, f2), arr), f"{name} e{e}: repack round-trip differs"
            assert np.array_equal(f2["d_raw"], f0["d_raw"]) and np.array_equal(f2["scales"], f0["scales"]), f"{name} e{e}: d/scales changed"
            gg = dequantize(arr.reshape(q1.shape[0], -1), getattr(QT, t)).reshape(q1.shape).astype(np.float32)
            assert int((gg != q1).sum()) == 0, f"{name} e{e}: gguf dequant != refined reconstruction"
        return name, e, arr

    def finalize(recs, saves):
        items = list(packer.map(save_one, saves))
        if sink is not None and hasattr(sink, "write_batch"):
            sink.write_batch(items)
        elif sink is not None:
            list(packer.map(lambda it: sink.write(*it), items))
        for rec in recs:                                                      # record only after its codes are written
            res.write(json.dumps(rec) + "\n")
        res.flush()

    for bi, bexp in enumerate(batches):
        tb = time.time()
        data = [f.result() for f in futq.pop(bi)]
        T["wait_load"] += time.time() - tb
        prefetch(bi + 1 + s.prefetch)
        E = len(data)
        torch.cuda.synchronize(dev); t1 = time.time()
        Hs = {}
        for g in groups:
            Htr = torch.stack([d[g.key][0] for d in data]).to(dev)
            Hte = torch.stack([d[g.key][1] for d in data]).to(dev)
            Htr = 0.5 * (Htr + Htr.transpose(1, 2))                         # exact symmetry for the kernels
            dg = Htr.diagonal(dim1=1, dim2=2)
            Hsrc = torch.diag_embed(dg) if s.hessian == "diag" else Htr
            ridge = (s.lam * dg.double().mean(-1)).float()
            Hs[g.key] = (Htr, Hte, Hsrc + ridge[:, None, None] * torch.eye(g.K, device=dev)[None])
        for d in data:
            for g in groups:
                d.pop(g.key, None)
        torch.cuda.synchronize(dev); T["hessian"] += time.time() - t1
        recs = [{"layer": layer, "expert": d["e"], **{k: d[k] for k in ("rows", "rows_train", "rows_test") if k in d},
                 "output": output, "lam": s.lam, "hessian": s.hessian, "sweeps": s.sweeps, "impl": "ggufrefine", **s.tag,
                 **({"codes_verified": True} if s.verify_codes else {})} for d in data]
        saves = []
        for g in groups:
            ts = time.time()
            t = types[g.key]
            Ws, mults, codes = [], [], []
            for d in data:
                wr, mr, cr = [], [], []
                for name in g.tensors:
                    md = d["mats"][name]
                    wr.append(torch.as_tensor(md["W"]).to(dev, non_blocking=True).float())
                    mr.append(torch.from_numpy(md["mult"]).to(dev)); cr.append(torch.from_numpy(md["code"]).to(dev))
                Ws.append(torch.cat(wr)); mults.append(torch.cat(mr)); codes.append(torch.cat(cr))
            W = torch.stack(Ws); mult = torch.stack(mults); code = torch.stack(codes)
            del Ws, mults, codes
            Htr, Hte, Hf = Hs[g.key]
            rf = refiners[t]
            Q0 = rf.recon(mult, code.clone())
            torch.cuda.synchronize(dev); T["setup"] += time.time() - ts; tr0 = time.time()
            sh = None
            if s.shadow_rows and (rf.parity or rf.iq1):
                srows = torch.from_numpy(np.sort(np.random.default_rng(1000 + bi).choice(W.shape[1], s.shadow_rows, replace=False))).to(dev)
                sh = en.ShadowCheck(W, Hf, srows, 8 // rf.gs)
            nz = s.noise if g.key in s.noise_groups else ""
            sweeps_g, sched = s.sweeps, None
            if nz:
                s0, Tn, Cn = float(nz.split(":")[0]), int(nz.split(":")[1]), int(nz.split(":")[2])
                sweeps_g = Tn + Cn
                res_rms = ((W - Q0) ** 2).mean(dim=(1, 2)).sqrt()
                gen = torch.Generator(device=dev).manual_seed(1_000_003 * layer + 7919 * bi + g.index)

                def sched(sw, T_=Tn, s0=s0, res_rms=res_rms, gen=gen, W=W):
                    if sw >= T_:
                        return None, None, False
                    sig = s0 * res_rms * (1 - sw / T_) ** 2
                    return W + sig[:, None, None] * torch.randn(W.shape, device=W.device, generator=gen), None, True
            flip = t == "IQ1_S" and s.iq1s_flip
            gl = _gate_level(s, t)
            Q1, moves, obj_log = rf.run(W, Hf, mult, code, len(g.tensors), sweeps=sweeps_g, seed=s.seed, gate=gl, margin=s.gate_margin,
                                        shadow=sh, refresh=s.refresh, **({"flip": True} if flip else {}),
                                        **({"schedule": sched} if sched else {}))
            if sh is not None:
                for r_ in recs:
                    r_[f"shadow_{g.key}"] = {k: v for k, v in sh.st.items() if isinstance(v, (int, float))}
                del sh
            torch.cuda.synchronize(dev); T["refine"] += time.time() - tr0; te = time.time()
            nm = len(g.tensors)
            Rm = g.rows[0]
            assert all(r == Rm for r in g.rows), "stacked tensors of one group must have equal row counts"
            sig_tr = en.quad(W, Htr, s.eval_fp64).reshape(E, nm, Rm).sum(-1)
            sig_te = en.quad(W, Hte, s.eval_fp64).reshape(E, nm, Rm).sum(-1)
            vals = {"train_sig": sig_tr, "test_sig": sig_te}
            for tag, Qx in (("base", Q0), ("ref", Q1)):
                Rx = W - Qx
                vals[f"train_nmse_{tag}"] = en.quad(Rx, Htr, s.eval_fp64).reshape(E, nm, Rm).sum(-1) / sig_tr
                vals[f"test_nmse_{tag}"] = en.quad(Rx, Hte, s.eval_fp64).reshape(E, nm, Rm).sum(-1) / sig_te
                vals[f"plain_mse_{tag}"] = (Rx.double() ** 2).reshape(E, nm, -1).mean(-1)
                del Rx
            vals = {k: v.cpu().numpy() for k, v in vals.items()}
            code_np = code.cpu().numpy() if (sink is not None or s.verify_codes) else None   # records-only: no pack
            q1_np = Q1.cpu().numpy() if s.verify_codes else None
            T["eval"] += time.time() - te
            for ei, d in enumerate(data):
                for k, name in enumerate(g.tensors):
                    m = {"type": t, "moves": moves[ei][k], "obj_log": obj_log[ei][k], **({"noise": nz} if nz else {}),
                         **{key: float(vals[key][ei, k]) for key in vals}}
                    if s.refresh != "fp32":
                        m["refresh"] = s.refresh
                    if flip:
                        m.update(iq1s_flip=True, flip_moves=rf.flip_moves[ei][k])
                    if gl:
                        m.update(gate=gl, gate_margin=s.gate_margin, gate_rejects=int(sum(rf.gate_rejects[ei][k])),
                                 gate_rejects_sweep=rf.gate_rejects[ei][k], gate_rejects_real=int(sum(rf.gate_rejects_real[ei][k])))
                    recs[ei][name] = m
                    if code_np is not None:
                        saves.append((name, d["e"], t, code_np[ei, k * Rm:(k + 1) * Rm], d["mats"][name]["f0"],
                                  q1_np[ei, k * Rm:(k + 1) * Rm] if q1_np is not None else None))
            del W, mult, code, Q0, Q1
        del Hs
        sec = time.time() - tb
        for r in recs:
            r["sec"] = round(sec / E, 2)
        tw = time.time()
        for p in pending:
            p.result()
        T["finalize_wait"] += time.time() - tw
        pending = [writer.submit(finalize, recs, saves)]
        n_done += E
        now = time.time()
        if now - t_last >= 10 or bi + 1 == len(batches):
            rate = n_done / (now - t_start)
            log(f"L{layer} progress {n_done}/{len(todo)} experts  {rate:.3f} exp/s  ETA {(len(todo) - n_done) / rate / 60:.1f} min  "
                + " ".join(f"{k}={v:.1f}s" for k, v in T.items()))
            t_last = now
    for p in pending:
        p.result()
    res.close()
    log(f"L{layer} done {n_done} experts in {time.time() - t_start:.1f}s; time split " + " ".join(f"{k}={v:.1f}s" for k, v in T.items()))
    return n_done
