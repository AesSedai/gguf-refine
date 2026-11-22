"""Per (layer, expert, type) error table for the type knapsack.

For a stratified sample of experts per layer (strata = equal-size bins of the routing counts, so rarely and
often used experts are both represented; a dense layer is one expert), every candidate type is quantized with
the stock quantizer + imatrix and, for refinable types, refined; train / held-out Hessian-weighted NMSE, plain
MSE and the signal tr(W H W^T) are recorded per tensor. Records are appended per (expert, type) as they finish
and skipped on restart. counts-L<L>.json next to the output holds the counts and strata for the estimator.

Record: {layer, expert, type, stratum, N_h, n_h, rows*, lam, sweeps, <tensor>: {train_nmse_base, test_nmse_base,
[train_nmse_ref, test_nmse_ref, moves], plain_mse_*, train_sig, test_sig}}
"""
import concurrent.futures as cf
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..quant import codes as iq
from ..quant import ggml as gq
from ..refine import engine as en
from gguf import GGMLQuantizationType as QT  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

REFINED = ("IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ1_S", "IQ1_M")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stratified_sample(counts, n_strata, per_stratum, seed):
    """[(expert, stratum, N_h, n_h)] - experts sorted by count, split into n_strata bins, per_stratum drawn per bin"""
    order = np.argsort(counts, kind="stable")
    strata = np.array_split(order, n_strata)
    rng = np.random.default_rng(seed)
    out = []
    for si, s in enumerate(strata):
        pick = rng.choice(s, min(per_stratum, len(s)), replace=False)
        out += [(int(e), si, len(s), len(pick)) for e in sorted(pick)]
    return out, strata


def errtab_layer(layer, groups, weights, hessians, imatrix, types, out, counts=None, n_strata=8, per_stratum=6,
                 lam=1.0, sweeps=8, batch=8, quant_threads=6, refined_types=REFINED, dev="cuda:0", experts=None):
    """groups: [Group] of one layer (equal n_experts); counts: routing counts per expert (None for dense / unknown:
    every expert is its own stratum when n_experts <= n_strata * per_stratum, else uniform strata by index)."""
    dev = torch.device(dev)
    out = Path(out)
    E_all = groups[0].n_experts
    done = set()
    if out.exists():
        with open(out) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add((r["layer"], r["expert"], r["type"]))
    cnt = np.zeros(E_all, np.int64) if counts is None else np.asarray(counts)
    sample, strata = stratified_sample(cnt, min(n_strata, E_all), per_stratum, 1000 + layer)
    if experts is not None:
        sample = [(int(e), -1, 0, 0) for e in experts]
    with open(out.with_name(f"counts-L{layer}.json"), "w") as fh:
        json.dump({"layer": layer, "counts": cnt.tolist(), "strata": [s.tolist() for s in strata]}, fh)
    todo = [x for x in sample if any((layer, x[0], t) not in done for t in types)]
    log(f"L{layer}: {len(sample)} sampled experts, {len(todo)} to do, types {types}")
    refiners = {t: en.FastRefiner(t, dev) for t in types if t in refined_types}
    fout = open(out, "a")
    t0 = time.time()
    for b0 in range(0, len(todo), batch):
        bt = todo[b0:b0 + batch]
        E = len(bt)
        meta, Hs, Wd, Wn = [], {}, {}, []
        per = [hessians.load(layer, groups, e) for e, *_ in bt]
        for (e, si, Nh, nh), h in zip(bt, per):
            meta.append(dict(layer=layer, expert=e, stratum=si, N_h=Nh, n_h=nh, lam=lam, sweeps=sweeps,
                             **{k: h[k] for k in ("rows", "rows_train", "rows_test") if k in h}))
            Wn.append({name: np.asarray(weights.weights(name, e), np.float32) for g in groups for name in g.tensors})
        for g in groups:
            Htr = torch.stack([h[g.key][0] for h in per]).to(dev)
            Hte = torch.stack([h[g.key][1] for h in per]).to(dev)
            Htr = 0.5 * (Htr + Htr.transpose(1, 2))
            ridge = (lam * Htr.diagonal(dim1=1, dim2=2).double().mean(-1)).float()
            Hs[g.key] = (Htr, Hte, Htr + ridge[:, None, None] * torch.eye(g.K, device=dev)[None])
            Wd[g.key] = torch.stack([torch.from_numpy(np.concatenate([wm[n] for n in g.tensors])) for wm in Wn]).to(dev)
        del per
        recs = {(e, t): dict(meta[i], type=t) for i, (e, *_r) in enumerate(bt) for t in types}

        def stock(t, ei, g, Wn=Wn):
            e = bt[ei][0]
            return [np.asarray(gq.quantize(Wn[ei][n], t, imatrix.row(n, e) if imatrix is not None and n in imatrix else None,
                                           quant_threads), np.uint8) for n in g.tensors]

        for t in types:
            if all((layer, e, t) in done for e, *_r in bt):
                continue
            for g in groups:
                Rm, K, nm = g.rows[0], g.K, len(g.tensors)
                Htr, Hte, Hf = Hs[g.key]
                W = Wd[g.key]
                with cf.ThreadPoolExecutor(E) as ex:
                    raws = list(ex.map(lambda ei: stock(t, ei, g), range(E)))
                moves = None
                if t in refiners:
                    mults, codes = [], []
                    for ei in range(E):
                        mr, cr = [], []
                        for raw in raws[ei]:
                            f0 = iq.unpack(t, raw.reshape(-1, iq.SPEC[t]["nbytes"]))
                            mu, co = en.fields_to_dev(t, f0, Rm, K, dev)
                            mr.append(mu); cr.append(co)
                        mults.append(torch.cat(mr)); codes.append(torch.cat(cr))
                    mult, code = torch.stack(mults), torch.stack(codes)
                    rfn = refiners[t]
                    Q0 = rfn.recon(mult, code.clone())
                    Q1, moves, _ = rfn.run(W, Hf, mult, code, nm, sweeps=sweeps, gate=1 if t in en.GATED else 0)
                    pairs = (("base", Q0), ("ref", Q1))
                else:
                    Q0 = torch.stack([torch.from_numpy(np.concatenate(
                        [dequantize(raw.reshape(Rm, -1), getattr(QT, t)).reshape(Rm, K).astype(np.float32) for raw in raws[ei]]))
                        for ei in range(E)]).to(dev)
                    pairs = (("base", Q0),)
                sig_tr = en.quad(W, Htr, False).reshape(E, nm, Rm).sum(-1)
                sig_te = en.quad(W, Hte, False).reshape(E, nm, Rm).sum(-1)
                vals = {"train_sig": sig_tr, "test_sig": sig_te}
                for tag, Qx in pairs:
                    Rx = W - Qx
                    vals[f"train_nmse_{tag}"] = en.quad(Rx, Htr, False).reshape(E, nm, Rm).sum(-1) / sig_tr
                    vals[f"test_nmse_{tag}"] = en.quad(Rx, Hte, False).reshape(E, nm, Rm).sum(-1) / sig_te
                    vals[f"plain_mse_{tag}"] = (Rx.double() ** 2).reshape(E, nm, -1).mean(-1)
                    del Rx
                vals = {k: v.cpu().numpy() for k, v in vals.items()}
                for ei, (e, *_r) in enumerate(bt):
                    for k, name in enumerate(g.tensors):
                        recs[(e, t)][name] = {key: float(v[ei, k]) for key, v in vals.items()}
                        if moves is not None:
                            recs[(e, t)][name]["moves"] = moves[ei][k]
                del W, Q0, pairs
            for e, *_r in bt:
                if (layer, e, t) not in done:
                    fout.write(json.dumps(recs[(e, t)]) + "\n")
                    done.add((layer, e, t))
            fout.flush()
        del Hs, Wd, Wn
        torch.cuda.empty_cache()
        n = b0 + E
        el = time.time() - t0
        log(f"L{layer}: {n}/{len(todo)} experts  {el / n:.1f} s/expert  ETA layer {(len(todo) - n) * el / n / 60:.1f} min")
    fout.close()
    log(f"L{layer}: done in {(time.time() - t0) / 60:.1f} min")
