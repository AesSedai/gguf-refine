#!/usr/bin/env python3
"""Full-depth native-precision Kimi-K3.1 reference, batched over sequences, for a
llama-perplexity --kl-divergence-base file.

Written from vLLM's implementation (vllm/models/kimi_k3/nvidia/{model,kda,dsa}.py), fp32, TF32 off,
batched so every layer's weights are read once for all sequences:
  - all sequences' hidden states stay on the GPU between layers
  - attention-residual mixes are computed source by source (no [N, S, D] stack)
  - KDA runs in sequence sub-batches, DSA per sequence, MoE over all tokens
  - state is checkpointed every --ckpt-every layers; --resume continues from it

Per layer it also records statistics that need the native forward pass:
  stats/moe_counts-<il>.npy      tokens routed to each expert              [E] i64
  stats/moe_gap-<il>.npy         biased-score gap rank16/rank17 per token  [N] f16
  stats/dsa_gap-<il>.npy         indexer score gap at the cutoff per token [N] f16 (nan if no cutoff)
  stats/dsa_mass-<il>.npy        dense attention mass inside the selection [N] f16 (nan with a capture
                                 option on: the dense softmax is skipped)
  stats/zeros-<il>.npz           MXFP4 zero structure of every expert tensor (see zero_stats)

Capture outputs and options (the fp32 default is bit-identical with or without them):
  --expert-dump-layers  routed-expert inputs of each token (ExpertDumpWriter below; reader ..expert_dump)
  --grams            dense-matmul input Grams per layer (ggufrefine.capture.grams), merged by merge_grams
  --imatrix          llama-imatrix sums per layer (joined by write_imatrix.py)
  prefetch           a reader thread streams each layer's dense tensors and expert chunks to the
                     GPU ahead of use (stream.Prefetcher; --no-prefetch reads on the main thread)
  --layer-cache DIR  read shards from the RAM copy kept by layer_cache.py
  async writes       expert-dump and Gram files are written by a background thread
  --fast             TF32 matmuls (not KDA recurrence, not Grams): NOT bit-identical, capture only
  --no-logits        after the last layer: final mix + norm (output.weight Gram) but no lm_head/KLD
  --profile          synchronising per-section timers (slower; for the breakdown only)

Output: part file of KLD-base records (positions [n_ctx/2, n_ctx-1) of each sequence,
in sequence order) plus tokens; merge_kld_base.py joins parts into the final file.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ....capture import grams as gram_io
from ....capture.grams import fp32_exact
from .. import expert_dump
from .. import model as km
from . import stream
rms_norm, situ, mxfp4_dequant, PFX = km.rms_norm, km.situ, km.mxfp4_dequant, km.PFX

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class IMatrix:
    """Sum of squared matmul inputs per GGUF weight, the quantity llama-imatrix stores as
    <name>.in_sum2 with <name>.counts. Dense 2D: [K] and one count per token. Expert stacks:
    [E, K] and per-expert token counts. Per-head 3D (attn_k_b / attn_v_b): [H*K] flattened
    head-major with one count, as llama-imatrix saves 3D dense tensors."""
    def __init__(self):
        self.on, self.acc, self.lin, self.cnt = False, {}, {}, {}   # sum x^2, sum x, counts

    def _put(self, name, s2, s1, c):
        if name in self.acc:
            self.acc[name] += s2; self.lin[name] += s1; self.cnt[name] += c
        else:
            self.acc[name] = s2.clone(); self.lin[name] = s1.clone(); self.cnt[name] = c

    def dense(self, names, x):
        if not self.on: return
        xd = x.double()
        s2, s1 = xd.pow(2).sum(0), xd.sum(0)
        for n in names: self._put(n, s2, s1, x.shape[0])

    def heads(self, name, x):                     # x [T, H, K]
        if not self.on: return
        xd = x.double()
        self._put(name, xd.pow(2).sum(0).reshape(-1), xd.sum(0).reshape(-1), x.shape[0])

    def expert(self, name, e, x, E):
        if not self.on: return
        if name not in self.acc:
            self.acc[name] = torch.zeros(E, x.shape[1], dtype=torch.float64, device=x.device)
            self.lin[name] = torch.zeros(E, x.shape[1], dtype=torch.float64, device=x.device)
            self.cnt[name] = torch.zeros(E, dtype=torch.float64)
        xd = x.double()
        self.acc[name][e] += xd.pow(2).sum(0); self.lin[name][e] += xd.sum(0); self.cnt[name][e] += x.shape[0]

    def flush(self):
        out = {}
        for n, v in self.acc.items():
            out[n + ".in_sum2"] = v.cpu().numpy()
            out[n + ".in_sum"] = self.lin[n].cpu().numpy()
            c = self.cnt[n]
            out[n + ".counts"] = c.numpy() if torch.is_tensor(c) else np.array([c], dtype=np.float64)
        self.acc, self.lin, self.cnt = {}, {}, {}
        return out


IM = IMatrix()


class ExpertDumpWriter:
    """Routed-expert dump (read by ..expert_dump.ExpertDump). Per dumped MoE layer
    and process, one directory layer-<il>/ of raw little-endian column files, one row per token
    (every token is routed at a MoE layer), rows in token order (sequence-major):
      h.f16      [n, 3584] float16  latent input of the routed experts' gate/up (fp16 of the fp32
                                    tensor `lat` = x @ routed_expert_down_proj^T that w1/w3 consume)
      ids.i16    [n, 16]   int16    top-16 expert ids, in topk order of the biased score
      w.f32      [n, 16]   float32  routing weight multiplying each expert's output (after renorm/scale)
      tok.i64    [n]       int64    global token index into the --tokens file: seq * n_ctx + pos
      seq.i32    [n]       int32    sequence index into the --tokens file
      pos.i32    [n]       int32    position in the sequence
      token.i32  [n]       int32    token id
      rms.f32    [n]       float32  sqrt(mean(acc^2)) of the summed routed output acc (latent space,
                                    the input of routed_expert_norm), no eps
      meta.json                     written last (tmp + rename) after all columns are fsynced; a layer
                                    without meta.json is incomplete and is rewritten on the next pass."""
    COLS = expert_dump.COLUMNS

    def __init__(self):
        self.layers, self.dir, self.toks, self.s0, self.tokens_path = set(), None, None, 0, None

    def on(self, il):
        return self.dir is not None and il in self.layers

    def done(self, il):
        m = self.dir / f"layer-{il}" / "meta.json"
        if not m.exists():
            return False
        n = json.load(open(m))["n"]
        return n == self.toks.size

    def write(self, il, lat, sel, w, acc, chunk=16384):
        """Device -> host copies here (same per-chunk ops as before, so the same bytes); the file
        writes run on RT["writer"] when there is one."""
        d = self.dir / f"layer-{il}"
        if self.done(il):
            log(f"expert dump layer {il}: complete dump exists, not rewritten")
            return
        B, T = self.toks.shape
        n = B * T
        assert lat.shape[0] == n and sel.shape == (n, 16) and w.shape == (n, 16), (lat.shape, sel.shape, w.shape)
        assert int(sel.max()) < 2 ** 15
        t0 = time.time()
        parts = []
        for r0 in range(0, n, chunk):
            r1 = min(n, r0 + chunk)
            g = np.arange(r0, r1, dtype=np.int64)
            b, t = g // T, g % T
            cols = {
                "h": lat[r0:r1].half().cpu().numpy(),
                "ids": sel[r0:r1].to(torch.int16).cpu().numpy(),
                "w": w[r0:r1].float().cpu().numpy(),
                "tok": (self.s0 + b) * T + t,
                "seq": (self.s0 + b).astype(np.int32),
                "pos": t.astype(np.int32),
                "token": self.toks.reshape(-1)[r0:r1].astype(np.int32),
                "rms": acc[r0:r1].pow(2).mean(-1).sqrt().float().cpu().numpy(),
            }
            for k, fn, dt in self.COLS:
                assert cols[k].dtype == np.dtype(dt), (k, cols[k].dtype)
            parts.append(cols)
        t_copy = time.time() - t0
        meta = {"format": expert_dump.FORMAT, "version": 1, "layer": il, "n": n, "n_ctx": T,
                "seqs": [self.s0, self.s0 + B], "tokens_file": str(self.tokens_path),
                "k_gate_up": int(lat.shape[1]), "topk": int(sel.shape[1]),
                "columns": {k: {"file": fn, "dtype": dt} for k, fn, dt in self.COLS},
                "shapes": {"h": [n, int(lat.shape[1])], "ids": [n, 16], "w": [n, 16]},
                "rms": "sqrt(mean(acc**2, -1)) of sum_e w_e * expert_e(lat), before routed_expert_norm, no eps",
                "row_order": "token order: seq-major, then position"}

        def job():
            t1 = time.time()
            d.mkdir(parents=True, exist_ok=True)
            (d / "meta.json").unlink(missing_ok=True)
            fs = {k: open(d / fn, "wb") for k, fn, _ in self.COLS}
            for cols in parts:
                for k, fn, dt in self.COLS:
                    fs[k].write(np.ascontiguousarray(cols[k]).tobytes())
                    fs[k].flush()
            for f in fs.values():
                os.fsync(f.fileno()); f.close()
            tmp = d / "meta.json.tmp"
            json.dump(meta, open(tmp, "w"), indent=1)
            os.replace(tmp, d / "meta.json")
            log(f"expert dump layer {il}: {n} tokens written in {time.time() - t1:.1f}s (device copy {t_copy:.1f}s)")

        if RT["writer"] is not None:
            RT["writer"].submit(job)
        else:
            job()


XDW = ExpertDumpWriter()
SKIP_DENSE_MASS = {"on": False}
GR = gram_io.Grams()
RT = {"pf": None, "writer": None, "chunk": 32}     # prefetcher, background writer, experts per prefetch chunk


class Prof:
    """--profile: wall time per section with a device sync on both ends (serialises host and GPU)."""
    def __init__(self):
        self.on, self.t = False, {}

    def sec(self, name):
        return _ProfSec(self, name)

    def take(self):
        t, self.t = self.t, {}
        return t


class _ProfSec:
    def __init__(self, p, name):
        self.p, self.name = p, name

    def __enter__(self):
        if self.p.on:
            torch.cuda.synchronize()
            self.t0 = time.perf_counter()

    def __exit__(self, *exc):
        if self.p.on:
            torch.cuda.synchronize()
            self.p.t[self.name] = self.p.t.get(self.name, 0.0) + time.perf_counter() - self.t0


PROF = Prof()


class StreamWeights:
    """model.Weights, except that the current layer's dense tensors come from the prefetcher's GPU
    buffer (same bytes; .float() on the GPU instead of after the H2D copy)."""
    def __init__(self, base):
        self.base, self.cur = base, None

    def begin_layer(self, il):
        if RT["pf"] is not None:
            self.cur = RT["pf"].get(("dense", il))

    def end_layer(self):
        if self.cur is not None:
            RT["pf"].release(self.cur)
            self.cur = None

    def raw(self, name):
        return self.base.raw(name)

    def f32(self, name):
        if self.cur is not None and self.cur.has(name):
            return self.cur.t(name).float()
        return self.base.f32(name)

    def has(self, name):
        return self.base.has(name)


_DEQ_LUT = {}


def mxfp4_dequant_lut(packed, scale):
    """model.mxfp4_dequant with one gather from a 256-entry byte -> (even, odd) value table kept on the
    device: same values bit for bit (the table holds where(sign, -mag, mag), -0.0 included), no
    host-to-device copy of the E2M1 table (a blocking copy that synchronised the host per call), and
    3 kernels instead of ~10."""
    lut = _DEQ_LUT.get(packed.device)
    if lut is None:
        b = torch.arange(256, dtype=torch.int64)
        codes = torch.stack((b & 0x0F, b >> 4), dim=-1)                       # [256, 2]
        mag = km.E2M1[codes & 0x07]
        lut = _DEQ_LUT[packed.device] = torch.where((codes & 0x08).bool(), -mag, mag).to(packed.device)
    val = lut.index_select(0, packed.reshape(-1).int()).view(packed.shape[0], -1, 32)
    s = torch.exp2(scale.float() - 127.0)
    return (val * s.unsqueeze(-1)).reshape(packed.shape[0], -1)


def gname(p):
    return f"blk.{int(__import__('re').search(r'layers\.(\d+)\.', p).group(1))}."


def res_mix_lite(cur, stack, norm_w, proj_w, eps):
    """model.res_mix without materialising [N, S, D]: softmax over [stack..., cur] of
    qk . RMSNorm(src), mixing the raw sources."""
    if not stack:
        return cur
    qk = norm_w * proj_w
    srcs = list(stack) + [cur]
    logits = torch.stack([(s * qk).sum(-1) * torch.rsqrt(s.pow(2).mean(-1) + eps) for s in srcs], dim=1)
    p = torch.softmax(logits, dim=-1)
    out = srcs[0] * p[:, 0:1]
    for i in range(1, len(srcs)):
        out.add_(srcs[i] * p[:, i:i + 1])
    return out


def chunked_mm(x, w, rows=16384):
    return torch.cat([x[i:i + rows] @ w.t() for i in range(0, x.shape[0], rows)])


def kda_block(x, W, p, cfg, B, T, kb):
    """KDA attention block, batched over B sequences of length T, kb sequences at a time."""
    H, D = cfg["kda_heads"], cfg["head_dim"]
    A_log = W.f32(p + "A_log")
    if A_log.numel() != H:
        assert torch.all(A_log[H:] == 0), "A_log padding expected zero"
        A_log = A_log[:H]
    a = torch.exp(A_log).view(1, 1, H, 1)
    w = {n: W.f32(p + n + ".weight") for n in ("q_proj", "k_proj", "v_proj", "g_proj", "b_proj", "f_a_proj", "f_b_proj", "o_proj")}
    cw = {n: W.f32(p + f"{n}_conv1d.weight") for n in ("q", "k", "v")}
    dt = W.f32(p + "dt_bias").view(1, 1, H, D)
    on_w = W.f32(p + "o_norm.weight")
    g = gname(p)
    IM.dense([g + n + ".weight" for n in ("attn_q", "attn_k", "attn_v", "ssm_g", "ssm_beta", "ssm_f_a")], x)
    with PROF.sec("grams"):
        GR.dense("attn_in", [g + n + ".weight" for n in ("attn_q", "attn_k", "attn_v", "ssm_g", "ssm_beta", "ssm_f_a")], x)
    out_all = torch.empty_like(x)
    for b0 in range(0, B, kb):
        nb = min(kb, B - b0)
        xs = x[b0 * T:(b0 + nb) * T]

        def conv(name):
            y = (xs @ w[f"{name}_proj"].t()).view(nb, T, -1).transpose(1, 2)
            C = y.shape[1]
            y = F.conv1d(y, cw[name], padding=cw[name].shape[-1] - 1, groups=C)[..., :T]
            return F.silu(y).transpose(1, 2).reshape(nb, T, H, D)

        with PROF.sec("kda.proj"):
            q, k, v = conv("q"), conv("k"), conv("v")
            q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * (D ** -0.5)
            k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
            fa = xs @ w["f_a_proj"].t()
        IM.dense([g + "ssm_f_b.weight"], fa)
        with PROF.sec("grams"):
            GR.dense("ssm_f_b_in", [g + "ssm_f_b.weight"], fa)
        with PROF.sec("kda.proj"):
            g_raw = (fa @ w["f_b_proj"].t()).view(nb, T, H, D) + dt
            del fa
            if RT.get("fused_kda"):
                logdecay = cfg["gate_lower_bound"] * torch.sigmoid(a * g_raw)
            else:
                decay = torch.exp(cfg["gate_lower_bound"] * torch.sigmoid(a * g_raw))
            del g_raw
            beta = torch.sigmoid(xs @ w["b_proj"].t()).view(nb, T, H)
        if RT.get("fused_kda"):
            # fla's fused recurrent KDA kernel: same recurrence (state [V, K], decay along K, fp32
            # state in registers), different summation order: NOT bit-identical (--fused-kda only)
            from fla.ops.kda.fused_recurrent import fused_recurrent_kda_fwd
            with PROF.sec("kda.recurrence"):
                o, _ = fused_recurrent_kda_fwd(q.contiguous(), k.contiguous(), v.contiguous(), logdecay.contiguous(),
                                               beta.contiguous(), scale=1.0, initial_state=None,
                                               output_final_state=False, inplace_final_state=False,
                                               state_v_first=True)
            del q, k, v, logdecay
        else:
            S = torch.zeros(nb, H, D, D, device=x.device)
            o = torch.empty(nb, T, H, D, device=x.device)
            with PROF.sec("kda.recurrence"), fp32_exact():                 # exact fp32 under --fast too
                for t in range(T):
                    S.mul_(decay[:, t].unsqueeze(-2))                            # decay along K
                    kt = k[:, t]
                    vt = (v[:, t] - torch.matmul(S, kt.unsqueeze(-1)).squeeze(-1)) * beta[:, t].unsqueeze(-1)
                    S.add_(vt.unsqueeze(-1) * kt.unsqueeze(-2))
                    o[:, t] = torch.matmul(S, q[:, t].unsqueeze(-1)).squeeze(-1)
            del q, k, v, decay, S
        with PROF.sec("kda.proj"):
            g2 = (xs @ w["g_proj"].t()).view(nb, T, H, D)
            o = (rms_norm(o, on_w, cfg["o_norm_eps"]) * torch.sigmoid(g2)).reshape(nb * T, H * D)
        IM.dense([g + "attn_output.weight"], o)
        with PROF.sec("grams"):
            GR.dense("attn_out_in", [g + "attn_output.weight"], o)
        with PROF.sec("kda.proj"):
            out_all[b0 * T:(b0 + nb) * T] = o @ w["o_proj"].t()
    return out_all


def dsa_block(x, W, p, cfg, B, T, stats):
    nh, dn, dr, dv = cfg["n_heads"], cfg["qk_nope"], cfg["qk_rope"], cfg["v_head"]
    eps, K = cfg["rms_eps"], cfg["idx_topk"]
    ip = p + "indexer."
    wqa, wqb = W.f32(p + "q_a_proj.weight"), W.f32(p + "q_b_proj.weight")
    wkva, wkvb = W.f32(p + "kv_a_proj_with_mqa.weight"), W.f32(p + "kv_b_proj.weight")
    wiq, wik, wiw = W.f32(ip + "wq_b.weight"), W.f32(ip + "wk.weight"), W.f32(ip + "weights_proj.weight")
    qa_n, kva_n = W.f32(p + "q_a_layernorm.weight"), W.f32(p + "kv_a_layernorm.weight")
    kn_w, kn_b = W.f32(ip + "k_norm.weight"), W.f32(ip + "k_norm.bias")
    wg, wo = W.f32(p + "g_proj.weight"), W.f32(p + "o_proj.weight")
    scale = (dn + dr) ** -0.5
    out_all = torch.empty_like(x)
    gap = torch.full((B * T,), float("nan"))
    mass = torch.full((B * T,), float("nan"))                            # stays nan when dense mass is skipped
    pos = torch.arange(T, device=x.device)
    gp = gname(p)
    IM.dense([gp + n + ".weight" for n in ("attn_q_a", "attn_kv_a_mqa", "attn_gate", "indexer.attn_k", "indexer.proj")], x)
    with PROF.sec("grams"):
        GR.dense("attn_in", [gp + n + ".weight" for n in ("attn_q_a", "attn_kv_a_mqa", "attn_gate", "indexer.attn_k", "indexer.proj")], x)
    lat_v = torch.empty(T, nh, cfg["kv_lora"], device=x.device) if GR.on else None
    for b in range(B):
        xs = x[b * T:(b + 1) * T]
        q_c = rms_norm(xs @ wqa.t(), qa_n, eps)
        IM.dense([gp + "attn_q_b.weight", gp + "indexer.attn_q_b.weight"], q_c)
        q = (q_c @ wqb.t()).view(T, nh, dn + dr)
        IM.heads(gp + "attn_k_b.weight", q[..., :dn])
        kv_a = xs @ wkva.t()
        c = rms_norm(kv_a[:, :cfg["kv_lora"]], kva_n, eps)
        with PROF.sec("grams"):
            GR.dense("q_c", [gp + "attn_q_b.weight", gp + "indexer.attn_q_b.weight"], q_c)
            GR.heads("q_nope", [gp + "attn_k_b.weight"], q[..., :dn])
            GR.dense("kv_c", [gp + "attn_kv_b.weight"], c)
        k_pe = kv_a[:, cfg["kv_lora"]:]
        kv = (c @ wkvb.t()).view(T, nh, dn + dv)
        k_nope, v = kv[..., :dn], kv[..., dn:]
        iq = (q_c @ wiq.t()).view(T, cfg["idx_heads"], cfg["idx_dim"])
        ik = F.layer_norm(xs @ wik.t(), (cfg["idx_dim"],), kn_w, kn_b, eps)
        iw = (xs @ wiw.t()) * (cfg["idx_heads"] ** -0.5) * (cfg["idx_dim"] ** -0.5)
        out = torch.empty(T, nh, dv, device=x.device)
        CH = 256
        for t0 in range(0, T, CH):
            t1 = min(T, t0 + CH)
            rows = pos[t0:t1]
            causal = pos.unsqueeze(0) <= rows.unsqueeze(1)
            sc = (torch.einsum("chd,sd->chs", iq[t0:t1], ik).relu() * iw[t0:t1].unsqueeze(-1)).sum(1)
            sc = sc.masked_fill(~causal, float("-inf"))
            big = (rows + 1) > K
            sel = causal.clone()
            if big.any():
                top_v, top_i = sc[big].topk(K + 1, dim=-1)
                sb = torch.zeros_like(causal[big])
                sb.scatter_(1, top_i[:, :K], True)
                sel[big] = sb
                g = torch.full((t1 - t0,), float("nan"), device=x.device)
                g[big] = top_v[:, K - 1] - top_v[:, K]
                gap[b * T + t0:b * T + t1] = g.cpu()
            s_attn = (torch.einsum("chd,shd->chs", q[t0:t1, :, :dn], k_nope)
                      + torch.einsum("chd,sd->chs", q[t0:t1, :, dn:], k_pe)) * scale
            if not SKIP_DENSE_MASS["on"]:
                dense = torch.softmax(s_attn.masked_fill(~causal.unsqueeze(1), float("-inf")), dim=-1)
                mass[b * T + t0:b * T + t1] = (dense * sel.unsqueeze(1)).sum(-1).mean(-1).cpu()
                del dense
            pr = torch.softmax(s_attn.masked_fill(~sel.unsqueeze(1), float("-inf")), dim=-1)
            out[t0:t1] = torch.einsum("chs,shd->chd", pr, v)
            if IM.on:   # attn_v_b consumes the attention output in latent space, per head
                lv = torch.einsum("chs,sd->chd", pr, c)
                IM.heads(gp + "attn_v_b.weight", lv)
                if lat_v is not None:
                    lat_v[t0:t1] = lv
            elif lat_v is not None:       # causal: pr is exactly 0 beyond column t1-1
                with PROF.sec("grams"):
                    lat_v[t0:t1] = torch.einsum("chs,sd->chd", pr[..., :t1], c[:t1])
        if lat_v is not None:
            with PROF.sec("grams"):
                GR.heads("attn_lat", [gp + "attn_v_b.weight"], lat_v)
        gate = torch.sigmoid(xs @ wg.t())
        og = out.reshape(T, nh * dv) * gate
        IM.dense([gp + "attn_output.weight"], og)
        with PROF.sec("grams"):
            GR.dense("attn_out_in", [gp + "attn_output.weight"], og)
        out_all[b * T:(b + 1) * T] = og @ wo.t()
    stats["dsa_gap"] = gap.half().numpy()
    stats["dsa_mass"] = mass.half().numpy()
    return out_all


class ZeroStats:
    """MXFP4 zero structure of one layer's experts, accumulated expert by expert:
    <w>_frac [E] zero fraction, <w>_col [E, in] per input channel, <w>_row [E, out] per
    output row, <w>_scale_vs_zeros [40, 33] joint histogram of the block's E8M0 scale
    (100..139, clamped) and its number of zero codes (0..32)."""
    LO = 100

    def __init__(self, E, dev):
        self.E, self.dev, self.acc = E, dev, {}

    def add(self, e, wname, pk, sc):
        iz = torch.stack(((pk & 7) == 0, ((pk >> 4) & 7) == 0), -1).reshape(pk.shape[0], -1)
        out_, in_ = iz.shape
        if wname not in self.acc:
            self.acc[wname] = dict(frac=torch.zeros(self.E), col=torch.zeros(self.E, in_, dtype=torch.float16),
                                   row=torch.zeros(self.E, out_, dtype=torch.float16),
                                   hist=torch.zeros(40 * 33, dtype=torch.long, device=self.dev))
        a = self.acc[wname]
        f = iz.float()
        a["frac"][e] = f.mean().item()
        a["col"][e] = f.mean(0).half().cpu()
        a["row"][e] = f.mean(1).half().cpu()
        blk = iz.view(out_, -1, 32).sum(-1)
        idx = ((sc.long().clamp(self.LO, self.LO + 39) - self.LO) * 33 + blk).view(-1)
        a["hist"].index_add_(0, idx, torch.ones_like(idx))

    def arrays(self):
        z = {}
        for w, a in self.acc.items():
            z[f"{w}_frac"] = a["frac"].numpy()
            z[f"{w}_col"] = a["col"].numpy()
            z[f"{w}_row"] = a["row"].numpy()
            z[f"{w}_scale_vs_zeros"] = a["hist"].view(40, 33).cpu().numpy()
        return z


def moe_block(x, W, p, cfg, stats, zs):
    E, k = cfg["n_experts"], cfg["topk"]
    g = gname(p)
    il = int(g.split(".")[1])
    IM.dense([g + n + ".weight" for n in ("ffn_gate_inp", "ffn_routed_down", "ffn_gate_shexp", "ffn_up_shexp")], x)
    with PROF.sec("grams"):
        GR.dense("ffn_in", [g + n + ".weight" for n in ("ffn_gate_inp", "ffn_routed_down", "ffn_gate_shexp", "ffn_up_shexp")], x)
    with PROF.sec("moe.router"):
        logits = x @ W.f32(p + "gate.weight").t()
        scores = torch.sigmoid(logits)
        biased = scores + W.f32(p + "gate.e_score_correction_bias")
        top_v, top_i = biased.topk(k + 1, dim=-1)
        sel = top_i[:, :k]
        w = scores.gather(1, sel)
        if cfg["renorm"]:
            w = w / w.sum(-1, keepdim=True)
        w = w * cfg["routed_scale"]
        stats["moe_counts"] = torch.bincount(sel.reshape(-1), minlength=E).cpu().numpy()
        stats["moe_gap"] = (top_v[:, k - 1] - top_v[:, k]).half().cpu().numpy()

        lat = chunked_mm(x, W.f32(p + "routed_expert_down_proj.weight"))
        acc = torch.zeros_like(lat)
        ep = p + "experts."
        order = torch.argsort(sel.reshape(-1))
        flat = sel.reshape(-1)[order]
        bounds = torch.searchsorted(flat, torch.arange(E + 1, device=x.device)).cpu().tolist()

    def expert(e, mats):
        if zs is not None:
            for n in ("w1", "w2", "w3"):
                zs.add(e, n, *mats[n])
        lo, hi = bounds[e], bounds[e + 1]
        if hi > lo:
            with PROF.sec("moe.experts.dequant"):
                W1, W3, W2 = (mxfp4_dequant_lut(*mats[n]) for n in ("w1", "w3", "w2"))
            with PROF.sec("moe.experts.mm"):
                idx = order[lo:hi]
                tok, slot = idx // k, idx % k
                h = lat[tok]
                IM.expert(g + "ffn_gate_exps.weight", e, h, E); IM.expert(g + "ffn_up_exps.weight", e, h, E)
                a_ = situ(h @ W1.t(), h @ W3.t(), cfg["situ_beta"], cfg["situ_lin_beta"])
                IM.expert(g + "ffn_down_exps.weight", e, a_, E)
                y = a_ @ W2.t()
                acc.index_add_(0, tok, y * w[tok, slot].unsqueeze(-1))

    pf = RT["pf"]
    with PROF.sec("moe.experts"):
        if pf is None:
            for e in range(E):
                with PROF.sec("moe.experts.fetch"):
                    mats = {}
                    for n in ("w1", "w2", "w3"):
                        pk = W.raw(f"{ep}{e}.{n}.weight_packed").to(x.device, non_blocking=True)
                        s = W.raw(f"{ep}{e}.{n}.weight_scale").to(x.device, non_blocking=True)
                        mats[n] = (pk, s)
                expert(e, mats)
        else:
            C = RT["chunk"]
            for c in range((E + C - 1) // C):
                with PROF.sec("moe.experts.fetch"):
                    hd = pf.get(("exp", il, c))
                for e in range(c * C, min(E, (c + 1) * C)):
                    expert(e, {n: (hd.t(f"{ep}{e}.{n}.weight_packed"), hd.t(f"{ep}{e}.{n}.weight_scale"))
                               for n in ("w1", "w2", "w3")})
                pf.release(hd)
    if XDW.on(il):
        with PROF.sec("dump.copy"):
            XDW.write(il, lat, sel, w, acc)
    with PROF.sec("moe.shared"):
        if W.has(p + "routed_expert_norm.weight"):
            acc = rms_norm(acc, W.f32(p + "routed_expert_norm.weight"), cfg["rms_eps"])
    IM.dense([g + "ffn_routed_up.weight"], acc)
    with PROF.sec("grams"):
        GR.dense("routed_up_in", [g + "ffn_routed_up.weight"], acc)
    with PROF.sec("moe.shared"):
        routed = chunked_mm(acc, W.f32(p + "routed_expert_up_proj.weight"))
        sp = p + "shared_experts."
        G, U, Dn = (W.f32(sp + n + "_proj.weight") for n in ("gate", "up", "down"))
    parts = []
    for i in range(0, x.shape[0], 16384):
        with PROF.sec("moe.shared"):
            a_ = situ(x[i:i + 16384] @ G.t(), x[i:i + 16384] @ U.t(), cfg["situ_beta"], cfg["situ_lin_beta"])
        IM.dense([g + "ffn_down_shexp.weight"], a_)
        with PROF.sec("grams"):
            GR.dense("shexp_act", [g + "ffn_down_shexp.weight"], a_)
        with PROF.sec("moe.shared"):
            parts.append(a_ @ Dn.t())
    shared = torch.cat(parts)
    return routed + shared


def dense_mlp(x, W, p, cfg):
    G, U, Dn = (W.f32(p + n + "_proj.weight") for n in ("gate", "up", "down"))
    g = gname(p)
    IM.dense([g + "ffn_gate.weight", g + "ffn_up.weight"], x)
    with PROF.sec("grams"):
        GR.dense("ffn_in", [g + "ffn_gate.weight", g + "ffn_up.weight"], x)
    parts = []
    for i in range(0, x.shape[0], 8192):
        a_ = situ(x[i:i + 8192] @ G.t(), x[i:i + 8192] @ U.t(), cfg["situ_beta"], cfg["situ_lin_beta"])
        IM.dense([g + "ffn_down.weight"], a_)
        with PROF.sec("grams"):
            GR.dense("ffn_act", [g + "ffn_down.weight"], a_)
        parts.append(a_ @ Dn.t())
    return torch.cat(parts)


def kld_records(logits):
    """llama-perplexity log_softmax(..., uint16_t *log_prob, ...) for each row: two floats
    (scale, min_log_prob) then n_vocab uint16 = round((logit - min_logit)/scale)."""
    mx = logits.max(-1, keepdim=True).values
    mn = torch.maximum(logits.min(-1, keepdim=True).values, mx - 16)
    lse = torch.log(torch.exp(logits.double() - mx.double()).sum(-1, keepdim=True)).float()
    min_log_prob = mn - mx - lse
    scale = (mx - mn) / 65535.0
    q = torch.where(logits > mn, torch.round((logits - mn) / torch.where(scale > 0, scale, torch.ones_like(scale))),
                    torch.zeros_like(logits))
    q = torch.where(scale > 0, q, torch.zeros_like(q)).to(torch.int32)
    head = torch.cat([scale, min_log_prob], -1).float().cpu().numpy().view(np.uint16)   # [rows, 4]
    body = q.cpu().numpy().astype(np.uint16)
    return np.concatenate([head, body], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--tokens", required=True, type=Path, help=".npy int32 [n_seq, n_ctx]")
    ap.add_argument("--seqs", default=None, help="slice of sequences, e.g. 0:10")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--kda-batch", type=int, default=5)
    ap.add_argument("--ckpt-every", type=int, default=8, help="checkpoint every N layers (0: never)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dump-lout", action="store_true", help="save l_out per layer (validation only)")
    ap.add_argument("--no-zero-stats", action="store_true")
    ap.add_argument("--imatrix", action="store_true", help="collect llama-imatrix sums per layer, skip KLD records and dense-mass")
    ap.add_argument("--expert-dump-layers", default="", help="comma list of MoE layers whose routed-expert inputs are dumped (one row per token); 'all' = every MoE layer")
    ap.add_argument("--expert-dump-dir", type=Path, default=None, help="output dir for --expert-dump-layers")
    ap.add_argument("--stop-after-layer", type=int, default=None, help="end the pass after this layer (no logits)")
    # capture additions
    ap.add_argument("--grams", action="store_true", help="accumulate dense-matmul input Grams per layer")
    ap.add_argument("--gram-dir", type=Path, default=None, help="per-process Gram output dir (default <out>/grams)")
    ap.add_argument("--gram-reserve-gb", type=float, default=8.0, help="block Gram writes while the gram dir has less free space")
    ap.add_argument("--part-id", type=int, default=None, help="part number recorded in the Gram metadata (merge_grams splits on it)")
    ap.add_argument("--no-logits", action="store_true", help="after the last layer: final mix + norm (output.weight Gram), no lm_head / KLD records")
    ap.add_argument("--fast", action="store_true", help="TF32 matmuls (KDA recurrence and Grams stay fp32). NOT bit-identical to the default")
    ap.add_argument("--fused-kda", action="store_true", help="KDA recurrence with fla's fused_recurrent_kda kernel. NOT bit-identical")
    ap.add_argument("--no-prefetch", action="store_true", help="read weights through safetensors on the main thread (no prefetch thread)")
    ap.add_argument("--expert-chunk", type=int, default=32, help="experts per prefetch read")
    ap.add_argument("--expert-ring", type=int, default=8, help="GPU buffers of --expert-chunk experts (~0.56 GB each at 32)")
    ap.add_argument("--io-threads", type=int, default=4)
    ap.add_argument("--layer-cache", type=Path, default=None, help="read layer shards from this layer_cache.py dir")
    ap.add_argument("--cache-tag", default=None, help="progress name in the layer cache (default: out dir name)")
    ap.add_argument("--sync-writes", action="store_true", help="write dumps / Grams on the main thread")
    ap.add_argument("--write-depth", type=int, default=2, help="pending background write jobs before the forward pass waits")
    ap.add_argument("--profile", action="store_true", help="synchronising per-section timers, logged per layer")
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    if a.fast:
        torch.set_float32_matmul_precision("high")       # TF32; cudnn (KDA short conv) stays exact
    PROF.on = a.profile
    RT["fused_kda"] = a.fused_kda
    (a.out / "stats").mkdir(parents=True, exist_ok=True)

    tc = json.load(open(a.model / "config.json"))["text_config"]
    lac, dsa = tc["linear_attn_config"], tc["dsa_config"]
    n_layer = a.layers or tc["num_hidden_layers"]
    if a.expert_dump_layers:
        XDW.layers = (set(range(tc["first_k_dense_replace"], n_layer)) if a.expert_dump_layers == "all"
                      else {int(x) for x in a.expert_dump_layers.split(",")})
        XDW.dir = a.expert_dump_dir
        assert XDW.dir is not None, "--expert-dump-layers needs --expert-dump-dir"
        SKIP_DENSE_MASS["on"] = True
    if a.imatrix:
        IM.on = True; SKIP_DENSE_MASS["on"] = True
        (a.out / "imatrix").mkdir(parents=True, exist_ok=True)
    gram_dir = a.gram_dir or (a.out / "grams")
    if a.grams:
        GR.on = True; SKIP_DENSE_MASS["on"] = True
        gram_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(
        rms_eps=tc["rms_norm_eps"], kda_heads=lac["num_heads"], head_dim=lac["head_dim"],
        gate_lower_bound=lac["gate_lower_bound"], o_norm_eps=1e-5,
        n_heads=tc["num_attention_heads"], qk_nope=tc["qk_nope_head_dim"], qk_rope=tc["qk_rope_head_dim"],
        v_head=tc["v_head_dim"], kv_lora=tc["kv_lora_rank"],
        idx_heads=dsa["index_num_heads"], idx_dim=dsa["index_head_dim"], idx_topk=dsa["index_topk"],
        n_experts=tc["num_experts"], topk=tc["num_experts_per_token"], renorm=tc["moe_renormalize"],
        routed_scale=tc["routed_scaling_factor"], situ_beta=tc["activation_situ_beta"],
        situ_lin_beta=tc["activation_situ_linear_beta"], res_bs=tc["attn_res_block_size"],
        dense_lead=tc["first_k_dense_replace"],
    )
    full_attn = set(x - 1 for x in lac["full_attn_layers"])
    W = StreamWeights(km.Weights(a.model, dev))

    toks_all = np.load(a.tokens)
    s0, s1 = (0, toks_all.shape[0]) if a.seqs is None else map(int, a.seqs.split(":"))
    toks = toks_all[s0:s1]
    B, T = toks.shape
    XDW.toks, XDW.s0, XDW.tokens_path = toks, s0, a.tokens.resolve()
    eps = cfg["rms_eps"]
    ck = a.out / "ckpt.pt"
    start = 0
    if a.resume and ck.exists():
        st = torch.load(ck, map_location=dev)
        prefix, stack, start = st["prefix"], st["stack"], st["next_layer"]
        log(f"resumed at layer {start} (stack {len(stack)})")
    else:
        emb = W.raw(PFX + "embed_tokens.weight")
        prefix = emb[torch.from_numpy(toks.reshape(-1).astype(np.int64))].to(dev).float()
        del emb
        stack = []
    last = n_layer - 1 if a.stop_after_layer is None else min(n_layer - 1, a.stop_after_layer)
    gram_meta = {"part": a.part_id, "seqs": [s0, s0 + B], "n_ctx": T, "tokens_file": str(a.tokens.resolve()),
                 "fast": bool(a.fast or a.fused_kda), "tf32": bool(a.fast), "fused_kda": bool(a.fused_kda),
                 "model": str(a.model)}

    resolver = None
    if a.layer_cache is not None:
        resolver = stream.LayerCacheResolver(a.model, a.layer_cache, a.cache_tag or f"{a.out.name}-{os.getpid()}")
    if not a.no_prefetch:
        smap = stream.ShardMap(a.model)
        RT["chunk"] = a.expert_chunk
        slot = stream.expert_chunk_bytes(smap, max(start, cfg["dense_lead"]), cfg["n_experts"], a.expert_chunk, PFX)
        plan = stream.build_plan(smap, range(start, last + 1), cfg["dense_lead"], cfg["n_experts"], a.expert_chunk, PFX)
        RT["pf"] = stream.Prefetcher(smap, dev, plan, a.expert_ring, slot, resolver=resolver, io_threads=a.io_threads)
    if not a.sync_writes:
        RT["writer"] = stream.AsyncWriter(a.write_depth)
    log(f"B={B} T={T} seqs={s0}:{s1} layers={n_layer} start={start} last={last} fast={a.fast} fused_kda={a.fused_kda} grams={a.grams} "
        f"prefetch={'off' if a.no_prefetch else f'{a.expert_ring}x{a.expert_chunk} experts'} "
        f"layer_cache={a.layer_cache} async_writes={not a.sync_writes}")

    def finish(msg):
        if RT["writer"] is not None:
            RT["writer"].close()
        if RT["pf"] is not None:
            RT["pf"].close()
        if resolver is not None:
            resolver.progress(n_layer, done=True)
        if ck.exists():
            ck.unlink()
        log(msg)
        return 0

    def save_grams(name):
        fl = GR.flush()
        meta = dict(gram_meta, saved=time.strftime("%Y-%m-%d %H:%M:%S"))
        job = lambda: gram_io.write_layer(gram_dir, name, fl, meta, a.gram_reserve_gb, wait=True)   # merge_grams --watch --delete frees space
        if RT["writer"] is not None:
            RT["writer"].submit(job)
        else:
            job()

    t_run = time.time()
    for il in range(start, last + 1):
        t0 = time.time()
        p = f"{PFX}layers.{il}."
        st = {}
        with PROF.sec("fetch.dense"):
            W.begin_layer(il)
        with PROF.sec("resmix"):
            cur = res_mix_lite(prefix, stack, W.f32(p + "self_attention_res_norm.weight"),
                               W.f32(p + "self_attention_res_proj.weight").view(-1), eps)
            banked = il % cfg["res_bs"] == 0
            if banked:
                stack.append(prefix)
            h = rms_norm(cur, W.f32(p + "input_layernorm.weight"), eps)
            del cur
        if il in full_attn:
            with PROF.sec("attn.dsa"):
                attn = dsa_block(h, W, p + "self_attn.", cfg, B, T, st)
        else:
            with PROF.sec("attn.kda"):
                attn = kda_block(h, W, p + "self_attn.", cfg, B, T, a.kda_batch)
        del h
        with PROF.sec("resmix"):
            prefix = attn if banked else prefix + attn
            del attn
            cur = res_mix_lite(prefix, stack, W.f32(p + "mlp_res_norm.weight"),
                               W.f32(p + "mlp_res_proj.weight").view(-1), eps)
            h = rms_norm(cur, W.f32(p + "post_attention_layernorm.weight"), eps)
            del cur
        zs = None if a.no_zero_stats or il < cfg["dense_lead"] else ZeroStats(cfg["n_experts"], dev)
        if il < cfg["dense_lead"]:
            with PROF.sec("ffn.dense"):
                ffn = dense_mlp(h, W, p + "mlp.", cfg)
        else:
            with PROF.sec("ffn.moe"):
                ffn = moe_block(h, W, p + "block_sparse_moe.", cfg, st, zs)
        del h
        prefix = prefix + ffn
        del ffn
        W.end_layer()
        torch.cuda.synchronize(dev)
        for k_, v_ in st.items():
            np.save(a.out / "stats" / f"{k_}-{il}.npy", v_)
        if IM.on:
            np.savez(a.out / "imatrix" / f"layer-{il}.npz", **IM.flush())
        if GR.on:
            with PROF.sec("grams.flush"):
                save_grams(il)
        if zs is not None:
            np.savez(a.out / "stats" / f"zeros-{il}.npz", **zs.arrays())
        if a.dump_lout:
            np.save(a.out / f"l_out-{il}.npy", prefix[:T].cpu().numpy())
        dt = time.time() - t0
        done = il + 1 - start
        eta = (time.time() - t_run) / done * (last - il)
        extra = ""
        if RT["pf"] is not None:
            ps = RT["pf"].stats
            extra = (f"  read {ps['read_bytes'] / 1e9:.0f}GB/{ps['read_s']:.0f}s wait_w {ps['wait_get_s']:.0f}s")
        if RT["writer"] is not None:
            extra += f" writer_busy {RT['writer'].busy_s:.0f}s"
        log(f"layer {il:2d}/{n_layer} {'dsa' if il in full_attn else 'kda'} {dt:6.1f}s  "
            f"rms={prefix.pow(2).mean().sqrt():.4f}  mem={torch.cuda.max_memory_allocated(dev)/2**30:.1f}GiB  eta {eta/60:.1f} min{extra}")
        if PROF.on:
            t = PROF.take()
            log(f"profile layer {il}: " + " ".join(f"{k}={v:.2f}" for k, v in sorted(t.items())))
        if a.ckpt_every and (il + 1) % a.ckpt_every == 0 and il < last:
            if RT["writer"] is not None:
                RT["writer"].drain()          # every file of layers <= il complete before the checkpoint says so
            tmp = a.out / "ckpt.pt.tmp"
            torch.save({"prefix": prefix, "stack": stack, "next_layer": il + 1}, tmp)
            os.replace(tmp, ck)
            log(f"checkpoint after layer {il}")
        torch.cuda.empty_cache()
        if a.stop_after_layer is not None and il >= a.stop_after_layer:
            return finish(f"stopping after layer {il} as requested")

    # final mix, norm, lm_head -> KLD-base records for positions [T/2, T-1)
    cur = res_mix_lite(prefix, stack, W.f32(PFX + "output_attn_res_norm.weight"),
                       W.f32(PFX + "output_attn_res_proj.weight").view(-1), eps)
    del prefix, stack
    cur = rms_norm(cur, W.f32(PFX + "norm.weight"), eps)
    if GR.on:
        GR.dense("output_in", ["output.weight"], cur)
        save_grams("output")
    if IM.on:
        IM.dense(["output.weight"], cur)
        np.savez(a.out / "imatrix" / "output.npz", **IM.flush())
        np.save(a.out / "tokens.npy", toks)
        return finish(f"imatrix done in {(time.time() - t_run)/60:.1f} min")
    if a.no_logits:
        np.save(a.out / "tokens.npy", toks)
        return finish(f"done (no logits) in {(time.time() - t_run)/60:.1f} min")
    lm = W.f32("language_model.lm_head.weight")
    first = T // 2
    part = a.out / "kld_part.bin"
    rec_per_seq = T - 1 - first
    nv = 2 * ((lm.shape[0] + 1) // 2) + 4
    done_seqs = (part.stat().st_size // (rec_per_seq * nv * 2)) if part.exists() else 0
    with open(part, "r+b" if part.exists() else "wb") as f:
        f.seek(done_seqs * rec_per_seq * nv * 2)
        f.truncate()
        for b in range(done_seqs, B):
            rows = cur[b * T + first: b * T + T - 1]
            for r0 in range(0, rows.shape[0], 512):
                rec = kld_records(rows[r0:r0 + 512] @ lm.t())
                assert rec.shape[1] == nv, rec.shape
                f.write(rec.tobytes())
            f.flush()
            log(f"kld records: sequence {b + 1}/{B} written")
    np.save(a.out / "tokens.npy", toks)
    if a.dump_lout:
        np.save(a.out / "logits_seq0.npy", (cur[:T] @ lm.t()).cpu().numpy())
    return finish(f"done in {(time.time() - t_run)/60:.1f} min")


if __name__ == "__main__":
    sys.exit(main())
