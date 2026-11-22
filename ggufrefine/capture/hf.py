"""Reference capture for checking ggufrefine.capture.ggml: input Grams of every Linear of a Hugging Face transformers
model, fp32 weights and activations, TF32 off (model on GPU 0, accumulators on the others), on the same token file, written in the same grams format and with the
same groups (modules reading the same input tensor share one Gram, named after the first one's GGUF name).

HF module names are mapped to GGUF tensor names with gguf-py's TensorNameMap for the GGUF's architecture, so any
architecture gguf-py converts is covered without model code here.

    python -m ggufrefine.capture.hf --model HF_DIR --gguf MODEL.gguf --tokens T.npy --seqs 0:2 --part-id 0 --out DIR
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .. import ggufio, paths  # noqa: F401  (paths: gguf-py on sys.path)
from . import grams as gram_io
from .ggml import layer_of


def gguf_name_map(gguf_path):
    import gguf
    from gguf import GGUFReader
    r = GGUFReader(str(gguf_path))
    arch = bytes(r.fields["general.architecture"].parts[-1]).decode()
    n_layer = int(r.fields[f"{arch}.block_count"].parts[-1][0])
    names = {t.name for t in r.tensors}
    arch_id = next(k for k, v in gguf.MODEL_ARCH_NAMES.items() if v == arch)
    tmap = gguf.get_tensor_name_map(arch_id, n_layer)

    def to_gguf(hf):
        parts = hf.split(".")
        for i in range(len(parts)):
            for cand in (".".join(parts[i:]), "model." + ".".join(parts[i:])):
                g = tmap.get_name(cand)
                if g is not None:
                    g += ".weight"
                    if g in names:
                        return g
                    if g == "output.weight" and "token_embd.weight" in names:
                        return "token_embd.weight"                    # tied output projection
        return None
    return to_gguf


def load_model(model_dir, dev):
    """the whole fp32 model on one GPU (Gram accumulators go to the others)"""
    import transformers
    try:
        m = transformers.AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    except (ValueError, KeyError):
        m = transformers.AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.float32)
    return m.to(dev)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--gguf", required=True, type=Path, help="GGUF of the same model (tensor names, architecture)")
    ap.add_argument("--tokens", required=True, type=Path)
    ap.add_argument("--seqs", default=None)
    ap.add_argument("--part-id", type=int, required=True)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    to_gguf = gguf_name_map(a.gguf)
    n_gpu = torch.cuda.device_count()
    model = load_model(a.model, torch.device("cuda", 0)).eval()

    def acc_dev(g):                                       # accumulators by layer over GPUs 1.. (0 if only one)
        L = layer_of(g)
        return torch.device("cuda", 0 if n_gpu == 1 else 1 + (0 if L == "output" else L) % (n_gpu - 1))
    G = gram_io.Grams()
    G.on = True
    groups, cur, at = {}, {}, [None]
    hooked = 0

    def hook(name):
        def pre(mod, args):
            x = args[0]
            if layer_of(name) != at[0]:                   # inputs are held until the layer ends, so an address
                cur.clear()                               # cannot be reused for another input of the same layer
                at[0] = layer_of(name)
            key = (x.data_ptr(), tuple(x.shape), x.dtype)
            hit = cur.get(key)
            if hit is not None:
                if name not in groups[hit[0]]:
                    groups[hit[0]].append(name)
                return
            g = name
            cur[key] = (g, x)
            groups.setdefault(g, [name])
            G.dense(g, [g], x.reshape(-1, x.shape[-1]).float().to(acc_dev(g)))
        return pre

    targets = [("model.", model.get_decoder())]          # the language decoder only (not a vision tower)
    out_emb = model.get_output_embeddings()
    for prefix, root in targets:
        for n, m in root.named_modules():
            if isinstance(m, torch.nn.Linear):
                g = to_gguf(prefix + n)
                if g is not None:
                    m.register_forward_pre_hook(hook(g))
                    hooked += 1
    if isinstance(out_emb, torch.nn.Linear):
        out_emb.register_forward_pre_hook(hook(to_gguf("lm_head")))
        hooked += 1
    toks = np.load(a.tokens)
    s0, s1 = (0, toks.shape[0]) if a.seqs is None else map(int, a.seqs.split(":"))
    print(f"[{time.strftime('%H:%M:%S')}] {hooked} Linear modules hooked; {s1 - s0} sequences x {toks.shape[1]}", flush=True)
    dev0 = torch.device("cuda", 0)
    t0 = time.time()
    with torch.no_grad():
        for i in range(s0, s1):
            cur.clear()
            at[0] = None
            model(input_ids=torch.from_numpy(toks[i:i + 1].astype(np.int64)).to(dev0), use_cache=False)
            done = i + 1 - s0
            el = time.time() - t0
            print(f"[{time.strftime('%H:%M:%S')}] {done}/{s1 - s0} sequences  {done * toks.shape[1] / el:.0f} tok/s", flush=True)
    meta = {"part": a.part_id, "seqs": [s0, s1], "n_ctx": int(toks.shape[1]), "tokens_file": str(a.tokens.resolve()),
            "capture": "transformers fp32 (ggufrefine.capture.hf)", "model": str(a.model)}
    out = {}
    for g, (info, tri, s) in G.flush().items():
        out.setdefault(layer_of(g), {})[g] = (dict(info, weights=list(groups[g])), tri, s)
    for layer, fl in out.items():
        gram_io.write_layer(a.out, layer, fl, dict(meta, saved=time.strftime("%Y-%m-%d %H:%M:%S")))
    print(f"[{time.strftime('%H:%M:%S')}] done in {(time.time() - t0) / 60:.1f} min -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
