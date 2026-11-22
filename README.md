# gguf-refine

Hessian-refined GGUF quantization: the same stock IQ block formats (the output loads in unmodified llama.cpp), with
codewords chosen to minimise tr((Q-W) H (Q-W)^T) under calibration activations and the stock scales fixed; plus
per-tensor type allocation and KLD evaluation against a native reference.

Kimi-K3.1-Pro, all 92 MoE layers refined (experts IQ2_XXS, non-experts Q6_K, 701 GiB), against an identically built
stock file: combined c8192 KLD 0.0675 -> 0.0493 (-27%, 20/20 chunks; -25% on tokens with no overlap with the
calibration text); wiki.test.raw mean KLD unchanged.

## Setup

    native/build.sh                                   # clone llama.cpp at native/LLAMA_COMMIT, apply native/patches, CUDA build
    python -m venv .venv && .venv/bin/pip install -e '.[kimi]'
    .venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v

The package reads GGUFs with `native/llama.cpp/gguf-py` and loads the ggml quantizer from `native/llama.cpp/build/bin`
(`GGUF_REFINE_LLAMA` / `GGUF_REFINE_BIN` point it at another build). Refinement needs a CUDA GPU (Triton kernels);
codecs, planning, patching and the CPU tests do not.

## Pipeline

1. **Capture**: input Grams of every weight matmul on calibration text, held out by part.
   - Any model llama.cpp runs with its layers in VRAM: `python -m ggufrefine.capture.ggml` evaluates llama.cpp's own
     graph (libllama-capture, native patch 0010) and accumulates the Grams on the GPUs from its buffers, then
     `python -m ggufrefine.capture.merge_grams` builds the train / test splits. `capture.hf` is an fp32 transformers
     reference for checking a capture (`capture.compare`).
   - Kimi-K3.1 (1.56 TB reference, larger than VRAM + RAM): `models/kimi_k3/capture/capture.sh`, a layer-major fp32
     PyTorch forward that dumps the routed-expert inputs. It streams each layer's weights once per round of tokens;
     llama.cpp runs every ubatch through all layers, so the weights that overflow VRAM are re-read per ubatch (measured:
     one 65k-token batch did not finish in 84 min, against 6 h for the whole 2.5M-token capture layer-major).
2. **Refine**: per group of tensors sharing one input, Hessians from the capture, stock codes from the quantizer +
   imatrix, coordinate descent on the codewords; codes written straight into a stock GGUF of the same recipe.
   Dense / ggml-captured models: `python -m ggufrefine.models.dense`; K3.1: `models/kimi_k3/refine_all.sh TYPE TARGET.gguf`.
3. **Plan** (optional): held-out error per (layer, role, type) on a stratified expert sample (`errtab_all.sh`), then an
   exact knapsack over types under a byte budget (`python -m ggufrefine.alloc.plan`).
4. **Evaluate**: `models/kimi_k3/matched_eval.sh` builds the stock file with the same recipe, checks that it differs
   from the refined one only in the refined tensors, and scores both against the same KLD bases (`llama-kld-swap`).

Each script lists its environment variables (model directory, capture root, imatrix, ...) in its header.

## Layout

| Path | What |
|---|---|
| `ggufrefine/refine/engine.py` | FastRefiner: batched coordinate descent with Triton kernels; model-independent |
| `ggufrefine/refine/problem.py` | Group (tensors sharing one input / Hessian), stock-code sources (GGUF, or quantizer + imatrix) |
| `ggufrefine/refine/driver.py` | resumable batched refinement of one layer's groups; npy or direct-to-GGUF output |
| `ggufrefine/quant/` | IQ block codecs (`codes.py`); Triton kernels per codebook family (`kernels/`); ggml quantizer via ctypes |
| `ggufrefine/ggufio.py` | tensor table, per-expert reads, dequantized weights, imatrix rows, verified in-place writes |
| `ggufrefine/fold.py`, `patch.py` | fold `<tensor>-e<E>.npy` outputs into a GGUF in place; in-place patch with backup / revert |
| `ggufrefine/capture/` | Gram storage format and merger; ggml-graph capture (`ggml.py`), fp32 transformers reference (`hf.py`), capture comparison (`compare.py`) |
| `ggufrefine/alloc/` | error tables (`errtab.py`), stratified estimates + type plan (`plan.py`), exact knapsack (`knapsack.py`) |
| `ggufrefine/eval/` | llama-kld-swap client (`kldswap.py`), multi-base runner (`kld.py`), calibration-overlap split (`overlap.py`), sensitivity probes (`probe.py`), build check (`check_build.py`) |
| `ggufrefine/models/dense.py` | groups and Hessians from merged Gram splits, weights from a reference GGUF: any ggml-captured model |
| `ggufrefine/models/kimi_k3/` | K3.1 adapter: groups and plan config, MXFP4 safetensors weights, Hessians from the expert dump (`expert_dump.py`), capture (`capture/`), CLIs and run scripts |
| `native/` | llama.cpp patch series on `native/LLAMA_COMMIT` and its build script |
| `tests/` | `test_cpu.py` (no GPU); `gpu/` (engine vs fp64 references, needs CUDA); `test_k31_data.py`, `acceptance_k31.py` (need K3.1 files, see below) |

## Model adapters

The driver needs three sources (`refine/problem.py`): weights by GGUF tensor name and expert, Hessians per group and
expert (train / held-out), and stock codes. `ggufio.Gguf` provides weights and codes from any GGUF, so a new model
needs its groups and a Hessian source. Dense models are groups with one expert. A model's `ALLOC` config (roles,
propagation factors, scale tensor) drives `alloc.plan` and `eval.probe`.

## Tests that need K3.1 data

`test_k31_data.py` runs when `K31_MODEL` (HF directory) and `K31_GGUF` (BF16/MXFP4 GGUF) are set.
`acceptance_k31.py` re-refines 16 experts of layer 46 and requires codes byte-identical to a reference refined GGUF
(`K31_REFINED`, `K31_MODEL`, `K31_DUMP`, `K31_IMATRIX`); byte identity holds for the environment in
`tests/requirements-k31-freeze.txt` (torch 2.9.1+cu130, Triton 3.5.1), since other versions can change fp32 reduction
order and so individual codes.
