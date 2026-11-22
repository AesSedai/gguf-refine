# Runbook: refine a model end to end

From a Hugging Face checkpoint to a stock-loadable refined GGUF and a KLD comparison against the stock quant of the
same recipe. `examples/run_dense.sh` runs every stage below; each stage skips itself when its artifact exists, so a
rerun after a failure resumes. This covers models that llama.cpp runs with their layers in VRAM; Kimi-K3.1 (a
reference larger than VRAM + RAM) uses the layer-major PyTorch capture in `ggufrefine/models/kimi_k3/` instead.

## 0. Setup

    native/build.sh                     # llama.cpp at native/LLAMA_COMMIT + native/patches, CUDA build of every tool
    python -m venv .venv && .venv/bin/pip install -e '.[kimi]'
    .venv/bin/python -m unittest discover -s tests -p 'test_*.py'          # CPU
    .venv/bin/python -m unittest discover -s tests/gpu -p 'test_*.py'      # one GPU

After changing a patch, rebuild with `native/build.sh` (all targets), not one tool: `libllama-common.so` is shared,
and a tool linked against an older layout of `common_params` crashes at start (exit 139, empty log).

## 1. Inputs

| Variable | What |
|---|---|
| `HF` | the Hugging Face model directory (weights for conversion; tokenizer and chat template for the corpus) |
| `WORK` | output directory (needs about 4x the BF16 size: BF16, stock, one refined copy per seed, Grams) |
| `TYPE` | llama-quantize type of the stock file, e.g. `IQ3_XXS`; every tensor the preset puts in a refinable IQ type (IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S / IQ1_S / IQ1_M) is refined, the rest are left as quantized |
| `CAL` | imatrix calibration text (e.g. `calibration_datav5.txt`) |
| `SPEC` | corpus spec, e.g. `examples/corpus-mix.json`, plus the variables its paths use (`DATASETS`, `KLD`, `PERSONAL_RP`, `HELDOUT_RP`) |
| `PY` | the venv's python (default `python3`) |
| optional | `GPUS`, `SEEDS` (default `0 1`), `EXTRA_BASES="combined=/path/combined_all_small.txt"`, `UB` (ubatch for capture, KLD base and evals; default 8192), `KLD_EXTRA`, `QUANT_ARGS`, `REPORT_WITHOUT`, `STOP_AFTER=<stage>`, `LOGS` |

    HF=/models/Qwen3.5-9B WORK=/data/qwen35-9b TYPE=IQ3_XXS CAL=/data/calibration_datav5.txt \
    SPEC=examples/corpus-mix.json DATASETS=/data/datasets KLD=/data PERSONAL_RP=/data/rp HELDOUT_RP=/data/rp-heldout \
    PY=.venv/bin/python EXTRA_BASES="combined=/data/combined_all_small.txt" UB=4096 \
    QUANT_ARGS='--tensor-type blk\.32\.=q8_0' REPORT_WITHOUT=agent_swe ./examples/run_dense.sh

Logs go to `logs/<model>/` in the repo (one per stage and GPU); `tail -f` them while a stage runs.

## 2. Stages

| # | Stage | Artifact | Command (what the script runs) |
|---|---|---|---|
| 1 | convert | `$WORK/<model>-BF16.gguf` | `convert_hf_to_gguf.py $HF --outtype bf16` |
| 2 | imatrix | `$WORK/imatrix.gguf` | `llama-imatrix -f $CAL -c 8192 -b 8192` on the BF16 file |
| 3 | corpus | `$WORK/corpus/{train,test,eval}.tokens.npy`, `manifest.json` | `python -m ggufrefine.corpus --spec $SPEC --tokenizer $HF --gguf <bf16>` |
| 4 | capture | `$WORK/grams/{train,test}/layer-*/` | `python -m ggufrefine.capture.ggml` per GPU (train sequences sharded over GPUs 0..n-2, test on the last), then `capture.merge_grams --test-parts 100` |
| 5 | KLD bases | `$WORK/base-eval.kld` (+ `base-<name>.kld`) | `llama-perplexity --tokens-file eval.tokens.bin -ub $UB --kl-divergence-base` (patch 0011) |
| 6 | stock quant | `$WORK/<model>-$TYPE-stock.gguf` | `llama-quantize --imatrix imatrix.gguf <bf16> <out> $TYPE` |
| 7 | refine | `$WORK/<model>-$TYPE-refined-s<seed>.gguf` | copy of the stock file; `python -m ggufrefine.models.dense --seed S --layers a-b --write-gguf <copy>`, layers sharded over GPUs |
| 8 | KLD + report | `$WORK/kld/<label>/`, `$WORK/report-$TYPE-*.txt` | `python -m ggufrefine.eval.kld` per file (stock, each refined seed, and the BF16 file as the null), `python -m ggufrefine.eval.report` |

What each stage keeps fixed, so that stock and refined differ only in the refined codes: both come from the same
imatrix and preset, the refinement keeps the stock scales and types and only rewrites codewords (in place, verified
by read-back), and both are scored against the same base with the same settings.

## 3. The corpus and its held-out splits

`ggufrefine.corpus` builds three splits from one spec: `train` (the Hessians), `test` (held-out Grams: lambda
selection and held-out error in the refine logs) and `eval` (the KLD base). Held-out data is clean in three ways:

- **By group, before packing.** A group is what must not straddle the split: a role-play character (SillyTavern
  directory, trailing digits dropped, so `Dave`, `Dave1` and every branch of a chat are one group), a shared system
  prompt / character card (`"group": "{conversations[0][value]}"`), by default the document. A group goes to one
  split; when a held-out split is full, its group's further documents are dropped, never moved to train.
- **By content.** Every held-out token inside a 64-token run that also occurs in train or in an `exclude` file (the
  imatrix text) is cut out. The manifest reports the cut fraction per domain (shared tool-prompt templates lose
  ~20-30%, most domains under 4%) and the residue after packing (`exposed_tokens_after_cut`, junction matches only).
- **Eval-only sources.** `"splits": ["eval"]` keeps a source out of train and test entirely (`heldout_rp`).

Each sequence holds one domain, so the report can split KLD by domain; `min_heldout` guarantees a domain test / eval
sequences (personal RP gets at least 3 of each). The eval base is built from the token file, not re-tokenized text:
chat markup stays as special tokens, and chunk i is eval sequence i. Chat domains are rendered with the model's own
template; thinking mode (`chat_kwargs`) only where the source has reasoning.

Writing a spec: one entry per source, `format` one of `text`, `chat`, `sharegpt`, `messages`, `sillytavern`,
`text_file` (fields in `ggufrefine/corpus.py`'s docstring). Smoke-test a new spec by building with small `sequences`
first; a source that runs out raises `source exhausted` (raise its `rows_per_file`).

## 4. Reading the report

`report-$TYPE-eval.txt`: per subset (all, `all_rp` = every domain ending in "rp", `all_without_<D>`, then each domain)
mean KLD, PPL ratio and same-top for each file, then each file minus stock with a chunk-bootstrap 95% interval.
Check the `bf16` column first: the reference against its own base must be ~0 KLD in every row, otherwise the base and
the evals did not compute the same thing (see the ubatch note below). PPLx is computed from the base file's log-probs,
which llama-perplexity clamps to 16 nats below the top logit, so it is biased upward on content with many very
improbable tokens (the `bf16` row shows by how much); KLD drops only terms with p < e^-16 and is not affected. Chunks are the
independent unit; per-domain rows with one or two chunks are indicative only. Compare refined seeds with each other
before reading a variant difference: the refinement's visit-order seed alone moves KLD by up to ~10% on gemma-3-12b
IQ2_XS. `report-$TYPE-combined.txt` is the same on an extra text base; combined_all_small shares passages with
calibration_datav5 (the imatrix text), so it is not held out.

## 5. Model-specific notes

- **Same ubatch for base and evals.** On Qwen3.5-9B a base written at llama-perplexity's default `-ub 512` disagrees
  with `-ub 4096` by up to 0.7 nats/token on long tool-call trajectories (fp32 transformers agrees with 4096), while
  other domains agree to KLD 1e-4. `UB` sets one ubatch for capture, base and evals; the BF16 null checks it.
- **Tensors the imatrix does not reach** (an MTP / nextn layer, `blk.32` on Qwen3.5-9B) make llama-quantize refuse
  low-bit types; give them a type with `QUANT_ARGS='--tensor-type blk\.32\.=q8_0'`. They are not refined (no Grams).

- **Final-logit softcap with a large vocabulary** (gemma-4, 262k vocab): the CUDA softcap kernel fails past 2^31
  logits per ubatch (`invalid argument` in `ggml_cuda_op_softcap`). Set `UB=4096`.
- **BOS** follows the GGUF (`tokenizer.ggml.add_bos_token`), not the HF config: the corpus puts BOS at the start of
  every sequence when llama.cpp would, as llama-imatrix and llama-perplexity do per chunk.
- **Chat-template-sensitive models** (gemma): PPL / KLD on text with chat markup typed as plain text measures
  something the model never sees in serving; use the token-file eval base (stage 5), which this pipeline does.
- **Tool calls**: `messages` sources with JSON-string `arguments` are parsed to mappings (templates iterate them).

## 6. Measured run: Qwen3.5-9B, IQ3_XXS (2026-11-22, 8 x 96 GB)

Stock: `llama-quantize` IQ3_XXS preset with the calibration_datav5 imatrix, MTP layer Q8_0: 4.06 GiB, 3.70 BPW
(ffn and ssm_out / ssm_beta_alpha IQ3_XXS, attn_q / attn_k IQ2_S, attn_output IQ3_S: refined; attn_qkv_gate /
attn_v Q4_K, output Q5_K, token_embd IQ3_S: unchanged). Refined: the same file with refined codes, two seeds.
Corpus: 305 x 8192 train, 24 test, 32 eval sequences; eval RP = 4 personal-RP sequences (21 characters held out of
train and test), 2 public RP, 3 from an eval-only RP set (8 characters).

| subset | chunks | stock KLD | refined s0 | refined s1 | BF16 null | refined vs stock (s0, s1) |
|---|---:|---:|---:|---:|---:|---|
| all | 32 | 0.5440 | 0.4597 | 0.4563 | 0.0000 | -15.5%, -16.1% |
| all without agent_swe | 27 | 0.1101 | 0.0735 | 0.0744 | 0.0000 | -33.2%, -32.4% |
| all RP | 9 | 0.1495 | 0.0880 | 0.0896 | 0.0000 | -41.2%, -40.1% |
| personal RP | 4 | 0.1353 | 0.0767 | 0.0763 | 0.0000 | -43.3%, -43.6% |
| agent_swe | 5 | 2.8871 | 2.5455 | 2.5187 | 0.0000 | -11.8%, -12.8% |
| code | 5 | 0.0652 | 0.0466 | 0.0461 | 0.0000 | -28.5%, -29.2% |
| combined_all_small (not held out) | 10 | 0.1395 | 0.1176 | 0.1156 | 0.0000 | -15.7%, -17.1% |

Every domain improves on both seeds (full table: `report-IQ3_XXS-eval.txt`). agent_swe (mid-trajectory windows of
SWE agent runs) has BF16 NLL 3-5 nats/token (fp32 transformers agrees) and 20x the KLD of other domains, so it
dominates the all-domain mean.

Wall time (8 GPUs): convert 2.5 min, imatrix 4 min, corpus 1 min, capture 16 min (2.5 min of forward, the rest
writing Grams) + merge 3 min, KLD bases 9 min, quantize 0.5 min, refine 2 min per seed, KLD 2 min.
