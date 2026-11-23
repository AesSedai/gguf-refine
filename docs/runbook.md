# Runbook: refine a model end to end

From a Hugging Face checkpoint to a stock-loadable refined GGUF and a KLD comparison against the stock quant of the
same recipe. `examples/run_dense.sh` runs every stage below. This covers models that llama.cpp runs with their layers
in VRAM of one device; Kimi-K3.1 (a reference larger than VRAM + RAM) uses the layer-major PyTorch capture in
`ggufrefine/models/kimi_k3/` instead.

## 0. Setup

    native/build.sh                     # llama.cpp at native/LLAMA_COMMIT + native/patches, CUDA build of every tool
    python -m venv .venv && .venv/bin/pip install -e .
    .venv/bin/python -m unittest discover -s tests -p 'test_*.py'          # CPU
    .venv/bin/python -m unittest discover -s tests/gpu -p 'test_*.py'      # one GPU

Python 3.12 or later. After changing or adding a patch, rebuild from a fresh tree with every target:
`rm -rf native/llama.cpp && native/build.sh` (build.sh refuses a tree patched from another patch set; building one
tool alone leaves the others linked against an older shared `libllama-common.so`, and they crash at start with exit
139 and an empty log). The shell script uses `native/llama.cpp/build/bin`; do not point `GGUF_REFINE_BIN` elsewhere
for a pipeline run.

## 1. Inputs

| Variable | What |
|---|---|
| `HF` | the Hugging Face model directory (weights for conversion; tokenizer and chat template for the corpus) |
| `WORK` | output directory, one per model and recipe (disk: see below) |
| `TYPE` | llama-quantize type of the stock file, e.g. `IQ3_XXS`; every tensor the preset puts in a refinable IQ type (IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S / IQ1_S / IQ1_M) and that has captured Grams is refined, the rest are left as quantized (the refine log lists refinable tensors without Grams, e.g. `token_embd`) |
| `CAL` | imatrix calibration text (e.g. `calibration_datav5.txt`) |
| `SPEC` | corpus spec, e.g. `examples/corpus-mix.json`, plus the variables its paths use (`DATASETS`, `KLD`, `PERSONAL_RP`, `HELDOUT_RP` for that one) |
| `PY` | the venv's python (default `python3`) |
| `GPUS` | CUDA device ids, comma-separated, distinct (default `0`). Each job runs on one device; more devices run jobs side by side (llama-quantize uses all of them). Do not set `CUDA_VISIBLE_DEVICES` (preflight refuses it) |
| `N_CTX` | sequence length; default the spec's `n_ctx`. Corpus, imatrix chunks, capture, KLD base and evals all follow it |
| `UB` | ubatch of capture, KLD base and evals alike; default `N_CTX` |
| `CAPTURE_SEQS` | sequences per capture decode (default 8; the capture context is `CAPTURE_SEQS x N_CTX`); lower it when a capture job does not fit a device |
| optional | `SEEDS` (default `0 1`; distinct integers), `QUANT_ARGS` (extra llama-quantize args), `EXTRA_BASES="combined=/path/combined_all_small.txt"` (name: letters, digits, `_`, `-`; first 10 chunks of each text), `KLD_EXTRA` (llama-kld-swap args other than the model, -c, -b, -ub), `REPORT_WITHOUT="agent_swe"`, `REPORT_POOLS="all_rp=*rp"`, `STOP_AFTER`, `LOGS` (default `logs/<WORK name>/`) |

`STOP_AFTER` takes one stage name from the table in section 2: `convert imatrix corpus capture base quantize refine`.
List settings (`SEEDS`, `EXTRA_BASES`, `REPORT_*`) are split on whitespace; paths in them cannot contain spaces.

One device, the measured Qwen3.5-9B run's settings:

    HF=/models/Qwen3.5-9B WORK=/data/qwen35-9b TYPE=IQ3_XXS CAL=/data/calibration_datav5.txt \
    SPEC=examples/corpus-mix.json DATASETS=/data/datasets KLD=/data PERSONAL_RP=/data/rp HELDOUT_RP=/data/rp-heldout \
    PY=.venv/bin/python UB=4096 QUANT_ARGS='--tensor-type blk\.32\.=q8_0' \
    REPORT_WITHOUT=agent_swe REPORT_POOLS='all_rp=*rp' EXTRA_BASES="combined=/data/combined_all_small.txt" \
    ./examples/run_dense.sh

`GPUS=0,1,2,3,4,5,6,7` runs the same on eight devices (section 6 has both wall times). `QUANT_ARGS` here is
Qwen3.5's (section 5); for another model run with `STOP_AFTER=imatrix` first and read the imatrix log for tensors it
saw no data for. `REPORT_*` name domains of the example spec; `EXTRA_BASES` adds a second, not held-out base.

The script starts with a preflight that checks, before any stage: the settings above, the devices, the tools in
`native/llama.cpp/build/bin`, the Python packages, and, until the corpus is built, the corpus spec
(`python -m ggufrefine.corpus --check`: every source glob matches files, every `$VAR` is set, switches are booleans,
each split's allocation adds up and the `min_heldout` minima fit; it prints the sequences per split and domain, and
notes when `CAL` is not among the spec's `exclude` files). It takes a lock on `$WORK/.run.lock`: a second run on the same WORK, or workers
left behind by a killed run, make it refuse to start (`fuser -v $WORK/.run.lock` lists them).

Logs go to `logs/<WORK name>/` in the repo: `run.log` (the script's own output, appended across runs) and one per
stage and job (overwritten when the stage reruns); `tail -f` them while a stage runs. Refine journals live next to
the refined file, not in the logs.

### Hardware and disk (estimates; the measured Qwen3.5-9B run in section 6)

- Per device: a capture job holds the whole BF16 model, the KV cache of `CAPTURE_SEQS` sequences of `N_CTX`, and an
  fp64 Gram accumulator per input group (K x K x 8 bytes). Qwen3.5-9B at n_ctx 8192, 8 sequences: 75 GB measured.
- Disk in WORK (decimal GB): the BF16 file (B); per capture part (each train part and the test part) the full Gram
  set, G (about 0.7 x B for Qwen3.5-9B), merged into train and test (2 G) and removed once merged; the eval base,
  `(N_CTX / 2) x n_vocab x 2` bytes per eval sequence (2.03 GB for a 248k vocabulary at 8192), and 10 such chunks per
  extra base; the stock file and one copy per seed. Peak during capture: B + (parts + 1) x G. Capture also keeps
  8 GB free beyond the layer it writes, and stops with the free / needed bytes when it cannot.

## 2. Stages

| # | Stage (`STOP_AFTER`) | Artifact | Command (what the script runs) |
|---|---|---|---|
| 1 | `convert` | `$WORK/<model>-BF16.gguf` | `convert_hf_to_gguf.py $HF --outtype bf16` |
| 2 | `imatrix` | `$WORK/imatrix.gguf` | `llama-imatrix -f $CAL -c N_CTX -b N_CTX` on the BF16 file |
| 3 | `corpus` | `$WORK/corpus/{train,test,eval}.tokens.npy`, `manifest.json` | `python -m ggufrefine.corpus --spec $SPEC --tokenizer $HF --gguf <bf16>` |
| 4 | `capture` | `$WORK/grams/{train,test}/layer-*/`, `index.json` | `python -m ggufrefine.capture.ggml` per part (train in one part per device but the last, at least 1, at most one per sequence; test as part 100), then `capture.merge_grams --test-parts 100 --delete --require-parts ...` |
| 5 | `base` | `$WORK/base-eval.kld` (+ `base-<name>.kld`) | `llama-perplexity --tokens-file eval.tokens.bin -ub $UB --kl-divergence-base` (patch 0011) |
| 6 | `quantize` | `$WORK/<model>-$TYPE-stock.gguf` | `llama-quantize --imatrix imatrix.gguf $QUANT_ARGS <bf16> <out> $TYPE` |
| 7 | `refine` | `$WORK/<model>-$TYPE-refined-s<seed>.gguf` (+ `.journal/`, `.done`) | copy of the stock file; `python -m ggufrefine.models.dense --stock <stock> --seed S --layers a-b --write-gguf <copy>` in 8 layer shards; fails when no tensor was refined |
| 8 | (KLD + report) | `$WORK/kld/$TYPE-<label>/`, `$WORK/report-$TYPE-<base>.txt` | `python -m ggufrefine.eval.kld` per file (stock, each refined seed, and the BF16 file as the null), `python -m ggufrefine.eval.report --null bf16`; a failed report is written to `report-$TYPE-<base>.FAILED.txt` instead |

What each stage keeps fixed, so that stock and refined differ only in the refined codes: both come from the same
imatrix and preset; the refinement reads the stock codes and types from the untouched stock file, keeps the stock
scales and types and only rewrites codewords in the copy (in place, verified by read-back; it refuses a copy whose
tensor layout differs from the stock file); both are scored against the same base with the same settings.

Jobs run one per device at a time (`GPUS`): capture parts, bases, refine shards and evals each spread over the
devices; with one device they run in turn.

### Resuming

Every stage skips itself when its artifact exists; artifacts are written under a temporary name and renamed when
complete, so a killed stage leaves nothing that reads as done. Inside a stage:

- capture: finished parts have a `grams-src/partNNN.done` marker and are not rerun. The train partition is fixed in
  `grams-src/plan.txt` at the first launch (checked to cover the train sequences exactly), so a resume with a
  different `GPUS` runs the same parts. The merge refuses a part that appears twice and writes `index.json` only
  after checking that every layer holds exactly the planned parts.
- refine: a seed's copy and its `.journal/` (layers done) resume together; deleting the copy deletes the journal, its
  `.done` marker and the copy's KLD results on the next run.
- KLD: an eval result records the GGUF and base files (path, size, mtime) and settings it was made with; a result for
  other files or settings is an error, not a skip (remove that result directory). A newly written base or stock file
  removes the results against it. When every result is in place, no model is loaded.

Nothing checks that an existing artifact was made with the current settings. After changing one, remove what it
affects:

| Changed | Remove |
|---|---|
| `HF` checkpoint, tokenizer | a new `WORK` |
| `CAL` | `imatrix.gguf`, the stock file, refined copies; and, if `CAL` is also in the spec (`exclude`, a train source), as for `SPEC` |
| `SPEC` | `corpus/`, `grams*`, `base-*.kld`, refined copies, `kld/` |
| `N_CTX` | as for `SPEC` (the script refuses a corpus of another n_ctx), and as for `CAL` (the imatrix is built in N_CTX chunks) |
| `UB` | `grams*`, `base-*.kld`, refined copies, `kld/` |
| `CAPTURE_SEQS` | `grams*`, refined copies |
| `TYPE` | nothing (files are named by type) |
| `QUANT_ARGS` | the stock file, refined copies |

"Refined copies" means each `<model>-$TYPE-refined-s*.gguf` with its `.journal/` and `.done`.

## 3. The corpus and its held-out splits

`ggufrefine.corpus` builds three splits from one spec: `train` (the Hessians), `test` (held-out Grams: the held-out
reconstruction error in the refine journals) and `eval` (the KLD base). A spec is a list of sources ("domains"); a
domain is a label for allocation and reporting, nothing depends on particular names or on there being role-play,
characters or several domains. Data is bring-your-own: the corpus builder reads plain text, rows with a text field,
and conversations as lists of role / content messages; converting other exports (chat frontends, custom logs) into
one of those is the operator's job. Held-out data is clean in three ways:

- **By group, before packing.** A group is what must not straddle the split. By default it is the document; a
  domain's `"group"` template names something shared over the row: a system prompt / character card
  (`"group": "{conversations[0][value]}"`), or a field you wrote when converting (`"group": "{group}"`, e.g. the
  character of a role-play chat, so every chat and branch of one character is one group). Group keys are global:
  the same key in two domains is one group, in one split. A group goes to one split; when that split is full, the
  group's further documents are dropped, never moved to train (counted in the manifest).
- **By content.** Every held-out token inside a 64-token run that also occurs in train or in an `exclude` file (the
  imatrix text) is cut out. The manifest reports the cut fraction per domain (shared tool-prompt templates lose
  ~20-30%, most domains under 4%) and the residue after packing (`exposed_tokens_after_cut`: windows across the
  junction of two packed documents).
- **Eval-only sources.** `"splits": ["eval"]` keeps a source out of train and test entirely (`heldout_rp`).

Each sequence holds one domain, so the report can split KLD by domain. `min_heldout` sets a minimum of test / eval
sequences for a domain; preflight fails when the minima do not fit, and the build fails when a held-out split ends
up short after cutting (raise `heldout_margin`, default 1.6, or add held-out data). The manifest counts per domain
and split the groups assigned (`groups`) and the groups with tokens in the packed sequences (`groups_packed`: a
group's documents can all fall past the last packed sequence).

The eval base is built from the token file, not re-tokenized text: chat markup stays as special tokens, and chunk i
is eval sequence i. Chat domains are rendered with the model's own template; thinking mode (`chat_kwargs`) only where
the source has reasoning. BOS: from the GGUF (`tokenizer.ggml.add_bos_token` and `bos_token_id`); when the GGUF has
no such key the spec must say `"bos": true` or `false` (llama.cpp would use a per-vocab default).

Formats are by what the chat template renders, not by where the data came from:

| `format` | rendered as | a row is |
|---|---|---|
| `text_file` | raw text, no template | one text file, cut into `n_ctx`-token documents |
| `text` | raw text, no template | a `"text"` template over the row's fields |
| `chat` | the model's chat template; system / user / assistant turns, optional `reasoning_content` | `"messages"`: the row field holding the message list, or `"turns"`: message templates over the row |
| `tool_chat` | as `chat`, plus assistant `tool_calls`, `tool` turns and the row's tool definitions (`"tools"`: a field) | as `chat` |

A tool call or tool turn in a `chat` domain is an error, so tool-calling data is always rendered with the
template's tool markup. Tool calls are not validated against the tool definitions (nothing is executed); their
arguments must be a JSON object, which the templates need to render them.

Rows are strict: a row that lacks a field its templates use, a message with an unknown role, a tool call in a
`chat` domain, tool-call arguments that are not a JSON object, or a conversation the chat template rejects stops the
build with the domain, the row id and the reason. A source with a few malformed rows can set `"skip_bad_rows": true`
on its domain; skipped rows are counted (`bad_rows_skipped`) and the first ones logged. A null field renders as an empty string (a missing one is an error);
a turn left with no content, reasoning or tool call is dropped, and consecutive turns of one role are joined.
Messages use `role` / `content` keys or ShareGPT `from` / `value`; roles `system`, `human`/`user`,
`gpt`/`assistant`/`model`, `function_call` (assistant), `observation`/`tool` (tool); `"roles": {"speaker": "user"}`
maps others.

Writing a spec: one entry per source, `format` one of `text_file`, `text`, `chat`, `tool_chat` (fields in
`ggufrefine/corpus.py`'s docstring). `python -m ggufrefine.corpus --spec S --check` checks
files and allocation without a tokenizer; a build into a scratch directory with small `sequences` checks rendering
(look at a few documents of each chat domain in `eval.txt`). A source that runs out raises `source exhausted` with
its group, bad-row and short-document counts: raise its `rows_per_file` (parquet / JSONL sources), add files or
groups, or lower its weight.

## 4. Reading the report

`report-$TYPE-eval.txt`: per subset (all, the `REPORT_POOLS` rows, `all_without_<D>` for each `REPORT_WITHOUT`, then
each domain) mean KLD, PPL ratio and same-top for each file, then each file minus stock: the absolute KLD difference
with a chunk-bootstrap 95% interval, and in parentheses the difference relative to stock.

The `bf16` run (the reference against its own base) must be ~0 KLD on every row; the report exits with an error
(`NULL CHECK FAILED`, after printing the table) when it exceeds 1e-3 on any row, and the script stops. That means the
base and the evals did not compute the same thing (ubatch, context, model file or build), and the differences between
the other runs are not meaningful. For reference: a base at `-ub 512` scored at `-ub 4096` on Qwen3.5-9B gave a null
of 0.0062 overall, 0.038 on agent_swe; with matching ubatch it is 3e-8.

KLD is reference-to-model, in nats per scored token (the second half of each chunk: 4095 tokens per 8192 chunk),
summed over the tokens the reference gives log-prob above -16 (the base stores no smaller ones, so the tail beyond
that is left out; compare KLD only between runs scored against the same base). PPLx is computed from the base
file's log-probs, which llama-perplexity clamps to 16 nats below the top logit, so it is biased upward on content
with many very improbable tokens (the `bf16` row shows by how much). Chunks are the independent unit; per-domain
rows with one or two chunks are indicative only. Compare refined seeds with each other before reading a variant
difference: the refinement's visit-order seed alone moves KLD by up to ~10% on gemma-3-12b IQ2_XS.
`report-$TYPE-<name>.txt` is the same on an extra base; combined_all_small shares passages with calibration_datav5
(the imatrix text), so it is not held out.

## 5. Model-specific notes

- **Same ubatch for base and evals.** On Qwen3.5-9B a base written at llama-perplexity's default `-ub 512` disagrees
  with `-ub 4096` by up to 0.7 nats/token on long tool-call trajectories (fp32 transformers agrees with 4096), while
  other domains agree to KLD 1e-4. `UB` sets one ubatch for capture, base and evals; the null check catches a mismatch.
- **Tensors the imatrix does not reach** (an MTP / nextn layer, `blk.32` on Qwen3.5-9B) make llama-quantize refuse
  low-bit types; give them a type with `QUANT_ARGS='--tensor-type blk\.32\.=q8_0'` (the layer index is the model's:
  the imatrix log lists the tensors it saw no data for, `STOP_AFTER=imatrix` stops there to look). They are not
  refined (no Grams).
- **Final-logit softcap with a large vocabulary** (gemma-4, 262k vocab): the CUDA softcap kernel fails past 2^31
  logits per ubatch (`invalid argument` in `ggml_cuda_op_softcap`). Set `UB=4096`.
- **Chat-template-sensitive models** (gemma): PPL / KLD on text with chat markup typed as plain text measures
  something the model never sees in serving; use the token-file eval base (stage 5), which this pipeline does.
- **Tool calls**: `tool_chat` sources with JSON-string `arguments` are parsed to mappings (templates iterate them).
- **Stock compatibility**: the refined file has the stock file's layout and types; the pipeline scores it with the
  patched build in `native/`, which does not build `llama-cli`. To check it in an unmodified llama.cpp build, run
  that build's `llama-cli -m <refined> -p hi -n 8`.

## 6. Measured run: Qwen3.5-9B, IQ3_XXS (2026-11-22, 8 x 96 GB)

Stock: `llama-quantize` IQ3_XXS preset with the calibration_datav5 imatrix, MTP layer Q8_0: 4060.85 MiB of tensors
(3.97 GiB), 3.70 BPW (ffn and ssm_out / ssm_beta_alpha IQ3_XXS, attn_q / attn_k IQ2_S, attn_output IQ3_S: refined;
attn_qkv_gate / attn_v Q4_K, output Q5_K: unchanged; token_embd IQ3_S: no Grams, unchanged). Refined: the same file
with refined codes, two seeds. Corpus: 305 x 8192 train, 24 test, 32 eval sequences. Eval RP: 4 personal-RP
sequences (21 character groups assigned to eval, 14 with tokens in them; none in train or test), 2 public RP, 3 from
an eval-only RP set (8 groups assigned, 6 in the packed tokens). The packed-group counts come from rebuilding the
corpus with the current builder, whose token arrays are byte-identical (sha256) to the run's.

| subset | chunks | stock KLD | refined s0 | refined s1 | BF16 null | refined vs stock (s0, s1) |
|---|---:|---:|---:|---:|---:|---|
| all | 32 | 0.5440 | 0.4597 | 0.4563 | 0.0000 | -15.5%, -16.1% |
| all_without_agent_swe | 27 | 0.1101 | 0.0735 | 0.0744 | 0.0000 | -33.2%, -32.4% |
| all_rp | 9 | 0.1495 | 0.0880 | 0.0896 | 0.0000 | -41.2%, -40.1% |
| personal_rp | 4 | 0.1353 | 0.0767 | 0.0763 | 0.0000 | -43.3%, -43.6% |
| agent_swe | 5 | 2.8871 | 2.5455 | 2.5187 | 0.0000 | -11.8%, -12.8% |
| code | 5 | 0.0652 | 0.0466 | 0.0461 | 0.0000 | -28.5%, -29.2% |
| combined_all_small (not held out) | 10 | 0.1395 | 0.1176 | 0.1156 | 0.0000 | -15.7%, -17.1% |

Every domain improves on both seeds (full table: `report-IQ3_XXS-eval.txt`). agent_swe (mid-trajectory windows of
SWE agent runs) has BF16 NLL 3-5 nats/token (fp32 transformers agrees) and 20x the KLD of other domains, so it
dominates the all-domain mean.

Wall time (8 devices): convert 2.5 min, imatrix 4 min, corpus 1 min, capture 16 min (2.5 min of forward, the rest
writing Grams) + merge 3 min, KLD bases 9 min, quantize 0.5 min, refine 2 min per seed, KLD 2 min. Stages 6-8 rerun
on 7 devices with the current script (same corpus, Grams and bases): quantize 0.5 min, refine 3.2 and 4.2 min; KLD on
8 devices 1.8 min; stock and refined seed-0 files byte-identical to the first run, report identical.

One device (GPUS=0), a 16 / 4 / 8-sequence version of the spec, every stage from imatrix on: 28 min end to end
(imatrix 4.5 min, corpus 0.3, capture of one train part and the test part 7 + merge 2, base 1.6, quantize 1, refine
11 for one seed, KLD 1).

Disk (WORK, decimal GB): BF16 18.4; Gram sources 12.9 per part (8 parts: 103, removed once merged by the current
script), merged Grams 25.9; eval base 65.1 (32 chunks of 2.03), combined base 20.3 (10 chunks); stock and each
refined copy 4.3. Peak during capture about 125, final about 143.
