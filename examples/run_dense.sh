#!/bin/bash
# End-to-end refinement of a model llama.cpp runs with its layers in VRAM: BF16 GGUF -> imatrix -> leak-free corpus
# -> Gram capture -> KLD base -> stock quant -> refined quant(s) -> KLD report. Every stage checks for its artifact and
# skips when it exists, so a rerun resumes; see docs/runbook.md.
#
# required env: HF (HF model dir), WORK (output dir), TYPE (llama-quantize type, e.g. IQ3_XXS), CAL (imatrix text),
#               SPEC (ggufrefine.corpus spec) plus the variables the spec's paths use
# optional env: PY (python3), GPUS (all), SEEDS ("0 1"), EXTRA_BASES ("name=text.txt ..." extra KLD bases from text),
#               N_CTX (8192), QUANT_ARGS (extra llama-quantize args, e.g. --tensor-type for tensors the imatrix does
#               not reach such as an MTP layer), STOP_AFTER (a stage name),
#               LOGS (logs/<model name> in the repo), UB (N_CTX: ubatch of capture, KLD base and KLD eval alike),
#               KLD_EXTRA (extra llama-kld-swap args), REPORT_WITHOUT (domains to also leave out of an all-domain row)
# The KLD base and the evals must use the same ubatch: on Qwen3.5 a base written at llama-perplexity's default -ub 512
# differs from -ub 4096 (and from fp32 transformers) by up to 0.7 nats/token on long tool trajectories.
# Models with a final-logit softcap and a large vocabulary (gemma-4: 262k) need UB=4096: the softcap kernel fails past
# 2^31 logits per ubatch.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
cd "$here"
: "${HF:?}" "${WORK:?}" "${TYPE:?}" "${CAL:?}" "${SPEC:?}"
PY=${PY:-python3}
B=native/llama.cpp/build/bin
GPUS=${GPUS:-$(nvidia-smi -L | wc -l)}
SEEDS=${SEEDS:-0 1}
N_CTX=${N_CTX:-8192}
UB=${UB:-$N_CTX}
NAME=$(basename "$HF")
M=$WORK/$NAME-BF16.gguf
L=${LOGS:-$here/logs/$NAME}
mkdir -p "$WORK" "$L"
log() { echo "[$(date +%H:%M:%S)] $*"; }
stop() { [ "${STOP_AFTER:-}" = "$1" ] && { log "stopping after $1"; exit 0; } || true; }
gpu_wait() { local rc=0; for p in "$@"; do wait "$p" || rc=1; done; return $rc; }
fail() { log "$1 FAILED (log: $2)"; exit 1; }

# 1. reference GGUF
if [ ! -s "$M" ]; then
    log "convert"
    $PY native/llama.cpp/convert_hf_to_gguf.py "$HF" --outtype bf16 --outfile "$M.tmp" > "$L/convert.log" 2>&1 || fail convert "$L/convert.log"
    mv "$M.tmp" "$M"
fi
stop convert

# 2. imatrix (the stock quant's, and the scales the refinement keeps)
IM=$WORK/imatrix.gguf
if [ ! -s "$IM" ]; then
    log "imatrix"
    CUDA_VISIBLE_DEVICES=0 $B/llama-imatrix -m "$M" -f "$CAL" -c "$N_CTX" -b "$N_CTX" -ngl 99 -fa on -o "$IM" > "$L/imatrix.log" 2>&1 \
        || fail imatrix "$L/imatrix.log"
fi
stop imatrix

# 3. corpus: train / test / eval, no held-out 64-token run in train or in the exclude files
C=$WORK/corpus
if [ ! -s "$C/manifest.json" ]; then
    log "corpus"
    $PY -u -m ggufrefine.corpus --spec "$SPEC" --tokenizer "$HF" --gguf "$M" --out "$C" > "$L/corpus.log" 2>&1 || fail corpus "$L/corpus.log"
fi
stop corpus

# 4. Grams: train sequences over GPUs 0..GPUS-2, test on the last GPU (part 100), then merge into train / test splits
G=$WORK/grams
if [ ! -s "$G/index.json" ]; then
    log "capture"
    LL=(-m "$M" -c $((8 * N_CTX)) -b "$N_CTX" -ub "$UB" -np 8 --kv-unified -fa on -ngl 99)
    ntr=$($PY -c "import numpy as np; print(np.load('$C/train.tokens.npy', mmap_mode='r').shape[0])")
    k=$((GPUS - 1)); pids=()
    for i in $(seq 0 $((k - 1))); do
        a=$((ntr * i / k)); b=$((ntr * (i + 1) / k)); d=$WORK/grams-src/part$(printf %03d $i)
        [ -e "$d.done" ] && continue
        CUDA_VISIBLE_DEVICES=$i $PY -u -m ggufrefine.capture.ggml --tokens "$C/train.tokens.npy" --seqs $a:$b --part-id $i \
            --out "$d" -- "${LL[@]}" > "$L/capture-part$i.log" 2>&1 && touch "$d.done" &
        pids+=($!)
    done
    d=$WORK/grams-src/part100
    if [ ! -e "$d.done" ]; then
        CUDA_VISIBLE_DEVICES=$k $PY -u -m ggufrefine.capture.ggml --tokens "$C/test.tokens.npy" --part-id 100 \
            --out "$d" -- "${LL[@]}" > "$L/capture-test.log" 2>&1 && touch "$d.done" &
        pids+=($!)
    fi
    gpu_wait "${pids[@]}" || fail capture "$L/capture-*.log"
    $PY -u -m ggufrefine.capture.merge_grams --src "$WORK/grams-src" --out "$G" --test-parts 100 > "$L/merge.log" 2>&1 || fail merge "$L/merge.log"
fi
stop capture

# 5. KLD bases: the eval split (exact sequences, chat markup as tokens) and any extra text bases
EB=$WORK/base-eval.kld
if [ ! -s "$EB" ]; then
    log "kld base"
    $PY -c "import numpy as np; np.load('$C/eval.tokens.npy').astype('<i4').tofile('$C/eval.tokens.bin')"
    CUDA_VISIBLE_DEVICES=0 $B/llama-perplexity -m "$M" --tokens-file "$C/eval.tokens.bin" -c "$N_CTX" -b "$N_CTX" -ub "$UB" -ngl 99 -fa on \
        --kl-divergence-base "$EB.tmp" > "$L/base-eval.log" 2>&1 || fail "kld base" "$L/base-eval.log"
    mv "$EB.tmp" "$EB"
fi
BASES=(--base "eval=$EB")
i=1
for nb in ${EXTRA_BASES:-}; do
    n=${nb%%=*}; f=${nb#*=}; xb=$WORK/base-$n.kld
    if [ ! -s "$xb" ]; then
        CUDA_VISIBLE_DEVICES=$i $B/llama-perplexity -m "$M" -f "$f" -c "$N_CTX" -b "$N_CTX" -ub "$UB" --chunks 10 -ngl 99 -fa on \
            --kl-divergence-base "$xb.tmp" > "$L/base-$n.log" 2>&1 || fail "kld base $n" "$L/base-$n.log"
        mv "$xb.tmp" "$xb"
    fi
    BASES+=(--base "$n=$xb"); i=$((i + 1))
done
stop base

# 6. stock quant (llama-quantize preset + imatrix)
S=$WORK/$NAME-$TYPE-stock.gguf
if [ ! -s "$S" ]; then
    log "quantize $TYPE"
    # shellcheck disable=SC2086
    $B/llama-quantize --imatrix "$IM" ${QUANT_ARGS:-} "$M" "$S.tmp" "$TYPE" "$(nproc)" > "$L/quantize.log" 2>&1 || fail quantize "$L/quantize.log"
    mv "$S.tmp" "$S"
fi
stop quantize

# 7. refine a copy of the stock file per seed, layers sharded over the GPUs (each tensor keeps its stock type)
nl=$($PY -c "import json; print(len([l for l in json.load(open('$G/index.json'))['layers'] if str(l).isdigit()]))")
for seed in $SEEDS; do
    R=$WORK/$NAME-$TYPE-refined-s$seed.gguf
    [ -e "$R.done" ] && continue
    log "refine seed $seed ($nl layers on $GPUS GPUs)"
    [ -s "$R" ] || cp --reflink=auto "$S" "$R"
    pids=()
    for i in $(seq 0 $((GPUS - 1))); do
        a=$((nl * i / GPUS)); b=$((nl * (i + 1) / GPUS - 1))
        [ $b -lt $a ] && continue
        CUDA_VISIBLE_DEVICES=$i $PY -u -m ggufrefine.models.dense --gguf "$M" --grams "$G" --imatrix "$IM" --seed "$seed" \
            --layers $a-$b --write-gguf "$R" --results "$L/refine-s$seed-g$i.jsonl" > "$L/refine-s$seed-g$i.log" 2>&1 &
        pids+=($!)
    done
    gpu_wait "${pids[@]}" || fail "refine seed $seed" "$L/refine-s$seed-g*.log"
    touch "$R.done"
done
stop refine

# 8. KLD of each file against every base (one GPU each), then the report. The BF16 file against its own base is the
#    null: it must be ~0 on every domain, or the base and the evals do not compute the same thing.
pids=(); i=0
for f in stock $(for s in $SEEDS; do echo refined-s$s; done) bf16; do
    X=$WORK/$NAME-$TYPE-$f.gguf
    [ $f = bf16 ] && X=$M
    CUDA_VISIBLE_DEVICES=$((i % GPUS)) $PY -u -m ggufrefine.eval.kld --gguf "$X" --out "$WORK/kld/$TYPE-$f" "${BASES[@]}" --extra="-ub $UB ${KLD_EXTRA:-}" \
        > "$L/kld-$TYPE-$f.log" 2>&1 &
    pids+=($!); i=$((i + 1))
done
gpu_wait "${pids[@]}" || fail kld "$L/kld-*.log"
runs=(stock="$WORK/kld/$TYPE-stock"); for s in $SEEDS; do runs+=(refined-s$s="$WORK/kld/$TYPE-refined-s$s"); done
runs+=(bf16="$WORK/kld/$TYPE-bf16")
# shellcheck disable=SC2046
$PY -m ggufrefine.eval.report --base eval --manifest "$C/manifest.json" $(for w in ${REPORT_WITHOUT:-}; do echo --without $w; done) \
    "${runs[@]}" | tee "$WORK/report-$TYPE-eval.txt"
for nb in ${EXTRA_BASES:-}; do
    $PY -m ggufrefine.eval.report --base "${nb%%=*}" "${runs[@]}" | tee "$WORK/report-$TYPE-${nb%%=*}.txt"
done
