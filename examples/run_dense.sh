#!/bin/bash
# End-to-end refinement of a model llama.cpp runs with its layers in VRAM: BF16 GGUF -> imatrix -> leak-free corpus
# -> Gram capture -> KLD base -> stock quant -> refined quant(s) -> KLD report. See docs/runbook.md.
#
# required env: HF (HF model dir), WORK (output dir), TYPE (llama-quantize type, e.g. IQ3_XXS), CAL (imatrix text),
#               SPEC (ggufrefine.corpus spec) plus the variables the spec's paths use
# optional env: PY (python3), GPUS (CUDA device ids, comma-separated; default 0), SEEDS ("0 1"),
#               N_CTX (the spec's n_ctx), UB (N_CTX: ubatch of capture, KLD base and KLD eval alike),
#               QUANT_ARGS (extra llama-quantize args, e.g. --tensor-type for tensors the imatrix does not reach),
#               EXTRA_BASES ("name=text.txt ...": extra KLD bases, the first 10 chunks of each text),
#               KLD_EXTRA (extra llama-kld-swap args, not -c / -b / -ub), REPORT_WITHOUT (domains), REPORT_POOLS
#               ("all_rp=*rp ...": report rows pooling domains by glob), STOP_AFTER (convert imatrix corpus capture
#               base quantize refine), CAPTURE_SEQS (8: sequences per capture decode), LOGS (logs/<WORK name> in
#               the repo)
# Every stage checks for its artifact and skips when it exists, so a rerun resumes; an artifact is written under a
# temporary name and renamed when complete. Nothing checks that an existing artifact was made with the current
# settings: after changing a setting, remove the artifacts it affects (runbook, "Resuming").
set -euo pipefail
set -f                                                  # no pathname expansion: list settings hold globs (pools)
here=$(cd "$(dirname "$0")/.." && pwd)
cd "$here"
: "${HF:?}" "${WORK:?}" "${TYPE:?}" "${CAL:?}" "${SPEC:?}"
PY=${PY:-python3}
B=native/llama.cpp/build/bin
GPUS=${GPUS:-0}
IFS=, read -ra GPU <<< "$GPUS"
NG=${#GPU[@]}
SEEDS=${SEEDS:-0 1}
NAME=$(basename "$HF")
M=$WORK/$NAME-BF16.gguf
L=${LOGS:-$here/logs/$(basename "$WORK")}
CAPTURE_SEQS=${CAPTURE_SEQS:-8}
STAGES="convert imatrix corpus capture base quantize refine"
mkdir -p "$WORK" "$L"
exec > >(tee -a "$L/run.log") 2>&1
log() { echo "[$(date +%H:%M:%S)] $*"; }
fail() { log "$1 FAILED${2:+ (log: $2)}"; exit 1; }
stop() { if [ "${STOP_AFTER:-}" = "$1" ]; then log "stopping after $1"; exit 0; fi; }

# lanes FUNC N: run FUNC 0 .. FUNC N-1, job j on device GPU[j % NG] (CUDA_VISIBLE_DEVICES set), one job per device at
# a time; fails if any job failed. The jobs run with errexit off (bash disables it under ||): each returns the
# status of its last command or returns 1 explicitly.
lanes() {
    local fn=$1 n=$2 g pids=() rc=0
    for g in $(seq 0 $((NG - 1))); do
        [ "$g" -lt "$n" ] || break
        (
            for j in $(seq "$g" "$NG" $((n - 1))); do
                CUDA_VISIBLE_DEVICES=${GPU[g]} "$fn" "$j" || { log "$fn $j failed (device ${GPU[g]})"; exit 1; }
            done
        ) &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
    return $rc
}

# 0. preflight: everything that can be checked before the first stage
exec 9>"$WORK/.run.lock"
flock -n 9 || fail "lock: another run_dense.sh on $WORK, or workers a killed one started, still hold $WORK/.run.lock (fuser -v $WORK/.run.lock lists them)"
SPEC_CTX=$($PY -c "import json, sys; print(json.load(open(sys.argv[1])).get('n_ctx') or '')" "$SPEC") || fail "preflight: cannot read SPEC=$SPEC"
N_CTX=${N_CTX:-$SPEC_CTX}
UB=${UB:-$N_CTX}
errs=()
[ -n "$N_CTX" ] || errs+=("no n_ctx in $SPEC and no N_CTX")
ok=; for st in $STAGES; do [ "${STOP_AFTER:-}" = "$st" ] && ok=1; done
[ -z "${STOP_AFTER:-}" ] || [ -n "$ok" ] || errs+=("STOP_AFTER=$STOP_AFTER: not one of $STAGES")
[ -z "${CUDA_VISIBLE_DEVICES+x}" ] || errs+=("CUDA_VISIBLE_DEVICES is set ($CUDA_VISIBLE_DEVICES): unset it and list the device ids in GPUS")
ndev=$(nvidia-smi -L | wc -l)
if [[ ! $GPUS =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    errs+=("GPUS=$GPUS: not a comma-separated list of device ids")
else
    for g in "${GPU[@]}"; do [ "$g" -lt "$ndev" ] || errs+=("GPUS=$GPUS: $g is not a device id 0..$((ndev - 1))"); done
    [ "$(printf '%s\n' "${GPU[@]}" | sort -u | wc -l)" = "$NG" ] || errs+=("GPUS=$GPUS: a device is listed twice")
fi
read -ra SEED <<< "$SEEDS"
[ ${#SEED[@]} -gt 0 ] || errs+=("SEEDS is empty")
for x in "${SEED[@]}"; do [[ $x =~ ^[0-9]+$ ]] || errs+=("SEEDS=$SEEDS: $x is not a non-negative integer"); done
[ "$(printf '%s\n' "${SEED[@]}" | sort -u | wc -l)" = "${#SEED[@]}" ] || errs+=("SEEDS=$SEEDS: a seed is listed twice")
[[ $CAPTURE_SEQS =~ ^[1-9][0-9]*$ ]] || errs+=("CAPTURE_SEQS=$CAPTURE_SEQS: not a positive integer")
[ -f "$HF/config.json" ] || errs+=("HF=$HF: no config.json")
[ -f "$CAL" ] || errs+=("CAL=$CAL: not found")
for t in llama-imatrix llama-perplexity llama-quantize llama-kld-swap libllama-capture.so; do
    [ -e "$B/$t" ] || errs+=("$B/$t missing: run native/build.sh")
done
[ -f native/llama.cpp/convert_hf_to_gguf.py ] || errs+=("native/llama.cpp/convert_hf_to_gguf.py missing: run native/build.sh")
read -ra KX <<< "${KLD_EXTRA:-}"
for x in "${KX[@]}"; do
    case $x in -c|-b|-ub|--ctx-size|--batch-size|--ubatch-size|-m|--model)
        errs+=("KLD_EXTRA sets $x: the model, context and ubatch are the script's (N_CTX / UB)");; esac
done
declare -A seen=([eval]=1)
for nb in ${EXTRA_BASES:-}; do
    n=${nb%%=*}; f=${nb#*=}
    [[ $nb == *=* && $n =~ ^[A-Za-z0-9_-]+$ && -f $f ]] || errs+=("EXTRA_BASES: $nb is not name=existing-text-file (name: letters, digits, _ -)")
    [ -z "${seen[$n]:-}" ] || errs+=("EXTRA_BASES: name $n is used twice or reserved")
    seen[$n]=1
done
pyerr=$($PY - 2>&1 <<'EOF'
import sys
assert sys.version_info >= (3, 12), f"python {sys.version.split()[0]} < 3.12"
import numpy, torch, triton, transformers, jinja2, pyarrow, safetensors, sentencepiece  # noqa
import ggufrefine.paths, gguf  # noqa
assert torch.cuda.is_available(), "torch sees no CUDA device"
EOF
) || errs+=("PY=$PY: ${pyerr##*$'\n'} (pip install -e . in the venv)")
if [ ${#errs[@]} -gt 0 ]; then printf '  %s\n' "${errs[@]}"; fail preflight; fi
if [ ! -s "$WORK/corpus/manifest.json" ]; then           # the sources are needed only to build the corpus
    $PY -m ggufrefine.corpus --spec "$SPEC" --n-ctx "$N_CTX" --check || fail "preflight: corpus spec"
    $PY -c "import json, os, sys; ex = [os.path.realpath(os.path.expandvars(p)) for p in json.load(open(sys.argv[1])).get('exclude', [])]
if os.path.realpath(sys.argv[2]) not in ex: print('note: CAL is not in the spec\'s exclude list: held-out text is not checked against the imatrix text')" "$SPEC" "$CAL"
fi
log "preflight ok: $NAME $TYPE, n_ctx $N_CTX, ubatch $UB, devices ${GPU[*]}, seeds $SEEDS; $(df -h --output=avail "$WORK" | tail -1 | tr -d ' ') free in $WORK"

# 1. reference GGUF
if [ ! -s "$M" ]; then
    log "convert"
    $PY native/llama.cpp/convert_hf_to_gguf.py "$HF" --outtype bf16 --outfile "$M.tmp" > "$L/convert.log" 2>&1 || fail convert "$L/convert.log"
    mv "$M.tmp" "$M"
fi
stop convert

# 2. imatrix (the stock quant's; llama-imatrix saves partial matrices along the way, so it writes to a temporary name)
IM=$WORK/imatrix.gguf
if [ ! -s "$IM" ]; then
    log "imatrix"
    CUDA_VISIBLE_DEVICES=${GPU[0]} $B/llama-imatrix -m "$M" -f "$CAL" -c "$N_CTX" -b "$N_CTX" -ngl 99 -fa on \
        --output-format gguf -o "$WORK/imatrix.tmp.gguf" > "$L/imatrix.log" 2>&1 || fail imatrix "$L/imatrix.log"
    mv "$WORK/imatrix.tmp.gguf" "$IM"
fi
stop imatrix

# 3. corpus: train / test / eval; held-out 64-token runs shared with train or the exclude files are cut out (the
#    manifest's exposed_tokens_after_cut counts what remains across packing junctions)
C=$WORK/corpus
if [ ! -s "$C/manifest.json" ]; then
    log "corpus"
    $PY -u -m ggufrefine.corpus --spec "$SPEC" --n-ctx "$N_CTX" --tokenizer "$HF" --gguf "$M" --out "$C" > "$L/corpus.log" 2>&1 \
        || fail corpus "$L/corpus.log"
fi
mctx=$($PY -c "import json, sys; print(json.load(open(sys.argv[1]))['n_ctx'])" "$C/manifest.json")
[ "$mctx" = "$N_CTX" ] || fail "corpus: $C was built with n_ctx $mctx, this run uses $N_CTX (remove $C, or N_CTX=$mctx)"
stop corpus

# 4. Grams: train sequences in parts (one per device but the last, fixed at the first launch in grams-src/plan.txt),
#    test as part 100; then merged into train / test splits (sources removed once merged)
G=$WORK/grams
SRC=$WORK/grams-src
if [ ! -s "$G/index.json" ]; then
    log "capture"
    mkdir -p "$SRC"
    LL=(-m "$M" -c $((CAPTURE_SEQS * N_CTX)) -b "$N_CTX" -ub "$UB" -np "$CAPTURE_SEQS" --kv-unified -fa on -ngl 99)
    ntr=$($PY -c "import numpy as np, sys; print(np.load(sys.argv[1], mmap_mode='r').shape[0])" "$C/train.tokens.npy")
    if [ ! -s "$SRC/plan.txt" ]; then
        [ -z "$(ls -A "$SRC")" ] || fail "capture: $SRC holds parts but no plan.txt (remove $SRC and $G)"
        k=$(( NG > 1 ? NG - 1 : 1 )); k=$(( k < ntr ? k : ntr ))
        { echo "$ntr"; for i in $(seq 0 $((k - 1))); do echo "$((ntr * i / k)) $((ntr * (i + 1) / k))"; done; } > "$SRC/plan.txt.tmp"
        mv "$SRC/plan.txt.tmp" "$SRC/plan.txt"
    fi
    mapfile -t PLAN < "$SRC/plan.txt"
    RANGES=("${PLAN[@]:1}"); k=${#RANGES[@]}
    prev=0; for r in "${RANGES[@]}"; do read -r a b <<< "$r"; [ "$a" = "$prev" ] && [ "$b" -gt "$a" ] || prev=bad; [ "$prev" = bad ] || prev=$b; done
    [ "${PLAN[0]}" = "$ntr" ] && [ "$prev" = "$ntr" ] \
        || fail "capture: $SRC/plan.txt does not partition the corpus's $ntr train sequences (remove $SRC and $G)"
    capture_job() {
        local j=$1 a b d
        if [ "$j" -lt "$k" ]; then
            read -r a b <<< "${RANGES[j]}"; d=$SRC/part$(printf %03d "$j")
            [ -e "$d.done" ] && return 0
            $PY -u -m ggufrefine.capture.ggml --tokens "$C/train.tokens.npy" --seqs "$a:$b" --part-id "$j" --out "$d" \
                -- "${LL[@]}" > "$L/capture-part$j.log" 2>&1 && touch "$d.done"
        else
            d=$SRC/part100
            [ -e "$d.done" ] && return 0
            $PY -u -m ggufrefine.capture.ggml --tokens "$C/test.tokens.npy" --part-id 100 --out "$d" \
                -- "${LL[@]}" > "$L/capture-test.log" 2>&1 && touch "$d.done"
        fi
    }
    log "capture: $k train parts + test on devices ${GPU[*]}"
    lanes capture_job $((k + 1)) || fail capture "$L/capture-*.log"
    $PY -u -m ggufrefine.capture.merge_grams --src "$SRC" --out "$G" --test-parts 100 --delete \
        --require-parts "$(seq -s, 0 $((k - 1))),100" > "$L/merge.log" 2>&1 || fail merge "$L/merge.log"
fi
stop capture

# 5. KLD bases: the eval split (exact sequences, chat markup as tokens) and any extra text bases. A new base makes the
#    scores against the old one stale, so they are removed.
BN=(eval); BT=("")
for nb in ${EXTRA_BASES:-}; do BN+=("${nb%%=*}"); BT+=("${nb#*=}"); done
base_job() {
    local n=${BN[$1]} xb=$WORK/base-${BN[$1]}.kld
    [ -s "$xb" ] && return 0
    [ -d "$WORK/kld" ] && find "$WORK/kld" -mindepth 2 -maxdepth 2 \( -name "$n.json" -o -name "$n.chunks.tsv" \
        -o -name "$n.tokens.bin" -o -name "$n.meta.json" \) -delete
    if [ "$n" = eval ]; then
        $PY -c "import numpy as np, sys; np.load(sys.argv[1]).astype('<i4').tofile(sys.argv[2])" "$C/eval.tokens.npy" \
            "$C/eval.tokens.bin" || return 1
        $B/llama-perplexity -m "$M" --tokens-file "$C/eval.tokens.bin" -c "$N_CTX" -b "$N_CTX" -ub "$UB" -ngl 99 -fa on \
            --kl-divergence-base "$xb.tmp" > "$L/base-eval.log" 2>&1 || return 1
    else
        $B/llama-perplexity -m "$M" -f "${BT[$1]}" -c "$N_CTX" -b "$N_CTX" -ub "$UB" --chunks 10 -ngl 99 -fa on \
            --kl-divergence-base "$xb.tmp" > "$L/base-$n.log" 2>&1 || return 1
    fi
    mv "$xb.tmp" "$xb"
}
log "kld bases: ${BN[*]}"
lanes base_job ${#BN[@]} || fail "kld base" "$L/base-*.log"
BASES=(); for n in "${BN[@]}"; do BASES+=(--base "$n=$WORK/base-$n.kld"); done
stop base

# 6. stock quant (llama-quantize preset + imatrix)
S=$WORK/$NAME-$TYPE-stock.gguf
if [ ! -s "$S" ]; then
    log "quantize $TYPE"
    rm -rf "$WORK/kld/$TYPE-stock"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$GPUS $B/llama-quantize --imatrix "$IM" ${QUANT_ARGS:-} "$M" "$S.tmp" "$TYPE" "$(nproc)" \
        > "$L/quantize.log" 2>&1 || fail quantize "$L/quantize.log"
    mv "$S.tmp" "$S"
fi
stop quantize

# 7. refine a copy of the stock file per seed (each tensor keeps its stock type; stock codes are read from the stock
#    file). Layers in 8 fixed shards over the devices; each shard's journal (in <copy>.journal/) lets an interrupted
#    run resume on the same copy, and goes away with it.
lay=$($PY -c "import json, sys; print('\n'.join(sorted((l for l in json.load(open(sys.argv[1]))['layers'] if l.isdigit()), key=int)))" "$G/index.json") \
    || fail "refine: cannot read $G/index.json"
mapfile -t LAYERS <<< "$lay"
nl=${#LAYERS[@]}
[ -n "$lay" ] && [ "$nl" -gt 0 ] || fail "refine: no numbered layers in $G/index.json"
NSH=$(( nl < 8 ? nl : 8 ))
refine_job() {
    local k=$1 a=${LAYERS[$((nl * $1 / NSH))]} b=${LAYERS[$((nl * ($1 + 1) / NSH - 1))]}
    $PY -u -m ggufrefine.models.dense --gguf "$M" --grams "$G" --stock "$S" --seed "$seed" --layers "$a-$b" \
        --allow-empty --write-gguf "$R" --results "$R.journal/shard$k.jsonl" > "$L/refine-s$seed-shard$k.log" 2>&1
}
for seed in "${SEED[@]}"; do
    R=$WORK/$NAME-$TYPE-refined-s$seed.gguf
    [ -e "$R.done" ] && [ -s "$R" ] && continue
    if [ ! -s "$R" ]; then
        rm -rf "$R.done" "$R.journal" "$WORK/kld/$TYPE-refined-s$seed"
        cp --reflink=auto "$S" "$R.tmp" && mv "$R.tmp" "$R"
    fi
    mkdir -p "$R.journal"
    log "refine seed $seed ($nl layers, $NSH shards on devices ${GPU[*]})"
    lanes refine_job "$NSH" || fail "refine seed $seed" "$L/refine-s$seed-shard*.log"
    n=$(for k in $(seq 0 $((NSH - 1))); do cat "$L/refine-s$seed-shard$k.log"; done | sed -n 's/^refined \([0-9]*\) tensors.*/\1/p' | awk '{t += $1} END {print t + 0}')
    [ "$n" -gt 0 ] || fail "refine seed $seed: no tensor refined (no captured tensor has a refinable type in $S)"
    log "refine seed $seed: $n tensors"
    touch "$R.done"
done
stop refine

# 8. KLD of each file against every base (one per device at a time), then the report. The BF16 file against its own
#    base is the null: the report fails when it is not ~0 on every row.
FILES=(stock); for s in "${SEED[@]}"; do FILES+=("refined-s$s"); done; FILES+=(bf16)
kld_job() {
    local f=${FILES[$1]} X=$WORK/$NAME-$TYPE-${FILES[$1]}.gguf
    [ "$f" = bf16 ] && X=$M
    $PY -u -m ggufrefine.eval.kld --gguf "$X" --out "$WORK/kld/$TYPE-$f" "${BASES[@]}" --extra="-c $N_CTX -b $N_CTX -ub $UB ${KLD_EXTRA:-}" \
        > "$L/kld-$TYPE-$f.log" 2>&1
}
log "kld: ${FILES[*]}"
lanes kld_job ${#FILES[@]} || fail kld "$L/kld-*.log"
runs=(); for f in "${FILES[@]}"; do runs+=("$f=$WORK/kld/$TYPE-$f"); done
ropt=()
for w in ${REPORT_WITHOUT:-}; do ropt+=(--without "$w"); done
for p in ${REPORT_POOLS:-}; do ropt+=(--pool "$p"); done
# report BASE ARGS...: the table goes to report-$TYPE-BASE.txt only when the report (and its null check) passes,
# else to report-$TYPE-BASE.FAILED.txt
report() {
    local n=$1 o=$WORK/report-$TYPE-$1; shift
    rm -f "$o.FAILED.txt"
    if $PY -m ggufrefine.eval.report --base "$n" --null bf16 "$@" "${runs[@]}" > "$o.tmp"; then
        mv "$o.tmp" "$o.txt"; cat "$o.txt"
    else
        mv "$o.tmp" "$o.FAILED.txt"; cat "$o.FAILED.txt"; fail "report ($n)" "$o.FAILED.txt"
    fi
}
report eval --manifest "$C/manifest.json" "${ropt[@]}"
for n in "${BN[@]:1}"; do report "$n"; done
log "done: $WORK/report-$TYPE-*.txt"
