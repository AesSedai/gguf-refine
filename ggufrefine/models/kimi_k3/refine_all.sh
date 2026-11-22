#!/bin/bash
# Refine every routed-expert tensor of K3.1 at one type, one process per GPU over interleaved layers. Codes are written straight into TARGET (a stock build of that type,
# e.g. build_stock.sh output, same layout) and stock codes come from the quantizer + imatrix, so TARGET can be
# patched while running and a restart resumes from the results jsonl (an expert is recorded only after its codes
# are written). Pass 2 retries layers that failed.
#   refine_all.sh IQ2_XXS TARGET.gguf      env (required): MODEL DUMP IMATRIX   (optional): PY GPUS LAYERS TEST_PARTS LOGS
set -u
T=$1; TARGET=$2
cd "$(dirname "$0")/../../.."
PY=${PY:-python3}
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}; LAYERS=${LAYERS:-$(seq 1 92)}
TEST_PARTS=${TEST_PARTS:-29,30}
: "${MODEL:?set MODEL to the Kimi-K3.1 HF directory}" "${DUMP:?set DUMP to the capture root}" "${IMATRIX:?set IMATRIX}"
LOGD=${LOGS:-logs/refine-$T}
mkdir -p $LOGD
ng=$(echo $GPUS | wc -w); i=0
for g in $GPUS; do
  mine=$(echo $LAYERS | tr ' ' '\n' | awk -v n=$ng -v i=$i 'NR%n==i%n' | tr '\n' ' ')
  (
    for pass in 1 2; do
    for L in $mine; do
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$g $PY -u -m ggufrefine.models.kimi_k3.refine \
        --layer $L --types gate_up=$T,down=$T --model "$MODEL" --dump "$DUMP" --test-part $TEST_PARTS --requantize --imatrix "$IMATRIX" \
        --write-gguf "$TARGET" --results $LOGD/results-g$g.jsonl --batch 16 --loaders 4 --prefetch 1 \
        || { echo "[$(date +%T)] L$L refine FAILED (pass $pass)"; continue; }
      echo "[$(date +%T)] L$L done"
    done
    done
  ) > $LOGD/g$g.log 2>&1 &
  i=$((i+1))
done
wait
echo "refine-all $T done"
