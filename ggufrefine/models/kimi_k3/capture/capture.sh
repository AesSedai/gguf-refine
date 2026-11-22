#!/bin/bash
# Calibration capture of Kimi-K3.1: the token file is split into parts of $PART sequences, one forward.py process per
# GPU and part, in rounds of one part per GPU. Per part: the routed-expert dump of every MoE layer ->
# $OUT/partNN/x2/layer-L, and dense Grams of every layer + output -> $SHM (per process) -> merge_grams --watch ->
# $OUT/grams/{train,test}. Layer weights are read once per round into RAM by layer_cache. Resumable: finished parts
# are skipped (status --check reads the outputs); a part that died resumes from its last checkpoint (every 32 layers).
#   capture.sh        env (required): MODEL (HF dir), TOKENS (.npy int32 [n_seq, n_ctx]), OUT
#                     env (optional): PY, GPUS ("0 1 ..."), PART (10), TEST_PARTS (held-out parts; default the last
#                     two), SHM (/dev/shm/gguf-refine-capture), LOGS (logs/capture)
# Logs (tail-able): $LOGS/{capture,partNN,merge-roundR,cache-roundR,disk}.log
set -u
: "${MODEL:?set MODEL to the Kimi-K3.1 HF directory}" "${TOKENS:?set TOKENS to the token .npy}" "${OUT:?set OUT}"
cd "$(dirname "$0")/../../../.."
PY=${PY:-python3}
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
PART=${PART:-10}
SHM=${SHM:-/dev/shm/gguf-refine-capture}
G=$OUT/grams; S=$SHM/grams; C=$SHM/layer-cache
L=${LOGS:-logs/capture}
K=ggufrefine.models.kimi_k3.capture
n_seq=$($PY -c "import numpy, sys; print(numpy.load(sys.argv[1], mmap_mode='r').shape[0])" "$TOKENS") || exit 1
n_part=$(( (n_seq + PART - 1) / PART )); ng=${#GPUS[@]}; n_round=$(( (n_part + ng - 1) / ng ))
TEST_PARTS=${TEST_PARTS:-$((n_part - 2)),$((n_part - 1))}
mkdir -p "$L" "$OUT" "$S"
echo "[$(date +%T)] capture start: $n_seq sequences, $n_part parts of $PART (test $TEST_PARTS), $n_round rounds on GPUs ${GPUS[*]}" | tee -a "$L/capture.log"

( while true; do
    echo "$(date +%T) shm=$(du -sb "$SHM" 2>/dev/null | cut -f1) merged=$(du -sb "$G" 2>/dev/null | cut -f1) out_free=$(df -B1 --output=avail "$OUT" | tail -1)"
    sleep 60; done ) >> "$L/disk.log" 2>&1 &
SAMPLER=$!

status() { $PY -m $K.status --out "$OUT" --grams "$G" --parts 0-$((n_part - 1)) --test-parts "$TEST_PARTS" "$@"; }
job() { k=$1; g=$2; lo=$((k*PART)); hi=$((lo+PART)); [ $hi -gt $n_seq ] && hi=$n_seq
  p=$(printf "part%02d" $k)
  if status --check $k > /dev/null; then echo "[$(date +%T)] $p complete, skipped" | tee -a "$L/capture.log"; return 0; fi
  CUDA_VISIBLE_DEVICES=$g PYTHONUNBUFFERED=1 $PY -u -m $K.forward --model "$MODEL" --tokens "$TOKENS" --seqs $lo:$hi --out "$OUT/$p" \
    --no-zero-stats --expert-dump-layers all --expert-dump-dir "$OUT/$p/x2" --grams --gram-dir "$S/$p" --part-id $k --no-logits \
    --ckpt-every 32 --resume --layer-cache "$C" --cache-tag $p >> "$L/$p.log" 2>&1
  rc=$?
  if [ $rc = 0 ]; then echo "[$(date +%T)] $p finished" | tee -a "$L/capture.log"
  else echo "[$(date +%T)] $p FAILED rc=$rc (rerun this script: it resumes)" | tee -a "$L/capture.log"; fi; }

for ((round = 0; round < n_round; round++)); do
  parts=""; for ((i = 0; i < ng; i++)); do k=$((round*ng+i)); [ $k -lt $n_part ] && parts="$parts${parts:+,}$k"; done
  rm -f "$L/round$round.done"
  $PY -u -m $K.layer_cache --model "$MODEL" --cache-dir "$C" --window 4 --max-gb 80 --exit-when-done >> "$L/cache-round$round.log" 2>&1 &
  CACHE=$!
  $PY -u -m ggufrefine.capture.merge_grams --src "$S" --out "$G" --test-parts "$TEST_PARTS" --delete --watch \
    --until-file "$L/round$round.done" --expect $parts --max-wait 120 >> "$L/merge-round$round.log" 2>&1 &
  MERGE=$!
  echo "[$(date +%T)] round $round: parts $parts" | tee -a "$L/capture.log"
  pids=""
  for ((i = 0; i < ng; i++)); do k=$((round*ng+i)); if [ $k -lt $n_part ]; then job $k ${GPUS[$i]} & pids="$pids $!"; fi; done
  wait $pids
  touch "$L/round$round.done"
  wait $MERGE || echo "[$(date +%T)] merger round $round FAILED" | tee -a "$L/capture.log"
  kill $CACHE 2>/dev/null; wait $CACHE 2>/dev/null
  rm -rf "$C"
  echo "[$(date +%T)] round $round done; merged $(du -sh "$G" | cut -f1), dumps $(du -sh --exclude=grams "$OUT" | cut -f1)" | tee -a "$L/capture.log"
done
kill $SAMPLER
status > "$L/status.jsonl"
echo "[$(date +%T)] capture done: $(grep -c '"complete": true' "$L/status.jsonl")/$n_part parts complete; see $L/status.jsonl" | tee -a "$L/capture.log"
