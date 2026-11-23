#!/bin/bash
# Matched stock-vs-refined comparison: build the stock file
# with the same recipe, check it against the refined one (layout, identical non-experts, fresh-quantization codes),
# then score it on the same KLD bases with the same settings.
#   matched_eval.sh REFINED.gguf STOCK_OUT.gguf KLD_OUT_DIR     env (required): BASES ("name=path ..."), SRC, IMATRIX
#   (optional): PY
set -uo pipefail
REFINED=$1; STOCK=$2; KOUT=$3
cd "$(dirname "$0")/../../.."
PY=${PY:-python3}
: "${BASES:?set BASES to \"name=path.kld ...\"}" "${SRC:?set SRC to the BF16/MXFP4 GGUF}" "${IMATRIX:?set IMATRIX}"
mkdir -p logs "$KOUT"
[ -s "$STOCK" ] || SRC="$SRC" IMATRIX="$IMATRIX" bash ggufrefine/models/kimi_k3/build_stock.sh "$STOCK" logs/build-$(basename "$STOCK" .gguf).log > /dev/null 2>&1 \
  || { echo "build FAILED"; exit 1; }
echo "[$(date +%T)] built $(stat -c %s "$STOCK") bytes; checking"
SAMPLE=$(for p in "1 0" "12 500" "36 77" "46 346" "68 801" "92 895"; do set -- $p; for t in ffn_gate_exps ffn_up_exps ffn_down_exps; do printf "blk.%s.%s.weight:%s," $1 $t $2; done; done)
CUDA_VISIBLE_DEVICES=0 $PY -u -m ggufrefine.eval.check_build --a "$STOCK" --b "$REFINED" --differ _exps. \
  --reference "$SRC" --imatrix "$IMATRIX" --sample "${SAMPLE%,}" 2>&1 | tee logs/check-$(basename "$STOCK" .gguf).log
grep -q "CHECK PASSED" logs/check-$(basename "$STOCK" .gguf).log || { echo "check FAILED, not running KLD"; exit 1; }
args=(); for b in $BASES; do args+=(--base "$b"); done
$PY -u -m ggufrefine.eval.kld --gguf "$STOCK" --out "$KOUT" "${args[@]}" 2>&1 | tee "$KOUT/driver.log" || { echo "kld FAILED"; exit 1; }
echo "[$(date +%T)] all done"
