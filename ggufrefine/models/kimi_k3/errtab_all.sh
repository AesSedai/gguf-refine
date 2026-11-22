#!/bin/bash
# Error tables for the type plan: all 92 MoE layers, 48 usage-stratified experts each, one process per GPU
#   errtab_all.sh     env (required): MODEL DUMP IMATRIX   (optional): PY TYPES OUT
set -u
cd "$(dirname "$0")/../../.."
PY=${PY:-python3}
: "${MODEL:?set MODEL to the Kimi-K3.1 HF directory}" "${DUMP:?set DUMP to the capture root}" "${IMATRIX:?set IMATRIX}"
TYPES=${TYPES:-IQ2_XXS,IQ2_XS,IQ2_S,IQ3_XXS,IQ3_S,IQ1_M,IQ4_XS}
OUT=${OUT:-logs/errtab}
mkdir -p $OUT
for g in 0 1 2 3 4 5 6 7; do
  L=$(seq $((g+1)) 8 92 | paste -sd,)
  CUDA_VISIBLE_DEVICES=$g $PY -u -m ggufrefine.models.kimi_k3.errtab --layers $L --quant-threads 2 --types $TYPES \
    --model "$MODEL" --dump "$DUMP" --imatrix "$IMATRIX" --out $OUT/errtab-g$g.jsonl > $OUT/errtab-g$g.log 2>&1 &
done
wait
echo "errtab all done"
