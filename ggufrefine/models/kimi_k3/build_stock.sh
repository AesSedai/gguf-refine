#!/bin/bash
# Stock (unrefined) K3.1 build with the recipe of the VRAM substrate:
# routed experts $EXPERT_TYPE (default IQ2_XXS), non-expert tensors Q6_K, small / precision-critical tensors F32.
# llama-quantize from native/ (patch 0009: GPU quantizer, byte-identical to the CPU path). ~753 GB at IQ2_XXS.
#   build_stock.sh OUT.gguf [LOG]       env (required): SRC (BF16/MXFP4 GGUF), IMATRIX   (optional): EXPERT_TYPE, GGUF_REFINE_BIN
set -eu
OUT=$1
LOG=${2:-logs/build-stock-$(basename "$OUT" .gguf).log}
BIN=${GGUF_REFINE_BIN:-$(cd "$(dirname "$0")/../../.." && pwd)/native/llama.cpp/build/bin}
: "${SRC:?set SRC to the BF16/MXFP4 GGUF}" "${IMATRIX:?set IMATRIX}"
T=${EXPERT_TYPE:-IQ2_XXS}
mkdir -p "$(dirname "$LOG")"
env GGUF_WRITER_PREFETCH=6 OMP_NUM_THREADS=48 "$BIN/llama-quantize" --allow-requantize --imatrix "$IMATRIX" \
  --tensor-type "ssm_g=Q6_K" --tensor-type "ssm_=F32" --tensor-type "indexer=F32" \
  --tensor-type "(attn_k_b|attn_v_b|attn_kv_a_mqa)=F32" \
  --tensor-type "ffn_(up|down|gate)_shexp=Q6_K" \
  --tensor-type "ffn_up_exps=$T" --tensor-type "ffn_gate_exps=$T" --tensor-type "ffn_down_exps=$T" \
  "$SRC" "$OUT" Q6_K 48 2>&1 | tee "$LOG"
