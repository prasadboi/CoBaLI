#!/usr/bin/env bash
# CoBaLI full profiling with 16 slots and prefill-chunk=128
# Uses a large context so long prompts don't blow up KV cache.

set -uo pipefail   # NOTE: no -e, so one failure won't abort the script

RUNNER="../build/cobali_runner"
MODEL="../models/qwen2.5-0.5b-instruct-q5_k_m.gguf"
PROMPTS="../workloads/prompts_200.txt"

MAX_SLOTS=16
PREFILL_CHUNK=128

# Try a big context so each slot has enough tokens.
# With ctx=32768 and 16 slots, your fork tends to set n_ctx_seq ≈ 2048.
CTX_SIZE=16384

ts=$(date +%Y%m%d_%H%M%S)
LOG="cobali_full_profile_128_chunk${PREFILL_CHUNK}_${ts}.txt"

echo "Writing log to: $LOG"
exec > >(tee "$LOG") 2>&1

echo "============================================================"
echo " CoBaLI full profile (max-slots=$MAX_SLOTS, chunk=$PREFILL_CHUNK)"
echo " date      : $(date)"
echo " runner    : $RUNNER"
echo " model     : $MODEL"
echo " prompts   : $PROMPTS"
echo " log file  : $LOG"
echo " ctx-size  : $CTX_SIZE"
echo "============================================================"
echo

echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
echo

echo "Prompts file stats:"
wc -l "$PROMPTS"
echo

# 1) SEQ baseline
echo "----- 1) SEQ mode (run-to-completion, max-slots=1) -----"
/usr/bin/time -f "seq wall=%E user=%U sys=%S" \
  "$RUNNER" \
  --model "$MODEL" \
  --prompts "$PROMPTS" \
  --mode seq \
  --max-slots 1 \
  --ctx "$CTX_SIZE"

echo

# 2) CONT, no prefill splitting (baseline continuous)
echo "----- 2) CONT mode, no prefill split (max-slots=$MAX_SLOTS) -----"
/usr/bin/time -f "cont-nosp wall=%E user=%U sys=%S" \
  "$RUNNER" \
  --model "$MODEL" \
  --prompts "$PROMPTS" \
  --mode cont \
  --max-slots "$MAX_SLOTS" \
  --ctx "$CTX_SIZE" \
  || echo "[WARN] cont (no prefill split) failed; continuing..."

echo

# 3) CONT + prefill splitting (chunk=128)
echo "----- 3) CONT mode, prefill split (max-slots=$MAX_SLOTS, chunk=$PREFILL_CHUNK) -----"
/usr/bin/time -f "cont-split${PREFILL_CHUNK} wall=%E user=%U sys=%S" \
  "$RUNNER" \
  --model "$MODEL" \
  --prompts "$PROMPTS" \
  --mode cont \
  --max-slots "$MAX_SLOTS" \
  --ctx "$CTX_SIZE" \
  --prefill-chunk "$PREFILL_CHUNK"

echo
echo "All done. Combined log is in: $LOG"

