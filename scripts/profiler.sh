#!/usr/bin/env bash
set -euo pipefail

# All-in-one profiling script for 16 prompts
# Run this from: /home/ld3074/lakshay/CoBaLI/scripts

MODEL="../models/qwen2.5-0.5b-instruct-q5_k_m.gguf"
PROMPTS_FULL="../workloads/prompts.txt"
PROMPTS_16="../workloads/prompts_16.txt"

LOG="profile_all_16prompts_$(date +%Y%m%d_%H%M%S).log"

{
  echo "========================================"
  echo " CoBaLI profiling run (16 prompts)"
  echo " Date: $(date)"
  echo " Log file will be: ${LOG}"
  echo "========================================"
  echo

  #############################################
  # 0) Prepare 16-prompt workload
  #############################################
  echo "=== Step 0: Preparing 16-prompt file ==="
  echo "Source: ${PROMPTS_FULL}"
  echo "Target: ${PROMPTS_16}"
  echo

  if [[ ! -f "${PROMPTS_FULL}" ]]; then
    echo "ERROR: ${PROMPTS_FULL} does not exist. Edit PROMPTS_FULL in this script."
    exit 1
  fi

  # NOTE: This assumes each prompt fits on a single line.
  # If prompts are multi-line blocks, we can switch to a different splitter later.
  head -n 16 "${PROMPTS_FULL}" > "${PROMPTS_16}"
  echo "Wrote first 16 lines of ${PROMPTS_FULL} to ${PROMPTS_16}"
  echo

  #############################################
  # 1) Timed runs (seq / cont-nosp / cont-split)
  #############################################
  echo "=== Step 1: Timed runs ==="
  echo

  echo "--- 1.1 Sequential (seq, max-slots=1) ---"
  /usr/bin/time -f "seq wall=%E user=%U sys=%S" \
    ../build/cobali_runner \
      --model "${MODEL}" \
      --prompts "${PROMPTS_16}" \
      --mode seq \
      --max-slots 1
  echo

  echo "--- 1.2 Continuous (cont, no prefill split, max-slots=16) ---"
  /usr/bin/time -f "cont-nosp wall=%E user=%U sys=%S" \
    ../build/cobali_runner \
      --model "${MODEL}" \
      --prompts "${PROMPTS_16}" \
      --mode cont \
      --max-slots 16 \
      --prefill-chunk 4096
  echo

  echo "--- 1.3 Continuous + prefill split (cont, max-slots=16, chunk=512) ---"
  /usr/bin/time -f "cont-split wall=%E user=%U sys=%S" \
    ../build/cobali_runner \
      --model "${MODEL}" \
      --prompts "${PROMPTS_16}" \
      --mode cont \
      --max-slots 16 \
      --prefill-chunk 512
  echo

  #############################################
  # 2) Nsight Systems profiling
  #############################################
  echo "=== Step 2: Nsight Systems profiling ==="
  echo

  echo "--- 2.1 nsys: seq (seq_16.nsys-rep) ---"
  nsys profile --stats=true -o seq_16 \
    ../build/cobali_runner \
      --model "${MODEL}" \
      --prompts "${PROMPTS_16}" \
      --mode seq \
      --max-slots 1
  echo

  echo "--- 2.2 nsys: cont-nosp (cont_nosplit_16.nsys-rep) ---"
  nsys profile --stats=true -o cont_nosplit_16 \
    ../build/cobali_runner \
      --model "${MODEL}" \
      --prompts "${PROMPTS_16}" \
      --mode cont \
      --max-slots 16 \
      --prefill-chunk 4096
  echo

  echo "--- 2.3 nsys: cont-split (cont_split_16.nsys-rep) ---"
  nsys profile --stats=true -o cont_split_16 \
    ../build/cobali_runner \
      --model "${MODEL}" \
      --prompts "${PROMPTS_16}" \
      --mode cont \
      --max-slots 16 \
      --prefill-chunk 512
  echo

  #############################################
  # 3) Extract Nsight stats to .stats.txt files
  #############################################
  echo "=== Step 3: Extracting Nsight stats ==="
  echo

  echo "--- stats: seq_16.nsys-rep -> seq_16.stats.txt ---"
  nsys stats --report osrt_sum,cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
    seq_16.nsys-rep > seq_16.stats.txt
  echo

  echo "--- stats: cont_nosplit_16.nsys-rep -> cont_nosplit_16.stats.txt ---"
  nsys stats --report osrt_sum,cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
    cont_nosplit_16.nsys-rep > cont_nosplit_16.stats.txt
  echo

  echo "--- stats: cont_split_16.nsys-rep -> cont_split_16.stats.txt ---"
  nsys stats --report osrt_sum,cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
    cont_split_16.nsys-rep > cont_split_16.stats.txt
  echo

  echo "========================================"
  echo " Done! Key outputs:"
  echo "  - Log: ${LOG}"
  echo "  - Timings: see seq/cont-nosp/cont-split wall=... lines in the log"
  echo "  - Nsight reps: seq_16.nsys-rep, cont_nosplit_16.nsys-rep, cont_split_16.nsys-rep"
  echo "  - Stats: seq_16.stats.txt, cont_nosplit_16.stats.txt, cont_split_16.stats.txt"
  echo "========================================"
  echo

} 2>&1 | tee "${LOG}"

