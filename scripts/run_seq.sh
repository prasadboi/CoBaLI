#!/usr/bin/env bash
set -euo pipefail
MODEL=../models/qwen2.5-0.5b-instruct-q5_k_m.gguf
while IFS= read -r P; do
  ../build/cobali_runner --model "$MODEL" --mode seq --max-slots 1 --prompt "$P"
done < ../workloads/prompts_short.txt
