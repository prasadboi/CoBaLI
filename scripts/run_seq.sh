#!/usr/bin/env bash
set -euo pipefail
MODEL=../models/qwen2.5-0.5b-instruct-q5_k_m.gguf

# all generations -> out.txt, all logs -> err.txt, and also see them live
: > out.txt
: > err.txt
while IFS= read -r P; do
  echo ">>> PROMPT:" "$P" | tee -a out.txt
  ../build/cobali_runner --model "$MODEL" --mode seq --max-slots 1 --prompts "$P" 1>>out.txt 2>>err.txt
  echo | tee -a out.txt
done < ../workloads/prompts_short.txt

echo "----- DONE -----"
echo "Generations saved to out.txt"
echo "Verbose logs saved to err.txt"