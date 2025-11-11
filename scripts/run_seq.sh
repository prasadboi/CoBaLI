#!/usr/bin/env bash
set -euo pipefail
../build/cobali_runner --model ../models/qwen2.5-0.5b-instruct-q5_k_m.gguf --mode seq --max-slots 1 --prompts ../workloads/prompts_short.txt
