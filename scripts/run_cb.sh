#!/usr/bin/env bash
set -euo pipefail
./build/cobali_runner --model models/your-model.gguf --mode cb --max-slots 8 --prompts workloads/prompts_short.txt
