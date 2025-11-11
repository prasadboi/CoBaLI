#!/usr/bin/env bash
set -euo pipefail
./build/cobali_runner --model models/your-model.gguf --mode seq --max-slots 1 --prompts workloads/prompts_short.txt
