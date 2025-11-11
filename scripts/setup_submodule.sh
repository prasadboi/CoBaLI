#!/usr/bin/env bash
set -euo pipefail
git submodule add https://github.com/ggml-org/llama.cpp.git external/llama.cpp || true
git submodule update --init --recursive
