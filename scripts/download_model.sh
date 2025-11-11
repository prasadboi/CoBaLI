#!/bin/bash

set -e

echo "================================================"
echo "CoBaLI Model Download Script"
echo "================================================"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_ROOT/models"

mkdir -p "$MODELS_DIR"

echo "This script will download Qwen 0.5B model in GGUF format"
echo "Model size: ~300 MB (Q4_0 quantization)"
echo ""

# Check if model already exists
MODEL_FILE="$MODELS_DIR/qwen2-0_5b-instruct-q4_0.gguf"

if [ -f "$MODEL_FILE" ]; then
    echo "Model already exists at: $MODEL_FILE"
    read -p "Re-download? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Using existing model."
        exit 0
    fi
fi

echo "Downloading from Hugging Face..."
echo ""

# Download using huggingface-cli or wget
if command -v huggingface-cli &> /dev/null; then
    echo "Using huggingface-cli..."
    huggingface-cli download \
        Qwen/Qwen2-0.5B-Instruct-GGUF \
        qwen2-0_5b-instruct-q4_0.gguf \
        --local-dir "$MODELS_DIR" \
        --local-dir-use-symlinks False
else
    echo "huggingface-cli not found, using wget..."
    
    # Direct download URL (may need to be updated)
    URL="https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF/resolve/main/qwen2-0_5b-instruct-q4_0.gguf"
    
    wget -O "$MODEL_FILE" "$URL" || {
        echo "Error: Download failed."
        echo "Please manually download from:"
        echo "https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF"
        exit 1
    }
fi

if [ -f "$MODEL_FILE" ]; then
    echo ""
    echo "✓ Model downloaded successfully!"
    echo "Location: $MODEL_FILE"
    echo "Size: $(du -h "$MODEL_FILE" | cut -f1)"
    echo ""
    echo "You can now run:"
    echo "  ./build/cobali_main baseline $MODEL_FILE"
else
    echo "Error: Download failed"
    exit 1
fi

