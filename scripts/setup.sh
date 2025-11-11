#!/bin/bash

set -e

echo "================================================"
echo "CoBaLI Setup Script"
echo "================================================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo -e "${GREEN}[1/5] Checking dependencies...${NC}"

# Check for CMake
if ! command -v cmake &> /dev/null; then
    echo "Error: CMake not found. Please install CMake 3.18 or later."
    exit 1
fi

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "Error: CUDA not found. Please install CUDA toolkit."
    exit 1
fi

# Check for Git
if ! command -v git &> /dev/null; then
    echo "Error: Git not found."
    exit 1
fi

echo "✓ CMake found: $(cmake --version | head -n1)"
echo "✓ CUDA found: $(nvcc --version | grep release)"

echo -e "\n${GREEN}[2/5] Initializing llama.cpp submodule...${NC}"

if [ ! -d "third_party/llama.cpp/.git" ]; then
    echo "Cloning llama.cpp..."
    git submodule update --init --recursive
else
    echo "✓ llama.cpp already initialized"
fi

echo -e "\n${GREEN}[3/5] Building llama.cpp...${NC}"

cd third_party/llama.cpp

if [ ! -d "build" ]; then
    mkdir build
fi

cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_CUBLAS=ON \
    -DCMAKE_CUDA_ARCHITECTURES=75

cmake --build . --config Release -j$(nproc)

echo "✓ llama.cpp built successfully"

cd "$PROJECT_ROOT"

echo -e "\n${GREEN}[4/5] Building CoBaLI...${NC}"

if [ ! -d "build" ]; then
    mkdir build
fi

cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=75 \
    -DBUILD_TESTS=ON \
    -DBUILD_EXAMPLES=ON

cmake --build . --config Release -j$(nproc)

echo "✓ CoBaLI built successfully"

cd "$PROJECT_ROOT"

echo -e "\n${GREEN}[5/5] Setting up Python environment (optional)...${NC}"

if command -v python3 &> /dev/null; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    
    echo "Installing Python dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    
    echo "✓ Python environment set up"
else
    echo -e "${YELLOW}⚠ Python3 not found, skipping Python setup${NC}"
fi

echo -e "\n${GREEN}================================================${NC}"
echo -e "${GREEN}Setup complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Download a model: ./scripts/download_model.sh"
echo "  2. Run baseline: ./build/cobali_main baseline models/qwen-0.5b-q4_0.gguf"
echo "  3. Run examples: ./build/examples/example_baseline models/qwen-0.5b-q4_0.gguf"
echo ""
echo "For Python benchmarks:"
echo "  source venv/bin/activate"
echo "  python benchmarks/run_baseline.py"

