#!/bin/bash

set -e

echo "Building CoBaLI C++ library and examples..."

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Create build directory
if [ ! -d "build" ]; then
    mkdir build
fi

cd build

# Configure
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=75 \
    -DBUILD_TESTS=ON \
    -DBUILD_EXAMPLES=ON \
    -DBUILD_PYTHON_BINDINGS=OFF

# Build
cmake --build . --config Release -j$(nproc)

echo ""
echo "✓ Build complete!"
echo "  Binary: ./build/cobali_main"
echo "  Examples: ./build/examples/"
echo "  Library: ./build/libcobali.so"

