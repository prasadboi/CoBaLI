# Third Party Dependencies

## llama.cpp

This directory contains llama.cpp as a git submodule.

### Initial Setup

```bash
# Initialize submodule
git submodule update --init --recursive

# Or if llama.cpp is not initialized yet
git submodule add https://github.com/ggerganov/llama.cpp.git llama.cpp
```

### Building llama.cpp

```bash
cd llama.cpp
mkdir build && cd build

# Configure with CUDA support
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_CUBLAS=ON \
    -DCMAKE_CUDA_ARCHITECTURES=75

# Build
cmake --build . --config Release -j$(nproc)
```

### Using in CoBaLI

The main CoBaLI CMakeLists.txt automatically links against llama.cpp:

```cmake
link_directories(${LLAMA_CPP_DIR}/build)
target_link_libraries(cobali llama)
```

### Version

We use llama.cpp master branch. Pin to a specific commit for reproducibility:

```bash
cd llama.cpp
git checkout <commit-hash>
```

### Why llama.cpp?

1. **Well-tested CUDA kernels**: Production-quality GPU kernels
2. **GGUF support**: Efficient model format
3. **Baseline reference**: Compare against mature implementation
4. **Focus on scheduling**: We implement batching/splitting, not kernels (initially)

### Phase 4 Note

In Phase 4, we'll implement custom CUDA kernels to replace some llama.cpp kernels:
- Batched attention
- Optimized prefill
- Fused operations

But we'll still use llama.cpp for:
- Model loading
- Tokenization
- Initial testing

