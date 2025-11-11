# Getting Started with CoBaLI

This guide will help you get CoBaLI up and running quickly.

## Prerequisites Checklist

- [ ] Linux OS (tested on Ubuntu 20.04+)
- [ ] NVIDIA GPU (RTX 2080 Ti or better)
- [ ] CUDA 12.4 or 13.0 installed
- [ ] CMake 3.18+ installed
- [ ] GCC 9+ or Clang 10+ installed
- [ ] Git installed
- [ ] Python 3.8+ (optional, for benchmarks)

## Step-by-Step Setup

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd CoBaLI
```

### 2. Run Setup Script

This script will:
- Initialize llama.cpp submodule
- Build llama.cpp with CUDA support
- Build CoBaLI library and executables
- Set up Python virtual environment (optional)

```bash
./scripts/setup.sh
```

**Expected output:**
```
[1/5] Checking dependencies...
✓ CMake found: cmake version 3.22.1
✓ CUDA found: release 12.4

[2/5] Initializing llama.cpp submodule...
✓ llama.cpp already initialized

[3/5] Building llama.cpp...
✓ llama.cpp built successfully

[4/5] Building CoBaLI...
✓ CoBaLI built successfully

[5/5] Setting up Python environment (optional)...
✓ Python environment set up

Setup complete!
```

### 3. Download Model

```bash
./scripts/download_model.sh
```

This downloads Qwen 0.5B (~300MB) to the `models/` directory.

### 4. Test Installation

```bash
# Quick test with baseline
./build/cobali_main baseline models/qwen2-0_5b-instruct-q4_0.gguf \
    --prompt "Hello, world!" \
    --max-tokens 32 \
    --verbose
```

**Expected output:**
```
[INFO] CoBaLI - Continuous Batching and Prefill Splitting
[INFO] Mode: baseline
[INFO] Model: models/qwen2-0_5b-instruct-q4_0.gguf
[INFO] Initializing engine...
[INFO] Model loaded successfully
[INFO] Running baseline inference...
[INFO] Generated 32 tokens in 85.32 ms
[INFO] Throughput: 375.12 tokens/sec
Done!
```

## Running Examples

### Example 1: Baseline Sequential Inference

```bash
./build/examples/example_baseline models/qwen2-0_5b-instruct-q4_0.gguf
```

This demonstrates Phase 1 - sequential inference with no batching.

### Example 2: Continuous Batching

```bash
./build/examples/example_continuous_batching models/qwen2-0_5b-instruct-q4_0.gguf
```

This demonstrates Phase 2 - dynamic batching of multiple requests.

### Example 3: Full CoBaLI

```bash
./build/examples/example_full_cobali models/qwen2-0_5b-instruct-q4_0.gguf
```

This demonstrates Phase 3 - batching + prefill splitting.

## Running Benchmarks

### Python Benchmark (Recommended)

```bash
# Activate virtual environment
source venv/bin/activate

# Run baseline benchmark
python benchmarks/run_baseline.py \
    --model models/qwen2-0_5b-instruct-q4_0.gguf \
    --num-requests 10
```

### Manual Benchmarking

```bash
# Phase 1: Baseline
time ./build/cobali_main baseline models/qwen2-0_5b-instruct-q4_0.gguf

# Phase 2: Continuous batching
time ./build/cobali_main batching models/qwen2-0_5b-instruct-q4_0.gguf

# Phase 3: Full CoBaLI
time ./build/cobali_main full models/qwen2-0_5b-instruct-q4_0.gguf
```

## Configuration

Edit configuration files in `configs/`:

```bash
# For baseline
vim configs/baseline_config.yaml

# For full CoBaLI
vim configs/cobali_config.yaml
```

Key parameters:
- `max_batch_size`: Maximum requests per batch (default: 32)
- `max_tokens_per_batch`: Maximum tokens per batch (default: 4096)
- `prefill_chunk_size`: Tokens per prefill chunk (default: 512)
- `decode_priority_weight`: Decode vs prefill priority (default: 0.7)

## Development Workflow

### Building

```bash
# Full rebuild
./scripts/build_cpp.sh

# Incremental build
cd build
make -j$(nproc)
```

### Testing

```bash
cd build
./test_request_queue
./test_batch_manager
./test_prefill_splitter
./test_kv_cache
```

### Adding New Code

1. **Headers**: Add to `include/cobali/`
2. **Implementation**: Add to `src/`
3. **Tests**: Add to `tests/cpp/`
4. **Update CMakeLists.txt**: Add new files

```cmake
set(COBALI_SOURCES
    src/your_new_file.cpp
    ...
)
```

## Troubleshooting

### Build Errors

**Error: "llama.cpp not found"**
```bash
git submodule update --init --recursive
cd third_party/llama.cpp
mkdir build && cd build
cmake .. -DLLAMA_CUBLAS=ON -DCMAKE_CUDA_ARCHITECTURES=75
make -j
```

**Error: "CUDA not found"**
```bash
# Check CUDA installation
nvcc --version

# Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Runtime Errors

**Error: "Failed to load model"**
- Check model path is correct
- Verify model file exists and is not corrupted
- Re-download with `./scripts/download_model.sh`

**Error: "Out of GPU memory"**
- Reduce `max_batch_size` in config
- Reduce `kv_cache_size_mb` in config
- Use smaller model (Qwen 0.5B instead of 3B)

**Error: "Slow inference"**
- Check GPU is being used: `nvidia-smi`
- Verify `n_gpu_layers = -1` (all layers on GPU)
- Check temperature isn't too low (causes slow sampling)

## Next Steps

1. **Phase 1**: Establish baseline performance
   ```bash
   python benchmarks/run_baseline.py --model models/qwen2-0_5b-instruct-q4_0.gguf
   ```

2. **Phase 2**: Implement continuous batching improvements
   - Tune `max_batch_size`
   - Measure throughput gains
   - Document in `docs/06_results_analysis.md`

3. **Phase 3**: Add prefill splitting
   - Tune `prefill_chunk_size`
   - Measure TTFT improvements
   - Compare fairness

4. **Phase 4**: Write custom CUDA kernels (future)
   - Batched attention
   - Profiling with Nsight Compute
   - Performance optimization

## Resources

- [Documentation](docs/)
- [Design Overview](docs/01_design_overview.md)
- [Baseline Implementation](docs/02_baseline_implementation.md)
- [Continuous Batching](docs/03_continuous_batching.md)
- [Prefill Splitting](docs/04_prefill_splitting.md)

## Getting Help

- Check documentation in `docs/`
- Look at example programs in `examples/cpp/`
- Read test files in `tests/cpp/` for usage examples

Good luck with your GPU programming project! 🚀

