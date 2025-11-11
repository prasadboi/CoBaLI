# CoBaLI: Continuous Batching and Prefill Splitting for LLM Inference

A GPU programming course project demonstrating continuous batching and prefill splitting optimizations for large language model inference.

## 🎯 Project Goals

Starting with a small open-source LLM (Qwen 0.5B in GGUF format), this project implements:

1. **Phase 1**: Baseline sequential inference (using llama.cpp)
2. **Phase 2**: Continuous batching scheduler (C++)
3. **Phase 3**: Prefill splitting/chunking (C++)
4. **Phase 4**: Custom CUDA kernels (future)

**Key Constraint**: No use of advanced libraries like vLLM or TensorRT-LLM — everything implemented manually using llama.cpp as baseline.

## 📊 Expected Performance Improvements

| Phase | Optimization | Expected Speedup |
|-------|--------------|------------------|
| 1 | Baseline | 1x (reference) |
| 2 | Continuous Batching | 2-3x throughput |
| 3 | + Prefill Splitting | 5-10x throughput |
| 4 | + Custom CUDA Kernels | 10-15x throughput |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Python Layer (Orchestration & Benchmarking)                 │
├─────────────────────────────────────────────────────────────┤
│ C++ Layer (Scheduling & Batch Management)                   │
│  - ContinuousBatcher: Dynamic batch formation               │
│  - PrefillSplitter: Chunked prefill scheduling              │
│  - KVCacheManager: GPU memory management                    │
├─────────────────────────────────────────────────────────────┤
│ CUDA Layer (GPU Execution)                                  │
│  - llama.cpp kernels (Phase 1-3)                           │
│  - Custom CUDA kernels (Phase 4)                           │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **OS**: Linux
- **GPU**: NVIDIA RTX 2080 Ti or better (compute capability 7.5+)
- **CUDA**: 12.4 or 13.0
- **CMake**: 3.18+
- **C++**: C++17 compiler (GCC 9+ or Clang 10+)
- **Python**: 3.8+ (optional, for benchmarking)

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd CoBaLI

# Run setup script (builds everything)
./scripts/setup.sh

# Download model (Qwen 0.5B, ~300MB)
./scripts/download_model.sh
```

### Running

```bash
# Phase 1: Baseline (sequential)
./build/cobali_main baseline models/qwen2-0_5b-instruct-q4_0.gguf

# Phase 2: Continuous batching
./build/cobali_main batching models/qwen2-0_5b-instruct-q4_0.gguf

# Phase 3: Full CoBaLI (batching + prefill splitting)
./build/cobali_main full models/qwen2-0_5b-instruct-q4_0.gguf
```

### Running Examples

```bash
# Baseline sequential inference
./build/examples/example_baseline models/qwen2-0_5b-instruct-q4_0.gguf

# Continuous batching demo
./build/examples/example_continuous_batching models/qwen2-0_5b-instruct-q4_0.gguf

# Full CoBaLI demo
./build/examples/example_full_cobali models/qwen2-0_5b-instruct-q4_0.gguf
```

## 📁 Repository Structure

```
cobali/
├── include/cobali/          # C++ header files
│   ├── common/              # Types, config, utilities
│   ├── scheduler/           # Batching and splitting logic
│   ├── memory/              # KV cache management
│   ├── baseline/            # Sequential engine
│   ├── engine/              # Main inference engine
│   └── kernels/             # CUDA kernel headers
│
├── src/                     # C++ implementation
│   ├── scheduler/           # YOUR CORE IMPLEMENTATIONS
│   │   ├── continuous_batcher.cpp
│   │   └── prefill_splitter.cpp
│   ├── kernels/             # YOUR CUDA KERNELS (Phase 4)
│   └── ...
│
├── examples/cpp/            # Example programs
├── tests/cpp/               # C++ unit tests
├── benchmarks/              # Python benchmarking scripts
├── docs/                    # Detailed documentation
├── configs/                 # Configuration files
└── scripts/                 # Setup and build scripts
```

## 📖 Documentation

- [01_design_overview.md](docs/01_design_overview.md) - Architecture and design
- [02_baseline_implementation.md](docs/02_baseline_implementation.md) - Phase 1: Baseline
- [03_continuous_batching.md](docs/03_continuous_batching.md) - Phase 2: Batching
- [04_prefill_splitting.md](docs/04_prefill_splitting.md) - Phase 3: Splitting
- [05_cuda_kernels.md](docs/05_cuda_kernels.md) - Phase 4: Custom kernels (TODO)
- [06_results_analysis.md](docs/06_results_analysis.md) - Benchmarks and results (TODO)

## 🔧 Configuration

Configuration files in `configs/`:

```yaml
# configs/cobali_config.yaml
batching:
  enable_continuous_batching: true
  max_batch_size: 32
  max_tokens_per_batch: 4096

prefill_splitting:
  enabled: true
  chunk_size: 512
  decode_priority_weight: 0.7  # 0.0-1.0
```

## 🧪 Testing

```bash
# Build tests
cd build
cmake --build . --target test_request_queue

# Run tests
./test_request_queue
./test_batch_manager
./test_prefill_splitter
```

## 📊 Benchmarking

```bash
# Python benchmarks (after setup)
source venv/bin/activate
python benchmarks/run_baseline.py
python benchmarks/run_continuous_batch.py
python benchmarks/compare_all.py
```

## 🎓 Academic Context

This project is for a GPU programming course and focuses on:

1. **Host-side optimization**: C++ scheduling algorithms
2. **GPU memory management**: KV cache allocation
3. **Batched execution**: Efficient GPU utilization
4. **Custom CUDA kernels**: Low-level GPU programming

## 🔍 Key Concepts

### Continuous Batching
Instead of waiting for all requests in a batch to complete:
- Dynamically add new requests to active batch
- Remove completed requests mid-execution
- Maximize GPU utilization

### Prefill Splitting
Break large prompt processing into chunks:
- Process 512 tokens at a time (configurable)
- Interleave with decode steps from other requests
- Improves fairness and reduces time-to-first-token

### KV Cache Management
Per-request KV cache allocation:
- Each request gets separate GPU memory slot
- Dynamic allocation/deallocation
- Enables concurrent request processing

## 📝 License

[Your License Here]

## 🙏 Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) for baseline implementation
- [Qwen](https://github.com/QwenLM/Qwen) for the model
- [Orca/vLLM paper](https://arxiv.org/abs/2309.06180) for continuous batching inspiration

## 📮 Contact

[Your contact information]

---

**Note**: This is an academic project for learning GPU programming. For production use, consider vLLM, TensorRT-LLM, or other mature inference frameworks.
