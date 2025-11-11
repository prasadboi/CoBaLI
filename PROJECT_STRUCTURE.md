# CoBaLI Project Structure

This document provides a complete overview of the CoBaLI repository structure.

## Directory Tree

```
cobali/
│
├── README.md                              # Main project documentation
├── GETTING_STARTED.md                    # Quick start guide
├── PROJECT_STRUCTURE.md                  # This file
├── LICENSE                               # Project license
├── .gitignore                            # Git ignore rules
│
├── CMakeLists.txt                        # Root CMake build configuration
├── requirements.txt                       # Python dependencies
├── requirements-dev.txt                   # Python dev dependencies
│
├── docs/                                 # Detailed documentation
│   ├── 01_design_overview.md            # Architecture and design philosophy
│   ├── 02_baseline_implementation.md    # Phase 1: Sequential baseline
│   ├── 03_continuous_batching.md        # Phase 2: Continuous batching
│   ├── 04_prefill_splitting.md          # Phase 3: Prefill splitting
│   ├── 05_cuda_kernels.md               # Phase 4: Custom CUDA kernels (TODO)
│   └── 06_results_analysis.md           # Benchmark results (TODO)
│
├── models/                               # GGUF model files (gitignored)
│   ├── .gitkeep
│   └── README.md                        # Model download instructions
│
├── third_party/                          # External dependencies
│   ├── llama.cpp/                       # Git submodule (CUDA-enabled)
│   └── README.md                        # llama.cpp build instructions
│
├── include/cobali/                       # C++ header files
│   ├── common/                          # Common types and utilities
│   │   ├── types.h                      # Request, Batch, Token types
│   │   ├── config.h                     # Configuration structures
│   │   └── utils.h                      # Logging and utilities
│   │
│   ├── baseline/                        # Phase 1: Sequential engine
│   │   └── sequential_engine.h          # Simple one-at-a-time inference
│   │
│   ├── scheduler/                       # Phase 2 & 3: Scheduling logic
│   │   ├── request_queue.h              # Thread-safe priority queue
│   │   ├── batch_manager.h              # Batch formation logic
│   │   ├── continuous_batcher.h         # ⭐ YOUR Phase 2 implementation
│   │   └── prefill_splitter.h           # ⭐ YOUR Phase 3 implementation
│   │
│   ├── memory/                          # GPU memory management
│   │   └── kv_cache_manager.h           # KV cache allocation/deallocation
│   │
│   ├── engine/                          # Main inference engine
│   │   ├── cobali_engine.h              # Orchestrates everything
│   │   └── executor.h                   # Batch execution on GPU
│   │
│   └── kernels/                         # Phase 4: Custom CUDA kernels (TODO)
│       ├── attention.cuh                # Batched attention kernel
│       ├── prefill.cuh                  # Chunked prefill kernel
│       ├── decode.cuh                   # Batched decode kernel
│       └── kernel_utils.cuh             # Common CUDA utilities
│
├── src/                                 # C++ implementation files
│   ├── common/
│   │   └── utils.cpp                    # Utility implementations
│   │
│   ├── scheduler/                       # ⭐ YOUR CORE IMPLEMENTATIONS
│   │   ├── request_queue.cpp            # Priority queue implementation
│   │   ├── batch_manager.cpp            # Batch formation logic
│   │   ├── continuous_batcher.cpp       # Continuous batching scheduler
│   │   └── prefill_splitter.cpp         # Prefill splitting algorithm
│   │
│   ├── memory/
│   │   └── kv_cache_manager.cpp         # KV cache management
│   │
│   ├── baseline/
│   │   └── sequential_engine.cpp        # Sequential baseline engine
│   │
│   ├── engine/
│   │   ├── executor.cpp                 # GPU batch execution
│   │   └── cobali_engine.cpp            # Main engine orchestration
│   │
│   ├── kernels/                         # Phase 4: CUDA implementations (TODO)
│   │   ├── attention.cu                 # Your batched attention kernel
│   │   ├── prefill.cu                   # Your chunked prefill kernel
│   │   ├── decode.cu                    # Your batched decode kernel
│   │   └── kernel_utils.cu              # Common CUDA code
│   │
│   └── main.cpp                         # Standalone C++ binary
│
├── examples/                            # Example programs
│   ├── cpp/                             # C++ examples
│   │   ├── CMakeLists.txt
│   │   ├── baseline_inference.cpp       # Phase 1 example
│   │   ├── continuous_batching.cpp      # Phase 2 example
│   │   └── full_cobali.cpp              # Phase 3 example
│   │
│   └── python/                          # Python examples (TODO)
│       ├── 01_baseline.py
│       ├── 02_batching.py
│       └── 03_full_cobali.py
│
├── tests/                               # Unit and integration tests
│   ├── cpp/                             # C++ tests (Google Test)
│   │   ├── CMakeLists.txt
│   │   ├── test_request_queue.cpp       # Request queue tests
│   │   ├── test_batch_manager.cpp       # Batch manager tests
│   │   ├── test_prefill_splitter.cpp    # Prefill splitter tests
│   │   └── test_kv_cache.cpp            # KV cache tests
│   │
│   └── python/                          # Python tests (TODO)
│       ├── test_engine_wrapper.py
│       └── test_correctness.py
│
├── benchmarks/                          # Performance benchmarking
│   ├── workloads/                       # Test workloads (TODO)
│   │   ├── synthetic_generator.py       # Generate synthetic requests
│   │   └── sharegpt_loader.py           # Load ShareGPT traces
│   │
│   ├── run_baseline.py                  # Benchmark Phase 1
│   ├── run_continuous_batch.py          # Benchmark Phase 2 (TODO)
│   ├── run_prefill_split.py             # Benchmark Phase 3 (TODO)
│   ├── compare_all.py                   # Side-by-side comparison (TODO)
│   └── visualize_results.py             # Generate graphs (TODO)
│
├── scripts/                             # Utility scripts
│   ├── setup.sh                         # ⭐ Complete setup script
│   ├── download_model.sh                # Download Qwen 0.5B model
│   ├── build_cpp.sh                     # Build C++ library
│   └── run_all_benchmarks.sh            # Run full benchmark suite (TODO)
│
├── configs/                             # Configuration files
│   ├── baseline_config.yaml             # Phase 1 config
│   ├── cobali_config.yaml               # Phase 2 & 3 config
│   └── benchmark_config.yaml            # Benchmark parameters
│
├── cmake/                               # CMake modules
│   ├── FindLlamaCpp.cmake               # (TODO)
│   └── CUDAConfig.cmake                 # (TODO)
│
└── python/                              # Python bindings (TODO)
    ├── cobali/
    │   ├── __init__.py
    │   ├── bindings.cpp                 # pybind11 bindings
    │   ├── engine_wrapper.py            # Python wrapper
    │   └── utils/
    │       ├── logging.py
    │       ├── metrics.py
    │       └── config.py
    └── setup.py                         # Python package setup
```

## File Statistics

- **Total C++ headers**: 11 files
- **Total C++ sources**: 10 files
- **Total CUDA files**: 0 (Phase 4 - to be implemented)
- **Example programs**: 3 C++
- **Test files**: 4 C++
- **Documentation**: 4 markdown files (+ 2 TODO)
- **Scripts**: 3 shell scripts
- **Config files**: 3 YAML files

## Key Files for GPU Course Project

### Phase 1: Baseline
- `src/baseline/sequential_engine.cpp` - Sequential inference engine
- `examples/cpp/baseline_inference.cpp` - Example usage

### Phase 2: Continuous Batching (YOUR WORK)
- `include/cobali/scheduler/continuous_batcher.h` - Header
- `src/scheduler/continuous_batcher.cpp` - ⭐ YOUR IMPLEMENTATION
- `src/scheduler/batch_manager.cpp` - Batch formation logic
- `examples/cpp/continuous_batching.cpp` - Example

### Phase 3: Prefill Splitting (YOUR WORK)
- `include/cobali/scheduler/prefill_splitter.h` - Header
- `src/scheduler/prefill_splitter.cpp` - ⭐ YOUR IMPLEMENTATION
- `examples/cpp/full_cobali.cpp` - Example

### Phase 4: Custom CUDA Kernels (FUTURE WORK)
- `include/cobali/kernels/*.cuh` - CUDA headers
- `src/kernels/*.cu` - ⭐ YOUR CUDA IMPLEMENTATIONS

## Build Artifacts (Not in Git)

```
build/                                   # CMake build directory
├── cobali_main                          # Main executable
├── libcobali.so                         # Shared library
├── examples/
│   ├── example_baseline
│   ├── example_continuous_batching
│   └── example_full_cobali
└── tests/
    ├── test_request_queue
    ├── test_batch_manager
    ├── test_prefill_splitter
    └── test_kv_cache
```

## Next Steps for Development

1. **Build everything**: `./scripts/setup.sh`
2. **Download model**: `./scripts/download_model.sh`
3. **Run baseline**: `./build/cobali_main baseline models/qwen2-0_5b-instruct-q4_0.gguf`
4. **Measure performance**: `python benchmarks/run_baseline.py`
5. **Optimize Phase 2**: Tune batch sizes, measure improvements
6. **Optimize Phase 3**: Tune chunk sizes, measure TTFT
7. **Implement Phase 4**: Write custom CUDA kernels
8. **Write report**: Document results in `docs/06_results_analysis.md`

