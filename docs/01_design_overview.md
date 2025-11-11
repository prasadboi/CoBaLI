# CoBaLI Design Overview

## Project Goal

CoBaLI (Continuous Batching and Prefill Splitting for LLM Inference) is a GPU programming course project that demonstrates how to implement two key optimizations for LLM inference:

1. **Continuous Batching**: Dynamic batch formation that adds/removes requests during execution
2. **Prefill Splitting**: Breaking large prefill operations into chunks to improve fairness

## Architecture

### Three-Layer Design

```
┌─────────────────────────────────────────────────────────────┐
│ Python Layer (Orchestration)                                 │
│  - Configuration management                                  │
│  - Benchmarking harness                                      │
│  - Visualization                                             │
└───────────────────────┬─────────────────────────────────────┘
                        │ pybind11
┌───────────────────────┴─────────────────────────────────────┐
│ C++ Layer (Scheduling & Control)                            │
│  - Request queue management                                  │
│  - Batch formation (BatchManager)                            │
│  - Continuous batching scheduler (ContinuousBatcher)         │
│  - Prefill splitting logic (PrefillSplitter)                 │
│  - KV cache management                                       │
└───────────────────────┬─────────────────────────────────────┘
                        │ CUDA API
┌───────────────────────┴─────────────────────────────────────┐
│ CUDA Layer (GPU Execution)                                  │
│  - llama.cpp kernels (Phase 1-3)                            │
│  - Custom CUDA kernels (Phase 4)                            │
│    * Batched attention                                       │
│    * Prefill kernel                                          │
│    * Decode kernel                                           │
└─────────────────────────────────────────────────────────────┘
```

## Progressive Implementation

### Phase 1: Baseline (Sequential)
- Single request at a time
- No batching
- Uses llama.cpp kernels directly
- **Goal**: Establish performance baseline

### Phase 2: Continuous Batching
- Dynamic batch formation
- Mix decode-phase requests in same batch
- Add new requests without waiting for batch completion
- **Goal**: 2-3x throughput improvement

### Phase 3: Prefill Splitting
- Break large prefills into chunks (e.g., 512 tokens)
- Interleave prefill chunks with decode steps
- Fairness scheduling (prioritize decode)
- **Goal**: 5-10x throughput improvement, better TTFT

### Phase 4: Custom CUDA Kernels (Future)
- Replace llama.cpp kernels with custom implementations
- Batched attention kernel
- Optimized memory access patterns
- **Goal**: Additional 1.5-2x speedup

## Key Components

### 1. Request Management
- **Request**: Represents a single inference request
- **RequestQueue**: Thread-safe priority queue
- **Lifecycle**: WAITING → PREFILL → DECODE → COMPLETED

### 2. Batch Scheduling
- **BatchManager**: Decides which requests go in next batch
- **ContinuousBatcher**: Maintains active requests, adds new ones
- **Constraints**: Max batch size, max tokens, memory limits

### 3. Prefill Splitting
- **PrefillSplitter**: Determines chunk sizes
- **Adaptive chunking**: Smaller chunks when many decode requests
- **Fairness weight**: Configurable priority (decode vs prefill)

### 4. Memory Management
- **KVCacheManager**: Allocates GPU memory for KV cache
- **Per-request slots**: Each request gets own KV cache
- **Dynamic allocation**: Allocate on request start, free on completion

### 5. Execution
- **Executor**: Calls GPU kernels (llama.cpp or custom)
- **Batched operations**: Process multiple requests together
- **Token sampling**: Per-request sampling (temperature, top-p, top-k)

## Design Decisions

### Why C++ for Scheduling?
- This is a GPU programming course
- Demonstrates host-side optimization
- Low-level control over memory and threads
- Performance critical for real systems

### Why llama.cpp Initially?
- Well-tested GPU kernels
- Focus on scheduling logic first
- Validate correctness before custom kernels
- Easier to debug

### Why Separate Prefill/Decode?
- Different compute characteristics:
  - Prefill: Large matrix multiply, compute-bound
  - Decode: Single token, memory-bound
- Different priority:
  - Decode is latency-sensitive
  - Prefill can be chunked

## Performance Metrics

### Throughput
- **Requests/second**: System throughput
- **Tokens/second**: Total token generation rate

### Latency
- **TTFT**: Time to first token (prefill latency)
- **TPOT**: Time per output token (decode latency)
- **E2E**: End-to-end request latency

### Utilization
- **Batch size**: Average batch size over time
- **GPU utilization**: GPU compute usage
- **Memory utilization**: KV cache memory usage

## References
- [Orca: vLLM Continuous Batching](https://arxiv.org/abs/2309.06180)
- [FlashAttention: Fast Attention](https://arxiv.org/abs/2205.14135)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)

