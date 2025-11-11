# Phase 3: Prefill Splitting

## Overview

Prefill splitting breaks large prefill operations into **chunks** and interleaves them with decode steps from other requests. This improves fairness and throughput.

## Motivation

### Problem with Continuous Batching Alone

When a request with a long prompt arrives:
1. Must process entire prefill before decode
2. Blocks other decode requests (high latency)
3. Large prefill can take 100-500ms
4. Decode requests wait → poor TTFT

### Solution: Chunked Prefill

Break prefill into chunks (e.g., 512 tokens):
```
Request A: [PREFILL-CHUNK-1] → [PREFILL-CHUNK-2] → [DECODE] → [DECODE] → ...
Request B:                      [DECODE] → [DECODE] → [DECODE] → ...
```

## Algorithm

### Chunk Size Calculation

```cpp
int PrefillSplitter::getChunkSize(
    const Request* req,
    int available_tokens,
    int num_decode_requests
) {
    int chunk_size = base_chunk_size_;  // e.g., 512
    
    // Adjust for fairness
    if (num_decode_requests > 0) {
        float fairness = calculateFairnessWeight(1, num_decode_requests);
        chunk_size = chunk_size * fairness;
    }
    
    // Respect constraints
    chunk_size = min(chunk_size, available_tokens);
    chunk_size = min(chunk_size, remainingTokens(req));
    
    return chunk_size;
}
```

### Fairness Weight

```cpp
float calculateFairnessWeight(int num_prefill, int num_decode) {
    float decode_ratio = num_decode / (num_prefill + num_decode);
    float fairness = 1.0 - (decode_priority_weight * decode_ratio);
    return max(0.2, fairness);  // Minimum 20% of base chunk
}
```

### Batch Formation with Splitting

```cpp
for (auto* req : batch.requests) {
    if (req->phase == PREFILL) {
        // Determine chunk size
        req->current_chunk_size = splitter->getChunkSize(
            req, 
            available_tokens,
            num_decode_requests
        );
        
        // Process chunk
        executor->executePrefillChunk(req);
        
        // Update state
        req->tokens_processed += req->current_chunk_size;
        
        // Transition to decode if done
        if (req->tokens_processed >= req->prompt_length) {
            req->phase = DECODE;
        }
    }
}
```

## Configuration Parameters

### Chunk Size
```yaml
prefill_chunk_size: 512  # Base chunk size
```
- **Smaller**: Better fairness, more overhead
- **Larger**: Better throughput, worse fairness
- **Typical**: 256-1024 tokens

### Decode Priority Weight
```yaml
decode_priority_weight: 0.7  # 0.0 - 1.0
```
- **0.0**: No priority (equal treatment)
- **1.0**: Maximum priority (decode first)
- **0.7**: Balanced (recommended)

## Performance Improvements

### Throughput
- **5-10x improvement** over baseline
- **2-3x improvement** over continuous batching alone
- Better batch utilization
- Reduced idle time

### Latency
- **Much better TTFT** for queued requests
- Decode latency unchanged
- Fair scheduling prevents starvation

### Fairness
- Long prompts don't block short requests
- Predictable latency
- Better user experience

## Example Usage

```bash
# Build
./scripts/build_cpp.sh

# Run full CoBaLI
./build/cobali_main full models/qwen-0.5b.gguf \
    --prefill-chunk-size 512 \
    --decode-priority 0.7

# Or use example
./build/examples/example_full_cobali models/qwen-0.5b.gguf
```

## Benchmark Results (Example)

```
Configuration:
  Model: Qwen 0.5B Q4_0
  GPU: RTX 2080 Ti
  Batch size: 32
  Chunk size: 512
  Workload: 100 requests (varying prompt lengths)

Baseline:
  Throughput: 4.4 req/sec
  TTFT (p50): 850 ms
  TTFT (p99): 4500 ms
  GPU util: 35%

Continuous Batching:
  Throughput: 15.2 req/sec
  TTFT (p50): 320 ms
  TTFT (p99): 2100 ms
  GPU util: 72%

Full CoBaLI (Batching + Splitting):
  Throughput: 38.5 req/sec (8.7x vs baseline)
  TTFT (p50): 180 ms (4.7x better)
  TTFT (p99): 680 ms (6.6x better)
  GPU util: 85%
```

## Code Structure

### Key Files
- `src/scheduler/prefill_splitter.cpp`: Chunking logic
- `src/engine/cobali_engine.cpp`: Main engine with splitting
- `src/engine/executor.cpp`: Chunked execution

### Key Functions

**PrefillSplitter:**
- `getChunkSize()`: Determine chunk size
- `shouldSplit()`: Check if splitting needed
- `calculateFairnessWeight()`: Fairness calculation
- `updateAfterChunk()`: Update request state

## Trade-offs

### Advantages
✅ Better throughput
✅ Better fairness
✅ Lower TTFT variance
✅ Higher GPU utilization

### Disadvantages
❌ More complex scheduling
❌ KV cache fragmentation
❌ Slightly higher overhead per chunk

## Tuning Guidelines

### For Maximum Throughput
- Larger chunk size (1024)
- Lower decode priority (0.5)
- Larger batch size (64)

### For Best Fairness
- Smaller chunk size (256)
- Higher decode priority (0.8)
- Moderate batch size (32)

### For Balanced Performance
- Medium chunk size (512)
- Balanced priority (0.7)
- Medium batch size (32)

## Next Steps

Phase 4 will add **custom CUDA kernels** to further optimize:
- Batched attention (variable length)
- Fused operations
- Memory access patterns

