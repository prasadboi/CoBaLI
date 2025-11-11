# Phase 2: Continuous Batching

## Overview

Continuous batching dynamically adds and removes requests from batches **during execution**, unlike traditional batching that waits for all requests to complete.

## Key Insight

In LLM inference:
- Decode phase generates **one token at a time**
- Different requests can be at different stages
- We can batch requests even if they started at different times

## Algorithm

### Batch Formation

```cpp
Batch ContinuousBatcher::getNextBatch() {
    Batch batch;
    
    // 1. Add active requests (already generating)
    for (auto* req : active_requests) {
        if (canFitInBatch(req, batch)) {
            batch.add(req);
        }
    }
    
    // 2. Add new requests from queue
    while (!pending_queue.empty() && batch.size() < max_batch_size) {
        Request* req = pending_queue.tryDequeue();
        if (canFitInBatch(req, batch)) {
            batch.add(req);
            active_requests.push_back(req);
        }
    }
    
    return batch;
}
```

### Execution Loop

```cpp
while (hasRequests()) {
    // Form batch
    Batch batch = batcher->getNextBatch();
    
    // Execute batch on GPU
    executor->executeMixedBatch(batch);
    
    // Remove completed requests
    batcher->removeCompletedRequests();
}
```

## Implementation Details

### Request States

```
WAITING → PREFILL → DECODE → COMPLETED
   ↓         ↓         ↓
   └─────────┴─────────┴─> Can be batched together!
```

### Batch Constraints

1. **Max batch size**: e.g., 32 requests
2. **Max tokens per batch**: e.g., 4096 tokens
3. **Memory**: Available KV cache slots

### Token Budget Calculation

```cpp
int getRequestTokenBudget(const Request* req) {
    if (req->phase == PREFILL) {
        return req->prompt_length - req->tokens_processed;
    } else if (req->phase == DECODE) {
        return 1;  // One token per decode step
    }
}
```

## Performance Improvements

### Throughput
- **3-5x improvement** over baseline
- Batches multiple decode requests
- Reduces GPU idle time
- Better GPU utilization

### Latency
- TTFT improved for concurrent requests
- Decode latency similar to baseline
- Queue time reduced

### GPU Utilization
- 60-80% utilization (vs 30-40% baseline)
- Batch size scales with load
- Better memory bandwidth usage

## Example Usage

```bash
# Build
./scripts/build_cpp.sh

# Run continuous batching
./build/cobali_main batching models/qwen-0.5b.gguf \
    --max-batch-size 16 \
    --max-tokens 4096

# Or use example
./build/examples/example_continuous_batching models/qwen-0.5b.gguf
```

## Benchmark Results (Example)

```
Configuration:
  Model: Qwen 0.5B Q4_0
  GPU: RTX 2080 Ti
  Batch size: 16
  Workload: 100 concurrent requests
  
Baseline:
  Throughput: 4.4 req/sec
  Avg latency: 1800 ms
  GPU util: 35%

Continuous Batching:
  Throughput: 15.2 req/sec (3.5x)
  Avg latency: 650 ms (2.8x better)
  GPU util: 72%
```

## Code Structure

### Key Files
- `src/scheduler/continuous_batcher.cpp`: Main batching logic
- `src/scheduler/batch_manager.cpp`: Batch formation
- `src/scheduler/request_queue.cpp`: Request queue
- `src/engine/executor.cpp`: Batch execution

### Key Functions

**ContinuousBatcher:**
- `addRequest()`: Add new request
- `getNextBatch()`: Form next batch
- `updateAfterExecution()`: Update request states
- `removeCompletedRequests()`: Cleanup

**BatchManager:**
- `formBatch()`: Create batch from requests
- `canFitInBatch()`: Check constraints
- `getRequestTokenBudget()`: Calculate tokens needed

## Challenges

### 1. KV Cache Management
- Each request needs separate KV cache
- Must track per-request cache positions
- Dynamic allocation/deallocation

### 2. Variable Sequence Lengths
- Requests at different positions
- Attention masks must be correct
- Batched operations must handle variable lengths

### 3. Request Completion
- Requests complete at different times
- Must remove from batch mid-execution
- Handle EOS tokens per-request

## Next Steps

Phase 3 adds **prefill splitting** to further improve throughput and fairness.

