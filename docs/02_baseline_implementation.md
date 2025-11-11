# Phase 1: Baseline Implementation

## Overview

The baseline implementation processes requests **one at a time** with no batching or optimizations. This serves as the performance baseline to measure improvements.

## Components

### SequentialEngine

```cpp
class SequentialEngine {
    bool initialize();
    std::vector<Token> generate(prompt_tokens, max_length, ...);
    void processRequest(Request* request);
};
```

**Behavior:**
1. Load model into GPU memory
2. Process single request at a time
3. Prefill entire prompt (no chunking)
4. Generate tokens one by one
5. Return result

## Implementation Details

### Model Loading
- Uses llama.cpp's `llama_load_model_from_file`
- Offloads all layers to GPU (`n_gpu_layers = -1`)
- Creates single context for inference

### Prefill Phase
```cpp
for (int i = 0; i < n_prompt; ++i) {
    llama_decode(ctx, llama_batch_get_one(&prompt_tokens[i], 1, i, 0));
}
```

### Decode Phase
```cpp
while (n_generated < max_output_length) {
    Token next_token = sampleToken(temperature, top_p, top_k);
    if (next_token == EOS) break;
    
    output_tokens.push_back(next_token);
    llama_decode(ctx, llama_batch_get_one(&next_token, 1, n_cur, 0));
}
```

## Performance Characteristics

### Throughput
- **Single request at a time**
- Throughput = 1 / (request_latency)
- Very low for concurrent workloads

### Latency
- Low latency for single request
- High latency when queue builds up
- No concurrency

### GPU Utilization
- Good for single large request
- Poor for small requests (underutilized)
- Idle time between requests

## Example Usage

```bash
# Build
./scripts/build_cpp.sh

# Run baseline
./build/cobali_main baseline models/qwen-0.5b.gguf \
    --prompt "Once upon a time" \
    --max-tokens 128 \
    --verbose

# Or use example
./build/examples/example_baseline models/qwen-0.5b.gguf
```

## Benchmark Results (Example)

```
Configuration:
  Model: Qwen 0.5B Q4_0
  GPU: RTX 2080 Ti
  Prompt length: 256 tokens
  Output length: 128 tokens

Results:
  Prefill time: 45 ms
  Decode time: 180 ms (1.4 ms/token)
  Total time: 225 ms
  Throughput: 4.4 requests/sec
  Token throughput: 568 tokens/sec
```

## Limitations

1. **No concurrency**: Wastes GPU cycles between requests
2. **No batching**: Can't process multiple requests together
3. **Long TTFT for queued requests**: Must wait for previous request
4. **Poor throughput**: Linear scaling with requests

## Next Steps

Phase 2 addresses these limitations with **continuous batching**.

