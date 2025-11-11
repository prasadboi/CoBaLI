#pragma once

#include <string>
#include <cstdint>

namespace cobali {

// Engine configuration
struct EngineConfig {
    // Model configuration
    std::string model_path;                 // Path to GGUF model file
    int n_gpu_layers;                       // Number of layers to offload to GPU (-1 = all)
    int n_ctx;                              // Context size (max sequence length)
    
    // Batching configuration
    int max_batch_size;                     // Maximum number of requests in a batch
    int max_tokens_per_batch;               // Maximum total tokens per batch
    bool enable_continuous_batching;        // Enable continuous batching
    
    // Prefill splitting configuration
    bool enable_prefill_splitting;          // Enable chunked prefill
    int prefill_chunk_size;                 // Tokens per prefill chunk (e.g., 512)
    float decode_priority_weight;           // Priority weight for decode vs prefill (0.0-1.0)
    
    // Memory configuration
    size_t kv_cache_size_mb;                // Total KV cache size in MB
    int max_concurrent_requests;            // Maximum concurrent requests
    
    // Performance tuning
    int num_threads;                        // CPU threads for computation
    bool use_mmap;                          // Use mmap for model loading
    bool use_mlock;                         // Lock model in RAM
    
    // Scheduling
    int scheduling_policy;                  // 0=FCFS, 1=Priority, 2=Fair
    int preemption_mode;                    // 0=None, 1=Recompute, 2=Swap
    
    // Logging
    bool verbose;                           // Enable verbose logging
    std::string log_file;                   // Log file path (empty = stdout)
    
    // Default constructor with reasonable defaults
    EngineConfig()
        : model_path("")
        , n_gpu_layers(-1)
        , n_ctx(2048)
        , max_batch_size(32)
        , max_tokens_per_batch(4096)
        , enable_continuous_batching(false)
        , enable_prefill_splitting(false)
        , prefill_chunk_size(512)
        , decode_priority_weight(0.7f)
        , kv_cache_size_mb(4096)
        , max_concurrent_requests(64)
        , num_threads(8)
        , use_mmap(true)
        , use_mlock(false)
        , scheduling_policy(0)
        , preemption_mode(0)
        , verbose(false)
        , log_file("")
    {}
};

// Benchmark configuration
struct BenchmarkConfig {
    // Workload
    int num_requests;                       // Total requests to send
    double request_rate;                    // Requests per second (0 = as fast as possible)
    int min_prompt_length;                  // Minimum prompt tokens
    int max_prompt_length;                  // Maximum prompt tokens
    int min_output_length;                  // Minimum output tokens
    int max_output_length;                  // Maximum output tokens
    
    // Timing
    double duration_seconds;                // Run for this many seconds (0 = until all requests done)
    double warmup_seconds;                  // Warmup period before measuring
    
    // Output
    std::string output_file;                // CSV file for results
    bool save_per_request_stats;            // Save detailed per-request statistics
    
    BenchmarkConfig()
        : num_requests(100)
        , request_rate(0.0)
        , min_prompt_length(128)
        , max_prompt_length(512)
        , min_output_length(32)
        , max_output_length(128)
        , duration_seconds(0.0)
        , warmup_seconds(5.0)
        , output_file("benchmark_results.csv")
        , save_per_request_stats(true)
    {}
};

} // namespace cobali

