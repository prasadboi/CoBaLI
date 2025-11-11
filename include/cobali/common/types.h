#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <chrono>

namespace cobali {

// Request phases
enum class Phase {
    WAITING,    // In queue, not yet started
    PREFILL,    // Processing prompt tokens
    DECODE,     // Generating output tokens
    COMPLETED,  // Generation finished
    FAILED      // Error occurred
};

// Request priority (for scheduling)
enum class Priority {
    LOW = 0,
    NORMAL = 1,
    HIGH = 2
};

// Forward declarations
class Request;
class KVCacheSlot;

// Request ID type
using RequestID = uint64_t;

// Token type
using Token = int32_t;

// Timestamp type
using Timestamp = std::chrono::time_point<std::chrono::steady_clock>;

// Request structure - represents a single inference request
struct Request {
    RequestID id;                           // Unique request identifier
    
    // Input
    std::vector<Token> prompt_tokens;       // Input prompt tokens
    int prompt_length;                      // Length of prompt
    
    // Output
    std::vector<Token> output_tokens;       // Generated tokens
    int max_output_length;                  // Maximum tokens to generate
    
    // Generation parameters
    float temperature;                      // Sampling temperature
    float top_p;                            // Nucleus sampling parameter
    int top_k;                              // Top-k sampling parameter
    
    // State tracking
    Phase phase;                            // Current phase
    Priority priority;                      // Scheduling priority
    int tokens_processed;                   // Tokens processed in prefill
    int tokens_generated;                   // Tokens generated in decode
    
    // Prefill splitting state
    int current_chunk_size;                 // Current chunk size for prefill
    int prefill_chunks_completed;           // Number of chunks processed
    
    // KV cache
    KVCacheSlot* kv_cache;                  // Pointer to KV cache slot
    
    // Timing
    Timestamp arrival_time;                 // When request arrived
    Timestamp prefill_start_time;           // When prefill started
    Timestamp first_token_time;             // When first token was generated
    Timestamp completion_time;              // When request completed
    
    // Constructor
    Request(RequestID req_id, 
            const std::vector<Token>& tokens,
            int max_output_len = 256,
            float temp = 1.0f,
            float topp = 0.9f,
            int topk = 40)
        : id(req_id)
        , prompt_tokens(tokens)
        , prompt_length(static_cast<int>(tokens.size()))
        , max_output_length(max_output_len)
        , temperature(temp)
        , top_p(topp)
        , top_k(topk)
        , phase(Phase::WAITING)
        , priority(Priority::NORMAL)
        , tokens_processed(0)
        , tokens_generated(0)
        , current_chunk_size(0)
        , prefill_chunks_completed(0)
        , kv_cache(nullptr)
        , arrival_time(std::chrono::steady_clock::now())
    {}
    
    // Check if request is completed
    bool isCompleted() const {
        return phase == Phase::COMPLETED || phase == Phase::FAILED;
    }
    
    // Check if in prefill phase
    bool isPrefilling() const {
        return phase == Phase::PREFILL && tokens_processed < prompt_length;
    }
    
    // Check if in decode phase
    bool isDecoding() const {
        return phase == Phase::DECODE;
    }
    
    // Get total tokens (prompt + generated)
    int getTotalTokens() const {
        return prompt_length + tokens_generated;
    }
};

// Batch of requests to process together
struct Batch {
    std::vector<Request*> requests;         // Requests in this batch
    int total_tokens;                       // Total tokens across all requests
    int num_prefill_requests;               // Number of prefill requests
    int num_decode_requests;                // Number of decode requests
    
    Batch() 
        : total_tokens(0)
        , num_prefill_requests(0)
        , num_decode_requests(0)
    {}
    
    void clear() {
        requests.clear();
        total_tokens = 0;
        num_prefill_requests = 0;
        num_decode_requests = 0;
    }
    
    bool empty() const {
        return requests.empty();
    }
    
    size_t size() const {
        return requests.size();
    }
};

// KV Cache slot - represents memory for a single request's KV cache
struct KVCacheSlot {
    int slot_id;                            // Slot identifier
    void* k_cache;                          // Key cache GPU pointer
    void* v_cache;                          // Value cache GPU pointer
    size_t max_seq_len;                     // Maximum sequence length
    size_t current_len;                     // Current sequence length
    bool is_allocated;                      // Is this slot in use?
    RequestID owner_id;                     // Which request owns this slot
    
    KVCacheSlot()
        : slot_id(-1)
        , k_cache(nullptr)
        , v_cache(nullptr)
        , max_seq_len(0)
        , current_len(0)
        , is_allocated(false)
        , owner_id(0)
    {}
};

// Performance metrics
struct Metrics {
    // Throughput
    double requests_per_second;
    double tokens_per_second;
    
    // Latency
    double avg_time_to_first_token_ms;      // TTFT - prefill latency
    double avg_per_token_latency_ms;        // Decode latency per token
    double avg_end_to_end_latency_ms;       // Total request latency
    
    // Batch statistics
    double avg_batch_size;
    double avg_prefill_ratio;               // Ratio of prefill vs decode
    
    // GPU utilization
    double gpu_utilization_percent;
    double memory_used_gb;
    
    Metrics()
        : requests_per_second(0.0)
        , tokens_per_second(0.0)
        , avg_time_to_first_token_ms(0.0)
        , avg_per_token_latency_ms(0.0)
        , avg_end_to_end_latency_ms(0.0)
        , avg_batch_size(0.0)
        , avg_prefill_ratio(0.0)
        , gpu_utilization_percent(0.0)
        , memory_used_gb(0.0)
    {}
};

} // namespace cobali

