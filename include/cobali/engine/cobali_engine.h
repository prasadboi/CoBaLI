#pragma once

#include "cobali/common/types.h"
#include "cobali/common/config.h"
#include "cobali/scheduler/continuous_batcher.h"
#include "cobali/scheduler/prefill_splitter.h"
#include "cobali/memory/kv_cache_manager.h"
#include "cobali/engine/executor.h"
#include <memory>
#include <thread>
#include <atomic>

namespace cobali {

// Main CoBaLI inference engine
// Orchestrates continuous batching and prefill splitting
// This is the complete system combining Phases 2 & 3
class CoBaLIEngine {
public:
    explicit CoBaLIEngine(const EngineConfig& config);
    ~CoBaLIEngine();
    
    // Initialize engine (load model, allocate memory)
    bool initialize();
    
    // Start the inference loop in a background thread
    void start();
    
    // Stop the inference loop
    void stop();
    
    // Submit a request for processing (non-blocking)
    RequestID submitRequest(const std::vector<Token>& prompt_tokens,
                           int max_output_length = 256,
                           float temperature = 1.0f,
                           float top_p = 0.9f,
                           int top_k = 40);
    
    // Check if a request is complete
    bool isRequestComplete(RequestID request_id);
    
    // Get the result of a completed request
    std::vector<Token> getRequestResult(RequestID request_id);
    
    // Wait for a request to complete and return result (blocking)
    std::vector<Token> waitForRequest(RequestID request_id);
    
    // Get current metrics
    Metrics getMetrics() const;
    
    // Cleanup
    void cleanup();
    
private:
    const EngineConfig& config_;
    
    // Core components
    std::unique_ptr<ContinuousBatcher> batcher_;
    std::unique_ptr<PrefillSplitter> splitter_;
    std::unique_ptr<KVCacheManager> kv_cache_manager_;
    std::unique_ptr<Executor> executor_;
    
    // Request management
    RequestID next_request_id_;
    std::map<RequestID, std::unique_ptr<Request>> requests_;
    std::map<RequestID, std::vector<Token>> completed_results_;
    mutable std::mutex requests_mutex_;
    
    // Inference loop
    std::thread inference_thread_;
    std::atomic<bool> running_;
    void inferenceLoop();
    
    // Metrics tracking
    mutable std::mutex metrics_mutex_;
    Metrics current_metrics_;
    void updateMetrics();
    
    // Helper methods
    void processBatch(Batch& batch);
    void handleCompletedRequests();
    
    bool initialized_;
};

} // namespace cobali

