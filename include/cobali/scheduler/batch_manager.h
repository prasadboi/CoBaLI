#pragma once

#include "cobali/common/types.h"
#include "cobali/common/config.h"
#include "cobali/scheduler/request_queue.h"
#include <vector>
#include <memory>

namespace cobali {

// Manages batch formation for continuous batching
// Decides which requests to include in the next batch
class BatchManager {
public:
    explicit BatchManager(const EngineConfig& config);
    ~BatchManager();
    
    // Form a batch from available requests
    // Returns a batch of requests ready for execution
    Batch formBatch(RequestQueue& pending_queue, 
                    std::vector<Request*>& active_requests);
    
    // Check if a request can fit in the current batch
    bool canFitInBatch(const Request* request, 
                       const Batch& current_batch) const;
    
    // Calculate token budget for a request
    // For prefill: returns chunk size
    // For decode: returns 1
    int getRequestTokenBudget(const Request* request) const;
    
    // Get maximum batch size
    int getMaxBatchSize() const { return max_batch_size_; }
    
    // Get maximum tokens per batch
    int getMaxTokensPerBatch() const { return max_tokens_per_batch_; }
    
private:
    // Calculate priority score for request (used for scheduling)
    float calculatePriorityScore(const Request* request) const;
    
    // Check memory constraints
    bool hasMemoryForRequest(const Request* request, 
                            const Batch& current_batch) const;
    
    const EngineConfig& config_;
    int max_batch_size_;
    int max_tokens_per_batch_;
};

} // namespace cobali

