#pragma once

#include "cobali/common/types.h"
#include "cobali/common/config.h"
#include "cobali/scheduler/request_queue.h"
#include "cobali/scheduler/batch_manager.h"
#include <vector>
#include <memory>

namespace cobali {

// Continuous batching scheduler
// Dynamically adds/removes requests from batches during execution
// This is YOUR Phase 2 implementation
class ContinuousBatcher {
public:
    explicit ContinuousBatcher(const EngineConfig& config);
    ~ContinuousBatcher();
    
    // Add a new request to the scheduler
    void addRequest(Request* request);
    
    // Get the next batch to execute
    // Combines:
    // 1. Active requests that are still generating
    // 2. New requests from pending queue
    Batch getNextBatch();
    
    // Update request states after batch execution
    // Moves completed requests out of active set
    void updateAfterExecution(const Batch& executed_batch);
    
    // Remove completed requests
    void removeCompletedRequests();
    
    // Get statistics
    size_t getActiveRequestCount() const { return active_requests_.size(); }
    size_t getPendingRequestCount() const { return pending_queue_.size(); }
    
    // Check if there are any requests to process
    bool hasRequests() const;
    
private:
    const EngineConfig& config_;
    
    // Request queues
    RequestQueue pending_queue_;           // Requests waiting to start
    std::vector<Request*> active_requests_; // Requests currently being processed
    
    // Batch formation
    std::unique_ptr<BatchManager> batch_manager_;
    
    // Helper methods
    void moveRequestToActive(Request* request);
    void removeFromActive(RequestID request_id);
};

} // namespace cobali

