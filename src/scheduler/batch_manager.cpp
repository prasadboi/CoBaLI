#include "cobali/scheduler/batch_manager.h"
#include "cobali/common/utils.h"
#include <algorithm>

namespace cobali {

BatchManager::BatchManager(const EngineConfig& config)
    : config_(config)
    , max_batch_size_(config.max_batch_size)
    , max_tokens_per_batch_(config.max_tokens_per_batch) {
}

BatchManager::~BatchManager() {
}

Batch BatchManager::formBatch(RequestQueue& pending_queue, 
                               std::vector<Request*>& active_requests) {
    Batch batch;
    
    // First, add all active requests (decode phase) to batch
    for (auto* req : active_requests) {
        if (batch.size() >= static_cast<size_t>(max_batch_size_)) {
            break;
        }
        
        if (canFitInBatch(req, batch)) {
            batch.requests.push_back(req);
            int tokens = getRequestTokenBudget(req);
            batch.total_tokens += tokens;
            
            if (req->phase == Phase::PREFILL) {
                batch.num_prefill_requests++;
            } else if (req->phase == Phase::DECODE) {
                batch.num_decode_requests++;
            }
        }
    }
    
    // Then, try to add new requests from pending queue
    while (batch.size() < static_cast<size_t>(max_batch_size_)) {
        Request* req = pending_queue.tryDequeue();
        if (req == nullptr) {
            break; // No more pending requests
        }
        
        if (canFitInBatch(req, batch)) {
            batch.requests.push_back(req);
            int tokens = getRequestTokenBudget(req);
            batch.total_tokens += tokens;
            
            if (req->phase == Phase::PREFILL || req->phase == Phase::WAITING) {
                batch.num_prefill_requests++;
                req->phase = Phase::PREFILL;
            } else if (req->phase == Phase::DECODE) {
                batch.num_decode_requests++;
            }
        } else {
            // Can't fit this request, put it back
            pending_queue.enqueue(req);
            break;
        }
    }
    
    return batch;
}

bool BatchManager::canFitInBatch(const Request* request, const Batch& current_batch) const {
    // Check batch size limit
    if (current_batch.size() >= static_cast<size_t>(max_batch_size_)) {
        return false;
    }
    
    // Check token budget
    int tokens_needed = getRequestTokenBudget(request);
    if (current_batch.total_tokens + tokens_needed > max_tokens_per_batch_) {
        return false;
    }
    
    // Check memory constraints (simplified - should check KV cache availability)
    // TODO: Add proper memory checking via KVCacheManager
    
    return true;
}

int BatchManager::getRequestTokenBudget(const Request* request) const {
    if (request->phase == Phase::PREFILL || request->phase == Phase::WAITING) {
        // For prefill, budget is the chunk size (will be determined by PrefillSplitter)
        // For now, use a default or remaining tokens
        int remaining = request->prompt_length - request->tokens_processed;
        if (config_.enable_prefill_splitting) {
            return std::min(remaining, config_.prefill_chunk_size);
        } else {
            return remaining;
        }
    } else if (request->phase == Phase::DECODE) {
        // For decode, budget is 1 token per step
        return 1;
    }
    
    return 0;
}

float BatchManager::calculatePriorityScore(const Request* request) const {
    float score = 0.0f;
    
    // Base priority
    score += static_cast<float>(request->priority);
    
    // Prefer decode over prefill (if configured)
    if (config_.enable_prefill_splitting && request->phase == Phase::DECODE) {
        score += config_.decode_priority_weight;
    }
    
    // Penalize waiting time (to prevent starvation)
    auto now = std::chrono::steady_clock::now();
    double wait_time_ms = utils::getElapsedMs(request->arrival_time, now);
    score += wait_time_ms / 1000.0f; // Add 1 point per second of waiting
    
    return score;
}

bool BatchManager::hasMemoryForRequest(const Request* request, 
                                       const Batch& current_batch) const {
    // Simplified memory check
    // TODO: Implement proper KV cache memory tracking
    return true;
}

} // namespace cobali

