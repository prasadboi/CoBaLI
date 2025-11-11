#include "cobali/scheduler/continuous_batcher.h"
#include "cobali/common/utils.h"

namespace cobali {

ContinuousBatcher::ContinuousBatcher(const EngineConfig& config)
    : config_(config)
    , batch_manager_(std::make_unique<BatchManager>(config)) {
}

ContinuousBatcher::~ContinuousBatcher() {
}

void ContinuousBatcher::addRequest(Request* request) {
    pending_queue_.enqueue(request);
    utils::Logger::getInstance().info(
        utils::format("Added request %lu to pending queue", request->id)
    );
}

Batch ContinuousBatcher::getNextBatch() {
    // Form batch from pending queue and active requests
    Batch batch = batch_manager_->formBatch(pending_queue_, active_requests_);
    
    // Move new requests to active
    for (auto* req : batch.requests) {
        bool is_new = true;
        for (auto* active : active_requests_) {
            if (active->id == req->id) {
                is_new = false;
                break;
            }
        }
        if (is_new) {
            moveRequestToActive(req);
        }
    }
    
    return batch;
}

void ContinuousBatcher::updateAfterExecution(const Batch& executed_batch) {
    // Update request states after execution
    // This will be called by the engine after processing a batch
    
    for (auto* req : executed_batch.requests) {
        // Check if request is completed
        if (req->phase == Phase::COMPLETED || req->phase == Phase::FAILED) {
            removeFromActive(req->id);
        }
    }
}

void ContinuousBatcher::removeCompletedRequests() {
    // Remove completed requests from active set
    auto it = active_requests_.begin();
    while (it != active_requests_.end()) {
        if ((*it)->isCompleted()) {
            utils::Logger::getInstance().info(
                utils::format("Request %lu completed", (*it)->id)
            );
            it = active_requests_.erase(it);
        } else {
            ++it;
        }
    }
}

bool ContinuousBatcher::hasRequests() const {
    return !pending_queue_.empty() || !active_requests_.empty();
}

void ContinuousBatcher::moveRequestToActive(Request* request) {
    // Check if already in active
    for (auto* active : active_requests_) {
        if (active->id == request->id) {
            return; // Already active
        }
    }
    
    active_requests_.push_back(request);
    utils::Logger::getInstance().debug(
        utils::format("Moved request %lu to active (total active: %zu)", 
                     request->id, active_requests_.size())
    );
}

void ContinuousBatcher::removeFromActive(RequestID request_id) {
    auto it = active_requests_.begin();
    while (it != active_requests_.end()) {
        if ((*it)->id == request_id) {
            it = active_requests_.erase(it);
            return;
        }
        ++it;
    }
}

} // namespace cobali

