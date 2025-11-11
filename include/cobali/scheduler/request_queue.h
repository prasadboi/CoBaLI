#pragma once

#include "cobali/common/types.h"
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <functional>

namespace cobali {

// Thread-safe request queue
// Supports priority-based and FIFO ordering
class RequestQueue {
public:
    RequestQueue();
    ~RequestQueue();
    
    // Add a request to the queue
    void enqueue(Request* request);
    
    // Remove and return the next request (blocks if empty)
    Request* dequeue();
    
    // Try to dequeue without blocking (returns nullptr if empty)
    Request* tryDequeue();
    
    // Peek at the next request without removing it
    Request* peek();
    
    // Get all requests matching a predicate
    std::vector<Request*> getMatching(std::function<bool(const Request*)> predicate);
    
    // Remove specific request from queue
    bool remove(RequestID request_id);
    
    // Check if queue is empty
    bool empty() const;
    
    // Get queue size
    size_t size() const;
    
    // Clear all requests
    void clear();
    
private:
    // Priority comparator for requests
    struct RequestComparator {
        bool operator()(const Request* a, const Request* b) const {
            // Higher priority first, then FIFO (earlier arrival time)
            if (a->priority != b->priority) {
                return a->priority < b->priority; // Lower priority value = higher priority in max heap
            }
            return a->arrival_time > b->arrival_time; // Earlier arrival first
        }
    };
    
    std::priority_queue<Request*, std::vector<Request*>, RequestComparator> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
};

} // namespace cobali

