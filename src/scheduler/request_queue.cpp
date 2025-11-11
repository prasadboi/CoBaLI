#include "cobali/scheduler/request_queue.h"

namespace cobali {

RequestQueue::RequestQueue() {
}

RequestQueue::~RequestQueue() {
    clear();
}

void RequestQueue::enqueue(Request* request) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(request);
    cv_.notify_one();
}

Request* RequestQueue::dequeue() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return !queue_.empty(); });
    
    Request* req = queue_.top();
    queue_.pop();
    return req;
}

Request* RequestQueue::tryDequeue() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
        return nullptr;
    }
    
    Request* req = queue_.top();
    queue_.pop();
    return req;
}

Request* RequestQueue::peek() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
        return nullptr;
    }
    return queue_.top();
}

std::vector<Request*> RequestQueue::getMatching(std::function<bool(const Request*)> predicate) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<Request*> matches;
    
    // Note: This is inefficient for priority_queue
    // For production, consider using a different data structure
    // For now, we'll just return empty vector
    // TODO: Implement more efficient matching if needed
    
    return matches;
}

bool RequestQueue::remove(RequestID request_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Note: priority_queue doesn't support efficient removal
    // For production, consider using a different data structure
    // For now, return false (not supported)
    // TODO: Implement if needed
    
    return false;
}

bool RequestQueue::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
}

size_t RequestQueue::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

void RequestQueue::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    while (!queue_.empty()) {
        queue_.pop();
    }
}

} // namespace cobali

