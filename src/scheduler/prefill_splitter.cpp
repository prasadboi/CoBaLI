#include "cobali/scheduler/prefill_splitter.h"
#include <algorithm>
#include <cmath>

namespace cobali {

PrefillSplitter::PrefillSplitter(const EngineConfig& config)
    : config_(config)
    , base_chunk_size_(config.prefill_chunk_size)
    , decode_priority_weight_(config.decode_priority_weight)
    , min_chunk_size_(std::max(64, base_chunk_size_ / 4))
    , max_chunk_size_(std::min(2048, base_chunk_size_ * 2)) {
}

PrefillSplitter::~PrefillSplitter() {
}

int PrefillSplitter::getChunkSize(const Request* request, 
                                  int available_tokens,
                                  int num_decode_requests) const {
    if (!shouldSplit(request)) {
        // Don't split, process all remaining tokens
        return getRemainingPrefillTokens(request);
    }
    
    // Start with base chunk size
    int chunk_size = base_chunk_size_;
    
    // Adjust based on fairness weight
    if (num_decode_requests > 0) {
        float fairness = calculateFairnessWeight(1, num_decode_requests);
        // Reduce chunk size if there are many decode requests waiting
        chunk_size = static_cast<int>(chunk_size * fairness);
    }
    
    // Respect available token budget
    chunk_size = std::min(chunk_size, available_tokens);
    
    // Don't exceed remaining tokens
    int remaining = getRemainingPrefillTokens(request);
    chunk_size = std::min(chunk_size, remaining);
    
    // Enforce min/max bounds
    chunk_size = std::max(min_chunk_size_, std::min(chunk_size, max_chunk_size_));
    
    // If we're close to finishing, just process all remaining tokens
    if (remaining - chunk_size < min_chunk_size_) {
        chunk_size = remaining;
    }
    
    return chunk_size;
}

bool PrefillSplitter::shouldSplit(const Request* request) const {
    if (!config_.enable_prefill_splitting) {
        return false;
    }
    
    int remaining = getRemainingPrefillTokens(request);
    
    // Only split if remaining tokens exceed chunk size
    return remaining > base_chunk_size_;
}

float PrefillSplitter::calculateFairnessWeight(int num_prefill, int num_decode) const {
    if (num_decode == 0) {
        return 1.0f; // No decode requests, no need to reduce
    }
    
    // Calculate ratio of decode to total requests
    float decode_ratio = static_cast<float>(num_decode) / (num_prefill + num_decode);
    
    // Weight: higher decode_priority_weight means smaller chunks for prefill
    // If decode_priority_weight = 0.7 and decode_ratio = 0.8, then fairness = 0.44
    float fairness = 1.0f - (decode_priority_weight_ * decode_ratio);
    
    // Ensure fairness is at least 0.2 (don't make chunks too small)
    return std::max(0.2f, fairness);
}

int PrefillSplitter::getRemainingPrefillTokens(const Request* request) const {
    return request->prompt_length - request->tokens_processed;
}

bool PrefillSplitter::isPrefillComplete(const Request* request) const {
    return request->tokens_processed >= request->prompt_length;
}

void PrefillSplitter::updateAfterChunk(Request* request, int tokens_processed) {
    request->tokens_processed += tokens_processed;
    request->prefill_chunks_completed++;
    
    // Transition to decode phase if prefill is complete
    if (isPrefillComplete(request)) {
        request->phase = Phase::DECODE;
        request->first_token_time = std::chrono::steady_clock::now();
    }
}

} // namespace cobali

