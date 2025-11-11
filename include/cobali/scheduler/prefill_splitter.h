#pragma once

#include "cobali/common/types.h"
#include "cobali/common/config.h"
#include <vector>

namespace cobali {

// Prefill splitting / chunking logic
// Breaks large prefills into smaller chunks and interleaves with decode
// This is YOUR Phase 3 implementation
class PrefillSplitter {
public:
    explicit PrefillSplitter(const EngineConfig& config);
    ~PrefillSplitter();
    
    // Determine chunk size for this prefill request
    // Takes into account:
    // - Base chunk size from config
    // - Available token budget
    // - Number of decode requests (fairness)
    int getChunkSize(const Request* request, 
                     int available_tokens,
                     int num_decode_requests) const;
    
    // Should this prefill request be chunked?
    bool shouldSplit(const Request* request) const;
    
    // Calculate fairness weight for prefill vs decode
    // Higher decode_priority_weight = favor decode requests
    float calculateFairnessWeight(int num_prefill, int num_decode) const;
    
    // Get remaining tokens to process for a prefill request
    int getRemainingPrefillTokens(const Request* request) const;
    
    // Check if prefill is complete for a request
    bool isPrefillComplete(const Request* request) const;
    
    // Update request state after processing a chunk
    void updateAfterChunk(Request* request, int tokens_processed);
    
private:
    const EngineConfig& config_;
    int base_chunk_size_;              // Base chunk size (e.g., 512)
    float decode_priority_weight_;     // Priority for decode vs prefill
    
    // Adaptive chunking parameters
    int min_chunk_size_;               // Minimum chunk size (e.g., 128)
    int max_chunk_size_;               // Maximum chunk size (e.g., 1024)
};

} // namespace cobali

