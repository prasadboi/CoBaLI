#pragma once

#include "cobali/common/types.h"
#include "cobali/common/config.h"
#include <vector>
#include <map>
#include <memory>

namespace cobali {

// Manages KV cache allocation for multiple requests
// Each request gets its own KV cache slot in GPU memory
class KVCacheManager {
public:
    explicit KVCacheManager(const EngineConfig& config);
    ~KVCacheManager();
    
    // Initialize GPU memory pool
    bool initialize(size_t model_kv_size_per_layer);
    
    // Allocate KV cache for a new request
    KVCacheSlot* allocate(RequestID request_id, size_t max_seq_len);
    
    // Free KV cache when request completes
    void free(RequestID request_id);
    
    // Get KV cache for a specific request
    KVCacheSlot* get(RequestID request_id);
    
    // Get KV cache pointers for a batch of requests
    std::vector<void*> getBatchKPointers(const std::vector<Request*>& batch);
    std::vector<void*> getBatchVPointers(const std::vector<Request*>& batch);
    
    // Check if we have memory for a new request
    bool hasMemoryFor(size_t max_seq_len) const;
    
    // Get memory statistics
    size_t getTotalMemoryBytes() const { return total_memory_bytes_; }
    size_t getUsedMemoryBytes() const { return used_memory_bytes_; }
    size_t getAvailableMemoryBytes() const { return total_memory_bytes_ - used_memory_bytes_; }
    float getMemoryUtilization() const { 
        return total_memory_bytes_ > 0 ? 
            static_cast<float>(used_memory_bytes_) / total_memory_bytes_ : 0.0f;
    }
    
    // Cleanup
    void cleanup();
    
private:
    const EngineConfig& config_;
    
    // Memory pool
    void* gpu_memory_pool_;                        // Large contiguous GPU buffer
    size_t total_memory_bytes_;                    // Total allocated memory
    size_t used_memory_bytes_;                     // Currently used memory
    size_t bytes_per_token_;                       // Bytes per token in KV cache
    
    // Slot management
    std::map<RequestID, std::unique_ptr<KVCacheSlot>> allocated_slots_;
    std::vector<int> free_slot_ids_;               // Available slot IDs
    int next_slot_id_;
    
    // Helper methods
    void* allocateGPUMemory(size_t size);
    void freeGPUMemory(void* ptr);
};

} // namespace cobali

