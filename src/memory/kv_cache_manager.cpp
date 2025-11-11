#include "cobali/memory/kv_cache_manager.h"
#include "cobali/common/utils.h"
#include <cuda_runtime.h>
#include <stdexcept>

namespace cobali {

KVCacheManager::KVCacheManager(const EngineConfig& config)
    : config_(config)
    , gpu_memory_pool_(nullptr)
    , total_memory_bytes_(0)
    , used_memory_bytes_(0)
    , bytes_per_token_(0)
    , next_slot_id_(0) {
}

KVCacheManager::~KVCacheManager() {
    cleanup();
}

bool KVCacheManager::initialize(size_t model_kv_size_per_layer) {
    // Calculate total memory needed
    // This is a simplified calculation - in practice, depends on model architecture
    bytes_per_token_ = model_kv_size_per_layer;
    
    // Allocate memory pool (convert MB to bytes)
    total_memory_bytes_ = config_.kv_cache_size_mb * 1024 * 1024;
    
    gpu_memory_pool_ = allocateGPUMemory(total_memory_bytes_);
    if (gpu_memory_pool_ == nullptr) {
        utils::Logger::getInstance().error("Failed to allocate GPU memory pool");
        return false;
    }
    
    utils::Logger::getInstance().info(
        utils::format("KV Cache Manager initialized: %.2f MB", 
                     total_memory_bytes_ / (1024.0 * 1024.0))
    );
    
    return true;
}

KVCacheSlot* KVCacheManager::allocate(RequestID request_id, size_t max_seq_len) {
    // Calculate memory needed for this request
    size_t memory_needed = max_seq_len * bytes_per_token_ * 2; // *2 for K and V
    
    if (used_memory_bytes_ + memory_needed > total_memory_bytes_) {
        utils::Logger::getInstance().warning(
            utils::format("Not enough memory for request %lu (need %zu, available %zu)",
                         request_id, memory_needed, 
                         total_memory_bytes_ - used_memory_bytes_)
        );
        return nullptr;
    }
    
    // Create new slot
    auto slot = std::make_unique<KVCacheSlot>();
    
    // Allocate slot ID
    if (!free_slot_ids_.empty()) {
        slot->slot_id = free_slot_ids_.back();
        free_slot_ids_.pop_back();
    } else {
        slot->slot_id = next_slot_id_++;
    }
    
    // Allocate K and V cache within the memory pool
    // For simplicity, allocating separate buffers (not from pool)
    // In production, this should be from the pool
    slot->k_cache = allocateGPUMemory(memory_needed / 2);
    slot->v_cache = allocateGPUMemory(memory_needed / 2);
    
    if (slot->k_cache == nullptr || slot->v_cache == nullptr) {
        utils::Logger::getInstance().error("Failed to allocate KV cache buffers");
        return nullptr;
    }
    
    slot->max_seq_len = max_seq_len;
    slot->current_len = 0;
    slot->is_allocated = true;
    slot->owner_id = request_id;
    
    used_memory_bytes_ += memory_needed;
    
    // Store and return
    KVCacheSlot* slot_ptr = slot.get();
    allocated_slots_[request_id] = std::move(slot);
    
    utils::Logger::getInstance().debug(
        utils::format("Allocated KV cache slot %d for request %lu (%.2f MB)",
                     slot_ptr->slot_id, request_id, 
                     memory_needed / (1024.0 * 1024.0))
    );
    
    return slot_ptr;
}

void KVCacheManager::free(RequestID request_id) {
    auto it = allocated_slots_.find(request_id);
    if (it == allocated_slots_.end()) {
        return;
    }
    
    KVCacheSlot* slot = it->second.get();
    
    // Free GPU memory
    if (slot->k_cache != nullptr) {
        freeGPUMemory(slot->k_cache);
    }
    if (slot->v_cache != nullptr) {
        freeGPUMemory(slot->v_cache);
    }
    
    size_t memory_freed = slot->max_seq_len * bytes_per_token_ * 2;
    used_memory_bytes_ -= memory_freed;
    
    // Return slot ID to free list
    free_slot_ids_.push_back(slot->slot_id);
    
    // Remove from allocated slots
    allocated_slots_.erase(it);
    
    utils::Logger::getInstance().debug(
        utils::format("Freed KV cache for request %lu (%.2f MB)",
                     request_id, memory_freed / (1024.0 * 1024.0))
    );
}

KVCacheSlot* KVCacheManager::get(RequestID request_id) {
    auto it = allocated_slots_.find(request_id);
    if (it == allocated_slots_.end()) {
        return nullptr;
    }
    return it->second.get();
}

std::vector<void*> KVCacheManager::getBatchKPointers(const std::vector<Request*>& batch) {
    std::vector<void*> pointers;
    for (auto* req : batch) {
        KVCacheSlot* slot = get(req->id);
        if (slot != nullptr) {
            pointers.push_back(slot->k_cache);
        } else {
            pointers.push_back(nullptr);
        }
    }
    return pointers;
}

std::vector<void*> KVCacheManager::getBatchVPointers(const std::vector<Request*>& batch) {
    std::vector<void*> pointers;
    for (auto* req : batch) {
        KVCacheSlot* slot = get(req->id);
        if (slot != nullptr) {
            pointers.push_back(slot->v_cache);
        } else {
            pointers.push_back(nullptr);
        }
    }
    return pointers;
}

bool KVCacheManager::hasMemoryFor(size_t max_seq_len) const {
    size_t memory_needed = max_seq_len * bytes_per_token_ * 2;
    return used_memory_bytes_ + memory_needed <= total_memory_bytes_;
}

void KVCacheManager::cleanup() {
    // Free all allocated slots
    for (auto& pair : allocated_slots_) {
        if (pair.second->k_cache != nullptr) {
            freeGPUMemory(pair.second->k_cache);
        }
        if (pair.second->v_cache != nullptr) {
            freeGPUMemory(pair.second->v_cache);
        }
    }
    allocated_slots_.clear();
    
    // Free memory pool
    if (gpu_memory_pool_ != nullptr) {
        freeGPUMemory(gpu_memory_pool_);
        gpu_memory_pool_ = nullptr;
    }
    
    used_memory_bytes_ = 0;
}

void* KVCacheManager::allocateGPUMemory(size_t size) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        utils::Logger::getInstance().error(
            utils::format("cudaMalloc failed: %s", cudaGetErrorString(err))
        );
        return nullptr;
    }
    return ptr;
}

void KVCacheManager::freeGPUMemory(void* ptr) {
    if (ptr != nullptr) {
        cudaFree(ptr);
    }
}

} // namespace cobali

