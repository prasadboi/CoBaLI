#pragma once

#include "cobali/common/types.h"
#include "cobali/common/config.h"
#include <vector>

// Forward declarations
struct llama_model;
struct llama_context;
struct llama_batch;

namespace cobali {

// Executor: Executes batches on GPU
// Calls CUDA kernels (initially llama.cpp's, later your custom ones)
class Executor {
public:
    explicit Executor(const EngineConfig& config);
    ~Executor();
    
    // Initialize executor (load model)
    bool initialize();
    
    // Execute a batch of prefill requests
    // Each request processes prefill_chunk_size tokens
    void executePrefillBatch(std::vector<Request*>& requests);
    
    // Execute a batch of decode requests
    // Each request generates one token
    void executeDecodeBatch(std::vector<Request*>& requests);
    
    // Execute mixed batch (both prefill and decode)
    void executeMixedBatch(std::vector<Request*>& prefill_requests,
                          std::vector<Request*>& decode_requests);
    
    // Get model info
    int getVocabSize() const;
    int getContextSize() const;
    
    // Cleanup
    void cleanup();
    
private:
    const EngineConfig& config_;
    
    // llama.cpp handles (for Phase 1-3)
    llama_model* model_;
    llama_context* ctx_;
    llama_batch* batch_;
    
    bool initialized_;
    
    // Helper methods
    bool loadModel();
    void prepareBatch(const std::vector<Request*>& requests);
    Token sampleToken(const Request* request);
    
    // Future: Your custom CUDA kernels will be called from here (Phase 4)
    // void executeCustomPrefillKernel(...);
    // void executeCustomDecodeKernel(...);
    // void executeCustomAttentionKernel(...);
};

} // namespace cobali

