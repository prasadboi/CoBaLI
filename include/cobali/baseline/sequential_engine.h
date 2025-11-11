#pragma once

#include "cobali/common/types.h"
#include "cobali/common/config.h"
#include <string>
#include <memory>

// Forward declare llama.cpp types to avoid exposing them in header
struct llama_model;
struct llama_context;

namespace cobali {

// Phase 1: Sequential baseline engine
// Processes one request at a time using llama.cpp
// No batching, no optimizations - this is the baseline for comparison
class SequentialEngine {
public:
    explicit SequentialEngine(const EngineConfig& config);
    ~SequentialEngine();
    
    // Initialize the engine (load model, allocate memory)
    bool initialize();
    
    // Process a single request synchronously
    // Returns the generated tokens
    std::vector<Token> generate(const std::vector<Token>& prompt_tokens,
                                int max_output_length,
                                float temperature = 1.0f,
                                float top_p = 0.9f,
                                int top_k = 40);
    
    // Process a request object
    void processRequest(Request* request);
    
    // Get model information
    int getContextSize() const;
    int getVocabSize() const;
    std::string getModelInfo() const;
    
    // Cleanup
    void cleanup();
    
private:
    const EngineConfig& config_;
    
    // llama.cpp handles
    llama_model* model_;
    llama_context* ctx_;
    
    bool initialized_;
    
    // Helper methods
    bool loadModel();
    Token sampleToken(float temperature, float top_p, int top_k);
};

} // namespace cobali

