#pragma once

#include "cobali/common/types.h"
#include "cobali/common/config.h"
#include "llama.h"
#include <string>
#include <memory>

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
    
    // Tokenize text to tokens
    std::vector<Token> tokenize(const std::string& text, bool add_bos = true);
    
    // Detokenize tokens to text
    std::string detokenize(const std::vector<Token>& tokens);
    
    // Generate text from text (convenience method)
    std::string generateText(const std::string& prompt,
                            int max_output_length = 128,
                            float temperature = 1.0f,
                            float top_p = 0.9f,
                            int top_k = 40);
    
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
    llama_batch batch_;   // Batch for inference
    bool batch_initialized_;
    
    bool initialized_;
    
    // Helper methods
    bool loadModel();
    Token sampleToken(float temperature, float top_p, int top_k);
};

} // namespace cobali


