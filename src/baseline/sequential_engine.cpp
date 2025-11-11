#include "cobali/baseline/sequential_engine.h"
#include "cobali/common/utils.h"

// Include llama.cpp headers
#include "llama.h"

#include <vector>
#include <string>
#include <stdexcept>

namespace cobali {

SequentialEngine::SequentialEngine(const EngineConfig& config)
    : config_(config)
    , model_(nullptr)
    , ctx_(nullptr)
    , initialized_(false) {
}

SequentialEngine::~SequentialEngine() {
    cleanup();
}

bool SequentialEngine::initialize() {
    if (initialized_) {
        utils::Logger::getInstance().warning("Engine already initialized");
        return true;
    }
    
    utils::Logger::getInstance().info("Initializing Sequential Engine...");
    
    if (!loadModel()) {
        return false;
    }
    
    initialized_ = true;
    utils::Logger::getInstance().info("Sequential Engine initialized successfully");
    
    return true;
}

bool SequentialEngine::loadModel() {
    // Initialize llama backend
    llama_backend_init();
    
    // Setup model parameters
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = config_.n_gpu_layers;
    model_params.use_mmap = config_.use_mmap;
    model_params.use_mlock = config_.use_mlock;
    
    // Load model
    utils::Logger::getInstance().info(
        utils::format("Loading model from: %s", config_.model_path.c_str())
    );
    
    model_ = llama_load_model_from_file(config_.model_path.c_str(), model_params);
    if (model_ == nullptr) {
        utils::Logger::getInstance().error("Failed to load model");
        return false;
    }
    
    // Create context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = config_.n_ctx;
    ctx_params.n_batch = 512; // Default batch size for baseline
    ctx_params.n_threads = config_.num_threads;
    
    ctx_ = llama_new_context_with_model(model_, ctx_params);
    if (ctx_ == nullptr) {
        utils::Logger::getInstance().error("Failed to create context");
        llama_free_model(model_);
        model_ = nullptr;
        return false;
    }
    
    utils::Logger::getInstance().info(
        utils::format("Model loaded: vocab_size=%d, ctx_size=%d",
                     llama_n_vocab(model_), llama_n_ctx(ctx_))
    );
    
    return true;
}

std::vector<Token> SequentialEngine::generate(const std::vector<Token>& prompt_tokens,
                                              int max_output_length,
                                              float temperature,
                                              float top_p,
                                              int top_k) {
    if (!initialized_) {
        throw std::runtime_error("Engine not initialized");
    }
    
    std::vector<Token> output_tokens;
    
    // Reset context
    llama_kv_cache_clear(ctx_);
    
    int n_ctx = llama_n_ctx(ctx_);
    int n_prompt = static_cast<int>(prompt_tokens.size());
    
    if (n_prompt >= n_ctx) {
        utils::Logger::getInstance().error(
            utils::format("Prompt too long: %d tokens (max: %d)", n_prompt, n_ctx)
        );
        return output_tokens;
    }
    
    // Process prompt (prefill phase)
    utils::Logger::getInstance().debug(
        utils::format("Processing prompt: %d tokens", n_prompt)
    );
    
    for (int i = 0; i < n_prompt; ++i) {
        if (llama_decode(ctx_, llama_batch_get_one(&prompt_tokens[i], 1, i, 0))) {
            utils::Logger::getInstance().error("Failed to decode prompt");
            return output_tokens;
        }
    }
    
    // Generate output tokens (decode phase)
    int n_generated = 0;
    int n_cur = n_prompt;
    
    while (n_generated < max_output_length && n_cur < n_ctx) {
        // Sample next token
        Token next_token = sampleToken(temperature, top_p, top_k);
        
        // Check for EOS
        if (next_token == llama_token_eos(model_)) {
            break;
        }
        
        output_tokens.push_back(next_token);
        n_generated++;
        
        // Decode the sampled token
        if (llama_decode(ctx_, llama_batch_get_one(&next_token, 1, n_cur, 0))) {
            utils::Logger::getInstance().error("Failed to decode token");
            break;
        }
        
        n_cur++;
    }
    
    utils::Logger::getInstance().debug(
        utils::format("Generated %d tokens", n_generated)
    );
    
    return output_tokens;
}

void SequentialEngine::processRequest(Request* request) {
    if (!initialized_) {
        request->phase = Phase::FAILED;
        return;
    }
    
    request->phase = Phase::PREFILL;
    request->prefill_start_time = std::chrono::steady_clock::now();
    
    // Generate output
    std::vector<Token> output = generate(
        request->prompt_tokens,
        request->max_output_length,
        request->temperature,
        request->top_p,
        request->top_k
    );
    
    // Update request
    request->output_tokens = output;
    request->tokens_generated = static_cast<int>(output.size());
    request->phase = Phase::COMPLETED;
    request->completion_time = std::chrono::steady_clock::now();
    
    if (!output.empty()) {
        request->first_token_time = request->prefill_start_time; // Approximate
    }
}

Token SequentialEngine::sampleToken(float temperature, float top_p, int top_k) {
    // Get logits
    float* logits = llama_get_logits(ctx_);
    int n_vocab = llama_n_vocab(model_);
    
    // Prepare candidates
    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);
    
    for (llama_token token_id = 0; token_id < n_vocab; ++token_id) {
        candidates.push_back({token_id, logits[token_id], 0.0f});
    }
    
    llama_token_data_array candidates_p = {
        candidates.data(),
        candidates.size(),
        false
    };
    
    // Apply sampling
    llama_sample_top_k(ctx_, &candidates_p, top_k, 1);
    llama_sample_top_p(ctx_, &candidates_p, top_p, 1);
    llama_sample_temp(ctx_, &candidates_p, temperature);
    
    // Sample token
    Token token = llama_sample_token(ctx_, &candidates_p);
    
    return token;
}

int SequentialEngine::getContextSize() const {
    if (ctx_ == nullptr) return 0;
    return llama_n_ctx(ctx_);
}

int SequentialEngine::getVocabSize() const {
    if (model_ == nullptr) return 0;
    return llama_n_vocab(model_);
}

std::string SequentialEngine::getModelInfo() const {
    if (model_ == nullptr) {
        return "No model loaded";
    }
    
    return utils::format("Model: vocab=%d, ctx=%d, layers=%d",
                        llama_n_vocab(model_),
                        getContextSize(),
                        config_.n_gpu_layers);
}

void SequentialEngine::cleanup() {
    if (ctx_ != nullptr) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    
    if (model_ != nullptr) {
        llama_free_model(model_);
        model_ = nullptr;
    }
    
    llama_backend_free();
    
    initialized_ = false;
}

} // namespace cobali

