#include "cobali/engine/executor.h"
#include "cobali/common/utils.h"
#include "llama.h"

namespace cobali {

Executor::Executor(const EngineConfig& config)
    : config_(config)
    , model_(nullptr)
    , ctx_(nullptr)
    , batch_(nullptr)
    , initialized_(false) {
}

Executor::~Executor() {
    cleanup();
}

bool Executor::initialize() {
    if (initialized_) {
        return true;
    }
    
    utils::Logger::getInstance().info("Initializing Executor...");
    
    if (!loadModel()) {
        return false;
    }
    
    // Allocate batch structure
    batch_ = new llama_batch;
    *batch_ = llama_batch_init(config_.max_tokens_per_batch, 0, 1);
    
    initialized_ = true;
    utils::Logger::getInstance().info("Executor initialized successfully");
    
    return true;
}

bool Executor::loadModel() {
    llama_backend_init();
    
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = config_.n_gpu_layers;
    model_params.use_mmap = config_.use_mmap;
    model_params.use_mlock = config_.use_mlock;
    
    model_ = llama_load_model_from_file(config_.model_path.c_str(), model_params);
    if (model_ == nullptr) {
        utils::Logger::getInstance().error("Failed to load model");
        return false;
    }
    
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = config_.n_ctx;
    ctx_params.n_batch = config_.max_tokens_per_batch;
    ctx_params.n_threads = config_.num_threads;
    
    ctx_ = llama_new_context_with_model(model_, ctx_params);
    if (ctx_ == nullptr) {
        utils::Logger::getInstance().error("Failed to create context");
        llama_free_model(model_);
        model_ = nullptr;
        return false;
    }
    
    return true;
}

void Executor::executePrefillBatch(std::vector<Request*>& requests) {
    if (!initialized_ || requests.empty()) {
        return;
    }
    
    // Clear batch manually
    batch_->n_tokens = 0;
    
    // Add tokens from each request
    for (auto* req : requests) {
        int start_pos = req->tokens_processed;
        int end_pos = std::min(start_pos + req->current_chunk_size, 
                               req->prompt_length);
        
        // Add tokens to batch manually
        for (int i = start_pos; i < end_pos; ++i) {
            if (batch_->n_tokens < config_.max_tokens_per_batch) {
                batch_->token[batch_->n_tokens] = req->prompt_tokens[i];
                batch_->pos[batch_->n_tokens] = i;
                batch_->seq_id[batch_->n_tokens] = static_cast<int>(req->id);
                batch_->logits[batch_->n_tokens] = (i == end_pos - 1); // logits for last token
                batch_->n_tokens++;
            }
        }
    }
    
    // Execute batch
    if (llama_decode(ctx_, *batch_)) {
        utils::Logger::getInstance().error("Failed to decode prefill batch");
        return;
    }
    
    // Update request states
    for (auto* req : requests) {
        int chunk_size = std::min(req->current_chunk_size, 
                                  req->prompt_length - req->tokens_processed);
        req->tokens_processed += chunk_size;
        
        // Check if prefill is complete
        if (req->tokens_processed >= req->prompt_length) {
            req->phase = Phase::DECODE;
            req->first_token_time = std::chrono::steady_clock::now();
        }
    }
}

void Executor::executeDecodeBatch(std::vector<Request*>& requests) {
    if (!initialized_ || requests.empty()) {
        return;
    }
    
    // Clear batch manually
    batch_->n_tokens = 0;
    
    // Add decode token for each request
    for (auto* req : requests) {
        if (batch_->n_tokens >= config_.max_tokens_per_batch) {
            break;
        }
        
        // Get last generated token (or last prompt token if just started decode)
        Token token;
        if (req->tokens_generated > 0) {
            token = req->output_tokens.back();
        } else {
            token = req->prompt_tokens.back();
        }
        
        int pos = req->tokens_processed + req->tokens_generated;
        
        batch_->token[batch_->n_tokens] = token;
        batch_->pos[batch_->n_tokens] = pos;
        batch_->seq_id[batch_->n_tokens] = static_cast<int>(req->id);
        batch_->logits[batch_->n_tokens] = true;  // generate logits
        batch_->n_tokens++;
    }
    
    // Execute batch
    if (llama_decode(ctx_, *batch_)) {
        utils::Logger::getInstance().error("Failed to decode batch");
        return;
    }
    
    // Sample tokens for each request
    for (size_t i = 0; i < requests.size(); ++i) {
        auto* req = requests[i];
        Token next_token = sampleToken(req);
        
        // Check for EOS
        if (next_token == llama_token_eos(model_) || 
            req->tokens_generated >= req->max_output_length) {
            req->phase = Phase::COMPLETED;
            req->completion_time = std::chrono::steady_clock::now();
        } else {
            req->output_tokens.push_back(next_token);
            req->tokens_generated++;
        }
    }
}

void Executor::executeMixedBatch(std::vector<Request*>& prefill_requests,
                                 std::vector<Request*>& decode_requests) {
    // For simplicity, execute prefill and decode separately
    // In a more advanced implementation, these could be truly mixed
    
    if (!prefill_requests.empty()) {
        executePrefillBatch(prefill_requests);
    }
    
    if (!decode_requests.empty()) {
        executeDecodeBatch(decode_requests);
    }
}

Token Executor::sampleToken(const Request* request) {
    // Get logits for this request's sequence
    float* logits = llama_get_logits_ith(ctx_, static_cast<int>(request->id));
    int n_vocab = llama_n_vocab(model_);
    
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
    
    llama_sample_top_k(ctx_, &candidates_p, request->top_k, 1);
    llama_sample_top_p(ctx_, &candidates_p, request->top_p, 1);
    llama_sample_temp(ctx_, &candidates_p, request->temperature);
    
    return llama_sample_token(ctx_, &candidates_p);
}

int Executor::getVocabSize() const {
    if (model_ == nullptr) return 0;
    return llama_n_vocab(model_);
}

int Executor::getContextSize() const {
    if (ctx_ == nullptr) return 0;
    return llama_n_ctx(ctx_);
}

void Executor::cleanup() {
    if (batch_ != nullptr) {
        llama_batch_free(*batch_);
        delete batch_;
        batch_ = nullptr;
    }
    
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

