#include "cobali/engine/cobali_engine.h"
#include "cobali/common/utils.h"
#include <chrono>
#include <thread>

namespace cobali {

CoBaLIEngine::CoBaLIEngine(const EngineConfig& config)
    : config_(config)
    , next_request_id_(1)
    , running_(false)
    , initialized_(false) {
}

CoBaLIEngine::~CoBaLIEngine() {
    stop();
    cleanup();
}

bool CoBaLIEngine::initialize() {
    if (initialized_) {
        return true;
    }
    
    utils::Logger::getInstance().info("Initializing CoBaLI Engine...");
    
    // Initialize components
    batcher_ = std::make_unique<ContinuousBatcher>(config_);
    splitter_ = std::make_unique<PrefillSplitter>(config_);
    kv_cache_manager_ = std::make_unique<KVCacheManager>(config_);
    executor_ = std::make_unique<Executor>(config_);
    
    // Initialize executor (loads model)
    if (!executor_->initialize()) {
        utils::Logger::getInstance().error("Failed to initialize executor");
        return false;
    }
    
    // Initialize KV cache manager
    // Simplified: assume 128 bytes per token per layer
    size_t kv_size_per_layer = 128;
    if (!kv_cache_manager_->initialize(kv_size_per_layer)) {
        utils::Logger::getInstance().error("Failed to initialize KV cache manager");
        return false;
    }
    
    initialized_ = true;
    utils::Logger::getInstance().info("CoBaLI Engine initialized successfully");
    
    return true;
}

void CoBaLIEngine::start() {
    if (running_) {
        utils::Logger::getInstance().warning("Engine already running");
        return;
    }
    
    if (!initialized_) {
        utils::Logger::getInstance().error("Engine not initialized");
        return;
    }
    
    running_ = true;
    inference_thread_ = std::thread(&CoBaLIEngine::inferenceLoop, this);
    
    utils::Logger::getInstance().info("CoBaLI Engine started");
}

void CoBaLIEngine::stop() {
    if (!running_) {
        return;
    }
    
    running_ = false;
    if (inference_thread_.joinable()) {
        inference_thread_.join();
    }
    
    utils::Logger::getInstance().info("CoBaLI Engine stopped");
}

RequestID CoBaLIEngine::submitRequest(const std::vector<Token>& prompt_tokens,
                                     int max_output_length,
                                     float temperature,
                                     float top_p,
                                     int top_k) {
    std::lock_guard<std::mutex> lock(requests_mutex_);
    
    RequestID request_id = next_request_id_++;
    
    auto request = std::make_unique<Request>(
        request_id, prompt_tokens, max_output_length,
        temperature, top_p, top_k
    );
    
    // Allocate KV cache for this request
    request->kv_cache = kv_cache_manager_->allocate(request_id, config_.n_ctx);
    
    Request* req_ptr = request.get();
    requests_[request_id] = std::move(request);
    
    // Add to batcher
    batcher_->addRequest(req_ptr);
    
    utils::Logger::getInstance().info(
        utils::format("Submitted request %lu (prompt_len=%zu, max_output=%d)",
                     request_id, prompt_tokens.size(), max_output_length)
    );
    
    return request_id;
}

bool CoBaLIEngine::isRequestComplete(RequestID request_id) {
    std::lock_guard<std::mutex> lock(requests_mutex_);
    
    auto it = requests_.find(request_id);
    if (it == requests_.end()) {
        // Check if in completed results
        return completed_results_.find(request_id) != completed_results_.end();
    }
    
    return it->second->isCompleted();
}

std::vector<Token> CoBaLIEngine::getRequestResult(RequestID request_id) {
    std::lock_guard<std::mutex> lock(requests_mutex_);
    
    auto it = completed_results_.find(request_id);
    if (it != completed_results_.end()) {
        return it->second;
    }
    
    return std::vector<Token>();
}

std::vector<Token> CoBaLIEngine::waitForRequest(RequestID request_id) {
    while (!isRequestComplete(request_id)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    return getRequestResult(request_id);
}

void CoBaLIEngine::inferenceLoop() {
    utils::Logger::getInstance().info("Inference loop started");
    
    while (running_) {
        // Check if there are requests to process
        if (!batcher_->hasRequests()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        
        // Get next batch
        Batch batch = batcher_->getNextBatch();
        
        if (batch.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        
        // Process batch
        processBatch(batch);
        
        // Update batch state
        batcher_->updateAfterExecution(batch);
        
        // Handle completed requests
        handleCompletedRequests();
        
        // Update metrics
        updateMetrics();
    }
    
    utils::Logger::getInstance().info("Inference loop stopped");
}

void CoBaLIEngine::processBatch(Batch& batch) {
    // Separate prefill and decode requests
    std::vector<Request*> prefill_requests;
    std::vector<Request*> decode_requests;
    
    for (auto* req : batch.requests) {
        if (req->phase == Phase::PREFILL || 
            (req->phase == Phase::WAITING && req->tokens_processed < req->prompt_length)) {
            
            // Set prefill phase
            if (req->phase == Phase::WAITING) {
                req->phase = Phase::PREFILL;
                req->prefill_start_time = std::chrono::steady_clock::now();
            }
            
            // Determine chunk size if prefill splitting is enabled
            if (config_.enable_prefill_splitting) {
                int available_tokens = config_.max_tokens_per_batch - batch.total_tokens;
                req->current_chunk_size = splitter_->getChunkSize(
                    req, available_tokens, static_cast<int>(decode_requests.size())
                );
            } else {
                req->current_chunk_size = req->prompt_length - req->tokens_processed;
            }
            
            prefill_requests.push_back(req);
            
        } else if (req->phase == Phase::DECODE) {
            decode_requests.push_back(req);
        }
    }
    
    // Execute batch
    if (config_.enable_continuous_batching || config_.enable_prefill_splitting) {
        executor_->executeMixedBatch(prefill_requests, decode_requests);
    } else {
        // Fallback to sequential
        if (!prefill_requests.empty()) {
            executor_->executePrefillBatch(prefill_requests);
        }
        if (!decode_requests.empty()) {
            executor_->executeDecodeBatch(decode_requests);
        }
    }
    
    // Update prefill splitting state
    if (config_.enable_prefill_splitting) {
        for (auto* req : prefill_requests) {
            splitter_->updateAfterChunk(req, req->current_chunk_size);
        }
    }
}

void CoBaLIEngine::handleCompletedRequests() {
    std::lock_guard<std::mutex> lock(requests_mutex_);
    
    auto it = requests_.begin();
    while (it != requests_.end()) {
        auto* req = it->second.get();
        
        if (req->isCompleted()) {
            // Store result
            completed_results_[req->id] = req->output_tokens;
            
            // Free KV cache
            kv_cache_manager_->free(req->id);
            
            utils::Logger::getInstance().info(
                utils::format("Request %lu completed (generated %d tokens)",
                             req->id, req->tokens_generated)
            );
            
            // Remove from requests map
            it = requests_.erase(it);
        } else {
            ++it;
        }
    }
    
    batcher_->removeCompletedRequests();
}

void CoBaLIEngine::updateMetrics() {
    // TODO: Implement proper metrics tracking
    // For now, just basic stats
    
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    current_metrics_.avg_batch_size = static_cast<double>(batcher_->getActiveRequestCount());
    current_metrics_.memory_used_gb = kv_cache_manager_->getUsedMemoryBytes() / (1024.0 * 1024.0 * 1024.0);
}

Metrics CoBaLIEngine::getMetrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return current_metrics_;
}

void CoBaLIEngine::cleanup() {
    // Cleanup components
    executor_.reset();
    kv_cache_manager_.reset();
    splitter_.reset();
    batcher_.reset();
    
    // Clear requests
    requests_.clear();
    completed_results_.clear();
    
    initialized_ = false;
}

} // namespace cobali

