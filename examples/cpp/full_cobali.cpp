#include "cobali/engine/cobali_engine.h"
#include "cobali/common/utils.h"
#include <iostream>
#include <vector>
#include <random>
#include <thread>

using namespace cobali;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
        return 1;
    }
    
    // Setup config for full CoBaLI (Phase 3: batching + prefill splitting)
    EngineConfig config;
    config.model_path = argv[1];
    config.n_gpu_layers = -1;
    config.n_ctx = 2048;
    config.enable_continuous_batching = true;
    config.enable_prefill_splitting = true;
    config.max_batch_size = 32;
    config.max_tokens_per_batch = 4096;
    config.prefill_chunk_size = 512;
    config.decode_priority_weight = 0.7f;
    config.verbose = true;
    
    utils::Logger::getInstance().setLogLevel(utils::LogLevel::INFO);
    
    // Create engine
    CoBaLIEngine engine(config);
    
    // Initialize
    utils::Logger::getInstance().info("Initializing full CoBaLI engine...");
    if (!engine.initialize()) {
        utils::Logger::getInstance().error("Failed to initialize");
        return 1;
    }
    
    // Start engine
    engine.start();
    
    utils::Logger::getInstance().info("Submitting requests with varying prompt lengths...");
    
    // Submit requests with varying prompt lengths to demonstrate prefill splitting
    std::vector<RequestID> request_ids;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> prompt_dist(128, 1024);
    std::uniform_int_distribution<> output_dist(32, 256);
    
    int num_requests = 10;
    
    for (int i = 0; i < num_requests; ++i) {
        int prompt_len = prompt_dist(gen);
        int output_len = output_dist(gen);
        
        std::vector<Token> prompt_tokens(prompt_len, i);
        
        RequestID req_id = engine.submitRequest(
            prompt_tokens,
            output_len,
            0.8f,
            0.95f,
            40
        );
        
        request_ids.push_back(req_id);
        
        utils::Logger::getInstance().info(
            utils::format("Submitted request %lu (prompt: %d, output: %d)",
                         req_id, prompt_len, output_len)
        );
        
        // Varying submission rate
        std::this_thread::sleep_for(std::chrono::milliseconds(50 + i * 10));
    }
    
    utils::Logger::getInstance().info("Waiting for requests to complete...");
    
    // Wait for all requests and measure metrics
    auto start = std::chrono::steady_clock::now();
    
    int total_tokens = 0;
    for (auto req_id : request_ids) {
        std::vector<Token> output = engine.waitForRequest(req_id);
        total_tokens += output.size();
        utils::Logger::getInstance().info(
            utils::format("Request %lu completed: %zu tokens", req_id, output.size())
        );
    }
    
    auto end = std::chrono::steady_clock::now();
    double elapsed_s = utils::getElapsedSeconds(start, end);
    
    // Get metrics
    Metrics metrics = engine.getMetrics();
    
    // Print results
    std::cout << "\n=== Full CoBaLI Results ===" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Batch size: " << config.max_batch_size << std::endl;
    std::cout << "  Tokens per batch: " << config.max_tokens_per_batch << std::endl;
    std::cout << "  Prefill chunk size: " << config.prefill_chunk_size << std::endl;
    std::cout << "  Decode priority weight: " << config.decode_priority_weight << std::endl;
    std::cout << "\nPerformance:" << std::endl;
    std::cout << "  Total requests: " << request_ids.size() << std::endl;
    std::cout << "  Total tokens generated: " << total_tokens << std::endl;
    std::cout << "  Total time: " << elapsed_s << " seconds" << std::endl;
    std::cout << "  Request throughput: " << (request_ids.size() / elapsed_s) << " req/sec" << std::endl;
    std::cout << "  Token throughput: " << (total_tokens / elapsed_s) << " tokens/sec" << std::endl;
    std::cout << "  Avg batch size: " << metrics.avg_batch_size << std::endl;
    std::cout << "  Memory used: " << metrics.memory_used_gb << " GB" << std::endl;
    
    // Stop engine
    engine.stop();
    
    return 0;
}

