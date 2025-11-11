#include "cobali/engine/cobali_engine.h"
#include "cobali/common/utils.h"
#include <iostream>
#include <vector>
#include <thread>

using namespace cobali;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
        return 1;
    }
    
    // Setup config for continuous batching (Phase 2)
    EngineConfig config;
    config.model_path = argv[1];
    config.n_gpu_layers = -1;
    config.n_ctx = 2048;
    config.enable_continuous_batching = true;
    config.enable_prefill_splitting = false;  // Phase 2: only batching
    config.max_batch_size = 16;
    config.max_tokens_per_batch = 2048;
    config.verbose = true;
    
    utils::Logger::getInstance().setLogLevel(utils::LogLevel::INFO);
    
    // Create engine
    CoBaLIEngine engine(config);
    
    // Initialize
    utils::Logger::getInstance().info("Initializing CoBaLI engine...");
    if (!engine.initialize()) {
        utils::Logger::getInstance().error("Failed to initialize");
        return 1;
    }
    
    // Start engine
    engine.start();
    
    utils::Logger::getInstance().info("Submitting multiple requests...");
    
    // Submit multiple requests to demonstrate batching
    std::vector<RequestID> request_ids;
    
    for (int i = 0; i < 5; ++i) {
        std::vector<Token> prompt_tokens(10 + i * 5, i);  // Varying prompt lengths
        
        RequestID req_id = engine.submitRequest(
            prompt_tokens,
            64,     // max_output_length
            0.8f,   // temperature
            0.95f,  // top_p
            40      // top_k
        );
        
        request_ids.push_back(req_id);
        
        // Small delay between submissions
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    utils::Logger::getInstance().info("Waiting for requests to complete...");
    
    // Wait for all requests
    auto start = std::chrono::steady_clock::now();
    
    for (auto req_id : request_ids) {
        std::vector<Token> output = engine.waitForRequest(req_id);
        utils::Logger::getInstance().info(
            utils::format("Request %lu completed: %zu tokens", req_id, output.size())
        );
    }
    
    auto end = std::chrono::steady_clock::now();
    double elapsed_s = utils::getElapsedSeconds(start, end);
    
    // Get metrics
    Metrics metrics = engine.getMetrics();
    
    // Print results
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Total requests: " << request_ids.size() << std::endl;
    std::cout << "Total time: " << elapsed_s << " seconds" << std::endl;
    std::cout << "Throughput: " << (request_ids.size() / elapsed_s) << " requests/sec" << std::endl;
    std::cout << "Avg batch size: " << metrics.avg_batch_size << std::endl;
    
    // Stop engine
    engine.stop();
    
    return 0;
}

