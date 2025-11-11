#include "cobali/baseline/sequential_engine.h"
#include "cobali/common/utils.h"
#include <iostream>
#include <chrono>

using namespace cobali;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
        return 1;
    }
    
    // Setup config
    EngineConfig config;
    config.model_path = argv[1];
    config.n_gpu_layers = -1;  // Use all GPU layers
    config.n_ctx = 2048;
    config.verbose = true;
    
    utils::Logger::getInstance().setLogLevel(utils::LogLevel::INFO);
    
    // Create engine
    SequentialEngine engine(config);
    
    // Initialize
    utils::Logger::getInstance().info("Initializing engine...");
    if (!engine.initialize()) {
        utils::Logger::getInstance().error("Failed to initialize");
        return 1;
    }
    
    // Dummy prompt tokens (in practice, you'd tokenize a string)
    std::vector<Token> prompt_tokens = {1, 2, 3, 4, 5, 6, 7, 8};
    
    utils::Logger::getInstance().info("Running inference...");
    
    auto start = std::chrono::steady_clock::now();
    
    // Generate
    std::vector<Token> output = engine.generate(
        prompt_tokens,
        128,    // max_output_length
        0.8f,   // temperature
        0.95f,  // top_p
        40      // top_k
    );
    
    auto end = std::chrono::steady_clock::now();
    double elapsed_ms = utils::getElapsedMs(start, end);
    
    // Print results
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Prompt tokens: " << prompt_tokens.size() << std::endl;
    std::cout << "Generated tokens: " << output.size() << std::endl;
    std::cout << "Time: " << elapsed_ms << " ms" << std::endl;
    std::cout << "Throughput: " << (output.size() / (elapsed_ms / 1000.0)) << " tokens/sec" << std::endl;
    
    return 0;
}

