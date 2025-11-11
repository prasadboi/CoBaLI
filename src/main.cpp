#include "cobali/baseline/sequential_engine.h"
#include "cobali/engine/cobali_engine.h"
#include "cobali/common/utils.h"
#include <iostream>
#include <string>
#include <vector>

using namespace cobali;

void printUsage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " <mode> <model_path> [options]\n";
    std::cout << "\nModes:\n";
    std::cout << "  baseline    - Sequential baseline (no batching)\n";
    std::cout << "  batching    - Continuous batching\n";
    std::cout << "  full        - Full CoBaLI (batching + prefill splitting)\n";
    std::cout << "\nOptions:\n";
    std::cout << "  --prompt <text>        - Input prompt\n";
    std::cout << "  --max-tokens <n>       - Maximum output tokens (default: 128)\n";
    std::cout << "  --temperature <f>      - Sampling temperature (default: 1.0)\n";
    std::cout << "  --n-gpu-layers <n>     - Number of GPU layers (default: -1 = all)\n";
    std::cout << "  --verbose              - Enable verbose logging\n";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }
    
    std::string mode = argv[1];
    std::string model_path = argv[2];
    
    // Parse options
    std::string prompt = "Once upon a time";
    int max_tokens = 128;
    float temperature = 1.0f;
    int n_gpu_layers = -1;
    bool verbose = false;
    
    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            max_tokens = std::stoi(argv[++i]);
        } else if (arg == "--temperature" && i + 1 < argc) {
            temperature = std::stof(argv[++i]);
        } else if (arg == "--n-gpu-layers" && i + 1 < argc) {
            n_gpu_layers = std::stoi(argv[++i]);
        } else if (arg == "--verbose") {
            verbose = true;
        }
    }
    
    // Setup logging
    if (verbose) {
        utils::Logger::getInstance().setLogLevel(utils::LogLevel::DEBUG);
    }
    
    utils::Logger::getInstance().info("CoBaLI - Continuous Batching and Prefill Splitting");
    utils::Logger::getInstance().info("=================================================");
    utils::Logger::getInstance().info(utils::format("Mode: %s", mode.c_str()));
    utils::Logger::getInstance().info(utils::format("Model: %s", model_path.c_str()));
    utils::Logger::getInstance().info(utils::format("Prompt: %s", prompt.c_str()));
    
    // Create config
    EngineConfig config;
    config.model_path = model_path;
    config.n_gpu_layers = n_gpu_layers;
    config.verbose = verbose;
    
    if (mode == "baseline") {
        // Baseline sequential engine
        config.enable_continuous_batching = false;
        config.enable_prefill_splitting = false;
        
        SequentialEngine engine(config);
        if (!engine.initialize()) {
            utils::Logger::getInstance().error("Failed to initialize engine");
            return 1;
        }
        
        utils::Logger::getInstance().info("Running baseline inference...");
        
        // Tokenize prompt (simplified - would need proper tokenizer)
        std::vector<Token> prompt_tokens = {1, 2, 3, 4, 5}; // Dummy tokens
        
        auto start = std::chrono::steady_clock::now();
        std::vector<Token> output = engine.generate(prompt_tokens, max_tokens, temperature);
        auto end = std::chrono::steady_clock::now();
        
        double elapsed_ms = utils::getElapsedMs(start, end);
        
        utils::Logger::getInstance().info(utils::format("Generated %zu tokens in %.2f ms", 
                                                        output.size(), elapsed_ms));
        utils::Logger::getInstance().info(utils::format("Throughput: %.2f tokens/sec", 
                                                        output.size() / (elapsed_ms / 1000.0)));
        
    } else if (mode == "batching") {
        // Continuous batching (Phase 2)
        config.enable_continuous_batching = true;
        config.enable_prefill_splitting = false;
        config.max_batch_size = 16;
        config.max_tokens_per_batch = 2048;
        
        CoBaLIEngine engine(config);
        if (!engine.initialize()) {
            utils::Logger::getInstance().error("Failed to initialize engine");
            return 1;
        }
        
        engine.start();
        
        utils::Logger::getInstance().info("Running continuous batching...");
        
        // Submit request
        std::vector<Token> prompt_tokens = {1, 2, 3, 4, 5};
        RequestID req_id = engine.submitRequest(prompt_tokens, max_tokens, temperature);
        
        // Wait for completion
        auto start = std::chrono::steady_clock::now();
        std::vector<Token> output = engine.waitForRequest(req_id);
        auto end = std::chrono::steady_clock::now();
        
        double elapsed_ms = utils::getElapsedMs(start, end);
        
        utils::Logger::getInstance().info(utils::format("Generated %zu tokens in %.2f ms", 
                                                        output.size(), elapsed_ms));
        
        engine.stop();
        
    } else if (mode == "full") {
        // Full CoBaLI (Phase 3)
        config.enable_continuous_batching = true;
        config.enable_prefill_splitting = true;
        config.max_batch_size = 32;
        config.max_tokens_per_batch = 4096;
        config.prefill_chunk_size = 512;
        config.decode_priority_weight = 0.7f;
        
        CoBaLIEngine engine(config);
        if (!engine.initialize()) {
            utils::Logger::getInstance().error("Failed to initialize engine");
            return 1;
        }
        
        engine.start();
        
        utils::Logger::getInstance().info("Running full CoBaLI (batching + prefill splitting)...");
        
        // Submit request
        std::vector<Token> prompt_tokens = {1, 2, 3, 4, 5};
        RequestID req_id = engine.submitRequest(prompt_tokens, max_tokens, temperature);
        
        // Wait for completion
        auto start = std::chrono::steady_clock::now();
        std::vector<Token> output = engine.waitForRequest(req_id);
        auto end = std::chrono::steady_clock::now();
        
        double elapsed_ms = utils::getElapsedMs(start, end);
        
        utils::Logger::getInstance().info(utils::format("Generated %zu tokens in %.2f ms", 
                                                        output.size(), elapsed_ms));
        
        Metrics metrics = engine.getMetrics();
        utils::Logger::getInstance().info(utils::format("Average batch size: %.2f", 
                                                        metrics.avg_batch_size));
        
        engine.stop();
        
    } else {
        std::cerr << "Unknown mode: " << mode << std::endl;
        printUsage(argv[0]);
        return 1;
    }
    
    utils::Logger::getInstance().info("Done!");
    
    return 0;
}

