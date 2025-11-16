#include "cobali/api.hpp"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include "llama.h"

static std::string slurp(const char* path) {
  std::ifstream in(path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

int main(int argc, char** argv) {
  RunnerConfig cfg;
  const char* prompts_path = "workloads/prompts_short.txt";

  // super light CLI parsing
  for (int i=1;i<argc;++i) {
    std::string a = argv[i];
    if (a == "--model" && i+1<argc) cfg.model_path = argv[++i];
    else if (a == "--ctx" && i+1<argc) cfg.n_ctx = std::stoi(argv[++i]);
    else if (a == "--gpu-layers" && i+1<argc) cfg.gpu_layers = std::stoi(argv[++i]);
    else if (a == "--max-slots" && i+1<argc) cfg.max_slots = std::stoi(argv[++i]);
    else if (a == "--prefill-chunk" && i+1<argc) cfg.prefill_chunk_tokens = std::stoi(argv[++i]);
    else if (a == "--mode" && i+1<argc) {
      std::string m = argv[++i];
      cfg.mode = (m=="seq") ? Mode::Sequential : Mode::Continuous;
    } else if (a == "--prompts" && i+1<argc) {
      prompts_path = argv[++i];
    }
  }

  if (cfg.model_path.empty()) {
    fprintf(stderr, "usage: %s --model models/your.gguf [--mode seq|cont] [--max-slots N] [--prefill-chunk T]\n", argv[0]);
    return 1;
  }

  Engine eng(cfg);

  // Add a few demo requests (one per line)
  {
    std::ifstream in(prompts_path);
    std::string line;
    while (std::getline(in, line)) {
      if (line.empty()) continue;
      eng.add_request({ line, 64 });
    }
  }

  auto results = eng.run();
  for (auto& r : results) {
    printf("=== req %llu ===\n%s\n\n", (unsigned long long)r.req_id, r.text.c_str());
  }
  return 0;
}