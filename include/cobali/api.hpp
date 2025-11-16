#pragma once
#include <string>
#include <vector>
#include <cstdint>

enum class Mode { Sequential, Continuous };

struct AddReq {
  std::string prompt;
  int max_new = 64;
};

struct Generated {
  uint64_t req_id;
  std::string text;
};

struct RunnerConfig {
  std::string model_path;
  int n_ctx = 4096;
  int gpu_layers = -1;    // -1 = all available
  int max_slots = 8;      // decode slots for CB
  int prefill_chunk_tokens = 128; // tokens per request chunk during CB prefill
  Mode mode = Mode::Continuous;
  int seed = 0;
};

class Engine {
public:
  Engine(const RunnerConfig& cfg);
  ~Engine();

  uint64_t add_request(const AddReq& r); // tokenizes request and queues it
  // run decoding until all active requests are done
  std::vector<Generated> run();

private:
  struct Impl;
  Impl* self_;
};
