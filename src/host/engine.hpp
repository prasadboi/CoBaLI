#pragma once
#include <cstdint>
#include <string>
#include <vector>

enum class Mode { Sequential, Continuous };

enum ReqState : int32_t { RS_PREFILL = 0, RS_DECODE = 1, RS_DONE = 2 };

struct HRequest {
    uint64_t id = 0;
    std::vector<int32_t> prompt_tokens;
    std::vector<int32_t> generated;
    int32_t max_new = 0;
    int32_t pos = 0;
    int32_t eos = 0;
    int32_t state = RS_PREFILL;
};

struct RunnerConfig {
    std::string model_path;
    int n_ctx      = 4096;
    int gpu_layers = -1;
    int max_slots  = 8;
    Mode mode      = Mode::Continuous;
};

struct Generated {
    uint64_t id;
    std::string text;
};

class Engine {
public:
    explicit Engine(const RunnerConfig &cfg);
    ~Engine();

    struct AddReq { std::string prompt; int max_new = 64; };
    uint64_t add_request(const AddReq &r);
    std::vector<Generated> run();

private:
    struct Impl;                  // <-- forward declare nested type
    std::unique_ptr<Impl> self_;  // <-- pimpl pointer
};
