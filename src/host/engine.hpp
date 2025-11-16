#pragma once
#include "cobali/api.hpp"
#include <string>
#include <vector>
#include <cuda_runtime.h>

// Internal host-side request struct
struct HRequest {
  uint64_t id;
  std::vector<int32_t> prompt_tokens;
  std::vector<int32_t> generated;
  int32_t pos = 0;       // absolute pos (grows each token)
  int32_t prefill_cursor = 0; // prompt tokens already submitted
  int32_t state = 0;     // RS_*
  int32_t eos = 0;       // 1 when finished
  int32_t max_new = 64;
  bool ttft_seen = false;
};

