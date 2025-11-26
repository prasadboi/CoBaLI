#pragma once
#include "cobali/api.hpp"
#include <string>
#include <vector>
#include <cuda_runtime.h>

struct HRequest {
  uint64_t id;
  std::vector<int32_t> prompt_tokens;
  std::vector<int32_t> generated;
  int32_t pos = 0;  
  int32_t prefill_cursor = 0; 
  int32_t state = 0;     
  int32_t eos = 0; 
  int32_t max_new = 64;
  bool ttft_seen = false;
  
  //here -1 implies that no slot has been currently assigned
  int32_t slot_id = -1; 
};
