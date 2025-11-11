#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Request states mirrored on device
enum ReqState : int32_t { RS_IDLE = 0, RS_PREFILL = 1, RS_DECODE = 2, RS_DONE = 3 };

// Device arrays you maintain per request i:
//   d_state[i] : ReqState
//   d_eos[i]   : 1 if finished (hit EOS / max_new), else 0
//   d_pos[i]   : current absolute position (used by host when building llama_batch.pos)
//   d_req_id[i]: stable request id (seq id)

// Outputs per-iteration:
//   d_selected_idx[k] : chosen request indices (k < max_slots)
//   d_selected_cnt    : device-side counter, number of chosen indices

void cobali_reset_selected_count(int32_t* d_selected_cnt, cudaStream_t stream);

void cobali_select_continuous(const int32_t* d_state,
                              const int32_t* d_eos,
                              int32_t n_requests,
                              int32_t max_slots,
                              int32_t* d_selected_idx,
                              int32_t* d_selected_cnt,
                              cudaStream_t stream);

void cobali_select_sequential(const int32_t* d_state,
                              const int32_t* d_eos,
                              int32_t n_requests,
                              int32_t current_active,  // -1 if none
                              int32_t* d_selected_idx, // size >= 1
                              int32_t* d_selected_cnt,
                              cudaStream_t stream);

void cobali_gather_metadata(const int32_t* d_selected_idx,
                            int32_t selected_cnt,
                            const int32_t* d_req_id,
                            const int32_t* d_pos,
                            int32_t* d_seq_id_out,
                            int32_t* d_pos_out,
                            cudaStream_t stream);

#ifdef __cplusplus
}
#endif
