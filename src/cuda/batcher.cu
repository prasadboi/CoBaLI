#include "batcher.cuh"
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ bool append_capped(int32_t idx, int32_t max_slots, int32_t* d_selected_idx, int32_t* d_selected_cnt) {
  int32_t pos = atomicAdd(d_selected_cnt, 1);
  if (pos < max_slots) {
    d_selected_idx[pos] = idx;
    return true;
  } else {
    atomicSub(d_selected_cnt, 1);
    return false;
  }
}

__global__ void k_select_continuous(const int32_t* __restrict__ d_state, const int32_t* __restrict__ d_eos, int32_t n_requests, int32_t max_slots, int32_t* __restrict__ d_selected_idx, int32_t* __restrict__ d_selected_cnt) {
  for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n_requests;
       i += blockDim.x * gridDim.x) {
    if (d_state[i] == RS_DECODE && d_eos[i] == 0) {
      append_capped(i, max_slots, d_selected_idx, d_selected_cnt);
      if (atomicAdd(d_selected_cnt, 0) >= max_slots) return;
    }
  }
}

__global__ void k_select_sequential(const int32_t* __restrict__ d_state, const int32_t* __restrict__ d_eos, int32_t n_requests, int32_t current_active, int32_t* __restrict__ d_selected_idx, int32_t* __restrict__ d_selected_cnt) {
  if (current_active >= 0 &&
      current_active < n_requests &&
      d_state[current_active] == RS_DECODE &&
      d_eos[current_active] == 0) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      d_selected_idx[0] = current_active;
      *d_selected_cnt = 1;
    }
    return;
  }

  __shared__ int32_t found;
  if (threadIdx.x == 0) found = -1;
  __syncthreads();

  for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n_requests;
       i += blockDim.x * gridDim.x) {
    if (found >= 0) break;
    if (d_state[i] == RS_DECODE && d_eos[i] == 0) {
      int32_t old = atomicCAS(&found, -1, i);
      if (old != -1) atomicMin(&found, i);
    }
  }
  __syncthreads();

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    if (found >= 0) {
      d_selected_idx[0] = found;
      *d_selected_cnt = 1;
    } else {
      *d_selected_cnt = 0;
    }
  }
}

__global__ void k_gather_metadata(const int32_t* __restrict__ d_selected_idx, int32_t selected_cnt, const int32_t* __restrict__ d_req_id, const int32_t* __restrict__ d_pos, int32_t* __restrict__ d_seq_id_out, int32_t* __restrict__ d_pos_out) {
  int32_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= selected_cnt) return;
  int32_t i = d_selected_idx[k];
  d_seq_id_out[k] = d_req_id[i];
  d_pos_out[k]    = d_pos[i];
}

extern "C" void cobali_reset_selected_count(int32_t* d_selected_cnt, cudaStream_t stream) {
  cudaMemsetAsync(d_selected_cnt, 0, sizeof(int32_t), stream);
}

extern "C" void cobali_select_continuous(const int32_t* d_state, const int32_t* d_eos, int32_t n_requests, int32_t max_slots, int32_t* d_selected_idx, int32_t* d_selected_cnt, cudaStream_t stream) {
  cobali_reset_selected_count(d_selected_cnt, stream);
  int threads = 256;
  int blocks  = (n_requests + threads - 1) / threads;
  blocks = max(1, min(blocks, 1024));
  k_select_continuous<<<blocks, threads, 0, stream>>>(
    d_state, d_eos, n_requests, max_slots, d_selected_idx, d_selected_cnt);
}

extern "C" void cobali_select_sequential(const int32_t* d_state, const int32_t* d_eos, int32_t n_requests, int32_t current_active, int32_t* d_selected_idx, int32_t* d_selected_cnt, cudaStream_t stream) {
  cobali_reset_selected_count(d_selected_cnt, stream);
  k_select_sequential<<<32, 128, 0, stream>>>(
    d_state, d_eos, n_requests, current_active, d_selected_idx, d_selected_cnt);
}

extern "C" void cobali_gather_metadata(const int32_t* d_selected_idx, int32_t selected_cnt, const int32_t* d_req_id, const int32_t* d_pos, int32_t* d_seq_id_out, int32_t* d_pos_out, cudaStream_t stream) {
  int threads = 128;
  int blocks  = (selected_cnt + threads - 1) / threads;
  k_gather_metadata<<<blocks, threads, 0, stream>>>(
    d_selected_idx, selected_cnt, d_req_id, d_pos, d_seq_id_out, d_pos_out);
}
