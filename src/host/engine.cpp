#include "engine.hpp"
#include "cobali/api.hpp"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <algorithm>

#include <cuda_runtime.h>
#include "../cuda/batcher.cuh"

// ---- llama.cpp C API
#include "llama.h"   // from external/llama.cpp

// Notes on API usage:
// - Batch interface: fill llama_batch.{token[], pos[], seq_id[], logits[]} and call llama_decode()
//   (see examples like simple.cpp / batched.cpp). :contentReference[oaicite:3]{index=3}
// - Retrieve logits: llama_get_logits() or llama_get_logits_ith(...), rows correspond to entries with
//   logits[i] == true in this batch. :contentReference[oaicite:4]{index=4}
// - Tokenization: llama_tokenize(...) commonly called once to get size (dst=null) then again to write. :contentReference[oaicite:5]{index=5}

struct Engine::Impl {
  RunnerConfig cfg;
  llama_model*   model  = nullptr;
  llama_context* ctx    = nullptr;
  int n_vocab = 0;

  std::vector<HRequest> reqs;
  // host mirrors for device selection
  std::vector<int32_t> h_state, h_eos, h_pos, h_req_id;

  // device buffers
  int32_t *d_state=nullptr, *d_eos=nullptr, *d_pos=nullptr, *d_req_id=nullptr;
  int32_t *d_selected_idx=nullptr, *d_selected_cnt=nullptr;
  int32_t *d_sel_seq=nullptr, *d_sel_pos=nullptr; // small metadata (optional)
  cudaStream_t stream = nullptr;

  // helpers
  uint64_t next_id = 1;

  Impl(const RunnerConfig& c): cfg(c) {}
};

static int32_t greedy_argmax(const float* row, int n) {
  int32_t best = 0;
  float bv = row[0];
  for (int i=1;i<n;++i) if (row[i] > bv) { bv=row[i]; best=i; }
  return best;
}

static std::vector<int32_t> tokenize(llama_model* model, const std::string& text) {
  // First call with null dst to get required size (negative means size). :contentReference[oaicite:6]{index=6}
  int n = llama_tokenize(model, text.c_str(), text.size(), nullptr, 0, true, true);
  if (n < 0) n = -n;
  std::vector<int32_t> out(n);
  int n2 = llama_tokenize(model, text.c_str(), text.size(), out.data(), out.size(), true, true);
  out.resize(n2);
  return out;
}

Engine::Engine(const RunnerConfig& cfg): self_(new Impl(cfg)) {
  auto& S = *self_;

  llama_backend_init();
  llama_model_params mp = llama_model_default_params();
  mp.vocab_only = false;
  S.model = llama_load_model_from_file(cfg.model_path.c_str(), mp);
  if (!S.model) { fprintf(stderr, "failed to load model: %s\n", cfg.model_path.c_str()); std::abort(); }

  llama_context_params cp = llama_context_default_params();
  cp.n_ctx = cfg.n_ctx;
  cp.seed  = cfg.seed;
  cp.n_gpu_layers = cfg.gpu_layers;
  cp.flash_attn = true;
  S.ctx = llama_new_context_with_model(S.model, cp);
  if (!S.ctx) { fprintf(stderr, "failed to create context\n"); std::abort(); }

  S.n_vocab = llama_n_vocab(S.model);

  cudaStreamCreate(&S.stream);

  // reserve device arrays for up to, say, 1024 requests (adjust as needed)
  const int N = 1024;
  S.h_state.assign(N, 0);
  S.h_eos.assign(N, 0);
  S.h_pos.assign(N, 0);
  S.h_req_id.assign(N, 0);

  cudaMalloc(&S.d_state, N * sizeof(int32_t));
  cudaMalloc(&S.d_eos,   N * sizeof(int32_t));
  cudaMalloc(&S.d_pos,   N * sizeof(int32_t));
  cudaMalloc(&S.d_req_id,N * sizeof(int32_t));

  cudaMalloc(&S.d_selected_idx, S.cfg.max_slots * sizeof(int32_t));
  cudaMalloc(&S.d_selected_cnt, sizeof(int32_t));
  cudaMalloc(&S.d_sel_seq, S.cfg.max_slots * sizeof(int32_t));
  cudaMalloc(&S.d_sel_pos, S.cfg.max_slots * sizeof(int32_t));
}

Engine::~Engine() {
  auto& S = *self_;
  if (S.ctx)   llama_free(S.ctx);
  if (S.model) llama_free_model(S.model);
  llama_backend_free();

  if (S.d_state) cudaFree(S.d_state);
  if (S.d_eos)   cudaFree(S.d_eos);
  if (S.d_pos)   cudaFree(S.d_pos);
  if (S.d_req_id)cudaFree(S.d_req_id);
  if (S.d_selected_idx) cudaFree(S.d_selected_idx);
  if (S.d_selected_cnt) cudaFree(S.d_selected_cnt);
  if (S.d_sel_seq) cudaFree(S.d_sel_seq);
  if (S.d_sel_pos) cudaFree(S.d_sel_pos);
  if (S.stream) cudaStreamDestroy(S.stream);
  delete self_;
}

uint64_t Engine::add_request(const AddReq& r) {
  auto& S = *self_;
  HRequest q;
  q.id = S.next_id++;
  q.prompt_tokens = tokenize(S.model, r.prompt);
  q.max_new = r.max_new;
  q.state = RS_PREFILL;
  q.pos   = 0;
  q.eos   = 0;

  // PREFILL (single shot per-request; we keep it simple)
  {
    llama_batch batch = llama_batch_init((int)q.prompt_tokens.size(), 0, 1);
    batch.n_tokens = (int)q.prompt_tokens.size();
    for (int i=0;i<batch.n_tokens;++i) {
      batch.token[i]  = q.prompt_tokens[i];
      batch.pos[i]    = q.pos++;
      batch.seq_id[i] = (int)q.id;
      batch.logits[i] = false; // only want logits during decode iterations
    }
    if (llama_decode(S.ctx, batch)) {
      fprintf(stderr, "llama_decode prefill failed\n"); std::abort();
    }
    llama_batch_free(batch);
  }

  // move to DECODE
  q.state = RS_DECODE;
  S.reqs.push_back(std::move(q));

  // copy host mirrors (only first K reqs matter)
  int idx = (int)S.reqs.size() - 1;
  S.h_state[idx]  = RS_DECODE;
  S.h_eos[idx]    = 0;
  S.h_pos[idx]    = S.reqs[idx].pos;
  S.h_req_id[idx] = (int32_t)S.reqs[idx].id;
  return S.reqs.back().id;
}

std::vector<Generated> Engine::run() {
  auto& S = *self_;
  const int N = (int)S.reqs.size();
  std::vector<Generated> out; out.reserve(N);

  int32_t current_active = -1;

  // decode loop until all EOS
  int unfinished = N;
  while (unfinished > 0) {
    // sync mirrors to device (small; ok to memcpy each iter)
    cudaMemcpyAsync(S.d_state,  S.h_state.data(),  N*sizeof(int32_t), cudaMemcpyHostToDevice, S.stream);
    cudaMemcpyAsync(S.d_eos,    S.h_eos.data(),    N*sizeof(int32_t), cudaMemcpyHostToDevice, S.stream);
    cudaMemcpyAsync(S.d_pos,    S.h_pos.data(),    N*sizeof(int32_t), cudaMemcpyHostToDevice, S.stream);
    cudaMemcpyAsync(S.d_req_id, S.h_req_id.data(), N*sizeof(int32_t), cudaMemcpyHostToDevice, S.stream);
    cudaStreamSynchronize(S.stream);

    if (S.cfg.mode == Mode::Continuous) {
      cobali_select_continuous(S.d_state, S.d_eos, N, S.cfg.max_slots,
                               S.d_selected_idx, S.d_selected_cnt, S.stream);
    } else {
      cobali_select_sequential(S.d_state, S.d_eos, N, current_active,
                               S.d_selected_idx, S.d_selected_cnt, S.stream);
    }

    int32_t h_count = 0;
    cudaMemcpyAsync(&h_count, S.d_selected_cnt, sizeof(int32_t), cudaMemcpyDeviceToHost, S.stream);
    cudaStreamSynchronize(S.stream);
    if (h_count == 0) break; // nothing to do

    std::vector<int32_t> h_sel(h_count);
    cudaMemcpyAsync(h_sel.data(), S.d_selected_idx, h_count*sizeof(int32_t), cudaMemcpyDeviceToHost, S.stream);
    cudaStreamSynchronize(S.stream);

    // Build llama_batch for 1-token-per-selected-seq (request next token)
    llama_batch batch = llama_batch_init(h_count, 0, 1);
    batch.n_tokens = h_count;

    // For logits retrieval: mark logits=true for every row so rows map 1:1. :contentReference[oaicite:7]{index=7}
    for (int k=0;k<h_count;++k) {
      int i = h_sel[k];
      HRequest& r = S.reqs[i];
      int32_t last_token = (r.generated.empty())
                         ? r.prompt_tokens.back()
                         : r.generated.back();
      batch.token[k]  = last_token;
      batch.pos[k]    = r.pos++;                 // absolute position
      batch.seq_id[k] = (int)r.id;
      batch.logits[k] = true;                    // we want logits for each row
    }

    if (llama_decode(S.ctx, batch)) {
      fprintf(stderr, "llama_decode failed\n"); std::abort();
    }

    // Get logits rows for each selected stream and sample next token (greedy)
    const float* logits_rows = llama_get_logits(S.ctx); // rows = #true in batch.logits[]
    for (int k=0;k<h_count;++k) {
      HRequest& r = S.reqs[h_sel[k]];
      const float* row = logits_rows + k * S.n_vocab;
      int32_t tok = greedy_argmax(row, S.n_vocab); // or build candidates and call llama_sample_token_greedy(...)
      r.generated.push_back(tok);

      if (tok == llama_token_eos(S.model) || (int)r.generated.size() >= r.max_new) {
        r.eos = 1; r.state = RS_DONE;
        S.h_eos[h_sel[k]] = 1;
        S.h_state[h_sel[k]] = RS_DONE;
        --unfinished;

        // collect text
        std::string text = llama_detokenize_text(S.model, r.generated.data(), (int)r.generated.size());
        out.push_back({ r.id, text });
      } else {
        S.h_pos[h_sel[k]] = r.pos;
        S.h_state[h_sel[k]] = RS_DECODE;
        if (S.cfg.mode == Mode::Sequential) current_active = h_sel[k];
      }
    }
    llama_batch_free(batch);
  }

  return out;
}
