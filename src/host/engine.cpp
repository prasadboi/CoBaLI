#include "engine.hpp"
#include "cobali/api.hpp"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <algorithm>
#include <vector>
#include <string>

#include <cuda_runtime.h>
#include "../cuda/batcher.cuh"

// llama.cpp C API
#include "llama.h"   // from external/llama.cpp

struct Engine::Impl {
  RunnerConfig cfg;

  llama_model       * model  = nullptr;
  const llama_vocab * vocab  = nullptr;
  llama_context     * ctx    = nullptr;

  int n_vocab = 0;

  std::vector<HRequest> reqs;

  // host mirrors used by CUDA selection
  std::vector<int32_t> h_state, h_eos, h_pos, h_req_id;

  // device buffers
  int32_t *d_state=nullptr, *d_eos=nullptr, *d_pos=nullptr, *d_req_id=nullptr;
  int32_t *d_selected_idx=nullptr, *d_selected_cnt=nullptr;
  int32_t *d_sel_seq=nullptr, *d_sel_pos=nullptr; // optional metadata
  cudaStream_t stream = nullptr;

  uint64_t next_id = 1;

  Impl(const RunnerConfig& c): cfg(c) {}
};

static int32_t greedy_argmax(const float* row, int n) {
  int32_t best = 0;
  float bv = row[0];
  for (int i=1;i<n;++i) if (row[i] > bv) { bv=row[i]; best=i; }
  return best;
}

// tokenize using vocab (2-pass size + write)
static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string& text) {
  int n = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(),
                         nullptr, 0, /*add_special*/true, /*parse_special*/true);
  if (n < 0) n = -n;
  std::vector<llama_token> out(n);
  int n2 = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(),
                          out.data(), (int32_t)out.size(),
                          /*add_special*/true, /*parse_special*/true);
  if (n2 < 0) n2 = -n2;
  out.resize(n2);
  return out;
}

// robust detokenize (auto-grow buffer)
static std::string detok_text(const llama_vocab * vocab, const std::vector<llama_token>& toks) {
  if (toks.empty()) return {};
  int cap = 256;
  std::string buf(cap, '\0');
  while (true) {
    int wrote = llama_detokenize(vocab, toks.data(), (int32_t)toks.size(),
                                 buf.data(), (int32_t)buf.size(),
                                 /*remove_special*/true, /*unparse_special*/false);
    if (wrote >= 0) { buf.resize(wrote); return buf; }
    cap *= 2; buf.assign(cap, '\0');
  }
}

Engine::Engine(const RunnerConfig& cfg): self_(new Impl(cfg)) {
  auto& S = *self_;

  llama_backend_init();

  // --- model params (GPU/offload etc. live here)
  llama_model_params mp = llama_model_default_params();
  mp.vocab_only = false;
  if (cfg.gpu_layers >= 0) {
    mp.n_gpu_layers = cfg.gpu_layers; // place GPU layers control here
  }
  // (optional) mp.main_gpu / mp.split_mode / mp.devices can be set as needed.  :contentReference[oaicite:4]{index=4}

  S.model = llama_model_load_from_file(cfg.model_path.c_str(), mp);
  if (!S.model) {
    std::fprintf(stderr, "failed to load model: %s\n", cfg.model_path.c_str());
    std::abort();
  }

  S.vocab = llama_model_get_vocab(S.model);
  if (!S.vocab) { std::fprintf(stderr, "failed to get vocab from model\n"); std::abort(); }

  // --- context params (no seed/flash_attn here)
  llama_context_params cp = llama_context_default_params();
  cp.n_ctx = cfg.n_ctx;
  const uint32_t MAX_REQS = 1024;
  cp.n_seq_max = MAX_REQS;
  // If you want to tune threads / ubatch for CPU parts:
  // cp.n_threads = cfg.threads; cp.n_threads_batch = cfg.threads;
  // cp.n_ubatch = cfg.ubatch;           

  // modern initializer
  S.ctx = llama_init_from_model(S.model, cp);
  if (!S.ctx) { std::fprintf(stderr, "failed to create context\n"); std::abort(); }

  S.n_vocab = llama_vocab_n_tokens(S.vocab);

  cudaStreamCreate(&S.stream);

  // reserve device arrays (capacity for up to 1024 requests; adjust to your needs)
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
  cudaMalloc(&S.d_sel_seq,      S.cfg.max_slots * sizeof(int32_t));
  cudaMalloc(&S.d_sel_pos,      S.cfg.max_slots * sizeof(int32_t));
}

Engine::~Engine() {
  auto& S = *self_;
  if (S.ctx)   llama_free(S.ctx);
  if (S.model) llama_model_free(S.model);
  llama_backend_free();

  if (S.d_state)        cudaFree(S.d_state);
  if (S.d_eos)          cudaFree(S.d_eos);
  if (S.d_pos)          cudaFree(S.d_pos);
  if (S.d_req_id)       cudaFree(S.d_req_id);
  if (S.d_selected_idx) cudaFree(S.d_selected_idx);
  if (S.d_selected_cnt) cudaFree(S.d_selected_cnt);
  if (S.d_sel_seq)      cudaFree(S.d_sel_seq);
  if (S.d_sel_pos)      cudaFree(S.d_sel_pos);
  if (S.stream)         cudaStreamDestroy(S.stream);
  delete self_;
}

uint64_t Engine::add_request(const AddReq& r) {
  auto& S = *self_;
  HRequest q;
  q.id = S.next_id++;
  q.prompt_tokens = tokenize(S.vocab, r.prompt);
  q.max_new = r.max_new;
  q.state = RS_PREFILL;
  q.pos   = 0;
  q.eos   = 0;

  // assign a stable sequence id = index in S.reqs
  const int idx = (int)S.reqs.size();
  S.reqs.push_back(std::move(q));        // push first to get stable index
  HRequest& ref = S.reqs[idx];

  // ---- PREFILL
  {
    llama_batch batch = llama_batch_init((int)q.prompt_tokens.size(), /*embd*/0, /*n_seq_max*/1);
    batch.n_tokens = (int)q.prompt_tokens.size();

    // std::vector<int32_t>      n_seq_id(batch.n_tokens, 1);
    // std::vector<llama_seq_id> seq_ids_flat(batch.n_tokens);
    // std::vector<llama_seq_id*> seq_id_ptrs(batch.n_tokens);

    for (int i=0;i<batch.n_tokens;++i) {
      batch.token[i]  = q.prompt_tokens[i];
      batch.pos[i]    = q.pos++;
      // seq_ids_flat[i] = (llama_seq_id)q.id;
      // seq_ids_flat[i] = 0;
      // seq_id_ptrs[i]  = &seq_ids_flat[i];
      // batch.logits[i] = 0;
      batch.n_seq_id[i]  = 1;
      batch.seq_id[i][0] = (llama_seq_id)idx;   // <-- stable per-request seq id
      batch.logits[i]    = 0;                   // no logits during prefill
    }
    // batch.n_seq_id = n_seq_id.data();
    // batch.seq_id   = seq_id_ptrs.data();

    if (llama_decode(S.ctx, batch)) {
      std::fprintf(stderr, "llama_decode prefill failed\n"); std::abort();
    }
    llama_batch_free(batch);
  }

  // q.state = RS_DECODE;
  // S.reqs.push_back(std::move(q));

  // const int idx = (int)S.reqs.size() - 1;
  // S.h_state[idx]  = RS_DECODE;
  // S.h_eos[idx]    = 0;
  // S.h_pos[idx]    = S.reqs[idx].pos;
  // S.h_req_id[idx] = (int32_t)S.reqs[idx].id;

  // return S.reqs.back().id;
    // move to DECODE and update mirrors
  ref.state = RS_DECODE;
  S.h_state[idx]  = RS_DECODE;
  S.h_eos[idx]    = 0;
  S.h_pos[idx]    = ref.pos;
  S.h_req_id[idx] = (int32_t)ref.id;
  return ref.id;
}

std::vector<Generated> Engine::run() {
  auto& S = *self_;
  const int N = (int)S.reqs.size();
  std::vector<Generated> out; out.reserve(N);

  int32_t current_active = -1;

  int unfinished = N;
  while (unfinished > 0) {
    // sync mirrors to device
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
    if (h_count == 0) break;

    std::vector<int32_t> h_sel(h_count);
    cudaMemcpyAsync(h_sel.data(), S.d_selected_idx, h_count*sizeof(int32_t), cudaMemcpyDeviceToHost, S.stream);
    cudaStreamSynchronize(S.stream);

    // request 1 token for each selected sequence
    llama_batch batch = llama_batch_init(h_count, /*embd*/0, /*n_seq_max*/1);
    batch.n_tokens = h_count;

    // std::vector<int32_t>       n_seq_id(h_count, 1);
    // std::vector<llama_seq_id>  seq_ids_flat(h_count);
    // std::vector<llama_seq_id*> seq_id_ptrs(h_count);

    for (int k=0;k<h_count;++k) {
      int i = h_sel[k];
      HRequest& r = S.reqs[i];

      llama_token last = r.generated.empty() ? r.prompt_tokens.back() : r.generated.back();

      batch.token[k]  = last;
      batch.pos[k]    = r.pos++;
      // seq_ids_flat[k] = (llama_seq_id)r.id;
      // seq_ids_flat[k] = 0;
      // seq_id_ptrs[k]  = &seq_ids_flat[k];
      // batch.logits[k] = 1; // fetch logits for each row
      batch.n_seq_id[k]  = 1;
      batch.seq_id[k][0] = (llama_seq_id)i;     // same stable seq id used at prefill
      batch.logits[k]    = 1;                   // request logits
    }
    // batch.n_seq_id = n_seq_id.data();
    // batch.seq_id   = seq_id_ptrs.data();

    if (llama_decode(S.ctx, batch)) {
      std::fprintf(stderr, "llama_decode failed\n"); std::abort();
    }

    const float* logits_rows = llama_get_logits(S.ctx);
    for (int k=0;k<h_count;++k) {
      HRequest& r = S.reqs[h_sel[k]];
      const float* row = logits_rows + k * S.n_vocab;
      llama_token tok = (llama_token)greedy_argmax(row, S.n_vocab);
      r.generated.push_back(tok);

      if (tok == llama_vocab_eos(S.vocab) || (int)r.generated.size() >= r.max_new) {
        r.eos = 1; r.state = RS_DONE;
        S.h_eos[h_sel[k]]   = 1;
        S.h_state[h_sel[k]] = RS_DONE;
        --unfinished;

        out.push_back({ r.id, detok_text(S.vocab, r.generated) });
      } else {
        S.h_pos[h_sel[k]]   = r.pos;
        S.h_state[h_sel[k]] = RS_DECODE;
        if (S.cfg.mode == Mode::Sequential) current_active = h_sel[k];
      }
    }
    llama_batch_free(batch);
  }

  return out;
}
