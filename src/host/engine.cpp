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
#include <deque>
#include <utility>
#include <string>

#include <cuda_runtime.h>
#include "../cuda/batcher.cuh"

#include "llama.h"   // this comes from external/llama.cpp

struct Engine::Impl {
  RunnerConfig cfg;

  llama_model       * model  = nullptr;
  const llama_vocab * vocab  = nullptr;
  llama_context     * ctx    = nullptr;

  int n_vocab = 0;

  std::vector<HRequest> reqs;
  
  // this line represents the pool of available KV cache slots
  std::deque<int> free_slots;

  // this is the host mirrors used by CUDA selection
  std::vector<int32_t> h_state, h_eos, h_pos, h_req_id;

  //device buffers
  int32_t *d_state=nullptr, *d_eos=nullptr, *d_pos=nullptr, *d_req_id=nullptr;
  int32_t *d_selected_idx=nullptr, *d_selected_cnt=nullptr;
  int32_t *d_sel_seq=nullptr, *d_sel_pos=nullptr;
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

//this has been tokenized using vocab -> 2-pass size + write
static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string& text) {
  int n = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(), nullptr, 0, true, true);
  if (n < 0) n = -n;
  std::vector<llama_token> out(n);
  int n2 = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(), out.data(), (int32_t)out.size(), true, true);
  if (n2 < 0) n2 = -n2;
  out.resize(n2);
  return out;
}

//detokenize
static std::string detok_text(const llama_vocab * vocab, const std::vector<llama_token>& toks) {
  if (toks.empty()) return {};
  int cap = 256;
  std::string buf(cap, '\0');
  while (true) {
    int wrote = llama_detokenize(vocab, toks.data(), (int32_t)toks.size(), buf.data(), (int32_t)buf.size(), true, false);
    if (wrote >= 0) { buf.resize(wrote); return buf; }
    cap *= 2; buf.assign(cap, '\0');
  }
}

Engine::Engine(const RunnerConfig& cfg): self_(new Impl(cfg)) {
  auto& S = *self_;

  if (S.cfg.prefill_chunk_tokens <= 0) {
    S.cfg.prefill_chunk_tokens = 1;
  }

  llama_backend_init();

  llama_model_params mp = llama_model_default_params();
  mp.vocab_only = false;
  if (cfg.gpu_layers >= 0) {
    mp.n_gpu_layers = cfg.gpu_layers;
  }

  S.model = llama_model_load_from_file(cfg.model_path.c_str(), mp);
  if (!S.model) {
    std::fprintf(stderr, "failed to load model: %s\n", cfg.model_path.c_str());
    std::abort();
  }

  S.vocab = llama_model_get_vocab(S.model);
  if (!S.vocab) { std::fprintf(stderr, "failed to get vocab from model\n"); std::abort(); }

  llama_context_params cp = llama_context_default_params();
  cp.n_ctx    = cfg.n_ctx;
  
  int effective_max_slots = (cfg.mode == Mode::Sequential) ? 1 : std::max(1, cfg.max_slots);
  cp.n_seq_max = effective_max_slots;
  
  for (int i = 0; i < effective_max_slots; ++i) {
    S.free_slots.push_back(i);
  }
  
  S.ctx = llama_init_from_model(S.model, cp);
  if (!S.ctx) { std::fprintf(stderr, "failed to create context\n"); std::abort(); }

  S.n_vocab = llama_vocab_n_tokens(S.vocab);

  cudaStreamCreate(&S.stream);

  //this is to reserve device arrays
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
  q.slot_id = -1;

  //this is for assigning a stable sequence id
  const int idx = (int)S.reqs.size();

  if (idx >= (int) S.h_state.size()) {
    std::fprintf(stderr,
      "Too many queued requests (%d). Increase engine capacity "
      "(currently %zu; see N in Engine::Engine).\n",
      idx + 1, S.h_state.size());
    std::abort();
  }

  S.reqs.push_back(std::move(q));
  HRequest &ref = S.reqs[idx];

  ref.state          = RS_PREFILL;
  ref.prefill_cursor = 0;
  S.h_state[idx]     = RS_PREFILL;
  S.h_eos[idx]       = 0;
  S.h_pos[idx]       = ref.pos;
  S.h_req_id[idx]    = (int32_t) ref.id;

  return ref.id;
}

std::vector<Generated> Engine::run() {
  auto& S = *self_;
  const int N = (int)S.reqs.size();
  std::vector<Generated> out; out.reserve(N);

  int32_t current_active = -1;

  const auto run_prefill_round = [&]() {
    if (S.cfg.mode != Mode::Continuous) return;

    struct WorkItem {
      int idx;
      int chunk;
    };
    std::vector<WorkItem> work;
    work.reserve(S.cfg.max_slots);

    const int chunk_cap = std::max(1, S.cfg.prefill_chunk_tokens);
    for (int i = 0; i < N; ++i) {
      if ((int)work.size() >= S.cfg.max_slots) break;
      
      HRequest & r = S.reqs[i];
      if (r.state != RS_PREFILL || r.eos) continue;
      if (r.slot_id < 0) {
        if (S.free_slots.empty()) {
          continue;
        }
        r.slot_id = S.free_slots.front();
        S.free_slots.pop_front();
        
        //this is for clearing the KV cache for this particular slot before new usage
        llama_memory_seq_rm(llama_get_memory(S.ctx), r.slot_id, 0, -1);
      }

      int remain = (int)r.prompt_tokens.size() - r.prefill_cursor;
      if (remain <= 0) {
        r.state = RS_DECODE;
        S.h_state[i] = RS_DECODE;
        continue;
      }
      work.push_back({ i, std::min(remain, chunk_cap) });
    }

    if (work.empty()) return;

    int total_tokens = 0;
    for (auto & w : work) total_tokens += w.chunk;
    if (total_tokens == 0) return;

    llama_batch batch = llama_batch_init(total_tokens, 0, 1);
    batch.n_tokens = total_tokens;

    int cursor = 0;
    for (auto & w : work) {
      HRequest & r = S.reqs[w.idx];
      for (int t = 0; t < w.chunk; ++t) {
        batch.token[cursor]     = r.prompt_tokens[r.prefill_cursor];
        batch.pos[cursor]       = r.pos++;
        batch.n_seq_id[cursor]  = 1;
        batch.seq_id[cursor][0] = (llama_seq_id)r.slot_id;
        batch.logits[cursor]    = 0;
        ++cursor;
        ++r.prefill_cursor;
      }
      if (r.prefill_cursor >= (int)r.prompt_tokens.size()) {
        r.state = RS_DECODE;
        S.h_state[w.idx] = RS_DECODE;
      }
      S.h_pos[w.idx] = r.pos;
    }

    if (llama_decode(S.ctx, batch)) {
      std::fprintf(stderr, "llama_decode prefill split failed\n"); std::abort();
    }
    llama_batch_free(batch);
  };


  int unfinished = N;
  while (unfinished > 0) {
    if (S.cfg.mode == Mode::Sequential && current_active == -1) {
        int pick = -1;
        for (int i = 0; i < (int) S.reqs.size(); ++i) {
            if (S.reqs[i].state == RS_PREFILL) { pick = i; break; }
        }
        if (pick != -1) {
            HRequest & r = S.reqs[pick];
            
            r.slot_id = 0; 
            
            //this is for clearing the KV for slot 0
            llama_memory_seq_rm(llama_get_memory(S.ctx), 0, 0, -1);
            
            llama_batch batch = llama_batch_init((int) r.prompt_tokens.size(), 0, 1);
            batch.n_tokens = (int) r.prompt_tokens.size();
            r.pos = 0;
            r.prefill_cursor = 0;
            for (int t = 0; t < batch.n_tokens; ++t) {
                batch.token[t]     = r.prompt_tokens[t];
                batch.pos[t]       = r.pos++;
                batch.n_seq_id[t]  = 1;
                batch.seq_id[t][0] = (llama_seq_id) 0;
                batch.logits[t]    = 0;
            }
            if (llama_decode(S.ctx, batch)) { std::fprintf(stderr, "llama_decode prefill failed\n"); std::abort(); }
            llama_batch_free(batch);

            r.state          = RS_DECODE;
            r.prefill_cursor = (int) r.prompt_tokens.size();
            S.h_state[pick]  = RS_DECODE;
            S.h_pos[pick]    = r.pos;
            S.h_eos[pick]    = 0;
            current_active   = pick;
        }
    }

    if (S.cfg.mode == Mode::Continuous) {
      run_prefill_round();
    }
    
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
    if (h_count == 0) continue;

    std::vector<int32_t> h_sel(h_count);
    cudaMemcpyAsync(h_sel.data(), S.d_selected_idx, h_count*sizeof(int32_t), cudaMemcpyDeviceToHost, S.stream);
    cudaStreamSynchronize(S.stream);

    llama_batch batch = llama_batch_init(h_count, 0, 1);
    batch.n_tokens = h_count;

    for (int k=0;k<h_count;++k) {
      int i = h_sel[k];
      HRequest& r = S.reqs[i];

      if (r.slot_id < 0) {
          fprintf(stderr, "Error: selected request %d has no slot\n", i);
          abort();
      }

      llama_token last = r.generated.empty() ? r.prompt_tokens.back() : r.generated.back();

      batch.token[k]  = last;
      batch.pos[k]    = r.pos++;
      batch.n_seq_id[k]  = 1;
      batch.seq_id[k][0] = (llama_seq_id)r.slot_id;
      batch.logits[k]    = 1;
    }

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
        
        if (r.slot_id >= 0) {
            S.free_slots.push_back(r.slot_id);
            r.slot_id = -1;
        }
        
        --unfinished;
        out.push_back({ r.id, detok_text(S.vocab, r.generated) });
        if (S.cfg.mode == Mode::Sequential) current_active = -1;
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
