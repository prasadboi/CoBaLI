#include "engine.hpp"
#include "cobali/api.hpp"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include "../cuda/batcher.cuh"

// llama.cpp C API
#include "llama.h"   // from external/llama.cpp

// ----------------- utilities -----------------

static int32_t greedy_argmax(const float *row, int n) {
    int32_t best = 0;
    float bv = row[0];
    for (int i = 1; i < n; ++i) if (row[i] > bv) { bv = row[i]; best = i; }
    return best;
}

// 2-pass tokenize using the vocab handle
static std::vector<llama_token> tok_with_vocab(const llama_vocab *vocab, const std::string &text) {
    int n = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(),
                           /*tokens*/nullptr, /*n_tokens*/0,
                           /*add_special*/true, /*parse_special*/true);
    if (n < 0) n = -n;
    std::vector<llama_token> out(n);
    int n2 = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(),
                            out.data(), (int32_t)out.size(),
                            /*add_special*/true, /*parse_special*/true);
    if (n2 < 0) n2 = -n2;
    out.resize(n2);
    return out;
}

// detokenize: grow buffer until it fits (llama_detokenize returns -needed on overflow)
static std::string detok_text(const llama_vocab *vocab, const std::vector<llama_token> &toks) {
    if (toks.empty()) return {};
    std::string buf(256, '\0');
    for (;;) {
        int wrote = llama_detokenize(vocab, toks.data(), (int32_t)toks.size(),
                                     buf.data(), (int32_t)buf.size(),
                                     /*remove_special*/true, /*unparse_special*/false);
        if (wrote >= 0) { buf.resize((size_t)wrote); return buf; }
        buf.assign(buf.size() * 2, '\0');
    }
}

// ----------------- PIMPL -----------------

struct Engine::Impl {
    RunnerConfig cfg;

    llama_model       *model  = nullptr;
    const llama_vocab *vocab  = nullptr;
    llama_context     *ctx    = nullptr;

    int n_vocab = 0;

    std::vector<HRequest> reqs;

    // host mirrors for CUDA selection
    std::vector<int32_t> h_state, h_eos, h_pos, h_req_id;

    // device buffers (selection only)
    int32_t *d_state         = nullptr;
    int32_t *d_eos           = nullptr;
    int32_t *d_pos           = nullptr;
    int32_t *d_req_id        = nullptr;
    int32_t *d_selected_idx  = nullptr;
    int32_t *d_selected_cnt  = nullptr;
    int32_t *d_sel_seq       = nullptr; // optional metadata
    int32_t *d_sel_pos       = nullptr; // optional metadata
    cudaStream_t stream      = nullptr;

    uint64_t next_id = 1;

    explicit Impl(const RunnerConfig &c) : cfg(c) {}
};

// ----------------- Engine -----------------

Engine::Engine(const RunnerConfig &cfg) : self_(new Impl(cfg)) {
    auto &S = *self_;

    // Backend init (no-arg variant in current headers)
    llama_backend_init();  // call once per process. See header. :contentReference[oaicite:0]{index=0}

    // Model params (GPU offload lives here)
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only  = false;
    if (cfg.gpu_layers >= 0) mp.n_gpu_layers = cfg.gpu_layers; // documented in llama_model_params :contentReference[oaicite:1]{index=1}

    S.model = llama_model_load_from_file(cfg.model_path.c_str(), mp);
    if (!S.model) {
        std::fprintf(stderr, "failed to load model: %s\n", cfg.model_path.c_str());
        std::abort();
    }

    S.vocab = llama_model_get_vocab(S.model);               // get vocab handle :contentReference[oaicite:2]{index=2}
    if (!S.vocab) { std::fprintf(stderr, "no vocab\n"); std::abort(); }
    S.n_vocab = llama_vocab_n_tokens(S.vocab);              // vocab size :contentReference[oaicite:3]{index=3}

    // Context params
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = cfg.n_ctx;

    // IMPORTANT: keep n_seq_max within the runtime cap. Current builds error if > 256.
    // We set it to 1 for sequential mode, else to min(max_slots, 256).
    // (see n_seq_max field in llama_context_params; cap discussed in issues / runtime error) :contentReference[oaicite:4]{index=4}
    {
        uint32_t want = (cfg.mode == Mode::Sequential) ? 1u : (uint32_t)cfg.max_slots;
        cp.n_seq_max = std::max<uint32_t>(1u, std::min<uint32_t>(want, 256u));
    }

    S.ctx = llama_init_from_model(S.model, cp);             // preferred initializer :contentReference[oaicite:5]{index=5}
    if (!S.ctx) { std::fprintf(stderr, "failed to create context\n"); std::abort(); }

    // CUDA stream + selection buffers
    cudaStreamCreate(&S.stream);

    const int CAP = 1024; // capacity of host mirrors; adjust as needed
    S.h_state.assign(CAP, 0);
    S.h_eos.assign(CAP, 0);
    S.h_pos.assign(CAP, 0);
    S.h_req_id.assign(CAP, 0);

    cudaMalloc(&S.d_state, CAP * sizeof(int32_t));
    cudaMalloc(&S.d_eos,   CAP * sizeof(int32_t));
    cudaMalloc(&S.d_pos,   CAP * sizeof(int32_t));
    cudaMalloc(&S.d_req_id,CAP * sizeof(int32_t));

    cudaMalloc(&S.d_selected_idx, S.cfg.max_slots * sizeof(int32_t));
    cudaMalloc(&S.d_selected_cnt, sizeof(int32_t));
    cudaMalloc(&S.d_sel_seq,      S.cfg.max_slots * sizeof(int32_t));
    cudaMalloc(&S.d_sel_pos,      S.cfg.max_slots * sizeof(int32_t));
}

Engine::~Engine() {
    auto &S = *self_;
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

uint64_t Engine::add_request(const AddReq &r) {
    auto &S = *self_;

    // Allocate slot (stable index == seq_id we’ll use)
    const int idx = (int)S.reqs.size();
    S.reqs.emplace_back();
    HRequest &q = S.reqs[idx];

    q.id            = S.next_id++;
    q.prompt_tokens = tok_with_vocab(S.vocab, r.prompt);
    q.max_new       = r.max_new;
    q.state         = RS_PREFILL;
    q.pos           = 0;
    q.eos           = 0;

    // ---- PREFILL: feed the entire prompt in one batch
    {
        const int n_tok = (int)q.prompt_tokens.size();
        llama_batch batch = llama_batch_init(n_tok, /*embd*/0, /*n_seq_max*/1); // 1 seq per row :contentReference[oaicite:6]{index=6}
        batch.n_tokens = n_tok;

        for (int i = 0; i < n_tok; ++i) {
            batch.token[i]     = q.prompt_tokens[i];
            batch.pos[i]       = q.pos++;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = (llama_seq_id)idx;    // stable per-request sequence id
            batch.logits[i]    = 0;                    // no logits on prefill
        }

        if (llama_decode(S.ctx, batch)) {              // run graph for prefill
            std::fprintf(stderr, "llama_decode prefill failed\n");
            std::abort();
        }
        llama_batch_free(batch);
    }

    // Move to DECODE and update host mirrors used by CUDA selectors
    q.state = RS_DECODE;
    S.h_state[idx]  = RS_DECODE;
    S.h_eos[idx]    = 0;
    S.h_pos[idx]    = q.pos;
    S.h_req_id[idx] = (int32_t)q.id;

    return q.id;
}

std::vector<Generated> Engine::run() {
    auto &S = *self_;
    const int N = (int)S.reqs.size();
    std::vector<Generated> out; out.reserve(N);

    int32_t current_active = -1;
    int unfinished = N;

    while (unfinished > 0) {
        // sync mirrors to device for selection
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

        // Build a decode batch: 1 row per selected sequence; ask for logits on every row
        llama_batch batch = llama_batch_init(h_count, /*embd*/0, /*n_seq_max*/1);
        batch.n_tokens = h_count;

        for (int k = 0; k < h_count; ++k) {
            const int i = h_sel[k];
            HRequest &r = S.reqs[i];

            // last token (prompt tail if no generated yet)
            const llama_token last = r.generated.empty() ? r.prompt_tokens.back() : r.generated.back();

            batch.token[k]     = last;
            batch.pos[k]       = r.pos++;
            batch.n_seq_id[k]  = 1;
            batch.seq_id[k][0] = (llama_seq_id)i;   // same stable seq id as prefill
            batch.logits[k]    = 1;                 // request logits for this row
        }

        if (llama_decode(S.ctx, batch)) {
            std::fprintf(stderr, "llama_decode failed\n");
            std::abort();
        }

        // Per the header: rows are stored contiguously for entries where logits!=0; cols = n_vocab. :contentReference[oaicite:7]{index=7}
        const float *rows = llama_get_logits(S.ctx);

        for (int k = 0; k < h_count; ++k) {
            HRequest &r = S.reqs[h_sel[k]];
            const float *row = rows + k * S.n_vocab;

            const llama_token tok = (llama_token)greedy_argmax(row, S.n_vocab); // or use samplers
            r.generated.push_back(tok);

            const bool hit_eos = llama_vocab_is_eog(S.vocab, tok) || (tok == llama_vocab_eos(S.vocab)); // conservative EOS/EOG check :contentReference[oaicite:8]{index=8}
            const bool hit_len = (int)r.generated.size() >= r.max_new;

            if (hit_eos || hit_len) {
                r.eos   = 1;
                r.state = RS_DONE;
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
