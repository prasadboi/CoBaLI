# CoBaLI: Continuous Batching and Prefill Splitting for LLM Inference on a Single GPU

Lakshay Dua (ld3074), Arjun Parasuram Prasad (ap9334)

## Abstract

LLM inference on commodity GPUs often leaves a large fraction of available compute idle due to uneven request patterns and rigid execution loops. Simple servers that handle one request at a time are easy to build, but they underutilise the GPU and give poor latency for mixed workloads. Continuous batching and prefill splitting are two known techniques that improve utilisation by overlapping work across requests, yet most open-source implementations come as part of large, complex runtimes.

We present CoBaLI, a small C++/CUDA engine that implements sequential serving, continuous batching, and continuous batching with prefill splitting on top of the existing llama.cpp backend. CoBaLI treats llama.cpp as a black box that owns model weights, attention kernels, and KV cache layout. We do not modify any kernels. We instead manage request state and KV cache slots explicitly on the host, select active requests with a lightweight CUDA helper, and construct `llama_batch` objects that drive the standard `llama_decode` API.

We evaluate CoBaLI on Qwen2.5-0.5B-Instruct quantised for llama.cpp on a single RTX 4070 GPU (cuda5.cims.nyu.edu). We use a long-prompt workload with 186 prompts for end-to-end timing and a 16-prompt workload for profiling with Nsight Systems and Nsight Compute. Our experiments show that continuous batching improves throughput over a sequential baseline and that prefill splitting provides an additional gain when we choose a moderate chunk size. All improvements come from scheduling benefits and KV cache management; the underlying GPU kernels (e.g. RoPE, Matrix Multiplications, etc.) and model implementation remain unchanged.

## 1 Introduction

LLM inference places an irregular and bursty workload on GPUs. Users send prompts with very different lengths and ask for very different numbers of output tokens. A serving system that processes requests strictly one by one is simple and robust, but it often wastes GPU capacity. A single long prompt can occupy the device for many milliseconds while a queue of short queries waits, even though the GPU could have processed several of them in parallel.

Most simple servers follow the same pattern. They run a full prefill over the input prompt, then enter a token-by-token decode loop for that request, and only when the response finishes do they start the next request. This design never overlaps work across users. The prefill phase processes many tokens at once and is often memory bound. The decode phase extends each sequence by one token and is often compute bound. When we keep these phases isolated per request, we fail to exploit the fact that different requests sit in different phases at the same time.

Continuous batching and prefill splitting aim to fix this scheduling problem. Continuous batching keeps a set of active requests on the GPU and advances several of them at every decode step. When one request finishes, the scheduler immediately pulls in a new one so that the GPU always sees a batch of useful work. Prefill splitting targets very long prompts by breaking each long prefill into smaller chunks. The server then interleaves these chunks with decode steps and short prefills from other requests. Together, these ideas increase the effective batch size, hide small idle periods, and reduce end-to-end latency for mixed workloads.

Existing systems such as vLLM and TensorRT-LLM already implement continuous batching and related techniques, but they usually come with custom runtimes, specialised memory managers, and many moving parts. We instead focus on a small, transparent implementation that stays close to an existing backend. We build on llama.cpp, which already provides efficient kernels, quantised model support, and a C++ API for running transformer models on commodity GPUs.

We introduce CoBaLI, a C++/CUDA engine that implements three execution modes on top of the llama.cpp C API: a sequential baseline, continuous batching, and continuous batching with prefill splitting. CoBaLI treats llama.cpp as a black box. We leave all matmul and attention kernels untouched. We manage request state, map requests to KV cache slots, and construct `llama_batch` objects that tell llama.cpp which tokens to process for which sequences. A small CUDA kernel helps choose which requests to advance on each iteration, while the main scheduling loop remains on the host.

Our implementation targets Qwen2.5-0.5B-Instruct quantised for llama.cpp and runs on a single RTX 4070 GPU with 12 GB of memory on the CIMS cuda5 node. We expose all scheduling choices through a single runner binary, `cobali_runner`, and support all three modes in the same code base. We evaluate CoBaLI on a prompt workload of 186 prompts for end-to-end timing and on a smaller 16-prompt workload for detailed profiling with Nsight Systems and Nsight Compute. The results in the following sections show that continuous batching and prefill splitting provide clear throughput and latency improvements over the sequential baseline, using only host-side scheduling and KV management on top of an unchanged llama.cpp backend.

## 2 Literature Survey

We place our work in the context of recent systems for large language model (LLM) serving. Most of these systems view inference as two main stages: a prefill stage that runs the full prompt and builds the key–value (KV) cache, and a decode stage that extends each sequence token by token from the cache. The main design questions are how to batch requests across these stages, how to manage KV memory, and how much complexity to push into the runtime and kernels.

Kwon et al. introduce PagedAttention and vLLM, which manage the KV cache with a paging abstraction and support token-level continuous batching in a custom runtime.[1] PagedAttention lets the server mix requests with very different prompt and generation lengths while keeping high GPU utilisation. However, vLLM owns the full inference stack: it defines its own KV layout, implements new attention kernels, and exposes a separate serving API. Our work borrows the idea of continuous batching but keeps the underlying kernels and layout from an existing backend, llama.cpp, instead of replacing them.

Other open-source servers also use continuous batching but again come with their own runtimes. Hugging Face Text Generation Inference (TGI) exposes a production-ready HTTP server with features such as routing, streaming, and dynamic batching across multiple backends.[4] TGI can schedule prefill and decode work for many concurrent clients, but it ties batching logic to a larger service stack with gRPC, token streaming, and cluster deployment options. Sarathi-Serve targets the throughput–latency trade-off directly and proposes stall-free batching strategies that split long prefills into chunks and combine them with decode tokens.[3] Sarathi-Serve runs on data-centre GPUs and relies on a specialised runtime to implement its policies. We share the basic motivation behind chunked prefill, but we aim for a smaller design that stays inside the llama.cpp C API and focuses on a single GPU.

FineInfer looks at another dimension: it wants to run fine-tuning and inference together on the same hardware.[2] The system introduces deferred continuous batching, which can delay some tokens slightly so that the scheduler can switch between fine-tuning iterations and inference steps without breaking latency objectives. FineInfer shows that careful scheduling alone can unlock a lot of utilisation in constrained environments. We follow the same spirit and treat scheduling as the main lever, but we only target pure inference and we do not change model-parallel or data-parallel layouts.

Multi-tenant LoRA serving systems add more structure to the workload. Punica serves many LoRA adapters on a shared GPU by introducing a new segmented gather matrix–vector kernel and a scheduler that consolidates LoRA requests.[5] CaraServe keeps the base model on the GPU, streams LoRA adapters from CPU memory, and uses CPU-assisted prefilling together with rank-aware scheduling to cut cold-start latency and meet per-request SLOs.[6] These systems show that co-designing kernels, memory movement, and scheduling gives large gains when many adapted models share one base model. In contrast, we only serve a single base model without LoRA adapters and focus on the simpler case where all requests use the same set of weights.

Our work also builds directly on llama.cpp, a widely used C/C++ inference library that provides quantised models, GPU backends, and a simple C API for running transformer models on commodity hardware.[7] We treat llama.cpp as a black-box backend that owns model weights, attention kernels, and the KV cache. We do not modify its kernels or internal data structures. Instead we manage request state and KV slots on the host and construct `llama_batch` objects that drive the existing `llama_decode` entry point.

Finally, we evaluate our engine on Qwen2.5-0.5B-Instruct, a small member of the Qwen2.5 family.[8] The Qwen2.5 technical report focuses on model architecture, training, and instruction tuning; it does not prescribe any serving strategy. We therefore treat Qwen2.5-0.5B-Instruct as a representative small LLM that fits comfortably on a single RTX 4070 and lets us isolate the effects of scheduling.

In summary, prior work shows that continuous batching, careful KV management, and prefill-aware scheduling are powerful tools for improving LLM serving. Most of these systems, however, introduce new runtimes, custom kernels, or cluster-level components. We instead target a narrow but practical setting: a single GPU, a single model, and an existing backend. We show that a thin C++/CUDA layer that manages request state and chunks prefills on top of llama.cpp can already bring many of the benefits of larger systems, without changing kernels or adding a complex serving stack.

## 3 Proposed Idea

We design CoBaLI as a small engine that wraps the llama.cpp C API. We keep the backend simple and focus on scheduling and KV cache management on the host side. Our goals are:

- Keep llama.cpp kernels and internal data structures untouched.
- Represent request state and KV cache slots explicitly on the host.
- Move the scheduling loop to the host side with a small CUDA helper kernel.
- Support a sequential baseline and continuous modes in the same implementation.

CoBaLI is implemented in C++/CUDA. The engine links against llama.cpp and is exposed through a single binary, `cobali_runner`, which controls the mode, number of slots, context length, and prefill chunk size.

### 3.1 Request and Slot Model

Each user request becomes an `HRequest` structure on the host. The structure stores:

- a unique request ID,
- the tokenised prompt (`prompt_tokens`),
- the list of generated tokens (`generated`),
- the current absolute position `pos`,
- a `prefill_cursor` index that tracks how many prompt tokens we have already fed,
- a request state flag (`RS_PREFILL`, `RS_DECODE`, or `RS_DONE`),
- an `eos` flag and a `max_new` limit,
- a `slot_id` that maps the request to a KV cache slot.

We fix the maximum number of KV cache slots to `max_slots`. Each slot holds the KV cache for one logical sequence. llama.cpp already supports multiple sequences via `seq_id` and `n_seq_max`. We reuse this mechanism and map our `slot_id` directly to the `seq_id` field in the `llama_batch` structure.

We track free slots in a deque on the host. When we assign a slot to a request, we clear its KV cache with `llama_memory_seq_rm` and then use that slot until the request finishes. When the request reaches EOS or hits its `max_new` limit, we mark it as `RS_DONE`, push the slot back into the free list, and clear its KV cache again. This model lets us reason in terms of requests and slots without touching any internal KV layout inside llama.cpp.

### 3.2 Sequential Baseline

Sequential mode provides a clear baseline and mirrors a simple one-request-at-a-time server. We configure llama.cpp with `n_seq_max = 1` and only use slot 0. The engine runs the following loop:

1. Pick the next request whose state is `RS_PREFILL`.
2. Assign it slot 0 and clear the KV cache for that slot.
3. Run a single prefill pass that feeds the entire prompt into `llama_decode` using a `llama_batch`.
4. Move the request to `RS_DECODE`.
5. Decode tokens one by one until EOS or `max_new`, using greedy argmax over the logits.
6. Move the request to `RS_DONE`, release the slot, and continue with the next request.

The decode loop uses the same selection code path as continuous mode, but with only one active request. Sequential mode always selects the single request in `RS_DECODE`, and the GPU sees one token per forward pass. This setup gives a straightforward baseline for the timing and profiling results in Section 4.

### 3.3 Continuous Batching

Continuous mode aims to keep several requests decoding at the same time. We keep the same request representation and slot abstraction and change only the scheduling logic.

The engine maintains small host arrays `h_state`, `h_eos`, `h_pos`, and `h_req_id`, one entry per request. We mirror these arrays on the device as `d_state`, `d_eos`, `d_pos`, and `d_req_id`. On each iteration of the main loop, we copy the host arrays to the device and launch a CUDA kernel that selects which requests should decode a token in the next step.

The kernel `cobali_select_continuous` scans all requests. It selects indices of requests that are in `RS_DECODE` and not yet finished. It appends these indices into a device buffer `d_selected_idx` until it has chosen `max_slots` requests or runs out of candidates. The kernel stores the final count in `d_selected_cnt` using an atomic counter.

We then copy the selected indices back to the host. If the count is zero, all requests are either still in prefill or already done, and the loop moves on to prefill work. If the count is positive, we build a `llama_batch` of that size:

- For each selected request, we take its last token: the last prompt token when it has generated nothing yet, or the last generated token otherwise.
- We place that token into the batch input for the sequence.
- We set the position for that row to the current `pos` of the request.
- We set `n_seq_id = 1` and `seq_id[0] = slot_id` so that the row writes into the correct KV slot.
- We set `logits = 1` so that llama.cpp produces logits for every row.

We call `llama_decode` with this batch. For each row we read the logits, select the token with maximum logit, append it to `generated`, and update the request state. If the token is EOS or the request has produced `max_new` tokens, we mark the request as `RS_DONE`, set its `eos` flag, and free the associated slot. Otherwise we keep it in `RS_DECODE` and increment its position `pos`.

Continuous mode therefore keeps the inner computation entirely inside llama.cpp, while CoBaLI only controls which sequences enter each `llama_decode` call and which KV slots they use. This is the mode we use for the “continuous, no prefill split” configuration in Section 4.

### 3.4 Prefill Splitting

Prefill splitting adds another layer of scheduling. Long prompts can make the GPU sit in prefill for a single request while other requests wait in `RS_PREFILL` or `RS_DECODE`. We want to overlap long prefills with the decode steps of other requests without touching any backend kernels.

We add a parameter `prefill_chunk_tokens`. We then introduce a helper function `run_prefill_round()` that we call on each loop iteration in continuous mode. This function works as follows:

1. Scan all requests on the host.
2. For each request in `RS_PREFILL`, assign a free KV slot if it does not already have one. When we assign a slot, we clear its KV cache.
3. Compute how many prompt tokens remain for that request by comparing the prompt length and `prefill_cursor`. If no tokens remain, move the request to `RS_DECODE`.
4. Otherwise create a work item with the pair (request index, chunk size), where the chunk size is `min(remaining prompt tokens, prefill_chunk_tokens)`.
5. Stop adding work items when we have scheduled `max_slots` requests or we run out of eligible prefills.

We build a `llama_batch` that concatenates all scheduled prefill chunks. For each work item we fill the token, `pos`, and `seq_id` fields so that the chunk writes into the KV region of the correct slot. We call `llama_decode` once for this combined batch. After the call we advance `prefill_cursor` and `pos` for every request that participated. If a request has consumed all prompt tokens, we move it to `RS_DECODE` and let the continuous decode scheduler handle it.

Prefill splitting allows multiple long prefills to overlap with each other and with ongoing decode steps. Chunk size controls the trade-off between scheduling overhead and overlap. Very small chunks increase host-side work and device–host copies but give more interleaving opportunities. Very large chunks behave like a full-prompt prefill and let a single request hold on to the GPU for longer.

We keep all attention and matmul kernels inside llama.cpp unchanged. CoBaLI uses only public entry points such as `llama_batch_init`, `llama_decode`, and the KV cache utilities. The experiments in Section 4 show that a moderate chunk size around 256 tokens gives the best performance on our long-prompt workload on the RTX 4070, while aggressive splitting at 128 tokens reveals how prefill splitting changes launch behaviour and GPU utilisation in the Nsight profiles.

## 4 Experimental Setup

We evaluate our engine on a single-machine setup.

| Item           | Value (example)                                                       |
| -------------- | --------------------------------------------------------------------- |
| GPU            | NVIDIA GPU with CUDA support (RTX 4070 (cuda5 on CIMS))              |
| CUDA toolkit   | CUDA 12.4 (matching the driver on the system)                        |
| Nsight Systems | `nsys` CLI in PATH (e.g., from CUDA toolkit or Nsight bundle)        |
| Nsight Compute | `ncu` CLI in PATH                                                    |
| TMPDIR         | `ncu_tmp`                                                            |
| Model          | `models/qwen2.5-0.5b-instruct-q5_k_m.gguf`                           |
| Runner binary  | `build/cobali_runner`                                                |
| Prompts file   | `workloads/prompts_*`                                                |

Table 1: Profiling environment assumptions for the Nsight Systems and Nsight Compute experiments.

We compile the project with CMake and link against llama.cpp. We expose a single binary `cobali_runner` with command-line options for model path, prompts file, mode, number of slots, context length, and prefill chunk size.

### 4.1 Workload

We use `prompts_186.txt` (186 prompts) for conducting large scale experiments. For profiling tasks we use a smaller input file – `prompts_16.txt` (16 prompts). Prompts span multiple sentences and contain many tokens.

### 4.2 Modes and Parameters

We expose all scheduling choices through a single runner binary, `cobali_runner`. The binary takes command-line flags for the model path, prompts file, mode, maximum number of slots, context length, and prefill chunk size. Unless stated otherwise, all experiments in Experiments section use the same quantized model (`qwen2.5-0.5b-instruct-q5_k_m.gguf`) and a fixed context length of 16,384 tokens.

For the large-scale timing experiments in Experiment’s section we compare three scheduling configurations that match the rows in Table 2:

- **Sequential (SEQ):**
  - `--mode seq`
  - `--max-slots 1`

  The engine runs one request to completion before it starts the next. This configuration is our baseline for all reported speedups.

- **Continuous, no prefill split (CONT-SPLIT512):**
  - `--mode cont`
  - `--max-slots 16`
  - `--prefill-chunk-tokens 512`

  We use 16 KV-cache slots to keep up to 16 requests active at once. The chunk size of 512 tokens is larger than all prompts in the long-prompt workload, so each prefill runs as a single chunk. This setting therefore behaves as “continuous batching without prefill splitting” and corresponds to row 2 in Table 2.

- **Continuous with prefill splitting (CONT-SPLIT128 / CONT-SPLIT256):**
  - `--mode cont`
  - `--max-slots 16`
  - `--prefill-chunk-tokens 128` or `256`

  We keep the number of slots fixed at 16 and only change the prefill chunk size. CONT-SPLIT128 (row 3 in Table 2) uses 128-token chunks and represents an aggressive splitting policy. CONT-SPLIT256 (row 4) uses 256-token chunks and turns out to be the best trade-off between batching efficiency and scheduling overhead for our long-prompt workload.

For the Nsight Systems and Nsight Compute profiles reported later in this section, we reuse the same mode settings but run on the smaller 16-prompt workload (`prompts_16.txt`) to keep trace sizes manageable while preserving the relative behaviour between modes. All long-run timing experiments use the large 186-prompt workload (`prompts_186.txt`), and all profiling experiments use the 16-prompt workload (`prompts_16.txt`).

## 5 Experiments & Analysis

We summarise the timing results for the long-prompt workload in Table 2. The run uses the workload file `../workloads/prompts_186.txt`, which contains 186 prompts. All continuous batching runs use a maximum of 16 parallel slots (`--max-slots 16`).

We measure wall time in seconds, and compute speedup relative to the sequential baseline (154.17 s).

The profiling results report:

- Sequential mode: wall time 154.17 s, user time 34.68 s.
- Continuous mode, prefill split 512 (`cont-split512`): wall time 32.64 s, user time 17.64 s.
- Continuous mode, prefill split 128 (`cont-split128`): wall time 43.00 s, user time 20.08 s.
- Continuous mode, prefill split 256 (`cont-split256`): wall time 24.69 s, user time 16.47 s.

| Mode | Description                      | Wall Time (s) | User Time (s) | Speedup vs. Seq |
| ---- | -------------------------------- | ------------- | ------------- | --------------- |
| 1) SEQ           | Sequential (1 slot)              | 154.17        | 34.68         | 1.00×           |
| 2) CONT-SPLIT512 | Continuous, no prefill split     | 32.64         | 17.64         | 4.72×           |
| 3) CONT-SPLIT128 | Continuous, prefill split 128    | 43.00         | 20.08         | 3.59×           |
| 4) CONT-SPLIT256 | Continuous, prefill split 256    | 24.69         | 16.47         | 6.24×           |

Table 2: Timing results for 186 prompts using Qwen2.5-0.5B-Instruct on an RTX 4070 with context length 16384.

We observe that `cont-split256` provides the best performance, achieving a 6.24× speedup over the sequential baseline. This indicates a clear performance “sweet spot” around a chunk size of 256. Chunk 128 is slower because splitting becomes too granular, increasing scheduling and kernel-launch overhead. Chunk 512 is faster than 128, but slightly suboptimal due to longer uninterrupted prefills that can stall other slots. Chunk 256 achieves the best balance between batching efficiency and prefill parallelism.

### 5.1 Profiler Plots from Nsight Systems and Nsight Compute

We also profile the engine with Nsight Systems and Nsight Compute to understand CPU launch behaviour and GPU kernel efficiency. Tables 3 and 4 summarise the launch-gap statistics and effective launch throughput.

### 5.2 Effect of Continuous Batching

Continuous batching already improves performance over the simple sequential baseline. For the long-prompt workload in Table 2, moving from sequential execution to continuous mode without prefill splitting (row 2) reduces wall-clock time from 154.17 s to 32.64 s, a 4.72× speedup. Given 186 prompts, this corresponds to an increase in throughput from roughly 1.2 to 5.7 requests per second, consistent with the speedup column in Table 2.

The main gain comes from overlapping decode steps of different requests. Short requests no longer wait behind long ones. The GPU sees more tokens per `llama_decode` call because multiple sequences decode in parallel as long as there are free slots.

| Metric             | Sequential | Continuous |
| ------------------ | ---------- | ---------- |
| Launch count       | 30,828     | 200,018    |
| Mean gap (ns)      | 170,192    | 29,575     |
| Median gap (ns)    | 4,918      | 2,759      |
| P75 gap (ns)       | 10,610     | 3,795      |
| P90 gap (ns)       | 32,763     | 4,897      |
| P95 gap (ns)       | 832,487    | 6,460      |
| Max gap (ns)       | 253,929,568| 184,487,963|
| Min gap (ns)       | 804        | 857        |
| Mean API dur (ns)  | 26,261     | 11,253     |
| Median API dur (ns)| 11,286     | 8,240      |

Table 3: Launch gap and CUDA runtime API duration statistics from Nsight Systems for sequential and continuous modes.

| Mode       | Launches/s | Timeline span (s) |
| ---------- | ---------- | ----------------- |
| Sequential | 5,090.4    | 6.056             |
| Continuous | 24,492.2   | 8.167             |

Table 4: Launch throughput and Nsight Systems timeline span for sequential and continuous modes.

We also observe that user CPU time drops. Table 2 shows that user time decreases from 34.68 s in sequential mode to 17.64 s in continuous mode without prefill splitting, and further to 16.47 s for the best prefill-chunk configuration (256 tokens). Host-side overhead per token becomes smaller when we run fewer separate calls and handle multiple sequences inside one batch. The Nsight Systems statistics in Tables 3 and 4 show that mean launch gaps shrink from 170,192 ns to 29,575 ns (about a 5.8× reduction), while high-percentile gaps and launch throughput improve substantially under continuous batching.

### 5.3 Effect of Prefill Splitting

While prefill splitting at 128 tokens per chunk drops the continuous batching model’s performance by approx. 24%, when we increase the prefill splitting to 256 tokens per chunk gives an additional improvement of 32% over the continuous mode without prefill splitting. This indicates that the prefill split performance varies with the nature of the input (in particular the number of tokens in the inputs).

Nsight Systems runtime API statistics for the two continuous (with and without prefill splitting) configurations support this picture. When we run continuous mode with prefill effectively disabled (the prefill chunk parameter set to be larger than all prompt lengths), we observe about 200k launches with a mean launch gap of 29.6 µs, a P90 gap of 4.9 µs, a P95 gap of 6.5 µs, and a launch rate of 24.5k launches per second over an 8.17 s span. With prefill splitting at 128 tokens, the launch count stays almost identical but the launch rate rises to 39.3k launches per second while the Nsight timeline span shrinks to 5.09 s. The mean launch gap drops by roughly a factor of three (from 29.6 µs to around 10 µs), while the P90 and P95 gaps increase modestly to 7.8 µs and 10.8 µs, respectively. These statistics are summarized in Tables 8 and 9 and visualised in Fig. 5.

The launch-gap histograms in Fig. 4 show that both continuous variants spend most of their time with sub-10 µs gaps, but prefill splitting yields a denser cluster of small gaps together with a slightly heavier tail. The sampled timelines in Fig. 6 indicate that the configuration without prefill splitting exhibits fewer but larger spikes (up to tens of milliseconds), whereas the prefill-split configuration has more frequent mid-sized spikes but a much lower average gap and a shorter overall runtime.

Chunked prefills therefore let us overlap prefill work for long prompts with the decode work of other requests without changing any of the backend kernels. Long prompts no longer monopolize the GPU for the duration of a full prefill; instead, the scheduler injects 128-token chunks from multiple prefilling requests into the same `llama_decode` calls and interleaves them with decode tokens. At this model size and hardware scale, most of the gains from prefill splitting come from smoothing out short idle periods and increasing the effective launch rate, while the modest increase in high-percentile launch gaps remains small in absolute terms.

### 5.4 GPU Metrics and Kernel Considerations

The GPU information block in the profile shows that the RTX 4070 runs at about 100% utilization for this workload while using around half of the available memory. We therefore treat the current llama.cpp kernels as good enough for our target in this stage of the project.

Nsight Compute confirms that the `mul_mat_vec_q` kernel dominates the decode path. Table 6 summarizes SM throughput, DRAM throughput, and active time for the largest-grid configuration, while Table 7 shows the relative contribution of `mul_mat_vec_q` and other kernels such as RMSNorm, RoPE, and quantization. These plots suggest that the kernel runs at moderate fractions of peak SM and DRAM bandwidth, so further improvements are more likely to come from kernel fusion, arithmetic-intensity changes, or launch-shape tuning than from pure memory-bandwidth optimizations.

### 5.5 Limitations

Our evaluation has several limitations.

We use a single model (Qwen2.5-0.5B-Instruct) and a single GPU (RTX 4070). We do not study larger models, multi-GPU setups, or mixed CPU/GPU offload configurations. Short-prompt workloads and mixed prompt lengths may show different profiles. We do not measure per-request latency distributions and only report aggregate wall-clock time and derived throughput.

We only explore a small set of prefill chunk sizes (128, 256, and 512 tokens) and treat the best-performing configuration (256 tokens) as our default in this study. Other chunk sizes might better balance scheduling overhead and overlap, especially for different prompt-length distributions or hardware. We leave a more systematic tuning of this parameter for future work.

## 6 Conclusions

We built CoBaLI, a small and transparent continuous batching engine on top of llama.cpp, with the goal of improving LLM inference efficiency on commodity GPUs without modifying any of the backend kernels. CoBaLI manages request state, KV cache slots, and scheduling entirely on the host side, and relies on a lightweight CUDA kernel only for selecting active requests. All model computation continues to run inside the standard `llama_decode` path.

Our experiments on Qwen2.5-0.5B-Instruct running on a single RTX 4070 show that scheduling alone can provide substantial gains. Continuous batching removes the long stalls created by sequential serving and increases throughput by keeping more sequences active in every decode step. Prefill splitting adds another opportunity for overlap by breaking long prefills into smaller chunks and mixing them with both decode tokens and shorter prompts. Across our long-prompt workload, we find that a moderate chunk size of 256 tokens provides the best balance between batching efficiency and scheduling overhead.

The profiling results from Nsight Systems and Nsight Compute confirm that these improvements come from better utilisation of existing kernels rather than changes to kernel internals. Continuous batching and prefill splitting densify kernel launch patterns and smooth out idle periods, while the dominant kernels in llama.cpp maintain essentially identical performance characteristics across all modes.

CoBaLI therefore demonstrates that meaningful speedups on small GPUs do not require rewriting attention or matmul kernels. A lightweight scheduler, explicit KV cache management, and careful batching can already deliver strong performance improvements on top of an unchanged backend. In future work we plan to extend these ideas with per-request latency measurements, adaptive chunk sizing, and experiments on larger models and multi-GPU setups. We also aim to explore whether additional gains are possible from deeper integration with llama.cpp once the scheduling layer reaches its limits.

## Figures

(a) Sequential mode  
(b) Continuous mode  

Figure 1: CUDA kernel launch gap histograms (gaps capped at 200 µs) extracted from Nsight Systems for sequential and continuous batching.

Figure 2: Summary of mean, P90, and P95 CUDA launch gaps (in µs) for sequential vs. continuous mode, derived from Nsight Systems runtime API traces.

(a) Sequential mode  
(b) Continuous mode  

Figure 3: Sampled timeline of CUDA launch gaps over the full run for sequential and continuous modes, showing how continuous batching densifies the submission pattern.

(a) Continuous without prefill splitting  
(b) Continuous with prefill splitting (128-token)  

Figure 4: CUDA launch gap histograms for continuous mode with and without prefill splitting (gaps capped at 200 µs).

Figure 5: Mean, P90, and P95 launch gaps (in µs) for continuous mode without prefill splitting and with 128-token prefill splitting, derived from Nsight Systems runtime API traces.

(a) Continuous without prefill splitting  
(b) Continuous with prefill splitting (128-token)  

Figure 6: Sampled timelines of CUDA launch gaps (in µs) for continuous mode with and without prefill splitting, showing how chunked prefills densify the submission pattern and shorten the overall runtime.

## Additional Tables

| Mode        | Grid        | Block      | Samp. | SM (%) | DRAM (%) | Time (µs) |
| ----------- | ----------- | ---------- | ----- | ------ | -------- | --------- |
| Continuous  | 128, 1, 1   | 32, 4, 1   | 7     | 8.21   | 6.57     | 23.69     |
| Continuous  | 4 864, 1, 1 | 32, 4, 1   | 3     | 52.03  | 57.79    | 69.31     |
| Continuous  | 896, 1, 1   | 32, 4, 1   | 10    | 37.30  | 31.17    | 80.32     |
| Sequential  | 128, 1, 1   | 32, 4, 1   | 7     | 8.31   | 6.66     | 23.53     |
| Sequential  | 4 864, 1, 1 | 32, 4, 1   | 3     | 51.09  | 58.81    | 70.62     |
| Sequential  | 896, 1, 1   | 32, 4, 1   | 10    | 37.60  | 32.38    | 79.70     |

Table 5: Nsight Compute metrics for `mul_mat_vec_q` aggregated by grid and block configuration under continuous and sequential modes.

Figure 2: Summary of mean, P90, and P95 CUDA launch gaps (in µs) for sequential vs. continuous mode, derived from Nsight Systems runtime API traces.

| Mode       | Grid        | Block    | Samples | SM (% peak) | DRAM (% peak) | Time (µs) |
| ---------- | ----------- | -------- | ------- | ----------- | ------------- | --------- |
| Continuous | 4 864, 1, 1 | 32, 4, 1 | 3       | 52.03       | 57.79         | 69.31     |
| Sequential | 4 864, 1, 1 | 32, 4, 1 | 3       | 51.09       | 58.81         | 70.62     |

Table 6: Key `mul_mat_vec_q` configuration (largest grid) in continuous and sequential modes. These launches dominate the kernel’s total active GPU time.

| Kernel name (truncated)   | Grid      | Block      | Time (µs) | SM (% peak) | DRAM (% peak) |
| ------------------------- | --------- | ---------- | --------- | ----------- | ------------- |
| `void mul_mat_vec_q<...`  | 448, 1, 1 | 32, 2, 1   | 9.63      | 28.73       | 13.43         |
| `void rms_norm_f32<...`   | 8, 1, 1   | 256, 1, 1  | 4.03      | 1.13        | 2.06          |
| `void k_bin_bcast<...`    | 4, 8, 1   | 128, 1, 1  | 2.82      | 1.44        | 2.69          |
| `void rope_neox<...`      | 112, 1, 1 | 1, 256, 1  | 2.59      | 2.18        | 2.66          |
| `quantize_q8_1...`        | 4, 8, 1   | 256, 1, 1  | 2.53      | 3.26        | 2.70          |

Table 7: Top kernels by GPU active time (`gpu__time_active.sum`) from the Nsight Compute scan, showing utilisation metrics.

| Metric             | Continuous, no prefill | Continuous + prefill 128 |
| ------------------ | ---------------------- | ------------------------ |
| Launch count       | 200 018                | 200 022                  |
| Mean gap (ns)      | 29 575.8               | 10 217.5                 |
| Median gap (ns)    | 2 759.0                | 3 779.0                  |
| P75 gap (ns)       | 3 795.0                | 5 489.0                  |
| P90 gap (ns)       | 4 897.0                | 7 830.0                  |
| P95 gap (ns)       | 6 460.0                | 10 840.0                 |
| Max gap (ns)       | 184 487 963            | 218 704 071              |
| Min gap (ns)       | 857                    | 885                      |
| Mean API dur (ns)  | 11 253.7               | 15 245.7                 |
| Median API dur (ns)| 8 240.0                | 10 461.0                 |

Table 8: Launch-gap and CUDA runtime API duration statistics for continuous mode without prefill splitting and with 128-token prefill splitting (Nsight Systems).

| Mode                   | Launches/s | Timeline span (s) |
| ---------------------- | ---------- | ----------------- |
| Continuous, no prefill | 24 492.2   | 8.17              |
| Continuous + prefill 128 | 39 272.5 | 5.09              |

Table 9: Launch throughput and Nsight Systems timeline span for continuous mode without prefill splitting and with 128-token prefill splitting.

## References

[1] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica, “Efficient Memory Management for Large Language Model Serving with PagedAttention,” in Proceedings of the 29th ACM Symposium on Operating Systems Principles (SOSP ’23), 2023.

[2] Y. He, Y. Lu, and G. Alonso, “Deferred Continuous Batching in Resource-Efficient Large Language Model Serving,” in Proceedings of the 4th Workshop on Machine Learning and Systems (EuroMLSys ’24), 2024.

[3] P. Patel, R. Gupta, G. Mishra, Z. Wan, V. Saraph, K. Kumar, A. Gupta, J. Mink, U. Gupta, and A. Shaikh, “Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve,” in Proceedings of the 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI ’24), 2024.

[4] Hugging Face, “Text Generation Inference,” GitHub repository `huggingface/text-generation-inference`, accessed Nov. 2025. Available at https://github.com/huggingface/text-generation-inference.

[5] L. Chen, Z. Ye, Y. Wu, D. Zhuo, L. Ceze, and A. Krishnamurthy, “Punica: Multi-Tenant LoRA Serving,” in Proceedings of the 7th Conference on Machine Learning and Systems (MLSys 2024), 2024.

[6] S. Li, H. Lu, T. Wu, M. Yu, Q. Weng, X. Chen, Y. Shan, B. Yuan, and W. Wang, “CaraServe: CPU-Assisted and Rank-Aware LoRA Serving for Generative LLM Inference,” arXiv preprint arXiv:2401.11240, 2024.

[7] G. Gerganov and the llama.cpp contributors, “llama.cpp: LLM Inference in C/C++,” GitHub repository `ggml-org/llama.cpp`, accessed Nov. 2025. Available at https://github.com/ggml-org/llama.cpp.

[8] A. Yang, S. Yang, B. Zhang, S. Hui, X. Zheng, Y. Li, et al., “Qwen2.5 Technical Report,” arXiv preprint arXiv:2412.15115, 2024.
