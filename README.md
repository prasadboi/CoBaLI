# CoBaLI
Continuous Batching and Prefill Splitting for Effecient LLM inference

# Modules to load in cuda5
1. `module load cuda-12.4`
2. `module load python-12.1`
3. `pip install pandas`
4. `pip install matplotlib`

# Install llama.cpp
1. Create a folder called `external` in the root directory
2. cd into `external`
3. run `git clone https://github.com/ggml-org/llama.cpp.git`

# Install the model
1. Run `mkdir models` from the root
2. Run `pip install -U huggingface_hub`
3. Run `python -c "from huggingface_hub import hf_hub_download; hf_hub_download('Qwen/Qwen2.5-0.5B-Instruct-GGUF', 'qwen2.5-0.5b-instruct-q5_k_m.gguf', local_dir='models', local_dir_use_symlinks=False)"`

# Instructions to run seq mode:
1. in project root directory run the following:
   1. `cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DLLAMA_CUDA=ON -DLLAMA_NATIVE=OFF`
   2. `cmake --build build -j`
2. now change directory to scripts
   1. `cd scripts`
3. run command:
   1. SEQUENTIAL: `../build/cobali_runner --model ../models/qwen2.5-0.5b-instruct-q5_k_m.gguf --mode seq --max-slots 1 --prefill-chunk 512 --prompts ../workloads/prompts_186.txt`

# Continuous batching with prefill splitting (run from within scripts directory)
1. Use the same build steps as above.
2. Run (example):\
  `../build/cobali_runner --model ../models/qwen2.5-0.5b-instruct-q5_k_m.gguf --mode cont --max-slots 16 --prefill-chunk 256 --prompts ../workloads/prompts_186.txt`
- `--prefill-chunk` controls how many prompt tokens per request are ingested per scheduler round, enabling overlap between long-prefill prompts and ongoing decode tokens.

# Profiling with NSight Systems (run from within scripts directory)
1. These are the runs used to compare sequential vs continuous decoding for a batch of prompts:
   1. SEQUENTIAL RUN: `nsys profile --trace=cuda,nvtx --sample=none -o seq_profile ../build/cobali_runner --model ../models/qwen2.5-0.5b-instruct-q5_k_m.gguf --prompts ../workloads/prompts_16.txt --mode seq --max-slots 1 --ctx 16384`
   2. CONTINUOUS MODE (WITHOUT PREFILL CHUNKING): `nsys profile --trace=cuda,nvtx --sample=none -o cont_profile ../build/cobali_runner --model ../models/qwen2.5-0.5b-instruct-q5_k_m.gguf --prompts ../workloads/prompts_16.txt --mode cont --max-slots 4 --ctx 16384 --prefill-chunk 256`
  
2. Export to SQLite (needs to be done for both runs)
   1. `nsys export -t sqlite -o outputfilename  inputfilename.nsys-rep`
  
3. Generate plots (from within profiler directory):
   1. Go to 
   2. `python nsys_profiler.py`

# Profiling with NSight Compute (from within scripts directory)
1. Nsight Compute is used to:
   1. Profile the heaviest attention kernel (mul_mat_vec_q) under different batching modes
   2. Identify the top kernels by GPU active time
2. mul_mat_vec_q under continuous batching:
   1. `ncu --metrics "gpu__time_active.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed" -k regex:"mul_mat_vec_q" -c 50 ../build/cobali_runner --model ../models/qwen2.5-0.5b-instruct-q5_k_m.gguf --prompts ../workloads/prompts_16.txt --mode cont --max-slots 4 --ctx 16384 --prefill-chunk 128`
3. mul_mat_vec_q under sequential mode:
   1. `ncu --metrics "gpu__time_active.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed" -k regex:"mul_mat_vec_q" -c 50 ../build/cobali_runner --model ../models/qwen2.5-0.5b-instruct-q5_k_m.gguf --prompts ../workloads/prompts_16.txt --mode seq --max-slots 1 --ctx 256`
4. Heaviest-kernel scan (full profile)
   1. `ncu --set full -c 5 ../build/cobali_runner --model ../models/qwen2.5-0.5b-instruct-q5_k_m.gguf --prompts ../workloads/prompts_16.txt --mode seq --max-slots 1 --ctx 16384`
