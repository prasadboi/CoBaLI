# CoBaLI
Continuous Batching for LLM inference

# Instructions to run seq mode:
1. in project root directory run the following:
   1. `cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DLLAMA_CUDA=ON -DLLAMA_NATIVE=OFF`
   2. `cmake --build build -j`
2. now change directory to scripts
   1. `cd scripts`
3. run command:
   1. `../build/cobali_runner --model ../models/qwen2.5-0.5b-instruct-q5_k_m.gguf --mode seq --max-slots 1 --prompts ../workloads/prompts_short.txt../workloads/prompts_short.txt`
