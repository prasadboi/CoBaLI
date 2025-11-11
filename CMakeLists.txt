cmake_minimum_required(VERSION 3.20)
project(cobali LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

option(COBALI_BUILD_SERVER "Build optional SSE server" OFF)

# Pull llama.cpp (expects 'llama' CMake target) :contentReference[oaicite:11]{index=11}
add_subdirectory(external/llama.cpp)

# CUDA selector lib
add_library(cobali_batcher STATIC
  src/cuda/batcher.cu
  src/cuda/batcher.cuh)
target_include_directories(cobali_batcher PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda)

# Runner
add_executable(cobali_runner
  src/host/engine.cpp
  src/host/main.cpp
  src/host/engine.hpp
  include/cobali/api.hpp)

target_include_directories(cobali_runner PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/external/llama.cpp) # for llama.h

target_link_libraries(cobali_runner PRIVATE cobali_batcher llama)

# If you later wire the SSE server, add the file and link a lightweight HTTP lib.
