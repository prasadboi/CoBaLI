cmake_minimum_required(VERSION 3.18)
project(CoBaLI LANGUAGES CXX CUDA)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CUDA setup
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -march=native")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

# RTX 2080 Ti has compute capability 7.5 (Turing)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Options
option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/third_party/llama.cpp)
include_directories(${CMAKE_SOURCE_DIR}/third_party/llama.cpp/include)
include_directories(${CMAKE_SOURCE_DIR}/third_party/llama.cpp/common)
include_directories(${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/include)

# llama.cpp integration
# Note: You need to build llama.cpp first and link against it
set(LLAMA_CPP_DIR ${CMAKE_SOURCE_DIR}/third_party/llama.cpp)
if(NOT EXISTS ${LLAMA_CPP_DIR})
    message(WARNING "llama.cpp not found at ${LLAMA_CPP_DIR}")
    message(WARNING "Please run: git submodule update --init --recursive")
endif()

# Add llama.cpp library path
link_directories(${LLAMA_CPP_DIR}/build)

# Source files
set(COBALI_SOURCES
    # Common
    src/common/utils.cpp
    
    # Scheduler
    src/scheduler/request_queue.cpp
    src/scheduler/batch_manager.cpp
    src/scheduler/continuous_batcher.cpp
    src/scheduler/prefill_splitter.cpp
    
    # Memory
    src/memory/kv_cache_manager.cpp
    
    # Baseline
    src/baseline/sequential_engine.cpp
    
    # Engine
    src/engine/executor.cpp
    src/engine/cobali_engine.cpp
    
    # CUDA kernels (Phase 4 - custom kernels)
    # src/kernels/attention.cu
    # src/kernels/prefill.cu
    # src/kernels/decode.cu
    # src/kernels/kernel_utils.cu
)

# Create core library
add_library(cobali SHARED ${COBALI_SOURCES})
target_link_libraries(cobali 
    CUDA::cudart
    CUDA::cublas
    llama
)

# Standalone C++ executable
add_executable(cobali_main src/main.cpp)
target_link_libraries(cobali_main cobali)

# Examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples/cpp)
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/cpp)
endif()

# Python bindings
if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 QUIET)
    if(pybind11_FOUND)
        add_subdirectory(python)
    else()
        message(WARNING "pybind11 not found, skipping Python bindings")
    endif()
endif()

# Benchmarks
if(BUILD_BENCHMARKS)
    # Benchmarks are in Python, but we might add C++ micro-benchmarks later
endif()

# Install targets
install(TARGETS cobali cobali_main
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)

# Print configuration
message(STATUS "CoBaLI Configuration:")
message(STATUS "  CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
message(STATUS "  CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")
message(STATUS "  CUDA_VERSION: ${CUDAToolkit_VERSION}")
message(STATUS "  CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  BUILD_TESTS: ${BUILD_TESTS}")
message(STATUS "  BUILD_EXAMPLES: ${BUILD_EXAMPLES}")
message(STATUS "  BUILD_PYTHON_BINDINGS: ${BUILD_PYTHON_BINDINGS}")

