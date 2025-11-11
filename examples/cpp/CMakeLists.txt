cmake_minimum_required(VERSION 3.18)

# Example 1: Baseline sequential inference
add_executable(example_baseline baseline_inference.cpp)
target_link_libraries(example_baseline cobali)

# Example 2: Continuous batching (Phase 2)
add_executable(example_continuous_batching continuous_batching.cpp)
target_link_libraries(example_continuous_batching cobali)

# Example 3: Full CoBaLI with prefill splitting (Phase 3)
add_executable(example_full_cobali full_cobali.cpp)
target_link_libraries(example_full_cobali cobali)

# Install examples
install(TARGETS 
    example_baseline 
    example_continuous_batching 
    example_full_cobali
    RUNTIME DESTINATION bin/examples
)

