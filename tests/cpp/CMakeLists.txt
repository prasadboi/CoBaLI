cmake_minimum_required(VERSION 3.18)

# Find Google Test
find_package(GTest QUIET)

if(NOT GTest_FOUND)
    # Download and build Google Test
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
    )
    FetchContent_MakeAvailable(googletest)
endif()

# Test executables
add_executable(test_request_queue test_request_queue.cpp)
target_link_libraries(test_request_queue cobali GTest::gtest_main)

add_executable(test_batch_manager test_batch_manager.cpp)
target_link_libraries(test_batch_manager cobali GTest::gtest_main)

add_executable(test_prefill_splitter test_prefill_splitter.cpp)
target_link_libraries(test_prefill_splitter cobali GTest::gtest_main)

add_executable(test_kv_cache test_kv_cache.cpp)
target_link_libraries(test_kv_cache cobali GTest::gtest_main)

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(test_request_queue)
gtest_discover_tests(test_batch_manager)
gtest_discover_tests(test_prefill_splitter)
gtest_discover_tests(test_kv_cache)

