# NeuralVFX: CPU-only neural visual effects for games (docs/PLAN.md).
#   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-14
# No third-party runtime dependency; GoogleTest only for the tests. Can also be added to another build with
# add_subdirectory().
cmake_minimum_required(VERSION 3.25)
project(neuralfx VERSION 0.1.0 DESCRIPTION "CPU-only neural visual effects (C++23)" LANGUAGES CXX)

if(PROJECT_IS_TOP_LEVEL)
  set(CMAKE_CXX_STANDARD 23)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
  if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  endif()
  include(CTest)
endif()

option(NEURALFX_BUILD_TESTS "Build the neuralfx GoogleTest suites" ${BUILD_TESTING})
option(NEURALFX_BUILD_BENCH "Build the neuralfx benchmarks" ON)
option(NEURALFX_WERROR "Treat warnings in neuralfx targets as errors (CI)" OFF)

function(neuralfx_defaults target)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wconversion
                                             $<$<BOOL:${NEURALFX_WERROR}>:-Werror>)
  endif()
endfunction()

# Phase 0 prototypes. One translation unit per ISA; dispatch.cpp picks at run time. ISO C++ mode turns off
# multiply-add contraction, which halves the SIMD throughput of these loops, so it is switched back on here: the
# ISAs then differ in the last bits, which the tests allow for.
add_library(neuralfx_proto STATIC
  src/proto/dispatch.cpp
  src/proto/kernels_base.cpp
  src/proto/kernels_avx2.cpp
  src/proto/kernels_avx512.cpp)
target_include_directories(neuralfx_proto PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
neuralfx_defaults(neuralfx_proto)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(neuralfx_proto PRIVATE -ffp-contract=fast -Wno-psabi)
endif()

if(NEURALFX_BUILD_BENCH)
  add_executable(neuralfx_arch_bench bench/arch_bench.cpp)
  target_link_libraries(neuralfx_arch_bench PRIVATE neuralfx_proto)
  neuralfx_defaults(neuralfx_arch_bench)
endif()

if(NEURALFX_BUILD_TESTS)
  find_package(GTest CONFIG REQUIRED)
  include(GoogleTest)
  add_executable(neuralfx_tests tests/test_proto.cpp)
  target_link_libraries(neuralfx_tests PRIVATE neuralfx_proto GTest::gtest_main)
  neuralfx_defaults(neuralfx_tests)
  gtest_discover_tests(neuralfx_tests TEST_PREFIX "neuralfx." DISCOVERY_MODE PRE_TEST PROPERTIES LABELS neuralfx)
endif()
