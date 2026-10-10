# NeuralVFX: CPU-only neural visual effects for games (docs/PLAN.md).
#   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-14
# No third-party runtime dependency. GoogleTest for the tests, zlib for PNG figures, GLFW and Dear ImGui (fetched)
# for the optional viewer. Can also be added to another build with add_subdirectory().
cmake_minimum_required(VERSION 3.25)
project(neuralfx VERSION 0.1.0 DESCRIPTION "CPU-only neural visual effects (C++23)" LANGUAGES C CXX)

if(PROJECT_IS_TOP_LEVEL)
  set(CMAKE_CXX_STANDARD 23)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  set(CMAKE_C_STANDARD 11)
  set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
  if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  endif()
  include(CTest)
endif()

option(NEURALFX_BUILD_TESTS "Build the neuralfx GoogleTest suites" ${BUILD_TESTING})
option(NEURALFX_BUILD_BENCH "Build the neuralfx benchmarks" ON)
option(NEURALFX_BUILD_TOOLS "Build the command-line tools" ON)
option(NEURALFX_BUILD_SHARED "Build libnvfx as a shared library for engines" ON)
option(NEURALFX_BUILD_VIEWER "Build the Dear ImGui viewer (needs GLFW and OpenGL; fetches Dear ImGui)" OFF)
option(NEURALFX_WERROR "Treat warnings in neuralfx targets as errors (CI)" OFF)

function(neuralfx_defaults target)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wshadow -Wconversion>
                                             $<$<COMPILE_LANGUAGE:C>:-Wall -Wextra -Wpedantic>
                                             $<$<BOOL:${NEURALFX_WERROR}>:-Werror>)
  endif()
endfunction()

# SIMD code: one translation unit per ISA, the ISA switched on with #pragma GCC target after the standard headers.
# GCC contracts a * b + c into one fused multiply-add in C++ (only its ISO C modes turn this off), so any code under
# a "fma" target pragma fuses unless contraction is switched off explicitly. It is asked for explicitly here because
# separate multiplies and adds would halve the throughput of these loops; the ISAs then differ in the last bits, which
# the tests allow for. Code that must be bit-exact across ISAs stays out of these targets.
function(neuralfx_simd target)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${target} PRIVATE -ffp-contract=fast -Wno-psabi)
  endif()
endfunction()

# Phase 0 architecture prototypes and their benchmark.
add_library(neuralfx_proto STATIC
  src/proto/dispatch.cpp src/proto/kernels_base.cpp src/proto/kernels_avx2.cpp src/proto/kernels_avx512.cpp)
target_include_directories(neuralfx_proto PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
neuralfx_defaults(neuralfx_proto)
neuralfx_simd(neuralfx_proto)

# The model description and .nvfx format: shared by the runtime, the trainer and the tools.
add_library(neuralfx_model STATIC src/core/model.cpp src/core/rollout.cpp src/core/cm.cpp)
target_include_directories(neuralfx_model PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
set_target_properties(neuralfx_model PROPERTIES POSITION_INDEPENDENT_CODE ON)
neuralfx_defaults(neuralfx_model)

# The runtime (what ships in a game): C API, per-ISA renderers, no allocation per frame.
set(NVFX_RUNTIME_SOURCES src/runtime/nvfx.cpp src/runtime/rt_base.cpp src/runtime/rt_avx2.cpp src/runtime/rt_avx512.cpp)
add_library(nvfx STATIC ${NVFX_RUNTIME_SOURCES})
target_link_libraries(nvfx PUBLIC neuralfx_model)
set_target_properties(nvfx PROPERTIES POSITION_INDEPENDENT_CODE ON)
neuralfx_defaults(nvfx)
neuralfx_simd(nvfx)
if(NEURALFX_BUILD_SHARED)
  add_library(nvfx_shared SHARED ${NVFX_RUNTIME_SOURCES})
  target_link_libraries(nvfx_shared PRIVATE neuralfx_model)
  target_include_directories(nvfx_shared PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
  target_compile_definitions(nvfx_shared PRIVATE NVFX_SHARED_BUILD)
  set_target_properties(nvfx_shared PROPERTIES OUTPUT_NAME nvfx CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
  neuralfx_defaults(nvfx_shared)
  neuralfx_simd(nvfx_shared)
endif()

# Core: clips, the simulation, metrics and flipbook baselines. Baseline ISA, no contraction: the simulation is
# deterministic and identical across machines with the same compiler.
add_library(neuralfx_core STATIC src/core/clip.cpp src/core/metrics.cpp src/core/flipbook.cpp src/sim/sim.cpp)
target_link_libraries(neuralfx_core PUBLIC neuralfx_model)
neuralfx_defaults(neuralfx_core)

# Trainer: hand-written gradients; its hot loops switch on AVX2 + FMA (train() checks the CPU).
add_library(neuralfx_train STATIC src/train/net.cpp src/train/train.cpp src/train/rollout_train.cpp)
target_link_libraries(neuralfx_train PUBLIC neuralfx_core)
neuralfx_defaults(neuralfx_train)
neuralfx_simd(neuralfx_train)

# Images and videos (PNG via zlib, mp4 via an ffmpeg pipe) for previews, figures and comparisons.
find_package(ZLIB REQUIRED)
add_library(neuralfx_io STATIC src/core/image_io.cpp src/core/ingest.cpp)
target_link_libraries(neuralfx_io PUBLIC neuralfx_core PRIVATE ZLIB::ZLIB)
neuralfx_defaults(neuralfx_io)

# Video codecs as compression baselines (results/compression, study F2): x264, x265, VP9, AV1 through an ffmpeg pipe.
add_library(neuralfx_codec STATIC src/codec/video_codec.cpp)
target_link_libraries(neuralfx_codec PUBLIC neuralfx_core)
neuralfx_defaults(neuralfx_codec)

# Composed effects (docs/COMPOSE.md): rollout effects coupled through their fields, with particles, light, distortion
# and bloom for whole scenes. A prototype over the runtime's internals, not part of the C API.
add_library(neuralfx_compose STATIC src/compose/compose.cpp)
target_include_directories(neuralfx_compose PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src/compose" "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime")
target_link_libraries(neuralfx_compose PUBLIC nvfx)
neuralfx_defaults(neuralfx_compose)

# Diffusion-context mixing (docs/DCM.md): the owner's PAQ-style mixer, its frozen inference copy and its search, ported
# from CameraDetector, with a value-domain mixer for generating and coding effects. The standard library only.
add_library(neuralfx_dcm STATIC src/dcm/mixer.cpp src/dcm/compact.cpp src/dcm/search.cpp src/dcm/roc.cpp src/dcm/kmeans.cpp)
target_include_directories(neuralfx_dcm PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
neuralfx_defaults(neuralfx_dcm)
# DCM-fine (docs/DCM.md G1): the mixer in place of the rollout detail layer's lock, on the reference rollout model.
add_library(neuralfx_dcm_fine STATIC src/dcm/fine.cpp)
target_link_libraries(neuralfx_dcm_fine PUBLIC neuralfx_dcm neuralfx_model)
neuralfx_defaults(neuralfx_dcm_fine)

# The denoiser on coarse states of rollout effects (docs/DCM.md, stage S5): forward and backward passes written by hand,
# AVX2 + FMA kernels (the trainer's requirement).
add_library(neuralfx_ddpm STATIC src/dcm/ddpm.cpp)
target_link_libraries(neuralfx_ddpm PUBLIC neuralfx_dcm neuralfx_train)
neuralfx_defaults(neuralfx_ddpm)
neuralfx_simd(neuralfx_ddpm)

if(NEURALFX_BUILD_TOOLS)
  foreach(tool nvfx_sim nvfx_train nvfx_eval nvfx_experiment nvfx_ingest)
    add_executable(${tool} tools/${tool}.cpp)
    target_link_libraries(${tool} PRIVATE neuralfx_io neuralfx_train nvfx)
    neuralfx_defaults(${tool})
  endforeach()
  target_sources(nvfx_experiment PRIVATE tools/experiment_d.cpp)  # study D: start points and learned dynamics
  target_sources(nvfx_experiment PRIVATE tools/experiment_g.cpp tools/fine_study.cpp)  # study G (docs/DCM.md)
  target_link_libraries(nvfx_experiment PRIVATE neuralfx_dcm neuralfx_dcm_fine neuralfx_ddpm)
  # DCM: the mixer search on synthetic problems, version stamps of released mixers, DCM-fine (G1) and the coarse-state
  # denoiser (G2).
  add_executable(nvfx_dcm tools/nvfx_dcm.cpp tools/fine_study.cpp)
  target_link_libraries(nvfx_dcm PRIVATE neuralfx_dcm neuralfx_dcm_fine neuralfx_ddpm neuralfx_io neuralfx_train)
  neuralfx_defaults(nvfx_dcm)
  # The engine-side example: plain C against the shared library's API.
  add_executable(nvfx_c_host examples/c_host.c)
  target_link_libraries(nvfx_c_host PRIVATE nvfx)
  set_target_properties(nvfx_c_host PROPERTIES LINKER_LANGUAGE CXX)
  neuralfx_defaults(nvfx_c_host)
  # A scripted scene of composed effects (a fireball with smoke, fires, embers), rendered to video and profiled.
  add_executable(nvfx_fireball tools/nvfx_fireball.cpp)
  target_link_libraries(nvfx_fireball PRIVATE neuralfx_compose neuralfx_io)
  neuralfx_defaults(nvfx_fireball)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(nvfx_fireball PRIVATE -Wno-mismatched-new-delete)  # counts allocations (replaces operator new)
  endif()
  # Lossless context-mixing compression of .nvfx files, and its measurement against flipbooks (results/compression).
  add_executable(nvfx_pack tools/nvfx_pack.cpp)
  target_link_libraries(nvfx_pack PRIVATE neuralfx_core ZLIB::ZLIB)
  neuralfx_defaults(nvfx_pack)
  # Study F2: compression pushed further (fewer bits, rate-aware training, smaller grids) against flipbooks and video
  # codecs, memory and disk apart.
  add_executable(nvfx_f2 tools/nvfx_f2.cpp)
  target_link_libraries(nvfx_f2 PRIVATE neuralfx_codec neuralfx_train nvfx)
  neuralfx_defaults(nvfx_f2)
  add_executable(nvfx_scene examples/scene_bench.cpp)
  target_link_libraries(nvfx_scene PRIVATE nvfx)
  neuralfx_defaults(nvfx_scene)
endif()

if(NEURALFX_BUILD_BENCH)
  add_executable(neuralfx_arch_bench bench/arch_bench.cpp)
  target_link_libraries(neuralfx_arch_bench PRIVATE neuralfx_proto)
  neuralfx_defaults(neuralfx_arch_bench)
endif()

if(NEURALFX_BUILD_VIEWER)
  add_subdirectory(viewer)
endif()

if(NEURALFX_BUILD_TESTS)
  find_package(GTest CONFIG REQUIRED)
  include(GoogleTest)
  add_executable(neuralfx_tests tests/test_proto.cpp tests/test_sim.cpp tests/test_baselines.cpp tests/test_train.cpp
                                tests/test_runtime.cpp tests/test_ingest.cpp tests/test_rollout.cpp tests/test_compose.cpp
                                tests/test_dcm.cpp tests/test_cm.cpp tests/test_dcm_fine.cpp tests/test_dcm_ddpm.cpp
                                tests/test_video_codec.cpp)
  target_link_libraries(neuralfx_tests PRIVATE neuralfx_proto neuralfx_io neuralfx_train nvfx neuralfx_compose neuralfx_dcm
                                               neuralfx_dcm_fine neuralfx_ddpm neuralfx_codec GTest::gtest_main)
  neuralfx_defaults(neuralfx_tests)
  gtest_discover_tests(neuralfx_tests TEST_PREFIX "neuralfx." DISCOVERY_MODE PRE_TEST PROPERTIES LABELS neuralfx)
  # Counts heap allocations around nvfx_render (replaces the global operator new, so it is its own program).
  add_executable(nvfx_alloc_test tests/alloc_test.cpp)
  target_link_libraries(nvfx_alloc_test PRIVATE nvfx neuralfx_compose)
  neuralfx_defaults(nvfx_alloc_test)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(nvfx_alloc_test PRIVATE -Wno-mismatched-new-delete)  # replacing operator new is the point
  endif()
  add_test(NAME neuralfx.ZeroAllocationPerFrame COMMAND nvfx_alloc_test)
  set_tests_properties(neuralfx.ZeroAllocationPerFrame PROPERTIES LABELS neuralfx)
  if(NEURALFX_BUILD_TOOLS)
    add_test(NAME neuralfx.CHostExample COMMAND nvfx_c_host --self-test)
    set_tests_properties(neuralfx.CHostExample PROPERTIES LABELS neuralfx)
  endif()
endif()
