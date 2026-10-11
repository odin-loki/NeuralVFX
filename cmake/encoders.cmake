# Production texture encoders for the flipbook baselines (src/core/block_formats.cpp; docs/DATA.md, third-party code).
# Fetched at configure time at pinned commits, each by a child CMake process running FetchContent (cmake/fetch_one.cmake),
# so a failed download leaves that format out instead of failing the configure: its flipbooks are not built, the tools
# skip them and the tests that need them skip. They are linked into neuralfx_core (study code) only, never into the
# runtime (nvfx), and compiled without this project's warning flags.
#
#   BC7   bc7e, Binomial's BC7 encoder (all eight modes), in its plain C++ port from Basis Universal (Apache-2.0),
#         decoded by the reference decoder of bc7enc_rdo (MIT or public domain)
#   ASTC  Arm's astc-encoder (Apache-2.0), which encodes and decodes; LDR profile, built for SSE4.1 on x86-64
#
# Sets NEURALFX_ENCODER_LIBS (targets to link) and NEURALFX_ENCODER_DEFS (NEURALFX_HAVE_BC7=1, NEURALFX_HAVE_ASTC=1).

set(NEURALFX_ENCODER_LIBS "")
set(NEURALFX_ENCODER_DEFS "")
if(NOT NEURALFX_FETCH_ENCODERS)
  return()
endif()

set(_nvfx_bc7enc_rdo_commit b9438627eef73a1157e84201b6fa6eb2ffd6d9f0)      # 30 July 2026
set(_nvfx_basisu_commit 99f52d63aa6799cbdaecfe977111dc5ec3b31d47)          # 1 September 2026
set(_nvfx_astcenc_commit baff485b0ff36d2f95d28961605106502c653966)         # tag 5.7.0, 31 July 2026

# neuralfx_fetch(<name> GIT <url> TAG <commit> | URL <url> SHA256 <hash>): sets <name>_FOUND and <name>_DIR.
function(neuralfx_fetch name)
  cmake_parse_arguments(F "" "GIT;TAG;URL;SHA256" "" ${ARGN})
  set(dir "${CMAKE_CURRENT_BINARY_DIR}/_encoders/${name}")
  set(stamp "${dir}/fetched.txt")
  set(key "${F_GIT}${F_URL} ${F_TAG}${F_SHA256}")
  set(have "")
  if(EXISTS "${stamp}")
    file(READ "${stamp}" have)
  endif()
  if(NOT have STREQUAL key)
    file(REMOVE_RECURSE "${dir}")
    set(args -DNAME=${name} "-DDIR=${dir}" "-DCMAKE_GENERATOR=${CMAKE_GENERATOR}")
    if(CMAKE_MAKE_PROGRAM)
      list(APPEND args "-DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}")
    endif()
    if(F_URL)
      list(APPEND args "-DURL=${F_URL}" -DSHA256=${F_SHA256})
    else()
      list(APPEND args "-DGIT=${F_GIT}" -DTAG=${F_TAG})
    endif()
    message(STATUS "Fetching ${name} (flipbook baselines)")
    execute_process(COMMAND "${CMAKE_COMMAND}" ${args} -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/fetch_one.cmake"
                    RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE out TIMEOUT 600)
    if(NOT rc EQUAL 0)
      file(WRITE "${dir}-fetch.log" "${out}")
      message(WARNING "Could not fetch ${name} (${F_GIT}${F_URL}); the flipbook baselines that need it are not built "
                      "(details: ${dir}-fetch.log).")
      set(${name}_FOUND FALSE PARENT_SCOPE)
      return()
    endif()
    file(WRITE "${stamp}" "${key}")
  endif()
  set(${name}_FOUND TRUE PARENT_SCOPE)
  set(${name}_DIR "${dir}/src" PARENT_SCOPE)
endfunction()

neuralfx_fetch(bc7enc_rdo GIT https://github.com/richgel999/bc7enc_rdo.git TAG ${_nvfx_bc7enc_rdo_commit})
set(_nvfx_raw https://raw.githubusercontent.com/BinomialLLC/basis_universal/${_nvfx_basisu_commit}/encoder)
neuralfx_fetch(bc7e_cpp URL ${_nvfx_raw}/basisu_bc7e_scalar.cpp SHA256 684b106e6037042bee065694e5956a97ba91cf2b2417c1a44217e0087a5e808f)
neuralfx_fetch(bc7e_h URL ${_nvfx_raw}/basisu_bc7e_scalar.h SHA256 5395f673cbdb9b4ce5115d18369f36b6002712bf88ac3ce71170cdeace889b79)
neuralfx_fetch(astcenc GIT https://github.com/ARM-software/astc-encoder.git TAG ${_nvfx_astcenc_commit})

# Third-party code: its own language level, no warning flags of ours (-w), position-independent like the rest.
function(neuralfx_third_party target)
  set_target_properties(${target} PROPERTIES CXX_STANDARD 17 CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE -w)
  endif()
endfunction()

if(bc7enc_rdo_FOUND AND bc7e_cpp_FOUND AND bc7e_h_FOUND)
  add_library(neuralfx_bc7 STATIC "${bc7e_cpp_DIR}/basisu_bc7e_scalar.cpp" "${bc7enc_rdo_DIR}/bc7decomp.cpp"
                                  "${bc7enc_rdo_DIR}/bc7decomp_ref.cpp")
  target_include_directories(neuralfx_bc7 SYSTEM PUBLIC "${bc7e_h_DIR}" "${bc7enc_rdo_DIR}")
  neuralfx_third_party(neuralfx_bc7)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(neuralfx_bc7 PRIVATE -fno-strict-aliasing)  # as bc7enc_rdo builds itself
  endif()
  list(APPEND NEURALFX_ENCODER_LIBS neuralfx_bc7)
  list(APPEND NEURALFX_ENCODER_DEFS NEURALFX_HAVE_BC7=1)
endif()

if(astcenc_FOUND)
  set(_s "${astcenc_DIR}/Source")
  add_library(neuralfx_astc STATIC
    ${_s}/astcenc_averages_and_directions.cpp ${_s}/astcenc_block_sizes.cpp ${_s}/astcenc_color_quantize.cpp
    ${_s}/astcenc_color_unquantize.cpp ${_s}/astcenc_compress_symbolic.cpp ${_s}/astcenc_compute_variance.cpp
    ${_s}/astcenc_decompress_symbolic.cpp ${_s}/astcenc_diagnostic_trace.cpp ${_s}/astcenc_entry.cpp
    ${_s}/astcenc_find_best_partitioning.cpp ${_s}/astcenc_ideal_endpoints_and_weights.cpp ${_s}/astcenc_image.cpp
    ${_s}/astcenc_integer_sequence.cpp ${_s}/astcenc_mathlib.cpp ${_s}/astcenc_mathlib_softfloat.cpp
    ${_s}/astcenc_partition_tables.cpp ${_s}/astcenc_percentile_tables.cpp ${_s}/astcenc_pick_best_endpoint_format.cpp
    ${_s}/astcenc_quantization.cpp ${_s}/astcenc_symbolic_physical.cpp ${_s}/astcenc_weight_align.cpp
    ${_s}/astcenc_weight_quant_xfer_tables.cpp)
  target_include_directories(neuralfx_astc SYSTEM PUBLIC "${_s}")
  neuralfx_third_party(neuralfx_astc)
  # Its invariant build (no contraction): the same blocks on every x86-64 machine. SSE4.1 where the CPU is x86-64,
  # its portable scalar code elsewhere.
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    target_compile_definitions(neuralfx_astc PRIVATE ASTCENC_NEON=0 ASTCENC_SVE=0 ASTCENC_SSE=41 ASTCENC_AVX=0
                                                     ASTCENC_POPCNT=1 ASTCENC_F16C=0)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
      target_compile_options(neuralfx_astc PRIVATE -msse4.1 -mpopcnt)
    endif()
  else()
    target_compile_definitions(neuralfx_astc PRIVATE ASTCENC_NEON=0 ASTCENC_SVE=0 ASTCENC_SSE=0 ASTCENC_AVX=0
                                                     ASTCENC_POPCNT=0 ASTCENC_F16C=0)
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(neuralfx_astc PRIVATE -ffp-contract=off -fno-math-errno)
  endif()
  list(APPEND NEURALFX_ENCODER_LIBS neuralfx_astc)
  list(APPEND NEURALFX_ENCODER_DEFS NEURALFX_HAVE_ASTC=1)
endif()

if(NEURALFX_ENCODER_DEFS)
  message(STATUS "Flipbook baselines in production formats: ${NEURALFX_ENCODER_DEFS}")
endif()
