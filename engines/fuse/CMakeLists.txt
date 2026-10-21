# The sample integration for FUSE (docs/FUSE.md): the adapter (nvfx_fuse.hpp, a static library FUSE would compile into
# itself), the headless demo with a stand-in FUSE (demo/), and its test. Linked to libnvfx (the shared library, whose
# version the adapter checks against the headers) when it is built, else to the static libraries.
add_library(nvfx_fuse STATIC nvfx_fuse.cpp)
target_include_directories(nvfx_fuse PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
if(NEURALFX_BUILD_SHARED)
  target_link_libraries(nvfx_fuse PUBLIC nvfx_shared)
else()
  target_link_libraries(nvfx_fuse PUBLIC nvfx_pack_api)
endif()
neuralfx_defaults(nvfx_fuse)

add_executable(nvfx_fuse_demo demo/fuse_demo.cpp demo/standin.cpp)
target_link_libraries(nvfx_fuse_demo PRIVATE nvfx_fuse ZLIB::ZLIB Threads::Threads)
target_include_directories(nvfx_fuse_demo PRIVATE "${PROJECT_SOURCE_DIR}/tools")  # label_font.hpp for the HUD
neuralfx_defaults(nvfx_fuse_demo)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(nvfx_fuse_demo PRIVATE -Wno-mismatched-new-delete)  # it counts allocations (replaces operator new)
endif()

# The tests. With the data: a pack of the effect library's models, the campfire script and study N2's 3D fire (all
# outside git: the pack is not made, and both tests skip, without them), then the demo's --check: the whole sequence
# twice (on three workers and on the game thread alone), no allocation in any frame, the same frames to the bit.
if(NEURALFX_BUILD_TESTS AND NEURALFX_BUILD_TOOLS)
  if(DEFINED ENV{NEURALVFX_DATA})
    set(_fuse_data "$ENV{NEURALVFX_DATA}")
  else()
    set(_fuse_data "$ENV{HOME}/nvfx-data")
  endif()
  set(NEURALFX_FUSE_VOLUME "${_fuse_data}/n2_3d/models/chosen.nvfx3" CACHE FILEPATH "The 3D fire the FUSE demo plays (study N2's C4, docs/STUDY_3D.md §14)")
  set(NEURALFX_FUSE_MODELS "${_fuse_data}/library/models" CACHE PATH "The effect library's models for the FUSE demo's pack")
  # the pack names its members by file name: the 3D fire goes in as fire3d.nvfx3
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/pack_files")
  file(CREATE_LINK "${NEURALFX_FUSE_VOLUME}" "${CMAKE_CURRENT_BINARY_DIR}/pack_files/fire3d.nvfx3" RESULT _fuse_link SYMBOLIC)
  set(_fuse_pack "${CMAKE_CURRENT_BINARY_DIR}/fuse_demo.nvfxp")
  add_test(NAME neuralfx.FuseDemoPack
           COMMAND nvfx_pack pack ${_fuse_pack} --library ${PROJECT_SOURCE_DIR}/library --models ${NEURALFX_FUSE_MODELS}
                   --add ${PROJECT_SOURCE_DIR}/examples/scenes/campfire.nvfxs,${CMAKE_CURRENT_BINARY_DIR}/pack_files/fire3d.nvfx3 --require-models --threads 2)
  set_tests_properties(neuralfx.FuseDemoPack PROPERTIES LABELS "neuralfx;data" SKIP_RETURN_CODE 77 TIMEOUT 600 FIXTURES_SETUP fuse_pack)
  add_test(NAME neuralfx.FuseDemoHeadless COMMAND nvfx_fuse_demo --pack ${_fuse_pack} --check --workers 3)
  set_tests_properties(neuralfx.FuseDemoHeadless PROPERTIES LABELS "neuralfx;data" SKIP_RETURN_CODE 77 TIMEOUT 1800 FIXTURES_REQUIRED fuse_pack)
  # Without any learned model (CI): a pack of demo/selftest's procedural entries, and the adapter's self-test on it (its
  # load errors, and a stage played on the stand-in: no allocation per frame, the same frames on 2 workers and on 0).
  set(_fuse_selftest_pack "${CMAKE_CURRENT_BINARY_DIR}/fuse_selftest.nvfxp")
  add_test(NAME neuralfx.FuseSelfTestPack COMMAND nvfx_pack pack ${_fuse_selftest_pack} --library ${CMAKE_CURRENT_SOURCE_DIR}/demo/selftest)
  set_tests_properties(neuralfx.FuseSelfTestPack PROPERTIES LABELS neuralfx FIXTURES_SETUP fuse_selftest_pack)
  add_test(NAME neuralfx.FuseSelfTest COMMAND nvfx_fuse_demo --self-test ${_fuse_selftest_pack})
  set_tests_properties(neuralfx.FuseSelfTest PROPERTIES LABELS neuralfx TIMEOUT 600 FIXTURES_REQUIRED fuse_selftest_pack)
endif()
