# nvfx_viewer: play neural effects live with sliders, and edit scene scripts live (Dear ImGui on GLFW + OpenGL 3).
# Optional (NEURALFX_BUILD_VIEWER).
include(FetchContent)
find_package(glfw3 3.3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Threads REQUIRED)

# Dear ImGui is fetched with git at a pinned tag (no CMake of its own: its sources are compiled here).
FetchContent_Declare(imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG v1.91.8
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(imgui)

add_library(neuralfx_imgui STATIC
  ${imgui_SOURCE_DIR}/imgui.cpp ${imgui_SOURCE_DIR}/imgui_draw.cpp ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp ${imgui_SOURCE_DIR}/misc/cpp/imgui_stdlib.cpp)
target_include_directories(neuralfx_imgui SYSTEM PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends ${imgui_SOURCE_DIR}/misc/cpp)
target_link_libraries(neuralfx_imgui PUBLIC glfw OpenGL::GL)

# The scene mode's work apart from drawing (a script's edits debounced, checked and rebuilt on a worker thread, through
# the scene C API): no GUI, so it is tested without a display.
add_library(nvfx_viewer_scene STATIC scene_session.cpp)
target_include_directories(nvfx_viewer_scene PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(nvfx_viewer_scene PUBLIC nvfx_scene_api Threads::Threads)
neuralfx_defaults(nvfx_viewer_scene)

add_executable(nvfx_viewer main.cpp scene_ui.cpp)
target_link_libraries(nvfx_viewer PRIVATE nvfx_viewer_scene nvfx neuralfx_io neuralfx_imgui)
neuralfx_defaults(nvfx_viewer)

if(NEURALFX_BUILD_TESTS)
  # The scene mode's reload logic: debouncing, a broken edit keeping the last good scene, errors with line and column,
  # inputs carried over, seeks and restarts off the frame, no allocation in a frame (untrained stand-in effects, no data
  # needed).
  find_package(GTest CONFIG REQUIRED)
  include(GoogleTest)
  add_executable(nvfx_viewer_tests scene_session_test.cpp)
  target_link_libraries(nvfx_viewer_tests PRIVATE nvfx_viewer_scene GTest::gtest_main)
  neuralfx_defaults(nvfx_viewer_tests)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(nvfx_viewer_tests PRIVATE -Wno-mismatched-new-delete)  # counts allocations (replaces operator new)
  endif()
  target_compile_definitions(nvfx_viewer_tests PRIVATE NEURALFX_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
  gtest_discover_tests(nvfx_viewer_tests TEST_PREFIX "neuralfx." DISCOVERY_MODE PRE_TEST PROPERTIES LABELS neuralfx)
endif()

# Headless screenshots under Xvfb, when available: the viewer starts, renders and saves a PNG.
#   ViewerScreenshot             the effects mode with the untrained demo effect
#   ViewerSceneScreenshot        the scene mode playing examples/scenes/fireball.nvfxs on untrained stand-ins; the scene's
#                                picture on the screenshot must not be flat
#   ViewerSceneBrokenEdit        the same, then an edit that breaks line 86: the error is shown at that line and the last
#                                good scene plays on
#   ViewerSceneScreenshotModels  the fireball with the study D effects, when they are on this machine (no data in CI)
find_program(XVFB_RUN xvfb-run)
if(NEURALFX_BUILD_TESTS AND XVFB_RUN AND NEURALFX_BUILD_TOOLS)
  set(_xvfb ${XVFB_RUN} -a -s "-screen 0 1920x1080x24")
  set(_fireball ${PROJECT_SOURCE_DIR}/examples/scenes/fireball.nvfxs)
  add_test(NAME neuralfx.ViewerScreenshot
           COMMAND ${XVFB_RUN} -a $<TARGET_FILE:nvfx_viewer> --demo --screenshot ${CMAKE_CURRENT_BINARY_DIR}/viewer_test.png --frames 30)
  add_test(NAME neuralfx.ViewerSceneScreenshot
           COMMAND ${_xvfb} $<TARGET_FILE:nvfx_viewer> --scene ${_fireball} --stand-ins --threads 2 --at 1.5
                   --screenshot ${CMAKE_CURRENT_BINARY_DIR}/viewer_scene_test.png --frames 20 --check-picture)
  add_test(NAME neuralfx.ViewerSceneBrokenEdit
           COMMAND ${_xvfb} $<TARGET_FILE:nvfx_viewer> --scene ${_fireball} --stand-ins --threads 2 --at 1.5
                   --edit-replace "stop blast2" --with "stop blast3" --expect-error-line 86
                   --screenshot ${CMAKE_CURRENT_BINARY_DIR}/viewer_scene_error_test.png --frames 20 --check-picture)
  set_tests_properties(neuralfx.ViewerScreenshot neuralfx.ViewerSceneScreenshot neuralfx.ViewerSceneBrokenEdit
                       PROPERTIES LABELS "neuralfx;gui" TIMEOUT 600)
  if(DEFINED ENV{NEURALVFX_DATA})
    set(_viewer_data "$ENV{NEURALVFX_DATA}")
  else()
    set(_viewer_data "$ENV{HOME}/nvfx-data")
  endif()
  if(EXISTS "${_viewer_data}/experiments/models/d/explosion.nvfx")
    add_test(NAME neuralfx.ViewerSceneScreenshotModels
             COMMAND ${_xvfb} $<TARGET_FILE:nvfx_viewer> --scene ${_fireball} --effects ${_viewer_data}/experiments/models/d --threads 2 --at 2.9
                     --paused --screenshot ${CMAKE_CURRENT_BINARY_DIR}/viewer_scene_models.png --frames 5 --check-picture)
    set_tests_properties(neuralfx.ViewerSceneScreenshotModels PROPERTIES LABELS "neuralfx;gui;data" TIMEOUT 1200)
  endif()
endif()
