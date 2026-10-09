# nvfx_viewer: play neural effects live with sliders (Dear ImGui on GLFW + OpenGL 3). Optional (NEURALFX_BUILD_VIEWER).
include(FetchContent)
find_package(glfw3 3.3 REQUIRED)
find_package(OpenGL REQUIRED)

# Dear ImGui is fetched with git at a pinned tag (no CMake of its own: its sources are compiled here).
FetchContent_Declare(imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG v1.91.8
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(imgui)

add_library(neuralfx_imgui STATIC
  ${imgui_SOURCE_DIR}/imgui.cpp ${imgui_SOURCE_DIR}/imgui_draw.cpp ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp)
target_include_directories(neuralfx_imgui SYSTEM PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_link_libraries(neuralfx_imgui PUBLIC glfw OpenGL::GL)

add_executable(nvfx_viewer main.cpp)
target_link_libraries(nvfx_viewer PRIVATE nvfx neuralfx_io neuralfx_imgui)
neuralfx_defaults(nvfx_viewer)

# A headless screenshot under Xvfb, when available: the viewer starts, renders 30 frames and saves a PNG.
find_program(XVFB_RUN xvfb-run)
if(NEURALFX_BUILD_TESTS AND XVFB_RUN AND NEURALFX_BUILD_TOOLS)
  add_test(NAME neuralfx.ViewerScreenshot
           COMMAND ${XVFB_RUN} -a $<TARGET_FILE:nvfx_viewer> --demo --screenshot ${CMAKE_CURRENT_BINARY_DIR}/viewer_test.png --frames 30)
  set_tests_properties(neuralfx.ViewerScreenshot PROPERTIES LABELS "neuralfx;gui" TIMEOUT 120)
endif()
