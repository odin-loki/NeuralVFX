# The Godot 4 GDExtension (docs/ENGINES.md §8): the NeuralVFXScene and NeuralVFXEffect nodes over libnvfx's C API.
# Optional (NEURALFX_BUILD_GODOT=ON): godot-cpp is fetched at a pinned tag and built with a build profile (only the
# engine classes the nodes use); its binding generator needs Python 3. The extension and libnvfx are copied into the
# demo project's bin/ (NEURALFX_GODOT_BIN), where demo/neuralvfx.gdextension finds them. engines/godot/test.sh builds
# it and runs the demo headless.
include(FetchContent)

if(NOT NEURALFX_BUILD_SHARED)
  message(FATAL_ERROR "NEURALFX_BUILD_GODOT needs NEURALFX_BUILD_SHARED (the extension links libnvfx)")
endif()

# Cache entries, so that godot-cpp's own (set without FORCE) keep them.
set(GODOTCPP_TARGET "template_debug" CACHE STRING "godot-cpp's target (template_debug: the editor and debug exports)")
set(GODOTCPP_BUILD_PROFILE "${CMAKE_CURRENT_SOURCE_DIR}/build_profile.json" CACHE PATH "The engine classes godot-cpp binds")
FetchContent_Declare(godot-cpp
  GIT_REPOSITORY https://github.com/godotengine/godot-cpp.git
  GIT_TAG godot-4.4.1-stable
  GIT_SHALLOW TRUE
  SYSTEM)
FetchContent_MakeAvailable(godot-cpp)

add_library(nvfx_godot SHARED src/register_types.cpp src/nvfx_scene_node.cpp src/nvfx_effect_node.cpp)
target_link_libraries(nvfx_godot PRIVATE godot::cpp nvfx_shared)
# C++17 as godot-cpp; hidden symbols but the entry point; libnvfx found next to the extension.
set_target_properties(nvfx_godot PROPERTIES CXX_STANDARD 17 CXX_EXTENSIONS OFF CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
                                            OUTPUT_NAME neuralvfx_godot BUILD_WITH_INSTALL_RPATH ON INSTALL_RPATH "$ORIGIN")
neuralfx_defaults(nvfx_godot)

set(NEURALFX_GODOT_BIN "${CMAKE_CURRENT_SOURCE_DIR}/demo/bin" CACHE PATH "Where the extension and libnvfx are copied for a Godot project")
add_custom_command(TARGET nvfx_godot POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "${NEURALFX_GODOT_BIN}"
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:nvfx_godot> $<TARGET_FILE:nvfx_shared> "${NEURALFX_GODOT_BIN}"
  COMMENT "Copying the extension and libnvfx to ${NEURALFX_GODOT_BIN}")
