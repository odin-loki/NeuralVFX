// Paths for a portable viewer install (exe + data/ + library/ next to each other).
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace nfx::viewer {

// Directory containing nvfx_viewer.exe (Windows) or the current binary's parent when discoverable.
std::filesystem::path bundle_root();

// Ordered data roots: NEURALVFX_DATA, bundle_root()/data, ~/nvfx-data, %USERPROFILE%/nvfx-data.
std::vector<std::filesystem::path> data_roots();

std::string default_effects_dir();
std::string default_library_dir();
std::string default_library_models();
std::string default_volume_model();
std::filesystem::path shipped_models_dir();

}  // namespace nfx::viewer
