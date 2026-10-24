#include "bundle_paths.hpp"

#include <cstdlib>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace nfx::viewer {
namespace {

std::filesystem::path env_path(const char* name) {
  if (const char* v = std::getenv(name); v && *v) return std::filesystem::path(v);
  return {};
}

bool has_library_models(const std::filesystem::path& p) {
  std::error_code ec;
  if (!std::filesystem::is_directory(p, ec)) return false;
  return std::filesystem::exists(p / "fire.nvfx", ec) || std::filesystem::exists(p / "smoke.nvfx", ec);
}

}  // namespace

std::filesystem::path bundle_root() {
  if (auto p = env_path("NEURALVFX_BUNDLE"); !p.empty()) return p;
#ifdef _WIN32
  wchar_t buf[MAX_PATH];
  const DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
  if (n > 0 && n < MAX_PATH) return std::filesystem::path(buf).parent_path();
#endif
  std::error_code ec;
  const auto exe = std::filesystem::read_symlink("/proc/self/exe", ec);
  if (!ec) return exe.parent_path();
  return std::filesystem::current_path();
}

std::vector<std::filesystem::path> data_roots() {
  std::vector<std::filesystem::path> roots;
  if (auto d = env_path("NEURALVFX_DATA"); !d.empty()) roots.push_back(d);
  const auto bundled = bundle_root() / "data";
  std::error_code ec;
  if (std::filesystem::is_directory(bundled, ec)) roots.push_back(bundled);
  if (auto h = env_path("HOME"); !h.empty()) roots.push_back(h / "nvfx-data");
  if (auto u = env_path("USERPROFILE"); !u.empty()) roots.push_back(u / "nvfx-data");
  return roots;
}

std::string default_effects_dir() {
  for (const auto& root : data_roots()) {
    for (const char* sub : {"v2/models", "experiments/models/d"}) {
      std::error_code ec;
      if (std::filesystem::is_directory(root / sub, ec)) return (root / sub).string();
    }
  }
  return "models";
}

std::string default_library_dir() {
  const auto lib = bundle_root() / "library";
  std::error_code ec;
  if (std::filesystem::is_directory(lib, ec)) return lib.string();
#ifdef NEURALFX_SOURCE_DIR
  const std::filesystem::path src = NEURALFX_SOURCE_DIR;
  if (std::filesystem::is_directory(src / "library", ec)) return (src / "library").string();
#endif
  return "library";
}

std::string default_library_models() {
  for (const auto& root : data_roots()) {
    const auto lib = root / "library/models";
    if (has_library_models(lib)) return lib.string();
  }
#ifdef NEURALFX_SOURCE_DIR
  const auto lib = std::filesystem::path(NEURALFX_SOURCE_DIR) / "models" / "library/models";
  if (has_library_models(lib)) return lib.string();
#endif
  return {};
}

std::string default_volume_model() {
  const auto path = shipped_models_dir() / "fire_hres.nvfx3";
  std::error_code ec;
  if (std::filesystem::is_regular_file(path, ec)) return path.string();
  return (shipped_models_dir() / "fire.nvfx3").string();
}

std::filesystem::path shipped_models_dir() {
  const auto shipped = bundle_root() / "models" / "3d";
  std::error_code ec;
  if (std::filesystem::is_directory(shipped, ec)) return shipped;
#ifdef NEURALFX_SOURCE_DIR
  const auto src = std::filesystem::path(NEURALFX_SOURCE_DIR) / "models" / "3d";
  if (std::filesystem::is_directory(src, ec)) return src;
#endif
  return shipped;
}

}  // namespace nfx::viewer
