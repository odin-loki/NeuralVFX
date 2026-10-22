// The public headers as C++ (the C ones and nvfx.hpp), built without exceptions or RTTI (-fno-exceptions -fno-rtti):
// the struct sizes of docs/INTEGRATION.md §8 again, the shared library's version against these headers, and the
// wrapper's success and error paths. With tests/abi_c99.c: nvfx_abi_test (exit status: the number of failed checks).
#include <neuralfx/nvfx.hpp>
#include <neuralfx/nvfx_volume.h>

#include <cstdint>
#include <cstdio>
#include <type_traits>

extern "C" int nvfx_abi_c99(void);

#if UINTPTR_MAX == 0xffffffffffffffffu
static_assert(sizeof(nvfx_effect_info) == 80);
static_assert(sizeof(nvfx_scene_effect) == 16);
static_assert(sizeof(nvfx_scene_input) == 16);
static_assert(sizeof(nvfx_scene_desc) == 88);
static_assert(sizeof(nvfx_scene_error) == 1032);
static_assert(sizeof(nvfx_scene_info) == 88);
static_assert(sizeof(nvfx_scene_fields) == 16);
static_assert(sizeof(nvfx_scene_module_info) == 60);
static_assert(sizeof(nvfx_library_file) == 16);
static_assert(sizeof(nvfx_library_desc) == 32);
static_assert(sizeof(nvfx_library_effect) == 88);
static_assert(sizeof(nvfx_library_param) == 32);
static_assert(sizeof(nvfx_stage_room) == 16);
static_assert(sizeof(nvfx_stage_desc) == 96);
static_assert(sizeof(nvfx_stage_fx_info) == 28);
static_assert(sizeof(nvfx_manager_desc) == 104);
static_assert(sizeof(nvfx_manager_model_desc) == 60);
static_assert(sizeof(nvfx_spawn_desc) == 72);
static_assert(sizeof(nvfx_manager_picture) == 56);
static_assert(sizeof(nvfx_manager_stats) == 184);
static_assert(sizeof(nvfx_pack_desc) == 40);
static_assert(sizeof(nvfx_pack_info) == 48);
static_assert(sizeof(nvfx_pack_member) == 48);
static_assert(sizeof(nvfx_volume_info) == 96);
static_assert(sizeof(nvfx_volume_camera) == 128);
static_assert(sizeof(nvfx_volume_look_at) == 52);
#endif
static_assert(sizeof(nvfx_status) == 4 && sizeof(nvfx_fx) == 4 && sizeof(nvfx_handle) == 4);
// The wrapper's handles are a pointer each, move-only; a Picture is the C struct.
static_assert(sizeof(nvfx::Effect) == sizeof(void*) && sizeof(nvfx::Pack) == sizeof(void*) && sizeof(nvfx::SceneView) == sizeof(void*));
static_assert(!std::is_copy_constructible_v<nvfx::Effect> && std::is_nothrow_move_constructible_v<nvfx::Effect>);
static_assert(!std::is_copy_constructible_v<nvfx::Scene> && std::is_nothrow_move_assignable_v<nvfx::Scene>);
static_assert(!std::is_copy_constructible_v<nvfx::Instance> && std::is_nothrow_move_constructible_v<nvfx::Instance>);
static_assert(std::is_copy_constructible_v<nvfx::SceneView>);
static_assert(sizeof(nvfx::Picture) == sizeof(nvfx_manager_picture));

int main() {
  int failures = nvfx_abi_c99();
  const nvfx::Version v = nvfx::version();
  if (!v.matches_headers() || !v.has(NVFX_FEATURE_SHARED)) {
    std::fprintf(stderr, "abi: the shared library is version %u (features 0x%x), the headers %d\n", static_cast<unsigned>(v.version),
                 static_cast<unsigned>(v.features), NVFX_VERSION);
    ++failures;
  }
  const auto missing = nvfx::Effect::load("/nonexistent/fire.nvfx");
  if (missing || missing.error().status != NVFX_ERROR_IO) ++failures;
  const auto bad = nvfx::Scene::check("scene size 64 x 64\nmodul x\n", "t.nvfxs");
  if (bad || bad.error().status != NVFX_ERROR_SCRIPT || bad.error().line != 2) ++failures;
  if (!nvfx::Scene::check("scene size 64 x 64\n")) ++failures;
  int listed = 0;  // the templates, instantiated without exceptions
  if (!nvfx::Scene::list_effects("effect a = \"a.nvfx\"\n", "t", [&](const char*, const char*) { ++listed; }) || listed != 1) ++failures;
  if (nvfx::Library().list_models(0, [&](const char*, const char*) { ++listed; })) ++failures;
  if (nvfx::Pack().entry_models(0, [&](const char*, const char*) { ++listed; })) ++failures;
  const auto pack = nvfx::Pack::open_memory(std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>("NVFXPACK"), 8));
  if (pack || pack.error().message != "pack: truncated: 8 bytes, shorter than a pack's header") ++failures;
  std::printf("C++ without exceptions: version %u, features 0x%x, %s\n", static_cast<unsigned>(v.version), static_cast<unsigned>(v.features),
              failures ? "FAILED" : "ok");
  return failures;
}
