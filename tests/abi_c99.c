/* The public C headers as C99 (built with -std=c99 -pedantic and every warning an error), and the sizes of the structs
 * an engine allocates or reads on 64-bit platforms, as docs/INTEGRATION.md §8 gives them (a binding in another
 * language, C# or Rust, declares them with these sizes). Checked at compile time; nvfx_abi_c99() also checks at run
 * time that the library is the one these headers describe. With tests/abi_cpp.cpp: nvfx_abi_test. */
#include <neuralfx/nvfx.h>
#include <neuralfx/nvfx_library.h>
#include <neuralfx/nvfx_manager.h>
#include <neuralfx/nvfx_pack.h>
#include <neuralfx/nvfx_scene.h>
#include <neuralfx/nvfx_volume.h>

#include <stdint.h>
#include <stdio.h>

#define NVFX_ABI_CHECK(name, cond) typedef char nvfx_abi_check_##name[(cond) ? 1 : -1]

#if UINTPTR_MAX == 0xffffffffffffffffu
NVFX_ABI_CHECK(effect_info, sizeof(nvfx_effect_info) == 80);
NVFX_ABI_CHECK(scene_effect, sizeof(nvfx_scene_effect) == 16);
NVFX_ABI_CHECK(scene_input, sizeof(nvfx_scene_input) == 16);
NVFX_ABI_CHECK(scene_desc, sizeof(nvfx_scene_desc) == 88);
NVFX_ABI_CHECK(scene_error, sizeof(nvfx_scene_error) == 1032);
NVFX_ABI_CHECK(scene_info, sizeof(nvfx_scene_info) == 88);
NVFX_ABI_CHECK(scene_fields, sizeof(nvfx_scene_fields) == 16);
NVFX_ABI_CHECK(scene_module_info, sizeof(nvfx_scene_module_info) == 60);
NVFX_ABI_CHECK(library_file, sizeof(nvfx_library_file) == 16);
NVFX_ABI_CHECK(library_desc, sizeof(nvfx_library_desc) == 32);
NVFX_ABI_CHECK(library_effect, sizeof(nvfx_library_effect) == 88);
NVFX_ABI_CHECK(library_param, sizeof(nvfx_library_param) == 32);
NVFX_ABI_CHECK(stage_room, sizeof(nvfx_stage_room) == 16);
NVFX_ABI_CHECK(stage_desc, sizeof(nvfx_stage_desc) == 96);
NVFX_ABI_CHECK(stage_fx_info, sizeof(nvfx_stage_fx_info) == 28);
NVFX_ABI_CHECK(manager_desc, sizeof(nvfx_manager_desc) == 104);
NVFX_ABI_CHECK(manager_model_desc, sizeof(nvfx_manager_model_desc) == 60);
NVFX_ABI_CHECK(spawn_desc, sizeof(nvfx_spawn_desc) == 72);
NVFX_ABI_CHECK(manager_picture, sizeof(nvfx_manager_picture) == 56);
NVFX_ABI_CHECK(manager_stats, sizeof(nvfx_manager_stats) == 184);
NVFX_ABI_CHECK(pack_desc, sizeof(nvfx_pack_desc) == 40);
NVFX_ABI_CHECK(pack_info, sizeof(nvfx_pack_info) == 48);
NVFX_ABI_CHECK(pack_member, sizeof(nvfx_pack_member) == 48);
NVFX_ABI_CHECK(volume_info, sizeof(nvfx_volume_info) == 96);
NVFX_ABI_CHECK(volume_camera, sizeof(nvfx_volume_camera) == 128);
NVFX_ABI_CHECK(volume_look_at, sizeof(nvfx_volume_look_at) == 52);
#endif
NVFX_ABI_CHECK(status, sizeof(nvfx_status) == 4);
NVFX_ABI_CHECK(fx, sizeof(nvfx_fx) == 4);
NVFX_ABI_CHECK(handle, sizeof(nvfx_handle) == 4);
NVFX_ABI_CHECK(volume_field, sizeof(nvfx_volume_field) == 4);

int nvfx_abi_c99(void);

/* The number of failed checks. */
int nvfx_abi_c99(void) {
  int failures = 0;
  uint32_t features = 0;
  const uint32_t version = nvfx_version(&features);
  const uint32_t apis = NVFX_FEATURE_SCENE | NVFX_FEATURE_LIBRARY | NVFX_FEATURE_MANAGER | NVFX_FEATURE_PACK | NVFX_FEATURE_VOLUME;
  if (version != NVFX_VERSION) {
    fprintf(stderr, "abi: the library is version %u, the headers %d\n", (unsigned)version, NVFX_VERSION);
    ++failures;
  }
  if ((features & apis) != apis) {
    fprintf(stderr, "abi: the library lacks APIs (features 0x%x)\n", (unsigned)features);
    ++failures;
  }
  printf("C99: version %u, features 0x%x\n", (unsigned)version, (unsigned)features);
  return failures;
}
