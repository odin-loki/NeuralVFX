#include "headless.hpp"

#include <neuralfx/nvfx.h>
#include <neuralfx/nvfx_library.h>
#include <neuralfx/nvfx_volume.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>

namespace nfx::viewer::headless {
namespace {

int ceil8(float v) { return static_cast<int>(std::ceil(v / 8.f)) * 8; }

struct Shot {
  int w = 320, h = 240;
  float ground = 210.f, x = 160.f, y = 210.f;
};

Shot shot_for(const nvfx_library_effect& e) {
  Shot s;
  s.w = ceil8(std::max(320.f, e.width * 1.12f + 24.f));
  s.h = ceil8(std::max(240.f, e.height * 1.16f + 24.f));
  if (static_cast<float>(s.w) < 4.f / 3.f * static_cast<float>(s.h)) s.w = ceil8(4.f / 3.f * static_cast<float>(s.h));
  else s.h = ceil8(0.75f * static_cast<float>(s.w));
  s.ground = std::round(0.9f * static_cast<float>(s.h));
  s.x = e.anchor == NVFX_ANCHOR_LEFT ? std::round(0.5f * (static_cast<float>(s.w) - e.width)) : 0.5f * static_cast<float>(s.w);
  s.y = e.anchor == NVFX_ANCHOR_BASE ? s.ground : std::round(s.ground - 0.5f * e.height - 0.05f * static_cast<float>(s.h));
  return s;
}

}  // namespace

LumaStats luma_stats(std::span<const std::uint8_t> rgba, int width, int height, int x0, int y0, int x1, int y1) {
  x0 = std::clamp(x0, 0, width);
  x1 = std::clamp(x1, 0, width);
  y0 = std::clamp(y0, 0, height);
  y1 = std::clamp(y1, 0, height);
  double sum = 0, sum2 = 0;
  long n = 0;
  for (int y = y0; y < y1; ++y) {
    for (int x = x0; x < x1; ++x) {
      const std::size_t i = (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)) * 4;
      const double l = 0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2];
      sum += l;
      sum2 += l * l;
      ++n;
    }
  }
  LumaStats out;
  if (n == 0) return out;
  out.mean = sum / static_cast<double>(n);
  out.spread = std::sqrt(std::max(0.0, sum2 / static_cast<double>(n) - out.mean * out.mean));
  return out;
}

bool picture_ok(const LumaStats& s, double min_mean, double min_spread) { return s.spread >= min_spread && s.mean >= min_mean; }

bool library_play(const std::string& catalogue_dir, const std::string& models_dir, const char* entry_name, int frames,
                  std::vector<std::uint8_t>& rgba, int& width, int& height, int threads, std::string* error) {
  if (error) error->clear();
#ifdef _WIN32
  nvfx_profile prof{};
  if (nvfx_profile_init(&prof, NVFX_PROFILE_LOCKSTEP_AVX2) == NVFX_OK) nvfx_set_profile(&prof);
#endif
  nvfx_library_desc ld;
  nvfx_library_desc_init(&ld);
  ld.catalogue_dir = catalogue_dir.c_str();
  ld.models_dir = models_dir.empty() ? nullptr : models_dir.c_str();
  nvfx_library* lib = nullptr;
  nvfx_scene_error err{};
  if (nvfx_library_open(&ld, &lib, &err) != NVFX_OK) {
    if (error) *error = err.message[0] ? err.message : "library open failed";
    return false;
  }
  const int idx = nvfx_library_find(lib, entry_name);
  if (idx < 0) {
    nvfx_library_free(lib);
    if (error) *error = std::string("no catalogue entry: ") + entry_name;
    return false;
  }
  nvfx_library_effect e{};
  nvfx_library_get_effect(lib, idx, &e);
  const Shot shot = shot_for(e);
  width = shot.w;
  height = shot.h;
  nvfx_stage_room room{e.name, 1, 0};
  nvfx_stage_desc sd;
  nvfx_stage_desc_init(&sd);
  sd.width = width;
  sd.height = height;
  sd.ground = shot.ground;
  sd.rooms = &room;
  sd.n_rooms = 1;
  sd.threads = threads;
  nvfx_stage* stage = nullptr;
  if (nvfx_stage_create(lib, &sd, &stage, &err) != NVFX_OK) {
    nvfx_library_free(lib);
    if (error) *error = err.message[0] ? err.message : "stage create failed";
    return false;
  }
  nvfx_scene* scene = nvfx_stage_scene(stage);
  nvfx_fx fx = 0;
  nvfx_stage_spawn(stage, e.name, shot.x, shot.y, 1.f, 0.2f, &fx);
  rgba.assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4, 0);
  const float dt = 1.f / 60.f;
  float t = 0.f;
  for (int f = 0; f < frames; ++f) {
    if (e.move_x != 0.f || e.move_y != 0.f) {
      const float T = std::max(e.show, 0.5f), half = 0.5f * T;
      float u = std::fmod(std::max(t, 0.f), 2.f * T);
      if (u > T) u = 2.f * T - u;
      nvfx_stage_move(stage, fx, shot.x + e.move_x * (u - half), shot.y + e.move_y * (u - half));
    }
    int computed = 0;
    nvfx_scene_step(scene, dt, &computed);
    if (computed > 0) nvfx_scene_render(scene, rgba.data(), static_cast<std::size_t>(width) * 4);
    t += dt;
  }
  nvfx_stage_free(stage);
  nvfx_library_free(lib);
  return true;
}

bool volume_play(const std::string& model_path, int frames, std::vector<std::uint8_t>& rgba, int& width, int& height, int threads,
                 std::string* error) {
  if (error) error->clear();
  nvfx_volume_effect* effect = nullptr;
  if (nvfx_volume_effect_load(model_path.c_str(), &effect) != NVFX_OK) {
    if (error) *error = "cannot load " + model_path;
    return false;
  }
  nvfx_volume* vol = nullptr;
  if (nvfx_volume_create(effect, threads, &vol) != NVFX_OK) {
    nvfx_volume_effect_free(effect);
    if (error) *error = "volume create failed";
    return false;
  }
  const float controls[3] = {0.75f, 0.5f, 0.55f};
  nvfx_volume_set_controls(vol, controls, 3);
  nvfx_volume_set_seed(vol, 9'960'001);
  width = 512;
  height = width * 9 / 16;
  rgba.assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4, 0);
  const float fps = 30.f;
  for (int f = 0; f < frames; ++f) {
    const float t = static_cast<float>(f) / fps;
    const float az = 0.6f + 0.15f * static_cast<float>(f) / static_cast<float>(std::max(1, frames));
    const float dist = 2.2f;
    const float cy = 0.62f;
    const float ex = dist * std::cos(0.35f) * std::sin(az);
    const float ey = dist * std::sin(0.35f) + cy;
    const float ez = dist * std::cos(0.35f) * std::cos(az);
    nvfx_volume_look_at la{{ex, ey, ez}, {0.f, cy, 0.f}, {0.f, 1.f, 0.f}, 42.f, static_cast<float>(width) / static_cast<float>(height), 0.f, 0.f};
    nvfx_volume_camera cam{};
    nvfx_volume_camera_look_at(&la, &cam);
    nvfx_volume_step_to(vol, static_cast<double>(t));
    nvfx_volume_render(vol, &cam, width, height, rgba.data(), static_cast<std::size_t>(width) * 4);
  }
  nvfx_volume_free(vol);
  nvfx_volume_effect_free(effect);
  return true;
}

}  // namespace nfx::viewer::headless
