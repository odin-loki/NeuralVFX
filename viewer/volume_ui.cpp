#include "volume_ui.hpp"

#include <neuralfx/nvfx_volume.h>

#include <imgui.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <numeric>
#include <filesystem>
#include <numbers>
#include <print>

namespace nfx::viewer {
namespace {

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

std::filesystem::path source_root() {
#ifdef NEURALFX_SOURCE_DIR
  return NEURALFX_SOURCE_DIR;
#else
  return std::filesystem::current_path();
#endif
}

std::filesystem::path data_root() {
  if (const char* d = std::getenv("NEURALVFX_DATA"); d && *d) return d;
  if (const char* h = std::getenv("HOME"); h && *h) return std::filesystem::path(h) / "nvfx-data";
  if (const char* u = std::getenv("USERPROFILE"); u && *u) return std::filesystem::path(u) / "nvfx-data";
  return {};
}

void add_if_exists(std::vector<VolumeUi::ModelPath>& out, const std::string& label, const std::filesystem::path& p) {
  std::error_code ec;
  if (p.empty() || !std::filesystem::is_regular_file(p, ec)) return;
  out.push_back({label, p.string()});
}

}  // namespace

VolumeUi::VolumeUi(Options o) : threads_(std::max(1, o.threads)) {
  discover_models();
  if (!o.initial_model.empty()) {
    for (std::size_t i = 0; i < models_.size(); ++i) {
      if (models_[i].path == o.initial_model) {
        model_index_ = static_cast<int>(i);
        break;
      }
    }
    if (model_index_ == 0 && models_.empty()) models_.push_back({"custom", o.initial_model});
  }
  if (!models_.empty()) load_model(models_[static_cast<std::size_t>(model_index_)].path);
}

VolumeUi::~VolumeUi() {
  if (vol_.volume) nvfx_volume_free(static_cast<nvfx_volume*>(vol_.volume));
  if (vol_.effect) nvfx_volume_effect_free(static_cast<nvfx_volume_effect*>(vol_.effect));
}

void VolumeUi::discover_models() {
  const auto root = source_root();
  add_if_exists(models_, "fire (N2)", root / "models/3d/fire.nvfx3");
  add_if_exists(models_, "fire hres (C2)", root / "models/3d/fire_hres.nvfx3");
  add_if_exists(models_, "smoke", root / "models/3d/smoke.nvfx3");
  add_if_exists(models_, "smoke hres", root / "models/3d/smoke_hres.nvfx3");
  const auto data = data_root();
  if (!data.empty()) {
    add_if_exists(models_, "hres fire (data)", data / "hres_3d/fire/models/hres_fire.nvfx3");
    add_if_exists(models_, "hres smoke (data)", data / "hres_3d/smoke/models/smoke_hres.nvfx3");
  }
}

bool VolumeUi::load_model(const std::string& path) {
  error_.clear();
  if (vol_.volume) {
    nvfx_volume_free(static_cast<nvfx_volume*>(vol_.volume));
    vol_.volume = nullptr;
  }
  if (vol_.effect) {
    nvfx_volume_effect_free(static_cast<nvfx_volume_effect*>(vol_.effect));
    vol_.effect = nullptr;
  }
  nvfx_volume_effect* e = nullptr;
  if (nvfx_volume_effect_load(path.c_str(), &e) != NVFX_OK) {
    error_ = "cannot load " + path;
    return false;
  }
  nvfx_volume* v = nullptr;
  if (nvfx_volume_create(e, threads_, &v) != NVFX_OK) {
    nvfx_volume_effect_free(e);
    error_ = "cannot create volume";
    return false;
  }
  vol_.effect = e;
  vol_.volume = v;
  nvfx_volume_info info{};
  nvfx_volume_effect_get_info(e, &info);
  vol_.fps = info.fps > 0.f ? info.fps : 30.f;
  vol_.n_controls = std::max(1, std::min(3, info.n_controls));
  std::memcpy(vol_.name, info.name, sizeof(vol_.name));
  nvfx_volume_set_controls(v, controls_, vol_.n_controls);
  nvfx_volume_set_seed(v, static_cast<std::uint64_t>(seed_));
  nvfx_volume_set_shard(v, shard_);
  time_ = 0.0;
  dirty_controls_ = true;
  nvfx_volume_step_to(v, 0.0);
  step_picture();
  return true;
}

void VolumeUi::step_picture() {
  if (!vol_.volume) return;
  auto* v = static_cast<nvfx_volume*>(vol_.volume);
  if (dirty_controls_) {
    nvfx_volume_set_controls(v, controls_, vol_.n_controls);
    nvfx_volume_set_seed(v, static_cast<std::uint64_t>(seed_));
    nvfx_volume_set_shard(v, shard_);
    dirty_controls_ = false;
  }
  const auto t0 = Clock::now();
  nvfx_volume_step_to(v, time_);
  step_ms_.push_back(ms_since(t0));
  if (step_ms_.size() > 60) step_ms_.pop_front();

  const float aspect = static_cast<float>(picture_w_) / static_cast<float>(std::max(1, picture_w_ * 9 / 16));
  const float cy = 0.62f;
  const float ex = orbit_dist_ * std::cos(orbit_pitch_) * std::sin(orbit_yaw_);
  const float ey = orbit_dist_ * std::sin(orbit_pitch_) + cy;
  const float ez = orbit_dist_ * std::cos(orbit_pitch_) * std::cos(orbit_yaw_);
  nvfx_volume_look_at la{{ex, ey, ez}, {0.f, cy, 0.f}, {0.f, 1.f, 0.f}, 42.f, aspect, 0.f, 0.f};
  nvfx_volume_camera cam{};
  nvfx_volume_camera_look_at(&la, &cam);

  const int h = picture_w_ * 9 / 16;
  rgba_.assign(static_cast<std::size_t>(picture_w_) * static_cast<std::size_t>(h) * 4, 0);
  const auto t1 = Clock::now();
  nvfx_volume_render(v, &cam, picture_w_, h, rgba_.data(), static_cast<std::size_t>(picture_w_) * 4);
  draw_ms_.push_back(ms_since(t1));
  if (draw_ms_.size() > 60) draw_ms_.pop_front();
  tex_.upload(rgba_, picture_w_, h);
}

void VolumeUi::frame(float x, float y, float w, float h, double dt, bool mouse_captured) {
  if (!paused_) {
    time_ += dt * speed_;
    if (loop_ && shard_ > 0.f && time_ > static_cast<double>(shard_)) time_ = std::fmod(time_, static_cast<double>(shard_));
  }
  step_picture();

  ImGui::SetNextWindowPos({x, y}, ImGuiCond_Always);
  ImGui::SetNextWindowSize({w, h}, ImGuiCond_Always);
  ImGui::Begin("Volume", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);

  if (!error_.empty()) ImGui::TextColored({1.f, 0.4f, 0.4f, 1.f}, "%s", error_.c_str());

  if (!models_.empty()) {
    const char* preview = models_[static_cast<std::size_t>(model_index_)].label.c_str();
    if (ImGui::BeginCombo("model", preview)) {
      for (int i = 0; i < static_cast<int>(models_.size()); ++i) {
        if (ImGui::Selectable(models_[static_cast<std::size_t>(i)].label.c_str(), i == model_index_)) {
          model_index_ = i;
          load_model(models_[static_cast<std::size_t>(i)].path);
        }
      }
      ImGui::EndCombo();
    }
  }

  ImGui::Text("%s", vol_.name[0] ? vol_.name : "no model");
  ImGui::SeparatorText("playback");
  ImGui::Checkbox("paused", &paused_);
  ImGui::SameLine();
  if (ImGui::Button("restart")) time_ = 0.0;
  ImGui::SliderFloat("speed", &speed_, 0.f, 3.f);
  ImGui::Checkbox("loop shard", &loop_);
  ImGui::SliderFloat("time (s)", reinterpret_cast<float*>(&time_), 0.f, std::max(8.f, shard_));
  ImGui::SliderFloat("shard (s)", &shard_, 0.f, 30.f);
  if (ImGui::IsItemDeactivatedAfterEdit()) dirty_controls_ = true;

  ImGui::SeparatorText("learned controls");
  for (int i = 0; i < vol_.n_controls; ++i) {
    const char* name = vol_.effect ? nvfx_volume_effect_control_name(static_cast<nvfx_volume_effect*>(vol_.effect), i) : "control";
    if (ImGui::SliderFloat(name && *name ? name : "control", &controls_[i], 0.f, 1.f)) dirty_controls_ = true;
  }
  ImGui::InputInt("seed", &seed_);
  if (ImGui::IsItemDeactivatedAfterEdit()) dirty_controls_ = true;
  ImGui::SliderInt("threads", &threads_, 1, 16);
  if (ImGui::IsItemDeactivatedAfterEdit() && !models_.empty()) load_model(models_[static_cast<std::size_t>(model_index_)].path);

  ImGui::SeparatorText("camera");
  if (ImGui::IsWindowHovered() && ImGui::IsMouseDragging(ImGuiMouseButton_Left) && !mouse_captured) {
    const ImVec2 d = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
    orbit_yaw_ += d.x * 0.01f;
    orbit_pitch_ = std::clamp(orbit_pitch_ + d.y * 0.01f, -0.2f, 1.2f);
    ImGui::ResetMouseDragDelta(ImGuiMouseButton_Left);
  }
  if (ImGui::IsWindowHovered() && !mouse_captured) orbit_dist_ = std::clamp(orbit_dist_ - ImGui::GetIO().MouseWheel * 0.15f, 1.2f, 5.f);
  ImGui::SliderFloat("yaw", &orbit_yaw_, -3.14f, 3.14f);
  ImGui::SliderFloat("pitch", &orbit_pitch_, -0.2f, 1.2f);
  ImGui::SliderFloat("distance", &orbit_dist_, 1.2f, 5.f);
  ImGui::SliderInt("picture width", &picture_w_, 256, 1024);
  if (picture_w_ % 16 != 0) picture_w_ = (picture_w_ / 16) * 16;

  const double step_mean = step_ms_.empty() ? 0.0 : std::accumulate(step_ms_.begin(), step_ms_.end(), 0.0) / static_cast<double>(step_ms_.size());
  const double draw_mean = draw_ms_.empty() ? 0.0 : std::accumulate(draw_ms_.begin(), draw_ms_.end(), 0.0) / static_cast<double>(draw_ms_.size());
  ImGui::SeparatorText("cost");
  ImGui::Text("step %.2f ms  render %.2f ms  (%dx%d)", step_mean, draw_mean, picture_w_, picture_w_ * 9 / 16);

  if (tex_.id()) {
    ImGui::SeparatorText("picture");
    const float max_w = w - 24.f;
    const float pic_h = static_cast<float>(picture_w_ * 9 / 16);
    const float scale = std::min(1.f, max_w / static_cast<float>(picture_w_));
    ImGui::Image(tex_.id(), {static_cast<float>(picture_w_) * scale, pic_h * scale});
  }
  ImGui::End();
}

}  // namespace nfx::viewer
