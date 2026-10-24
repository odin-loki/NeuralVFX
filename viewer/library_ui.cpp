#include "library_ui.hpp"

#include <imgui.h>
#include <misc/cpp/imgui_stdlib.h>

#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <numeric>

namespace nfx::viewer {
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

std::array<float, 2> path(const nvfx_library_effect& e, float x, float y, float s) {
  const float T = std::max(e.show, 0.5f), half = 0.5f * T;
  float u = std::fmod(std::max(s, 0.f), 2.f * T);
  if (u > T) u = 2.f * T - u;
  return {x + e.move_x * (u - half), y + e.move_y * (u - half)};
}

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

LibraryUi::LibraryUi(Options o) : opt_(std::move(o)), models_field_(opt_.models_dir) { open_library(); }

void LibraryUi::refresh_missing_models() {
  missing_models_.clear();
  if (!lib_ || selected_ < 0 || opt_.models_dir.empty()) return;
  struct Ctx {
    std::filesystem::path dir;
    std::vector<std::string>* miss;
  } ctx{opt_.models_dir, &missing_models_};
  nvfx_library_list_models(
      lib_, selected_,
      [](void* u, const char*, const char* f) {
        auto* c = static_cast<Ctx*>(u);
        const std::filesystem::path p = std::filesystem::path(f).is_absolute() ? std::filesystem::path(f) : c->dir / f;
        std::error_code ec;
        if (!std::filesystem::exists(p, ec)) c->miss->push_back(f);
      },
      &ctx);
}

LibraryUi::~LibraryUi() {
  if (stage_) nvfx_stage_free(stage_);
  if (lib_) nvfx_library_free(lib_);
}

void LibraryUi::open_library() {
  load_error_.clear();
  if (lib_) {
    nvfx_library_free(lib_);
    lib_ = nullptr;
  }
  if (stage_) {
    nvfx_stage_free(stage_);
    stage_ = nullptr;
    scene_ = nullptr;
  }
  nvfx_library_desc d;
  nvfx_library_desc_init(&d);
  d.catalogue_dir = opt_.catalogue_dir.c_str();
  d.models_dir = opt_.models_dir.empty() ? nullptr : opt_.models_dir.c_str();
  nvfx_scene_error err{};
  if (nvfx_library_open(&d, &lib_, &err) != NVFX_OK) {
    load_error_ = err.message[0] ? err.message : "library open failed";
    return;
  }
  categories_.clear();
  categories_.push_back("all");
  for (int c = 0; c < nvfx_library_category_count(lib_); ++c) {
    const char* name = nvfx_library_category(lib_, c);
    if (name) categories_.push_back(name);
  }
  if (nvfx_library_count(lib_) > 0) select_effect(0);
}

bool LibraryUi::select_entry(const char* name) {
  if (!lib_ || !name) return false;
  const int idx = nvfx_library_find(lib_, name);
  if (idx < 0) return false;
  select_effect(idx);
  return true;
}

void LibraryUi::select_effect(int index) {
  if (!lib_ || index < 0 || index >= nvfx_library_count(lib_)) return;
  selected_ = index;
  nvfx_library_effect e{};
  nvfx_library_get_effect(lib_, index, &e);
  params_.resize(static_cast<std::size_t>(e.n_params));
  for (int i = 0; i < e.n_params; ++i) {
    nvfx_library_param p{};
    nvfx_library_get_param(lib_, index, i, &p);
    params_[static_cast<std::size_t>(i)] = p.value;
  }
  time_ = 0.0;
  refresh_missing_models();
  stage_dirty_ = true;
}

void LibraryUi::rebuild_stage() {
  error_.clear();
  if (stage_) {
    nvfx_stage_free(stage_);
    stage_ = nullptr;
    scene_ = nullptr;
    fx_ = 0;
  }
  if (!lib_ || selected_ < 0) return;
  refresh_missing_models();
  if (!missing_models_.empty()) {
    error_ = "missing in " + opt_.models_dir + ": " + missing_models_.front();
    if (missing_models_.size() > 1) error_ += " (+" + std::to_string(missing_models_.size() - 1) + " more)";
    return;
  }
  nvfx_library_effect e{};
  nvfx_library_get_effect(lib_, selected_, &e);
  const Shot shot = shot_for(e);
  stage_w_ = shot.w;
  stage_h_ = shot.h;
  ground_ = shot.ground;
  spawn_x_ = shot.x;
  spawn_y_ = shot.y;
  nvfx_stage_room room{e.name, 1, 0};
  nvfx_stage_desc d;
  nvfx_stage_desc_init(&d);
  d.width = stage_w_;
  d.height = stage_h_;
  d.ground = ground_;
  d.rooms = &room;
  d.n_rooms = 1;
  d.threads = opt_.threads;
  nvfx_scene_error err{};
  if (nvfx_stage_create(lib_, &d, &stage_, &err) != NVFX_OK) {
    error_ = err.message[0] ? err.message : "stage create failed";
    return;
  }
  scene_ = nvfx_stage_scene(stage_);
  frame_.assign(static_cast<std::size_t>(stage_w_) * static_cast<std::size_t>(stage_h_) * 4, 0);
  nvfx_stage_spawn(stage_, e.name, spawn_x_, spawn_y_, 1.f, 0.2, &fx_);
  for (int i = 0; i < e.n_params; ++i) {
    nvfx_library_param p{};
    nvfx_library_get_param(lib_, selected_, i, &p);
    nvfx_stage_set_param(stage_, fx_, p.name, params_[static_cast<std::size_t>(i)]);
  }
}

void LibraryUi::advance(double dt) {
  if (stage_dirty_) {
    rebuild_stage();
    stage_dirty_ = false;
  }
  if (!stage_ || !scene_ || paused_) return;
  const auto t0 = std::chrono::steady_clock::now();
  nvfx_library_effect e{};
  if (lib_ && selected_ >= 0) nvfx_library_get_effect(lib_, selected_, &e);
  if (e.move_x != 0.f || e.move_y != 0.f) {
    const auto [mx, my] = path(e, spawn_x_, spawn_y_, static_cast<float>(time_));
    nvfx_stage_move(stage_, fx_, mx, my);
  }
  int computed = 0;
  nvfx_scene_step(scene_, static_cast<float>(dt), &computed);
  if (computed > 0) nvfx_scene_render(scene_, frame_.data(), static_cast<std::size_t>(stage_w_) * 4);
  frame_ms_.push_back(ms_since(t0));
  if (frame_ms_.size() > 60) frame_ms_.pop_front();
  tex_.upload(frame_, stage_w_, stage_h_);
  time_ += dt * speed_;
  if (!e.looping && e.duration > 0.f && time_ > e.duration + 1.f) {
    if (loop_) {
      time_ = 0.0;
      nvfx_stage_stop(stage_, fx_, -1.0);
      nvfx_stage_spawn(stage_, e.name, spawn_x_, spawn_y_, 1.f, 0.2, &fx_);
    } else {
      paused_ = true;
    }
  }
}

bool LibraryUi::preview_luma(headless::LumaStats& out) const {
  if (frame_.empty() || stage_w_ <= 0 || stage_h_ <= 0) return false;
  out = headless::luma_stats(frame_, stage_w_, stage_h_, 0, 0, stage_w_, stage_h_);
  return true;
}

void LibraryUi::frame(float x, float y, float w, float h, double /*dt*/) {
  ImGui::SetNextWindowPos({x, y}, ImGuiCond_Always);
  ImGui::SetNextWindowSize({w, h}, ImGuiCond_Always);
  ImGui::Begin("Library", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);

  if (!load_error_.empty()) ImGui::TextColored({1.f, 0.4f, 0.4f, 1.f}, "%s", load_error_.c_str());
  if (!error_.empty()) ImGui::TextColored({1.f, 0.5f, 0.3f, 1.f}, "%s", error_.c_str());

  const float list_w = std::min(360.f, w * 0.35f);
  ImGui::BeginChild("list", {list_w, h - 40.f}, true);
  ImGui::InputTextWithHint("filter", "search…", &filter_);
  if (ImGui::BeginCombo("category", categories_[static_cast<std::size_t>(category_)].c_str())) {
    for (int i = 0; i < static_cast<int>(categories_.size()); ++i) {
      if (ImGui::Selectable(categories_[static_cast<std::size_t>(i)].c_str(), category_ == i)) category_ = i;
    }
    ImGui::EndCombo();
  }
  if (lib_) {
    for (int i = 0; i < nvfx_library_count(lib_); ++i) {
      nvfx_library_effect e{};
      nvfx_library_get_effect(lib_, i, &e);
      if (category_ > 0 && e.category != categories_[static_cast<std::size_t>(category_)]) continue;
      if (!filter_.empty()) {
        const std::string hay = std::string(e.name) + " " + e.description;
        if (hay.find(filter_) == std::string::npos) continue;
      }
      const bool sel = i == selected_;
      if (ImGui::Selectable(e.name, sel)) select_effect(i);
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", e.description);
    }
  }
  ImGui::EndChild();
  ImGui::SameLine();

  ImGui::BeginChild("detail", {w - list_w - 16.f, h - 40.f}, true);
  if (lib_ && selected_ >= 0) {
    nvfx_library_effect e{};
    nvfx_library_get_effect(lib_, selected_, &e);
    ImGui::Text("%s  (%s)", e.name, e.category);
    ImGui::TextWrapped("%s", e.description);
    ImGui::Text("models: %s", e.models && *e.models ? e.models : "procedural");
    if (e.elements && *e.elements) ImGui::Text("elements: %s", e.elements);
    ImGui::Text("%.0fx%.0f  anchor %d  %s  show %.1fs", e.width, e.height, static_cast<int>(e.anchor), e.looping ? "loop" : "one-shot", e.show);
    if (e.move_x != 0.f || e.move_y != 0.f) ImGui::Text("moving %.0f, %.0f px/s", e.move_x, e.move_y);

    ImGui::SeparatorText("parameters");
    for (int i = 0; i < e.n_params; ++i) {
      nvfx_library_param p{};
      nvfx_library_get_param(lib_, selected_, i, &p);
      if (stage_ && ImGui::SliderFloat(p.name, &params_[static_cast<std::size_t>(i)], p.min, p.max))
        nvfx_stage_set_param(stage_, fx_, p.name, params_[static_cast<std::size_t>(i)]);
      if (p.description && *p.description && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", p.description);
    }

    ImGui::SeparatorText("playback");
    ImGui::Checkbox("paused", &paused_);
    ImGui::SameLine();
    if (stage_ && ImGui::Button("respawn")) {
      time_ = 0.0;
      nvfx_stage_stop(stage_, fx_, -1.0);
      nvfx_stage_spawn(stage_, e.name, spawn_x_, spawn_y_, 1.f, 0.2, &fx_);
    }
    ImGui::SliderFloat("speed", &speed_, 0.f, 3.f);
    ImGui::Checkbox("loop one-shot", &loop_);
    ImGui::Text("time %.2f s", time_);
    const double mean = frame_ms_.empty() ? 0.0 : std::accumulate(frame_ms_.begin(), frame_ms_.end(), 0.0) / static_cast<double>(frame_ms_.size());
    ImGui::Text("stage %.2f ms/frame (%d threads)", mean, opt_.threads);

    if (tex_.id()) {
      ImGui::SeparatorText("preview");
      const float max_w = w - list_w - 40.f;
      const float scale = std::min(1.f, max_w / static_cast<float>(stage_w_));
      ImGui::Image(tex_.id(), {static_cast<float>(stage_w_) * scale, static_cast<float>(stage_h_) * scale});
    }
  }
  ImGui::EndChild();

  ImGui::SeparatorText("paths");
  ImGui::TextDisabled("catalogue %s", opt_.catalogue_dir.c_str());
  ImGui::InputTextWithHint("models dir", "$NEURALVFX_DATA/library/models", &models_field_);
  if (ImGui::Button("apply models path")) {
    opt_.models_dir = models_field_;
    if (selected_ >= 0) select_effect(selected_);
    else open_library();
  }
  ImGui::SameLine();
  if (ImGui::Button("reload catalogue")) open_library();
  if (opt_.models_dir.empty()) {
    ImGui::TextColored({1.f, 0.7f, 0.4f, 1.f}, "No models folder: run  models/restore.sh  from the repo (see docs/LIBRARY.md).");
  } else if (!missing_models_.empty() && selected_ >= 0) {
    ImGui::TextColored({1.f, 0.5f, 0.3f, 1.f}, "This entry needs: %s", missing_models_.front().c_str());
  }

  ImGui::End();
}

}  // namespace nfx::viewer
