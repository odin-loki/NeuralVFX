// Library mode for nvfx_viewer (docs/VIEWER_PLAN.md §3.3): browse library/*.nvfxl and play entries live.
#pragma once

#include "headless.hpp"
#include "texture.hpp"

#include <neuralfx/nvfx_library.h>

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace nfx::viewer {

class LibraryUi {
 public:
  struct Options {
    std::string catalogue_dir;
    std::string models_dir;
    int threads = 2;
  };
  explicit LibraryUi(Options o);
  ~LibraryUi();

  void frame(float x, float y, float w, float h, double dt);

  bool preview_luma(headless::LumaStats& out) const;
  int stage_width() const { return stage_w_; }
  int stage_height() const { return stage_h_; }
  bool select_entry(const char* name);

 private:
  void open_library();
  void select_effect(int index);
  void rebuild_stage();
  void advance(double dt);
  void refresh_missing_models();

  Options opt_;
  nvfx_library* lib_ = nullptr;
  nvfx_stage* stage_ = nullptr;
  nvfx_scene* scene_ = nullptr;
  int stage_w_ = 320, stage_h_ = 240;
  float ground_ = 210.f;
  nvfx_fx fx_ = 0;
  float spawn_x_ = 160.f, spawn_y_ = 210.f;

  int selected_ = -1;
  std::string filter_;
  int category_ = 0;  // 0 = all
  std::vector<std::string> categories_;

  double time_ = 0.0;
  bool paused_ = false;
  float speed_ = 1.f;
  bool loop_ = true;
  std::vector<float> params_;

  std::vector<std::uint8_t> frame_;
  Texture tex_;
  std::deque<double> frame_ms_;
  std::string error_;
  std::string load_error_;
  std::string models_field_;
  std::vector<std::string> missing_models_;
};

}  // namespace nfx::viewer
