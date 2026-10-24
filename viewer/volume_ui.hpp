// Volume mode for nvfx_viewer (docs/VIEWER_PLAN.md §3.4): play .nvfx3 learned 3D effects live.
#pragma once

#include "texture.hpp"

#include <chrono>
#include <deque>
#include <string>
#include <vector>

namespace nfx::viewer {

class VolumeUi {
 public:
  struct Options {
    std::string initial_model;  // empty: first discovered model
    int threads = 4;
  };
  explicit VolumeUi(Options o);

  void frame(float x, float y, float w, float h, double dt, bool mouse_captured);

  ~VolumeUi();

  struct ModelPath {
    std::string label;
    std::string path;
  };

 private:
  void discover_models();
  bool load_model(const std::string& path);
  void step_picture();
  std::vector<ModelPath> models_;
  int model_index_ = 0;

  struct Vol {
    void* effect = nullptr;  // nvfx_volume_effect*
    void* volume = nullptr;    // nvfx_volume*
    float fps = 30.f;
    int n_controls = 3;
    char name[32]{};
  } vol_{};

  float controls_[3]{0.75f, 0.5f, 0.55f};
  int seed_ = 9'960'001;
  float shard_ = 8.f;
  double time_ = 0.0;
  float speed_ = 1.f;
  bool paused_ = false;
  bool loop_ = true;
  float orbit_yaw_ = 0.6f;
  float orbit_pitch_ = 0.35f;
  float orbit_dist_ = 2.2f;
  int picture_w_ = 512;
  int threads_ = 4;

  std::vector<std::uint8_t> rgba_;
  Texture tex_;
  std::deque<double> step_ms_;
  std::deque<double> draw_ms_;
  std::string error_;
  bool dirty_controls_ = true;

  using Clock = std::chrono::steady_clock;
  Clock::time_point last_step_{};
};

}  // namespace nfx::viewer
