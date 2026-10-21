// A tiny stand-in for FUSE (docs/FUSE.md §8): just enough engine to run the integration headless.
//
//   ThreadPool    fuse::JobSystem: N worker threads and the caller, jobs taken in index order from an atomic counter
//   FixedClock    fuse::FrameClock: a fixed step (60 Hz), so that runs are repeatable
//   LookAtCamera  fuse::View: a perspective camera from an eye, a target and a field of view
//   SoftRenderer  fuse::GpuUpload and fuse::RenderHooks: textures in CPU memory, a render thread that draws each frame
//                 the game thread hands it (FUSE's pipeline: frame N drawn while N + 1 is updated), calls the hooks,
//                 sorts the quads back to front and draws them into a linear-light float framebuffer with
//                 premultiplied blending (texels decoded as sRGB, bilinear), over a sky and a ground of its own, then
//                 encodes the frame to sRGB bytes and hands it to a callback (hashing, PNG keyframes, video).
// Nothing allocates per frame. The rasteriser is plain C++ on one thread: it is the stand-in, not a renderer.
#pragma once

#include "../fuse_engine.hpp"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace standin {

using fuse::Vec3;
using Mat4 = std::array<float, 16>;  // column-major

class ThreadPool final : public fuse::JobSystem {
 public:
  explicit ThreadPool(int workers);  // workers besides the calling thread (0: everything on the caller)
  ~ThreadPool() override;
  int concurrency() const noexcept override { return workers_ + 1; }
  void run(int jobs, Job fn, void* context) noexcept override;

 private:
  void work() noexcept;
  void loop() noexcept;
  int workers_;
  std::vector<std::thread> threads_;
  std::atomic<std::uint32_t> generation_{0};
  std::atomic<int> next_{0}, finished_{0};
  std::atomic<bool> quit_{false};
  int jobs_ = 0;
  Job fn_ = nullptr;
  void* context_ = nullptr;
};

class FixedClock final : public fuse::FrameClock {
 public:
  explicit FixedClock(double hz) : dt_(1.0 / hz) {}
  fuse::FrameTime now() const noexcept override { return {frame_, static_cast<double>(frame_) * dt_, frame_ ? dt_ : 0.0}; }
  void advance() noexcept { ++frame_; }

 private:
  double dt_;
  std::uint64_t frame_ = 0;
};

class LookAtCamera final : public fuse::View {
 public:
  LookAtCamera(int width, int height, float fov_y_radians) : w_(width), h_(height), fov_(fov_y_radians) {}
  void look_at(Vec3 eye, Vec3 target);
  int width() const noexcept override { return w_; }
  int height() const noexcept override { return h_; }
  fuse::ScreenPoint world_to_screen(const Vec3& p) const noexcept override;
  fuse::CameraPose pose() const noexcept override { return pose_; }
  void matrices(float world_to_view[16], float view_to_clip[16]) const noexcept override;
  const Mat4& view_proj() const noexcept { return vp_; }

 private:
  int w_, h_;
  float fov_;
  fuse::CameraPose pose_;
  Mat4 view_{}, proj_{}, vp_{};
};

class SoftRenderer final : public fuse::GpuUpload, public fuse::RenderHooks {
 public:
  // fn(context, frame, rgb, width, height) on the render thread after each frame is drawn: rgb is width x height x 3
  // sRGB bytes, valid during the call.
  using FrameDone = void (*)(void* context, std::uint64_t frame, const std::uint8_t* rgb, int width, int height) noexcept;

  SoftRenderer(int width, int height, FrameDone done, void* context);
  ~SoftRenderer() override;

  // fuse::GpuUpload
  fuse::Texture create_texture(const fuse::TextureDesc& desc) override;
  void destroy_texture(fuse::Texture texture) noexcept override;
  void update(fuse::Texture texture, const fuse::TextureRegion& region, const void* pixels, std::size_t row_bytes) noexcept override;
  // fuse::RenderHooks
  bool add(Hook fn, void* context) override;
  void remove(Hook fn, void* context) noexcept override;

  // The game thread hands frame `frame` (0, 1, 2, ...) over with its camera and a caption for the HUD; returns at once
  // unless the render thread is two frames behind. finish() waits until every frame handed over is drawn.
  void submit(std::uint64_t frame, const LookAtCamera& camera, const char* caption);
  void finish();
  std::size_t texture_bytes() const noexcept { return texture_bytes_; }
  int live_textures() const noexcept;  // textures created and not destroyed
  int hooks() noexcept;                // hooks registered
  std::uint64_t quads_drawn() const noexcept { return quads_drawn_.load(std::memory_order_relaxed); }

 private:
  struct Texture {
    fuse::TextureDesc desc;
    std::vector<std::uint8_t> data;
    bool live = false;
  };
  struct Sink final : fuse::DrawSink {
    SoftRenderer* r = nullptr;
    void draw(const fuse::Quad& quad) noexcept override;
  };
  struct FrameIn {
    Mat4 vp{};
    fuse::CameraPose pose;
    char caption[320] = {};
  };
  struct Sorted {
    float depth;
    int index;
  };

  void loop() noexcept;
  void draw(std::uint64_t frame) noexcept;
  void sky_and_ground(const FrameIn& in) noexcept;
  void rasterise(const fuse::Quad& q, const Mat4& vp) noexcept;
  void encode() noexcept;

  int w_, h_;
  FrameDone done_;
  void* done_ctx_;
  std::vector<Texture> textures_;
  std::size_t texture_bytes_ = 0;
  std::mutex hooks_mu_;
  std::array<std::pair<Hook, void*>, 4> hooks_{};
  Sink sink_;
  std::vector<fuse::Quad> quads_;
  std::vector<Sorted> order_;
  int n_quads_ = 0;
  std::vector<float> fb_;           // linear rgb
  std::vector<std::uint8_t> out_;   // sRGB bytes
  std::array<float, 256> decode_{};       // sRGB byte -> linear
  std::array<std::uint8_t, 4096> encode_{};  // linear (12 bits) -> sRGB byte
  std::array<FrameIn, 4> in_{};
  std::mutex queue_mu_;  // the frames handed over and drawn
  std::condition_variable queue_cv_;
  std::uint64_t submitted_ = 0, drawn_ = 0;
  std::atomic<std::uint64_t> quads_drawn_{0};
  std::atomic<bool> quit_{false};
  std::thread thread_;
};

// PNG (RGB, 8 bits, zlib); false when it cannot be written.
bool write_png(const std::string& path, const std::uint8_t* rgb, int width, int height);

// 64-bit FNV-1a.
std::uint64_t fnv1a(std::span<const std::uint8_t> bytes, std::uint64_t h = 1469598103934665603ull) noexcept;

}  // namespace standin
