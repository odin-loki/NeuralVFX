// NeuralVFX in FUSE: the adapter (nvfx_fuse.hpp, docs/FUSE.md).
#include "nvfx_fuse.hpp"

#include <neuralfx/nvfx_volume.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace nvfx_fuse {
namespace {

using Clock = std::chrono::steady_clock;
double ms_between(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

constexpr int kSpriteTexture = 256;  // the manager's largest picture: every sprite texture is this size
constexpr int kMaxJobs = 3 + 16;     // the volume, two cards and up to 16 manager slots

// --- vectors and 4 x 4 matrices (column-major: m[col * 4 + row]) --------------------------------------------------------

Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec3 cross(Vec3 a, Vec3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
Vec3 normalized(Vec3 a) {
  const float l = std::sqrt(dot(a, a));
  return l > 1e-12f ? a * (1.f / l) : Vec3{};
}

using Mat4 = std::array<float, 16>;

Mat4 mul(const Mat4& a, const Mat4& b) {
  Mat4 r{};
  for (int c = 0; c < 4; ++c)
    for (int i = 0; i < 4; ++i) {
      float s = 0.f;
      for (int k = 0; k < 4; ++k) s += a[static_cast<std::size_t>(k * 4 + i)] * b[static_cast<std::size_t>(c * 4 + k)];
      r[static_cast<std::size_t>(c * 4 + i)] = s;
    }
  return r;
}

std::array<float, 4> apply(const Mat4& m, float x, float y, float z, float w) {
  std::array<float, 4> r{};
  for (std::size_t i = 0; i < 4; ++i) r[i] = m[i] * x + m[4 + i] * y + m[8 + i] * z + m[12 + i] * w;
  return r;
}

// The inverse by cofactors (in doubles); false when singular.
bool invert(const Mat4& mf, Mat4& out) {
  double m[16], inv[16];
  for (std::size_t i = 0; i < 16; ++i) m[i] = mf[i];
  inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
  inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
  inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
  inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
  inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
  inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
  inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
  inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
  inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
  inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
  inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
  inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
  inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
  inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
  inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
  inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
  const double det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
  if (!(std::abs(det) > 1e-30)) return false;
  for (std::size_t i = 0; i < 16; ++i) out[i] = static_cast<float>(inv[i] / det);
  return true;
}

nvfx::Error load_error(nvfx_status s, std::string message) { return nvfx::Error{s, std::move(message), 0, 0}; }

// Pixels with their version: the version of the content they hold (0: none yet).
struct Image {
  std::vector<std::uint8_t> bytes;
  int w = 0, h = 0, d = 1;
  std::size_t row = 0;  // bytes per row
  std::uint64_t version = 0;
};

// Copy the content of `src` into `dst` (same capacity) when dst is stale.
std::size_t refresh(Image& dst, const Image& src) {
  if (dst.version == src.version) return 0;
  const std::size_t n = src.row * static_cast<std::size_t>(src.h) * static_cast<std::size_t>(src.d);
  std::memcpy(dst.bytes.data(), src.bytes.data(), n);
  dst.w = src.w;
  dst.h = src.h;
  dst.d = src.d;
  dst.row = src.row;
  dst.version = src.version;
  return n;
}

enum class JobKind : std::uint8_t { volume, card, manager };
struct JobEntry {
  JobKind kind = JobKind::manager;
  int arg = 0;
};

}  // namespace

// --- the state -----------------------------------------------------------------------------------------------------------

struct System::Impl {
  fuse::Engine engine;
  int K = 2;  // packets

  // NeuralVFX: effects before the manager, the library before the stage (destroyed in reverse order).
  std::vector<nvfx::Effect> effects;
  std::vector<std::string> model_names;
  std::vector<int> model_controls;
  nvfx::Manager manager;
  std::vector<int> manager_model;
  nvfx::Scene scene;
  nvfx::Library library;
  nvfx::Stage stage;
  nvfx_volume_effect* vfx = nullptr;
  nvfx_volume* vol = nullptr;

  // --- sprites
  struct SpriteSlot {
    bool live = false;
    std::uint16_t gen = 0;
    nvfx_handle handle = 0;
    int model = 0;
    Vec3 position;
    float size = 1.f;
    bool base = true, auto_despawn = true;
    // the content stream of its own texture
    const std::uint8_t* last_rgba = nullptr;
    int last_size = 0;
    std::size_t last_stride = 0;
    std::uint64_t version = 0;
    // this update
    const std::uint8_t* rgba = nullptr;
    int pic_size = 0;
    std::size_t stride = 0;
    int owner = -1;  // the slot whose texture shows it (itself, or a copy sharing its picture); -1: not drawn
  };
  std::vector<SpriteSlot> sprites;
  std::vector<fuse::Texture> sprite_tex;
  int job_slots = 0;
  float edge_fade = 0.f;
  nvfx_status manager_status[16] = {};

  // --- cards (0: the scene, 1: the stage)
  struct CardState {
    bool on = false;
    nvfx_scene* s = nullptr;
    Card card;
    Vec3 normal;
    float cam_x = 0.f, cam_y = 0.f, ground = 0.f;
    int scene_w = 0, scene_h = 0;
    Image staging;
    fuse::Texture tex = 0;
    double dt = 0.0, ms = 0.0;
    nvfx_status status = NVFX_OK;
  };
  CardState cards[2];

  // --- the volume
  struct VolumeState {
    bool on = false, picture = true, fields = false;
    nvfx_volume_info info{};
    Vec3 position;
    float scale = 1.f, picture_scale = 0.5f;
    int max_picture = 320;
    double t0 = -1.0, t = 0.0;
    int frame_drawn = -1;      // the volume frame the picture (and fields) show
    bool placement_changed = false;
    // this update's work
    bool due = false, draw = false;
    nvfx_volume_camera cam{};
    int pw = 0, ph = 0;
    fuse::Quad pending{};
    // what is shown
    bool has_quad = false;
    fuse::Quad quad{};
    Image picture_img, field_img[2];
    fuse::Texture tex = 0, field_tex[2] = {0, 0};
    double ms = 0.0, step_ms = 0.0, picture_ms = 0.0, fields_ms = 0.0;
    nvfx_status status = NVFX_OK;
  } volume;

  // --- jobs
  JobEntry jobs[kMaxJobs];
  int n_jobs = 0;

  // --- packets between the game thread and the render thread
  struct Packet {
    std::uint64_t frame = 0;
    std::vector<fuse::Quad> quads;
    int n_quads = 0;
    std::vector<Image> sprite_px;
    Image card_px[2], volume_px, field_px[2];
  };
  std::vector<Packet> packets;
  std::uint64_t seq = 0;                       // packets published (game thread)
  std::atomic<std::uint64_t> published{0};     // the same, for the render thread
  std::atomic<std::uint64_t> released{0};      // the newest packet drawn + 1 (render thread)
  // render thread: the versions uploaded, per texture stream (sprites, then the cards, the picture, the fields)
  std::vector<std::uint64_t> uploaded;
  std::atomic<std::uint64_t> uploaded_bytes{0}, frames_drawn{0};

  Stats stats;
  FrameError error;
  bool registered = false;

  ~Impl() {
    if (registered) engine.hooks->remove(&render_hook, this);
    if (engine.gpu) {
      for (const fuse::Texture t : sprite_tex) engine.gpu->destroy_texture(t);
      for (const CardState& c : cards)
        if (c.tex) engine.gpu->destroy_texture(c.tex);
      if (volume.tex) engine.gpu->destroy_texture(volume.tex);
      for (const fuse::Texture t : volume.field_tex)
        if (t) engine.gpu->destroy_texture(t);
    }
    if (vol) nvfx_volume_free(vol);
    if (vfx) nvfx_volume_effect_free(vfx);
  }

  // A frame error, kept in fixed storage.
  void fail(nvfx_status s, const char* fmt, ...) noexcept __attribute__((format(printf, 3, 4))) {
    if (s == NVFX_OK) return;
    ++stats.errors;
    error.status = s;
    error.frame = stats.frame;
    va_list ap;
    va_start(ap, fmt);
    const int n = std::vsnprintf(error.what, sizeof(error.what), fmt, ap);
    va_end(ap);
    if (n >= 0 && static_cast<std::size_t>(n) < sizeof(error.what) - 24)
      std::snprintf(error.what + n, sizeof(error.what) - static_cast<std::size_t>(n), ": %s", nvfx_status_string(s));
  }

  // --- placement on a card
  // A world point to the card's scene world pixels; false when the point is not within half a unit of the card.
  bool to_card(const CardState& c, Vec3 p, float& x, float& y, float max_off_plane) const noexcept {
    const Vec3 d = p - c.card.base;
    if (std::abs(dot(d, c.normal)) > max_off_plane) return false;
    const float k = c.card.units_per_px;
    x = c.cam_x + 0.5f * static_cast<float>(c.scene_w) + dot(d, c.card.right) / k;
    y = c.ground - dot(d, c.card.up) / k;
    return true;
  }

  fuse::Quad card_quad(const CardState& c, const fuse::View& view) const noexcept {
    const float k = c.card.units_per_px;
    fuse::Quad q;
    q.texture = c.tex;
    q.centre = c.card.base + c.card.up * ((c.ground - c.cam_y - 0.5f * static_cast<float>(c.scene_h)) * k);
    q.axis_u = c.card.right * (0.5f * static_cast<float>(c.scene_w) * k);
    q.axis_v = c.card.up * (0.5f * static_cast<float>(c.scene_h) * k);
    q.depth = view.world_to_screen(q.centre).depth;
    return q;
  }

  // --- the jobs (FUSE's workers)
  static void job(void* ctx, int j) noexcept {
    auto& s = *static_cast<Impl*>(ctx);
    const JobEntry e = s.jobs[j];
    switch (e.kind) {
      case JobKind::volume: s.run_volume(); break;
      case JobKind::card: s.run_card(s.cards[e.arg]); break;
      case JobKind::manager: s.manager_status[e.arg] = nvfx_manager_run_jobs(s.manager.get(), e.arg); break;
    }
  }

  void run_volume() noexcept {
    const auto t0 = Clock::now();
    VolumeState& v = volume;
    v.status = nvfx_volume_step_to(vol, v.t);
    const auto t1 = Clock::now();
    v.step_ms = ms_between(t0, t1);
    if (v.status == NVFX_OK && v.draw) {
      v.status = nvfx_volume_render(vol, &v.cam, v.pw, v.ph, v.picture_img.bytes.data(), static_cast<std::size_t>(v.pw) * 4);
      if (v.status == NVFX_OK) {
        v.picture_img.w = v.pw;
        v.picture_img.h = v.ph;
        v.picture_img.row = static_cast<std::size_t>(v.pw) * 4;
        ++v.picture_img.version;
      }
    }
    const auto t2 = Clock::now();
    v.picture_ms = v.draw ? ms_between(t1, t2) : 0.0;
    if (v.status == NVFX_OK && v.fields) {
      const nvfx_volume_field which[2] = {NVFX_VOLUME_TEMPERATURE, NVFX_VOLUME_EXTINCTION};
      for (int i = 0; i < 2 && v.status == NVFX_OK; ++i) {
        Image& f = v.field_img[i];
        v.status = nvfx_volume_read(vol, which[i], NVFX_VOLUME_HALF, f.bytes.data(), f.bytes.size());
        if (v.status == NVFX_OK) ++f.version;
      }
    }
    const auto t3 = Clock::now();
    v.fields_ms = v.fields ? ms_between(t2, t3) : 0.0;
    v.ms = ms_between(t0, t3);
  }

  static void run_card(CardState& c) noexcept {
    const auto t0 = Clock::now();
    int computed = 0;
    c.status = nvfx_scene_step(c.s, c.dt, &computed);
    if (c.status == NVFX_OK && computed > 0) {
      c.status = nvfx_scene_render_layer(c.s, NVFX_LAYER_COLOUR, NVFX_FORMAT_RGBA8, c.staging.bytes.data(), c.staging.row);
      if (c.status == NVFX_OK) ++c.staging.version;
    }
    c.ms = computed > 0 ? ms_between(t0, Clock::now()) : 0.0;
  }

  // --- the volume's camera for this update: its transform folded into the view, the projection cropped to the box's
  // rectangle on screen, and the quad facing the camera that will show the picture.
  bool plan_volume(const fuse::View& view) noexcept {
    VolumeState& v = volume;
    Mat4 world_to_view, view_to_clip;
    view.matrices(world_to_view.data(), view_to_clip.data());
    const float s = v.scale;
    const Mat4 model = {s, 0, 0, 0, 0, s, 0, 0, 0, 0, s, 0, v.position.x, v.position.y, v.position.z, 1};
    const Mat4 vol_view = mul(world_to_view, model);
    const Mat4 vol_clip = mul(view_to_clip, vol_view);
    const float ex = v.info.extent[0], ey = v.info.extent[1], ez = v.info.extent[2];
    float x0 = 1e30f, x1 = -1e30f, y0 = 1e30f, y1 = -1e30f;
    for (int i = 0; i < 8; ++i) {
      const auto c = apply(vol_clip, (i & 1) ? 0.5f * ex : -0.5f * ex, (i & 2) ? ey : 0.f, (i & 4) ? 0.5f * ez : -0.5f * ez, 1.f);
      if (!(c[3] > 1e-4f)) return false;  // the camera is inside or beside the box: no picture (a limit of the quad)
      x0 = std::min(x0, c[0] / c[3]);
      x1 = std::max(x1, c[0] / c[3]);
      y0 = std::min(y0, c[1] / c[3]);
      y1 = std::max(y1, c[1] / c[3]);
    }
    x0 = std::max(x0, -1.f);
    y0 = std::max(y0, -1.f);
    x1 = std::min(x1, 1.f);
    y1 = std::min(y1, 1.f);
    if (!(x1 > x0 && y1 > y0)) return false;  // off screen
    const float W = static_cast<float>(view.width()), H = static_cast<float>(view.height());
    float pw = 0.5f * (x1 - x0) * W * v.picture_scale, ph = 0.5f * (y1 - y0) * H * v.picture_scale;
    const float big = std::max(pw, ph), cap = static_cast<float>(v.max_picture);
    if (big > cap) {
      pw *= cap / big;
      ph *= cap / big;
    }
    v.pw = std::clamp(static_cast<int>(std::ceil(pw)), 4, v.max_picture);
    v.ph = std::clamp(static_cast<int>(std::ceil(ph)), 4, v.max_picture);
    // the projection cropped to [x0, x1] x [y0, y1] of the screen
    const float sx = 2.f / (x1 - x0), sy = 2.f / (y1 - y0), tx = -(x1 + x0) / (x1 - x0), ty = -(y1 + y0) / (y1 - y0);
    const Mat4 crop = {sx, 0, 0, 0, 0, sy, 0, 0, 0, 0, 1, 0, tx, ty, 0, 1};
    const Mat4 cropped = mul(crop, view_to_clip);
    std::memcpy(v.cam.view, vol_view.data(), sizeof(v.cam.view));
    std::memcpy(v.cam.proj, cropped.data(), sizeof(v.cam.proj));
    // the quad: the rectangle's corners on the plane through the box's centre, facing the camera
    Mat4 inv;
    if (!invert(mul(view_to_clip, world_to_view), inv)) return false;
    const fuse::CameraPose pose = view.pose();
    const Vec3 centre = v.position + Vec3{0.f, 0.5f * ey * s, 0.f};
    const auto on_plane = [&](float nx, float ny) {
      const auto a = apply(inv, nx, ny, -1.f, 1.f), b = apply(inv, nx, ny, 1.f, 1.f);
      const Vec3 pa{a[0] / a[3], a[1] / a[3], a[2] / a[3]}, pb{b[0] / b[3], b[1] / b[3], b[2] / b[3]};
      const Vec3 d = pb - pa;
      const float den = dot(d, pose.forward);
      return std::abs(den) > 1e-12f ? pa + d * (dot(centre - pa, pose.forward) / den) : centre;
    };
    const Vec3 p00 = on_plane(x0, y0), p10 = on_plane(x1, y0), p01 = on_plane(x0, y1);
    fuse::Quad& q = v.pending;
    q.texture = v.tex;
    q.axis_u = (p10 - p00) * 0.5f;
    q.axis_v = (p01 - p00) * 0.5f;
    q.centre = p00 + q.axis_u + q.axis_v;
    // half a texel in from each edge: the texture beyond the picture holds older pixels
    const float T = static_cast<float>(v.max_picture);
    q.u0 = 0.5f / T;
    q.v0 = 0.5f / T;
    q.u1 = (static_cast<float>(v.pw) - 0.5f) / T;
    q.v1 = (static_cast<float>(v.ph) - 0.5f) / T;
    q.depth = view.world_to_screen(centre).depth;
    return true;
  }

  // --- one game frame
  void update(const fuse::FrameTime& time, const fuse::View& view) noexcept {
    const auto t_start = Clock::now();
    stats.frame = time.frame;
    // the packet this frame fills must have been drawn K frames ago
    for (std::uint64_t r = released.load(std::memory_order_acquire); r + static_cast<std::uint64_t>(K) <= seq; r = released.load(std::memory_order_acquire))
      released.wait(r, std::memory_order_acquire);
    Packet& pk = packets[static_cast<std::size_t>(seq % static_cast<std::uint64_t>(K))];

    // what runs: the volume when its frame changes, the cards, the manager's jobs
    n_jobs = 0;
    VolumeState& v = volume;
    if (v.on) {
      if (v.t0 < 0.0) v.t0 = time.time;
      v.t = std::max(0.0, time.time - v.t0);
      const int vf = static_cast<int>(std::floor(v.t * static_cast<double>(v.info.fps)));
      v.due = vf != v.frame_drawn || v.placement_changed;
      v.draw = false;
      if (v.due) {
        v.draw = v.picture && plan_volume(view);
        if (v.picture && !v.draw) v.has_quad = false;  // off screen or too close
        v.frame_drawn = vf;
        v.placement_changed = false;
        jobs[n_jobs++] = {JobKind::volume, 0};
      }
    }
    for (int c = 0; c < 2; ++c)
      if (cards[c].on) {
        cards[c].dt = time.dt;
        jobs[n_jobs++] = {JobKind::card, c};
      }
    int manager_jobs = 0;
    if (manager) {
      const fuse::CameraPose pose = view.pose();
      const float pos[3] = {pose.eye.x, pose.eye.y, pose.eye.z}, fwd[3] = {pose.forward.x, pose.forward.y, pose.forward.z}, up[3] = {pose.up.x, pose.up.y, pose.up.z};
      if (const nvfx_status st = nvfx_manager_set_camera(manager.get(), pos, fwd, up, pose.fov_y, static_cast<float>(view.width()), static_cast<float>(view.height()));
          st != NVFX_OK)
        fail(st, "manager: set_camera");
      if (const nvfx_status st = nvfx_manager_tick_begin(manager.get(), time.dt, &manager_jobs); st != NVFX_OK) {
        fail(st, "manager: tick_begin");
        manager_jobs = -1;
      }
      for (int i = 0; i < std::min(std::max(manager_jobs, 0), job_slots); ++i) {
        manager_status[i] = NVFX_OK;
        jobs[n_jobs++] = {JobKind::manager, i};
      }
    }
    v.ms = 0.0;
    if (n_jobs > 0) engine.jobs->run(n_jobs, &job, this);
    if (manager && manager_jobs >= 0)
      if (const nvfx_status st = nvfx_manager_tick_end(manager.get()); st != NVFX_OK) fail(st, "manager: tick_end");
    for (int i = 0; i < n_jobs; ++i) {
      if (jobs[i].kind == JobKind::manager && manager_status[jobs[i].arg] != NVFX_OK) fail(manager_status[jobs[i].arg], "manager: run_jobs(%d)", jobs[i].arg);
      if (jobs[i].kind == JobKind::card && cards[jobs[i].arg].status != NVFX_OK) fail(cards[jobs[i].arg].status, "%s: step and draw", jobs[i].arg ? "stage" : "scene");
      if (jobs[i].kind == JobKind::volume && v.status != NVFX_OK) fail(v.status, "volume: step and draw");
    }
    if (v.draw && v.status == NVFX_OK) {
      v.quad = v.pending;
      v.has_quad = true;
    }

    // the sprites' pictures: versions, copies that share a picture, one-shots that have finished
    std::size_t copied = 0;
    int drawn = 0, live = 0;
    pk.n_quads = 0;
    const fuse::CameraPose pose = view.pose();
    for (std::size_t i = 0; i < sprites.size(); ++i) {
      SpriteSlot& sp = sprites[i];
      sp.owner = -1;
      if (!sp.live) continue;
      nvfx_manager_picture pic{};
      if (const nvfx_status st = nvfx_manager_get_picture(manager.get(), sp.handle, &pic); st != NVFX_OK) {
        fail(st, "manager: get_picture");
        continue;
      }
      if ((pic.flags & NVFX_PICTURE_FINISHED) && sp.auto_despawn) {
        nvfx_manager_stop(manager.get(), sp.handle);
        sp.live = false;
        ++sp.gen;
        continue;
      }
      ++live;
      if (!pic.rgba || pic.size <= 0 || !pic.visible) {
        sp.last_rgba = nullptr;
        continue;
      }
      sp.rgba = pic.rgba;
      sp.pic_size = pic.size;
      sp.stride = pic.stride;
      for (std::size_t j = 0; j < i && sp.owner < 0; ++j)  // a copy without a time offset: the same picture, one texture
        if (sprites[j].owner == static_cast<int>(j) && sprites[j].rgba == pic.rgba) sp.owner = static_cast<int>(j);
      if (sp.owner >= 0) {
        sp.last_rgba = nullptr;
      } else {
        sp.owner = static_cast<int>(i);
        if ((pic.flags & NVFX_PICTURE_NEW) || pic.rgba != sp.last_rgba || pic.size != sp.last_size || pic.stride != sp.last_stride) ++sp.version;
        sp.last_rgba = pic.rgba;
        sp.last_size = pic.size;
        sp.last_stride = pic.stride;
        Image& dst = pk.sprite_px[i];
        if (dst.version != sp.version) {  // this packet's copy is stale
          const std::size_t row = static_cast<std::size_t>(pic.size) * 4;
          for (int y = 0; y < pic.size; ++y) std::memcpy(dst.bytes.data() + static_cast<std::size_t>(y) * row, pic.rgba + static_cast<std::size_t>(y) * pic.stride, row);
          dst.w = dst.h = pic.size;
          dst.row = row;
          dst.version = sp.version;
          copied += row * static_cast<std::size_t>(pic.size);
        }
      }
      // an upright quad facing the camera
      const SpriteSlot& own = sprites[static_cast<std::size_t>(sp.owner)];
      fuse::Quad& q = pk.quads[static_cast<std::size_t>(pk.n_quads++)];
      const float half = 0.5f * sp.size;
      q.centre = sp.base ? sp.position + Vec3{0.f, half, 0.f} : sp.position;
      Vec3 to_eye = pose.eye - q.centre;
      to_eye.y = 0.f;
      const Vec3 right = dot(to_eye, to_eye) > 1e-8f ? normalized(Vec3{to_eye.z, 0.f, -to_eye.x}) : normalized(cross(pose.forward, pose.up));
      q.axis_u = right * half;
      q.axis_v = Vec3{0.f, half, 0.f};
      q.texture = sprite_tex[static_cast<std::size_t>(sp.owner)];
      const float T = static_cast<float>(kSpriteTexture);
      q.u0 = q.v0 = 0.5f / T;  // half a texel in: the texture beyond a smaller picture holds older pixels
      q.u1 = q.v1 = (static_cast<float>(own.pic_size) - 0.5f) / T;
      q.depth = view.world_to_screen(q.centre).depth;
      q.opacity = 1.f;
      q.edge = edge_fade;
      ++drawn;
    }
    // the cards and the volume
    for (int c = 0; c < 2; ++c) {
      CardState& cs = cards[c];
      if (!cs.on) continue;
      copied += refresh(pk.card_px[c], cs.staging);
      pk.quads[static_cast<std::size_t>(pk.n_quads++)] = card_quad(cs, view);
    }
    if (v.on) {
      if (v.picture) copied += refresh(pk.volume_px, v.picture_img);
      if (v.fields)
        for (int i = 0; i < 2; ++i) copied += refresh(pk.field_px[i], v.field_img[i]);
      if (v.has_quad) {
        v.quad.depth = view.world_to_screen(v.quad.centre).depth;
        pk.quads[static_cast<std::size_t>(pk.n_quads++)] = v.quad;
      }
    }
    pk.frame = time.frame;
    ++seq;
    published.store(seq, std::memory_order_release);

    // stats
    stats.sprites = live;
    stats.sprites_drawn = drawn;
    if (manager) {
      nvfx_manager_get_stats(manager.get(), &stats.manager);
      stats.memory.manager = stats.manager.memory_bytes;  // reported from its first tick on
      stats.memory_bytes = stats.memory.models + stats.memory.manager + stats.memory.scenes + stats.memory.volume + stats.memory.packets;
    }
    stats.copied_bytes = copied;
    stats.volume_ms = v.ms;
    stats.volume_step_ms = v.due ? v.step_ms : 0.0;
    stats.volume_picture_ms = v.due ? v.picture_ms : 0.0;
    stats.volume_fields_ms = v.due ? v.fields_ms : 0.0;
    stats.scene_ms = cards[0].ms;
    stats.stage_ms = cards[1].ms;
    stats.uploaded_bytes = uploaded_bytes.load(std::memory_order_relaxed);
    stats.frames_drawn = frames_drawn.load(std::memory_order_relaxed);
    stats.update_ms = ms_between(t_start, Clock::now());
  }

  // --- the render hook (FUSE's render thread)
  static void render_hook(void* ctx, const fuse::RenderFrame& frame) noexcept { static_cast<Impl*>(ctx)->render(frame); }

  void upload(fuse::Texture t, std::size_t stream, const Image& img, std::uint64_t& bytes) noexcept {
    if (img.version == 0 || img.version == uploaded[stream]) return;
    engine.gpu->update(t, fuse::TextureRegion{0, 0, 0, img.w, img.h, img.d}, img.bytes.data(), img.row);
    uploaded[stream] = img.version;
    bytes += img.row * static_cast<std::size_t>(img.h) * static_cast<std::size_t>(img.d);
  }

  void render(const fuse::RenderFrame& frame) noexcept {
    const std::uint64_t pub = published.load(std::memory_order_acquire), r = released.load(std::memory_order_relaxed);
    // the newest packet not drawn yet whose frame is not after the one FUSE draws (the game thread does not touch these)
    std::uint64_t chosen = pub;
    for (std::uint64_t s = pub; s > r; --s)
      if (packets[static_cast<std::size_t>((s - 1) % static_cast<std::uint64_t>(K))].frame <= frame.frame) {
        chosen = s - 1;
        break;
      }
    if (chosen == pub) return;  // nothing published for this frame
    const Packet& pk = packets[static_cast<std::size_t>(chosen % static_cast<std::uint64_t>(K))];
    std::uint64_t bytes = 0;
    const std::size_t n = sprites.size();
    for (std::size_t i = 0; i < n; ++i) upload(sprite_tex[i], i, pk.sprite_px[i], bytes);
    for (std::size_t c = 0; c < 2; ++c)
      if (cards[c].on) upload(cards[c].tex, n + c, pk.card_px[c], bytes);
    if (volume.on && volume.picture) upload(volume.tex, n + 2, pk.volume_px, bytes);
    if (volume.on && volume.fields)
      for (std::size_t i = 0; i < 2; ++i) upload(volume.field_tex[i], n + 3 + i, pk.field_px[i], bytes);
    if (frame.sink)
      for (int i = 0; i < pk.n_quads; ++i) frame.sink->draw(pk.quads[static_cast<std::size_t>(i)]);
    uploaded_bytes.fetch_add(bytes, std::memory_order_relaxed);
    frames_drawn.fetch_add(1, std::memory_order_relaxed);
    released.store(chosen + 1, std::memory_order_release);
    released.notify_all();
  }
};

// --- loading -------------------------------------------------------------------------------------------------------------

namespace {

std::expected<void, nvfx::Error> open_card(System::Impl::CardState& c, nvfx_scene* s, const Card& card, fuse::GpuUpload& gpu, const char* name) {
  c.on = true;
  c.s = s;
  c.card = card;
  c.card.right = normalized(card.right);
  c.card.up = normalized(card.up);
  c.normal = cross(c.card.right, c.card.up);
  if (!(card.units_per_px > 0.f) || dot(c.normal, c.normal) < 0.5f) return std::unexpected(load_error(NVFX_ERROR_ARGUMENT, std::string(name) + ": the card needs units_per_px > 0 and two axes at right angles"));
  nvfx_scene_info info{};
  nvfx_scene_get_info(s, &info);
  nvfx_scene_camera(s, &c.cam_x, &c.cam_y);
  c.scene_w = info.scene_width;
  c.scene_h = info.scene_height;
  c.ground = info.ground;
  Image& img = c.staging;
  img.w = info.width;
  img.h = info.height;
  img.row = static_cast<std::size_t>(info.width) * 4;
  img.bytes.assign(img.row * static_cast<std::size_t>(info.height), 0);
  // frame 0, computed by the scene's creation: its layers drawn now, so the first update has a picture
  if (const nvfx_status st = nvfx_scene_render_layer(s, NVFX_LAYER_COLOUR, NVFX_FORMAT_RGBA8, img.bytes.data(), img.row); st != NVFX_OK)
    return std::unexpected(load_error(st, std::string(name) + ": the first picture: " + nvfx_status_string(st)));
  img.version = 1;
  c.tex = gpu.create_texture({info.width, info.height, 1, fuse::TextureFormat::Rgba8Srgb, name});
  if (!c.tex) return std::unexpected(load_error(NVFX_ERROR_MEMORY, std::string(name) + ": FUSE could not create its texture"));
  return {};
}

}  // namespace

System::System(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
System::~System() = default;

std::expected<std::unique_ptr<System>, nvfx::Error> System::create(const fuse::Engine& engine, const Config& cfg) {
  using std::unexpected;
  if (!engine.jobs || !engine.hooks || !engine.gpu) return unexpected(load_error(NVFX_ERROR_ARGUMENT, "the engine's job system, render hooks and GPU upload are all needed"));
  if (cfg.frames_in_flight < 2) return unexpected(load_error(NVFX_ERROR_ARGUMENT, "frames_in_flight must be 2 or more"));

  // 1. libnvfx is the one these headers describe, with what this level needs
  const nvfx::Version ver = nvfx::version();
  if (!ver.matches_headers())
    return unexpected(load_error(NVFX_ERROR_UNSUPPORTED, "libnvfx is version " + std::to_string(ver.version) + ", the headers " + std::to_string(NVFX_VERSION)));
  std::uint32_t need = NVFX_FEATURE_PACK;
  if (!cfg.sprites.models.empty()) need |= NVFX_FEATURE_MANAGER;
  if (cfg.scene.script) need |= NVFX_FEATURE_SCENE;
  if (!cfg.stage.rooms.empty()) need |= NVFX_FEATURE_SCENE | NVFX_FEATURE_LIBRARY;
  if (cfg.volume.member) need |= NVFX_FEATURE_VOLUME;
  if (!ver.has(need)) return unexpected(load_error(NVFX_ERROR_UNSUPPORTED, "libnvfx lacks a feature this level needs (features 0x" + std::format("{:x}", ver.features) + ")"));

  auto s = std::make_unique<Impl>();
  s->engine = engine;
  s->K = cfg.frames_in_flight;

  // 2. the pack: one read, every member's checksums checked
  auto pd = nvfx::pack_desc();
  pd.path = cfg.pack_path;
  pd.data = cfg.pack_bytes.data();
  pd.bytes = cfg.pack_bytes.size();
  pd.threads = cfg.load_threads;
  auto pack = nvfx::Pack::open(pd);
  if (!pack) return unexpected(pack.error());

  // 3. sprites: the models and the manager, its costs from the pack, else measured here or the defaults
  if (!cfg.sprites.models.empty()) {
    const SpriteConfig& sc = cfg.sprites;
    if (sc.max_sprites < 1 || sc.job_slots < 1 || sc.job_slots > 16) return unexpected(load_error(NVFX_ERROR_ARGUMENT, "sprites: max_sprites >= 1 and job_slots in 1..16"));
    auto md = nvfx::manager_desc();
    md.threads = 0;  // FUSE's job system runs the work
    md.job_slots = sc.job_slots;
    md.max_instances = sc.max_sprites;
    md.budget_ms = sc.budget_ms;
    md.texel_scale = sc.texel_scale;
    if (sc.half_px >= 0.f) md.half_px = sc.half_px;  // else the manager's default
    md.half_share = sc.half_share;
    md.half_upscale = sc.half_upscale;
    auto mgr = nvfx::Manager::create(md);
    if (!mgr) return unexpected(mgr.error());
    for (const SpriteModel& m : sc.models) {
      auto e = pack->load_effect(m.name ? m.name : "");
      if (!e) return unexpected(load_error(e.error().status, std::string("sprites: the pack has no model '") + (m.name ? m.name : "") + "'"));
      const nvfx_effect_info info = e->get_info().value_or(nvfx_effect_info{});
      s->model_names.emplace_back(m.name);
      s->model_controls.push_back(info.n_controls);
      s->effects.push_back(std::move(*e));
    }
    s->manager = std::move(*mgr);
    for (std::size_t i = 0; i < sc.models.size(); ++i) {
      auto mm = nvfx::manager_model_desc();
      if (sc.models[i].max_sims > 0) mm.max_sims = sc.models[i].max_sims;
      mm.precision = sc.models[i].precision;
      mm.half_px = sc.models[i].half_px;
      auto model = s->manager.add_model(s->effects[i], &mm);
      if (!model) return unexpected(load_error(model.error().status, "sprites: " + s->model_names[i] + " is not a rollout effect the manager plays"));
      s->manager_model.push_back(*model);
      const int from_pack = s->manager.set_costs_from(*pack, *model, s->model_names[i]).value_or(0);
      if (from_pack == 0 && sc.costs == Costs::pack_or_calibrate)
        if (auto r = s->manager.calibrate(*model); !r) return unexpected(r.error());
    }
    s->job_slots = sc.job_slots;
    s->edge_fade = std::clamp(sc.edge_fade, 0.f, 1.f);
    s->sprites.resize(static_cast<std::size_t>(sc.max_sprites));
    for (int i = 0; i < sc.max_sprites; ++i) {
      const fuse::Texture t = engine.gpu->create_texture({kSpriteTexture, kSpriteTexture, 1, fuse::TextureFormat::Rgba8Srgb, "nvfx sprite"});
      if (!t) return unexpected(load_error(NVFX_ERROR_MEMORY, "sprites: FUSE could not create a texture"));
      s->sprite_tex.push_back(t);
    }
  }

  // 4. the scripted scene: on one thread (it runs as one FUSE job), its engine layers on
  if (cfg.scene.script) {
    auto d = nvfx::scene_desc();
    d.script_path = cfg.scene.script;
    d.inputs = cfg.scene.inputs.data();
    d.n_inputs = static_cast<int>(cfg.scene.inputs.size());
    d.threads = 1;
    d.overlap = 0;
    d.layers = 1;
    d.width = cfg.scene.width;
    d.height = cfg.scene.height;
    d.precision = cfg.scene.precision;
    auto sc = nvfx::Scene::create_from_pack(*pack, d);
    if (!sc) return unexpected(sc.error());
    s->scene = std::move(*sc);
    if (auto r = open_card(s->cards[0], s->scene.get(), cfg.scene.card, *engine.gpu, "nvfx scene"); !r) return unexpected(r.error());
  }

  // 5. the stage of the pack's effect library, likewise
  if (!cfg.stage.rooms.empty()) {
    auto lib = nvfx::Library::open_pack(*pack);
    if (!lib) return unexpected(lib.error());
    s->library = std::move(*lib);
    auto d = nvfx::stage_desc();
    d.rooms = cfg.stage.rooms.data();
    d.n_rooms = static_cast<int>(cfg.stage.rooms.size());
    d.width = cfg.stage.width;
    d.height = cfg.stage.height;
    d.threads = 1;
    d.overlap = 0;
    d.layers = 1;
    d.precision = cfg.stage.precision;
    auto st = nvfx::Stage::create(s->library, d);
    if (!st) return unexpected(st.error());
    s->stage = std::move(*st);
    if (auto r = open_card(s->cards[1], nvfx_stage_scene(s->stage.get()), cfg.stage.card, *engine.gpu, "nvfx stage"); !r) return unexpected(r.error());
  }

  // 6. the volume: its file a member of the pack, warmed up now (frame 0)
  if (cfg.volume.member) {
    const VolumeConfig& vc = cfg.volume;
    auto& v = s->volume;
    const int idx = pack->find(vc.member);
    if (idx < 0) return unexpected(load_error(NVFX_ERROR_IO, std::string("volume: the pack has no member '") + vc.member + "'"));
    const nvfx_pack_member m = pack->get_member(idx).value_or(nvfx_pack_member{});
    if (const nvfx_status st = nvfx_volume_effect_load_memory(m.data, m.bytes, &s->vfx); st != NVFX_OK)
      return unexpected(load_error(st, std::string("volume: ") + vc.member + ": " + nvfx_status_string(st)));
    if (const nvfx_status st = nvfx_volume_create(s->vfx, std::max(1, vc.threads), &s->vol); st != NVFX_OK) return unexpected(load_error(st, "volume: cannot create it"));
    nvfx_volume_effect_get_info(s->vfx, &v.info);
    nvfx_volume_set_controls(s->vol, vc.controls.data(), std::min(3, v.info.n_controls));
    nvfx_volume_set_seed(s->vol, vc.seed);
    nvfx_volume_set_shard(s->vol, vc.shard_seconds);
    if (const nvfx_status st = nvfx_volume_step_to(s->vol, 0.0); st != NVFX_OK) return unexpected(load_error(st, "volume: the warm-up"));
    v.on = true;
    v.picture = vc.picture;
    v.fields = vc.fields;
    v.position = vc.position;
    v.scale = vc.scale;
    v.picture_scale = vc.picture_scale;
    v.max_picture = std::clamp(vc.max_picture, 16, 2048);
    if (v.picture) {
      v.picture_img.bytes.assign(static_cast<std::size_t>(v.max_picture) * static_cast<std::size_t>(v.max_picture) * 4, 0);
      v.tex = engine.gpu->create_texture({v.max_picture, v.max_picture, 1, fuse::TextureFormat::Rgba8Srgb, "nvfx volume"});
      if (!v.tex) return unexpected(load_error(NVFX_ERROR_MEMORY, "volume: FUSE could not create its texture"));
    }
    if (v.fields)
      for (int i = 0; i < 2; ++i) {
        Image& f = v.field_img[i];
        f.w = v.info.size[0];
        f.h = v.info.size[1];
        f.d = v.info.size[2];
        f.row = static_cast<std::size_t>(f.w) * 2;
        f.bytes.assign(f.row * static_cast<std::size_t>(f.h) * static_cast<std::size_t>(f.d), 0);
        v.field_tex[i] = engine.gpu->create_texture({f.w, f.h, f.d, fuse::TextureFormat::R16Float3d, i ? "nvfx volume extinction" : "nvfx volume temperature"});
        if (!v.field_tex[i]) return unexpected(load_error(NVFX_ERROR_MEMORY, "volume: FUSE could not create a 3D texture"));
      }
  }

  // 7. the packets: each holds the newest pixels of everything it shows, so its copies are made now
  const std::size_t n_sprites = s->sprites.size();
  s->packets.resize(static_cast<std::size_t>(s->K));
  for (Impl::Packet& p : s->packets) {
    p.quads.resize(n_sprites + 3);
    p.sprite_px.resize(n_sprites);
    for (Image& img : p.sprite_px) img.bytes.assign(static_cast<std::size_t>(kSpriteTexture) * kSpriteTexture * 4, 0);
    for (int c = 0; c < 2; ++c) p.card_px[c].bytes.assign(s->cards[c].staging.bytes.size(), 0);
    p.volume_px.bytes.assign(s->volume.picture_img.bytes.size(), 0);
    for (int i = 0; i < 2; ++i) p.field_px[i].bytes.assign(s->volume.field_img[i].bytes.size(), 0);
  }
  s->uploaded.assign(n_sprites + 5, 0);

  // what this level holds
  Stats::Memory& mb = s->stats.memory;
  if (s->manager) mb.manager = s->manager.get_stats().value_or(nvfx_manager_stats{}).memory_bytes;
  for (const nvfx::Effect& e : s->effects) mb.models += e.get_info().value_or(nvfx_effect_info{}).resident_bytes;
  for (const Impl::CardState& c : s->cards)
    if (c.on) {
      nvfx_scene_info info{};
      nvfx_scene_get_info(c.s, &info);
      mb.scenes += info.scratch_bytes + c.staging.bytes.size();
    }
  if (s->vol) mb.volume = nvfx_volume_scratch_bytes(s->vol) + s->volume.info.resident_bytes + s->volume.picture_img.bytes.size() + 2 * s->volume.field_img[0].bytes.size();
  std::size_t tex = n_sprites * kSpriteTexture * kSpriteTexture * 4 + s->volume.picture_img.bytes.size() + 2 * s->volume.field_img[0].bytes.size();
  for (const Impl::CardState& c : s->cards) tex += c.staging.bytes.size();
  for (const Impl::Packet& p : s->packets) {
    for (const Image& img : p.sprite_px) mb.packets += img.bytes.size();
    mb.packets += p.card_px[0].bytes.size() + p.card_px[1].bytes.size() + p.volume_px.bytes.size() + p.field_px[0].bytes.size() + p.field_px[1].bytes.size();
  }
  s->stats.memory_bytes = mb.models + mb.manager + mb.scenes + mb.volume + mb.packets;
  s->stats.texture_bytes = tex;

  // 8. the render hook; the pack is freed here (the library, the scene and the effects keep what they need)
  if (!engine.hooks->add(&Impl::render_hook, s.get())) return unexpected(load_error(NVFX_ERROR_MEMORY, "FUSE has no room for another render hook"));
  s->registered = true;
  return std::unique_ptr<System>(new System(std::move(s)));
}

// --- the calls -----------------------------------------------------------------------------------------------------------

int System::model(std::string_view name) const noexcept {
  for (std::size_t i = 0; i < impl_->model_names.size(); ++i)
    if (impl_->model_names[i] == name) return static_cast<int>(i);
  return -1;
}

namespace {
std::uint32_t sprite_id(std::size_t slot, std::uint16_t gen) { return (static_cast<std::uint32_t>(gen) << 16) | static_cast<std::uint32_t>(slot + 1); }
}  // namespace

Sprite System::spawn(const SpriteDesc& d) noexcept {
  Impl& s = *impl_;
  if (!s.manager || d.model < 0 || d.model >= static_cast<int>(s.manager_model.size())) {
    s.fail(NVFX_ERROR_ARGUMENT, "spawn: no sprite model %d", d.model);
    return {};
  }
  std::size_t slot = 0;
  while (slot < s.sprites.size() && s.sprites[slot].live) ++slot;
  if (slot == s.sprites.size()) {
    s.fail(NVFX_ERROR_MEMORY, "spawn: max_sprites (%zu) are playing", s.sprites.size());
    return {};
  }
  auto sd = nvfx::spawn_desc();
  sd.model = s.manager_model[static_cast<std::size_t>(d.model)];
  sd.controls = d.controls.data();
  sd.n_controls = std::min(3, s.model_controls[static_cast<std::size_t>(d.model)]);
  sd.seed = d.seed;
  sd.share = d.share ? 1 : 0;
  sd.time_offset = d.time_offset;
  sd.allow_proxy = d.allow_proxy ? 1 : 0;
  sd.priority = d.priority;
  const Vec3 c = d.anchor_base ? d.position + Vec3{0.f, 0.5f * d.size, 0.f} : d.position;
  sd.position[0] = c.x;
  sd.position[1] = c.y;
  sd.position[2] = c.z;
  sd.world_size = d.size;
  nvfx_handle h = 0;
  if (const nvfx_status st = nvfx_manager_spawn(s.manager.get(), &sd, &h); st != NVFX_OK) {
    s.fail(st, "spawn: %s", s.model_names[static_cast<std::size_t>(d.model)].c_str());
    return {};
  }
  Impl::SpriteSlot& sp = s.sprites[slot];
  sp.live = true;
  sp.handle = h;
  sp.model = d.model;
  sp.position = d.position;
  sp.size = d.size;
  sp.base = d.anchor_base;
  sp.auto_despawn = d.despawn_when_finished;
  sp.last_rgba = nullptr;
  return Sprite{sprite_id(slot, sp.gen)};
}

namespace {
System::Impl::SpriteSlot* find(System::Impl& s, Sprite h) noexcept {
  const std::size_t slot = (h.id & 0xffffu);
  if (slot == 0 || slot > s.sprites.size()) return nullptr;
  System::Impl::SpriteSlot& sp = s.sprites[slot - 1];
  return sp.live && sprite_id(slot - 1, sp.gen) == h.id ? &sp : nullptr;
}
}  // namespace

bool System::despawn(Sprite h) noexcept {
  Impl::SpriteSlot* sp = find(*impl_, h);
  if (!sp) return false;
  nvfx_manager_stop(impl_->manager.get(), sp->handle);
  sp->live = false;
  ++sp->gen;
  return true;
}

bool System::move(Sprite h, Vec3 p) noexcept {
  Impl::SpriteSlot* sp = find(*impl_, h);
  if (!sp) return false;
  sp->position = p;
  const Vec3 c = sp->base ? p + Vec3{0.f, 0.5f * sp->size, 0.f} : p;
  const float pos[3] = {c.x, c.y, c.z};
  const nvfx_status st = nvfx_manager_set_position(impl_->manager.get(), sp->handle, pos, sp->size);
  impl_->fail(st, "move");
  return st == NVFX_OK;
}

bool System::set_controls(Sprite h, std::span<const float> controls) noexcept {
  Impl::SpriteSlot* sp = find(*impl_, h);
  if (!sp) return false;
  const nvfx_status st = nvfx_manager_set_controls(impl_->manager.get(), sp->handle, controls.data(), static_cast<int>(controls.size()));
  impl_->fail(st, "set_controls");
  return st == NVFX_OK;
}

bool System::alive(Sprite h) const noexcept { return find(*impl_, h) != nullptr; }

bool System::set_input(const char* name, float value) noexcept {
  if (!impl_->scene) return false;
  const nvfx_status st = nvfx_scene_set_input(impl_->scene.get(), name, value);
  impl_->fail(st, "scene: set_input %s", name ? name : "(null)");
  return st == NVFX_OK;
}

bool System::trigger(const char* rule) noexcept {
  if (!impl_->scene) return false;
  const nvfx_status st = nvfx_scene_trigger(impl_->scene.get(), rule);
  impl_->fail(st, "scene: trigger %s", rule ? rule : "(null)");
  return st == NVFX_OK;
}

nvfx::SceneView System::scene() noexcept { return nvfx::SceneView(impl_->scene.get()); }
nvfx::SceneView System::stage_scene() noexcept { return nvfx::SceneView(impl_->cards[1].on ? impl_->cards[1].s : nullptr); }

StageFx System::spawn_entry(const char* entry, Vec3 at, float scale) noexcept {
  Impl& s = *impl_;
  if (!s.stage) return {};
  float x = 0.f, y = 0.f;
  s.to_card(s.cards[1], at, x, y, 1e30f);
  nvfx_fx fx = 0;
  const nvfx_status st = nvfx_stage_spawn(s.stage.get(), entry, x, y, scale, -1.0, &fx);
  s.fail(st, "stage: spawn %s", entry ? entry : "(null)");
  return st == NVFX_OK ? StageFx{fx} : StageFx{};
}

bool System::stop_entry(StageFx fx) noexcept {
  if (!impl_->stage) return false;
  const nvfx_status st = nvfx_stage_stop(impl_->stage.get(), fx.id, -1.0);
  impl_->fail(st, "stage: stop");
  return st == NVFX_OK;
}

bool System::set_entry_param(StageFx fx, const char* param, float value) noexcept {
  if (!impl_->stage) return false;
  const nvfx_status st = nvfx_stage_set_param(impl_->stage.get(), fx.id, param, value);
  impl_->fail(st, "stage: set_param %s", param ? param : "(null)");
  return st == NVFX_OK;
}

float System::heat_at(Vec3 p) const noexcept {
  float heat = 0.f;
  for (const Impl::CardState& c : impl_->cards) {
    float x = 0.f, y = 0.f;
    nvfx_scene_fields f{};
    if (c.on && impl_->to_card(c, p, x, y, 0.5f) && nvfx_scene_sample(c.s, x, y, &f) == NVFX_OK) heat = std::max(heat, f.heat);
  }
  return heat;
}

bool System::set_volume_controls(std::span<const float> controls) noexcept {
  if (!impl_->vol) return false;
  const nvfx_status st = nvfx_volume_set_controls(impl_->vol, controls.data(), static_cast<int>(controls.size()));
  impl_->fail(st, "volume: set_controls");
  return st == NVFX_OK;
}

void System::place_volume(Vec3 position, float scale) noexcept {
  impl_->volume.position = position;
  impl_->volume.scale = scale;
  impl_->volume.placement_changed = true;
}

void System::update(const fuse::FrameTime& time, const fuse::View& view) noexcept { impl_->update(time, view); }
const Stats& System::stats() const noexcept {
  impl_->stats.uploaded_bytes = impl_->uploaded_bytes.load(std::memory_order_relaxed);  // the render thread's counters, now
  impl_->stats.frames_drawn = impl_->frames_drawn.load(std::memory_order_relaxed);
  return impl_->stats;
}
const FrameError& System::last_error() const noexcept { return impl_->error; }

}  // namespace nvfx_fuse
