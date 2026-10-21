// The stand-in FUSE of the demo (standin.hpp).
#include "standin.hpp"

#include "label_font.hpp"

#include <zlib.h>

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string_view>

namespace standin {
namespace {

Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec3 cross(Vec3 a, Vec3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
Vec3 normalized(Vec3 a) { return a * (1.f / std::sqrt(dot(a, a))); }

Mat4 mul(const Mat4& a, const Mat4& b) {
  Mat4 r{};
  for (std::size_t c = 0; c < 4; ++c)
    for (std::size_t i = 0; i < 4; ++i) {
      float s = 0.f;
      for (std::size_t k = 0; k < 4; ++k) s += a[k * 4 + i] * b[c * 4 + k];
      r[c * 4 + i] = s;
    }
  return r;
}

std::array<float, 4> apply(const Mat4& m, Vec3 p, float w) {
  std::array<float, 4> r{};
  for (std::size_t i = 0; i < 4; ++i) r[i] = m[i] * p.x + m[4 + i] * p.y + m[8 + i] * p.z + m[12 + i] * w;
  return r;
}

float fract(float x) { return x - std::floor(x); }


}  // namespace

// --- the thread pool --------------------------------------------------------------------------------------------------

ThreadPool::ThreadPool(int workers) : workers_(std::max(0, workers)) {
  for (int i = 0; i < workers_; ++i) threads_.emplace_back([this] { loop(); });
}

ThreadPool::~ThreadPool() {
  quit_.store(true, std::memory_order_relaxed);
  generation_.fetch_add(1, std::memory_order_release);
  generation_.notify_all();
  for (std::thread& t : threads_) t.join();
}

void ThreadPool::work() noexcept {
  for (;;) {
    const int j = next_.fetch_add(1, std::memory_order_relaxed);
    if (j >= jobs_) return;
    fn_(context_, j);
  }
}

void ThreadPool::loop() noexcept {
  std::uint32_t seen = 0;
  for (;;) {
    generation_.wait(seen, std::memory_order_acquire);
    seen = generation_.load(std::memory_order_acquire);
    if (quit_.load(std::memory_order_relaxed)) return;
    work();
    finished_.fetch_add(1, std::memory_order_release);
    finished_.notify_all();
  }
}

void ThreadPool::run(int jobs, Job fn, void* context) noexcept {
  if (jobs <= 0) return;
  jobs_ = jobs;
  fn_ = fn;
  context_ = context;
  next_.store(0, std::memory_order_relaxed);
  finished_.store(0, std::memory_order_relaxed);
  generation_.fetch_add(1, std::memory_order_release);  // every worker takes part in every run, so none misses one
  generation_.notify_all();
  work();
  for (int f = finished_.load(std::memory_order_acquire); f < workers_; f = finished_.load(std::memory_order_acquire)) finished_.wait(f, std::memory_order_acquire);
}

// --- the camera -------------------------------------------------------------------------------------------------------

void LookAtCamera::look_at(Vec3 eye, Vec3 target) {
  const Vec3 f = normalized(target - eye), r = normalized(cross(f, Vec3{0.f, 1.f, 0.f})), u = cross(r, f);
  pose_ = {eye, f, u, fov_};
  view_ = {r.x, u.x, -f.x, 0.f, r.y, u.y, -f.y, 0.f, r.z, u.z, -f.z, 0.f, -dot(r, eye), -dot(u, eye), dot(f, eye), 1.f};
  const float n = 0.05f, far = 200.f, t = 1.f / std::tan(0.5f * fov_), aspect = static_cast<float>(w_) / static_cast<float>(h_);
  proj_ = {t / aspect, 0.f, 0.f, 0.f, 0.f, t, 0.f, 0.f, 0.f, 0.f, (far + n) / (n - far), -1.f, 0.f, 0.f, 2.f * far * n / (n - far), 0.f};
  vp_ = mul(proj_, view_);
}

fuse::ScreenPoint LookAtCamera::world_to_screen(const Vec3& p) const noexcept {
  const auto c = apply(vp_, p, 1.f);
  if (!(c[3] > 1e-6f)) return {0.f, 0.f, c[3]};
  return {(c[0] / c[3] * 0.5f + 0.5f) * static_cast<float>(w_), (0.5f - 0.5f * c[1] / c[3]) * static_cast<float>(h_), c[3]};
}

void LookAtCamera::matrices(float world_to_view[16], float view_to_clip[16]) const noexcept {
  std::memcpy(world_to_view, view_.data(), sizeof(view_));
  std::memcpy(view_to_clip, proj_.data(), sizeof(proj_));
}

// --- the renderer -----------------------------------------------------------------------------------------------------

SoftRenderer::SoftRenderer(int width, int height, FrameDone done, void* context)
    : w_(width), h_(height), done_(done), done_ctx_(context), quads_(512), order_(512),
      fb_(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3), out_(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3) {
  sink_.r = this;
  for (int i = 0; i < 256; ++i) {
    const double c = i / 255.0;
    decode_[static_cast<std::size_t>(i)] = static_cast<float>(c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4));
  }
  for (int i = 0; i < 4096; ++i) {
    const double l = i / 4095.0, c = l <= 0.0031308 ? 12.92 * l : 1.055 * std::pow(l, 1.0 / 2.4) - 0.055;
    encode_[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(std::clamp(static_cast<int>(std::lround(c * 255.0)), 0, 255));
  }
  thread_ = std::thread([this] { loop(); });
}

SoftRenderer::~SoftRenderer() {
  finish();
  {
    std::lock_guard lk(queue_mu_);
    quit_.store(true);
  }
  queue_cv_.notify_all();
  thread_.join();
}

fuse::Texture SoftRenderer::create_texture(const fuse::TextureDesc& d) {
  const std::size_t texel = d.format == fuse::TextureFormat::Rgba8Srgb ? 4 : 2;
  Texture t{d, std::vector<std::uint8_t>(static_cast<std::size_t>(d.width) * static_cast<std::size_t>(d.height) * static_cast<std::size_t>(d.depth) * texel, 0), true};
  texture_bytes_ += t.data.size();
  textures_.push_back(std::move(t));
  return static_cast<fuse::Texture>(textures_.size());
}

void SoftRenderer::destroy_texture(fuse::Texture t) noexcept {
  if (t == 0 || t > textures_.size()) return;
  Texture& x = textures_[t - 1];
  texture_bytes_ -= x.data.size();
  x.live = false;
  x.data = {};
}

void SoftRenderer::update(fuse::Texture t, const fuse::TextureRegion& r, const void* pixels, std::size_t row_bytes) noexcept {
  if (t == 0 || t > textures_.size() || !textures_[t - 1].live) return;
  Texture& x = textures_[t - 1];
  const std::size_t texel = x.desc.format == fuse::TextureFormat::Rgba8Srgb ? 4 : 2, w = static_cast<std::size_t>(x.desc.width), h = static_cast<std::size_t>(x.desc.height);
  if (r.x < 0 || r.y < 0 || r.z < 0 || r.x + r.width > x.desc.width || r.y + r.height > x.desc.height || r.z + r.depth > x.desc.depth) return;
  const auto* src = static_cast<const std::uint8_t*>(pixels);
  for (int z = 0; z < r.depth; ++z)
    for (int y = 0; y < r.height; ++y)
      std::memcpy(x.data.data() + ((static_cast<std::size_t>(r.z + z) * h + static_cast<std::size_t>(r.y + y)) * w + static_cast<std::size_t>(r.x)) * texel,
                  src + (static_cast<std::size_t>(z) * static_cast<std::size_t>(r.height) + static_cast<std::size_t>(y)) * row_bytes, static_cast<std::size_t>(r.width) * texel);
}

bool SoftRenderer::add(Hook fn, void* context) {
  std::lock_guard lk(hooks_mu_);
  for (auto& h : hooks_)
    if (!h.first) {
      h = {fn, context};
      return true;
    }
  return false;
}

void SoftRenderer::remove(Hook fn, void* context) noexcept {
  std::lock_guard lk(hooks_mu_);  // the render thread holds it while it calls the hooks
  for (auto& h : hooks_)
    if (h.first == fn && h.second == context) h = {nullptr, nullptr};
}

int SoftRenderer::live_textures() const noexcept {
  int n = 0;
  for (const Texture& t : textures_) n += t.live ? 1 : 0;
  return n;
}

int SoftRenderer::hooks() noexcept {
  std::lock_guard lk(hooks_mu_);
  int n = 0;
  for (const auto& h : hooks_) n += h.first ? 1 : 0;
  return n;
}

void SoftRenderer::Sink::draw(const fuse::Quad& q) noexcept {
  if (r->n_quads_ < static_cast<int>(r->quads_.size())) r->quads_[static_cast<std::size_t>(r->n_quads_++)] = q;
}

void SoftRenderer::submit(std::uint64_t frame, const LookAtCamera& camera, const char* caption) {
  std::unique_lock lk(queue_mu_);
  queue_cv_.wait(lk, [&] { return submitted_ - drawn_ < 2; });  // FUSE's pipeline: at most one frame queued behind the one drawn
  FrameIn& in = in_[static_cast<std::size_t>(frame % in_.size())];
  in.vp = camera.view_proj();
  in.pose = camera.pose();
  std::snprintf(in.caption, sizeof(in.caption), "%s", caption ? caption : "");
  submitted_ = frame + 1;
  queue_cv_.notify_all();
}

void SoftRenderer::finish() {
  std::unique_lock lk(queue_mu_);
  queue_cv_.wait(lk, [&] { return drawn_ == submitted_; });
}

void SoftRenderer::loop() noexcept {
  std::unique_lock lk(queue_mu_);
  for (;;) {
    queue_cv_.wait(lk, [&] { return quit_.load() || submitted_ > drawn_; });
    if (submitted_ == drawn_) return;  // quit, with nothing left to draw
    const std::uint64_t f = drawn_;
    lk.unlock();
    draw(f);
    lk.lock();
    drawn_ = f + 1;
    queue_cv_.notify_all();
  }
}

void SoftRenderer::draw(std::uint64_t frame) noexcept {
  const FrameIn& in = in_[static_cast<std::size_t>(frame % in_.size())];
  sky_and_ground(in);
  n_quads_ = 0;
  {
    std::lock_guard lk(hooks_mu_);
    for (const auto& h : hooks_)
      if (h.first) h.first(h.second, fuse::RenderFrame{frame, &sink_});
  }
  // back to front; equal depths in the order drawn
  for (int i = 0; i < n_quads_; ++i) order_[static_cast<std::size_t>(i)] = {quads_[static_cast<std::size_t>(i)].depth, i};
  std::sort(order_.begin(), order_.begin() + n_quads_, [](const Sorted& a, const Sorted& b) { return a.depth != b.depth ? a.depth > b.depth : a.index < b.index; });
  for (int i = 0; i < n_quads_; ++i) rasterise(quads_[static_cast<std::size_t>(order_[static_cast<std::size_t>(i)].index)], in.vp);
  quads_drawn_.fetch_add(static_cast<std::uint64_t>(n_quads_), std::memory_order_relaxed);
  encode();
  // the HUD: the caption's lines ('|' between them) at the top left, light grey with a dark shadow
  std::string_view text(in.caption);
  for (int line = 0; !text.empty(); ++line) {
    const std::size_t bar = text.find('|');
    const std::string_view l = text.substr(0, bar);
    text = bar == std::string_view::npos ? std::string_view{} : text.substr(bar + 1);
    for (int pass = 0; pass < 2; ++pass)
      nfx::tools::draw_text(l, 16 + (pass ? 0 : 2), 14 + line * 22 + (pass ? 0 : 2), 2, [&](int x, int yy) {
        if (x < 0 || yy < 0 || x >= w_ || yy >= h_) return;
        std::uint8_t* p = &out_[(static_cast<std::size_t>(yy) * static_cast<std::size_t>(w_) + static_cast<std::size_t>(x)) * 3];
        p[0] = p[1] = p[2] = pass ? 225 : 8;
      });
  }
  done_(done_ctx_, frame, out_.data(), w_, h_);
}

void SoftRenderer::sky_and_ground(const FrameIn& in) noexcept {
  const fuse::CameraPose& p = in.pose;
  const Vec3 right = cross(p.forward, p.up);
  const float ty = std::tan(0.5f * p.fov_y), tx = ty * static_cast<float>(w_) / static_cast<float>(h_);
  const float sky_top[3] = {0.004f, 0.006f, 0.018f}, horizon[3] = {0.030f, 0.026f, 0.036f}, ground[3] = {0.012f, 0.010f, 0.009f}, grid[3] = {0.040f, 0.036f, 0.033f};
  for (int y = 0; y < h_; ++y) {
    const float ny = 1.f - (static_cast<float>(y) + 0.5f) / static_cast<float>(h_) * 2.f;
    for (int x = 0; x < w_; ++x) {
      const float nx = (static_cast<float>(x) + 0.5f) / static_cast<float>(w_) * 2.f - 1.f;
      const Vec3 d = p.forward + right * (nx * tx) + p.up * (ny * ty);
      const float len = std::sqrt(dot(d, d));
      float* o = &fb_[(static_cast<std::size_t>(y) * static_cast<std::size_t>(w_) + static_cast<std::size_t>(x)) * 3];
      if (d.y < -1e-5f && p.eye.y > 0.f) {
        const float t = -p.eye.y / d.y, dist = t * len;
        const float gx = p.eye.x + t * d.x, gz = p.eye.z + t * d.z;
        const float wline = 0.012f + 0.0025f * dist;  // wider with distance: a cheap anti-aliasing
        const float lx = std::max(0.f, 1.f - std::min(fract(gx), 1.f - fract(gx)) / wline), lz = std::max(0.f, 1.f - std::min(fract(gz), 1.f - fract(gz)) / wline);
        const float line = std::max(lx, lz) * std::exp(-dist / 30.f);
        const float fog = 1.f - std::exp(-dist / 45.f);
        for (int c = 0; c < 3; ++c) o[c] = (ground[c] + line * (grid[c] - ground[c])) * (1.f - fog) + horizon[c] * fog;
      } else {
        const float s = std::clamp(d.y / len * 3.f, 0.f, 1.f);
        for (int c = 0; c < 3; ++c) o[c] = horizon[c] + s * (sky_top[c] - horizon[c]);
      }
    }
  }
}

void SoftRenderer::rasterise(const fuse::Quad& q, const Mat4& vp) noexcept {
  if (q.texture == 0 || q.texture > textures_.size()) return;
  const Texture& t = textures_[q.texture - 1];
  if (!t.live || t.desc.format != fuse::TextureFormat::Rgba8Srgb) return;
  // screen (px * w, py * w, w) = M (s, t, 1) for the quad's point centre + s axis_u + t axis_v; its inverse maps pixels back
  const auto P0 = apply(vp, q.centre, 1.f), Pu = apply(vp, q.axis_u, 0.f), Pv = apply(vp, q.axis_v, 0.f);
  const double W = w_, H = h_;
  const double M[9] = {0.5 * W * (Pu[0] + Pu[3]), 0.5 * W * (Pv[0] + Pv[3]), 0.5 * W * (P0[0] + P0[3]),
                       0.5 * H * (Pu[3] - Pu[1]), 0.5 * H * (Pv[3] - Pv[1]), 0.5 * H * (P0[3] - P0[1]),
                       Pu[3], Pv[3], P0[3]};
  const double det = M[0] * (M[4] * M[8] - M[5] * M[7]) - M[1] * (M[3] * M[8] - M[5] * M[6]) + M[2] * (M[3] * M[7] - M[4] * M[6]);
  if (!(std::abs(det) > 1e-12)) return;  // seen edge on
  const double inv[9] = {(M[4] * M[8] - M[5] * M[7]) / det, (M[2] * M[7] - M[1] * M[8]) / det, (M[1] * M[5] - M[2] * M[4]) / det,
                         (M[5] * M[6] - M[3] * M[8]) / det, (M[0] * M[8] - M[2] * M[6]) / det, (M[2] * M[3] - M[0] * M[5]) / det,
                         (M[3] * M[7] - M[4] * M[6]) / det, (M[1] * M[6] - M[0] * M[7]) / det, (M[0] * M[4] - M[1] * M[3]) / det};
  // the pixels to visit: the corners' box when all four are in front of the camera, else the whole screen
  int bx0 = 0, by0 = 0, bx1 = w_, by1 = h_;
  bool front = true;
  double mnx = 1e30, mny = 1e30, mxx = -1e30, mxy = -1e30;
  for (int k = 0; k < 4; ++k) {
    const double s = (k & 1) ? 1.0 : -1.0, tt = (k & 2) ? 1.0 : -1.0;
    const double X = M[0] * s + M[1] * tt + M[2], Y = M[3] * s + M[4] * tt + M[5], Z = M[6] * s + M[7] * tt + M[8];
    if (!(Z > 0.05)) {
      front = false;
      break;
    }
    mnx = std::min(mnx, X / Z);
    mxx = std::max(mxx, X / Z);
    mny = std::min(mny, Y / Z);
    mxy = std::max(mxy, Y / Z);
  }
  if (front) {
    bx0 = std::max(0, static_cast<int>(std::floor(mnx)));
    by0 = std::max(0, static_cast<int>(std::floor(mny)));
    bx1 = std::min(w_, static_cast<int>(std::ceil(mxx)) + 1);
    by1 = std::min(h_, static_cast<int>(std::ceil(mxy)) + 1);
  }
  const int tw = t.desc.width, th = t.desc.height;
  const float op = q.opacity;
  const std::uint8_t* tex = t.data.data();
  const auto texel = [&](int x, int y) { return tex + (static_cast<std::size_t>(y) * static_cast<std::size_t>(tw) + static_cast<std::size_t>(x)) * 4; };
  for (int y = by0; y < by1; ++y) {
    const double py = y + 0.5;
    for (int x = bx0; x < bx1; ++x) {
      const double px = x + 0.5;
      const double A = inv[0] * px + inv[1] * py + inv[2], B = inv[3] * px + inv[4] * py + inv[5], C = inv[6] * px + inv[7] * py + inv[8];
      if (std::abs(C) < 1e-30) continue;
      const double s = A / C, tt = B / C;
      if (s < -1.0 || s > 1.0 || tt < -1.0 || tt > 1.0) continue;
      if (!(M[6] * s + M[7] * tt + M[8] > 0.05)) continue;  // behind the camera
      float opx = op;
      if (q.edge > 0.f) {  // soft edges
        const auto ramp = [&](double d) { const float r = std::clamp(static_cast<float>(d) / q.edge, 0.f, 1.f); return r * r * (3.f - 2.f * r); };
        opx *= ramp(1.0 - std::abs(s)) * ramp(1.0 - std::abs(tt));
        if (!(opx > 0.f)) continue;
      }
      const float u = q.u0 + static_cast<float>((s + 1.0) * 0.5) * (q.u1 - q.u0), v = q.v0 + static_cast<float>((1.0 - tt) * 0.5) * (q.v1 - q.v0);
      const float fu = u * static_cast<float>(tw) - 0.5f, fv = v * static_cast<float>(th) - 0.5f;
      const int x0 = static_cast<int>(std::floor(fu)), y0 = static_cast<int>(std::floor(fv));
      const float ax = fu - static_cast<float>(x0), ay = fv - static_cast<float>(y0);
      const int xa = std::clamp(x0, 0, tw - 1), xb = std::clamp(x0 + 1, 0, tw - 1), ya = std::clamp(y0, 0, th - 1), yb = std::clamp(y0 + 1, 0, th - 1);
      const std::uint8_t *t00 = texel(xa, ya), *t10 = texel(xb, ya), *t01 = texel(xa, yb), *t11 = texel(xb, yb);
      if ((t00[0] | t00[1] | t00[2] | t00[3] | t10[0] | t10[1] | t10[2] | t10[3] | t01[0] | t01[1] | t01[2] | t01[3] | t11[0] | t11[1] | t11[2] | t11[3]) == 0) continue;
      const float w00 = (1.f - ax) * (1.f - ay), w10 = ax * (1.f - ay), w01 = (1.f - ax) * ay, w11 = ax * ay;
      float* o = &fb_[(static_cast<std::size_t>(y) * static_cast<std::size_t>(w_) + static_cast<std::size_t>(x)) * 3];
      const float a = (w00 * t00[3] + w10 * t10[3] + w01 * t01[3] + w11 * t11[3]) * (opx / 255.f);
      for (std::size_t c = 0; c < 3; ++c) {
        const float src = w00 * decode_[t00[c]] + w10 * decode_[t10[c]] + w01 * decode_[t01[c]] + w11 * decode_[t11[c]];
        o[c] = src * opx + o[c] * (1.f - a);  // premultiplied: ONE, ONE_MINUS_SRC_ALPHA
      }
    }
  }
}

void SoftRenderer::encode() noexcept {
  const std::size_t n = fb_.size();
  for (std::size_t i = 0; i < n; ++i) {
    const float v = std::clamp(fb_[i], 0.f, 1.f);
    out_[i] = encode_[static_cast<std::size_t>(v * 4095.f + 0.5f)];
  }
}

// --- files --------------------------------------------------------------------------------------------------------------

bool write_png(const std::string& path, const std::uint8_t* rgb, int width, int height) {
  const std::size_t row = static_cast<std::size_t>(width) * 3;
  std::vector<std::uint8_t> raw((row + 1) * static_cast<std::size_t>(height));
  for (int y = 0; y < height; ++y) {
    raw[static_cast<std::size_t>(y) * (row + 1)] = 0;
    std::memcpy(&raw[static_cast<std::size_t>(y) * (row + 1) + 1], rgb + static_cast<std::size_t>(y) * row, row);
  }
  uLongf zn = compressBound(static_cast<uLong>(raw.size()));
  std::vector<std::uint8_t> z(zn);
  if (compress2(z.data(), &zn, raw.data(), static_cast<uLong>(raw.size()), 6) != Z_OK) return false;
  z.resize(zn);
  std::ofstream f(path, std::ios::binary);
  const auto be = [&](std::uint32_t v) {
    const char b[4] = {static_cast<char>(v >> 24), static_cast<char>(v >> 16), static_cast<char>(v >> 8), static_cast<char>(v)};
    f.write(b, 4);
  };
  const auto chunk = [&](const char* type, const std::uint8_t* data, std::size_t n) {
    be(static_cast<std::uint32_t>(n));
    f.write(type, 4);
    if (n) f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n));
    uLong c = crc32(0L, reinterpret_cast<const Bytef*>(type), 4);
    if (n) c = crc32(c, data, static_cast<uInt>(n));
    be(static_cast<std::uint32_t>(c));
  };
  f.write("\x89PNG\r\n\x1a\n", 8);
  const std::uint8_t ihdr[13] = {static_cast<std::uint8_t>(width >> 24), static_cast<std::uint8_t>(width >> 16), static_cast<std::uint8_t>(width >> 8), static_cast<std::uint8_t>(width),
                                 static_cast<std::uint8_t>(height >> 24), static_cast<std::uint8_t>(height >> 16), static_cast<std::uint8_t>(height >> 8), static_cast<std::uint8_t>(height),
                                 8, 2, 0, 0, 0};
  chunk("IHDR", ihdr, sizeof(ihdr));
  chunk("IDAT", z.data(), z.size());
  chunk("IEND", nullptr, 0);
  return static_cast<bool>(f);
}

std::uint64_t fnv1a(std::span<const std::uint8_t> bytes, std::uint64_t h) noexcept {
  for (const std::uint8_t b : bytes) h = (h ^ b) * 1099511628211ull;
  return h;
}

}  // namespace standin
