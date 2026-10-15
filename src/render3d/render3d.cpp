// Study N's volume renderer (include/neuralfx/render3d.hpp; docs/STUDY_3D.md §3). Plain C++ at the baseline ISA,
// compiled once, so every build draws the same pictures; each pixel is marched alone and every sum runs in a fixed
// order, so the pictures do not depend on the number of threads either.
#include <neuralfx/render3d.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace nfx::render3d {

namespace {

std::size_t zu(int v) { return static_cast<std::size_t>(v); }
float fl(int v) { return static_cast<float>(v); }
float smoothstep(float a, float b, float x) {
  const float t = std::clamp((x - a) / (b - a), 0.f, 1.f);
  return t * t * (3.f - 2.f * t);
}

// The CIE 1931 2-degree colour matching functions, Wyman, Sloan and Shirley's multi-lobe fit (JCGT 2(2), 2013).
double lobe(double l, double mu, double s1, double s2) {
  const double t = (l - mu) / (l < mu ? s1 : s2);
  return std::exp(-0.5 * t * t);
}
std::array<double, 3> cmf(double nm) {
  const double x = 1.056 * lobe(nm, 599.8, 37.9, 31.0) + 0.362 * lobe(nm, 442.0, 16.0, 26.7) - 0.065 * lobe(nm, 501.1, 20.4, 26.2);
  const double y = 0.821 * lobe(nm, 568.8, 46.9, 40.5) + 0.286 * lobe(nm, 530.9, 16.3, 31.1);
  const double z = 1.217 * lobe(nm, 437.0, 11.8, 36.0) + 0.681 * lobe(nm, 459.0, 26.0, 13.8);
  return {x, y, z};
}

// XYZ of a blackbody (Planck's law, W / (sr m^2 m) integrated over 360 to 830 nm in 1 nm steps against the CMFs).
std::array<double, 3> blackbody_xyz(double kelvin) {
  std::array<double, 3> s{};
  if (kelvin < 1.0) return s;
  constexpr double c1 = 1.191042972e-16, c2 = 1.438776877e-2;
  for (int nm = 360; nm <= 830; ++nm) {
    const double l = nm * 1e-9;
    const double e = c2 / (l * kelvin);
    if (e > 700.0) continue;
    const double b = c1 / (l * l * l * l * l * std::expm1(e));
    const auto c = cmf(nm);
    for (int k = 0; k < 3; ++k) s[zu(k)] += b * c[zu(k)] * 1e-9;
  }
  return s;
}

std::array<double, 3> xyz_to_srgb(const std::array<double, 3>& c) {
  return {3.2404542 * c[0] - 1.5371385 * c[1] - 0.4985314 * c[2], -0.9692660 * c[0] + 1.8760108 * c[1] + 0.0415560 * c[2],
          0.0556434 * c[0] - 0.2040259 * c[1] + 1.0572252 * c[2]};
}

constexpr float kTableStep = 5.f;  // kelvin
constexpr int kTableSize = 1601;   // 0 to 8000 K

float srgb_encode(float v) { return v <= 0.0031308f ? 12.92f * v : 1.055f * std::pow(v, 1.f / 2.4f) - 0.055f; }
std::uint8_t to8(float v) { return static_cast<std::uint8_t>(std::clamp(v, 0.f, 1.f) * 255.f + 0.5f); }

using Mat = std::array<double, 16>;
Mat mul(const std::array<float, 16>& a, const std::array<float, 16>& b) {  // column-major a * b
  Mat r{};
  for (int c = 0; c < 4; ++c) {
    for (int rr = 0; rr < 4; ++rr) {
      double s = 0;
      for (int k = 0; k < 4; ++k) s += static_cast<double>(a[zu(k * 4 + rr)]) * static_cast<double>(b[zu(c * 4 + k)]);
      r[zu(c * 4 + rr)] = s;
    }
  }
  return r;
}
Mat inverse(const Mat& m) {  // Gauss-Jordan with partial pivoting on a row-major copy
  double a[4][8];
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      a[r][c] = m[zu(c * 4 + r)];
      a[r][c + 4] = r == c ? 1.0 : 0.0;
    }
  }
  for (int c = 0; c < 4; ++c) {
    int piv = c;
    for (int r = c + 1; r < 4; ++r) {
      if (std::abs(a[r][c]) > std::abs(a[piv][c])) piv = r;
    }
    if (std::abs(a[piv][c]) < 1e-30) throw std::invalid_argument("render3d: the camera's matrices are singular");
    for (int k = 0; k < 8; ++k) std::swap(a[c][k], a[piv][k]);
    const double d = a[c][c];
    for (int k = 0; k < 8; ++k) a[c][k] /= d;
    for (int r = 0; r < 4; ++r) {
      if (r == c) continue;
      const double f = a[r][c];
      for (int k = 0; k < 8; ++k) a[r][k] -= f * a[c][k];
    }
  }
  Mat inv{};
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) inv[zu(c * 4 + r)] = a[r][c + 4];
  }
  return inv;
}
std::array<double, 3> unproject(const Mat& inv, double x, double y, double z) {
  double v[4];
  for (int r = 0; r < 4; ++r) v[r] = inv[zu(r)] * x + inv[zu(4 + r)] * y + inv[zu(8 + r)] * z + inv[zu(12 + r)];
  return {v[0] / v[3], v[1] / v[3], v[2] / v[3]};
}

std::array<float, 16> view_matrix(std::array<float, 3> eye, std::array<float, 3> target, std::array<float, 3> up) {
  const auto norm = [](std::array<float, 3> v) {
    const float l = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    return std::array<float, 3>{v[0] / l, v[1] / l, v[2] / l};
  };
  const auto cross = [](std::array<float, 3> a, std::array<float, 3> b) {
    return std::array<float, 3>{a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
  };
  const auto f = norm({target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]});
  const auto s = norm(cross(f, up));
  const auto u = cross(s, f);
  std::array<float, 16> m{};
  m[0] = s[0];
  m[4] = s[1];
  m[8] = s[2];
  m[1] = u[0];
  m[5] = u[1];
  m[9] = u[2];
  m[2] = -f[0];
  m[6] = -f[1];
  m[10] = -f[2];
  m[12] = -(s[0] * eye[0] + s[1] * eye[1] + s[2] * eye[2]);
  m[13] = -(u[0] * eye[0] + u[1] * eye[1] + u[2] * eye[2]);
  m[14] = f[0] * eye[0] + f[1] * eye[1] + f[2] * eye[2];
  m[15] = 1.f;
  return m;
}

}  // namespace

std::array<float, 3> blackbody_rgb(float kelvin) {
  const auto c = xyz_to_srgb(blackbody_xyz(kelvin));
  const double ref = blackbody_xyz(1800.0)[1];
  return {static_cast<float>(std::max(0.0, c[0] / ref)), static_cast<float>(std::max(0.0, c[1] / ref)), static_cast<float>(std::max(0.0, c[2] / ref))};
}

Camera look_at(std::array<float, 3> eye, std::array<float, 3> target, std::array<float, 3> up, float fov_y_degrees, float aspect, float near_plane,
               float far_plane) {
  Camera c;
  c.view = view_matrix(eye, target, up);
  const float f = 1.f / std::tan(0.5f * fov_y_degrees * 3.14159265f / 180.f);
  c.proj[0] = f / aspect;
  c.proj[5] = f;
  c.proj[10] = (far_plane + near_plane) / (near_plane - far_plane);
  c.proj[11] = -1.f;
  c.proj[14] = 2.f * far_plane * near_plane / (near_plane - far_plane);
  return c;
}

Camera orthographic(std::array<float, 3> eye, std::array<float, 3> target, std::array<float, 3> up, float half_width, float half_height, float near_plane,
                    float far_plane) {
  Camera c;
  c.view = view_matrix(eye, target, up);
  c.proj[0] = 1.f / half_width;
  c.proj[5] = 1.f / half_height;
  c.proj[10] = -2.f / (far_plane - near_plane);
  c.proj[14] = -(far_plane + near_plane) / (far_plane - near_plane);
  c.proj[15] = 1.f;
  return c;
}

Renderer::Renderer(int nx, int ny, int nz, const Optics& optics) : nx_(nx), ny_(ny), nz_(nz), o_(optics) {
  if (nx < 2 || ny < 2 || nz < 2 || nx > 1024 || ny > 1024 || nz > 1024) throw std::invalid_argument("render3d: grid size out of range");
  if (!(o_.step > 0.05f) || !(o_.light_dir[1] > 0.1f)) throw std::invalid_argument("render3d: bad step or light direction");
  const double ref = blackbody_xyz(o_.reference_kelvin)[1];
  table_.resize(zu(kTableSize) * 3);
  for (int i = 0; i < kTableSize; ++i) {
    const auto c = xyz_to_srgb(blackbody_xyz(static_cast<double>(i) * kTableStep));
    for (int k = 0; k < 3; ++k) table_[zu(i) * 3 + zu(k)] = static_cast<float>(std::max(0.0, c[zu(k)] / ref) * o_.emission);
  }
  const std::size_t n = zu(nx) * zu(ny) * zu(nz);
  sigma_.assign(n, 0.f);
  tlight_.assign(n, 1.f);
  vol_.assign(n * 4, 0.f);
  mx_ = (nx + kMacro - 1) / kMacro;
  my_ = (ny + kMacro - 1) / kMacro;
  mz_ = (nz + kMacro - 1) / kMacro;
  macro_.assign(zu(mx_) * zu(my_) * zu(mz_), 0.f);
  sums_.assign(zu(ny) * 7, 0.0);
}

std::size_t Renderer::bytes() const {
  return sizeof(float) * (table_.size() + sigma_.size() + tlight_.size() + vol_.size() + macro_.size()) + sizeof(double) * sums_.size();
}

void Renderer::prepare(std::span<const float> temperature, std::span<const float> soot, Pool* pool) {
  const std::size_t n = zu(nx_) * zu(ny_) * zu(nz_);
  if (temperature.size() < n || soot.size() < n) throw std::invalid_argument("render3d: field buffers too small");
  const float cell = 1.f / fl(nx_);  // world units per cell
  const float ks = o_.soot_extinction, kf = o_.flame_extinction, a = o_.albedo;
  // 1. extinction and emission per cell; per slab, the emitted power and its centre (for the fire's light)
  parallel_for(pool, ny_, [&](int y) {
    double* s = sums_.data() + zu(y) * 7;
    for (int k = 0; k < 7; ++k) s[k] = 0.0;
    for (int z = 0; z < nz_; ++z) {
      for (int x = 0; x < nx_; ++x) {
        const std::size_t i = (zu(y) * zu(nz_) + zu(z)) * zu(nx_) + zu(x);
        const float T = std::max(0.f, temperature[i]), D = std::max(0.f, soot[i]);
        const float flame = kf * smoothstep(o_.flame0, o_.flame1, T);
        const float sig = ks * D + flame, sabs = (1.f - a) * ks * D + flame;
        sigma_[i] = sig;
        float* v = vol_.data() + 4 * i;
        v[0] = sig;
        if (sabs > 0.f) {
          const float kel = std::clamp((o_.kelvin0 + o_.kelvin_per_unit * T) / kTableStep, 0.f, fl(kTableSize - 1) - 1e-3f);
          const int k0 = static_cast<int>(kel);
          const float f = kel - fl(k0);
          const float* t0 = table_.data() + zu(k0) * 3;
          for (int c = 0; c < 3; ++c) v[1 + c] = sabs * (t0[c] + f * (t0[3 + c] - t0[c]));
          const double lum = 0.2126 * v[1] + 0.7152 * v[2] + 0.0722 * v[3];
          s[0] += lum;
          s[1] += lum * x;
          s[2] += lum * y;
          s[3] += lum * z;
          s[4] += v[1];
          s[5] += v[2];
          s[6] += v[3];
        } else {
          v[1] = v[2] = v[3] = 0.f;
        }
      }
    }
  });
  // 2. transmittance towards the key light: a sweep from the top, each slab from the one above, sheared along the light
  const float lx0 = o_.light_dir[0], ly0 = o_.light_dir[1], lz0 = o_.light_dir[2];
  const float ll = std::sqrt(lx0 * lx0 + ly0 * ly0 + lz0 * lz0);
  const float lx = lx0 / ll, ly = ly0 / ll, lz = lz0 / ll;
  const float shx = lx / ly, shz = lz / ly, path = cell / ly;  // offset per slab (cells), path per slab (world units)
  for (int y = ny_ - 1; y >= 0; --y) {
    parallel_for(pool, nz_, [&](int z) {
      for (int x = 0; x < nx_; ++x) {
        const std::size_t i = (zu(y) * zu(nz_) + zu(z)) * zu(nx_) + zu(x);
        if (y == ny_ - 1) {
          tlight_[i] = std::exp(-0.5f * sigma_[i] * path);
          continue;
        }
        const float px = fl(x) + shx, pz = fl(z) + shz;
        float above = 1.f, sab = 0.f;
        if (px >= 0.f && pz >= 0.f && px <= fl(nx_ - 1) && pz <= fl(nz_ - 1)) {
          const int x0 = std::min(static_cast<int>(px), nx_ - 2), z0 = std::min(static_cast<int>(pz), nz_ - 2);
          const float fx = px - fl(x0), fz = pz - fl(z0);
          const std::size_t j = (zu(y + 1) * zu(nz_) + zu(z0)) * zu(nx_) + zu(x0);
          const auto bil = [&](const std::vector<float>& f) {
            const float a0 = f[j] + fx * (f[j + 1] - f[j]), a1 = f[j + zu(nx_)] + fx * (f[j + zu(nx_) + 1] - f[j + zu(nx_)]);
            return a0 + fz * (a1 - a0);
          };
          above = bil(tlight_);
          sab = bil(sigma_);
        }
        tlight_[i] = above * std::exp(-0.5f * (sigma_[i] + sab) * path);
      }
    });
  }
  // 3. the fire's light: its emitted power at the emission's centre (slab sums added in order)
  double tot = 0, cx = 0, cy = 0, cz = 0, pr = 0, pg = 0, pb = 0;
  for (int y = 0; y < ny_; ++y) {
    const double* s = sums_.data() + zu(y) * 7;
    tot += s[0];
    cx += s[1];
    cy += s[2];
    cz += s[3];
    pr += s[4];
    pg += s[5];
    pb += s[6];
  }
  const double vol = static_cast<double>(cell) * cell * cell;
  const float fire_on = tot > 0.0 && o_.fire_light > 0.f ? 1.f : 0.f;
  const float fcx = tot > 0 ? static_cast<float>(cx / tot) : 0.f, fcy = tot > 0 ? static_cast<float>(cy / tot) : 0.f, fcz = tot > 0 ? static_cast<float>(cz / tot) : 0.f;
  const float k4pi = o_.fire_light / (4.f * 3.14159265f);
  const std::array<float, 3> P{static_cast<float>(pr * vol) * k4pi * fire_on, static_cast<float>(pg * vol) * k4pi * fire_on, static_cast<float>(pb * vol) * k4pi * fire_on};
  // 4. light scattered by soot: the key light (shadowed), the ambient light and the fire's light
  parallel_for(pool, ny_, [&](int y) {
    for (int z = 0; z < nz_; ++z) {
      for (int x = 0; x < nx_; ++x) {
        const std::size_t i = (zu(y) * zu(nz_) + zu(z)) * zu(nx_) + zu(x);
        const float D = std::max(0.f, soot[i]);
        if (D <= 0.f) continue;
        const float sscat = a * ks * D;
        const float dx = (fl(x) - fcx) * cell, dy = (fl(y) - fcy) * cell, dz = (fl(z) - fcz) * cell;
        const float att = 1.f / (dx * dx + dy * dy + dz * dz + 0.0025f);
        float* v = vol_.data() + 4 * i;
        for (int c = 0; c < 3; ++c) v[1 + c] += sscat * (o_.light_rgb[zu(c)] * tlight_[i] + o_.ambient_rgb[zu(c)] + P[zu(c)] * att);
      }
    }
  });
  // 5. the largest extinction over each macro cell and a cell around it (what a trilinear sample inside it can read)
  parallel_for(pool, my_, [&](int my) {
    for (int mz = 0; mz < mz_; ++mz) {
      for (int mx = 0; mx < mx_; ++mx) {
        float m = 0.f;
        for (int y = std::max(0, my * kMacro - 1); y <= std::min(ny_ - 1, (my + 1) * kMacro); ++y) {
          for (int z = std::max(0, mz * kMacro - 1); z <= std::min(nz_ - 1, (mz + 1) * kMacro); ++z) {
            const float* s = sigma_.data() + (zu(y) * zu(nz_) + zu(z)) * zu(nx_);
            for (int x = std::max(0, mx * kMacro - 1); x <= std::min(nx_ - 1, (mx + 1) * kMacro); ++x) m = std::max(m, s[x]);
          }
        }
        macro_[(zu(my) * zu(mz_) + zu(mz)) * zu(mx_) + zu(mx)] = m;
      }
    }
  });
}

bool invertible(const Camera& cam) {
  for (const float x : cam.view) {
    if (!std::isfinite(x)) return false;
  }
  for (const float x : cam.proj) {
    if (!std::isfinite(x)) return false;
  }
  try {
    const Mat inv = inverse(mul(cam.proj, cam.view));
    for (const double x : inv) {
      if (!std::isfinite(x)) return false;
    }
    return true;
  } catch (const std::invalid_argument&) {
    return false;
  }
}

template <bool kTone>
void Renderer::render_rows(const std::array<double, 16>& inv, int width, int height, int y0, int y1, void* out, std::size_t stride) const {
  const float N = fl(nx_), zoff = 0.5f * fl(nz_) / N;
  const float h = o_.step;  // step in cells
  const float lim_x = fl(nx_ - 1), lim_y = fl(ny_ - 1), lim_z = fl(nz_ - 1);
  for (int py = y0; py < y1; ++py) {
    for (int px = 0; px < width; ++px) {
      const double ndx = (px + 0.5) / width * 2.0 - 1.0, ndy = 1.0 - (py + 0.5) / height * 2.0;
      const auto a = unproject(inv, ndx, ndy, -1.0), b = unproject(inv, ndx, ndy, 1.0);
      // the ray in cell coordinates (x across from the box's left face, y up from the floor, z from its back face)
      float o[3] = {static_cast<float>((a[0] + 0.5) * N), static_cast<float>(a[1] * N), static_cast<float>((a[2] + zoff) * N)};
      float d[3] = {static_cast<float>(b[0] - a[0]), static_cast<float>(b[1] - a[1]), static_cast<float>(b[2] - a[2])};
      const float dl = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
      for (float& c : d) c /= dl;
      float t0 = 0.f, t1 = 1e30f;
      const float hi[3] = {N, fl(ny_), fl(nz_)};
      bool hit = true;
      for (int k = 0; k < 3 && hit; ++k) {
        if (std::abs(d[k]) < 1e-12f) {
          if (o[k] < 0.f || o[k] > hi[k]) hit = false;
          continue;
        }
        float ta = (0.f - o[k]) / d[k], tb = (hi[k] - o[k]) / d[k];
        if (ta > tb) std::swap(ta, tb);
        t0 = std::max(t0, ta);
        t1 = std::min(t1, tb);
        if (t0 >= t1) hit = false;
      }
      float L[3] = {0.f, 0.f, 0.f}, T = 1.f;
      if (hit) {
        const float first = t0 + 0.5f * h;
        float t = first;
        while (t - 0.5f * h < t1) {  // each sample stands for [t - h/2, t + h/2]
          const float p[3] = {o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]};
          const int mxi = std::clamp(static_cast<int>(p[0]) / kMacro, 0, mx_ - 1), myi = std::clamp(static_cast<int>(p[1]) / kMacro, 0, my_ - 1),
                    mzi = std::clamp(static_cast<int>(p[2]) / kMacro, 0, mz_ - 1);
          if (macro_[(zu(myi) * zu(mz_) + zu(mzi)) * zu(mx_) + zu(mxi)] <= 0.f) {
            // empty: jump to the first step past this macro cell
            const int mi[3] = {mxi, myi, mzi};
            float te = t1;
            for (int k = 0; k < 3; ++k) {
              if (std::abs(d[k]) < 1e-12f) continue;
              const float face = fl((mi[k] + (d[k] > 0.f ? 1 : 0)) * kMacro);
              te = std::min(te, (face - o[k]) / d[k]);
            }
            const float jump = std::max(1.f, std::ceil((te - t) / h + 1e-4f));
            t += jump * h;
            continue;
          }
          const float cx = std::clamp(p[0] - 0.5f, 0.f, lim_x), cy = std::clamp(p[1] - 0.5f, 0.f, lim_y), cz = std::clamp(p[2] - 0.5f, 0.f, lim_z);
          const int x0 = std::min(static_cast<int>(cx), nx_ - 2), y0i = std::min(static_cast<int>(cy), ny_ - 2), z0 = std::min(static_cast<int>(cz), nz_ - 2);
          const float fx = cx - fl(x0), fy = cy - fl(y0i), fz = cz - fl(z0);
          const float* c000 = vol_.data() + 4 * ((zu(y0i) * zu(nz_) + zu(z0)) * zu(nx_) + zu(x0));
          const std::size_t sx = 4, sz = 4 * zu(nx_), sy = 4 * zu(nx_) * zu(nz_);
          float s[4];
          for (int c = 0; c < 4; ++c) {
            const float* q = c000 + c;
            const float a00 = q[0] + fx * (q[sx] - q[0]), a10 = q[sz] + fx * (q[sz + sx] - q[sz]);
            const float a01 = q[sy] + fx * (q[sy + sx] - q[sy]), a11 = q[sy + sz] + fx * (q[sy + sz + sx] - q[sy + sz]);
            const float b0 = a00 + fz * (a10 - a00), b1 = a01 + fz * (a11 - a01);
            s[c] = b0 + fy * (b1 - b0);
          }
          if (s[0] > 0.f) {
            // the sample stands for the segment [t - h/2, t + h/2], clipped at the box's far side
            const float seg = std::min(h, t1 - (t - 0.5f * h)) / N;
            const float tau = s[0] * seg, e = std::exp(-tau);
            const float w = tau > 1e-4f ? T * (1.f - e) / s[0] : T * seg * (1.f - 0.5f * tau);
            L[0] += w * s[1];
            L[1] += w * s[2];
            L[2] += w * s[3];
            T *= e;
            if (T < 1e-4f) {
              T = 0.f;
              break;
            }
          }
          t += h;
        }
      }
      if constexpr (kTone) {
        std::uint8_t* q = static_cast<std::uint8_t*>(out) + zu(py) * stride + zu(px) * 4;
        for (int c = 0; c < 3; ++c) q[c] = to8(srgb_encode(1.f - std::exp(-o_.exposure * L[c])));
        q[3] = to8(1.f - T);
      } else {
        float* q = static_cast<float*>(out) + (zu(py) * zu(width) + zu(px)) * 4;
        q[0] = L[0];
        q[1] = L[1];
        q[2] = L[2];
        q[3] = 1.f - T;
      }
    }
  }
}

void Renderer::render(const Camera& cam, int width, int height, std::uint8_t* rgba, std::size_t stride, Pool* pool) const {
  if (width < 1 || height < 1 || !rgba || stride < zu(width) * 4) throw std::invalid_argument("render3d: bad picture size or buffer");
  const Mat inv = inverse(mul(cam.proj, cam.view));
  constexpr int kRows = 4;
  parallel_for(pool, (height + kRows - 1) / kRows, [&](int b) { render_rows<true>(inv, width, height, b * kRows, std::min(height, (b + 1) * kRows), rgba, stride); });
}

void Renderer::render_linear(const Camera& cam, int width, int height, float* rgba, Pool* pool) const {
  if (width < 1 || height < 1 || !rgba) throw std::invalid_argument("render3d: bad picture size or buffer");
  const Mat inv = inverse(mul(cam.proj, cam.view));
  constexpr int kRows = 4;
  parallel_for(pool, (height + kRows - 1) / kRows, [&](int b) { render_rows<false>(inv, width, height, b * kRows, std::min(height, (b + 1) * kRows), rgba, 0); });
}

}  // namespace nfx::render3d
