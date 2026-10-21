// Study N's 3D simulator (include/neuralfx/sim3d.hpp; docs/STUDY_3D.md). Our own solver, tuned by eye, not a validated
// fire code. Every operation is a loop over cells whose result depends only on the cell, so splitting the slabs over
// threads never changes a value: the same fields on any number of threads.
#include <neuralfx/sim3d.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace nfx::sim3d {

std::string_view effect_name(Effect e) {
  switch (e) {
    case Effect::fire: return "fire";
    case Effect::smoke: return "smoke";
    case Effect::smoke_r: return "smoke_r";
  }
  std::unreachable();
}

bool parse_effect(std::string_view text, Effect& out) {
  for (const Effect e : {Effect::fire, Effect::smoke, Effect::smoke_r}) {
    if (effect_name(e) == text) {
      out = e;
      return true;
    }
  }
  return false;
}

Forcing3 forcing(Effect e) {
  Forcing3 f;
  f.flicker_rate = e == Effect::fire ? 2.6f : 1.4f;
  return f;
}

namespace {

float fl(int i) { return static_cast<float>(i); }
std::size_t zu(int i) { return static_cast<std::size_t>(i); }
float clamp01(float x) { return std::clamp(x, 0.f, 1.f); }
float smoothstep(float a, float b, float x) {
  const float t = clamp01((x - a) / (b - a));
  return t * t * (3.f - 2.f * t);
}

// Per-effect constants, in solver cells and seconds for a 128-cell width (lengths and speeds scaled by width / 128).
struct Tune {
  float buoyancy, soot_weight, cooling, soot_decay, wind_speed, wind_drag, confinement, curl_force, vel_damp, expansion;
};

Tune tune(const Params& p) {
  const float turb = p.turbulence, I = p.intensity;
  switch (p.effect) {
    case Effect::fire: return {260.f, 8.f, 0.9f, 0.6f, 90.f, 2.5f, 1.2f + 6.f * turb, 10.f + 60.f * turb, 0.15f, 1.5f + 1.5f * I};
    // smoke: warm, not hot (it does not glow), so more buoyancy per unit of heat; a stronger push at the source
    case Effect::smoke: return {420.f, 0.6f, 0.30f, 0.30f, 70.f, 2.0f, 0.8f + 4.f * turb, 6.f + 40.f * turb, 0.10f, 0.f};
    // study R's smoke (docs/STUDY_R.md §1): study N's without its sinks. Warm smoke a few tens of kelvin above the air
    // does not cool or fade in seconds, and a free plume has no drag on it: so no cooling, no soot decay and no velocity
    // damping, which conserves the plume's buoyancy flux as Morton, Taylor and Turner's theory requires.
    case Effect::smoke_r: return {420.f, 0.6f, 0.f, 0.f, 70.f, 2.0f, 0.8f + 4.f * turb, 6.f + 40.f * turb, 0.f, 0.f};
  }
  std::unreachable();
}

}  // namespace

void Fluid::Grid::init(int x, int y, int z) {
  nx = x;
  ny = y;
  nz = z;
  stride_z = zu(x + 2);
  stride_y = zu(x + 2) * zu(z + 2);
  v.assign(stride_y * zu(y + 2), 0.f);
}

float Fluid::Grid::sample(float x, float y, float z) const {
  x = std::clamp(x, 0.5f, fl(nx) + 0.5f);
  y = std::clamp(y, 0.5f, fl(ny) + 0.5f);
  z = std::clamp(z, 0.5f, fl(nz) + 0.5f);
  const int x0 = static_cast<int>(x), y0 = static_cast<int>(y), z0 = static_cast<int>(z);
  const float fx = x - fl(x0), fy = y - fl(y0), fz = z - fl(z0);
  const float* p = v.data() + zu(y0) * stride_y + zu(z0) * stride_z + zu(x0);
  const float c00 = p[0] + fx * (p[1] - p[0]);
  const float c10 = p[stride_z] + fx * (p[stride_z + 1] - p[stride_z]);
  const float c01 = p[stride_y] + fx * (p[stride_y + 1] - p[stride_y]);
  const float c11 = p[stride_y + stride_z] + fx * (p[stride_y + stride_z + 1] - p[stride_y + stride_z]);
  const float c0 = c00 + fz * (c10 - c00), c1 = c01 + fz * (c11 - c01);
  return c0 + fy * (c1 - c0);
}

template <class F>
void Fluid::slabs(F&& fn) {
  pool_->run(ny_, [&](int i) { fn(i + 1); });
}

Fluid::Fluid(const Params& p) : p_(p), nx_(p.width), ny_(p.height), nz_(p.width), s_(fl(p.width) / p.span), forcing_(forcing(p.effect)) {
  if (p.width < 8 || p.width > 512 || p.height < 8 || p.height > 768) throw std::invalid_argument("sim3d: size out of range");
  if (!(p.span >= 16.f && p.span <= 4096.f) || !(p.source_x >= 0.f && p.source_x <= 1.f)) throw std::invalid_argument("sim3d: bad span or source position");
  if (p.substeps < 1 || !(p.fps > 0.f) || p.pressure_iters < 0) throw std::invalid_argument("sim3d: bad substeps, fps or iterations");
  for (Grid* g : {&u_, &v_, &w_, &temp_, &soot_, &pressure_, &div_, &ta_, &tb_, &tc_, &wx_, &wy_, &wz_, &wn_}) g->init(nx_, ny_, nz_);
  curl_.resize(forcing_, p.span, p.span * fl(ny_) / fl(nx_), p.span);
  pool_ = std::make_unique<Pool>(std::max(1, p.threads));
}

Fluid::~Fluid() = default;

State Fluid::state() const {
  State s;
  s.nx = nx_;
  s.ny = ny_;
  s.nz = nz_;
  s.frame = frame_;
  s.time = time_;
  const auto inside = [this](const Grid& g) {
    std::vector<float> o(zu(nx_) * zu(ny_) * zu(nz_));
    for (int y = 1; y <= ny_; ++y) {
      for (int z = 1; z <= nz_; ++z) {
        for (int x = 1; x <= nx_; ++x) o[(zu(y - 1) * zu(nz_) + zu(z - 1)) * zu(nx_) + zu(x - 1)] = g.at(x, y, z);
      }
    }
    return o;
  };
  s.u = inside(u_);
  s.v = inside(v_);
  s.w = inside(w_);
  s.temp = inside(temp_);
  s.soot = inside(soot_);
  s.pressure = inside(pressure_);
  return s;
}

void Fluid::set_state(const State& s) {
  const std::size_t n = zu(nx_) * zu(ny_) * zu(nz_);
  if (s.nx != nx_ || s.ny != ny_ || s.nz != nz_ || s.u.size() != n || s.v.size() != n || s.w.size() != n || s.temp.size() != n ||
      s.soot.size() != n || s.pressure.size() != n) {
    throw std::invalid_argument("sim3d: state does not match the solver's size");
  }
  const auto put = [this](Grid& g, const std::vector<float>& from) {
    for (int y = 1; y <= ny_; ++y) {
      for (int z = 1; z <= nz_; ++z) {
        for (int x = 1; x <= nx_; ++x) g.at(x, y, z) = from[(zu(y - 1) * zu(nz_) + zu(z - 1)) * zu(nx_) + zu(x - 1)];
      }
    }
  };
  put(u_, s.u);
  put(v_, s.v);
  put(w_, s.w);
  put(temp_, s.temp);
  put(soot_, s.soot);
  put(pressure_, s.pressure);
  for (int z = 1; z <= nz_; ++z) {
    for (int x = 1; x <= nx_; ++x) pressure_.at(x, 0, z) = pressure_.at(x, 1, z);
  }
  velocity_border();
  frame_ = s.frame;
  time_ = s.time;
}

void Fluid::fields(std::span<float> temp, std::span<float> soot) const {
  const std::size_t n = zu(nx_) * zu(ny_) * zu(nz_);
  if (temp.size() < n || soot.size() < n) throw std::invalid_argument("sim3d: field buffers too small");
  pool_->run(ny_, [&](int yi) {
    const int y = yi + 1;
    for (int z = 1; z <= nz_; ++z) {
      const std::size_t o = (zu(y - 1) * zu(nz_) + zu(z - 1)) * zu(nx_);
      for (int x = 1; x <= nx_; ++x) {
        temp[o + zu(x - 1)] = temp_.at(x, y, z);
        soot[o + zu(x - 1)] = soot_.at(x, y, z);
      }
    }
  });
}

void Fluid::velocity(std::span<float> uvw) const {
  const std::size_t n = zu(nx_) * zu(ny_) * zu(nz_);
  if (uvw.size() < 3 * n) throw std::invalid_argument("sim3d: velocity buffer too small");
  for (int y = 1; y <= ny_; ++y) {
    for (int z = 1; z <= nz_; ++z) {
      for (int x = 1; x <= nx_; ++x) {
        const std::size_t i = (zu(y - 1) * zu(nz_) + zu(z - 1)) * zu(nx_) + zu(x - 1);
        uvw[3 * i] = u_.at(x, y, z);
        uvw[3 * i + 1] = v_.at(x, y, z);
        uvw[3 * i + 2] = w_.at(x, y, z);
      }
    }
  }
}

std::array<float, 2> Fluid::buoyancy_coefficients() const {
  const Tune t = tune(p_);
  return {t.buoyancy * s_, t.soot_weight * s_};
}

void Fluid::velocity_border() {
  // Sides and top: outflow (the neighbour's value). Floor: closed, slip along it (u, w copied, v reflected).
  for (int y = 0; y <= ny_ + 1; ++y) {
    const int yc = std::clamp(y, 1, ny_);
    for (int z = 0; z <= nz_ + 1; ++z) {
      const int zc = std::clamp(z, 1, nz_);
      for (int x = 0; x <= nx_ + 1; ++x) {
        const bool border = x == 0 || z == 0 || y == 0 || x == nx_ + 1 || y == ny_ + 1 || z == nz_ + 1;
        if (!border) {
          x = nx_;  // skip the inside of the row
          continue;
        }
        const int xc = std::clamp(x, 1, nx_);
        u_.at(x, y, z) = u_.at(xc, yc, zc);
        w_.at(x, y, z) = w_.at(xc, yc, zc);
        v_.at(x, y, z) = y == 0 ? -v_.at(xc, 1, zc) : v_.at(xc, yc, zc);
      }
    }
  }
}

void Fluid::set_controls(float intensity, float wind, float turbulence) {
  p_.intensity = intensity;
  p_.wind = wind;
  p_.turbulence = turbulence;
}

void Fluid::step_frame() {
  const float dt = 1.f / (p_.fps * fl(p_.substeps));
  for (int s = 0; s < p_.substeps; ++s) {
    steps_ = frame_ * p_.substeps + s;  // the flux form's sweep order follows it (a restored state goes on the same way)
    step(dt);
  }
  ++frame_;
}

void Fluid::step(float dt) {
  add_sources(dt);
  add_forces(dt);
  advect_velocity(dt);
  project();
  const bool flux = p_.scalars == 1 || (p_.scalars < 0 && p_.effect == Effect::smoke_r);
  if (flux) {
    advect_scalar_flux(temp_, dt);
    advect_scalar_flux(soot_, dt);
  } else {
    advect_scalar(temp_, dt);
    advect_scalar(soot_, dt);
  }
  const Tune t = tune(p_);
  const float cool = std::exp(-t.cooling * dt), decay = std::exp(-t.soot_decay * dt);
  slabs([&](int y) {
    for (int z = 1; z <= nz_; ++z) {
      for (int x = 1; x <= nx_; ++x) {
        temp_.at(x, y, z) *= cool;
        soot_.at(x, y, z) *= decay;
      }
    }
  });
  time_ += dt;
}

void Fluid::add_sources(float dt) {
  const float s = s_, I = p_.intensity;
  const bool fire = p_.effect == Effect::fire;
  // a round burner on the floor, in world units (128 across by default; cell x has its centre at (x - 0.5) / s)
  const float cx = p_.span * p_.source_x, cz = p_.span * 0.5f, cy = fire ? 6.f : 5.f;
  const float r = fire ? (0.10f + 0.08f * I) * 128.f : (0.09f + 0.07f * I) * 128.f, hh = 0.035f * 128.f;
  const int x_lo = std::max(1, static_cast<int>((cx - 1.2f * r) * s)), x_hi = std::min(nx_, static_cast<int>((cx + 1.2f * r) * s) + 2);
  const int z_lo = std::max(1, static_cast<int>((cz - 1.2f * r) * s)), z_hi = std::min(nz_, static_cast<int>((cz + 1.2f * r) * s) + 2);
  const int y_hi = std::min(ny_, static_cast<int>((cy + 1.2f * hh) * s) + 2);
  const std::uint64_t seed = p_.seed;
  pool_->run(y_hi, [&](int yi) {
    const int y = yi + 1;
    for (int z = z_lo; z <= z_hi; ++z) {
      for (int x = x_lo; x <= x_hi; ++x) {
        const float X = (fl(x) - 0.5f) / s, Y = (fl(y) - 0.5f) / s, Z = (fl(z) - 0.5f) / s;
        const float ex = (X - cx) / r, ez = (Z - cz) / r, ey = (Y - cy) / hh;
        const float shape = 1.f - smoothstep(0.6f, 1.f, std::sqrt(ex * ex + ez * ez + ey * ey));
        if (shape <= 0.f) continue;
        const float k = shape * smoothstep(-0.25f, 0.6f, flicker3(forcing_, seed, X, Y, Z, time_));
        if (fire) {
          temp_.at(x, y, z) = std::max(temp_.at(x, y, z), (0.95f + 0.55f * I) * k);
          soot_.at(x, y, z) += dt * (0.35f + 0.25f * I) * k;
          v_.at(x, y, z) = std::max(v_.at(x, y, z), (40.f + 20.f * I) * s * k);
        } else {
          temp_.at(x, y, z) = std::max(temp_.at(x, y, z), (0.30f + 0.15f * I) * k);
          soot_.at(x, y, z) = std::max(soot_.at(x, y, z), (0.7f + 0.5f * I) * k);
          v_.at(x, y, z) = std::max(v_.at(x, y, z), (55.f + 20.f * I) * s * k);
        }
      }
    }
  });
}

void Fluid::add_forces(float dt) {
  const Tune t = tune(p_);
  const float s = s_;
  const float wind = (2.f * p_.wind - 1.f) * t.wind_speed * s;
  const float wdrag = 1.f - std::exp(-t.wind_drag * dt), damp = std::exp(-t.vel_damp * dt);
  const float conf = t.confinement * s, amp = t.curl_force * s;
  curl_.build(forcing_, p_.seed, time_, pool_.get());
  // vorticity (central differences) and its magnitude
  slabs([&](int y) {
    for (int z = 1; z <= nz_; ++z) {
      for (int x = 1; x <= nx_; ++x) {
        const float ox = 0.5f * ((w_.at(x, y + 1, z) - w_.at(x, y - 1, z)) - (v_.at(x, y, z + 1) - v_.at(x, y, z - 1)));
        const float oy = 0.5f * ((u_.at(x, y, z + 1) - u_.at(x, y, z - 1)) - (w_.at(x + 1, y, z) - w_.at(x - 1, y, z)));
        const float oz = 0.5f * ((v_.at(x + 1, y, z) - v_.at(x - 1, y, z)) - (u_.at(x, y + 1, z) - u_.at(x, y - 1, z)));
        wx_.at(x, y, z) = ox;
        wy_.at(x, y, z) = oy;
        wz_.at(x, y, z) = oz;
        wn_.at(x, y, z) = std::sqrt(ox * ox + oy * oy + oz * oz);
      }
    }
  });
  // forces: buoyancy, confinement, curl noise where there is material, wind drag, damping (written to ta, tb, tc, then
  // swapped in, so every cell reads the old velocity)
  slabs([&](int y) {
    for (int z = 1; z <= nz_; ++z) {
      for (int x = 1; x <= nx_; ++x) {
        const float T = temp_.at(x, y, z), D = soot_.at(x, y, z);
        float fx = 0.f, fy = t.buoyancy * s * T - t.soot_weight * s * D, fz = 0.f;
        if (x > 1 && x < nx_ && y > 1 && y < ny_ && z > 1 && z < nz_) {
          const float gx = 0.5f * (wn_.at(x + 1, y, z) - wn_.at(x - 1, y, z));
          const float gy = 0.5f * (wn_.at(x, y + 1, z) - wn_.at(x, y - 1, z));
          const float gz = 0.5f * (wn_.at(x, y, z + 1) - wn_.at(x, y, z - 1));
          const float len = std::sqrt(gx * gx + gy * gy + gz * gz) + 1e-5f;
          const float Nx = gx / len, Ny = gy / len, Nz = gz / len;
          const float ox = wx_.at(x, y, z), oy = wy_.at(x, y, z), oz = wz_.at(x, y, z);
          fx += conf * (Ny * oz - Nz * oy);
          fy += conf * (Nz * ox - Nx * oz);
          fz += conf * (Nx * oy - Ny * ox);
        }
        const float mat = std::min(1.f, 2.f * (T + D));
        if (mat > 0.01f) {
          const auto c = curl_.sample((fl(x) - 0.5f) / s, (fl(y) - 0.5f) / s, (fl(z) - 0.5f) / s);
          fx += mat * amp * c[0];
          fy += mat * amp * c[1];
          fz += mat * amp * c[2];
        }
        float u = (u_.at(x, y, z) + dt * fx) * damp;
        const float v = (v_.at(x, y, z) + dt * fy) * damp;
        const float w = (w_.at(x, y, z) + dt * fz) * damp;
        u += wdrag * (wind - u) * mat;  // air drags material towards the wind speed
        ta_.at(x, y, z) = u;
        tb_.at(x, y, z) = v;
        tc_.at(x, y, z) = w;
      }
    }
  });
  std::swap(u_.v, ta_.v);
  std::swap(v_.v, tb_.v);
  std::swap(w_.v, tc_.v);
  velocity_border();
}

void Fluid::project() {
  const Tune t = tune(p_);
  // the divergence; combustion: hot gas expands, so the solve targets a divergence of `expansion` (1/s) there
  slabs([&](int y) {
    for (int z = 1; z <= nz_; ++z) {
      for (int x = 1; x <= nx_; ++x) {
        float d = -0.5f * (u_.at(x + 1, y, z) - u_.at(x - 1, y, z) + v_.at(x, y + 1, z) - v_.at(x, y - 1, z) + w_.at(x, y, z + 1) - w_.at(x, y, z - 1));
        if (t.expansion > 0.f) d += t.expansion * smoothstep(0.4f, 1.0f, temp_.at(x, y, z));
        div_.at(x, y, z) = d;
      }
    }
  });
  // Red-black Gauss-Seidel with over-relaxation, warm-started from the last solve. p = 0 in the border of the sides
  // and the top (never written); the floor's border copies the cell above it (no flow through the floor).
  constexpr float omega = 1.7f, sixth = 1.f / 6.f;
  for (int it = 0; it < p_.pressure_iters; ++it) {
    for (const int colour : {0, 1}) {
      slabs([&](int y) {
        for (int z = 1; z <= nz_; ++z) {
          for (int x = 1 + ((y + z + colour) & 1); x <= nx_; x += 2) {
            const float g = sixth * (div_.at(x, y, z) + pressure_.at(x - 1, y, z) + pressure_.at(x + 1, y, z) + pressure_.at(x, y - 1, z) +
                                     pressure_.at(x, y + 1, z) + pressure_.at(x, y, z - 1) + pressure_.at(x, y, z + 1));
            float& p = pressure_.at(x, y, z);
            p += omega * (g - p);
          }
        }
      });
      for (int z = 1; z <= nz_; ++z) {
        for (int x = 1; x <= nx_; ++x) pressure_.at(x, 0, z) = pressure_.at(x, 1, z);
      }
    }
  }
  slabs([&](int y) {
    for (int z = 1; z <= nz_; ++z) {
      for (int x = 1; x <= nx_; ++x) {
        u_.at(x, y, z) -= 0.5f * (pressure_.at(x + 1, y, z) - pressure_.at(x - 1, y, z));
        v_.at(x, y, z) -= 0.5f * (pressure_.at(x, y + 1, z) - pressure_.at(x, y - 1, z));
        w_.at(x, y, z) -= 0.5f * (pressure_.at(x, y, z + 1) - pressure_.at(x, y, z - 1));
      }
    }
  });
  velocity_border();
}

void Fluid::advect_velocity(float dt) {
  slabs([&](int y) {
    for (int z = 1; z <= nz_; ++z) {
      for (int x = 1; x <= nx_; ++x) {
        const float bx = fl(x) - dt * u_.at(x, y, z), by = fl(y) - dt * v_.at(x, y, z), bz = fl(z) - dt * w_.at(x, y, z);
        ta_.at(x, y, z) = u_.sample(bx, by, bz);
        tb_.at(x, y, z) = v_.sample(bx, by, bz);
        tc_.at(x, y, z) = w_.sample(bx, by, bz);
      }
    }
  });
  std::swap(u_.v, ta_.v);
  std::swap(v_.v, tb_.v);
  std::swap(w_.v, tc_.v);
  velocity_border();
}

void Fluid::advect_scalar(Grid& q, float dt) {
  // MacCormack: a forward step, a backward step, half the round-trip error added back, clamped to the corners of the
  // forward step's sample so no new extrema appear. The scalars' border stays zero (nothing flows in).
  slabs([&](int y) {
    for (int z = 1; z <= nz_; ++z) {
      for (int x = 1; x <= nx_; ++x) ta_.at(x, y, z) = q.sample(fl(x) - dt * u_.at(x, y, z), fl(y) - dt * v_.at(x, y, z), fl(z) - dt * w_.at(x, y, z));
    }
  });
  slabs([&](int y) {
    for (int z = 1; z <= nz_; ++z) {
      for (int x = 1; x <= nx_; ++x) tb_.at(x, y, z) = ta_.sample(fl(x) + dt * u_.at(x, y, z), fl(y) + dt * v_.at(x, y, z), fl(z) + dt * w_.at(x, y, z));
    }
  });
  slabs([&](int y) {
    for (int z = 1; z <= nz_; ++z) {
      for (int x = 1; x <= nx_; ++x) {
        const float bx = std::clamp(fl(x) - dt * u_.at(x, y, z), 0.5f, fl(nx_) + 0.5f);
        const float by = std::clamp(fl(y) - dt * v_.at(x, y, z), 0.5f, fl(ny_) + 0.5f);
        const float bz = std::clamp(fl(z) - dt * w_.at(x, y, z), 0.5f, fl(nz_) + 0.5f);
        const int x0 = static_cast<int>(bx), y0 = static_cast<int>(by), z0 = static_cast<int>(bz);
        float lo = q.at(x0, y0, z0), hi = lo;
        for (int c = 1; c < 8; ++c) {
          const float a = q.at(x0 + (c & 1), y0 + ((c >> 2) & 1), z0 + ((c >> 1) & 1));
          lo = std::min(lo, a);
          hi = std::max(hi, a);
        }
        tc_.at(x, y, z) = std::max(0.f, std::clamp(ta_.at(x, y, z) + 0.5f * (q.at(x, y, z) - tb_.at(x, y, z)), lo, hi));
      }
    }
  });
  slabs([&](int y) {
    for (int z = 1; z <= nz_; ++z) {
      for (int x = 1; x <= nx_; ++x) q.at(x, y, z) = tc_.at(x, y, z);
    }
  });
}

void Fluid::advect_scalar_flux(Grid& q, float dt) {
  // Flux form: one sweep per axis (x, y, z on even steps, z, y, x on odd ones), each q_i -= dt (F_i+1/2 - F_i-1/2) with
  // the face's velocity the mean of its two cells' and the face's value upwind, second order with the monotonised
  // central limiter (Sweby's form, so the Courant number enters). A face's flux is computed the same way from either
  // side, so what leaves one cell enters the next exactly. The border stays zero: inflow at an open side brings
  // nothing, outflow carries the edge cell's value out; the floor's face has no velocity (v is reflected there).
  // Faces next to the border fall back to first order (there is no cell beyond it).
  const auto mc = [](float r) { return std::max(0.f, std::min({2.f * r, 0.5f * (1.f + r), 2.f})); };
  const bool forward = (steps_ & 1) == 0;
  for (int k = 0; k < 3; ++k) {
    const int axis = forward ? k : 2 - k;  // 0 x, 1 y, 2 z
    const Grid& c = axis == 0 ? u_ : axis == 1 ? v_ : w_;
    const std::size_t st = axis == 0 ? 1 : axis == 1 ? q.stride_y : q.stride_z;
    const int n = axis == 0 ? nx_ : axis == 1 ? ny_ : nz_;
    // the largest Courant number of the sweep (a maximum: the same on any number of threads)
    std::vector<float> slab_max(zu(ny_) + 1, 0.f);
    slabs([&](int y) {
      float m = 0.f;
      for (int z = 1; z <= nz_; ++z) {
        for (int x = 1; x <= nx_; ++x) m = std::max(m, std::abs(c.at(x, y, z)));
      }
      slab_max[zu(y)] = m;
    });
    const float cmax = *std::max_element(slab_max.begin(), slab_max.end()) * dt;
    const int sub = std::max(1, static_cast<int>(std::ceil(cmax / 0.95f)));
    const float h = dt / fl(sub);
    for (int s = 0; s < sub; ++s) {
      // the flux through the face between cell a (coordinate ia along the axis, 0..n) and the next
      const auto face = [&](std::size_t a, int ia) {
        const float* qq = q.v.data();
        const float uf = 0.5f * (c.v[a] + c.v[a + st]), nu = std::abs(uf) * h;
        float qf;
        if (uf >= 0.f) {
          const float d = qq[a + st] - qq[a];
          float phi = 0.f;
          if (ia >= 1 && d != 0.f) phi = mc((qq[a] - qq[a - st]) / d);
          qf = qq[a] + 0.5f * (1.f - nu) * phi * d;
        } else {
          const float d = qq[a] - qq[a + st];
          float phi = 0.f;
          if (ia <= n - 1 && d != 0.f) phi = mc((qq[a + st] - qq[a + 2 * st]) / d);
          qf = qq[a + st] + 0.5f * (1.f - nu) * phi * d;
        }
        return uf * qf;
      };
      slabs([&](int y) {
        for (int z = 1; z <= nz_; ++z) {
          for (int x = 1; x <= nx_; ++x) {
            const std::size_t i = zu(y) * q.stride_y + zu(z) * q.stride_z + zu(x);
            const int ic = axis == 0 ? x : axis == 1 ? y : z;
            const float out = face(i, ic), in = face(i - st, ic - 1);
            ta_.v[i] = std::max(0.f, q.v[i] - h * (out - in));
          }
        }
      });
      slabs([&](int y) {
        for (int z = 1; z <= nz_; ++z) {
          for (int x = 1; x <= nx_; ++x) q.at(x, y, z) = ta_.at(x, y, z);
        }
      });
    }
  }
}

}  // namespace nfx::sim3d
