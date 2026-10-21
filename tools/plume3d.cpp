// Study R (docs/STUDY_R.md): the 3D solver's smoke plume against Morton, Taylor and Turner's theory (tools/plume3d.hpp).
#include "plume3d.hpp"

#include "label_font.hpp"

#include <neuralfx/metrics.hpp>
#include <neuralfx/render3d.hpp>
#include <neuralfx/sim3d.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <format>
#include <fstream>
#include <limits>
#include <map>
#include <numbers>
#include <print>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace nfx::tools::plume3d {

namespace fs = std::filesystem;

namespace {

std::size_t zu(int v) { return static_cast<std::size_t>(v); }
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// One run's time averages (STUDY_R.md §1.3). Radial bin j holds the cells whose centres lie [j, j + 1) cells from the
// source's axis; its sums are over those cells of the time means. Heights are cell rows from the floor (row y's centre
// is y + 0.5 cells up).
struct Profile {
  std::int32_t nx = 0, ny = 0, nz = 0, frames = 0, nb = 0, effect = 0;
  float s = 0, cx = 0, cz = 0, kT = 0, kD = 0;
  float intensity = 0, wind = 0, turbulence = 0, span = 0, source_x = 0, spin = 0, seconds = 0, wind_speed = 0;
  std::uint64_t seed = 0;
  std::vector<double> cnt, r, v, g, d, t, vg, vv, vd;     // ny * nb
  std::vector<double> F, Fd, Q, M, heat, soot, xd, zd, ud;  // ny: sums over the whole plane
  std::vector<double> slice_v, slice_d, slice_t;           // nx * ny: the plane through the axis (z = cz)
  std::vector<double> depth_d, depth_t, depth_u;           // nx * ny: summed over depth (z)
  std::vector<double> div, div_abs;                        // ny: sums over the plane of soot x and heat x the projected flow's divergence (central differences, 1/s)
  // ny, the buoyancy budget (§1.4; files from before it have none): the buoyancy g' carried out through the four side
  // faces of each row, the sum over the row's edge cells of max(0, outward velocity) g' (cells^4 / s^3 per row, time
  // mean); and each plane's sum of g' at the first and at the last averaged frame (what the box stores)
  std::vector<double> side, b_first, b_last;
  // study R2 (docs/STUDY_R2.md §1; files from before it have none): ny, each plane's sum of the projection's pressure
  // (cells^2 / s^2, time mean: the pressure's part of the momentum budget); and the vertical momentum budget of the fit's
  // slab per operator over the averaged frames (sim3d::MomentumAudit: y_lo, y_hi, sources, buoyancy, confinement, curl,
  // damping, advection, projection, seconds, flux_bottom, flux_top, flux_sides)
  std::vector<double> P, audit;
};

template <class T>
void put(std::ofstream& o, const T& v) {
  o.write(reinterpret_cast<const char*>(&v), sizeof(v));
}
void putv(std::ofstream& o, const std::vector<double>& v) {
  const std::uint64_t n = v.size();
  put(o, n);
  o.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(n * sizeof(double)));
}
template <class T>
void get(std::ifstream& i, T& v) {
  i.read(reinterpret_cast<char*>(&v), sizeof(v));
}
void getv(std::ifstream& i, std::vector<double>& v) {
  std::uint64_t n = 0;
  get(i, n);
  if (n > (1u << 26)) throw std::runtime_error("plume profile: bad size");
  v.resize(n);
  i.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(double)));
}

constexpr std::uint32_t kMagic = 0x52504C4Du;  // "MLPR"

template <class Fn>
void each_vec(Profile& p, Fn&& fn) {
  for (auto* v : {&p.cnt, &p.r, &p.v, &p.g, &p.d, &p.t, &p.vg, &p.vv, &p.vd, &p.F, &p.Fd, &p.Q, &p.M, &p.heat, &p.soot, &p.xd, &p.zd, &p.ud, &p.slice_v,
                  &p.slice_d, &p.slice_t, &p.depth_d, &p.depth_t, &p.depth_u, &p.div, &p.div_abs, &p.side, &p.b_first, &p.b_last, &p.P, &p.audit})
    fn(*v);
}

void save(const fs::path& f, Profile p) {
  fs::create_directories(f.parent_path());
  const fs::path tmp = f.string() + ".tmp";
  {
    std::ofstream o(tmp, std::ios::binary);
    put(o, kMagic);
    for (auto x : {p.nx, p.ny, p.nz, p.frames, p.nb, p.effect}) put(o, x);
    for (auto x : {p.s, p.cx, p.cz, p.kT, p.kD, p.intensity, p.wind, p.turbulence, p.span, p.source_x, p.spin, p.seconds, p.wind_speed}) put(o, x);
    put(o, p.seed);
    each_vec(p, [&](std::vector<double>& v) { putv(o, v); });
    if (!o) throw std::runtime_error("cannot write " + tmp.string());
  }
  fs::rename(tmp, f);
}

Profile load(const fs::path& f) {
  std::ifstream i(f, std::ios::binary);
  if (!i) throw std::runtime_error("cannot open " + f.string());
  std::uint32_t magic = 0;
  get(i, magic);
  if (magic != kMagic) throw std::runtime_error(f.string() + ": not a plume profile");
  Profile p;
  for (auto* x : {&p.nx, &p.ny, &p.nz, &p.frames, &p.nb, &p.effect}) get(i, *x);
  for (auto* x : {&p.s, &p.cx, &p.cz, &p.kT, &p.kD, &p.intensity, &p.wind, &p.turbulence, &p.span, &p.source_x, &p.spin, &p.seconds, &p.wind_speed}) get(i, *x);
  get(i, p.seed);
  bool optional = false;  // the divergence's sums and the budget's came later: older files end before them
  each_vec(p, [&](std::vector<double>& v) {
    if (&v == &p.div) optional = true;
    if (optional && !i.eof() && i.peek() == std::char_traits<char>::eof()) i.clear(std::ios::eofbit);
    if (i.eof()) return;
    getv(i, v);
  });
  if (i.fail()) throw std::runtime_error(f.string() + ": truncated");
  return p;
}

std::vector<std::string> split(const std::string& s, char c) {
  std::vector<std::string> out;
  std::istringstream ss(s);
  std::string x;
  while (std::getline(ss, x, c)) {
    if (!x.empty()) out.push_back(x);
  }
  return out;
}

}  // namespace

// --- the measurement -----------------------------------------------------------------------------------------------

int measure(const Args& a, const fs::path& root) {
  const std::string label = a.need("label");
  sim3d::Params p;
  if (!sim3d::parse_effect(a.str("effect", a.str("study", "r") == "r2" ? "smoke_r2" : "smoke_r"), p.effect)) throw std::invalid_argument("--effect fire|smoke|smoke_r|smoke_r2");
  p.intensity = a.f("intensity", 0.5f);
  p.wind = a.f("wind", 0.5f);
  p.turbulence = a.f("turbulence", 0.5f);
  p.width = a.i("width", 64);
  p.height = a.i("height", 96);
  p.span = a.f("span", 128.f);
  p.source_x = a.f("source-x", 0.5f);
  p.threads = a.i("threads", 2);
  p.pressure_iters = a.i("pressure-iters", p.pressure_iters);
  p.substeps = a.i("substeps", p.substeps);
  p.scalars = a.i("scalars", p.scalars);
  p.velocity = a.i("velocity", p.velocity);
  const int seeds = a.i("seeds", 6), from = a.i("from", 0);
  const std::uint64_t base = std::stoull(a.need("seed-base"));
  const float spin = a.f("spin", 6.f), seconds = a.f("seconds", 20.f);
  const fs::path dir = root / "plume" / label;
  for (int k = from; k < seeds; ++k) {
    const fs::path f = dir / std::format("seed_{:02}.bin", k);
    if (fs::exists(f)) continue;
    p.seed = base + static_cast<std::uint64_t>(k);
    const double t0 = now();
    sim3d::Fluid sim(p);
    const int nx = sim.nx(), ny = sim.ny(), nz = sim.nz();
    const float s = static_cast<float>(nx) / p.span;
    Profile pr;
    pr.nx = nx;
    pr.ny = ny;
    pr.nz = nz;
    pr.s = s;
    pr.cx = p.span * p.source_x * s;
    pr.cz = p.span * 0.5f * s;
    const auto kb = sim.buoyancy_coefficients();
    pr.kT = kb[0];
    pr.kD = kb[1];
    pr.intensity = p.intensity;
    pr.wind = p.wind;
    pr.turbulence = p.turbulence;
    pr.span = p.span;
    pr.source_x = p.source_x;
    pr.spin = spin;
    pr.seconds = seconds;
    pr.seed = p.seed;
    pr.effect = static_cast<std::int32_t>(p.effect);
    pr.wind_speed = a.f("wind-speed", 0.f);  // for the record: the preset's wind speed at this control (cells per second)
    pr.nb = static_cast<std::int32_t>(std::floor(std::min({pr.cx, static_cast<float>(nx) - pr.cx, pr.cz, static_cast<float>(nz) - pr.cz})));
    const int nb = pr.nb;
    const std::size_t pb = zu(ny) * zu(nb), pl = zu(nx) * zu(ny);
    for (auto* v : {&pr.cnt, &pr.r, &pr.v, &pr.g, &pr.d, &pr.t, &pr.vg, &pr.vv, &pr.vd}) v->assign(pb, 0.0);
    for (auto* v : {&pr.F, &pr.Fd, &pr.Q, &pr.M, &pr.heat, &pr.soot, &pr.xd, &pr.zd, &pr.ud, &pr.div, &pr.div_abs, &pr.side, &pr.b_first, &pr.b_last, &pr.P})
      v->assign(zu(ny), 0.0);
    // the momentum budget's slab (study R2): the rows whose centres lie in [audit-lo, audit-hi] cells above the floor
    sim3d::MomentumAudit audit;
    audit.y_lo = static_cast<int>(std::ceil(a.f("audit-lo", 24.f) - 0.5f));
    audit.y_hi = std::min(ny - 1, static_cast<int>(std::floor(a.f("audit-hi", 92.f) - 0.5f)));
    for (auto* v : {&pr.slice_v, &pr.slice_d, &pr.slice_t, &pr.depth_d, &pr.depth_t, &pr.depth_u}) v->assign(pl, 0.0);
    // each cell's radial bin (-1 beyond the inscribed circle), and the two rows through the axis with their weights
    std::vector<int> bin(zu(nx) * zu(nz));
    for (int z = 0; z < nz; ++z) {
      for (int x = 0; x < nx; ++x) {
        const double r = std::hypot(x + 0.5 - pr.cx, z + 0.5 - pr.cz);
        const int b = static_cast<int>(std::floor(r));
        bin[zu(z) * zu(nx) + zu(x)] = b < nb ? b : -1;
        if (b < nb) {
          for (int y = 0; y < ny; ++y) {
            pr.cnt[zu(y) * zu(nb) + zu(b)] += 1;
            pr.r[zu(y) * zu(nb) + zu(b)] += r;
          }
        }
      }
    }
    const int z0 = static_cast<int>(std::floor(pr.cz - 0.5f)), z1 = std::min(nz - 1, z0 + 1);
    const double w1 = static_cast<double>(pr.cz - 0.5f) - z0, w0 = 1.0 - w1;
    const int spin_frames = static_cast<int>(std::lround(spin * p.fps)), frames = static_cast<int>(std::lround(seconds * p.fps));
    for (int fr = 0; fr < spin_frames; ++fr) sim.step_frame();
    const std::size_t n = zu(nx) * zu(ny) * zu(nz);
    std::vector<float> T(n), D(n), U(3 * n), Pr(n);
    sim.set_audit(&audit);
    for (int fr = 0; fr < frames; ++fr) {
      sim.step_frame();
      sim.fields(T, D);
      sim.velocity(U);
      sim.pressure(Pr);
      for (std::size_t i = 0; i < n; ++i) pr.P[i / (zu(nx) * zu(nz))] += Pr[i];
      for (int y = 0; y < ny; ++y) {
        double F = 0, Fd = 0, Q = 0, M = 0, H = 0, S = 0, X = 0, Z = 0, UD = 0, SIDE = 0, B = 0;
        for (int z = 0; z < nz; ++z) {
          const double wz = z == z0 ? w0 : z == z1 ? w1 : 0.0;
          for (int x = 0; x < nx; ++x) {
            const std::size_t i = (zu(y) * zu(nz) + zu(z)) * zu(nx) + zu(x);
            const double v = U[3 * i + 1], u = U[3 * i], t = T[i], dd = D[i];
            const double g = static_cast<double>(pr.kT) * t - static_cast<double>(pr.kD) * dd;
            B += g;
            if (x == 0) SIDE += std::max(0.0, -u) * g;  // out through the side faces (an edge cell may have two)
            if (x == nx - 1) SIDE += std::max(0.0, u) * g;
            if (z == 0) SIDE += std::max(0.0, -static_cast<double>(U[3 * i + 2])) * g;
            if (z == nz - 1) SIDE += std::max(0.0, static_cast<double>(U[3 * i + 2])) * g;
            F += v * g;
            Fd += v * dd;
            Q += v;
            M += v * v;
            H += t;
            S += dd;
            X += (x + 0.5) * dd;
            Z += (z + 0.5) * dd;
            UD += u * dd;
            const int b = bin[zu(z) * zu(nx) + zu(x)];
            if (b >= 0) {
              const std::size_t j = zu(y) * zu(nb) + zu(b);
              pr.v[j] += v;
              pr.g[j] += g;
              pr.d[j] += dd;
              pr.t[j] += t;
              pr.vg[j] += v * g;
              pr.vv[j] += v * v;
              pr.vd[j] += v * dd;
            }
            const std::size_t q = zu(y) * zu(nx) + zu(x);
            if (wz > 0) {
              pr.slice_v[q] += wz * v;
              pr.slice_d[q] += wz * dd;
              pr.slice_t[q] += wz * t;
            }
            pr.depth_d[q] += dd;
            pr.depth_t[q] += t;
            pr.depth_u[q] += u * dd;
            if (x > 0 && x < nx - 1 && z > 0 && z < nz - 1 && y > 0 && y < ny - 1) {  // the solver's central differences
              const std::size_t sx = 3, sz = 3 * zu(nx), sy = 3 * zu(nx) * zu(nz);
              const double dv = 0.5 * (U[3 * i + sx] - U[3 * i - sx] + U[3 * i + 1 + sy] - U[3 * i + 1 - sy] + U[3 * i + 2 + sz] - U[3 * i + 2 - sz]);
              pr.div[zu(y)] += dd * dv;  // the soot the flow's divergence adds per second (advective form)
              pr.div_abs[zu(y)] += t * dv;
            }
          }
        }
        pr.F[zu(y)] += F;
        pr.Fd[zu(y)] += Fd;
        pr.Q[zu(y)] += Q;
        pr.M[zu(y)] += M;
        pr.heat[zu(y)] += H;
        pr.soot[zu(y)] += S;
        pr.xd[zu(y)] += X;
        pr.zd[zu(y)] += Z;
        pr.ud[zu(y)] += UD;
        pr.side[zu(y)] += SIDE;
        if (fr == 0) pr.b_first[zu(y)] = B;
        if (fr == frames - 1) pr.b_last[zu(y)] = B;
      }
    }
    sim.set_audit(nullptr);
    const double inv = 1.0 / frames;
    for (auto* v : {&pr.v, &pr.g, &pr.d, &pr.t, &pr.vg, &pr.vv, &pr.vd, &pr.F, &pr.Fd, &pr.Q, &pr.M, &pr.heat, &pr.soot, &pr.xd, &pr.zd, &pr.ud, &pr.slice_v,
                    &pr.slice_d, &pr.slice_t, &pr.depth_d, &pr.depth_t, &pr.depth_u, &pr.div, &pr.div_abs, &pr.side, &pr.P}) {
      for (double& x : *v) x *= inv;
    }
    pr.audit = {static_cast<double>(audit.y_lo), static_cast<double>(audit.y_hi), audit.sources, audit.buoyancy, audit.confinement, audit.curl, audit.damping,
                audit.advection, audit.projection, audit.seconds, audit.flux_bottom, audit.flux_top, audit.flux_sides,
                static_cast<double>(p.velocity >= 0 ? p.velocity : p.effect == sim3d::Effect::smoke_r2 ? 2 : 0)};  // [13]: how the velocity was carried
    pr.frames = frames;
    save(f, pr);
    const std::string line = std::format("plume {} seed {} ({}x{}x{}, span {:.0f}, controls {:.2f} {:.2f} {:.2f}): {} frames after {} of spin-up, {:.0f} s", label,
                                         p.seed, nx, ny, nz, p.span, p.intensity, p.wind, p.turbulence, frames, spin_frames, now() - t0);
    std::println("{}", line);
    std::fflush(stdout);
    std::ofstream(root / "logs" / "plume.log", std::ios::app) << line << "\n";
  }
  return 0;
}

// --- the fit -------------------------------------------------------------------------------------------------------

namespace {

// Pooled time means of several runs (a bootstrap resample may repeat a run).
struct Pooled {
  int ny = 0, nb = 0, nx = 0;
  std::vector<double> cnt, r, v, g, d, F, Fd, depth_d, depth_u;
  std::vector<double> M, B;  // per height: the plane's momentum flux (w^2) and buoyancy (g'), time means
};
Pooled pool_of(const std::vector<Profile>& runs, const std::vector<int>& pick) {
  Pooled o;
  const Profile& p0 = runs.at(zu(pick.at(0)));
  o.ny = p0.ny;
  o.nb = p0.nb;
  o.nx = p0.nx;
  o.cnt = p0.cnt;
  o.r = p0.r;
  for (auto* v : {&o.v, &o.g, &o.d}) v->assign(p0.v.size(), 0.0);
  for (auto* v : {&o.F, &o.Fd, &o.M, &o.B}) v->assign(p0.F.size(), 0.0);
  for (auto* v : {&o.depth_d, &o.depth_u}) v->assign(p0.depth_d.size(), 0.0);
  const double w = 1.0 / static_cast<double>(pick.size());
  for (const int k : pick) {
    const Profile& p = runs[zu(k)];
    for (std::size_t i = 0; i < p.v.size(); ++i) {
      o.v[i] += w * p.v[i];
      o.g[i] += w * p.g[i];
      o.d[i] += w * p.d[i];
    }
    for (std::size_t i = 0; i < p.F.size(); ++i) {
      o.F[i] += w * p.F[i];
      o.Fd[i] += w * p.Fd[i];
      o.M[i] += w * p.M[i];
      o.B[i] += w * (static_cast<double>(p.kT) * p.heat[i] - static_cast<double>(p.kD) * p.soot[i]);
    }
    for (std::size_t i = 0; i < p.depth_d.size(); ++i) {
      o.depth_d[i] += w * p.depth_d[i];
      o.depth_u[i] += w * p.depth_u[i];
    }
  }
  return o;
}

// A Gaussian fitted to the core of a radial profile: ln f = ln c - r^2 / b^2 by least squares (weights: the bins'
// cells) over the bins from the axis out to where f first falls below e^-2 of the innermost bin's (r up to about 1.4 b).
struct Gauss {
  double c = kNaN, b = kNaN;
};
Gauss gauss_core(const double* sum, const double* cnt, const double* rsum, int nb) {
  if (cnt[0] <= 0) return {};
  const double f0 = sum[0] / cnt[0];
  if (!(f0 > 0)) return {};
  double sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
  int used = 0;
  for (int j = 0; j < nb; ++j) {
    if (cnt[j] <= 0) break;
    const double f = sum[j] / cnt[j], r = rsum[j] / cnt[j];
    if (!(f > f0 * std::exp(-2.0))) break;
    const double w = cnt[j], x = r * r, y = std::log(f);
    sw += w;
    sx += w * x;
    sy += w * y;
    sxx += w * x * x;
    sxy += w * x * y;
    ++used;
  }
  if (used < 3) return {};
  const double den = sw * sxx - sx * sx;
  if (!(std::abs(den) > 0)) return {};
  const double m = (sw * sxy - sx * sy) / den, A = (sy - m * sx) / sw;
  if (!(m < 0)) return {};
  return {std::exp(A), std::sqrt(-1.0 / m)};
}

struct Lines {
  std::vector<double> z, wc, bw, gc, bg, dc, bd, F, Fd, M, B;
};
Lines lines_of(const Pooled& p) {
  Lines L;
  for (int y = 0; y < p.ny; ++y) {
    const std::size_t o = zu(y) * zu(p.nb);
    const Gauss w = gauss_core(p.v.data() + o, p.cnt.data() + o, p.r.data() + o, p.nb);
    const Gauss g = gauss_core(p.g.data() + o, p.cnt.data() + o, p.r.data() + o, p.nb);
    const Gauss d = gauss_core(p.d.data() + o, p.cnt.data() + o, p.r.data() + o, p.nb);
    L.z.push_back(y + 0.5);
    L.wc.push_back(w.c);
    L.bw.push_back(w.b);
    L.gc.push_back(g.c);
    L.bg.push_back(g.b);
    L.dc.push_back(d.c);
    L.bd.push_back(d.b);
    L.F.push_back(p.F[zu(y)]);
    L.Fd.push_back(p.Fd[zu(y)]);
    L.M.push_back(p.M.empty() ? kNaN : p.M[zu(y)]);
    L.B.push_back(p.B.empty() ? kNaN : p.B[zu(y)]);
  }
  return L;
}

// Least squares y = a + b x.
struct LinFit {
  double a = kNaN, b = kNaN;
  int n = 0;
};
LinFit linfit(const std::vector<double>& x, const std::vector<double>& y) {
  double sx = 0, sy = 0, sxx = 0, sxy = 0;
  int n = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (!std::isfinite(x[i]) || !std::isfinite(y[i])) continue;
    sx += x[i];
    sy += y[i];
    sxx += x[i] * x[i];
    sxy += x[i] * y[i];
    ++n;
  }
  LinFit f;
  f.n = n;
  if (n < 3) return f;
  const double den = n * sxx - sx * sx;
  f.b = (n * sxy - sx * sy) / den;
  f.a = (sy - f.b * sx) / n;
  return f;
}

// MTT (1956) for Gaussian profiles of equal width (velocity w = W exp(-r^2/b^2), buoyancy likewise, entrainment
// velocity alpha W): b = (6/5) alpha z, W = (25 F / (12 pi alpha^2))^(1/3) z^(-1/3), G = 2 F / (pi b^2 W)
// = (2 / (pi (36/25) alpha^2 Cw)) F^(2/3) z^(-5/3), with F = pi b^2 W G / 2 the buoyancy flux.
double mtt_cw(double alpha) { return std::cbrt(25.0 / (12.0 * std::numbers::pi * alpha * alpha)); }
double mtt_cg(double alpha) { return 2.0 / (std::numbers::pi * 1.44 * alpha * alpha * mtt_cw(alpha)); }

struct Fit {
  double beta = kNaN, z0 = kNaN, alpha_g = kNaN, alpha_t = kNaN;  // the velocity width b_w = beta (z - z0); alpha = beta / 1.2
  double beta_g = kNaN, z0_bg = kNaN;                              // the buoyancy width's
  double pw = kNaN, pg = kNaN, pd = kNaN, pF = kNaN;               // exponents against z - z0 (MTT: -1/3, -5/3, -5/3, 0)
  double pw_free = kNaN, z0_free_w = kNaN, pg_free = kNaN, z0_free_g = kNaN;  // exponent and origin fitted together
  double z0w = kNaN, z0g = kNaN;                                   // origins from w^-3 and g^-3/5 linear in z
  double F = kNaN, Fspread = kNaN;                                 // mean buoyancy flux over the range; (max - min) / mean
  double Cw = kNaN, Cg = kNaN, lambda = kNaN, Aw = kNaN;           // W (z - z0)^(1/3) / F^(1/3), G (z - z0)^(5/3) / F^(2/3), b_g / b_w; Aw = W (z - z0)^(1/3)
  double Cw_mtt = kNaN, Cg_mtt = kNaN;                             // MTT's at the fitted alpha
  // the flux-balance parameter (Morton 1959), top-hat: 5 F / (8 pi alpha_T b_T w_T^3) with b_T = sqrt(2) b_w and
  // w_T = W / 2 (1 for a pure plume, above for a lazy one, below for a forced one): the means of the range's lowest and
  // highest four heights
  double gamma_lo = kNaN, gamma_hi = kNaN;
  // the momentum budget: dM/dz (central differences of the plane's w^2) over the plane's buoyancy, the mean over the
  // range (1 if buoyancy alone changes the momentum flux, as in MTT's plume)
  double momentum = kNaN;
};

// The exponent p and origin z0 of f = A (z - z0)^p fitted together (z0 on a grid below the range: the least squares
// of ln f against ln (z - z0) at each, the best kept).
std::pair<double, double> power_free(const std::vector<double>& z, const std::vector<double>& f, double zlo) {
  double best = 1e300, bp = kNaN, bz = kNaN;
  for (double z0 = zlo - 400.0; z0 < zlo - 0.5; z0 += 0.5) {
    std::vector<double> lx, ly;
    for (std::size_t i = 0; i < z.size(); ++i) {
      if (!(f[i] > 0)) continue;
      lx.push_back(std::log(z[i] - z0));
      ly.push_back(std::log(f[i]));
    }
    const LinFit l = linfit(lx, ly);
    if (l.n < 3) continue;
    double e = 0;
    for (std::size_t i = 0; i < lx.size(); ++i) e += (ly[i] - l.a - l.b * lx[i]) * (ly[i] - l.a - l.b * lx[i]);
    if (e < best) {
      best = e;
      bp = l.b;
      bz = z0;
    }
  }
  return {bp, bz};
}

Fit fit_range(const Lines& L, double lo, double hi) {
  Fit f;
  std::vector<double> z, bw, bg, wc, gc, dc, F;
  for (std::size_t i = 0; i < L.z.size(); ++i) {
    if (L.z[i] < lo || L.z[i] > hi) continue;
    z.push_back(L.z[i]);
    bw.push_back(L.bw[i]);
    bg.push_back(L.bg[i]);
    wc.push_back(L.wc[i]);
    gc.push_back(L.gc[i]);
    dc.push_back(L.dc[i]);
    F.push_back(L.F[i]);
  }
  const LinFit b = linfit(z, bw);
  if (b.n < 3 || !(b.b > 0)) return f;
  f.beta = b.b;
  f.z0 = -b.a / b.b;
  f.alpha_g = f.beta / 1.2;
  f.alpha_t = std::numbers::sqrt2 * f.alpha_g;
  const LinFit bgf = linfit(z, bg);
  if (bgf.n >= 3 && bgf.b > 0) {
    f.beta_g = bgf.b;
    f.z0_bg = -bgf.a / bgf.b;
  }
  const auto expo = [&](const std::vector<double>& q) {
    std::vector<double> lx, ly;
    for (std::size_t i = 0; i < z.size(); ++i) {
      if (!(q[i] > 0) || !(z[i] - f.z0 > 0)) continue;
      lx.push_back(std::log(z[i] - f.z0));
      ly.push_back(std::log(q[i]));
    }
    return linfit(lx, ly).b;
  };
  f.pw = expo(wc);
  f.pg = expo(gc);
  f.pd = expo(dc);
  f.pF = expo(F);
  std::tie(f.pw_free, f.z0_free_w) = power_free(z, wc, lo);
  std::tie(f.pg_free, f.z0_free_g) = power_free(z, gc, lo);
  std::vector<double> w3, g35;
  for (std::size_t i = 0; i < z.size(); ++i) {
    w3.push_back(wc[i] > 0 ? std::pow(wc[i], -3.0) : kNaN);
    g35.push_back(gc[i] > 0 ? std::pow(gc[i], -0.6) : kNaN);
  }
  const LinFit lw = linfit(z, w3), lg = linfit(z, g35);
  if (lw.n >= 3 && lw.b > 0) f.z0w = -lw.a / lw.b;
  if (lg.n >= 3 && lg.b > 0) f.z0g = -lg.a / lg.b;
  double fs = 0, fmn = 1e300, fmx = -1e300, cw = 0, cg = 0, lam = 0, aw = 0;
  int n = 0;
  for (std::size_t i = 0; i < z.size(); ++i) {
    fs += F[i];
    fmn = std::min(fmn, F[i]);
    fmx = std::max(fmx, F[i]);
  }
  f.F = fs / static_cast<double>(z.size());
  f.Fspread = (fmx - fmn) / f.F;
  for (std::size_t i = 0; i < z.size(); ++i) {
    const double zz = z[i] - f.z0;
    if (!(zz > 0)) continue;
    cw += wc[i] * std::cbrt(zz) / std::cbrt(f.F);
    aw += wc[i] * std::cbrt(zz);
    cg += gc[i] * std::pow(zz, 5.0 / 3.0) / std::pow(f.F, 2.0 / 3.0);
    lam += bg[i] / bw[i];
    ++n;
  }
  if (n > 0) {
    f.Cw = cw / n;
    f.Cg = cg / n;
    f.lambda = lam / n;
    f.Aw = aw / n;
  }
  f.Cw_mtt = mtt_cw(f.alpha_g);
  f.Cg_mtt = mtt_cg(f.alpha_g);
  {
    double sm = 0;
    int nm = 0;
    for (std::size_t i = 1; i + 1 < L.z.size(); ++i) {
      if (L.z[i] < lo || L.z[i] > hi || !(L.B[i] > 0)) continue;
      sm += 0.5 * (L.M[i + 1] - L.M[i - 1]) / L.B[i];
      ++nm;
    }
    if (nm > 0) f.momentum = sm / nm;
  }
  std::vector<double> gam;
  for (std::size_t i = 0; i < z.size(); ++i) {
    const double bt = std::numbers::sqrt2 * bw[i], wt = 0.5 * wc[i];
    gam.push_back(5.0 * F[i] / (8.0 * std::numbers::pi * f.alpha_t * bt * wt * wt * wt));
  }
  if (gam.size() >= 8) {
    double a0 = 0, a1 = 0;
    for (int k = 0; k < 4; ++k) {
      a0 += gam[zu(k)] / 4;
      a1 += gam[gam.size() - 1 - zu(k)] / 4;
    }
    f.gamma_lo = a0;
    f.gamma_hi = a1;
  }
  return f;
}

// The wind-bent plume's path: per column x downwind of the source, the height of the soot's centroid (summed over
// depth), and the soot-weighted horizontal speed over the plume.
struct Path {
  std::vector<double> x, zc;
  double u_plume = kNaN;
};
Path path_of(const Pooled& p, double xs, int y_max) {
  Path P;
  double su = 0, sd = 0;
  for (int x = 0; x < p.nx; ++x) {
    double m = 0, zm = 0;
    for (int y = 0; y < y_max; ++y) {
      const double d = p.depth_d[zu(y) * zu(p.nx) + zu(x)];
      m += d;
      zm += d * (y + 0.5);
      su += p.depth_u[zu(y) * zu(p.nx) + zu(x)];
      sd += d;
    }
    P.x.push_back(x + 0.5 - xs);
    P.zc.push_back(m > 0 ? zm / m : kNaN);
  }
  P.u_plume = sd > 0 ? su / sd : kNaN;
  return P;
}

struct Bent {
  double p = kNaN, C = kNaN, u = kNaN;  // z - zs = k x^p; C: Briggs's coefficient (z - zs) U / ((F / pi)^(1/3) x^(2/3)) (Briggs: 1.6)
};
Bent bent_fit(const Path& P, double xlo, double xhi, double zs, double F, double U) {
  Bent b;
  std::vector<double> lx, ly;
  double c = 0;
  int n = 0;
  for (std::size_t i = 0; i < P.x.size(); ++i) {
    if (P.x[i] < xlo || P.x[i] > xhi || !(P.zc[i] - zs > 0)) continue;
    lx.push_back(std::log(P.x[i]));
    ly.push_back(std::log(P.zc[i] - zs));
    c += (P.zc[i] - zs) * std::abs(U) / (std::cbrt(F / std::numbers::pi) * std::pow(P.x[i], 2.0 / 3.0));
    ++n;
  }
  b.p = linfit(lx, ly).b;
  b.C = n ? c / n : kNaN;
  b.u = P.u_plume;
  return b;
}

struct Iv {
  double est = kNaN, lo = kNaN, hi = kNaN;
};
Iv interval(double est, std::vector<double> v) {
  std::erase_if(v, [](double x) { return !std::isfinite(x); });
  Iv iv;
  iv.est = est;
  if (v.size() < 20) return iv;
  std::ranges::sort(v);
  const auto q = [&](double p) { return v[std::min(v.size() - 1, static_cast<std::size_t>(std::floor(p * static_cast<double>(v.size() - 1) + 0.5)))]; };
  iv.lo = q(0.025);
  iv.hi = q(0.975);
  return iv;
}
std::string fmt(const Iv& v, int d) { return std::format("{:.{}f} [{:.{}f}, {:.{}f}]", v.est, d, v.lo, d, v.hi, d); }

struct Config {
  std::string label;
  std::vector<Profile> runs;
};

std::vector<Profile> load_config(const fs::path& dir) {
  std::vector<fs::path> files;
  if (!fs::exists(dir)) throw std::runtime_error("no plume runs in " + dir.string());
  for (const auto& e : fs::directory_iterator(dir)) {
    if (e.path().extension() == ".bin") files.push_back(e.path());
  }
  std::ranges::sort(files);
  std::vector<Profile> out;
  for (const auto& f : files) out.push_back(load(f));
  if (out.empty()) throw std::runtime_error("no plume runs in " + dir.string());
  return out;
}

// How a run's velocity was carried (study R2's audit records it; Params::velocity).
std::string velocity_name(const Profile& p) {
  if (p.audit.size() < 14) return "-";
  const int v = static_cast<int>(p.audit[13]);
  return v == 2 ? "flux form" : v == 1 ? "MacCormack" : "semi-Lagrangian";
}

std::vector<int> all_of(std::size_t n) {
  std::vector<int> v(n);
  for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<int>(i);
  return v;
}

}  // namespace

// nvfx_3d plumefit --configs A,B,C [--strengths A,B,C] [--box REF:CMP,...] [--bent W1,W2 --still S] [--budget A,B] --fit-lo Z --fit-hi Z
//                  --out DIR [--resamples 10000]
int fit(const Args& a, const fs::path& root) {
  const auto labels = split(a.need("configs"), ',');
  const double lo = a.f("fit-lo", 20.f), hi = a.f("fit-hi", 70.f);
  const int resamples = a.i("resamples", 10000);
  const fs::path out = a.str("out", "results/3d/r");
  fs::create_directories(out);
  std::map<std::string, Config> cfg;
  for (const auto& l : labels) cfg[l] = Config{l, load_config(root / "plume" / l)};
  std::mt19937_64 rng(2026);
  const auto resample = [&](std::size_t n) {
    std::vector<int> v(n);
    std::uniform_int_distribution<int> u(0, static_cast<int>(n) - 1);
    for (auto& x : v) x = u(rng);
    return v;
  };
  std::ofstream md(out / a.str("md", "plume_fit.md"));
  const auto both = [&](const std::string& s) {
    std::println("{}", s);
    md << s << "\n";
  };
  both(std::format("# The smoke plume against Morton, Taylor and Turner (1956)\n\nTime means of each run, pooled over the runs (seeds); fits over "
                   "heights {:.1f} to {:.1f} cells above the floor; 95% intervals by resampling the runs ({} resamples, seed 2026). Gaussian "
                   "widths are 1/e radii of the core (STUDY_R.md §1.3); alpha_G = beta / 1.2, alpha_T = sqrt(2) alpha_G.\n",
                   lo, hi, resamples));
  both("| config | runs | controls | F (cells^4/s^3) | F spread | beta (b_w / z) | alpha_G | alpha_T | z0 (cells) | p_w (MTT -1/3) | p_g (MTT -5/3) | "
       "p_soot (-5/3) | p_F (0) | z0 from w^-3 | z0 from g^-3/5 | p_w, z0 free | lambda = b_g / b_w | C_w (MTT at alpha_G) | C_g (MTT at alpha_G) | "
       "Gamma, lowest / highest heights (pure plume 1) | dM/dz over buoyancy (1) |");
  both("|---|---:|---|---:|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
  std::ofstream csv(out / a.str("csv", "plume_profiles.csv"));
  csv << "config,z,wc,bw,gc,bg,dc,bd,F,Fd\n";
  std::map<std::string, std::vector<Fit>> boots;
  std::map<std::string, Fit> point;
  for (const auto& l : labels) {
    const Config& c = cfg[l];
    const Pooled P = pool_of(c.runs, all_of(c.runs.size()));
    const Lines L = lines_of(P);
    for (std::size_t i = 0; i < L.z.size(); ++i) {
      csv << std::format("{},{:.1f},{:.6g},{:.6g},{:.6g},{:.6g},{:.6g},{:.6g},{:.6g},{:.6g}\n", l, L.z[i], L.wc[i], L.bw[i], L.gc[i], L.bg[i], L.dc[i], L.bd[i], L.F[i],
                         L.Fd[i]);
    }
    const Fit f = fit_range(L, lo, hi);
    point[l] = f;
    auto& bs = boots[l];
    for (int r = 0; r < resamples; ++r) bs.push_back(fit_range(lines_of(pool_of(c.runs, resample(c.runs.size()))), lo, hi));
    const auto iv = [&](double Fit::* m) {
      std::vector<double> v;
      for (const Fit& b : bs) v.push_back(b.*m);
      return interval(f.*m, v);
    };
    const Profile& p0 = c.runs[0];
    both(std::format("| {} | {} | {:.2f}, {:.2f}, {:.2f} ({}x{}x{}, span {:.0f}) | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} ({:.2f}) | {} ({:.2f}) |", l,
                     c.runs.size(), p0.intensity, p0.wind, p0.turbulence, p0.nx, p0.ny, p0.nz, p0.span, fmt(iv(&Fit::F), 0), fmt(iv(&Fit::Fspread), 2),
                     fmt(iv(&Fit::beta), 3), fmt(iv(&Fit::alpha_g), 3), fmt(iv(&Fit::alpha_t), 3), fmt(iv(&Fit::z0), 1), fmt(iv(&Fit::pw), 3),
                     fmt(iv(&Fit::pg), 3), fmt(iv(&Fit::pd), 3), fmt(iv(&Fit::pF), 3), fmt(iv(&Fit::z0w), 1), fmt(iv(&Fit::z0g), 1), fmt(iv(&Fit::pw_free), 3),
                     fmt(iv(&Fit::lambda), 2), fmt(iv(&Fit::Cw), 2), f.Cw_mtt, fmt(iv(&Fit::Cg), 2), f.Cg_mtt) +
         std::format(" {} / {} | {} |", fmt(iv(&Fit::gamma_lo), 2), fmt(iv(&Fit::gamma_hi), 2), fmt(iv(&Fit::momentum), 2)));
  }
  // the source strengths: W (z - z0)^(1/3) against F across configs (MTT: slope 1/3)
  if (a.has("strengths")) {
    const auto st = split(a.str("strengths"), ',');
    const auto slope = [&](const std::map<std::string, Fit>& fits) {
      std::vector<double> x, y;
      for (const auto& l : st) {
        x.push_back(std::log(fits.at(l).F));
        y.push_back(std::log(fits.at(l).Aw));
      }
      return linfit(x, y).b;
    };
    const double s0 = slope(point);
    std::vector<double> v;
    for (int r = 0; r < resamples; ++r) {
      std::map<std::string, Fit> m;
      for (const auto& l : st) m[l] = boots[l][zu(r)];
      v.push_back(slope(m));
    }
    both(std::format("\nThe source strengths ({}): the slope of ln (W (z - z0)^(1/3)) against ln F, MTT 1/3: {}", a.str("strengths"), fmt(interval(s0, v), 3)));
  }
  // the box: per height, the centreline velocity and the velocity width in the study's box over a larger box's
  if (a.has("box")) {
    for (const auto& pair : split(a.str("box"), ',')) {
      const auto ab = split(pair, ':');
      const Config &A = cfg.at(ab.at(0)), &B = cfg.at(ab.at(1));
      both(std::format("\nThe box: {} over {} (ratios of the pooled profiles at each height, 95% intervals):\n", ab[0], ab[1]));
      both("| z (cells) | W ratio | b_w ratio | G ratio | F ratio |");
      both("|---:|---|---|---|---|");
      const Lines LA = lines_of(pool_of(A.runs, all_of(A.runs.size()))), LB = lines_of(pool_of(B.runs, all_of(B.runs.size())));
      std::vector<Lines> ba, bb;
      for (int r = 0; r < std::min(resamples, 2000); ++r) {
        ba.push_back(lines_of(pool_of(A.runs, resample(A.runs.size()))));
        bb.push_back(lines_of(pool_of(B.runs, resample(B.runs.size()))));
      }
      std::ofstream bcsv(out / std::format("plume_box_{}_{}.csv", ab[0], ab[1]));
      bcsv << "z,w_ratio,w_lo,w_hi,b_ratio,b_lo,b_hi,g_ratio,g_lo,g_hi,F_ratio,F_lo,F_hi\n";
      const std::size_t n = std::min(LA.z.size(), LB.z.size());
      double agree_to = kNaN;  // STUDY_R.md §1.3: the highest height to which both ratios' intervals reach into [0.95, 1.05]
      bool agreeing = true;
      for (std::size_t i = 0; i < n; ++i) {
        const auto ratio = [&](std::vector<double> Lines::* m) {
          std::vector<double> v;
          for (std::size_t r = 0; r < ba.size(); ++r) v.push_back((ba[r].*m)[i] / (bb[r].*m)[i]);
          return interval((LA.*m)[i] / (LB.*m)[i], v);
        };
        const Iv w = ratio(&Lines::wc), b = ratio(&Lines::bw), g = ratio(&Lines::gc), F = ratio(&Lines::F);
        if (LA.z[i] >= lo && agreeing) {
          const auto reaches = [](const Iv& v) { return v.hi >= 0.95 && v.lo <= 1.05; };
          if (reaches(w) && reaches(b)) agree_to = LA.z[i];
          else agreeing = false;
        }
        bcsv << std::format("{:.1f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n", LA.z[i], w.est, w.lo, w.hi, b.est, b.lo,
                            b.hi, g.est, g.lo, g.hi, F.est, F.lo, F.hi);
        if (i % 4 == 2) both(std::format("| {:.1f} | {} | {} | {} | {} |", LA.z[i], fmt(w, 3), fmt(b, 3), fmt(g, 3), fmt(F, 3)));
      }
      both(std::format("\nThe boxes agree (both ratios' intervals reach into [0.95, 1.05] at every height from {:.0f}) up to z = {:.1f} cells; rounded down "
                       "to a multiple of 4: {:.0f}.",
                       lo, agree_to, std::isfinite(agree_to) ? 4.0 * std::floor(agree_to / 4.0) : kNaN));
    }
  }
  // the buoyancy budget between the range's ends (§1.4): what the planes' flux loses from lo to hi against what leaves
  // through the box's sides in between and what the box stores there over the averaging (runs with the budget's sums)
  if (a.has("budget")) {
    both(std::format("\nThe buoyancy budget from {:.0f} to {:.0f} cells (each as a fraction of F at {:.0f}; 95% intervals by resampling the runs): the "
                     "planes' flux lost, what leaves through the four sides in between, what the box stores there, and the rest (made or lost "
                     "by the numerics):\n",
                     lo, hi, lo));
    both("| config | runs | F at lo | F at hi | lost, (F(lo) - F(hi)) / F(lo) | out through the sides | stored | the rest |");
    both("|---|---:|---:|---:|---|---|---|---|");
    struct Budget {
      double Flo = kNaN, Fhi = kNaN, lost = kNaN, side = kNaN, stored = kNaN, rest = kNaN;
    };
    for (const auto& l : split(a.str("budget"), ',')) {
      const Config& c = cfg.at(l);
      const Profile& p0 = c.runs[0];
      for (const Profile& p : c.runs) {
        if (p.side.size() != zu(p.ny)) throw std::runtime_error(l + ": a run without the budget's sums");
      }
      int ylo = -1, yhi = -1;
      for (int y = 0; y < p0.ny; ++y) {
        if (y + 0.5 >= lo && ylo < 0) ylo = y;
        if (y + 0.5 <= hi) yhi = y;
      }
      if (ylo < 0 || yhi <= ylo) throw std::runtime_error("--budget: the range has no rows");
      const auto budget = [&](const std::vector<int>& pick) {
        const double w = 1.0 / static_cast<double>(pick.size());
        Budget b{0, 0, 0, 0, 0, 0};
        for (const int k : pick) {
          const Profile& p = c.runs[zu(k)];
          const double span = (p.frames - 1) * static_cast<double>(p.seconds) / p.frames;  // first to last averaged frame
          b.Flo += w * p.F[zu(ylo)];
          b.Fhi += w * p.F[zu(yhi)];
          for (int y = ylo; y <= yhi; ++y) {
            const double h = y == ylo || y == yhi ? 0.5 : 1.0;  // the planes through the end rows' centres
            b.side += w * h * p.side[zu(y)];
            b.stored += w * h * (p.b_last[zu(y)] - p.b_first[zu(y)]) / span;
          }
        }
        b.lost = (b.Flo - b.Fhi) / b.Flo;
        b.side /= b.Flo;
        b.stored /= b.Flo;
        b.rest = b.lost - b.side - b.stored;
        return b;
      };
      const Budget b0 = budget(all_of(c.runs.size()));
      std::vector<Budget> bs;
      for (int r = 0; r < resamples; ++r) bs.push_back(budget(resample(c.runs.size())));
      const auto iv = [&](double Budget::* m) {
        std::vector<double> v;
        for (const Budget& b : bs) v.push_back(b.*m);
        return fmt(interval(b0.*m, v), 3);
      };
      both(std::format("| {} | {} | {:.0f} | {:.0f} | {} | {} | {} | {} |", l, c.runs.size(), b0.Flo, b0.Fhi, iv(&Budget::lost), iv(&Budget::side),
                       iv(&Budget::stored), iv(&Budget::rest)));
    }
  }
  // Study R2 (docs/STUDY_R2.md §1): the vertical momentum budget of the slab between the fit's heights, per operator
  // (runs with the audit), each as a fraction of what the buoyancy gives the slab; and the momentum flux's growth with
  // height against the planes' buoyancy, without and with the pressure's part
  if (a.has("momentum")) {
    both(std::format("\nThe vertical momentum budget from {:.0f} to {:.0f} cells (study R2): what each operator adds to the slab's vertical "
                     "momentum, as a fraction of what the buoyancy adds (rates over the averaged frames, pooled over the runs; 95% intervals by "
                     "resampling the runs). Steady, they sum to what the slab stores (about 0). dM/dz over B is study R's ratio (M the plane's "
                     "sum of w^2, B its buoyancy); with P, the plane's sum of the pressure, added to M; the advection against the momentum "
                     "flux the planes carry in and out of the slab (1 if the advection conserves momentum); and the flux form's own face "
                     "fluxes where the velocity is in flux form:\n",
                     lo, hi));
    both("| config | runs | velocity | dM/dz over B | d(M + P)/dz over B | advection | projection (pressure) | confinement | curl noise | damping | "
         "sum (stored) | advection over the planes' net flux | advection over its face fluxes |");
    both("|---|---:|---|---|---|---|---|---|---|---|---|---|---|");
    struct Mom {
      double ratio = kNaN, ratio_p = kNaN, adv = kNaN, proj = kNaN, conf = kNaN, curl = kNaN, damp = kNaN, sum = kNaN, adv_planes = kNaN, adv_faces = kNaN;
    };
    for (const auto& l : split(a.str("momentum"), ',')) {
      const Config& c = cfg.at(l);
      for (const Profile& p : c.runs) {
        if (p.audit.size() < 13 || p.P.size() != zu(p.ny)) throw std::runtime_error(l + ": a run without the momentum audit");
      }
      const auto mom = [&](const std::vector<int>& pick) {
        const double w = 1.0 / static_cast<double>(pick.size());
        const Profile& p0 = c.runs[zu(pick[0])];
        const int ny = p0.ny;
        std::vector<double> M(zu(ny), 0.0), P(zu(ny), 0.0), B(zu(ny), 0.0);
        std::array<double, 13> au{};
        for (const int k : pick) {
          const Profile& p = c.runs[zu(k)];
          for (int y = 0; y < ny; ++y) {
            M[zu(y)] += w * p.M[zu(y)];
            P[zu(y)] += w * p.P[zu(y)];
            B[zu(y)] += w * (static_cast<double>(p.kT) * p.heat[zu(y)] - static_cast<double>(p.kD) * p.soot[zu(y)]);
          }
          for (std::size_t j = 2; j < 13; ++j) au[j] += w * p.audit[j] / p.audit[9];  // rates (per second); [9] is the time
        }
        Mom m;
        double sm = 0, smp = 0;
        int nm = 0;
        for (int y = 1; y + 1 < ny; ++y) {
          if (y + 0.5 < lo || y + 0.5 > hi || !(B[zu(y)] > 0)) continue;
          sm += 0.5 * (M[zu(y) + 1] - M[zu(y) - 1]) / B[zu(y)];
          smp += 0.5 * (M[zu(y) + 1] + P[zu(y) + 1] - M[zu(y) - 1] - P[zu(y) - 1]) / B[zu(y)];
          ++nm;
        }
        if (nm) {
          m.ratio = sm / nm;
          m.ratio_p = smp / nm;
        }
        const double b = au[3];
        m.adv = au[7] / b;
        m.proj = au[8] / b;
        m.conf = au[4] / b;
        m.curl = au[5] / b;
        m.damp = au[6] / b;
        m.sum = (au[2] + au[3] + au[4] + au[5] + au[6] + au[7] + au[8]) / b;
        const int ylo = static_cast<int>(p0.audit[0]), yhi = static_cast<int>(p0.audit[1]);
        if (ylo >= 1 && yhi + 1 < ny) {
          const double flux_in = 0.5 * (M[zu(ylo) - 1] + M[zu(ylo)]), flux_out = 0.5 * (M[zu(yhi)] + M[zu(yhi) + 1]);
          m.adv_planes = au[7] / (flux_in - flux_out);
        }
        const double faces = au[10] - au[11] + au[12];
        if (std::abs(faces) > 0) m.adv_faces = au[7] / faces;
        return m;
      };
      const Mom m0 = mom(all_of(c.runs.size()));
      std::vector<Mom> ms;
      for (int r = 0; r < resamples; ++r) ms.push_back(mom(resample(c.runs.size())));
      const auto iv = [&](double Mom::* f) {
        std::vector<double> v;
        for (const Mom& m : ms) v.push_back(m.*f);
        return fmt(interval(m0.*f, v), 3);
      };
      both(std::format("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |", l, c.runs.size(), velocity_name(c.runs[0]), iv(&Mom::ratio),
                       iv(&Mom::ratio_p), iv(&Mom::adv), iv(&Mom::proj), iv(&Mom::conf), iv(&Mom::curl), iv(&Mom::damp), iv(&Mom::sum), iv(&Mom::adv_planes),
                       std::isfinite(m0.adv_faces) ? iv(&Mom::adv_faces) : std::string("-")));
    }
  }
  // the wind-bent plume against Briggs's two-thirds law: z - zs = 1.6 (F / pi)^(1/3) x^(2/3) / U
  if (a.has("bent")) {
    const auto still = cfg.at(a.need("still"));
    const Fit fs0 = point.at(still.label);
    const double zs = a.f("source-z", 4.7f), xlo = a.f("bent-lo", 4.f), xhi = a.f("bent-hi", 30.f);
    const int ymax = a.i("bent-ymax", 88);
    both(std::format("\nThe wind-bent plume against Briggs's two-thirds law (z - z_s = 1.6 (F / pi)^(1/3) x^(2/3) / U; x from {:.0f} to {:.0f} cells "
                     "downwind of the source; F from {} over the fit range; U the preset's wind speed):\n",
                     xlo, xhi, still.label));
    both("| config | wind | U (cells/s) | plume's mean speed (cells/s) | exponent p (Briggs 2/3) | coefficient (Briggs 1.6) |");
    both("|---|---:|---:|---:|---|---|");
    std::ofstream pcsv(out / "plume_bent_paths.csv");
    pcsv << "config,x,zc\n";
    for (const auto& l : split(a.str("bent"), ',')) {
      const Config& c = cfg.at(l);
      const double U = c.runs[0].wind_speed;
      const Pooled P = pool_of(c.runs, all_of(c.runs.size()));
      const double xs = c.runs[0].cx;
      const Path path = path_of(P, xs, ymax);
      for (std::size_t i = 0; i < path.x.size(); ++i) pcsv << std::format("{},{:.1f},{:.3f}\n", l, path.x[i], path.zc[i]);
      const Bent b = bent_fit(path, xlo, xhi, zs, fs0.F, U);
      std::vector<double> vp, vc;
      for (int r = 0; r < resamples; ++r) {
        const Bent q = bent_fit(path_of(pool_of(c.runs, resample(c.runs.size())), xs, ymax), xlo, xhi, zs, fs0.F, U);
        vp.push_back(q.p);
        vc.push_back(q.C);
      }
      both(std::format("| {} | {:.2f} | {:.1f} | {:.1f} | {} | {} |", l, c.runs[0].wind, U, b.u, fmt(interval(b.p, vp), 3), fmt(interval(b.C, vc), 2)));
    }
  }
  return 0;
}

// --- the figure ----------------------------------------------------------------------------------------------------

namespace {

struct Canvas {
  int w = 0, h = 0;
  std::vector<std::uint8_t> rgb;
  void put(int x, int y, std::array<std::uint8_t, 3> c, float a = 1.f) {
    if (x < 0 || y < 0 || x >= w || y >= h) return;
    std::uint8_t* d = rgb.data() + (zu(y) * zu(w) + zu(x)) * 3;
    for (int k = 0; k < 3; ++k) d[k] = static_cast<std::uint8_t>(std::lround(d[k] + a * (c[zu(k)] - d[k])));
  }
  void fill(int x0, int y0, int fw, int fh, std::array<std::uint8_t, 3> c) {
    for (int y = y0; y < y0 + fh; ++y)
      for (int x = x0; x < x0 + fw; ++x) put(x, y, c);
  }
  void text(std::string_view t, int x, int y, int scale, std::array<std::uint8_t, 3> c) {
    nfx::tools::draw_text(t, x, y, scale, [&](int px, int py) { put(px, py, c); });
  }
  // a line of width `wd` pixels (dash: on/off lengths in pixels, 0 for solid)
  void line(double x0, double y0, double x1, double y1, std::array<std::uint8_t, 3> c, double wd = 1.5, double dash = 0, float a = 1.f) {
    const double len = std::hypot(x1 - x0, y1 - y0);
    const int n = std::max(1, static_cast<int>(std::ceil(len * 2)));
    for (int i = 0; i <= n; ++i) {
      const double t = static_cast<double>(i) / n;
      if (dash > 0 && std::fmod(t * len, 2 * dash) > dash) continue;
      const double x = x0 + t * (x1 - x0), y = y0 + t * (y1 - y0);
      const int r = static_cast<int>(std::ceil(wd / 2));
      for (int dy = -r; dy <= r; ++dy)
        for (int dx = -r; dx <= r; ++dx) {
          const double d = std::hypot(static_cast<double>(dx), static_cast<double>(dy));
          if (d <= wd / 2 + 0.5) put(static_cast<int>(std::lround(x)) + dx, static_cast<int>(std::lround(y)) + dy, c, a * static_cast<float>(std::clamp(wd / 2 + 0.5 - d, 0.0, 1.0)));
        }
    }
  }
  void dot(double x, double y, double r, std::array<std::uint8_t, 3> c, float a = 1.f) {
    for (int dy = -static_cast<int>(r) - 1; dy <= static_cast<int>(r) + 1; ++dy)
      for (int dx = -static_cast<int>(r) - 1; dx <= static_cast<int>(r) + 1; ++dx) {
        const double d = std::hypot(static_cast<double>(dx), static_cast<double>(dy));
        if (d <= r + 0.5) put(static_cast<int>(std::lround(x)) + dx, static_cast<int>(std::lround(y)) + dy, c, a * static_cast<float>(std::clamp(r + 0.5 - d, 0.0, 1.0)));
      }
  }
};

// A plot panel: data coordinates to pixels, log or linear on each axis.
struct Panel {
  int x0, y0, w, h;
  double xlo, xhi, ylo, yhi;
  bool logx, logy;
  double px(double x) const {
    const double t = logx ? (std::log(x) - std::log(xlo)) / (std::log(xhi) - std::log(xlo)) : (x - xlo) / (xhi - xlo);
    return x0 + t * w;
  }
  double py(double y) const {
    const double t = logy ? (std::log(y) - std::log(ylo)) / (std::log(yhi) - std::log(ylo)) : (y - ylo) / (yhi - ylo);
    return y0 + h - t * h;
  }
  bool inside(double x, double y) const { return x >= xlo && x <= xhi && y >= ylo && y <= yhi && std::isfinite(x) && std::isfinite(y); }
  void frame(Canvas& c, const std::string& xl, const std::string& yl, const std::vector<double>& xt, const std::vector<double>& yt) const {
    c.fill(x0, y0, w, h, {22, 23, 28});
    const std::array<std::uint8_t, 3> ax{150, 150, 158}, grid{40, 42, 50};
    for (const double t : xt) {
      if (t < xlo || t > xhi) continue;
      c.line(px(t), y0, px(t), y0 + h, grid, 1);
      const std::string s = std::format("{:g}", t);
      c.text(s, static_cast<int>(px(t)) - nfx::tools::text_width(s, 1) / 2, y0 + h + 4, 1, ax);
    }
    for (const double t : yt) {
      if (t < ylo || t > yhi) continue;
      c.line(x0, py(t), x0 + w, py(t), grid, 1);
      const std::string s = std::format("{:g}", t);
      c.text(s, x0 - 6 - nfx::tools::text_width(s, 1), static_cast<int>(py(t)) - 3, 1, ax);
    }
    c.line(x0, y0 + h, x0 + w, y0 + h, ax, 1);
    c.line(x0, y0, x0, y0 + h, ax, 1);
    c.text(xl, x0 + w - nfx::tools::text_width(xl, 1), y0 + h + 16, 1, ax);
    c.text(yl, x0, y0 - 14, 1, {210, 210, 216});
  }
  // y = f(x) as a curve over the panel's x range
  template <class F>
  void curve(Canvas& c, F&& f, std::array<std::uint8_t, 3> col, double wd, double dash = 0) const {
    double lx = kNaN, ly = kNaN;
    for (int i = 0; i <= 200; ++i) {
      const double t = i / 200.0;
      const double x = logx ? std::exp(std::log(xlo) + t * (std::log(xhi) - std::log(xlo))) : xlo + t * (xhi - xlo);
      const double y = f(x);
      if (inside(x, y) && std::isfinite(lx)) c.line(px(lx), py(ly), px(x), py(y), col, wd, dash);
      if (inside(x, y)) {
        lx = x;
        ly = y;
      } else {
        lx = kNaN;
      }
    }
  }
};

std::array<double, 2> project(const render3d::Camera& cam, double x, double y, double z, int W, int H) {
  double clip[4];
  double v[4];
  const double p[4] = {x, y, z, 1};
  for (int r = 0; r < 4; ++r) {
    v[r] = 0;
    for (int k = 0; k < 4; ++k) v[r] += cam.view[zu(k * 4 + r)] * p[k];
  }
  for (int r = 0; r < 4; ++r) {
    clip[r] = 0;
    for (int k = 0; k < 4; ++k) clip[r] += cam.proj[zu(k * 4 + r)] * v[k];
  }
  return {(clip[0] / clip[3] + 1) / 2 * W, (1 - clip[1] / clip[3]) / 2 * H};
}

}  // namespace

// nvfx_3d plumevideo --configs A,B,C --fit-lo Z --fit-hi Z --out F.mp4 [--seconds 16 --seed S --resamples 2000]: on the
// left the smoke simulated live in still air at the first config's controls, MTT's top-hat edges b_T = (6/5) alpha_T (z - z0)
// drawn over it with the fitted alpha and origin; on the right the pooled profiles of every config against the theory's
// power laws, and the live run's own running average converging onto them.
int video(const Args& a, const fs::path& root) {
  const auto labels = split(a.need("configs"), ',');
  const double lo = a.f("fit-lo", 20.f), hi = a.f("fit-hi", 70.f);
  const fs::path out = a.need("out");
  const int W = a.i("width", 1280), H = a.i("height", 720), threads = a.i("threads", 3), crf = a.i("crf", 26);
  const float seconds = a.f("seconds", 16.f);
  const int resamples = a.i("resamples", 2000);
  std::vector<Config> cfg;
  std::vector<Lines> lines;
  std::vector<Fit> fits;
  for (const auto& l : labels) {
    cfg.push_back(Config{l, load_config(root / "plume" / l)});
    lines.push_back(lines_of(pool_of(cfg.back().runs, all_of(cfg.back().runs.size()))));
    fits.push_back(fit_range(lines.back(), lo, hi));
  }
  // the first config's intervals (shown on the figure)
  std::mt19937_64 rng(2026);
  std::vector<Fit> bs;
  {
    const auto& runs = cfg[0].runs;
    std::uniform_int_distribution<int> u(0, static_cast<int>(runs.size()) - 1);
    for (int r = 0; r < resamples; ++r) {
      std::vector<int> pick(runs.size());
      for (auto& x : pick) x = u(rng);
      bs.push_back(fit_range(lines_of(pool_of(runs, pick)), lo, hi));
    }
  }
  const auto iv = [&](double Fit::* m) {
    std::vector<double> v;
    for (const Fit& b : bs) v.push_back(b.*m);
    return interval(fits[0].*m, v);
  };
  const Iv i_pw = iv(&Fit::pw), i_pg = iv(&Fit::pg), i_at = iv(&Fit::alpha_t), i_pF = iv(&Fit::pF);
  const Profile& p0 = cfg[0].runs[0];
  sim3d::Params p;
  if (!sim3d::parse_effect(a.str("effect", "smoke_r"), p.effect)) throw std::invalid_argument("--effect");
  p.intensity = p0.intensity;
  p.wind = p0.wind;
  p.turbulence = p0.turbulence;
  p.width = p0.nx;
  p.height = p0.ny;
  p.span = p0.span;
  p.threads = threads;
  p.seed = a.u64("seed", 29'900'100);
  sim3d::Fluid sim(p);
  const int nx = sim.nx(), ny = sim.ny(), nz = sim.nz();
  render3d::Renderer rend(nx, ny, nz, render3d::optics_for("smoke3d"));
  nfx::Pool pool(threads);
  const std::size_t n = zu(nx) * zu(ny) * zu(nz);
  std::vector<float> T(n), D(n), U(3 * n);
  // the live run's running average of the vertical velocity in the radial bins (as the measurement's)
  std::vector<double> vsum(p0.v.size(), 0.0), gsum(p0.v.size(), 0.0);
  std::vector<int> bin(zu(nx) * zu(nz));
  for (int z = 0; z < nz; ++z)
    for (int x = 0; x < nx; ++x) {
      const int b = static_cast<int>(std::floor(std::hypot(x + 0.5 - p0.cx, z + 0.5 - p0.cz)));
      bin[zu(z) * zu(nx) + zu(x)] = b < p0.nb ? b : -1;
    }
  const auto kb = sim.buoyancy_coefficients();
  int averaged = 0;
  const float avg_from = a.f("average-from", 4.f);
  const std::string cmd = std::format("ffmpeg -loglevel error -y -f rawvideo -pix_fmt rgb24 -s {}x{} -r 30 -i - -an -c:v libx264 -preset slow -crf {} -threads 2 "
                                      "-pix_fmt yuv420p -movflags +faststart \"{}\"",
                                      W, H, crf, out.string());
  std::FILE* pipe = popen(cmd.c_str(), "w");
  if (!pipe) throw std::runtime_error("cannot start ffmpeg");
  Canvas cv{W, H, std::vector<std::uint8_t>(zu(W) * zu(H) * 3)};
  const int lw = W / 2, frames = static_cast<int>(std::lround(seconds * 30.f));
  std::vector<std::uint8_t> pic(zu(lw) * zu(H) * 4);
  const std::array<std::array<std::uint8_t, 3>, 4> col{{{255, 170, 70}, {110, 190, 255}, {150, 230, 120}, {230, 130, 220}}};
  const double t0 = now();
  for (int f = 0; f < frames; ++f) {
    sim.step_frame();
    sim.fields(T, D);
    sim.velocity(U);
    const float t = sim.time();
    if (t >= avg_from) {
      ++averaged;
      for (int y = 0; y < ny; ++y)
        for (int z = 0; z < nz; ++z)
          for (int x = 0; x < nx; ++x) {
            const int b = bin[zu(z) * zu(nx) + zu(x)];
            if (b < 0) continue;
            const std::size_t i = (zu(y) * zu(nz) + zu(z)) * zu(nx) + zu(x);
            vsum[zu(y) * zu(p0.nb) + zu(b)] += U[3 * i + 1];
            gsum[zu(y) * zu(p0.nb) + zu(b)] += kb[0] * T[i] - kb[1] * D[i];
          }
    }
    // left: the plume from the side, over a dark backdrop, with MTT's top-hat edges
    rend.prepare(T, D, &pool);
    const render3d::Camera cam = render3d::look_at({0.f, 0.72f, 2.35f}, {0.f, 0.72f, 0.f}, {0.f, 1.f, 0.f}, 42.f, static_cast<float>(lw) / static_cast<float>(H));
    rend.render(cam, lw, H, pic.data(), zu(lw) * 4, &pool);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < lw; ++x) {
        const std::uint8_t* s2 = pic.data() + (zu(y) * zu(lw) + zu(x)) * 4;
        const float al = s2[3] / 255.f, bg = 16.f + 14.f * static_cast<float>(y) / static_cast<float>(H);
        std::uint8_t* d = cv.rgb.data() + (zu(y) * zu(W) + zu(x)) * 3;
        for (int k = 0; k < 3; ++k) d[k] = static_cast<std::uint8_t>(std::clamp(s2[k] + bg * (1.f - al), 0.f, 255.f));
      }
    {
      const Fit& F0 = fits[0];
      const double bt = 1.2 * F0.alpha_t;
      for (const int side : {-1, 1}) {
        std::array<double, 2> last{kNaN, kNaN};
        for (int k = 0; k <= 40; ++k) {
          const double zc = lo + (hi - lo) * k / 40.0;
          const double xc = side * bt * (zc - F0.z0);
          const auto q = project(cam, xc / nx, zc / nx, 0.0, lw, H);
          if (k > 0) cv.line(last[0], last[1], q[0], q[1], {255, 225, 150}, 2, 6);
          last = q;
        }
      }
      for (const double zc : {lo, hi}) {
        const auto a1 = project(cam, -0.5, zc / nx, 0.0, lw, H), a2 = project(cam, -0.45, zc / nx, 0.0, lw, H);
        cv.line(a1[0], a1[1], a2[0], a2[1], {200, 200, 206}, 1.5);
      }
    }
    cv.text("SIMULATED SMOKE, STILL AIR", 20, 18, 2, {255, 255, 255});
    cv.text(std::format("OUR 3D SOLVER, {}X{}X{}, CONTROLS {:.2f} {:.2f} {:.2f}", nx, ny, nz, p.intensity, p.wind, p.turbulence), 20, 42, 1, {190, 190, 196});
    cv.text("DASHED: MTT TOP-HAT EDGE B = 1.2 ALPHA (Z - Z0), FITTED", 20, H - 40, 1, {255, 225, 150});
    cv.text(std::format("{:.1f} S", t), lw - 90, 18, 2, {255, 225, 150});
    // right: three panels against the theory
    cv.fill(lw, 0, W - lw, H, {12, 13, 17});
    cv.fill(lw - 1, 0, 2, H, {0, 0, 0});
    const int px0 = lw + 70, pw = W - lw - 100, ph = 160;
    const Fit& F0 = fits[0];
    double zmax = 0;
    for (const auto& L : lines) zmax = std::max(zmax, L.z.back() - F0.z0);
    const double zmin = std::max(4.0, lo - F0.z0 - 20);
    const Panel pw_(px0, 44, pw, ph, zmin, zmax * 1.05, 8, 80, true, true);
    const Panel pb(px0, 44 + ph + 58, pw, ph, 0, static_cast<double>(ny), 0, 24, false, false);
    const Panel pg(px0, 44 + 2 * (ph + 58), pw, ph, zmin, zmax * 1.05, 1, 100, true, true);
    pw_.frame(cv, "Z - Z0 (CELLS)", "CENTRELINE VELOCITY W (CELLS/S)", {10, 20, 50, 100, 200}, {10, 20, 50});
    pb.frame(cv, "Z (CELLS ABOVE THE FLOOR)", "TOP-HAT RADIUS SQRT(2) B_W (CELLS)", {0, 20, 40, 60, 80}, {0, 5, 10, 15, 20});
    pg.frame(cv, "Z - Z0 (CELLS)", "CENTRELINE BUOYANCY G' (CELLS/S^2)", {10, 20, 50, 100, 200}, {1, 10, 100});
    for (std::size_t c = 0; c < lines.size(); ++c) {
      const Lines& L = lines[c];
      const Fit& Fc = fits[c];
      const auto colr = col[c % col.size()];
      // MTT's laws through this config's points: W = A (z - z0)^(-1/3), G = B (z - z0)^(-5/3), amplitudes fitted
      double sa = 0, sb = 0;
      int na = 0;
      for (std::size_t i = 0; i < L.z.size(); ++i) {
        if (L.z[i] < lo || L.z[i] > hi || !(L.wc[i] > 0) || !(L.gc[i] > 0)) continue;
        sa += std::log(L.wc[i]) + std::log(L.z[i] - F0.z0) / 3.0;
        sb += std::log(L.gc[i]) + 5.0 * std::log(L.z[i] - F0.z0) / 3.0;
        ++na;
      }
      const double A = na ? std::exp(sa / na) : kNaN, B = na ? std::exp(sb / na) : kNaN;
      pw_.curve(cv, [&](double x) { return A * std::pow(x, -1.0 / 3.0); }, colr, 1.2, 5);
      pg.curve(cv, [&](double x) { return B * std::pow(x, -5.0 / 3.0); }, colr, 1.2, 5);
      for (std::size_t i = 0; i < L.z.size(); i += 2) {
        const bool in = L.z[i] >= lo && L.z[i] <= hi;
        const double zz = L.z[i] - F0.z0;
        if (pw_.inside(zz, L.wc[i])) cv.dot(pw_.px(zz), pw_.py(L.wc[i]), in ? 2.4 : 1.4, colr, in ? 1.f : 0.45f);
        if (pg.inside(zz, L.gc[i])) cv.dot(pg.px(zz), pg.py(L.gc[i]), in ? 2.4 : 1.4, colr, in ? 1.f : 0.45f);
        const double bt = std::numbers::sqrt2 * L.bw[i];
        if (pb.inside(L.z[i], bt)) cv.dot(pb.px(L.z[i]), pb.py(bt), in ? 2.4 : 1.4, colr, in ? 1.f : 0.45f);
      }
      (void)Fc;
      cv.text(std::format("INTENSITY {:.2f}", cfg[c].runs[0].intensity), px0 + pw - 100, pw_.y0 + 22 + 12 * static_cast<int>(c), 1, colr);
    }
    // the literature's band of top-hat entrainment for pure plumes (0.10 to 0.16) about the fitted origin, and the fit
    for (const double al : {0.10, 0.16}) pb.curve(cv, [&](double z) { return 1.2 * al * (z - F0.z0); }, {120, 120, 130}, 1.2, 4);
    pb.curve(cv, [&](double z) { return 1.2 * F0.alpha_t * (z - F0.z0); }, {255, 225, 150}, 1.6);
    cv.text("GREY: ALPHA_T 0.10 AND 0.16 (CARAZZO ET AL. 2006)", px0 + 6, pb.y0 + 18, 1, {150, 150, 158});
    for (const double zc : {lo, hi}) cv.line(pb.px(zc), pb.y0, pb.px(zc), pb.y0 + pb.h, {90, 90, 100}, 1, 3);
    // the live run's running average (one seed): white
    if (averaged > 0) {
      Pooled lp;
      lp.ny = ny;
      lp.nb = p0.nb;
      lp.cnt = p0.cnt;
      lp.r = p0.r;
      lp.v.resize(vsum.size());
      lp.g.resize(gsum.size());
      lp.d.assign(vsum.size(), 0.0);
      for (std::size_t i = 0; i < vsum.size(); ++i) {
        lp.v[i] = vsum[i] / averaged;
        lp.g[i] = gsum[i] / averaged;
      }
      lp.F.assign(zu(ny), 0.0);
      lp.Fd.assign(zu(ny), 0.0);
      const Lines L = lines_of(lp);
      double lx = kNaN, ly = kNaN, gx = kNaN, gy = kNaN;
      for (std::size_t i = 0; i < L.z.size(); ++i) {
        const double zz = L.z[i] - F0.z0;
        if (pw_.inside(zz, L.wc[i])) {
          if (std::isfinite(lx)) cv.line(pw_.px(lx), pw_.py(ly), pw_.px(zz), pw_.py(L.wc[i]), {255, 255, 255}, 1.4);
          lx = zz;
          ly = L.wc[i];
        }
        if (pg.inside(zz, L.gc[i])) {
          if (std::isfinite(gx)) cv.line(pg.px(gx), pg.py(gy), pg.px(zz), pg.py(L.gc[i]), {255, 255, 255}, 1.4);
          gx = zz;
          gy = L.gc[i];
        }
      }
      cv.text(std::format("WHITE: THIS RUN'S TIME AVERAGE, {:.1f} S", averaged / 30.0), px0 + 6, pw_.y0 + ph - 12, 1, {255, 255, 255});
    } else {
      cv.text(std::format("WHITE: THIS RUN'S TIME AVERAGE (FROM {:.0f} S)", avg_from), px0 + 6, pw_.y0 + ph - 12, 1, {200, 200, 200});
    }
    cv.text(std::format("MTT: EXPONENT -0.33. FITTED {:.2f} [{:.2f}, {:.2f}]", i_pw.est, i_pw.lo, i_pw.hi), px0 + 6, pw_.y0 + 6, 1, {255, 225, 150});
    cv.text(std::format("MTT: B = 1.2 ALPHA (Z - Z0). FITTED ALPHA_T {:.3f} [{:.3f}, {:.3f}], Z0 {:.0f}", i_at.est, i_at.lo, i_at.hi, F0.z0), px0 + 6, pb.y0 + 6, 1,
            {255, 225, 150});
    cv.text(std::format("MTT: EXPONENT -1.67. FITTED {:.2f} [{:.2f}, {:.2f}]", i_pg.est, i_pg.lo, i_pg.hi), px0 + 6, pg.y0 + 6, 1, {255, 225, 150});
    cv.text(std::format("BUOYANCY FLUX F: EXPONENT {:.2f} [{:.2f}, {:.2f}] (MTT 0)", i_pF.est, i_pF.lo, i_pF.hi), px0 + 6, pg.y0 + 18, 1, {255, 225, 150});
    cv.text(std::format("DOTS: POOLED TIME AVERAGES, {} SEEDS EACH; BOLD: THE FIT'S HEIGHTS, {:.0f} TO {:.0f} CELLS", cfg[0].runs.size(), lo, hi), lw + 20, H - 22, 1,
            {170, 170, 176});
    cv.text("DASHED: MTT'S POWER LAW THROUGH THE BOLD DOTS (SLOPE FIXED)", lw + 20, H - 10, 1, {170, 170, 176});
    if (std::fwrite(cv.rgb.data(), 1, cv.rgb.size(), pipe) != cv.rgb.size()) throw std::runtime_error("ffmpeg stopped reading");
    if (f % 60 == 0) std::println("  frame {} ({:.0f} s)", f, now() - t0);
  }
  if (pclose(pipe) != 0) throw std::runtime_error("ffmpeg failed");
  std::println("wrote {} ({:.0f} s)", out.string(), now() - t0);
  return 0;
}

}  // namespace nfx::tools::plume3d
