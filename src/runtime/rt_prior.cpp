// The prior's file reader and schedule (rt_prior.hpp). Baseline ISA: nothing here is per frame.
#include "rt_prior.hpp"

#include <neuralfx/binio.hpp>

#include <cmath>
#include <cstring>
#include <numbers>
#include <spanstream>

namespace nfx::rt {

namespace {

constexpr char kMagic[8] = {'N', 'V', 'F', 'X', 'D', 'D', 'P', 'M'};
constexpr std::uint32_t kVersion = 1;

std::size_t z(int v) { return static_cast<std::size_t>(v); }

// The weight offsets of dcm::ddpm::layout, in the same order.
void lay_out(PriorNet& n) {
  std::size_t o = 0;
  const auto take = [&](std::size_t k) {
    const std::size_t r = o;
    o += k;
    return r;
  };
  const std::size_t I = z(n.inputs()), C0 = z(n.c0), C1 = z(n.c1), C2 = z(n.c2);
  n.stem_w = take(9 * I * C0);
  n.stem_b = take(C0);
  const int widths[PriorNet::kBlocks] = {n.c0, n.c1, n.c2, n.c2, n.c1, n.c0};
  const int sides[PriorNet::kBlocks] = {n.res, n.res / 2, n.res / 4, n.res / 4, n.res / 2, n.res};
  std::size_t film = 0;
  for (int k = 0; k < PriorNet::kBlocks; ++k) {
    PriorNet::Block& B = n.blocks[z(k)];
    const std::size_t C = z(widths[k]);
    B.width = widths[k];
    B.side = sides[k];
    B.wa = take(9 * C * C);
    B.ba = take(C);
    B.wb = take(9 * C * C);
    B.bb = take(C);
    B.film = film;
    film += 2 * C;
  }
  n.down1_w = take(C0 * C1);
  n.down1_b = take(C1);
  n.down2_w = take(C1 * C2);
  n.down2_b = take(C2);
  n.up2_w = take(C2 * C1);
  n.up2_b = take(C1);
  n.up1_w = take(C1 * C0);
  n.up1_b = take(C0);
  n.out_w = take(9 * C0 * z(n.channels));
  n.out_b = take(z(n.channels));
  n.film_size = film;
  n.mlp1_w = take(z(n.embed()) * z(n.film_hidden));
  n.mlp1_b = take(z(n.film_hidden));
  n.mlp2_w = take(z(n.film_hidden) * film);
  n.mlp2_b = take(film);
  n.size = o;
}

}  // namespace

double PriorNet::macs() const {
  const double R0 = res, R1 = res / 2, R2 = res / 4;
  const double n0 = R0 * R0, n1 = R1 * R1, n2 = R2 * R2;
  double m = n0 * 9.0 * inputs() * c0;
  m += 2.0 * 9.0 * (n0 * c0 * c0 * 2 + n1 * c1 * c1 * 2 + n2 * c2 * c2 * 2);
  m += n1 * c0 * c1 + n2 * c1 * c2 + n2 * c2 * c1 + n1 * c1 * c0;
  m += n0 * 9.0 * c0 * channels;
  m += static_cast<double>(embed()) * film_hidden + static_cast<double>(film_hidden) * static_cast<double>(film_size);
  return m;
}

std::expected<PriorNet, std::string> parse_prior(std::span<const char> bytes) {
  std::ispanstream in(bytes);
  char magic[8] = {};
  in.read(magic, 8);
  if (!in || std::memcmp(magic, kMagic, 8) != 0) return std::unexpected("not a denoiser file (NVFXDDPM)");
  const auto ver = bin::get<std::uint32_t>(in);
  if (!ver || *ver != kVersion) return std::unexpected("unsupported denoiser version");
  PriorNet n;
  int* const sizes[9] = {&n.res, &n.channels, &n.c0, &n.c1, &n.c2, &n.cond, &n.freqs, &n.film_hidden, &n.timesteps};
  for (int* v : sizes) {
    const auto r = bin::get<std::int32_t>(in);
    if (!r) return std::unexpected(r.error());
    *v = *r;
  }
  // dcm::ddpm::Config::validate, and bounds that keep every buffer small
  if (n.res < 4 || n.res % 4 != 0 || n.res > 256) return std::unexpected("denoiser: res is a multiple of 4 up to 256");
  for (const int v : {n.channels, n.c0, n.c1, n.c2, n.freqs, n.film_hidden}) {
    if (v < 1 || v > 1024) return std::unexpected("denoiser: channels and widths in [1, 1024]");
  }
  if (n.c0 % 8 != 0 || n.c1 % 8 != 0 || n.c2 % 8 != 0) return std::unexpected("denoiser: the runtime needs widths that are multiples of 8");
  if (n.cond < 0 || n.cond > 64) return std::unexpected("denoiser: condition values in [0, 64]");
  if (n.timesteps < 10 || n.timesteps > 100000) return std::unexpected("denoiser: timesteps in [10, 100000]");
  for (std::vector<float>* v : {&n.scale, &n.lo, &n.hi}) {
    v->resize(z(n.channels));
    for (float& x : *v) {
      const auto r = bin::get<float>(in);
      if (!r) return std::unexpected(r.error());
      x = *r;
    }
  }
  for (const float s : n.scale) {
    if (!(s != 0.f) || !std::isfinite(s)) return std::unexpected("denoiser: a channel scale is zero or not finite");
  }
  lay_out(n);
  const auto count = bin::get<std::uint64_t>(in);
  if (!count || *count != n.size) return std::unexpected("denoiser: weight count does not match the configuration");
  n.w.resize(n.size);
  if (auto r = bin::get_array<float>(in, n.w); !r) return std::unexpected(r.error());
  n.file_bytes = bytes.size();
  return n;
}

// dcm::ddpm::cosine_alpha_bar, one value: the same operations in the same order (each value depends on the one before
// through the clip).
double prior_alpha_bar(int timesteps, int t) {
  constexpr double s = 0.008;
  const auto f = [&](double u) {
    const double v = std::cos(((u / timesteps) + s) / (1.0 + s) * std::numbers::pi / 2.0);
    return v * v;
  };
  double ab = 1.0;
  const double f0 = f(0.0);
  for (int k = 1; k <= t && k <= timesteps; ++k) {
    double v = f(k) / f0;
    const double prev = ab;
    if (1.0 - v / prev > 0.999) v = prev * 0.001;  // beta clip
    ab = v;
  }
  return ab;
}

}  // namespace nfx::rt
