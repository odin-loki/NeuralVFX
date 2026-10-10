// The prior against drift in the runtime (src/runtime/rt_prior*.{hpp,cpp}, docs/DCM.md G2.13): the runtime reads the
// trainer's .ddpm format with the trainer's layout; its prior step equals dcm::ddpm::prior_step bit for bit on AVX2 and
// AVX-512 (the reference's own arithmetic) and within a stated tolerance on the baseline; nvfx_render applies it where
// study G did (after every N-th frame of one continuous rollout, counted from the end of the warm-up) and only there;
// the C API checks its arguments. With study G's data present: on the released fire denoiser and study G's test run,
// the same holds, and the runtime's rollout with the prior stays with the reference's.
#include "rt_common.hpp"
#include "rt_prior.hpp"

#include <neuralfx/dcm/ddpm.hpp>
#include <neuralfx/nvfx.h>
#include <neuralfx/rollout.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace dd = nfx::dcm::ddpm;
namespace rt = nfx::rt;
namespace ro = nfx::rollout;

namespace {

// Largest difference the baseline build (no FMA: every multiply-add rounded twice) may show after one prior step, in
// network units. Measured: at most 1.2e-6 on the fire denoiser (112 passes of study G's test run, nvfx_prior parity
// --isa baseline) and 7.7e-7 on the random networks below.
constexpr double kBaselineTolerance = 1e-5;

// A denoiser with every weight random (the zero-initialised layers too), so every path of the network matters.
dd::Denoiser random_denoiser(const dd::Config& c, std::uint64_t seed, float sd) {
  dd::Denoiser d = dd::init_denoiser(c, seed);
  std::vector<float> g(d.w.size());
  dd::gaussian(seed * 7 + 1, g);
  for (std::size_t i = 0; i < g.size(); ++i) d.w[i] += sd * g[i];
  d.scale = {0.4f, 0.4f, 0.5f, 0.3f};
  d.lo = {-6.f, -6.f, -0.5f, -0.5f};
  d.hi = {6.f, 6.f, 8.f, 8.f};
  return d;
}

dd::Config small_config() {
  dd::Config c;
  c.res = 16;
  c.c0 = 8;
  c.c1 = 16;
  c.c2 = 24;  // blocks of 8, 16 and 8 + 16 output channels
  c.freqs = 4;
  c.film_hidden = 8;
  return c;
}

struct IsaPrior {
  nvfx_isa isa;
  const char* name;
  std::unique_ptr<rt::Prior> (*make)(const rt::PriorNet&);
};
// The ISAs this CPU runs.
std::vector<IsaPrior> isas() {
  std::vector<IsaPrior> v;
  for (const IsaPrior& p : {IsaPrior{NVFX_ISA_BASELINE, "baseline", &rt::isa_base::make_prior}, IsaPrior{NVFX_ISA_AVX2, "avx2", &rt::isa_avx2::make_prior},
                            IsaPrior{NVFX_ISA_AVX512, "avx512", &rt::isa_avx512::make_prior}}) {
    if (nvfx_set_isa(p.isa) == NVFX_OK) v.push_back(p);
  }
  nvfx_set_isa(NVFX_ISA_AUTO);
  return v;
}

std::size_t bits_differ(std::span<const float> a, std::span<const float> b) {
  std::size_t n = 0;
  for (std::size_t i = 0; i < a.size(); ++i) n += std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(b[i]);
  return n;
}
double max_abs(std::span<const float> a, std::span<const float> b) {
  double m = 0;
  for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
  return m;
}

rt::PriorNet parse(const dd::Denoiser& d) {
  const std::string bytes = dd::serialise(d);
  auto n = rt::parse_prior(std::span(bytes.data(), bytes.size()));
  EXPECT_TRUE(n) << (n ? "" : n.error());
  return n ? std::move(*n) : rt::PriorNet{};
}

// Every ISA's prior step and predicted noise against the reference, on random states.
void check_steps(const dd::Denoiser& d, std::uint64_t seed) {
  const rt::PriorNet net = parse(d);
  const std::size_t n = static_cast<std::size_t>(d.cfg.res) * static_cast<std::size_t>(d.cfg.res) * static_cast<std::size_t>(d.cfg.channels);
  std::vector<float> x(n);
  dd::gaussian(seed, x);
  for (std::size_t i = 0; i < n; ++i) x[i] = std::clamp(1.5f * x[i] + (i % 4 >= 2 ? 1.f : 0.f), d.lo[i % 4], d.hi[i % 4]);
  const std::vector<float> cond = {0.31f, 0.88f, 0.74f};
  for (const IsaPrior& p : isas()) {
    const auto prior = p.make(net);
    for (const int t : {1, 20, 100, 400}) {
      std::vector<float> want(n), got(n);
      dd::predict_eps(d, x, t, cond, want);
      prior->predict(x, t, cond, got);
      for (const float beta : {1.f, 0.5f, 0.25f}) {
        std::vector<float> a = x, b = x;
        dd::prior_step(d, a, t, beta, cond);
        prior->step(b, t, beta, cond);
        if (p.isa == NVFX_ISA_BASELINE) {
          EXPECT_LE(max_abs(a, b), kBaselineTolerance) << p.name << " t " << t << " beta " << beta;
        } else {
          EXPECT_EQ(bits_differ(a, b), 0u) << p.name << " t " << t << " beta " << beta << ": largest difference " << max_abs(a, b);
        }
      }
      if (p.isa != NVFX_ISA_BASELINE) {
        EXPECT_EQ(bits_differ(want, got), 0u) << p.name << " t " << t;
      }
      EXPECT_GT(max_abs(want, std::vector<float>(n, 0.f)), 0.01) << "the network predicts something";
    }
  }
}

// A small rollout effect (tests/test_rollout.cpp's runtime model): start point 0 grows its fine fields (3 warm-up
// frames), start point 1 has them.
ro::Model rollout_model() {
  ro::Hyper h;
  h.res = 16;
  h.hidden = 8;
  h.memory = 2;
  h.jacobi = 10;
  h.render_hidden = 6;
  h.start_fine = 16;
  h.warmup = 3;
  ro::Model m = ro::init_model(h, 21);
  std::mt19937_64 rng(22);
  std::normal_distribution<float> nd(0.f, 0.1f);
  for (float& w : m.step_w) w += nd(rng);
  for (float& w : m.render_w) w = 0.5f * w + 0.05f;
  m.effect = "test";
  m.control_names = {"intensity", "wind", "turbulence"};
  m.scale = {0.2f, 0.2f, 0.4f, 0.3f};
  m.lo = {-2.f, -2.f, 0.f, 0.f};
  m.hi = {2.f, 2.f, 3.f, 3.f};
  m.render_scale = {1.f, 1.f};
  m.detail.swirl_control = 2;
  m.detail.swirl = 0.6f;
  const int R = h.res;
  for (int k = 0; k < 2; ++k) {
    ro::StartPoint sp;
    sp.controls = {0.3f + 0.4f * static_cast<float>(k), 0.5f, 0.6f};
    sp.seed = 1000 + static_cast<std::uint64_t>(k);
    sp.time = 2.f;
    sp.coarse.resize(static_cast<std::size_t>(R) * R * ro::kPhys);
    for (int y = 0; y < R; ++y) {
      for (int x = 0; x < R; ++x) {
        float* c = sp.coarse.data() + (static_cast<std::size_t>(y) * R + x) * ro::kPhys;
        c[0] = 0.1f * std::sin(0.5f * static_cast<float>(x + k));
        c[1] = 0.2f + 0.1f * std::cos(0.4f * static_cast<float>(y));
        c[2] = std::max(0.f, 0.8f - 0.05f * static_cast<float>(std::abs(x - 8) + y));
        c[3] = std::max(0.f, 0.5f - 0.04f * static_cast<float>(std::abs(x - 7) + std::abs(y - 6)));
      }
    }
    if (k == 1) {
      sp.fine_t.resize(256);
      sp.fine_d.resize(256);
      for (int i = 0; i < 256; ++i) {
        sp.fine_t[static_cast<std::size_t>(i)] = std::max(0.f, 0.9f - 0.03f * static_cast<float>(std::abs(i % 16 - 8) + i / 16));
        sp.fine_d[static_cast<std::size_t>(i)] = 0.3f + 0.2f * std::sin(0.7f * static_cast<float>(i));
      }
    }
    m.starts.push_back(sp);
  }
  ro::quantise_like_storage(m);
  return m;
}

// A denoiser for that effect: its grid, its channel scales, its condition (3 controls).
dd::Denoiser rollout_denoiser(const ro::Model& m) {
  dd::Config c = small_config();
  c.res = m.h.res;
  c.cond = m.h.cond();
  dd::Denoiser d = random_denoiser(c, 31, 0.04f);
  d.scale.assign(m.scale.begin(), m.scale.end());
  d.lo = {-8.f, -8.f, 0.f, 0.f};
  d.hi = {8.f, 8.f, 7.f, 9.f};
  return d;
}

// The reference prior on a rollout state, as tools/experiment_g.cpp's long_run applies it.
void reference_prior(const dd::Denoiser& d, const ro::Model& m, std::span<const float> ctl, ro::State& s, int t, float beta) {
  std::vector<float> x(static_cast<std::size_t>(d.cfg.res * d.cfg.res * d.cfg.channels)), cond(static_cast<std::size_t>(m.h.cond()));
  dd::to_network(d, s.coarse, m.h.channels(), x);
  ro::condition(m, ctl, s.time, cond);
  dd::prior_step(d, x, t, beta, cond);
  dd::to_physical(d, x, m.h.channels(), s.coarse);
  const auto C = static_cast<std::size_t>(m.h.channels());
  for (std::size_t i = 0; i < s.coarse.size(); ++i) {
    if (i % C < ro::kPhys) s.coarse[i] = std::clamp(s.coarse[i], m.lo[i % C], m.hi[i % C]);
  }
}

std::vector<std::uint8_t> to_bytes(std::span<const float> rgba) {
  std::vector<std::uint8_t> out(rgba.size());
  for (std::size_t i = 0; i < rgba.size(); ++i) out[i] = static_cast<std::uint8_t>(rgba[i] * 255.f + 0.5f);
  return out;
}

struct Effect {
  nvfx_effect* e = nullptr;
  explicit Effect(const ro::Model& m) {
    std::ostringstream os;
    EXPECT_TRUE(ro::save_model(os, m));
    const std::string b = os.str();
    EXPECT_EQ(nvfx_effect_load_memory(b.data(), b.size(), &e), NVFX_OK);
  }
  ~Effect() { nvfx_effect_free(e); }
  Effect(const Effect&) = delete;
  Effect& operator=(const Effect&) = delete;
};

std::vector<std::uint8_t> render(nvfx_instance* in, int frame, int size) {
  std::vector<std::uint8_t> px(static_cast<std::size_t>(size) * static_cast<std::size_t>(size) * 4);
  EXPECT_EQ(nvfx_render(in, frame / 30.0, px.data(), static_cast<std::size_t>(size) * 4), NVFX_OK);
  return px;
}

int max_diff(std::span<const std::uint8_t> a, std::span<const std::uint8_t> b) {
  int m = 0;
  for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(int(a[i]) - int(b[i])));
  return m;
}

}  // namespace

TEST(RuntimePrior, ReadsTheTrainersFormatWithTheTrainersLayout) {
  for (const dd::Config& c : {small_config(), dd::Config{}}) {
    const dd::Denoiser d = random_denoiser(c, 5, 0.03f);
    const rt::PriorNet n = parse(d);
    const dd::Layout L = dd::layout(c);
    EXPECT_EQ(n.res, c.res);
    EXPECT_EQ(n.channels, c.channels);
    EXPECT_EQ(n.c0, c.c0);
    EXPECT_EQ(n.c1, c.c1);
    EXPECT_EQ(n.c2, c.c2);
    EXPECT_EQ(n.cond, c.cond);
    EXPECT_EQ(n.freqs, c.freqs);
    EXPECT_EQ(n.film_hidden, c.film_hidden);
    EXPECT_EQ(n.timesteps, c.timesteps);
    EXPECT_EQ(n.scale, d.scale);
    EXPECT_EQ(n.lo, d.lo);
    EXPECT_EQ(n.hi, d.hi);
    EXPECT_EQ(n.w, d.w);
    EXPECT_EQ(n.size, L.size);
    for (const auto& [a, b] : {std::pair{n.stem_w, L.stem_w}, {n.stem_b, L.stem_b}, {n.down1_w, L.down1_w}, {n.down1_b, L.down1_b}, {n.down2_w, L.down2_w},
                               {n.down2_b, L.down2_b}, {n.up2_w, L.up2_w}, {n.up2_b, L.up2_b}, {n.up1_w, L.up1_w}, {n.up1_b, L.up1_b},
                               {n.out_w, L.out_w}, {n.out_b, L.out_b}, {n.mlp1_w, L.mlp1_w}, {n.mlp1_b, L.mlp1_b}, {n.mlp2_w, L.mlp2_w},
                               {n.mlp2_b, L.mlp2_b}, {n.film_size, L.film_size}}) {
      EXPECT_EQ(a, b);
    }
    for (int k = 0; k < dd::kBlocks; ++k) {
      const auto& x = n.blocks[static_cast<std::size_t>(k)];
      const auto& y = L.blocks[static_cast<std::size_t>(k)];
      EXPECT_EQ(x.width, y.width);
      EXPECT_EQ(x.wa, y.wa);
      EXPECT_EQ(x.ba, y.ba);
      EXPECT_EQ(x.wb, y.wb);
      EXPECT_EQ(x.bb, y.bb);
      EXPECT_EQ(x.film, y.film);
    }
    EXPECT_DOUBLE_EQ(n.macs(), dd::forward_macs(c));
    for (const int t : {0, 1, 100, 999, 1000}) EXPECT_EQ(rt::prior_alpha_bar(c.timesteps, t), dd::cosine_alpha_bar(c.timesteps)[static_cast<std::size_t>(t)]);
  }
  // what it refuses
  const std::string good = dd::serialise(random_denoiser(small_config(), 5, 0.03f));
  const auto refused = [](std::string b) { return !rt::parse_prior(std::span(b.data(), b.size())); };
  EXPECT_TRUE(refused(good.substr(0, good.size() - 4)));  // truncated
  EXPECT_TRUE(refused("NVFXMDL1" + good.substr(8)));      // another format
  std::string v2 = good;
  v2[8] = 2;  // another version
  EXPECT_TRUE(refused(v2));
  dd::Config odd = small_config();
  odd.c0 = 12;  // the runtime's kernels need multiples of 8
  EXPECT_TRUE(refused(dd::serialise(random_denoiser(odd, 5, 0.03f))));
}

TEST(RuntimePrior, StepIsTheReferencesBitForBitWithFmaAndCloseWithout) {
  check_steps(random_denoiser(small_config(), 11, 0.05f), 12);
  check_steps(random_denoiser(dd::Config{}, 13, 0.02f), 14);  // the released network's shape: 391,748 weights
}

TEST(RuntimePrior, AppliedToACoarseStateAsStudyGAppliedIt) {
  // A rollout state with memory channels: the runtime's apply() against to_network, prior_step, to_physical and the
  // stepper's clamp (tools/experiment_g.cpp).
  const ro::Model m = rollout_model();
  const dd::Denoiser d = rollout_denoiser(m);
  const rt::PriorNet net = parse(d);
  const std::vector<float> ctl = {0.6f, 0.2f, 0.9f};
  ro::State s = ro::start(m, 0, 32, ctl, 5);
  for (int f = 0; f < 5; ++f) ro::step(m, s, ctl, 5);
  for (std::size_t i = 0; i < s.coarse.size(); ++i) {  // push some values beyond the stepper's range
    if (i % 37 == 0) s.coarse[i] += 2.5f;
  }
  std::vector<float> cond(3);
  ro::condition(m, ctl, s.time, cond);
  for (const IsaPrior& p : isas()) {
    ro::State want = s;
    reference_prior(d, m, ctl, want, 100, 1.f);
    std::vector<float> got = s.coarse;
    p.make(net)->apply(got, m.h.channels(), m.lo, m.hi, 100, 1.f, cond);
    if (p.isa == NVFX_ISA_BASELINE) {
      EXPECT_LE(max_abs(want.coarse, got), kBaselineTolerance) << p.name;
    } else {
      EXPECT_EQ(bits_differ(want.coarse, got), 0u) << p.name;
    }
    EXPECT_GT(max_abs(s.coarse, got), 0.01) << "the prior moves the state";
  }
}

TEST(RuntimePrior, ContinuousInstanceMatchesTheReferenceRolloutWithThePrior) {
  // nvfx_render with the prior against study G's loop (rollout::start, then rollout::step and the prior after every
  // N-th frame), on every ISA. The runtime's rollout is not bit-exact with the reference (it runs the stepper in
  // another order; tests/test_rollout.cpp allows 3 levels); the prior must not widen that.
  const ro::Model m = rollout_model();
  const dd::Denoiser d = rollout_denoiser(m);
  const std::string prior = dd::serialise(d);
  Effect fx(m);
  ASSERT_EQ(nvfx_effect_attach_prior_memory(fx.e, prior.data(), prior.size()), NVFX_OK);
  const int size = 32, every = 4, t = 100, last = 21;
  for (const IsaPrior& p : isas()) {
    ASSERT_EQ(nvfx_set_isa(p.isa), NVFX_OK);
    for (const int start : {0, 1}) {
      const auto& sp = m.starts[static_cast<std::size_t>(start)];
      nvfx_instance* in = nullptr;
      ASSERT_EQ(nvfx_instance_create(fx.e, size, &in), NVFX_OK);
      ASSERT_EQ(nvfx_instance_set_variation(in, start), NVFX_OK);  // start point `start` with its run's seed
      nvfx_instance_set_controls(in, sp.controls.data(), 3);
      ASSERT_EQ(nvfx_instance_set_drift(in, 0.f), NVFX_OK);
      ASSERT_EQ(nvfx_instance_set_prior(in, every, t, 1.f), NVFX_OK);
      ro::State s = ro::start(m, start, size, sp.controls, sp.seed);
      ro::State plain = s;
      std::vector<float> rgba(static_cast<std::size_t>(size) * size * 4);
      int moved = 0;
      for (int f = 0; f <= last; ++f) {
        if (f > 0) {
          ro::step(m, s, sp.controls, sp.seed);
          ro::step(m, plain, sp.controls, sp.seed);
          if (f % every == 0) reference_prior(d, m, sp.controls, s, t, 1.f);
        }
        const auto got = render(in, f, size);
        ro::render(m, s, rgba);
        const auto want = to_bytes(rgba);
        EXPECT_LE(max_diff(got, want), 3) << p.name << " start " << start << " frame " << f;
        ro::render(m, plain, rgba);
        moved = std::max(moved, max_diff(want, to_bytes(rgba)));
      }
      EXPECT_GT(moved, 10) << "the prior changes the picture";
      nvfx_instance_free(in);
    }
  }
  nvfx_set_isa(NVFX_ISA_AUTO);
}

TEST(RuntimePrior, ActsOnlyInOneContinuousRolloutAndKeepsFramesAFunctionOfTime) {
  const ro::Model m = rollout_model();
  const std::string prior = dd::serialise(rollout_denoiser(m));
  Effect with(m), without(m);
  ASSERT_EQ(nvfx_effect_attach_prior_memory(with.e, prior.data(), prior.size()), NVFX_OK);
  const int size = 32;
  const auto make = [&](nvfx_effect* e, float drift) {
    nvfx_instance* in = nullptr;
    EXPECT_EQ(nvfx_instance_create(e, size, &in), NVFX_OK);
    nvfx_instance_set_seed(in, 77);
    nvfx_instance_set_drift(in, drift);
    return in;
  };
  // shards (the default 6 s, and 2 s): the prior is idle
  for (const float drift : {6.f, 2.f}) {
    nvfx_instance *a = make(with.e, drift), *b = make(without.e, drift);
    for (const int f : {0, 16, 32, 47, 52, 75}) EXPECT_EQ(render(a, f, size), render(b, f, size)) << "drift " << drift << " frame " << f;
    nvfx_instance_free(a);
    nvfx_instance_free(b);
  }
  // one continuous rollout: on by default once attached (16, 100, 1); frames before the first pass are untouched
  nvfx_instance *a = make(with.e, 0.f), *b = make(without.e, 0.f), *c = make(with.e, 0.f);
  for (int f = 0; f < 16; ++f) EXPECT_EQ(render(a, f, size), render(b, f, size)) << "frame " << f;
  EXPECT_NE(render(a, 16, size), render(b, 16, size));
  // played frame by frame or sought: the same frames (backwards replays from the start)
  std::vector<std::uint8_t> played;
  for (int f = 17; f <= 40; ++f) played = render(a, f, size);
  EXPECT_EQ(render(c, 40, size), played);
  EXPECT_EQ(render(a, 20, size), render(c, 20, size));
  // off again: the plain rollout
  ASSERT_EQ(nvfx_instance_set_prior(a, 0, 100, 1.f), NVFX_OK);
  EXPECT_EQ(render(a, 40, size), render(b, 40, size));
  // another setting: another run
  ASSERT_EQ(nvfx_instance_set_prior(a, 8, 50, 0.5f), NVFX_OK);
  EXPECT_NE(render(a, 40, size), render(c, 40, size));
  ASSERT_EQ(nvfx_instance_set_prior(a, 16, 100, 1.f), NVFX_OK);
  EXPECT_EQ(render(a, 40, size), played);
  for (nvfx_instance* in : {a, b, c}) nvfx_instance_free(in);
}

TEST(RuntimePrior, ApiChecksItsArguments) {
  const ro::Model m = rollout_model();
  const dd::Denoiser d = rollout_denoiser(m);
  const std::string prior = dd::serialise(d);
  Effect fx(m);
  nvfx_instance* early = nullptr;
  ASSERT_EQ(nvfx_instance_create(fx.e, 32, &early), NVFX_OK);  // created before the prior is attached
  const std::size_t plain_scratch = nvfx_instance_scratch_bytes(early);
  nvfx_effect_info info{};
  nvfx_effect_get_info(fx.e, &info);
  const std::size_t resident = info.resident_bytes;
  EXPECT_EQ(nvfx_instance_set_prior(early, 16, 100, 1.f), NVFX_ERROR_ARGUMENT);  // no prior attached
  EXPECT_EQ(nvfx_instance_set_prior(early, 0, 100, 1.f), NVFX_OK);               // off is always fine
  EXPECT_EQ(nvfx_effect_attach_prior(nullptr, "x"), NVFX_ERROR_ARGUMENT);
  EXPECT_EQ(nvfx_effect_attach_prior(fx.e, nullptr), NVFX_ERROR_ARGUMENT);
  EXPECT_EQ(nvfx_effect_attach_prior(fx.e, "/nonexistent/fire.ddpm"), NVFX_ERROR_IO);
  EXPECT_EQ(nvfx_effect_attach_prior_memory(fx.e, "garbage!garbage!", 16), NVFX_ERROR_FORMAT);
  {
    dd::Denoiser other = d;  // another grid
    other.cfg.res = 32;
    other.w.assign(dd::layout(other.cfg).size, 0.01f);
    const std::string b = dd::serialise(other);
    EXPECT_EQ(nvfx_effect_attach_prior_memory(fx.e, b.data(), b.size()), NVFX_ERROR_FORMAT);
    other = d;  // another condition (a one-shot effect's age features)
    other.cfg.cond = 5;
    other.w.assign(dd::layout(other.cfg).size, 0.01f);
    const std::string c = dd::serialise(other);
    EXPECT_EQ(nvfx_effect_attach_prior_memory(fx.e, c.data(), c.size()), NVFX_ERROR_FORMAT);
  }
  ASSERT_EQ(nvfx_effect_attach_prior_memory(fx.e, prior.data(), prior.size()), NVFX_OK);
  EXPECT_EQ(nvfx_effect_attach_prior_memory(fx.e, prior.data(), prior.size()), NVFX_ERROR_ARGUMENT);  // one prior per effect
  nvfx_effect_get_info(fx.e, &info);
  EXPECT_EQ(info.resident_bytes, resident + 4 * (d.w.size() + 12));
  // the early instance gets its buffers from set_prior; a later one has them from the start
  EXPECT_EQ(nvfx_instance_scratch_bytes(early), plain_scratch);
  EXPECT_EQ(nvfx_instance_set_prior(early, 16, 100, 1.f), NVFX_OK);
  EXPECT_GT(nvfx_instance_scratch_bytes(early), plain_scratch);
  nvfx_instance* later = nullptr;
  ASSERT_EQ(nvfx_instance_create(fx.e, 32, &later), NVFX_OK);
  EXPECT_EQ(nvfx_instance_scratch_bytes(later), nvfx_instance_scratch_bytes(early));
  EXPECT_EQ(nvfx_instance_set_prior(later, -1, 100, 1.f), NVFX_ERROR_ARGUMENT);
  EXPECT_EQ(nvfx_instance_set_prior(later, 16, 0, 1.f), NVFX_ERROR_ARGUMENT);
  EXPECT_EQ(nvfx_instance_set_prior(later, 16, 1001, 1.f), NVFX_ERROR_ARGUMENT);
  EXPECT_EQ(nvfx_instance_set_prior(later, 16, 100, 1.5f), NVFX_ERROR_ARGUMENT);
  EXPECT_EQ(nvfx_instance_set_prior(later, 16, 100, -0.1f), NVFX_ERROR_ARGUMENT);
  EXPECT_EQ(nvfx_instance_set_prior(later, 16, 100, std::nanf("")), NVFX_ERROR_ARGUMENT);
  EXPECT_EQ(nvfx_instance_set_prior(nullptr, 16, 100, 1.f), NVFX_ERROR_ARGUMENT);
  EXPECT_EQ(nvfx_instance_set_prior(later, 1, 1000, 0.f), NVFX_OK);
  nvfx_instance_free(early);
  nvfx_instance_free(later);
  // frame models have no coarse state
  nfx::Hyper g;
  g.arch = nfx::Arch::grid;
  g.size = 32;
  g.frames = 8;
  g.n_controls = 3;
  g.bases = 1;
  g.grid_t = 2;
  g.grid = 8;
  g.channels = 4;
  g.hidden = 8;
  std::ostringstream os;
  ASSERT_TRUE(nfx::save_model(os, nfx::init_model(g, 1)));
  const std::string b = os.str();
  nvfx_effect* frame_model = nullptr;
  ASSERT_EQ(nvfx_effect_load_memory(b.data(), b.size(), &frame_model), NVFX_OK);
  EXPECT_EQ(nvfx_effect_attach_prior_memory(frame_model, prior.data(), prior.size()), NVFX_ERROR_ARGUMENT);
  nvfx_instance* fm = nullptr;
  ASSERT_EQ(nvfx_instance_create(frame_model, 32, &fm), NVFX_OK);
  EXPECT_EQ(nvfx_instance_set_prior(fm, 16, 100, 1.f), NVFX_ERROR_ARGUMENT);
  nvfx_instance_free(fm);
  nvfx_effect_free(frame_model);
}

// --- with study G's data (skipped without it) ---------------------------------------------------------------------------

namespace {

std::filesystem::path data_root() {
  if (const char* d = std::getenv("NEURALVFX_DATA")) return d;
  const char* home = std::getenv("HOME");
  return std::filesystem::path(home ? home : ".") / "nvfx-data";
}

}  // namespace

TEST(RuntimePrior, FireReproducesStudyGsTestRun) {
  // The released fire effect and denoiser (docs/DCM.md G2.6), study G's test run at study B's held-out setting 1
  // (0.312 / 0.882 / 0.737), model seed 2,970,000, the start point nearest the controls, 128 px, the prior every 16
  // frames at t = 100 with beta = 1, for 96 frames (6 passes). At every pass the runtime's prior step on the runtime's
  // own state equals the reference's (bits on AVX2 and AVX-512, kBaselineTolerance on the baseline); the runtime's
  // rollout with the prior stays within 1e-4 network units and 2 levels of 255 of the reference's rollout with it
  // (measured: about 1e-5 and 1 level).
  const auto model_path = data_root() / "experiments" / "models" / "d" / "fire.nvfx";
  const auto prior_path = data_root() / "g" / "diff" / "fire.ddpm";
  if (!std::filesystem::exists(model_path) || !std::filesystem::exists(prior_path)) GTEST_SKIP() << "study G's data is not in " << data_root();
  auto m = ro::load_model(model_path);
  ASSERT_TRUE(m) << m.error();
  auto d = dd::load(prior_path);
  ASSERT_TRUE(d) << d.error();
  std::ifstream f(prior_path, std::ios::binary);
  const std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  auto net = rt::parse_prior(std::span(bytes.data(), bytes.size()));
  ASSERT_TRUE(net) << net.error();
  EXPECT_EQ(net->size, 391748u);
  std::vector<float> ctl;
  {  // study B's held-out setting 1: the first draw that is off study B's training grid (tools/experiment_g.cpp)
    std::mt19937_64 rng(2026);
    std::uniform_real_distribution<float> u(0.05f, 0.95f);
    const auto off = [](float x, std::initializer_list<float> g) { return std::ranges::all_of(g, [x](float y) { return std::abs(x - y) >= 0.05f; }); };
    while (ctl.empty()) {
      const float a = u(rng), b = u(rng), c = u(rng);
      if (off(a, {0.f, 0.5f, 1.f}) && off(b, {0.f, 0.25f, 0.5f, 0.75f, 1.f}) && off(c, {0.f, 0.5f, 1.f})) ctl = {a, b, c};
    }
    ASSERT_NEAR(ctl[0], 0.312, 1e-3);
    ASSERT_NEAR(ctl[1], 0.882, 1e-3);
    ASSERT_NEAR(ctl[2], 0.737, 1e-3);
  }
  int idx = 0;
  float best = 1e30f;
  for (std::size_t k = 0; k < m->starts.size(); ++k) {
    float dist = 0;
    for (std::size_t q = 0; q < 3; ++q) dist += (m->starts[k].controls[q] - ctl[q]) * (m->starts[k].controls[q] - ctl[q]);
    if (dist < best) {
      best = dist;
      idx = static_cast<int>(k);
    }
  }
  const std::uint64_t seed = 2970000;
  const int size = 128, every = 16, frames = 96;
  rt::RolloutEffect re;
  re.m = *m;
  std::vector<float> cond(3), rgba(static_cast<std::size_t>(size) * size * 4);
  std::vector<std::uint8_t> px(rgba.size());
  for (const IsaPrior& p : isas()) {
    ro::State ref = ro::start(*m, idx, size, ctl, seed);
    std::unique_ptr<rt::RolloutRunner> run = p.isa == NVFX_ISA_AVX512 ? rt::isa_avx512::make_rollout(re, size)
                                             : p.isa == NVFX_ISA_AVX2  ? rt::isa_avx2::make_rollout(re, size)
                                                                       : rt::isa_base::make_rollout(re, size);
    const auto prior = p.make(*net);
    run->start(idx, ctl, seed);
    for (int k = 1; k <= frames; ++k) {
      ro::step(*m, ref, ctl, seed);
      run->step(ctl, seed);
      if (k % every != 0) continue;
      ro::State same = ref;  // the runtime's state through the reference's prior
      same.coarse.assign(run->coarse().begin(), run->coarse().end());
      same.time = run->time();
      reference_prior(*d, *m, ctl, same, 100, 1.f);
      ro::condition(*m, ctl, run->time(), cond);
      prior->apply(run->coarse_mut(), m->h.channels(), m->lo, m->hi, 100, 1.f, cond);
      if (p.isa == NVFX_ISA_BASELINE) {
        EXPECT_LE(max_abs(same.coarse, run->coarse()), kBaselineTolerance) << p.name << " frame " << k;
      } else {
        EXPECT_EQ(bits_differ(same.coarse, run->coarse()), 0u) << p.name << " frame " << k;
      }
      reference_prior(*d, *m, ctl, ref, 100, 1.f);
      double apart = 0;  // the two rollouts, in network units
      const auto C = static_cast<std::size_t>(m->h.channels());
      for (std::size_t i = 0; i < ref.coarse.size(); ++i) {
        const double unit = i % C < ro::kPhys ? static_cast<double>(d->scale[i % C]) : 1.0;
        apart = std::max(apart, std::abs(static_cast<double>(ref.coarse[i]) - static_cast<double>(run->coarse()[i])) / unit);
      }
      EXPECT_LE(apart, 1e-4) << p.name << " frame " << k;
      run->render(rt::FrameInput{}, px.data(), static_cast<std::size_t>(size) * 4);
      ro::render(*m, ref, rgba);
      EXPECT_LE(max_diff(px, to_bytes(rgba)), 2) << p.name << " frame " << k;
    }
  }
}
