// Training samples for the inverse network from the built-in simulation (include/neuralfx/footage.hpp): frames drawn by
// the simulator's own renderer with the true fine fields, some degraded as a camera would.
#include <neuralfx/footage.hpp>
#include <neuralfx/rollout_train.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <random>
#include <thread>

namespace nfx::footage {

std::vector<InverseSample> simulate_samples(sim::Effect e, const SampleOptions& o) {
  rollout::SimRecipe r = rollout::recipe_for(e);
  r.salt = o.salt;
  const bool loops = sim::effect_loops(e);
  const int first = loops ? 20 : 0, last = loops ? 240 : 89;  // frames [first, last)
  std::vector<std::vector<InverseSample>> per(static_cast<std::size_t>(o.runs));
  std::atomic<int> next{0};
  const auto work = [&] {
    for (int i; (i = next++) < o.runs;) {
      sim::Params p = rollout::recipe_run(r, static_cast<std::uint64_t>(i));
      p.size = o.size;
      std::mt19937_64 rng(o.salt * 0x2545F4914F6CDD1DULL + static_cast<std::uint64_t>(i) * 7919ULL + 17);
      std::vector<int> at(static_cast<std::size_t>(o.per_run));
      for (int& f : at) f = first + static_cast<int>(rng() % static_cast<std::uint64_t>(last - first));
      std::ranges::sort(at);
      sim::Fluid fluid(p);
      std::vector<std::uint8_t> rgba(static_cast<std::size_t>(o.size) * static_cast<std::size_t>(o.size) * 4);
      std::size_t k = 0;
      for (int f = 0; k < at.size(); ++f) {
        fluid.step_frame();  // state after f + 1 frames
        while (k < at.size() && at[k] == f) {
          fluid.render(rgba);
          Degradation d;
          std::uniform_real_distribution<float> u(0.f, 1.f);
          if (u(rng) < o.degrade_share) {
            if (u(rng) < 0.5f) d.blur = 1.2f * u(rng);
            if (u(rng) < 0.5f) d.noise = 3.f / 255.f * u(rng);
          }
          d.drop_alpha = o.drop_alpha;
          std::vector<std::uint8_t> shown = rgba;
          degrade(shown, o.size, d, rng());
          const sim::State st = fluid.state();
          InverseSample s;
          s.size = o.size;
          s.frame = to_float(shown);
          s.truth.size = o.size;
          s.truth.heat = st.temp;
          s.truth.soot = st.soot;
          per[static_cast<std::size_t>(i)].push_back(std::move(s));
          ++k;
        }
      }
    }
  };
  {
    std::vector<std::jthread> pool;
    for (int t = 0; t < std::max(1, o.threads); ++t) pool.emplace_back(work);
  }
  std::vector<InverseSample> out;
  for (auto& v : per) {
    for (auto& s : v) out.push_back(std::move(s));
  }
  return out;
}

std::vector<MotionSample> simulate_motion_samples(sim::Effect e, const Inverse& inv, const SampleOptions& o, int cells) {
  rollout::SimRecipe r = rollout::recipe_for(e);
  r.salt = o.salt;
  const bool loops = sim::effect_loops(e);
  const int first = loops ? 20 : 2, last = loops ? 240 : 89;  // frames [first, last), each with the two before it
  constexpr int R = 32;
  std::vector<std::vector<MotionSample>> per(static_cast<std::size_t>(o.runs));
  std::atomic<int> next{0};
  const auto work = [&] {
    for (int i; (i = next++) < o.runs;) {
      sim::Params p = rollout::recipe_run(r, static_cast<std::uint64_t>(i));
      p.size = o.size;
      std::mt19937_64 rng(o.salt * 0x9E3779B97F4A7C15ULL + static_cast<std::uint64_t>(i) * 104729ULL + 23);
      std::vector<int> at(static_cast<std::size_t>(o.per_run));
      for (int& f : at) f = first + static_cast<int>(rng() % static_cast<std::uint64_t>(last - first));
      std::ranges::sort(at);
      sim::Fluid fluid(p);
      const std::size_t bytes = static_cast<std::size_t>(o.size) * static_cast<std::size_t>(o.size) * 4;
      std::array<std::vector<std::uint8_t>, 3> ring;
      for (auto& b : ring) b.resize(bytes);  // resize, not assign: GCC 14 warns falsely on assign here
      std::size_t k = 0;
      for (int f = 0; k < at.size(); ++f) {
        fluid.step_frame();  // state after f + 1 frames
        fluid.render(ring[static_cast<std::size_t>(f % 3)]);
        while (k < at.size() && at[k] == f) {
          ++k;
          if (f < 2) continue;
          Degradation d;
          std::uniform_real_distribution<float> u(0.f, 1.f);
          if (u(rng) < o.degrade_share) {
            if (u(rng) < 0.5f) d.blur = 1.2f * u(rng);
            if (u(rng) < 0.5f) d.noise = 3.f / 255.f * u(rng);
          }
          d.drop_alpha = !inv.spec.alpha;
          std::vector<Fields> three;
          for (int q = 2; q >= 0; --q) {
            std::vector<std::uint8_t> shown = ring[static_cast<std::size_t>((f - q) % 3)];
            degrade(shown, o.size, d, rng());
            three.push_back(apply_inverse(inv, to_float(shown), o.size));
          }
          const Flow a = block_flow(three[0], three[1], R, inv.scale, {}), b = block_flow(three[1], three[2], R, inv.scale, {});
          const std::vector<float> in = motion_inputs(inv.motion, three, a, b, R);
          std::vector<float> truth(static_cast<std::size_t>(R) * R * rollout::kPhys);
          rollout::coarse_from_sim(fluid.state(), R, p.fps, truth);
          // cells: half near material (coarse heat or soot of the last frame within two cells), half anywhere
          const std::vector<float> c = coarse_fields(three[2], R);
          std::vector<int> near, any;
          for (int y = 0; y < R; ++y) {
            for (int x = 0; x < R; ++x) {
              float mass = 0.f;
              for (int dy = -2; dy <= 2; ++dy) {
                for (int dx = -2; dx <= 2; ++dx) {
                  const int xx = x + dx, yy = y + dy;
                  if (xx >= 0 && yy >= 0 && xx < R && yy < R) mass += c[static_cast<std::size_t>(yy * R + xx) * 2] / inv.scale[0] + c[static_cast<std::size_t>(yy * R + xx) * 2 + 1] / inv.scale[1];
                }
              }
              any.push_back(y * R + x);
              if (mass > 0.05f) near.push_back(y * R + x);
            }
          }
          MotionSample s;
          for (int q = 0; q < cells; ++q) {
            const std::vector<int>& from = (q % 2 == 0 && !near.empty()) ? near : any;
            const int cell = from[rng() % from.size()];
            const float* row = in.data() + static_cast<std::size_t>(cell) * Motion::kInputs;
            s.inputs.insert(s.inputs.end(), row, row + Motion::kInputs);
            s.target.push_back(truth[static_cast<std::size_t>(cell) * rollout::kPhys]);
            s.target.push_back(truth[static_cast<std::size_t>(cell) * rollout::kPhys + 1]);
          }
          per[static_cast<std::size_t>(i)].push_back(std::move(s));
        }
      }
    }
  };
  {
    std::vector<std::jthread> pool;
    for (int t = 0; t < std::max(1, o.threads); ++t) pool.emplace_back(work);
  }
  std::vector<MotionSample> out;
  for (auto& v : per) {
    for (auto& s : v) out.push_back(std::move(s));
  }
  return out;
}

}  // namespace nfx::footage
