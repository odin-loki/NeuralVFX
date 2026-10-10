// Study H, H2: thread CPU time of N rollout steps of one effect at one size, built against main's runtime or this
// branch's, to compare the two in separate processes (tools/study_h/h2_vs_main.sh). Not part of the CMake build:
//   g++-14 -O3 -std=c++23 -DNFX_HAS_SKIP -Iinclude -Isrc/compose -Isrc/runtime tools/study_h/ab_step.cpp -o ab_new
//          build/libneuralfx_compose.a build/libnvfx.a build/libneuralfx_model.a -lpthread   (one line)
//   (against main: the same without -DNFX_HAS_SKIP, with main's include directories and libraries)
// Usage: ab_step EFFECT.nvfx SIZE SKIP STEPS  ->  prints "SKIP ms-per-step"
#include "compose.hpp"

#include <neuralfx/rollout.hpp>

#include <cstdio>
#include <cstdlib>
#include <ctime>

using namespace nfx;

static double cpu() {
  timespec t{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
  return static_cast<double>(t.tv_sec) + 1e-9 * static_cast<double>(t.tv_nsec);
}

int main(int argc, char** argv) {
  if (argc < 5) return 2;
  auto m = rollout::load_model(argv[1]);
  if (!m) return 1;
  rt::RolloutEffect e;
  e.m = std::move(*m);
  const int size = std::atoi(argv[2]), steps = std::atoi(argv[4]);
  const bool skip = std::atoi(argv[3]) != 0;
  auto r = compose::make_runner(e, size, compose::Isa::avx2);
#ifdef NFX_HAS_SKIP
  r->skip_empty(skip);
#endif
  r->start(0, e.m.starts[0].controls, 11);
  const double t0 = cpu();
  for (int f = 0; f < steps; ++f) r->step(e.m.starts[0].controls, 11);
  std::printf("%d %.4f\n", skip ? 1 : 0, (cpu() - t0) * 1e3 / steps);
}
