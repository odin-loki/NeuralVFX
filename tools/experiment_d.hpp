// Study D of nvfx_experiment: start points and learned dynamics (tools/experiment_d.cpp).
#pragma once

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <ostream>
#include <string>

namespace nfx::study_d {

struct Ctx {
  std::filesystem::path data;     // runs, models, videos (outside git)
  std::filesystem::path results;  // CSVs (in git)
  int threads = 4;
  bool quick = false;
  std::string effects;  // comma-separated effects ("fire,smoke", or later ones: "steam,magic"); empty = study D's three
  std::filesystem::path figures = "docs/figures";  // d-eval's comparison sheets
  bool stages12 = false;  // d-train: stepper stages 1 and 2 only, as study D's models were first trained (then d-tune,
                          // d-finish), so each step's result is saved
  // d-eval: the test's seeds from this base instead of the studies' (0: theirs). The held-out runs keep their settings
  // (salt 2) and take seeds base + run; the real runs at the held-out settings base + 100000 + setting and
  // base + 110000 + setting, the neural runs base + 120000 + setting, the coarse simulation's base + 130000 + setting,
  // the long runs base + 140000 + setting. The water effects' test (docs/EFFECTS_WATER.md) uses 7000000: seeds no study
  // had used. d-chaos too: its runs keep their settings (salt 77) and take seeds base + 150000 + run (the other seed
  // base + 150999 + run); the earth effects (docs/EFFECTS_EARTH.md) use 6000000.
  std::uint64_t seed_base = 0;
  int iters(int full) const { return quick ? std::max(50, full / 20) : full; }
};

void step_chaos(const Ctx& c);
void step_train(const Ctx& c);
void step_tune(const Ctx& c);    // stage 3 of the stepper (activity) on models trained by an older d-train
void step_finish(const Ctx& c);  // renderer, detail layer and start points again, on trained steppers
void step_eval(const Ctx& c);
void step_timing(const Ctx& c);
void report(const Ctx& c, std::ostream& md);  // the D section of SUMMARY.md

}  // namespace nfx::study_d
