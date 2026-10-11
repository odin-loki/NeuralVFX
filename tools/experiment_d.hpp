// Study D of nvfx_experiment: start points and learned dynamics (tools/experiment_d.cpp).
#pragma once

#include <algorithm>
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
