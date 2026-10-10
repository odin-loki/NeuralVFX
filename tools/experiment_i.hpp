// Study I of nvfx_experiment: couplings in training (docs/COMPOSE.md §9, tools/experiment_i.cpp).
#pragma once

#include <filesystem>
#include <ostream>
#include <string>

namespace nfx::study_i {

struct Ctx {
  std::filesystem::path data;     // runs, candidate and chosen models (outside git): the data root's i/
  std::filesystem::path v1;       // study D's frozen models (read only)
  std::filesystem::path results;  // CSVs (in git)
  int threads = 1;
  bool quick = false;
  std::string effects;  // comma-separated subset; empty = all
  // i-train
  float share = 0.5f;   // probability that a window comes from a coupled run
  float lr = 3e-4f;
  int iters = 400, checkpoint = 100, batch = 16;
  int stop = 0;         // stop after this many iterations of the `iters` schedule (the plain control)
  std::string tag;      // candidate name; empty: from share and lr
};

void step_data(const Ctx& c);   // i-data   plain runs (study D's recipe, salt 1), forced runs, hand-over runs (smoke)
void step_probe(const Ctx& c);  // i-probe  seconds per fine-tuning iteration, plain and coupled
void step_train(const Ctx& c);  // i-train  fine-tune v1's stepper on a mix of plain and coupled runs (one candidate)
void step_val(const Ctx& c);    // i-val    v1 and every candidate on validation; chooses v2c by the rule of §9
void step_test(const Ctx& c);   // i-test   the test, once: v1, v2c and the plain control; the decisions

}  // namespace nfx::study_i
