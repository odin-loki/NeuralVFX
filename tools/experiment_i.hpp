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
  // round 2 (docs/COMPOSE.md §10): its data under the data root's i2/; round 1's recorded runs reused (read only)
  std::filesystem::path data2;  // runs, candidates, chosen files, logs of round 2 (outside git)
  std::filesystem::path v2;     // the v2 files (read only): the baselines of round 2
  float anchor = 0.f;           // i2-train: weight of the anchor to v1's plain one-step predictions
  bool aim = false;             // i2-train: windows from forced runs placed where a coupling acts
  bool strong = true;           // i2-train: the strong (scene-like) forced runs join the coupled runs
  std::string split = "val";    // i2-handoff: val or test
};

void step_data(const Ctx& c);   // i-data   plain runs (study D's recipe, salt 1), forced runs, hand-over runs (smoke)
void step_probe(const Ctx& c);  // i-probe  seconds per fine-tuning iteration, plain and coupled
void step_train(const Ctx& c);  // i-train  fine-tune v1's stepper on a mix of plain and coupled runs (one candidate)
void step_val(const Ctx& c);    // i-val    v1 and every candidate on validation; chooses v2c by the rule of §9
void step_test(const Ctx& c);   // i-test   the test, once: v1, v2c and the plain control; the decisions

// Round 2 (docs/COMPOSE.md §10): fire and smoke sturdier under couplings; the explosion's first second.
void step_data2(const Ctx& c);     // i2-data     strong (scene-like) forced runs of fire and smoke (salt 1)
void step_probe2(const Ctx& c);    // i2-probe    v1 under strong pushes on training-salt runs; the anchor's scale
void step_train2(const Ctx& c);    // i2-train    one candidate (--share --lr --anchor --aim --tag), checkpoints
void step_val2(const Ctx& c);      // i2-val      validation of every candidate; the choice by the rule of §10
void step_test2(const Ctx& c);     // i2-test     the test, once, with new seeds; the decisions
void step_handoff(const Ctx& c);   // i2-handoff  the explosion's first second in the simulator's look (--split val|test)
void step_cost2(const Ctx& c);     // i2-cost     the hand-off's cost per frame through the runtime

}  // namespace nfx::study_i
