// Study G of nvfx_experiment: diffusion-context mixing for generating effects (docs/DCM.md, tools/experiment_g.cpp).
#pragma once

#include <algorithm>
#include <filesystem>
#include <ostream>
#include <string>

namespace nfx::study_g {

struct Ctx {
  std::filesystem::path data;     // pixel rows, mixers, videos (outside git)
  std::filesystem::path results;  // CSVs (in git)
  int threads = 4;
  bool quick = false;
  std::string effects;  // comma-separated subset ("fire,smoke"); empty = all
  int iters(int full) const { return quick ? std::max(50, full / 20) : full; }
};

// Each step prints "not yet implemented" until its stage (docs/DCM.md §5) fills it.
void step_data(const Ctx& c);    // g-data    pixel rows (fine experts, contexts, true next fine field) from study D's runs
void step_pilot(const Ctx& c);   // g-pilot   the default mixer against the hand-made detail layer on fire
void step_search(const Ctx& c);  // g-search  the nested leave-one-control-bin-out mixer search, held-out bits per pixel
void step_eval(const Ctx& c);    // g-eval    G1 and G1c on held-out settings through the runtime
void step_timing(const Ctx& c);  // g-timing  milliseconds per frame (quiet machine)
void step_fine(const Ctx& c);    // g-fine    design G1 (DCM-fine) end to end: rows, pilot, cost, search, generation, test
void step_diff(const Ctx& c);       // g-diff       stage S5 on validation: the denoiser as a prior against drift (G2b) and as
                                    //              a source of start points (G2c), against the alternatives without diffusion
void step_diff_test(const Ctx& c);  // g-diff-test  the test protocol, once, for a use that passed validation
void step_prior(const Ctx& c);       // g-prior       round 2: G2b for another looping effect (--effects smoke) on validation
void step_prior_test(const Ctx& c);  // g-prior-test  its test, once, if a prior passed validation
void report(const Ctx& c, std::ostream& md);  // the G section of SUMMARY.md (nothing yet)

}  // namespace nfx::study_g
