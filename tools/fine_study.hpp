// Study G, design G1 (docs/DCM.md): DCM-fine, the owner's context mixer in place of the rollout detail layer's lock.
// The pipeline behind nvfx_dcm's record, experts, search-fine, train-fine, eval-fine and bench-experts, and behind
// nvfx_experiment g-fine. Data (pixel rows, windows, mixers) lives under NEURALVFX_DATA/g/fine, never in git; CSVs go to
// results/experiments/g_fine_*.csv.
#pragma once

#include <neuralfx/dcm/fine.hpp>
#include <neuralfx/sim.hpp>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace nfx::fine_study {

struct Ctx {
  std::filesystem::path data;     // pixel rows, windows, mixers (NEURALVFX_DATA/g/fine)
  std::filesystem::path models;   // study D's trained rollout effects (read only)
  std::filesystem::path results;  // CSVs (in git)
  std::filesystem::path figures;  // docs/figures
  int threads = 2;
  bool quick = false;
  std::uint64_t seed = 0;         // search seed
  std::string family = "hand+macro";
  std::string domain = "linear";
  int configs = 200, refine = 2;
  int max_rows = 0;               // cap on search rows (0: all)
  double budget_ms = 1.0;         // the search's cost budget per 128 x 128 frame; <= 0: none (results labelled family/free)
};

// 1. Pixel rows of 48 training runs (salt 1) and 16 validation runs (salt 3) at 128 px, and the own-rollout windows of
//    the training runs.
void record(const Ctx& c, sim::Effect e);
// 2. The spec (bins, k-means of regional statistics) from the training rows; each expert alone as a predictor on the
//    validation rows; the mutual information of the macro clusters with the hand-made contexts.
void experts(const Ctx& c, sim::Effect e);
// 3. The pilot: the default mixer against the v1 lock used as a predictor, held-out bits on validation runs.
void pilot(const Ctx& c, sim::Effect e);
// 4. The nested mixer search (sites: control bins of the training runs) for a context family, a domain and a seed.
void search(const Ctx& c, sim::Effect e);
// 5. Own-rollout pass, tau and relock on validation settings, re-ranking of the global top 10; releases the mixer.
void train(const Ctx& c, sim::Effect e);
// 6-7. G1c and the test (once): held-out settings and tracking runs, v1 against DCM-fine through the reference.
void eval(const Ctx& c, sim::Effect e);
// Cost: microseconds per pixel of every expert group, context and the mixer, ms per 128 x 128 frame.
void bench(const Ctx& c, sim::Effect e);
// A generation diagnostic: mean and peak fine heat of v1 and of a mixer from the first validation setting.
void probe(const Ctx& c, sim::Effect e, const std::string& mixer, double tau, int relock, int frames);
// Paired-bootstrap summary of the CSVs (docs/DCM.md G1): printed and written to results/.../g_fine_summary.md.
void summary(const Ctx& c);
std::string summary_markdown(const std::filesystem::path& results);  // the same tables, for SUMMARY.md

}  // namespace nfx::fine_study
