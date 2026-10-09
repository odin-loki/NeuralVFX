// Study G (docs/DCM.md): diffusion-context mixing for generating effects. Part of nvfx_experiment. The steps are
// registered as g-data, g-pilot, g-search, g-eval and g-timing, and later stages fill them in; g-fine runs design G1
// (DCM-fine, tools/fine_study.cpp) end to end.
#include "experiment_g.hpp"
#include "fine_study.hpp"

#include <neuralfx/clip.hpp>

#include <filesystem>
#include <print>

namespace nfx::study_g {

namespace {

void not_yet(std::string_view step) { std::println("{}: not yet implemented", step); }

}  // namespace

void step_data(const Ctx&) { not_yet("g-data"); }
void step_pilot(const Ctx&) { not_yet("g-pilot"); }
void step_search(const Ctx&) { not_yet("g-search"); }
void step_eval(const Ctx&) { not_yet("g-eval"); }
void step_timing(const Ctx&) { not_yet("g-timing"); }

// G1 (docs/DCM.md): per effect, the pixel rows, the spec, the cost of each part, the nested search, the own-rollout
// pass with tau and relock on validation settings, and the test; on fire also the pilot, the search's seed noise and
// the context families. Each piece is also an nvfx_dcm subcommand.
void step_fine(const Ctx& c) {
  fine_study::Ctx f;
  f.data = c.data / "fine";
  f.models = (c.quick ? c.data.parent_path().parent_path() : c.data.parent_path()) / "experiments" / "models" / "d";
  f.results = c.results;
  f.figures = c.quick ? f.data / "figures" : std::filesystem::path("docs/figures");
  f.threads = std::min(c.threads, 2);
  f.quick = c.quick;
  f.max_rows = 100000;
  for (const auto e : sim::kEffects) {
    const std::string name(sim::effect_name(e));
    if (!c.effects.empty() && c.effects.find(name) == std::string::npos) continue;
    if (!std::filesystem::exists(f.data / (name + "_train.rows"))) fine_study::record(f, e);
    fine_study::experts(f, e);
    if (e == sim::Effect::fire) fine_study::pilot(f, e);
    fine_study::bench(f, e);
    fine_study::search(f, e);
    if (e == sim::Effect::fire) {
      for (const std::uint64_t seed : {1, 2}) {
        fine_study::Ctx s = f;
        s.seed = seed;
        fine_study::search(s, e);
      }
      for (const std::string family : {"none", "hand"}) {
        fine_study::Ctx s = f;
        s.family = family;
        fine_study::search(s, e);
      }
    }
    fine_study::train(f, e);
    fine_study::eval(f, e);
  }
  fine_study::summary(f);
}

void report(const Ctx& c, std::ostream& md) {
  if (!std::filesystem::exists(c.results / "g_fine_test_stats.csv") && !std::filesystem::exists(c.results / "g_fine_pilot.csv")) return;
  md << "\n## G1: DCM-fine (docs/DCM.md)\n\n" << fine_study::summary_markdown(c.results);
}

}  // namespace nfx::study_g
