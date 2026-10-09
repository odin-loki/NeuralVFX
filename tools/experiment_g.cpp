// Study G (docs/DCM.md): diffusion-context mixing for generating effects. Part of nvfx_experiment. A skeleton: the
// steps are registered as g-data, g-pilot, g-search, g-eval and g-timing, and later stages fill them in.
#include "experiment_g.hpp"

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

void report(const Ctx&, std::ostream&) {}

}  // namespace nfx::study_g
