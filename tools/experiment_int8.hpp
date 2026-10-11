// The int8 path of the frame models (src/runtime/rt_int8.hpp, docs/REPORT.md §7) in nvfx_experiment: its quality against
// the float path on studies A, B and C, and its cost (tools/experiment_int8.cpp).
#pragma once

#include <filesystem>

namespace nfx::study_int8 {

struct Ctx {
  std::filesystem::path data;     // the trained models and clips of studies A to C (read only)
  std::filesystem::path results;  // CSVs (in git)
  int runs = 5;                   // timing: the least of this many runs' medians
  int core = 3;                   // timing: the core the thread is pinned to
};

// Every frame of every model of studies A, B and C that has a target, through the runtime at float and at int8:
// active PSNR, PSNR and SSIM against the clips, the largest and mean pixel difference between the two, and whether the
// int8 frames are the same on every ISA. Paired bootstrap of the change; the rule for the default (results/experiments/
// int8_summary.csv): A's 12 clips, mean change within -0.05 dB and its interval not entirely below that.
void step_quality(const Ctx& c);
// ms per frame, thread CPU time on one pinned core, the least of `runs` runs' medians (200 frames, the first 20 left
// out), float and int8, per ISA and size, for the fire models of studies A to C. Wall-clock medians beside them.
void step_timing(const Ctx& c);

}  // namespace nfx::study_int8
