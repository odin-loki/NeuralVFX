// Study R (docs/STUDY_R.md): the 3D solver's smoke plume against established plume physics. `nvfx_3d plume` measures
// the steady plume (time averages of one run per seed: radial profiles about the source's axis at every height, the
// buoyancy flux through every horizontal plane, a slice through the axis and the field summed over depth), `nvfx_3d
// plumefit` fits Morton, Taylor and Turner's (1956) point-source plume to the pooled profiles with a virtual origin
// (the width's growth, the centreline's velocity and buoyancy, their constants, the buoyancy flux's scaling) and the
// wind-bent plume's rise against Briggs's two-thirds law, with intervals by resampling the seeds, and `nvfx_3d
// plumevideo` draws the figure.
#pragma once

#include "args.hpp"

#include <filesystem>

namespace nfx::tools::plume3d {

int measure(const Args& a, const std::filesystem::path& root);
int fit(const Args& a, const std::filesystem::path& root);
int video(const Args& a, const std::filesystem::path& root);

}  // namespace nfx::tools::plume3d
