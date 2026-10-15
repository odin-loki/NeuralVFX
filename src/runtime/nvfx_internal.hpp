// Inside the library only: what the scene API (src/compose/nvfx_scene.cpp) needs of the effects the C API loads.
#pragma once

#include <neuralfx/nvfx.h>

#include "rt_common.hpp"

namespace nfx::rt {

// The rollout effect an nvfx_effect holds (nullptr: a frame effect, or null).
const RolloutEffect* rollout_of(const nvfx_effect* e);

// Study K (tools/nvfx_study_k.cpp): which of the stepper's convolutions a rollout instance runs in 8 bits when its
// precision is int8 (the study's candidates; by default kInt8Convs). A set-up call: new runners and a new timeline when
// it changes what the instance runs. NVFX_ERROR_ARGUMENT for a frame model's instance.
nvfx_status set_rollout_int8_convs(nvfx_instance* in, Int8Convs convs);

// The precision an nvfx_precision asks of a rollout effect (DEFAULT: default_rollout_precision()), and the ISA a rollout runner
// at a precision takes: the one forced (nvfx_set_isa); with none forced, the AVX-512 build where the CPU has AVX-512
// (at int8 only where it also has VNNI), else AVX2, else the baseline (docs/REPORT.md §13.10). Scenes and the effect
// manager create their runners with these, as instances do.
Precision rollout_precision(nvfx_precision p);
nvfx_isa rollout_isa(Precision p);

// The ISA a rollout instance's runners were built for (at its present precision); NVFX_ISA_AUTO for a frame model's
// instance or null.
nvfx_isa rollout_instance_isa(const nvfx_instance* in);

}  // namespace nfx::rt
