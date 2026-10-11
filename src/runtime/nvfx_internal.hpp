// Inside the library only: what the scene API (src/compose/nvfx_scene.cpp) needs of the effects the C API loads.
#pragma once

#include <neuralfx/nvfx.h>

#include "rt_common.hpp"

namespace nfx::rt {

// The rollout effect an nvfx_effect holds (nullptr: a frame effect, or null).
const RolloutEffect* rollout_of(const nvfx_effect* e);

}  // namespace nfx::rt
