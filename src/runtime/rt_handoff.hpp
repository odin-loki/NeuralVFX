// The simulator's own look for a rollout effect's first frames (study I round 2, docs/COMPOSE.md §10).
//
// The learned renderer draws a fresh explosion poorly: its first frames are rare in its training samples (REPORT §6.4),
// while the simulator's renderer, drawn on the model's own fine fields, is about 1 dB closer in the first second (study
// G extras, DCM §10.7). A hand-off draws the first `frames` frames of an effect's timeline with the simulator's look
// and crossfades to the learned renderer over `fade` frames. Compiled at the baseline ISA, like the simulation, so the
// look is the simulator's to the bit. Nothing here allocates.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace nfx::rt {

// Which of the simulator's looks (sim::Fluid::render) an effect has; none for effects the simulator does not make.
enum class SimLook : int { none = -1, fire = 0, smoke = 1, explosion = 2 };
SimLook sim_look_for(std::string_view effect_name);

// sim::Fluid::render of fine heat and soot fields of size x size (rows from the bottom, as the runtime keeps them),
// as a solver of that size holding these fields (with its zero border) draws them: the same bytes. Premultiplied RGBA8,
// rows top to bottom, `stride` bytes apart. With a colour matrix (row-major 3 x 3, the instance's hue and brightness)
// the colour is multiplied by it before rounding, as the learned renderer does.
void draw_sim_look(SimLook look, std::span<const float> heat, std::span<const float> soot, int size, std::uint8_t* rgba,
                   std::size_t stride, const float* colour = nullptr);

// The learned renderer's weight at frame f of the timeline (f = 0: the start point) for a hand-off after `frames`
// frames over `fade` frames: 0 for f < frames, then (f - frames + 1) / (fade + 1), and 1 from f = frames + fade on. So
// frames + fade frames show some of the simulator's look.
float handoff_weight(std::int64_t f, int frames, int fade);

// The hand-off of one frame: learned (in rgba) and the simulator's look (in sim), byte by byte as the shard crossfade
// mixes, rgba = sim * (1 - w) + learned * w, rounded half up.
void blend_handoff(std::uint8_t* rgba, std::size_t stride, const std::uint8_t* sim, std::size_t sim_stride, int size, float w);

}  // namespace nfx::rt
