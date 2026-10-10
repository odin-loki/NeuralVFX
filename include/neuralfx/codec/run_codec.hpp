// G3a (docs/DCM.md §8): a codec for authored runs of a rollout effect, built from its learned dynamics.
//
// An authored run is one real simulation run at a chosen setting and seed. The stream holds:
//   - a header: the controls, the noise seed, the start time, the number of frames and the codec's settings;
//   - the start: the true coarse state at the first frame, quantised and coded as a correction of the nearest stored
//     start point of the effect (or of zero), and optionally the fine heat and soot at a lower resolution, coded as a
//     correction of the upsampled coarse state;
//   - every k frames, the coarse correction: the true coarse state minus the effect's own stepper's prediction (driven
//     by the simulator's forcing noise with the run's seed), quantised with a step in units of each channel's scale;
//   - optionally, every kf frames, a fine residual: the true fine fields minus the detail layer's, block-averaged to a
//     lower resolution and quantised.
// Everything else (the fine detail between corrections, the picture) is synthesised by the effect's detail layer and
// renderer, as when it plays endlessly. Integers are coded by the context-mixing coder of rcoder.hpp.
//
// Closed loop: the encoder runs the decoder's reconstruction and computes each correction against it, so the decoder
// reproduces exactly what the encoder saw. The reconstruction is the reference implementation of rollout.hpp (plain
// float C++, built for the baseline instruction set without contraction): bit-exact between the encoder and the decoder
// of one build. A hash of the encoder's final state lets the decoder check this (Decoded::verified). Without the side
// context, the entropy-coded integers never depend on floating-point state, so a stream decodes on any build (one that
// rounds differently reconstructs a slightly different picture, which the next correction pulls back). With it (the
// option), the coder's contexts include a class of the decoder's own prediction, so a build that rounds differently
// decodes garbage, which the checksum of the coded integers refuses.
#pragma once

#include <neuralfx/rollout.hpp>

#include <array>
#include <cstdint>
#include <expected>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace nfx::codec {

// The run as the encoder sees it: true states from the simulation.
struct TrueRun {
  std::array<float, 3> controls{};
  std::uint64_t seed = 0;
  float time0 = 0.f;   // seconds since the effect began, at the start state
  int frames = 0;      // frames coded after the start
  int size = 128;      // side of the fine fields and of the rendered frames
  std::vector<float> coarse;          // (frames + 1) * res * res * kPhys: index 0 the start, i after i frames
  std::vector<float> fine_t, fine_d;  // (frames + 1) * size * size, rows from the bottom (the simulation's layout)
};

struct Settings {
  int k = 4;                   // a coarse correction every k frames (0: none after the start)
  float q = 0.1f;              // coarse step, in channel scales (rollout::Model::scale)
  float q_vel = 1.f;           // velocity step as a multiple of q (0: velocity never corrected)
  float q_mat = 1.f;           // heat and soot step as a multiple of q (0: only velocity is corrected)
  float q_start = 0.f;         // coarse start step in channel scales (0: q)
  bool start_nearest = true;   // the coarse start is coded as a correction of the nearest stored start (else of zero)
  int start_fine = 64;         // side of the coded fine start (0: the fine fields start as the upsampled coarse state)
  float q_fine_start = 0.05f;  // fine start step, in the renderer's input scale (rollout::Model::render_scale)
  int kf = 0;                  // a fine residual every kf frames (0: none)
  int fine_res = 64;           // side of the fine residual (divides the frame size)
  float qf = 0.05f;            // fine residual step, in the renderer's input scale
  bool correct_flow = true;    // a velocity correction also moves the flow that carries the fine fields this frame
  bool fine_sets_coarse = true;  // after a fine residual, the coarse heat and soot become the fine fields' block means
  bool side_context = false;   // the coder also conditions on a class of the decoder's own prediction at each cell (the
                               // stream then decodes only where the reconstruction rounds alike: see above)
  float round = 0.5f;          // quantiser rounding offset: r = sign(x) floor(|x| / step + round); 0.5 rounds to nearest,
                               // less leaves more zeros (a dead zone)
};

// Steps are stored as 8-bit codes on a log scale (2^(1/16) apart); the codec uses the stored value. This returns the
// settings as they will be used.
Settings stored_settings(const Settings& s);
// A short name of the settings for tables ("k4_q0.100_..." with the stored steps).
std::string describe(const Settings& s);

struct Encoded {
  std::vector<std::uint8_t> stream;
  std::size_t header_bytes = 0;  // header and trailer
  // ideal code lengths of each part (bytes; the arithmetic coder adds at most a few bytes in all)
  double coarse_start_bytes = 0, fine_start_bytes = 0, coarse_bytes = 0, fine_bytes = 0;
  int coarse_planes = 0, fine_planes = 0;
  std::vector<std::uint8_t> frames;  // the reconstruction (when asked): frames * size * size * 4, RGBA8 premultiplied,
                                     // rows top to bottom
};

// Encodes `run` with effect `m` (its stepper, detail layer, renderer and start points). render: also return the
// reconstructed frames.
Encoded encode(const rollout::Model& m, const TrueRun& run, const Settings& s, bool render = false);

struct Decoded {
  Settings settings;
  std::array<float, 3> controls{};
  std::uint64_t seed = 0;
  float time0 = 0.f;
  int frames = 0, size = 0;
  std::vector<std::uint8_t> rgba;  // frames * size * size * 4 (empty when a frame callback was given)
  bool verified = false;           // the reconstruction equals the encoder's (hash of the final state)
  std::size_t working_bytes = 0;   // decoder memory while it ran: one plane of integers, coder model, reconstruction state
  double entropy_seconds = 0;      // thread CPU time spent decoding integers (the rest is the effect's own step and render)
};

// Decodes a stream. Refuses a stream made for another model, or one that is damaged or truncated (the decoder runs
// past the end of its input, or the checksum of every coded integer differs): no frames are returned. on_frame, if
// given, receives each frame as it is made instead (streaming): damage is then reported after the frames.
std::expected<Decoded, std::string> decode(const rollout::Model& m, std::span<const std::uint8_t> stream,
                                           const std::function<void(int, std::span<const std::uint8_t>)>& on_frame = {});

// The effect's renderer as the codec runs it: the 8-bit output of rollout::render (rounded as v * 255 + 0.5), byte for
// byte, without its per-pixel cost (the directional soot sums are made once per coarse cell, no allocation per pixel,
// and pixels the material gate closes are not run through the MLP). out: size * size * 4, rows top to bottom. scratch is
// reused between calls.
void render_u8(const rollout::Model& m, const rollout::State& s, std::span<std::uint8_t> out, std::vector<float>& scratch);

// A 32-bit tag of the model's stored form (weights, scales, detail constants, start points): a stream names the model
// it was made with.
std::uint32_t model_tag(const rollout::Model& m);

// The stored start point nearest to the controls (squared distance in control space).
int nearest_start(const rollout::Model& m, std::span<const float> controls);

}  // namespace nfx::codec
