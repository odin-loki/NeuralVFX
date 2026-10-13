// Training neural effects on the CPU from clips (docs/PLAN.md §5.2): hand-written gradients for the two model
// families of model.hpp, Adam, data-parallel minibatches over threads. No third-party libraries.
//
// Each training example is a clip with its learned controls. Every clip also gets its own variation code z
// (an auto-decoder: the codes are learned together with the weights, with a small |z|^2 prior), so the model can
// reproduce each training variation and new variations come from new codes at run time.
//
// The hot loops are compiled for x86-64-v3 (AVX2 + FMA); train() refuses to run on a CPU without it.
#pragma once

#include <neuralfx/clip.hpp>
#include <neuralfx/model.hpp>

#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <vector>

namespace nfx::train {

struct Example {
  const Clip* clip = nullptr;
  std::vector<float> controls;  // n_controls values (empty when the model has none)
};

struct Options {
  int iterations = 3000;
  int batch_frames = 8;    // frames per step, spread over the threads
  int pixels = 4096;       // sampled pixels per frame (grid family; the conv family uses whole frames)
  int threads = 0;         // 0 = all hardware threads
  float lr = 3e-3f;        // MLP / conv weights, basis, FiLM
  float lr_features = 2e-2f;
  float lr_codes = 1e-2f;
  float z_prior = 1e-3f;   // weight of mean |z|^2
  float final_lr_scale = 0.05f;  // cosine decay to this fraction
  std::vector<int> frames; // frame indices usable for training (same for every clip); empty = all
  std::uint64_t seed = 1;
  int log_every = 250;
  std::optional<Model> init;   // start from this model instead of a fresh one
  bool freeze_model = false;   // learn only the codes (fit variations of a trained model)
  // Quantisation-aware training (results/compression, study F2): with qat_bits in 2..8 the forward pass sees the
  // features as they will be stored at that precision (per-plane min/max, uniform codes; model.cpp), and the
  // gradients pass straight through the rounding to the float features (straight-through estimator). Quantisation
  // starts after the first qat_start fraction of the iterations.
  int qat_bits = 0;
  float qat_start = 0.f;
  bool qat_trim = false;  // quantise with Model::feature_trim ranges (the model must then be saved with it)
  // Rate term: rate_lambda times the estimated bits per feature value is added to the loss, so the features become
  // predictable for the lossless coder (src/core/cm.cpp). The estimate per value is log2(1 + |r| / step), with r the
  // smallest residual of three causal predictors the coder also uses (the median edge predictor in the plane, the
  // previous time slice, and the previous time slice plus the change to the left), and step the plane's quantisation
  // step at rate_bits (its min/max range over 2^rate_bits - 1; held constant within a step).
  float rate_lambda = 0.f;
  int rate_bits = 8;
  // Vector quantisation (Model::vq_bits): after the first vq_start fraction of the iterations, a codebook of
  // 2^vq_bits vectors per group of vq_dim channels is fitted by k-means to the features; from then on the forward pass
  // sees every feature vector as its nearest codeword (gradients pass straight through to the float features) and
  // each codeword moves towards the mean of the features assigned to it (exponential average); codewords unused for
  // 50 steps restart at a random feature vector. The result carries the codebook.
  int vq_bits = 0, vq_dim = 0;
  float vq_start = 0.5f;
  // Mixed precision (study F3): with mixed_bits > 0 (and qat_bits 0), when quantisation starts (qat_start) every feature
  // plane gets its own bits, mixed_min to mixed_max, for an average of mixed_bits per value (allocate_plane_bits), and
  // from then on the forward pass sees each plane at its bits. The result carries them (Model::plane_bits).
  float mixed_bits = 0.f;
  int mixed_min = 0, mixed_max = 8;
  int mixed_size = 0;  // frame size at which the allocation renders (0: the clips' size; the conv family: always native)
  // Sparse features (study F3, grid family): only the grid points each time slice needs are stored (feature_support of
  // the examples), every other point of a plane holds the plane's fill (the mean of the plane's values there). The
  // forward pass sees the features that way from the first step; with qat_bits (or mixed_bits) the stored points are
  // also quantised. The result carries the mask (Model::feature_mask) and the widths (Model::plane_bits).
  bool sparse = false;
  // The multi family (study F4): with qat_bits, every feature plane is trained at qat_bits, or at level_bits[l] for the
  // planes of level l when given; with mlp_qat the forward pass also sees the first, hidden and output layers' weights
  // as stored at 8 bits (Model::mlp_bits; per output unit a scale, straight-through gradients) once quantisation starts.
  std::vector<int> level_bits;
  bool mlp_qat = false;
  int sparse_threshold = 0;  // a pixel counts when a channel is above this (0 to 255)
  int sparse_dilate = 0;     // grow the mask by this many grid points
  std::function<void(int iteration, double loss)> progress;
};

struct Result {
  Model model;                            // codes, z_mean and z_std filled in
  std::vector<std::vector<float>> codes;  // one per example
  double final_loss = 0;                  // mean squared error per channel over the last 5% of steps
  double final_rate = 0;                  // estimated bits per feature value at the end (see Options::rate_lambda)
  double seconds = 0;
  double alloc_seconds = 0;               // of which the bit allocation (Options::mixed_bits)
  std::vector<std::pair<int, double>> curve;
};

// Throws std::invalid_argument for inconsistent data, std::runtime_error without AVX2 + FMA.
Result train(const Hyper& h, std::span<const Example> data, const Options& options);

// Render a whole clip with the trainer's (fast, float) forward pass, quantised to RGBA8, on `threads` threads (0: every
// hardware thread).
Clip render_clip(const Model& m, std::span<const float> controls, std::span<const float> z, int frames, int size, int threads = 0);

bool cpu_supported();

// The rate estimate of Options::rate_lambda for a model's features: total estimated bits, and (when `grad` is not
// empty, same size as the features) adds the gradient of those bits times `weight` to it. With `plane_bits` (one per
// plane), each plane's step is that of its own bits, and planes at 0 bits cost nothing.
double feature_rate(const Model& m, int bits, std::span<float> grad = {}, float weight = 0.f, bool trim = false,
                    std::span<const std::uint8_t> plane_bits = {});

// Features as stored at `bits` (2 to 8) and read back, in place (the forward pass of quantisation-aware training),
// with min/max ranges or (trim) the ranges of Model::feature_trim. The second form: each plane at its own bits (0 to 8).
void fake_quantise(Model& m, int bits, bool trim = false);
void fake_quantise(Model& m, std::span<const std::uint8_t> plane_bits, bool trim = false);

// The grid points each time slice needs (grid family), [grid_t][grid][grid], 1 = needed: point (x, y) of slice t is
// needed when a pixel that samples it with a weight above zero (bilinear, as the trainer and the runtime sample at the
// clips' size) has a channel above `threshold` in a frame of an example that blends slice t with a weight above zero;
// then grown by `dilate` points (a 3 x 3 neighbourhood per step).
std::vector<std::uint8_t> feature_support(const Hyper& h, std::span<const Example> data, int threshold, int dilate = 0);

// Features as stored per plane, in place: each plane at plane_bits[k] (empty: unquantised), and with m.feature_mask
// the points outside the mask at their plane's fill.
void store_planes(Model& m, std::span<const std::uint8_t> plane_bits, bool trim = false);

// Mixed precision: the bits of every feature plane [basis][slice][channel] for an average of `avg_bits` per value,
// each from min_bits to max_bits. Every plane's distortion at every width is measured alone, as the squared change of
// the rendered frames (every frame of every example whose time slices use the plane) when only that plane is quantised
// (feature_plane_range ranges); then bits go to the plane whose distortion falls most per bit (along each plane's lower
// convex hull, so a step of several bits counts at its mean slope) until the budget is spent or no plane gains.
// `codes`: the examples' variation codes (empty: zero codes). `size`: render size (0: the clips'). `distortion`, when
// given, receives the table [plane][bits - min_bits].
std::vector<std::uint8_t> allocate_plane_bits(const Model& m, std::span<const Example> data, std::span<const std::vector<float>> codes,
                                              double avg_bits, int min_bits, int max_bits, bool trim, int threads, int size = 0,
                                              std::vector<double>* distortion = nullptr);

}  // namespace nfx::train
