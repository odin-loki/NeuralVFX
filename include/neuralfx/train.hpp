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
  std::function<void(int iteration, double loss)> progress;
};

struct Result {
  Model model;                            // codes, z_mean and z_std filled in
  std::vector<std::vector<float>> codes;  // one per example
  double final_loss = 0;                  // mean squared error per channel over the last 5% of steps
  double final_rate = 0;                  // estimated bits per feature value at the end (see Options::rate_lambda)
  double seconds = 0;
  std::vector<std::pair<int, double>> curve;
};

// Throws std::invalid_argument for inconsistent data, std::runtime_error without AVX2 + FMA.
Result train(const Hyper& h, std::span<const Example> data, const Options& options);

// Render a whole clip with the trainer's (fast, float) forward pass, quantised to RGBA8.
Clip render_clip(const Model& m, std::span<const float> controls, std::span<const float> z, int frames, int size);

bool cpu_supported();

// The rate estimate of Options::rate_lambda for a model's features: total estimated bits, and (when `grad` is not
// empty, same size as the features) adds the gradient of those bits times `weight` to it.
double feature_rate(const Model& m, int bits, std::span<float> grad = {}, float weight = 0.f, bool trim = false);

// Features as stored at `bits` (2 to 8) and read back, in place (the forward pass of quantisation-aware training),
// with min/max ranges or (trim) the ranges of Model::feature_trim.
void fake_quantise(Model& m, int bits, bool trim = false);

}  // namespace nfx::train
