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
  std::function<void(int iteration, double loss)> progress;
};

struct Result {
  Model model;                            // codes, z_mean and z_std filled in
  std::vector<std::vector<float>> codes;  // one per example
  double final_loss = 0;                  // mean squared error per channel over the last 5% of steps
  double seconds = 0;
  std::vector<std::pair<int, double>> curve;
};

// Throws std::invalid_argument for inconsistent data, std::runtime_error without AVX2 + FMA.
Result train(const Hyper& h, std::span<const Example> data, const Options& options);

// Render a whole clip with the trainer's (fast, float) forward pass, quantised to RGBA8.
Clip render_clip(const Model& m, std::span<const float> controls, std::span<const float> z, int frames, int size);

bool cpu_supported();

}  // namespace nfx::train
