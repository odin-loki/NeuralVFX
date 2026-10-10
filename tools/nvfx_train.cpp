// nvfx_train: train a neural effect from clips and save it as .nvfx.
//
//   nvfx_train --rollout fire|smoke|explosion --out effect.nvfx [--runs 160] [--frames 240] [--iters 2500]
//              [--finetune 1500] [--render-iters 2500] [--starts 8] [--threads 0]
//     trains a rollout effect (include/neuralfx/rollout.hpp) from runs of the built-in simulation instead: start
//     points, a learned coarse stepper, the detail layer and a renderer (about 40 minutes per effect on 4 cores).
//
//   nvfx_train --clips a.nfxclip[,b.nfxclip...] | --clip-dir DIR   --out model.nvfx
//              [--arch grid|conv] [--grid 32] [--channels 8] [--hidden 32] [--layers 2] [--grid-t 16] [--bases 1]
//              [--latent 16] [--c0 32] [--c1 16] [--c2 8] [--latent-dims 0] [--no-controls]
//              [--iters 3000] [--batch 8] [--pixels 4096] [--threads 0] [--lr 3e-3] [--lr-features 2e-2]
//              [--lr-codes 1e-2] [--z-prior 1e-3] [--bits 16] [--frames all|even] [--seed 1]
//              [--qat] [--qat-start 0] [--trim] [--rate LAMBDA]
//
// --bits: feature storage, 16 (fp16) or 2 to 8 (per-plane affine codes; below 8 bit-packed). --qat trains for that
// precision (the forward pass sees the stored features, after the first --qat-start fraction of the iterations);
// --trim stores each plane with its best-quantising range (tails clipped) instead of its min and max; --rate adds
// LAMBDA times the estimated bits per feature value to the loss (results/compression, study F2).
//
// Every clip must have the same size and length. The clips' controls become the model's controls unless
// --no-controls; --latent-dims gives each clip its own learned variation code. Prints the training curve and,
// at the end, the PSNR of the reloaded model on each training clip.
#include "args.hpp"

#include <neuralfx/metrics.hpp>
#include <neuralfx/model.hpp>
#include <neuralfx/rollout_train.hpp>
#include <neuralfx/sim.hpp>
#include <neuralfx/train.hpp>

#include <algorithm>
#include <filesystem>
#include <print>
#include <ranges>

using namespace nfx;

int main(int argc, char** argv) try {
  const tools::Args a(argc, argv, {"no-controls", "help", "quiet", "qat", "trim"});
  if (a.flag("help")) {
    std::println("nvfx_train --clips a,b | --clip-dir DIR --out model.nvfx [--arch grid|conv ...]");
    std::println("nvfx_train --rollout fire|smoke|explosion --out effect.nvfx [--runs N --iters N ...] (see the source header)");
    return 0;
  }
  if (a.has("rollout")) {
    sim::Effect e;
    if (!sim::parse_effect(a.str("rollout"), e)) throw std::invalid_argument("--rollout: fire, smoke or explosion");
    if (!train::cpu_supported()) throw std::runtime_error("training needs AVX2 + FMA");
    rollout::SimRecipe r = rollout::recipe_for(e);
    r.runs = a.i("runs", r.runs);
    r.frames = a.i("frames", r.frames);
    r.starts = a.i("starts", r.starts);
    r.threads = a.i("threads", 0);
    r.stepper.iterations = a.i("iters", r.stepper.iterations);
    r.stepper.finetune = a.i("finetune", r.stepper.finetune);
    r.stepper.threads = r.threads;
    r.renderer.iterations = a.i("render-iters", r.renderer.iterations);
    r.renderer.threads = r.threads;
    r.start_frame = std::min(r.start_frame, r.frames - 1);
    const bool quiet = a.flag("quiet");
    std::println("recording {} runs of {} frames ({:.1f} minutes of simulation)", r.runs, r.frames, r.runs * r.frames / 30.0 / 60.0);
    const auto runs = rollout::record_runs(r);
    rollout::Model m = rollout::recipe_model(r);
    r.stepper.progress = [quiet](int it, int unroll, double loss) {
      if (!quiet) std::println("  stepper {:5}  unroll {:2}  loss {:.5f}", it, unroll, loss);
    };
    r.renderer.progress = [quiet](int it, double psnr) {
      if (!quiet) std::println("  renderer {:5}  psnr {:.2f}", it, psnr);
    };
    const auto sr = rollout::train_stepper(m, runs, r.stepper);
    const auto fin = rollout::finish_model(m, r, runs);
    const auto out = a.need("out");
    if (auto s = rollout::save_model(out, m); !s) throw std::runtime_error(s.error());
    std::println("stepper loss {:.5f} ({:.0f} s), renderer {:.2f} dB, detail contrast {} swirl {}, {} start points, {:.1f} KB stored",
                 sr.final_loss, sr.seconds, fin.render_psnr, m.detail.contrast, m.detail.swirl, m.starts.size(),
                 static_cast<double>(m.storage_bytes()) / 1024.0);
    a.warn_unused();
    return 0;
  }
  std::vector<std::filesystem::path> paths;
  if (a.has("clips")) {
    for (const auto part : std::views::split(a.str("clips"), ',')) paths.emplace_back(std::string_view(part));
  } else {
    for (const auto& e : std::filesystem::directory_iterator(a.need("clip-dir"))) {
      if (e.path().extension() == ".nfxclip") paths.push_back(e.path());
    }
    std::ranges::sort(paths);
  }
  if (paths.empty()) throw std::invalid_argument("no clips");
  std::vector<Clip> clips;
  for (const auto& p : paths) {
    auto c = read_clip(p);
    if (!c) throw std::runtime_error(c.error());
    clips.push_back(std::move(*c));
  }
  const Clip& first = clips.front();
  Hyper h;
  const std::string arch = a.str("arch", "grid");
  h.arch = arch == "conv" ? Arch::conv : Arch::grid;
  h.size = first.size;
  h.frames = first.frames;
  h.loop = first.loop;
  h.n_controls = a.flag("no-controls") ? 0 : first.n_controls;
  h.n_latent = a.i("latent-dims", 0);
  h.bases = a.i("bases", 1);
  h.grid_t = a.i("grid-t", 16);
  h.grid = a.i("grid", 32);
  h.channels = a.i("channels", 8);
  h.hidden = a.i("hidden", 32);
  h.layers = a.i("layers", 2);
  h.latent = a.i("latent", first.size / 8);
  h.c0 = a.i("c0", 32);
  h.c1 = a.i("c1", 16);
  h.c2 = a.i("c2", 8);

  std::vector<train::Example> data;
  for (const Clip& c : clips) {
    data.push_back({&c, std::vector<float>(c.controls.begin(), c.controls.begin() + h.n_controls)});
  }
  train::Options o;
  o.iterations = a.i("iters", o.iterations);
  o.batch_frames = a.i("batch", o.batch_frames);
  o.pixels = a.i("pixels", o.pixels);
  o.threads = a.i("threads", o.threads);
  o.lr = a.f("lr", o.lr);
  o.lr_features = a.f("lr-features", o.lr_features);
  o.lr_codes = a.f("lr-codes", o.lr_codes);
  o.z_prior = a.f("z-prior", o.z_prior);
  o.seed = a.u64("seed", o.seed);
  o.log_every = a.i("log-every", 250);
  const int bits = a.i("bits", 16);
  if (!valid_feature_bits(bits)) throw std::invalid_argument("--bits must be 16 or 2 to 8");
  if (a.flag("qat")) {
    if (bits >= 16) throw std::invalid_argument("--qat needs --bits 2 to 8");
    o.qat_bits = bits;
    o.qat_start = a.f("qat-start", 0.f);
  }
  o.qat_trim = a.flag("trim");
  o.rate_lambda = a.f("rate", 0.f);
  o.rate_bits = std::min(bits, 8);
  if (a.str("frames", "all") == "even") {
    for (int f = 0; f < h.frames; f += 2) o.frames.push_back(f);
  }
  const bool quiet = a.flag("quiet");
  o.progress = [quiet](int it, double loss) {
    if (!quiet) std::println("  iter {:5}  mse {:.6f}  psnr {:.2f}", it, loss, metrics::psnr_from_mse(loss));
  };
  std::println("training {} on {} clips of {}x{}, {} frames ({})", h.describe(), clips.size(), h.size, h.size, h.frames,
               o.frames.empty() ? "all frames" : "even frames");
  auto r = train::train(h, data, o);
  r.model.effect = first.effect;
  r.model.fps = first.fps;
  r.model.feature_bits = bits;
  r.model.feature_trim = o.qat_trim;
  if (first.source == "sim" && h.n_controls == sim::kControls) {
    r.model.control_names.assign(sim::kControlNames.begin(), sim::kControlNames.end());
  }
  const auto out = a.need("out");
  if (auto s = save_model(out, r.model); !s) throw std::runtime_error(s.error());
  auto m = load_model(out);
  if (!m) throw std::runtime_error(m.error());
  std::println("trained in {:.1f} s; {} parameters, {:.1f} KB stored ({} -bit features), {:.0f} MAC/px at {}",
               r.seconds, m->param_count(), static_cast<double>(m->storage_bytes()) / 1024.0, m->feature_bits,
               m->macs_per_pixel(h.size), h.size);
  for (std::size_t i = 0; i < clips.size() && i < 8; ++i) {
    const std::span<const float> z = m->z_train.empty() ? std::span<const float>{} : m->z_train[i];
    const Clip rc = train::render_clip(*m, data[i].controls, z, clips[i].frames, clips[i].size);
    const auto s = metrics::score(clips[i], rc);
    std::println("  {}: psnr {:.2f} dB, ssim {:.4f}, temporal psnr {:.2f}, flicker {:.2f}", paths[i].filename().string(),
                 s.psnr, s.ssim, s.temporal_psnr, s.flicker);
  }
  a.warn_unused();
  return 0;
} catch (const std::exception& e) {
  std::println(stderr, "nvfx_train: {}", e.what());
  return 2;
}
