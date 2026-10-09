// nvfx_eval: score flipbook baselines and trained models against a reference clip.
//
//   nvfx_eval baselines --clip ref.nfxclip [--keep even] [--csv out.csv]
//   nvfx_eval model --clip ref.nfxclip --model m.nvfx [--code K] [--keep even]
//
// --keep even scores the frame-interpolation test: baselines keep only the even frames, and only odd (held-out)
// frames are scored. Prints memory, full-frame and active-region PSNR, SSIM, temporal PSNR and flicker.
#include "args.hpp"

#include <neuralfx/flipbook.hpp>
#include <neuralfx/metrics.hpp>
#include <neuralfx/model.hpp>
#include <neuralfx/train.hpp>

#include <fstream>
#include <print>
#include <ranges>

using namespace nfx;

namespace {

void row(std::string_view name, double kb, const metrics::ClipScores& s, std::ofstream* csv) {
  std::println("| {:<28} | {:>8.1f} | {:>6.2f} | {:>6.2f} | {:.4f} | {:>6.2f} | {:>5.2f} |", name, kb, s.psnr, s.active_psnr,
               s.ssim, s.temporal_psnr, s.flicker);
  if (csv) *csv << std::format("{},{:.1f},{:.3f},{:.3f},{:.5f},{:.3f},{:.3f}\n", name, kb, s.psnr, s.active_psnr, s.ssim, s.temporal_psnr, s.flicker);
}

void header() {
  std::println("| {:<28} | {:>8} | {:>6} | {:>6} | {:>6} | {:>6} | {:>5} |", "method", "KB", "PSNR", "active", "SSIM", "tPSNR", "flick");
  std::println("|---|---:|---:|---:|---:|---:|---:|");
}

}  // namespace

int main(int argc, char** argv) try {
  const tools::Args a(argc, argv, {"help"});
  if (a.flag("help") || a.positional().empty()) {
    std::println("nvfx_eval baselines|model --clip ref.nfxclip [--model m.nvfx] [--code K] [--keep even] [--csv out]");
    return 0;
  }
  auto ref = read_clip(a.need("clip"));
  if (!ref) throw std::runtime_error(ref.error());
  std::vector<int> keep, scored;
  if (a.str("keep", "all") == "even") {
    for (int f = 0; f < ref->frames; ++f) (f % 2 ? scored : keep).push_back(f);
  }
  std::ofstream csv_file;
  if (a.has("csv")) csv_file.open(a.str("csv"));
  std::ofstream* csv = csv_file.is_open() ? &csv_file : nullptr;
  const std::string cmd = a.positional()[0];
  header();
  if (cmd == "baselines") {
    for (const auto& spec : flipbook::ladder(ref->size, ref->frames)) {
      if (!keep.empty() && spec.frames != ref->frames) continue;  // interpolation test: the even frames, all kept
      const auto fb = flipbook::build(*ref, spec, keep);
      row(spec.describe(), static_cast<double>(fb.bytes) / 1024.0, metrics::score(*ref, flipbook::play(fb), scored), csv);
    }
  } else if (cmd == "model") {
    auto m = load_model(a.need("model"));
    if (!m) throw std::runtime_error(m.error());
    const int code = a.i("code", 0);
    const std::span<const float> z = m->z_train.empty() ? std::span<const float>{} : m->z_train[static_cast<std::size_t>(code)];
    const std::span<const float> controls(ref->controls.data(), static_cast<std::size_t>(m->h.n_controls));
    const Clip out = train::render_clip(*m, controls, z, ref->frames, ref->size);
    row(m->h.describe(), static_cast<double>(m->storage_bytes()) / 1024.0, metrics::score(*ref, out, scored), csv);
  } else {
    throw std::invalid_argument("unknown command " + cmd);
  }
  a.warn_unused();
  return 0;
} catch (const std::exception& e) {
  std::println(stderr, "nvfx_eval: {}", e.what());
  return 2;
}
