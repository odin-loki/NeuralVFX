// nvfx_estimate: start points for a rollout effect estimated from footage (docs/FOOTAGE.md). Writes a copy of the effect
// whose start points are states estimated from the footage, so the effect continues from what the footage shows.
//
//   nvfx_estimate --effect fire.nvfx --clip take3.nfxclip --out fire_take3.nvfx
//                 [--at 63[,40,...]] [--context 8] [--controls 0.5,0.5,0.5] [--age 0.033] [--alpha auto|yes|no]
//                 [--inverse FILE | --train-inverse FILE] [--fine 64] [--keep-starts] [--no-refine] [--preview sheet.png]
//                 [--threads 1]
//   Footage straight from a video or a folder of frames goes through the ingest path, with its licence check and a row in
//   the licence register (docs/DATA.md §3):
//                 --input take3.mov | --frames-dir DIR   --licence own|CC0-1.0|CC-BY-4.0|commercial --source WHERE
//                 [--author NAME] [--licence-note TEXT] [--crop w:h:x:y] [--start 0] [--duration 0] [--register FILE]
//
// --at: frames of the footage (0-based) to start from; default the last. Each uses up to --context frames up to it.
// --controls: the effect's controls that match the footage (the start point's controls; the runtime picks start points
//   by the nearest controls). --age (one-shot effects): seconds since the effect began at the footage's first frame
//   (default: one frame, footage that starts at the detonation).
// --alpha: whether the footage has a real alpha channel; auto: no when alpha is max(r, g, b) everywhere (nvfx_ingest
//   --alpha luma) or zero everywhere.
// The inverse network maps frames to fine heat and soot. It is trained from the built-in simulation for fire, smoke and
// explosion effects, with the motion network that measures velocity (--train-inverse FILE, about 6 minutes on one core;
// cached at <data root>/j/inverse/ by nvfx_study_j inverse and motion). Validated on simulated renders only: real footage is untested (docs/FOOTAGE.md §6).
#include "args.hpp"

#include <neuralfx/footage.hpp>
#include <neuralfx/image_io.hpp>
#include <neuralfx/ingest.hpp>
#include <neuralfx/nvfx.h>

#include <algorithm>
#include <format>
#include <print>
#include <sstream>

namespace fs = std::filesystem;
using namespace nfx;

namespace {

std::vector<int> parse_ints(const std::string& s) {
  std::vector<int> v;
  for (const float f : tools::parse_floats(s)) v.push_back(static_cast<int>(f));
  return v;
}

bool footage_has_alpha(const Clip& c) {
  bool all_zero = true, all_luma = true;
  for (std::size_t i = 0; i < c.rgba.size(); i += 4) {
    const std::uint8_t a = c.rgba[i + 3];
    if (a != 0) all_zero = false;
    if (a != std::max({c.rgba[i], c.rgba[i + 1], c.rgba[i + 2]})) all_luma = false;
  }
  return !(all_zero || all_luma);
}

}  // namespace

int main(int argc, char** argv) try {
  const tools::Args a(argc, argv, {"help", "keep-starts", "no-refine"});
  if (a.flag("help") || !a.has("effect")) {
    std::println("nvfx_estimate --effect EFFECT.nvfx (--clip CLIP.nfxclip | --input VIDEO | --frames-dir DIR) --out OUT.nvfx [options; see the source header]");
    return a.flag("help") ? 0 : 1;
  }
  auto loaded = rollout::load_model(fs::path(a.str("effect")));
  if (!loaded) throw std::runtime_error(a.str("effect") + ": " + loaded.error() + " (nvfx_estimate needs a rollout effect)");
  const rollout::Model M = std::move(*loaded);
  const fs::path out = a.need("out");

  // --- the footage
  Clip clip;
  if (a.has("clip")) {
    auto c = read_clip(a.str("clip"));
    if (!c) throw std::runtime_error(c.error());
    clip = std::move(*c);
  } else {
    ingest::Licence lic{a.str("licence"), a.str("source"), a.str("author"), a.str("licence-note")};
    if (auto ok = ingest::check_licence(lic); !ok) {
      std::println(stderr, "nvfx_estimate: refused: {}", ok.error());
      return 3;
    }
    ingest::Options o;
    o.name = fs::path(a.has("input") ? a.str("input") : a.need("frames-dir")).stem().string();
    o.size = 128;
    o.fps = M.fps;
    o.start = a.f("start", 0.f);
    o.duration = a.f("duration", 0.f);
    o.crop = a.str("crop");
    o.alpha = a.str("alpha", "auto") == "yes" ? ingest::AlphaMode::keep : ingest::AlphaMode::luma;
    const fs::path input = a.has("input") ? fs::path(a.str("input")) : fs::path(a.str("frames-dir"));
    auto c = a.has("input") ? ingest::from_video(input, o) : ingest::from_frames(input, o);
    if (!c) throw std::runtime_error(c.error());
    clip = std::move(*c);
    const fs::path reg = a.has("register") ? fs::path(a.str("register")) : data_root() / "licences.tsv";
    if (auto r = ingest::register_clip(reg, lic, input, out, clip); !r) throw std::runtime_error(r.error());
  }
  if (clip.frames < 1) throw std::runtime_error("the footage has no frames");

  // --- the inverse network
  const std::string alpha_opt = a.str("alpha", "auto");
  const bool alpha = alpha_opt == "yes" || (alpha_opt == "auto" && footage_has_alpha(clip));
  const std::string mode = alpha ? "rgba" : "rgb";
  footage::Inverse inv;
  if (a.has("train-inverse")) {
    sim::Effect e{};
    if (!sim::parse_effect(M.effect, e)) throw std::runtime_error(std::format("--train-inverse: the simulation makes fire, smoke and explosion, not '{}'", M.effect));
    std::println("training the inverse network for {} ({}) from the simulation ...", M.effect, mode);
    footage::SampleOptions so;  // as nvfx_study_j inverse and motion (docs/FOOTAGE.md §2)
    so.runs = 64;
    so.drop_alpha = !alpha;
    so.threads = a.i("threads", 1);
    footage::InverseSpec spec;
    spec.alpha = alpha;
    inv = footage::init_inverse(spec, 11);
    inv.effect = M.effect;
    footage::InverseTrainOptions to;
    to.iterations = 12000;
    to.threads = so.threads;
    to.progress = [](int it, double loss) { std::println("  {:5d} loss {:.5f}", it, loss); };
    footage::train_inverse(inv, footage::simulate_samples(e, so), to);
    std::println("training the motion network ...");
    inv.motion.in_scale = inv.scale;
    footage::train_motion(inv.motion, footage::simulate_motion_samples(e, inv, so), to);
    if (auto w = footage::save_inverse(a.str("train-inverse"), inv); !w) throw std::runtime_error(w.error());
  } else {
    const fs::path p = a.has("inverse") ? fs::path(a.str("inverse")) : data_root() / "j" / "inverse" / std::format("{}_{}.nvfxinv", M.effect, mode);
    auto r = footage::load_inverse(p);
    if (!r) throw std::runtime_error(std::format("{} (give --inverse FILE, or train one with --train-inverse FILE)", r.error()));
    inv = std::move(*r);
  }
  if (inv.spec.alpha != alpha) throw std::runtime_error(std::format("the inverse network is for footage {} alpha; this footage is {} alpha (see --alpha)",
                                                                    inv.spec.alpha ? "with" : "without", alpha ? "with" : "without"));
  if (inv.effect != M.effect) std::println(stderr, "warning: the inverse network was trained for '{}', the effect is '{}'", inv.effect, M.effect);
  if (clip.size != inv.size) throw std::runtime_error(std::format("the footage is {} px; the inverse network works at {} px (ingest with --size {})", clip.size, inv.size, inv.size));

  // --- the estimates
  std::vector<int> at = a.has("at") ? parse_ints(a.str("at")) : std::vector<int>{clip.frames - 1};
  std::vector<float> controls = a.has("controls") ? tools::parse_floats(a.str("controls")) : std::vector<float>{};
  controls.resize(static_cast<std::size_t>(M.h.n_controls), 0.5f);
  const float age0 = a.f("age", 1.f / M.fps);
  footage::EstimateOptions eo;
  eo.context = a.i("context", eo.context);
  eo.refine = !a.flag("no-refine");
  eo.assim.controls = controls;
  std::vector<footage::Frame> frames;
  for (int i = 0; i < clip.frames; ++i) frames.push_back(footage::to_float(clip.frame(i)));
  std::vector<rollout::StartPoint> starts;
  for (const int f : at) {
    if (f < 0 || f >= clip.frames) throw std::runtime_error(std::format("--at {}: the footage has frames 0 to {}", f, clip.frames - 1));
    eo.assim.time = M.loop ? static_cast<float>(f) / M.fps : age0 + static_cast<float>(f) / M.fps;
    eo.assim.seed = 1;
    const std::size_t first = static_cast<std::size_t>(std::max(0, f - eo.context + 1));
    const footage::Estimate est = footage::estimate_start(M, inv, std::span(frames).subspan(first, static_cast<std::size_t>(f) + 1 - first), clip.size, eo);
    rollout::StartPoint sp = est.start;
    sp.seed = 1000 + static_cast<std::uint64_t>(starts.size());
    double heat = 0, soot = 0;
    for (std::size_t j = 0; j < sp.coarse.size(); j += rollout::kPhys) {
      heat += sp.coarse[j + 2];
      soot += sp.coarse[j + 3];
    }
    std::println("frame {}: {} context frames, time {:.3f} s, mean coarse heat {:.4f}, soot {:.4f}", f, est.fields.size(), sp.time,
                 heat / static_cast<double>(sp.coarse.size() / rollout::kPhys), soot / static_cast<double>(sp.coarse.size() / rollout::kPhys));
    starts.push_back(std::move(sp));
  }
  const int fine = a.i("fine", M.h.start_fine > 0 ? M.h.start_fine : 64);
  const rollout::Model R = footage::with_starts(M, starts, fine, a.flag("keep-starts"));
  if (auto w = rollout::save_model(out, R); !w) throw std::runtime_error(w.error());
  std::println("{}: {} with {} start points ({} estimated from footage), {:.1f} KB", out.string(), R.effect, R.starts.size(), starts.size(),
               static_cast<double>(R.storage_bytes()) / 1024.0);

  if (a.has("preview")) {  // per start point: the footage frame, then the effect from it at 0, 8, 30 and 60 frames
    nvfx_effect* fx = nullptr;
    std::ostringstream os;
    if (auto r = rollout::save_model(os, R); !r) throw std::runtime_error(r.error());
    const std::string b = os.str();
    if (nvfx_effect_load_memory(b.data(), b.size(), &fx) != NVFX_OK) throw std::runtime_error("runtime load failed");
    std::vector<Clip> rows;
    const int first_new = static_cast<int>(R.starts.size() - starts.size());
    const std::vector<int> cols = {0, 8, 30, 60};
    for (std::size_t k = 0; k < starts.size(); ++k) {
      Clip row;
      row.allocate(clip.size, 5);
      std::ranges::copy(clip.frame(at[k]), row.frame(0).begin());
      nvfx_instance* in = nullptr;
      nvfx_instance_create(fx, clip.size, &in);
      nvfx_instance_set_controls(in, controls.data(), static_cast<int>(controls.size()));
      nvfx_instance_set_drift(in, 0.f);
      nvfx_instance_set_variation(in, first_new + static_cast<int>(k));
      for (std::size_t c = 0; c < cols.size(); ++c) nvfx_render(in, cols[c] / static_cast<double>(M.fps), row.frame(static_cast<int>(c) + 1).data(), static_cast<std::size_t>(clip.size) * 4);
      nvfx_instance_free(in);
      rows.push_back(std::move(row));
    }
    nvfx_effect_free(fx);
    std::vector<const Clip*> ptrs;
    for (const Clip& c : rows) ptrs.push_back(&c);
    if (auto w = write_png(a.str("preview"), comparison_sheet(ptrs, 5, Background::black, 2)); !w) std::println(stderr, "preview: {}", w.error());
  }
  std::println("Validated on simulated renders only (docs/FOOTAGE.md): check the result by eye before shipping it.");
  a.warn_unused();
  return 0;
} catch (const std::exception& e) {
  std::println(stderr, "nvfx_estimate: {}", e.what());
  return 2;
}
