// nvfx_ingest: an effect video (or a folder of PAM/PPM frames) into a training clip, with a licence check and a
// row in the licence register (docs/DATA.md).
//
//   nvfx_ingest --input fire.mov | --frames-dir DIR   --name fire_owner_01
//               --licence own|CC0-1.0|CC-BY-4.0|CC-BY-SA-4.0|commercial  --source URL-or-description
//               [--author NAME] [--licence-note TEXT] [--size 128] [--fps 30] [--start 0] [--duration 0]
//               [--crop w:h:x:y] [--alpha additive|luma|keep] [--loop-blend 16] [--max-frames 256]
//               [--out clip.nfxclip] [--register licences.tsv] [--sheet sheet.png]
//
// Output defaults to <data root>/clips/ingest/<name>.nfxclip and the register to <data root>/licences.tsv.
#include "args.hpp"

#include <neuralfx/image_io.hpp>
#include <neuralfx/ingest.hpp>

#include <print>

using namespace nfx;

int main(int argc, char** argv) try {
  const tools::Args a(argc, argv, {"help"});
  if (a.flag("help")) {
    std::println("nvfx_ingest --input VIDEO | --frames-dir DIR --name NAME --licence ID --source WHERE [options; see the source header]");
    return 0;
  }
  ingest::Licence lic{a.str("licence"), a.str("source"), a.str("author"), a.str("licence-note")};
  if (auto ok = ingest::check_licence(lic); !ok) {
    std::println(stderr, "nvfx_ingest: refused: {}", ok.error());
    return 3;
  }
  ingest::Options o;
  o.name = a.need("name");
  o.size = a.i("size", o.size);
  o.fps = a.f("fps", o.fps);
  o.start = a.f("start", o.start);
  o.duration = a.f("duration", o.duration);
  o.max_frames = a.i("max-frames", o.max_frames);
  o.crop = a.str("crop");
  o.loop_blend = a.i("loop-blend", 0);
  if (!ingest::parse_alpha(a.str("alpha", "luma"), o.alpha)) throw std::invalid_argument("--alpha: additive, luma or keep");
  std::filesystem::path input;
  std::expected<Clip, std::string> clip = std::unexpected("no input");
  if (a.has("input")) {
    input = a.str("input");
    clip = ingest::from_video(input, o);
  } else {
    input = a.need("frames-dir");
    clip = ingest::from_frames(input, o);
  }
  if (!clip) throw std::runtime_error(clip.error());
  const auto out = a.has("out") ? std::filesystem::path(a.str("out")) : data_root() / "clips" / "ingest" / (o.name + ".nfxclip");
  if (auto r = write_clip(out, *clip); !r) throw std::runtime_error(r.error());
  const auto reg = a.has("register") ? std::filesystem::path(a.str("register")) : data_root() / "licences.tsv";
  if (auto r = ingest::register_clip(reg, lic, input, out, *clip); !r) throw std::runtime_error(r.error());
  if (a.has("sheet")) {
    if (auto r = write_png(a.str("sheet"), contact_sheet(*clip, 8, std::max(1, clip->frames / 16), Background::checker)); !r) {
      std::println(stderr, "warning: {}", r.error());
    }
  }
  std::println("{}: {} frames of {}x{} at {} fps{} -> {} (registered in {})", o.name, clip->frames, clip->size, clip->size, clip->fps,
               clip->loop ? ", looping" : "", out.string(), reg.string());
  a.warn_unused();
  return 0;
} catch (const std::exception& e) {
  std::println(stderr, "nvfx_ingest: {}", e.what());
  return 2;
}
