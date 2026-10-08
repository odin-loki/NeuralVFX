#include <neuralfx/binio.hpp>
#include <neuralfx/clip.hpp>

#include <algorithm>
#include <cstdlib>
#include <format>
#include <fstream>
#include <stdexcept>

namespace nfx {

namespace {

constexpr std::string_view kMagic = "NFXCLIP1";
constexpr std::uint32_t kVersion = 1;

}  // namespace

std::expected<void, std::string> write_clip(const std::filesystem::path& path, const Clip& clip) {
  if (clip.rgba.size() != clip.frame_bytes() * static_cast<std::size_t>(clip.frames)) {
    return std::unexpected("nfxclip: pixel buffer does not match size and frames");
  }
  std::error_code ec;
  if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path(), ec);
  std::ofstream o(path, std::ios::binary);
  if (!o) return std::unexpected(std::format("nfxclip: cannot write {}", path.string()));
  o.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
  bin::put(o, kVersion);
  bin::put(o, static_cast<std::uint32_t>(clip.size));
  bin::put(o, static_cast<std::uint32_t>(clip.frames));
  bin::put(o, clip.fps);
  bin::put(o, static_cast<std::uint32_t>(clip.loop));
  bin::put_str(o, clip.effect, 32);
  bin::put_str(o, clip.source, 64);
  bin::put(o, static_cast<std::uint32_t>(clip.n_controls));
  bin::put_array<float>(o, clip.controls);
  bin::put(o, clip.seed);
  bin::put_array<std::uint8_t>(o, clip.rgba);
  if (!o) return std::unexpected(std::format("nfxclip: write failed for {}", path.string()));
  return {};
}

std::expected<Clip, std::string> read_clip(const std::filesystem::path& path) {
  std::ifstream i(path, std::ios::binary);
  if (!i) return std::unexpected(std::format("nfxclip: cannot open {}", path.string()));
  std::string magic(kMagic.size(), '\0');
  i.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  if (!i || magic != kMagic) return std::unexpected(std::format("nfxclip: not a clip: {}", path.string()));
  // Every header field in order; the first failure ends the read.
  auto version = bin::get<std::uint32_t>(i);
  if (!version) return std::unexpected(version.error());
  if (*version != kVersion) return std::unexpected("nfxclip: unsupported version");
  const auto size = bin::get<std::uint32_t>(i);
  const auto frames = bin::get<std::uint32_t>(i);
  const auto fps = bin::get<float>(i);
  const auto loop = bin::get<std::uint32_t>(i);
  const auto effect = bin::get_str(i, 32);
  const auto source = bin::get_str(i, 64);
  const auto n_controls = bin::get<std::uint32_t>(i);
  if (!size || !frames || !fps || !loop || !effect || !source || !n_controls) return std::unexpected("nfxclip: truncated header");
  if (*size == 0 || *size > 4096 || *frames == 0 || *frames > 100000) return std::unexpected("nfxclip: bad dimensions");
  if (*n_controls > kMaxControls) return std::unexpected("nfxclip: bad control count");
  Clip c;
  c.fps = *fps;
  c.loop = *loop != 0;
  c.effect = *effect;
  c.source = *source;
  c.n_controls = static_cast<int>(*n_controls);
  if (auto r = bin::get_array<float>(i, c.controls); !r) return std::unexpected(r.error());
  const auto seed = bin::get<std::uint64_t>(i);
  if (!seed) return std::unexpected(seed.error());
  c.seed = *seed;
  c.allocate(static_cast<int>(*size), static_cast<int>(*frames));
  if (auto r = bin::get_array<std::uint8_t>(i, c.rgba); !r) {
    return std::unexpected(std::format("nfxclip: truncated pixels in {}", path.string()));
  }
  return c;
}

Clip slice_clip(const Clip& clip, int first, int count) {
  if (first < 0 || count <= 0 || first + count > clip.frames) throw std::invalid_argument("slice_clip: bad range");
  Clip out = clip;
  out.frames = count;
  out.loop = clip.loop && first == 0 && count == clip.frames;
  const auto begin = clip.rgba.begin() + static_cast<std::ptrdiff_t>(clip.frame_bytes() * static_cast<std::size_t>(first));
  out.rgba.assign(begin, begin + static_cast<std::ptrdiff_t>(clip.frame_bytes() * static_cast<std::size_t>(count)));
  return out;
}

std::filesystem::path data_root() {
  if (const char* e = std::getenv("NEURALVFX_DATA"); e && *e) return e;
  if (const char* h = std::getenv("HOME"); h && *h) return std::filesystem::path(h) / "nvfx-data";
  return "nvfx-data";
}

}  // namespace nfx
