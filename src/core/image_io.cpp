#include <neuralfx/image_io.hpp>

#include <zlib.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <format>
#include <fstream>
#include <ranges>

namespace nfx {

bool parse_background(std::string_view text, Background& out) {
  if (text == "black") out = Background::black;
  else if (text == "grey") out = Background::grey;
  else if (text == "checker") out = Background::checker;
  else return false;
  return true;
}

void composite(std::span<const std::uint8_t> premul, int size, Background bg, std::span<std::uint8_t> out) {
  for (int y = 0; y < size; ++y) {
    for (int x = 0; x < size; ++x) {
      const std::size_t i = (static_cast<std::size_t>(y) * size + x) * 4;
      int b = 0;
      if (bg == Background::grey) b = 96;
      if (bg == Background::checker) b = ((x / 8 + y / 8) & 1) ? 112 : 72;
      const int a = premul[i + 3];
      for (int c = 0; c < 3; ++c) out[i + c] = static_cast<std::uint8_t>(std::min(255, premul[i + c] + (b * (255 - a) + 127) / 255));
      out[i + 3] = 255;
    }
  }
}

void blit(Image& dst, std::span<const std::uint8_t> src, int size, int x0, int y0, int scale) {
  for (int y = 0; y < size * scale; ++y) {
    const int dy = y0 + y;
    if (dy < 0 || dy >= dst.height) continue;
    for (int x = 0; x < size * scale; ++x) {
      const int dx = x0 + x;
      if (dx < 0 || dx >= dst.width) continue;
      const auto s = src.subspan((static_cast<std::size_t>(y / scale) * size + x / scale) * 4, 4);
      std::ranges::copy(s, dst.pixel(dx, dy));
    }
  }
}

namespace {

void store_be32(std::span<unsigned char, 4> dst, std::uint32_t x) {
  dst[0] = static_cast<unsigned char>(x >> 24);
  dst[1] = static_cast<unsigned char>(x >> 16);
  dst[2] = static_cast<unsigned char>(x >> 8);
  dst[3] = static_cast<unsigned char>(x);
}

// One PNG chunk: length, type, data, CRC of type and data. Sized up front (no incremental growth).
void write_chunk(std::ofstream& o, std::string_view type, std::span<const unsigned char> data) {
  std::vector<unsigned char> buf(data.size() + 12);
  const std::span all(buf);
  store_be32(all.first<4>(), static_cast<std::uint32_t>(data.size()));
  std::ranges::transform(type, all.begin() + 4, [](char c) { return static_cast<unsigned char>(c); });
  std::ranges::copy(data, all.begin() + 8);
  const uLong crc = crc32(crc32(0L, Z_NULL, 0), buf.data() + 4, static_cast<uInt>(data.size() + 4));
  store_be32(all.last<4>(), static_cast<std::uint32_t>(crc));
  o.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
}

std::expected<std::ofstream, std::string> open_out(const std::filesystem::path& path) {
  std::error_code ec;
  if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path(), ec);
  std::ofstream o(path, std::ios::binary);
  if (!o) return std::unexpected(std::format("cannot write {}", path.string()));
  return o;
}

}  // namespace

std::expected<void, std::string> write_png(const std::filesystem::path& path, const Image& img) {
  auto o = open_out(path);
  if (!o) return std::unexpected(o.error());
  constexpr std::array<unsigned char, 8> sig{137, 80, 78, 71, 13, 10, 26, 10};
  o->write(reinterpret_cast<const char*>(sig.data()), sig.size());
  std::array<unsigned char, 13> ihdr{};
  store_be32(std::span(ihdr).first<4>(), static_cast<std::uint32_t>(img.width));
  store_be32(std::span(ihdr).subspan<4, 4>(), static_cast<std::uint32_t>(img.height));
  ihdr[8] = 8;  // bit depth
  ihdr[9] = 6;  // RGBA; compression, filter and interlace stay 0
  write_chunk(*o, "IHDR", ihdr);
  const std::size_t row = static_cast<std::size_t>(img.width) * 4;
  std::vector<unsigned char> raw((row + 1) * static_cast<std::size_t>(img.height));
  for (int y = 0; y < img.height; ++y) {
    unsigned char* d = raw.data() + (row + 1) * static_cast<std::size_t>(y);
    const unsigned char* r = img.rgba.data() + row * static_cast<std::size_t>(y);
    d[0] = 1;  // Sub filter: smaller files for smooth images
    for (std::size_t i = 0; i < row; ++i) d[i + 1] = static_cast<unsigned char>(r[i] - (i >= 4 ? r[i - 4] : 0));
  }
  uLongf zlen = compressBound(static_cast<uLong>(raw.size()));
  std::vector<unsigned char> z(zlen);
  if (compress2(z.data(), &zlen, raw.data(), static_cast<uLong>(raw.size()), 9) != Z_OK) {
    return std::unexpected("png: deflate failed");
  }
  z.resize(zlen);
  write_chunk(*o, "IDAT", z);
  write_chunk(*o, "IEND", {});
  if (!*o) return std::unexpected(std::format("png: write failed for {}", path.string()));
  return {};
}

std::expected<void, std::string> write_pam(const std::filesystem::path& path, const Image& img) {
  auto o = open_out(path);
  if (!o) return std::unexpected(o.error());
  *o << std::format("P7\nWIDTH {}\nHEIGHT {}\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n", img.width, img.height);
  o->write(reinterpret_cast<const char*>(img.rgba.data()), static_cast<std::streamsize>(img.rgba.size()));
  if (!*o) return std::unexpected(std::format("pam: write failed for {}", path.string()));
  return {};
}

Image contact_sheet(const Clip& clip, int columns, int step, Background bg, int scale) {
  const int shown = (clip.frames + step - 1) / step;
  const int rows = (shown + columns - 1) / columns;
  const int cell = clip.size * scale + 2;
  Image img;
  img.allocate(columns * cell, rows * cell);
  std::vector<std::uint8_t> tile(clip.frame_bytes());
  for (const int k : std::views::iota(0, shown)) {
    composite(clip.frame(k * step), clip.size, bg, tile);
    blit(img, tile, clip.size, (k % columns) * cell + 1, (k / columns) * cell + 1, scale);
  }
  return img;
}

Image comparison_sheet(std::span<const Clip* const> rows, int columns, Background bg, int scale) {
  if (rows.empty()) return {};
  const int size = rows[0]->size;
  const int cell = size * scale + 2;
  Image img;
  img.allocate(columns * cell, static_cast<int>(rows.size()) * cell);
  std::vector<std::uint8_t> tile(rows[0]->frame_bytes());
  for (const auto [r, clip] : std::views::enumerate(rows)) {
    if (clip->size != size) continue;
    for (const int k : std::views::iota(0, columns)) {
      const int f = columns > 1 ? k * (clip->frames - 1) / (columns - 1) : 0;
      composite(clip->frame(f), size, bg, tile);
      blit(img, tile, size, k * cell + 1, static_cast<int>(r) * cell + 1, scale);
    }
  }
  return img;
}

std::expected<void, std::string> write_video(const std::filesystem::path& path, std::span<const Image> frames, float fps) {
  if (frames.empty()) return std::unexpected("video: no frames");
  std::error_code ec;
  if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path(), ec);
  const int w = frames[0].width & ~1, h = frames[0].height & ~1;  // H.264 wants even sizes
  const std::string cmd = std::format(
      "ffmpeg -loglevel error -y -f rawvideo -pix_fmt rgba -s {}x{} -r {} -i - -pix_fmt yuv420p -c:v libx264 -crf 18 "
      "\"{}\"",
      w, h, fps, path.string());
  FILE* p = popen(cmd.c_str(), "w");
  if (!p) return std::unexpected("video: cannot start ffmpeg");
  bool ok = true;
  for (const Image& f : frames) {
    for (int y = 0; y < h && ok; ++y) {
      const std::size_t n = static_cast<std::size_t>(w) * 4;
      ok = std::fwrite(f.rgba.data() + static_cast<std::size_t>(y) * f.width * 4, 1, n, p) == n;
    }
  }
  if (pclose(p) != 0 || !ok) return std::unexpected("video: ffmpeg failed (is it installed?)");
  return {};
}

std::expected<void, std::string> write_comparison_video(const std::filesystem::path& path,
                                                        std::span<const Clip* const> clips, Background bg, int scale,
                                                        int loops) {
  if (clips.empty()) return std::unexpected("video: no clips");
  const int size = clips[0]->size;
  const int frames = std::ranges::max(clips | std::views::transform([](const Clip* c) { return c->frames; }));
  const int cell = size * scale + 4;
  std::vector<Image> out;
  std::vector<std::uint8_t> tile(clips[0]->frame_bytes());
  for (int l = 0; l < loops; ++l) {
    for (int f = 0; f < frames; ++f) {
      Image img;
      img.allocate(static_cast<int>(clips.size()) * cell, cell);
      for (const auto [c, clip] : std::views::enumerate(clips)) {
        if (clip->size != size) continue;
        composite(clip->frame(std::min(f, clip->frames - 1)), size, bg, tile);
        blit(img, tile, size, static_cast<int>(c) * cell + 2, 2, scale);
      }
      out.push_back(std::move(img));
    }
  }
  return write_video(path, out, clips[0]->fps);
}

}  // namespace nfx
