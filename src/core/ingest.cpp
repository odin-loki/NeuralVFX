#include <neuralfx/ingest.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <format>
#include <fstream>
#include <ranges>
#include <sstream>

namespace nfx::ingest {

bool parse_alpha(std::string_view text, AlphaMode& out) {
  if (text == "additive") out = AlphaMode::additive;
  else if (text == "luma") out = AlphaMode::luma;
  else if (text == "keep") out = AlphaMode::keep;
  else return false;
  return true;
}

namespace {

std::string upper(std::string_view s) {
  std::string u(s);
  std::ranges::transform(u, u.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return u;
}

}  // namespace

std::expected<void, std::string> check_licence(const Licence& l) {
  const std::string id = upper(l.id);
  if (id.empty() || id == "UNKNOWN") return std::unexpected("licence missing or unknown: footage of unknown licence cannot be used");
  if (id.find("-NC") != std::string::npos || id.find("NONCOMMERCIAL") != std::string::npos) {
    return std::unexpected(std::format("licence {} forbids commercial use: a model shipped in a game is commercial", l.id));
  }
  if (id.find("-ND") != std::string::npos || id.find("NODERIV") != std::string::npos) {
    return std::unexpected(std::format("licence {} forbids derivatives: a trained model is one", l.id));
  }
  if (id.find("ALL RIGHTS RESERVED") != std::string::npos || id == "ARR") return std::unexpected("all rights reserved: needs a licence");
  if (l.source.empty()) return std::unexpected("record where the footage came from (--source)");
  if (id == "OWN" || id.starts_with("CC0") || id == "PUBLIC-DOMAIN") return {};
  if (id.starts_with("CC-BY")) {
    if (l.author.empty()) return std::unexpected(std::format("{} needs the author for attribution (--author)", l.id));
    return {};  // CC-BY-SA is accepted but flagged in the register
  }
  if (id == "COMMERCIAL") {
    if (l.note.empty()) return std::unexpected("commercial footage needs --licence-note naming the agreement that allows ML training");
    return {};
  }
  return std::unexpected(std::format("licence {} is not on the accepted list (own, CC0, CC-BY, CC-BY-SA, commercial)", l.id));
}

void finish(Clip& clip, const Options& o) {
  for (std::size_t i = 0; i < clip.rgba.size(); i += 4) {
    std::uint8_t* p = clip.rgba.data() + i;
    switch (o.alpha) {
      case AlphaMode::additive: p[3] = 0; break;
      case AlphaMode::luma: p[3] = std::max({p[0], p[1], p[2]}); break;  // colour on black is already premultiplied
      case AlphaMode::keep:
        for (int c = 0; c < 3; ++c) p[c] = static_cast<std::uint8_t>((p[c] * p[3] + 127) / 255);
        break;
    }
  }
  if (o.loop_blend > 0 && clip.frames > 2 * o.loop_blend) {
    const int n = clip.frames - o.loop_blend;  // the extra frames fade into the start
    Clip out = slice_clip(clip, 0, n);
    for (int i = 0; i < o.loop_blend; ++i) {
      const float w = (static_cast<float>(i) + 0.5f) / static_cast<float>(o.loop_blend);
      const auto a = clip.frame(i), e = clip.frame(n + i);
      auto d = out.frame(i);
      for (std::size_t k = 0; k < d.size(); ++k) d[k] = static_cast<std::uint8_t>(static_cast<float>(e[k]) * (1.f - w) + static_cast<float>(a[k]) * w + 0.5f);
    }
    out.loop = true;
    clip = std::move(out);
  }
}

std::expected<Clip, std::string> from_video(const std::filesystem::path& video, const Options& o) {
  if (!std::filesystem::exists(video)) return std::unexpected(std::format("no such file: {}", video.string()));
  if (o.size < 16 || o.size > 1024) return std::unexpected("size out of range");
  const std::string crop = o.crop.empty() ? "crop='min(iw,ih)':'min(iw,ih)'" : std::format("crop={}", o.crop);
  std::string cmd = std::format("ffmpeg -loglevel error -ss {} ", o.start);
  if (o.duration > 0) cmd += std::format("-t {} ", o.duration);
  cmd += std::format("-i \"{}\" -vf \"fps={},{},scale={}:{}:flags=area,format=rgba\" -frames:v {} -f rawvideo -pix_fmt rgba -",
                     video.string(), o.fps, crop, o.size, o.size, o.max_frames);
  FILE* p = popen(cmd.c_str(), "r");
  if (!p) return std::unexpected("cannot start ffmpeg");
  Clip clip;
  clip.size = o.size;
  clip.fps = o.fps;
  clip.effect = o.name;
  clip.source = "ingest";
  std::vector<std::uint8_t> frame(static_cast<std::size_t>(o.size) * o.size * 4);
  while (std::fread(frame.data(), 1, frame.size(), p) == frame.size()) {
    clip.rgba.insert(clip.rgba.end(), frame.begin(), frame.end());
    ++clip.frames;
  }
  if (pclose(p) != 0 && clip.frames == 0) return std::unexpected("ffmpeg failed (is it installed, and is the file a video?)");
  if (clip.frames == 0) return std::unexpected("no frames decoded");
  finish(clip, o);
  return clip;
}

namespace {

// A PAM (P7) or binary PPM (P6) image; returns RGBA.
std::expected<std::vector<std::uint8_t>, std::string> read_image(const std::filesystem::path& path, int& w, int& h) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return std::unexpected(std::format("cannot open {}", path.string()));
  std::string magic;
  in >> magic;
  int depth = 3, maxval = 255;
  if (magic == "P7") {
    std::string key;
    while (in >> key && key != "ENDHDR") {
      if (key == "WIDTH") in >> w;
      else if (key == "HEIGHT") in >> h;
      else if (key == "DEPTH") in >> depth;
      else if (key == "MAXVAL") in >> maxval;
      else if (key == "TUPLTYPE") in >> key;
    }
  } else if (magic == "P6") {
    in >> w >> h >> maxval;
  } else {
    return std::unexpected(std::format("{}: not a PAM or PPM image", path.string()));
  }
  in.get();
  if (w <= 0 || h <= 0 || maxval != 255 || (depth != 3 && depth != 4)) return std::unexpected(std::format("{}: unsupported image", path.string()));
  std::vector<std::uint8_t> raw(static_cast<std::size_t>(w) * h * depth);
  in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
  if (!in) return std::unexpected(std::format("{}: truncated", path.string()));
  if (depth == 4) return raw;
  std::vector<std::uint8_t> rgba(static_cast<std::size_t>(w) * h * 4, 255);
  for (std::size_t i = 0; i < static_cast<std::size_t>(w) * h; ++i) std::copy_n(raw.begin() + static_cast<std::ptrdiff_t>(i * 3), 3, rgba.begin() + static_cast<std::ptrdiff_t>(i * 4));
  return rgba;
}

}  // namespace

std::expected<Clip, std::string> from_frames(const std::filesystem::path& dir, const Options& o) {
  std::vector<std::filesystem::path> files;
  for (const auto& e : std::filesystem::directory_iterator(dir)) {
    const auto ext = e.path().extension();
    if (ext == ".pam" || ext == ".ppm") files.push_back(e.path());
  }
  std::ranges::sort(files);
  if (files.empty()) return std::unexpected(std::format("no .pam or .ppm frames in {}", dir.string()));
  Clip clip;
  clip.fps = o.fps;
  clip.effect = o.name;
  clip.source = "ingest";
  for (const auto& f : files | std::views::take(o.max_frames)) {
    int w = 0, h = 0;
    auto img = read_image(f, w, h);
    if (!img) return std::unexpected(img.error());
    if (w != h) return std::unexpected(std::format("{}: frames must be square (use the video path to crop)", f.string()));
    if (clip.frames == 0) clip.size = w;
    if (w != clip.size) return std::unexpected("frames differ in size");
    clip.rgba.insert(clip.rgba.end(), img->begin(), img->end());
    ++clip.frames;
  }
  finish(clip, o);
  return clip;
}

std::expected<void, std::string> register_clip(const std::filesystem::path& tsv, const Licence& l, const std::filesystem::path& input,
                                               const std::filesystem::path& output, const Clip& clip) {
  if (auto ok = check_licence(l); !ok) return ok;
  std::error_code ec;
  if (tsv.has_parent_path()) std::filesystem::create_directories(tsv.parent_path(), ec);
  const bool fresh = !std::filesystem::exists(tsv);
  std::ofstream out(tsv, std::ios::app);
  if (!out) return std::unexpected(std::format("cannot write {}", tsv.string()));
  if (fresh) out << "date\tname\tlicence\tauthor\tsource\tnote\tflags\tinput\toutput\tframes\tsize\n";
  const auto clean = [](std::string s) {
    std::ranges::replace(s, '\t', ' ');
    std::ranges::replace(s, '\n', ' ');
    return s;
  };
  const std::string flags = upper(l.id).find("-SA") != std::string::npos ? "share-alike: check that it does not reach the shipped model" : "";
  out << std::format("{:%F}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n", std::chrono::floor<std::chrono::days>(std::chrono::system_clock::now()),
                     clean(clip.effect), clean(l.id), clean(l.author), clean(l.source), clean(l.note), flags, clean(input.string()),
                     clean(output.string()), clip.frames, clip.size);
  return {};
}

}  // namespace nfx::ingest
