// A small ffmpeg-pipe harness for video-codec baselines (study G3, docs/DCM.md §8): premultiplied RGBA frames in,
// coded bytes, decoded RGBA frames and decode time out. Self-contained (the standard library and POSIX popen); the
// ffmpeg binary must be on the PATH.
//
// Alpha: codecs without an alpha plane get the colour (premultiplied RGB, i.e. over black) on top and the alpha as grey
// below, in one frame twice as tall (the usual trick for alpha video in games); VP9 also codes alpha natively (WebM).
// Bytes: the elementary stream only (SEI messages with encoder settings stripped from H.264 and H.265, the IVF framing
// of VP9 and AV1 subtracted), except native VP9 alpha, whose alpha lives in the WebM container (whole file counted).
#pragma once

#include <sys/resource.h>
#include <sys/wait.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <format>
#include <span>
#include <string>
#include <vector>

namespace nfx::video {

struct Codec {
  std::string name;       // "x264", "x265", "vp9", "vp9a", "aom", "svt", "rav1e"
  std::string ext;        // file extension: "264", "hevc", "ivf", "webm"
  std::string args;       // encoder arguments; "{q}" is replaced by the quality value
  bool alpha = false;     // native alpha (yuva420p, WebM); else stacked colour over alpha
  std::string format;     // muxer: "h264", "hevc", "ivf", "webm"
  std::string strip;      // bitstream filter applied to the output (removes SEI), or empty
  std::string decoder;    // decoder to force on input ("libvpx-vp9" keeps the alpha), or empty
  std::vector<int> ladder;  // quality values, best first
};

// The codecs of the study, each at a ladder of quality values (CRF or QP), one thread, constant quality, a single
// keyframe for the whole run (as a stored effect would be played: from the start).
inline std::vector<Codec> study_codecs() {
  return {
      {"x264", "264", "-c:v libx264 -preset veryslow -crf {q} -g 1000 -pix_fmt yuv420p -threads 1", false, "h264",
       "filter_units=remove_types=6", "", {12, 18, 24, 30, 36, 42, 47, 51}},
      {"x265", "hevc", "-c:v libx265 -preset slow -crf {q} -pix_fmt yuv420p -x265-params pools=1:frame-threads=1:keyint=1000:info=0:log-level=error",
       false, "hevc", "filter_units=remove_types=39|40", "", {12, 18, 24, 30, 36, 42, 47, 51}},
      {"vp9", "ivf", "-c:v libvpx-vp9 -b:v 0 -crf {q} -deadline good -cpu-used 1 -row-mt 0 -g 1000 -pix_fmt yuv420p -threads 1", false, "ivf", "", "",
       {10, 20, 30, 40, 48, 55, 60, 63}},
      {"vp9a", "webm", "-c:v libvpx-vp9 -b:v 0 -crf {q} -deadline good -cpu-used 1 -row-mt 0 -g 1000 -pix_fmt yuva420p -auto-alt-ref 0 -threads 1", true,
       "webm", "", "libvpx-vp9", {10, 20, 30, 40, 48, 55, 60, 63}},
      {"aom", "ivf", "-c:v libaom-av1 -b:v 0 -crf {q} -cpu-used 4 -row-mt 0 -g 1000 -pix_fmt yuv420p -threads 1", false, "ivf", "", "",
       {10, 20, 30, 40, 48, 55, 60, 63}},
      {"svt", "ivf", "-c:v libsvtav1 -crf {q} -preset 6 -g 1000 -pix_fmt yuv420p -svtav1-params lp=1", false, "ivf", "", "",
       {10, 20, 30, 40, 48, 55, 60, 63}},
      // 4:4:4 and RGB variants: fire's saturated colours and hard edges lose most to chroma subsampling
      {"x264_444", "264", "-c:v libx264 -preset veryslow -crf {q} -g 1000 -pix_fmt yuv444p -threads 1", false, "h264", "filter_units=remove_types=6", "",
       {12, 18, 24, 30, 36, 42, 47, 51}},
      {"x264_rgb", "264", "-c:v libx264rgb -preset veryslow -crf {q} -g 1000 -pix_fmt bgr0 -threads 1", false, "h264", "filter_units=remove_types=6", "",
       {12, 18, 24, 30, 36, 42, 47, 51}},
      {"x265_444", "hevc", "-c:v libx265 -preset slow -crf {q} -pix_fmt yuv444p -x265-params pools=1:frame-threads=1:keyint=1000:info=0:log-level=error",
       false, "hevc", "filter_units=remove_types=39|40", "", {12, 18, 24, 30, 36, 42, 47, 51}},
      {"vp9_444", "ivf", "-c:v libvpx-vp9 -b:v 0 -crf {q} -deadline good -cpu-used 1 -row-mt 0 -g 1000 -pix_fmt yuv444p -threads 1", false, "ivf", "", "",
       {10, 20, 30, 40, 48, 55, 60, 63}},
      {"aom_444", "ivf", "-c:v libaom-av1 -b:v 0 -crf {q} -cpu-used 4 -row-mt 0 -g 1000 -pix_fmt yuv444p -threads 1", false, "ivf", "", "",
       {10, 20, 30, 40, 48, 55, 60, 63}},
      {"rav1e", "ivf", "-c:v librav1e -qp {q} -speed 6 -g 1000 -pix_fmt yuv420p -threads 1", false, "ivf", "", "",
       {40, 80, 120, 160, 200, 230, 255}},
  };
}

struct Result {
  bool ok = false;
  std::string error;
  std::size_t file_bytes = 0;  // as written
  std::size_t bytes = 0;       // the elementary stream (see above)
  std::vector<std::uint8_t> rgba;  // decoded, frames * size * size * 4, premultiplied
  double encode_cpu_s = 0, decode_cpu_s = 0;  // CPU time of the ffmpeg processes (user + system)
};

namespace detail {
inline double child_cpu() {
  rusage u{};
  getrusage(RUSAGE_CHILDREN, &u);
  return static_cast<double>(u.ru_utime.tv_sec + u.ru_stime.tv_sec) + 1e-6 * static_cast<double>(u.ru_utime.tv_usec + u.ru_stime.tv_usec);
}
inline std::string replace_q(std::string s, int q) {
  const auto p = s.find("{q}");
  if (p != std::string::npos) s.replace(p, 3, std::to_string(q));
  return s;
}
}  // namespace detail

// Codes `frames` premultiplied RGBA frames of side `size` and decodes them. `work` is a scratch directory; `tag` names
// the files. Child CPU times are measured with getrusage, so callers that run codecs in several threads at once get
// mixed figures: time with one thread (or use decode_cpu_seconds below).
inline Result roundtrip(const Codec& c, int q, std::span<const std::uint8_t> rgba, int size, int frames, float fps,
                        const std::filesystem::path& work, const std::string& tag) {
  Result r;
  const std::size_t fb = static_cast<std::size_t>(size) * static_cast<std::size_t>(size) * 4;
  std::filesystem::create_directories(work);
  const std::filesystem::path file = work / std::format("{}_{}_q{}.{}", tag, c.name, q, c.ext);
  const int h = c.alpha ? size : 2 * size;
  const std::string in_fmt = c.alpha ? "rgba" : "rgb24";
  std::string enc = std::format("SVT_LOG=1 ffmpeg -loglevel error -y -f rawvideo -pix_fmt {} -s {}x{} -r {} -i - {} ", in_fmt, size, h, fps, detail::replace_q(c.args, q));
  if (!c.strip.empty()) enc += std::format("-bsf:v \"{}\" ", c.strip);
  enc += std::format("-f {} \"{}\"", c.format, file.string());
  const double c0 = detail::child_cpu();
  FILE* p = popen(enc.c_str(), "w");
  if (!p) {
    r.error = "cannot start ffmpeg";
    return r;
  }
  std::vector<std::uint8_t> buf(static_cast<std::size_t>(size) * static_cast<std::size_t>(h) * (c.alpha ? 4 : 3));
  bool wrote = true;
  for (int f = 0; f < frames && wrote; ++f) {
    const std::uint8_t* src = rgba.data() + static_cast<std::size_t>(f) * fb;
    if (c.alpha) {
      std::copy(src, src + fb, buf.begin());
    } else {
      const std::size_t n = static_cast<std::size_t>(size) * static_cast<std::size_t>(size);
      for (std::size_t i = 0; i < n; ++i) {
        buf[i * 3] = src[i * 4];
        buf[i * 3 + 1] = src[i * 4 + 1];
        buf[i * 3 + 2] = src[i * 4 + 2];
        const std::size_t j = (n + i) * 3;
        buf[j] = buf[j + 1] = buf[j + 2] = src[i * 4 + 3];
      }
    }
    wrote = std::fwrite(buf.data(), 1, buf.size(), p) == buf.size();
  }
  if (pclose(p) != 0 || !wrote) {
    r.error = "ffmpeg encode failed: " + enc;
    return r;
  }
  r.encode_cpu_s = detail::child_cpu() - c0;
  r.file_bytes = static_cast<std::size_t>(std::filesystem::file_size(file));
  r.bytes = r.file_bytes;
  if (c.format == "ivf") {
    const std::size_t framing = 32 + 12 * static_cast<std::size_t>(frames);
    r.bytes = r.file_bytes > framing ? r.file_bytes - framing : 0;
  }
  const std::string dec = std::format("ffmpeg -loglevel error -threads 1 {} -i \"{}\" -f rawvideo -pix_fmt {} -", c.decoder.empty() ? "" : "-c:v " + c.decoder,
                                      file.string(), in_fmt);
  const double c1 = detail::child_cpu();
  FILE* d = popen(dec.c_str(), "r");
  if (!d) {
    r.error = "cannot start ffmpeg";
    return r;
  }
  r.rgba.assign(static_cast<std::size_t>(frames) * fb, 0);
  int got = 0;
  for (; got < frames; ++got) {
    if (std::fread(buf.data(), 1, buf.size(), d) != buf.size()) break;
    std::uint8_t* dst = r.rgba.data() + static_cast<std::size_t>(got) * fb;
    if (c.alpha) {
      std::copy(buf.begin(), buf.end(), dst);
    } else {
      const std::size_t n = static_cast<std::size_t>(size) * static_cast<std::size_t>(size);
      for (std::size_t i = 0; i < n; ++i) {
        const std::size_t j = (n + i) * 3;
        dst[i * 4] = buf[i * 3];
        dst[i * 4 + 1] = buf[i * 3 + 1];
        dst[i * 4 + 2] = buf[i * 3 + 2];
        dst[i * 4 + 3] = static_cast<std::uint8_t>((buf[j] + buf[j + 1] + buf[j + 2] + 1) / 3);
      }
    }
  }
  std::vector<std::uint8_t> rest(4096);
  while (std::fread(rest.data(), 1, rest.size(), d) > 0) {
  }
  const int status = pclose(d);
  r.decode_cpu_s = detail::child_cpu() - c1;
  if (got != frames || status != 0) {
    r.error = std::format("ffmpeg decode gave {} of {} frames", got, frames);
    return r;
  }
  r.ok = true;
  return r;
}

// Decode-only CPU time of a coded file (no conversion to RGBA, null output), the least of `reps` runs, pinned to `cpu`
// (taskset) when cpu >= 0. Returns ffmpeg's own user + system time (-benchmark) in seconds and its peak resident memory
// in KiB, or negative values on failure.
inline std::pair<double, double> decode_cpu_seconds(const std::filesystem::path& file, const std::string& decoder, int reps, int cpu) {
  double best = -1, rss = -1;
  for (int i = 0; i < reps; ++i) {
    const std::string cmd = std::format("{}ffmpeg -hide_banner -nostats -benchmark -threads 1 {} -i \"{}\" -f null - 2>&1",
                                        cpu >= 0 ? std::format("taskset -c {} ", cpu) : "", decoder.empty() ? "" : "-c:v " + decoder, file.string());
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) return {-1, -1};
    std::string out;
    char line[512];
    while (std::fgets(line, sizeof line, p)) out += line;
    if (pclose(p) != 0) return {-1, -1};
    double ut = 0, st = 0, mr = 0;
    if (const auto a = out.find("utime="); a != std::string::npos) ut = std::stod(out.substr(a + 6));
    if (const auto a = out.find("stime="); a != std::string::npos) st = std::stod(out.substr(a + 6));
    if (const auto a = out.find("maxrss="); a != std::string::npos) mr = std::stod(out.substr(a + 7));
    if (best < 0 || ut + st < best) best = ut + st;
    rss = std::max(rss, mr);
  }
  return {best, rss};
}

}  // namespace nfx::video
