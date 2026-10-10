// Video codecs through an ffmpeg pipe (include/neuralfx/video_codec.hpp).
#include <neuralfx/video_codec.hpp>

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cstring>
#include <format>
#include <fcntl.h>
#include <fstream>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

namespace nfx::video {

namespace fs = std::filesystem;

std::vector<Codec> codecs() {
  // Single-threaded, slow presets (the codecs' best compression), one keyframe for the whole 64-frame clip. Each
  // ladder spans roughly 22 to 45 dB of active PSNR on the study's clips, best quality first.
  const std::vector<int> crf63 = {6, 12, 18, 24, 30, 36, 42, 48, 54, 60};
  return {
      {"x264", "libx264", "-preset veryslow -crf {q} -threads 1 -bsf:v filter_units=remove_types=6", "h264", "", false, "yuv444p",
       {4, 8, 12, 16, 20, 24, 28, 32, 36, 40}, 16},
      {"x265", "libx265", "-preset veryslow -crf {q} -x265-params pools=1:frame-threads=1:info=0:log-level=error", "hevc", "",
       false, "yuv444p", {4, 8, 12, 16, 20, 24, 28, 32, 36, 40}, 5},
      {"vp9", "libvpx-vp9", "-crf {q} -b:v 0 -cpu-used 1 -deadline good -row-mt 0 -threads 1 -auto-alt-ref 0", "ivf", "", false,
       "yuv444p", crf63, 8},
      {"vp9a", "libvpx-vp9", "-crf {q} -b:v 0 -cpu-used 1 -deadline good -row-mt 0 -threads 1 -auto-alt-ref 0", "webm",
       "libvpx-vp9", true, "yuva420p", crf63, 8},
      {"aom", "libaom-av1", "-crf {q} -b:v 0 -cpu-used 3 -row-mt 0 -threads 1 -tiles 1x1", "ivf", "", false, "yuv444p", crf63, 8},
      {"svt", "libsvtav1", "-crf {q} -preset 4 -svtav1-params lp=1", "ivf", "", false, "yuv420p", crf63, 8},
      {"rav1e", "librav1e", "-qp {q} -speed 4 -rav1e-params threads=1", "ivf", "", false, "yuv444p",
       {10, 35, 60, 85, 110, 135, 160, 185, 210, 235}, 8},
  };
}

std::expected<RunResult, std::string> run(const std::string& command, std::span<const std::uint8_t> input) {
  int in_pipe[2], out_pipe[2];
  if (pipe(in_pipe) != 0) return std::unexpected("pipe failed");
  if (pipe(out_pipe) != 0) {
    close(in_pipe[0]);
    close(in_pipe[1]);
    return std::unexpected("pipe failed");
  }
  const pid_t pid = fork();
  if (pid < 0) return std::unexpected("fork failed");
  if (pid == 0) {
    dup2(in_pipe[0], 0);
    dup2(out_pipe[1], 1);
    close(in_pipe[0]);
    close(in_pipe[1]);
    close(out_pipe[0]);
    close(out_pipe[1]);
    const std::string cmd = "exec " + command;
    execl("/bin/sh", "sh", "-c", cmd.c_str(), static_cast<char*>(nullptr));
    _exit(127);
  }
  close(in_pipe[0]);
  close(out_pipe[1]);
  RunResult r;
  // Feed stdin and drain stdout without deadlocking: a writer thread would do; here the input is written in pieces
  // between reads using non-blocking descriptors kept simple with poll-free alternation.
  std::size_t sent = 0;
  std::vector<std::uint8_t> buf(1 << 16);
  bool in_open = true;
  if (input.empty()) {
    close(in_pipe[1]);
    in_open = false;
  }
  // Make stdout non-blocking so the loop can alternate.
  const int flags = fcntl(out_pipe[0], F_GETFL, 0);
  fcntl(out_pipe[0], F_SETFL, flags | O_NONBLOCK);
  bool out_open = true;
  while (out_open) {
    bool progress = false;
    if (in_open) {
      const std::size_t n = std::min<std::size_t>(input.size() - sent, 1 << 16);
      const ssize_t w = write(in_pipe[1], input.data() + sent, n);
      if (w > 0) {
        sent += static_cast<std::size_t>(w);
        progress = true;
      } else if (w < 0 && errno != EINTR) {
        sent = input.size();  // the reader went away; stop feeding
      }
      if (sent >= input.size()) {
        close(in_pipe[1]);
        in_open = false;
      }
    }
    const ssize_t rd = read(out_pipe[0], buf.data(), buf.size());
    if (rd > 0) {
      r.out.insert(r.out.end(), buf.begin(), buf.begin() + rd);
      progress = true;
    } else if (rd == 0) {
      out_open = false;
    } else if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
      out_open = false;
    }
    if (!progress && out_open && !in_open) {
      // Only output left: block on it.
      fcntl(out_pipe[0], F_SETFL, flags);
    } else if (!progress) {
      usleep(200);
    }
  }
  if (in_open) close(in_pipe[1]);
  close(out_pipe[0]);
  int status = 0;
  rusage ru{};
  if (wait4(pid, &status, 0, &ru) < 0) return std::unexpected("wait failed");
  r.status = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
  r.cpu_s = static_cast<double>(ru.ru_utime.tv_sec) + 1e-6 * static_cast<double>(ru.ru_utime.tv_usec) +
            static_cast<double>(ru.ru_stime.tv_sec) + 1e-6 * static_cast<double>(ru.ru_stime.tv_usec);
  r.max_rss_kb = ru.ru_maxrss;
  return r;
}

namespace {

std::uint64_t le(std::span<const std::uint8_t> b, std::size_t at, int n) {
  std::uint64_t v = 0;
  for (int i = 0; i < n; ++i) v |= static_cast<std::uint64_t>(b[at + static_cast<std::size_t>(i)]) << (8 * i);
  return v;
}

// An EBML variable-length integer at b[at]: its value (marker bit removed) and length; length 0 on error.
std::pair<std::uint64_t, int> vint(std::span<const std::uint8_t> b, std::size_t at) {
  if (at >= b.size() || b[at] == 0) return {0, 0};
  const int len = std::countl_zero(static_cast<unsigned>(b[at])) - 23;  // leading zeros within the byte, plus one
  if (len < 1 || len > 8 || at + static_cast<std::size_t>(len) > b.size()) return {0, 0};
  std::uint64_t v = b[at] & (0xffu >> len);
  for (int i = 1; i < len; ++i) v = (v << 8) | b[at + static_cast<std::size_t>(i)];
  return {v, len};
}

// An EBML element id (marker bit kept) and its length.
std::pair<std::uint32_t, int> ebml_id(std::span<const std::uint8_t> b, std::size_t at) {
  if (at >= b.size() || b[at] == 0) return {0, 0};
  const int len = std::countl_zero(static_cast<unsigned>(b[at])) - 23;
  if (len < 1 || len > 4 || at + static_cast<std::size_t>(len) > b.size()) return {0, 0};
  std::uint32_t id = 0;
  for (int i = 0; i < len; ++i) id = (id << 8) | b[at + static_cast<std::size_t>(i)];
  return {id, len};
}

// Sum of block payloads in [at, end): SimpleBlock and Block data after their track, timecode and flags, plus
// BlockAdditional data. Master elements are entered; everything else is skipped.
bool webm_walk(std::span<const std::uint8_t> b, std::size_t at, std::size_t end, std::size_t& sum, int depth) {
  if (depth > 16) return false;
  while (at < end) {
    const auto [id, il] = ebml_id(b, at);
    if (il == 0) return false;
    const auto [size, sl] = vint(b, at + static_cast<std::size_t>(il));
    if (sl == 0) return false;
    const std::size_t body = at + static_cast<std::size_t>(il + sl);
    const bool unknown = size == (std::uint64_t{1} << (7 * sl)) - 1;  // "unknown size": runs to the parent's end
    const std::size_t stop = unknown ? end : body + static_cast<std::size_t>(size);
    if (stop > end || stop < body) return false;
    switch (id) {
      case 0x18538067:  // Segment
      case 0x1F43B675:  // Cluster
      case 0xA0:        // BlockGroup
      case 0x75A1:      // BlockAdditions
      case 0xA6:        // BlockMore
        if (!webm_walk(b, body, stop, sum, depth + 1)) return false;
        break;
      case 0xA3:    // SimpleBlock
      case 0xA1: {  // Block: track number (vint), 16-bit timecode, flags
        const auto [track, tl] = vint(b, body);
        (void)track;
        if (tl == 0 || body + static_cast<std::size_t>(tl) + 3 > stop) return false;
        sum += stop - (body + static_cast<std::size_t>(tl) + 3);
        break;
      }
      case 0xA5:  // BlockAdditional (the alpha frame)
        sum += stop - body;
        break;
      default:
        break;
    }
    at = stop;
  }
  return true;
}

std::vector<std::uint8_t> read_all(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), {}};
}

std::string replace_q(std::string s, int q) {
  const auto at = s.find("{q}");
  if (at != std::string::npos) s.replace(at, 3, std::to_string(q));
  return s;
}

}  // namespace

std::expected<std::size_t, std::string> ivf_payload(std::span<const std::uint8_t> f) {
  if (f.size() < 32 || std::memcmp(f.data(), "DKIF", 4) != 0) return std::unexpected("not an IVF file");
  const std::size_t header = le(f, 6, 2);
  std::size_t at = header, sum = 0;
  while (at + 12 <= f.size()) {
    const std::size_t n = le(f, at, 4);
    if (at + 12 + n > f.size()) return std::unexpected("truncated IVF frame");
    sum += n;
    at += 12 + n;
  }
  if (at != f.size()) return std::unexpected("trailing bytes in IVF file");
  return sum;
}

std::expected<std::size_t, std::string> webm_payload(std::span<const std::uint8_t> f) {
  std::size_t sum = 0;
  if (!webm_walk(f, 0, f.size(), sum, 0)) return std::unexpected("WebM does not parse");
  return sum;
}

std::expected<Coded, std::string> code(const Clip& clip, const Codec& c, int q, const fs::path& work) {
  std::error_code ec;
  fs::create_directories(work, ec);
  const int S = clip.size, F = clip.frames;
  const bool alpha = c.native_alpha;
  const int H = alpha ? S : 2 * S;
  // Input frames: RGBA as is (native alpha), or colour above alpha (stacked, opaque).
  std::vector<std::uint8_t> in(static_cast<std::size_t>(F) * S * H * 4);
  for (int f = 0; f < F; ++f) {
    const auto src = clip.frame(f);
    std::uint8_t* dst = in.data() + static_cast<std::size_t>(f) * S * H * 4;
    for (std::size_t i = 0; i < clip.pixels(); ++i) {
      if (alpha) {
        std::memcpy(dst + i * 4, src.data() + i * 4, 4);
      } else {
        dst[i * 4] = src[i * 4];
        dst[i * 4 + 1] = src[i * 4 + 1];
        dst[i * 4 + 2] = src[i * 4 + 2];
        dst[i * 4 + 3] = 255;
        std::uint8_t* d2 = dst + (clip.pixels() + i) * 4;
        d2[0] = d2[1] = d2[2] = src[i * 4 + 3];
        d2[3] = 255;
      }
    }
  }
  const fs::path file = work / std::format("{}_q{}.{}", c.name, q, c.format == "h264" ? "264" : c.format == "hevc" ? "265" : c.format);
  const std::string enc = std::format(
      "ffmpeg -nostdin -hide_banner -loglevel error -y -f rawvideo -pix_fmt rgba -s {}x{} -r {} -i - -pix_fmt {} -c:v {} {} -g {} "
      "-f {} '{}' 2>/dev/null",
      S, H, clip.fps, alpha ? "yuva420p" : c.pix_fmt, c.encoder, replace_q(c.args, q), F, c.format, file.string());
  auto er = run(enc, in);
  if (!er) return std::unexpected(er.error());
  if (er->status != 0 || !fs::exists(file)) return std::unexpected(std::format("encoding failed ({}): {}", er->status, enc));
  Coded out;
  out.encode_cpu_s = er->cpu_s;
  const auto bytes = read_all(file);
  out.file_bytes = bytes.size();
  if (c.format == "ivf") {
    auto p = ivf_payload(bytes);
    if (!p) return std::unexpected(p.error());
    out.payload_bytes = *p;
  } else if (c.format == "webm") {
    auto p = webm_payload(bytes);
    if (!p) return std::unexpected(p.error());
    out.payload_bytes = *p;
  } else {
    out.payload_bytes = bytes.size();  // an elementary stream
  }
  const std::string dec = std::format("ffmpeg -nostdin -hide_banner -loglevel error -threads 1 {} -i '{}' -f rawvideo -pix_fmt rgba - 2>/dev/null",
                                      c.decoder.empty() ? std::string{} : "-c:v " + c.decoder, file.string());
  auto dr = run(dec);
  if (!dr) return std::unexpected(dr.error());
  const std::size_t want = static_cast<std::size_t>(F) * S * H * 4;
  if (dr->status != 0 || dr->out.size() != want) {
    return std::unexpected(std::format("decoding failed ({}, {} of {} bytes): {}", dr->status, dr->out.size(), want, dec));
  }
  out.decode_cpu_s = dr->cpu_s;
  out.decode_max_rss_kb = dr->max_rss_kb;
  out.decoded = clip;  // metadata
  for (int f = 0; f < F; ++f) {
    const std::uint8_t* src = dr->out.data() + static_cast<std::size_t>(f) * S * H * 4;
    auto dst = out.decoded.frame(f);
    for (std::size_t i = 0; i < clip.pixels(); ++i) {
      if (alpha) {
        std::memcpy(dst.data() + i * 4, src + i * 4, 4);
      } else {
        dst[i * 4] = src[i * 4];
        dst[i * 4 + 1] = src[i * 4 + 1];
        dst[i * 4 + 2] = src[i * 4 + 2];
        dst[i * 4 + 3] = src[(clip.pixels() + i) * 4 + 1];  // the alpha half, from green (full-resolution luma)
      }
    }
  }
  fs::remove(file, ec);
  return out;
}

}  // namespace nfx::video
