// Video codecs as baselines for effect compression (results/compression, study F2): a clip goes through ffmpeg
// (x264, x265, VP9, AV1 by libaom, SVT-AV1 and rav1e) at one quality setting and comes back decoded, with the bytes a
// game would store and what decoding cost. Everything runs through an ffmpeg pipe; nothing here links a codec.
//
// Alpha. VP9 carries it natively (yuva420p: a second VP9 stream for the alpha plane, stored by WebM as a block
// addition). The other codecs have no alpha in this ffmpeg, so the clip is stacked: premultiplied colour above, alpha
// as grey below, in one frame of twice the height (the usual way games ship alpha in video).
//
// Chroma. 4:2:0 alone (no codec) caps a fire clip at about 29 dB of active PSNR: small flames change colour faster
// than every other pixel. So the stacked codecs code 4:4:4 where they can (x264, x265, VP9, libaom, rav1e); SVT-AV1
// codes 4:2:0 only, and VP9 with native alpha is 4:2:0 (yuva420p), as WebM stores it.
//
// Bytes. `payload` counts what the codec produced: the elementary stream for H.264 and HEVC (Annex B, parameter sets
// included, encoder-information messages removed), the frames of an IVF file (its 32-byte header and 12 bytes per
// frame excluded), and for WebM the blocks and block additions (EBML structure excluded). `file` is the file as
// written.
#pragma once

#include <neuralfx/clip.hpp>

#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace nfx::video {

struct Codec {
  std::string name;              // short id: x264, x265, vp9, aom, svt, rav1e
  std::string encoder;           // ffmpeg encoder
  std::string args;              // encoder options; "{q}" is replaced by the quality setting
  std::string format;            // ffmpeg muxer: h264, hevc, ivf, webm
  std::string decoder;           // ffmpeg decoder to force ("" = ffmpeg's choice)
  bool native_alpha = false;     // yuva420p in WebM; otherwise stacked colour over alpha
  std::string pix_fmt;           // the stacked frame's pixel format (yuv444p or yuv420p)
  std::vector<int> ladder;       // quality settings (CRF or QP), best quality first
  int refs = 0;                  // reference frames the decoder keeps (its stream's limit): sizes the decoder's state
};

// The codecs of study F2 with their quality ladders. Encoders run single-threaded at slow, high-efficiency settings.
std::vector<Codec> codecs();

struct Coded {
  Clip decoded;
  std::size_t file_bytes = 0, payload_bytes = 0;
  double encode_cpu_s = 0, decode_cpu_s = 0;  // user + system seconds of the ffmpeg process
  long decode_max_rss_kb = 0;                 // peak resident memory of the decoding ffmpeg process
};

// Encode `clip` with `codec` at quality `q` into `work` (a scratch directory), decode it back. Errors: ffmpeg missing
// or failing, or a stream that does not parse.
std::expected<Coded, std::string> code(const Clip& clip, const Codec& codec, int q, const std::filesystem::path& work);

// Payload of a container (see above); nullopt-like 0 with an error when it does not parse.
std::expected<std::size_t, std::string> ivf_payload(std::span<const std::uint8_t> file);
std::expected<std::size_t, std::string> webm_payload(std::span<const std::uint8_t> file);

// Run a shell command with optional stdin bytes, capturing stdout; returns its exit status, user + system CPU seconds
// and peak resident memory (KB). The command is exec'd by /bin/sh, so the measured process is the command itself.
struct RunResult {
  int status = -1;
  double cpu_s = 0;
  long max_rss_kb = 0;
  std::vector<std::uint8_t> out;
};
std::expected<RunResult, std::string> run(const std::string& command, std::span<const std::uint8_t> input = {});

}  // namespace nfx::video
