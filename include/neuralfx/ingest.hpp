// Phase 1b: the owner's own effect videos into clips (docs/PLAN.md §5.3, docs/DATA.md).
//
// Licences come first: a clip is only made when its licence allows training a model that ships in a game, and
// every ingested clip gets a row in the licence register (<data root>/licences.tsv). Raw videos, frames and clips
// stay under the data root, never in git.
#pragma once

#include <neuralfx/clip.hpp>

#include <expected>
#include <filesystem>
#include <string>
#include <vector>

namespace nfx::ingest {

// How alpha is made for footage without an alpha channel.
enum class AlphaMode {
  additive,  // alpha = 0: pure emission on black (fire, sparks, magic glows)
  luma,      // alpha = max(r, g, b): black-keyed footage with coverage (smoke and fire on black)
  keep,      // the footage has alpha (straight); it is premultiplied here
};
bool parse_alpha(std::string_view text, AlphaMode& out);

struct Licence {
  std::string id;           // e.g. "own", "CC0-1.0", "CC-BY-4.0", "commercial"
  std::string source;       // URL or a description of where the footage came from
  std::string author;       // for attribution
  std::string note;         // required for "commercial": the agreement that allows ML training
};

// Whether a licence allows training a shipped model. Accepted: own footage, CC0, CC-BY (attribution recorded),
// CC-BY-SA (flagged: share-alike may reach the trained model), "commercial" with a note naming the agreement.
// Refused: NonCommercial and NoDerivatives licences, "unknown", "all rights reserved", anything else.
std::expected<void, std::string> check_licence(const Licence& l);

struct Options {
  int size = 128;
  float fps = 30.f;
  float start = 0.f;        // seconds into the video
  float duration = 0.f;     // seconds; 0 = to the end (capped by max_frames)
  int max_frames = 256;
  std::string crop;         // ffmpeg crop "w:h:x:y"; empty = centre square
  AlphaMode alpha = AlphaMode::luma;
  int loop_blend = 0;       // > 0: make a seamless loop by crossfading this many extra frames into the start
  std::string name;         // effect name stored in the clip
};

// Decode with ffmpeg (any format it reads) into a clip at size x size, square-cropped, resampled to fps.
std::expected<Clip, std::string> from_video(const std::filesystem::path& video, const Options& o);
// Read a directory of PAM (RGB_ALPHA or RGB) or binary PPM frames, sorted by name, already square.
std::expected<Clip, std::string> from_frames(const std::filesystem::path& dir, const Options& o);

// Convert straight RGB(A) frames to the premultiplied clip convention per the alpha mode; make a loop if asked.
void finish(Clip& clip, const Options& o);

// Append a row to the licence register; creates it with a header if missing.
std::expected<void, std::string> register_clip(const std::filesystem::path& register_tsv, const Licence& l,
                                               const std::filesystem::path& input, const std::filesystem::path& output,
                                               const Clip& clip);

}  // namespace nfx::ingest
