// The effect pack (include/neuralfx/nvfx_pack.h, docs/INTEGRATION.md §5): one file that holds everything a set of
// effects needs, so that an engine ships and loads one file.
//
// Layout (all integers little-endian):
//
//   header (96 bytes)   "NVFXPACK", u32 format (1), u32 header bytes (96), u64 manifest bytes, u64 data bytes,
//                       u64 file bytes (the three sums: a truncated or extended file is refused), the manifest's SHA-256,
//                       24 bytes reserved (zero)
//   manifest            text, one record per line (below), checked by its SHA-256 in the header
//   data                the members' stored bytes, each at its offset from the start of the data, 8-byte aligned
//
// The manifest:
//
//   nvfx-pack 1                                              the format, first
//   member NAME KIND METHOD BYTES STORED OFFSET SHA256 STORED_SHA256
//   needs ENTRY [MODEL ...]                                  the model members a catalogue entry plays
//   cost MODEL SIZE STEP_MS RENDER_MS                        default costs for the effect manager (optional)
//
// KIND is model (.nvfx), catalogue (.nvfxl), script (.nvfxs), licence (licence or credit text) or file; METHOD is stored
// or cm (the lossless context-mixing coder, include/neuralfx/cm.hpp, either of its formats). SHA256 is that of the
// member's bytes, STORED_SHA256 that of its bytes in the pack (the same when stored): a reader checks the stored bytes
// before decoding them and the decoded bytes after. Lines a reader does not know are ignored (a later writer may add
// optional records without a new format); `#` starts a comment line. Names are relative paths without spaces.
#pragma once

#include "sha256.hpp"

#include <neuralfx/nvfx.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace nfx::pack {

inline constexpr int kFormat = 1;
inline constexpr std::size_t kHeaderBytes = 96;
inline constexpr std::size_t kMaxManifestBytes = std::size_t{16} << 20;

enum class Kind : std::uint8_t { model = 0, catalogue = 1, script = 2, licence = 3, file = 4 };
enum class Method : std::uint8_t { stored = 0, cm = 1 };

const char* kind_name(Kind k);
std::optional<Kind> kind_from(std::string_view s);
// The kind a file's extension suggests (.nvfx model, .nvfxl catalogue, .nvfxs script, else file).
Kind kind_of_file(std::string_view name);

// An error with the status the C API returns and, for the manifest, its line (0: not about a line).
class Error : public std::runtime_error {
 public:
  Error(nvfx_status s, const std::string& message, int at_line = 0) : std::runtime_error(message), status(s), line(at_line) {}
  nvfx_status status;
  int line;
};

struct Record {
  std::string name;
  Kind kind = Kind::file;
  Method method = Method::stored;
  std::uint64_t bytes = 0, stored_bytes = 0, offset = 0;
  Digest sha256{}, stored_sha256{};
};

struct Need {
  std::string entry;
  std::vector<std::string> models;
};

struct Cost {
  std::string model;
  int size = 0;
  float step_ms = 0.f, render_ms = 0.f;
};

struct Manifest {
  int format = kFormat;
  std::vector<Record> members;
  std::vector<Need> needs;
  std::vector<Cost> costs;
  int find(std::string_view name) const;  // index of the member, or -1
};

std::string write_manifest(const Manifest& m);
// Throws Error (NVFX_ERROR_FORMAT with the line, NVFX_ERROR_UNSUPPORTED for a newer format).
Manifest parse_manifest(std::string_view text);
// A member name the writer accepts: a relative path of printable characters without spaces, backslashes, '.' or '..'
// parts, at most 255 bytes.
bool valid_name(std::string_view name);

// --- writing ----------------------------------------------------------------------------------------------------------

// How members are coded: stored, or by the context-mixing coder's full model (format 1), its light model with LZ
// tokens, or its fast model with LZ tokens (format 2). A member is coded only when that pays (PackOptions).
enum class Coder { store, full, light, fast };
const char* coder_name(Coder c);
std::optional<Coder> coder_from(std::string_view s);

struct Input {
  std::string name;
  Kind kind = Kind::file;
  std::vector<std::uint8_t> data;
};

struct PackOptions {
  Coder coder = Coder::fast;          // the default by measurement (docs/INTEGRATION.md §5)
  double min_saving = 0.05;          // coded only if it saves at least this fraction of the member ...
  std::size_t min_saved_bytes = 256; // ... and at least this many bytes
  int threads = 1;                   // members coded at once
};

struct Written {
  std::vector<std::uint8_t> file;
  Manifest manifest;
  double encode_seconds = 0;  // summed over members
};

// Codes the members (each round trip checked), checks the records (names, needs and costs refer to model members)
// and writes the pack. Members stay in the order given. Throws Error.
Written write(const std::vector<Input>& members, const std::vector<Need>& needs, const std::vector<Cost>& costs, const PackOptions& o);

// --- reading ----------------------------------------------------------------------------------------------------------

struct Member {
  Record rec;
  std::vector<std::uint8_t> buf;  // its bytes and a NUL after them (text members are C strings)
  std::span<const std::uint8_t> bytes() const { return {buf.data(), buf.size() - 1}; }
  std::string_view text() const { return {reinterpret_cast<const char*>(buf.data()), buf.size() - 1}; }
};

struct Pack {
  Manifest manifest;
  std::string manifest_text;
  std::vector<Member> members;  // in the manifest's order
  std::size_t file_bytes = 0;
  // A member by its name, or a model by its name without the folder and extension ("fire" for "fire.nvfx"); -1: none.
  int find(std::string_view name) const;
};

struct ReadOptions {
  int threads = 1;  // members decoded at once (the calling thread is one)
};

// Checks the header, the size, the manifest's checksum and records, every member's stored and decoded checksums, and
// that the members every entry needs are models of the pack; decodes every member. Throws Error with a message that
// names what is wrong.
Pack read(std::span<const std::uint8_t> file, const ReadOptions& o = {});

}  // namespace nfx::pack
