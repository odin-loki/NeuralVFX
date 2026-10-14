// The effect pack: its manifest, the writer and the reader (pack.hpp).
#include "pack.hpp"

#include <neuralfx/cm.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <charconv>
#include <cmath>
#include <cstring>
#include <exception>
#include <format>
#include <thread>

namespace nfx::pack {

namespace {

constexpr char kMagic[8] = {'N', 'V', 'F', 'X', 'P', 'A', 'C', 'K'};

void put32(std::uint8_t* p, std::uint32_t v) {
  for (int i = 0; i < 4; ++i) p[i] = static_cast<std::uint8_t>(v >> (8 * i));
}
void put64(std::uint8_t* p, std::uint64_t v) {
  for (int i = 0; i < 8; ++i) p[i] = static_cast<std::uint8_t>(v >> (8 * i));
}
std::uint32_t get32(const std::uint8_t* p) {
  std::uint32_t v = 0;
  for (int i = 0; i < 4; ++i) v |= std::uint32_t{p[i]} << (8 * i);
  return v;
}
std::uint64_t get64(const std::uint8_t* p) {
  std::uint64_t v = 0;
  for (int i = 0; i < 8; ++i) v |= std::uint64_t{p[i]} << (8 * i);
  return v;
}

const char* method_name(Method m) { return m == Method::cm ? "cm" : "stored"; }

std::vector<std::string_view> words(std::string_view line) {
  std::vector<std::string_view> w;
  std::size_t i = 0;
  while (i < line.size()) {
    while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) ++i;
    std::size_t j = i;
    while (j < line.size() && line[j] != ' ' && line[j] != '\t') ++j;
    if (j > i) w.push_back(line.substr(i, j - i));
    i = j;
  }
  return w;
}

template <class T>
bool number(std::string_view s, T& out) {
  const auto r = std::from_chars(s.data(), s.data() + s.size(), out);
  return r.ec == std::errc() && r.ptr == s.data() + s.size();
}

// Runs fn(i) for i in [0, n) on up to `threads` threads (the caller's included); the first failure by index is rethrown.
template <class F>
void parallel(std::size_t n, int threads, F&& fn) {
  std::vector<std::exception_ptr> failed(n);
  std::atomic<std::size_t> next{0};
  const auto work = [&] {
    for (std::size_t i; (i = next.fetch_add(1)) < n;) {
      try {
        fn(i);
      } catch (...) {
        failed[i] = std::current_exception();
      }
    }
  };
  const std::size_t extra = std::min<std::size_t>(n > 0 ? n - 1 : 0, static_cast<std::size_t>(std::max(1, threads) - 1));
  std::vector<std::jthread> pool;
  pool.reserve(extra);
  for (std::size_t t = 0; t < extra; ++t) pool.emplace_back(work);
  work();
  pool.clear();
  for (const auto& e : failed)
    if (e) std::rethrow_exception(e);
}

std::string short_name(std::string_view name) {  // "fx/fire.nvfx" -> "fire"
  const std::size_t slash = name.rfind('/');
  std::string_view base = slash == std::string_view::npos ? name : name.substr(slash + 1);
  const std::size_t dot = base.rfind('.');
  return std::string(dot == std::string_view::npos || dot == 0 ? base : base.substr(0, dot));
}

}  // namespace

const char* kind_name(Kind k) {
  switch (k) {
    case Kind::model: return "model";
    case Kind::catalogue: return "catalogue";
    case Kind::script: return "script";
    case Kind::licence: return "licence";
    case Kind::file: return "file";
  }
  return "file";
}

std::optional<Kind> kind_from(std::string_view s) {
  for (const Kind k : {Kind::model, Kind::catalogue, Kind::script, Kind::licence, Kind::file})
    if (s == kind_name(k)) return k;
  return std::nullopt;
}

Kind kind_of_file(std::string_view name) {
  if (name.ends_with(".nvfx")) return Kind::model;
  if (name.ends_with(".nvfxl")) return Kind::catalogue;
  if (name.ends_with(".nvfxs")) return Kind::script;
  return Kind::file;
}

const char* coder_name(Coder c) {
  switch (c) {
    case Coder::store: return "store";
    case Coder::full: return "full";
    case Coder::light: return "light";
    case Coder::fast: return "fast";
  }
  return "store";
}

std::optional<Coder> coder_from(std::string_view s) {
  for (const Coder c : {Coder::store, Coder::full, Coder::light, Coder::fast})
    if (s == coder_name(c)) return c;
  return std::nullopt;
}

bool valid_name(std::string_view name) {
  if (name.empty() || name.size() > 255 || name.front() == '/') return false;
  for (const char c : name) {
    const auto u = static_cast<unsigned char>(c);
    if (u <= 0x20 || u == 0x7f || c == '\\' || c == ':') return false;
  }
  std::size_t i = 0;
  while (i <= name.size()) {
    const std::size_t j = std::min(name.find('/', i), name.size());
    const std::string_view part = name.substr(i, j - i);
    if (part.empty() || part == "." || part == "..") return false;
    i = j + 1;
  }
  return true;
}

int Manifest::find(std::string_view name) const {
  for (std::size_t i = 0; i < members.size(); ++i)
    if (members[i].name == name) return static_cast<int>(i);
  return -1;
}

int Pack::find(std::string_view name) const {
  if (const int i = manifest.find(name); i >= 0) return i;
  for (std::size_t i = 0; i < members.size(); ++i)
    if (members[i].rec.kind == Kind::model && short_name(members[i].rec.name) == name) return static_cast<int>(i);
  return -1;
}

std::string write_manifest(const Manifest& m) {
  std::string s = std::format("nvfx-pack {}\n", m.format);
  for (const Record& r : m.members)
    s += std::format("member {} {} {} {} {} {} {} {}\n", r.name, kind_name(r.kind), method_name(r.method), r.bytes, r.stored_bytes, r.offset, hex(r.sha256),
                     hex(r.stored_sha256));
  for (const Need& n : m.needs) {
    s += "needs " + n.entry;
    for (const std::string& f : n.models) s += " " + f;
    s += "\n";
  }
  for (const Cost& c : m.costs) s += std::format("cost {} {} {} {}\n", c.model, c.size, c.step_ms, c.render_ms);
  return s;
}

Manifest parse_manifest(std::string_view text) {
  Manifest m;
  bool seen_format = false;
  int line_no = 0;
  std::size_t at = 0;
  while (at < text.size()) {
    const std::size_t end = std::min(text.find('\n', at), text.size());
    std::string_view line = text.substr(at, end - at);
    at = end + 1;
    ++line_no;
    if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
    const auto w = words(line);
    if (w.empty() || w[0].front() == '#') continue;
    const auto bad = [&](const std::string& what) { return Error(NVFX_ERROR_FORMAT, std::format("manifest line {}: {}", line_no, what), line_no); };
    if (!seen_format) {
      int f = 0;
      if (w[0] != "nvfx-pack" || w.size() < 2 || !number(w[1], f)) throw bad("expected 'nvfx-pack FORMAT' first");
      if (f < 1) throw bad(std::format("format {} is not a pack format", f));
      if (f > kFormat) throw Error(NVFX_ERROR_UNSUPPORTED, std::format("pack format {} is newer than this library reads ({})", f, kFormat), line_no);
      m.format = f;
      seen_format = true;
      continue;
    }
    if (w[0] == "member") {
      if (w.size() != 9) throw bad("a member takes NAME KIND METHOD BYTES STORED OFFSET SHA256 STORED_SHA256");
      Record r;
      r.name = std::string(w[1]);
      if (!valid_name(r.name)) throw bad(std::format("'{}' is not a member name", r.name));
      r.kind = kind_from(w[2]).value_or(Kind::file);
      if (w[3] == "stored") r.method = Method::stored;
      else if (w[3] == "cm") r.method = Method::cm;
      else throw Error(NVFX_ERROR_UNSUPPORTED, std::format("manifest line {}: member {} is coded by '{}', which this library does not know", line_no, r.name, w[3]), line_no);
      if (!number(w[4], r.bytes) || !number(w[5], r.stored_bytes) || !number(w[6], r.offset)) throw bad("sizes and offsets are whole numbers");
      if (!parse_hex(w[7], r.sha256) || !parse_hex(w[8], r.stored_sha256)) throw bad("a SHA-256 is 64 hex digits");
      if (r.method == Method::stored && (r.bytes != r.stored_bytes || r.sha256 != r.stored_sha256)) throw bad(std::format("stored member {} has two sizes or checksums", r.name));
      if (m.find(r.name) >= 0) throw bad(std::format("member {} is listed twice", r.name));
      m.members.push_back(std::move(r));
    } else if (w[0] == "needs") {
      if (w.size() < 2) throw bad("needs takes an entry and its models");
      Need n{std::string(w[1]), {}};
      for (std::size_t i = 2; i < w.size(); ++i) n.models.emplace_back(w[i]);
      m.needs.push_back(std::move(n));
    } else if (w[0] == "cost") {
      Cost c;
      if (w.size() != 5 || !number(w[2], c.size) || !number(w[3], c.step_ms) || !number(w[4], c.render_ms)) throw bad("cost takes MODEL SIZE STEP_MS RENDER_MS");
      c.model = std::string(w[1]);
      m.costs.push_back(std::move(c));
    }
    // other records: a later writer's optional ones, ignored
  }
  if (!seen_format) throw Error(NVFX_ERROR_FORMAT, "the manifest is empty");
  return m;
}

namespace {

void check_records(const Manifest& m) {
  for (const Need& n : m.needs)
    for (const std::string& f : n.models) {
      const int i = m.find(f);
      if (i < 0 || m.members[static_cast<std::size_t>(i)].kind != Kind::model)
        throw Error(NVFX_ERROR_FORMAT, std::format("entry '{}' needs {}, which is not a model of the pack", n.entry, f));
    }
  for (const Cost& c : m.costs) {
    const int i = m.find(c.model);
    if (i < 0 || m.members[static_cast<std::size_t>(i)].kind != Kind::model)
      throw Error(NVFX_ERROR_FORMAT, std::format("a cost names {}, which is not a model of the pack", c.model));
    if (c.size != 32 && c.size != 64 && c.size != 128 && c.size != 256) throw Error(NVFX_ERROR_FORMAT, std::format("a cost of {} is for size {} (32, 64, 128 or 256)", c.model, c.size));
    if (!std::isfinite(c.step_ms) || !std::isfinite(c.render_ms) || c.step_ms < 0.f || c.render_ms < 0.f)
      throw Error(NVFX_ERROR_FORMAT, std::format("a cost of {} is not a number of milliseconds", c.model));
  }
}

}  // namespace

Written write(const std::vector<Input>& in, const std::vector<Need>& needs, const std::vector<Cost>& costs, const PackOptions& o) {
  Written w;
  Manifest& m = w.manifest;
  for (const Input& x : in) {
    if (!valid_name(x.name)) throw Error(NVFX_ERROR_ARGUMENT, std::format("'{}' cannot be a member's name (a relative path without spaces)", x.name));
    if (m.find(x.name) >= 0) throw Error(NVFX_ERROR_ARGUMENT, std::format("two members are named {}", x.name));
    Record r;
    r.name = x.name;
    r.kind = x.kind;
    r.bytes = x.data.size();
    m.members.push_back(std::move(r));
  }
  m.needs = needs;
  m.costs = costs;
  check_records(m);
  for (const Need& n : needs)
    if (n.entry.empty() || !valid_name(n.entry) || n.entry.find('/') != std::string::npos) throw Error(NVFX_ERROR_ARGUMENT, std::format("'{}' cannot be an entry's name", n.entry));
  // code the members
  std::vector<std::vector<std::uint8_t>> stored(in.size());
  std::vector<double> seconds(in.size(), 0.0);
  parallel(in.size(), o.threads, [&](std::size_t i) {
    const std::vector<std::uint8_t>& data = in[i].data;
    Record& r = m.members[i];
    r.sha256 = sha256(data);
    r.method = Method::stored;
    if (o.coder != Coder::store && !data.empty()) {
      const auto t0 = std::chrono::steady_clock::now();
      cm::Packed p;
      if (o.coder == Coder::full) p = cm::pack_model(data);
      else p = cm::pack_model(data, cm::Options{o.coder == Coder::fast ? cm::Literal::fast : cm::Literal::light, true, false, std::size_t{1} << 16});
      seconds[i] = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      const std::size_t need = std::max(o.min_saved_bytes, static_cast<std::size_t>(std::ceil(o.min_saving * static_cast<double>(data.size()))));
      if (p.data.size() + need <= data.size()) {
        const auto back = cm::unpack_model(p.data);
        if (!back || *back != data) throw Error(NVFX_ERROR_FORMAT, std::format("the coder did not reproduce {} (kept stored would be safe: --coder store)", r.name));
        r.method = Method::cm;
        stored[i] = std::move(p.data);
      }
    }
    if (r.method == Method::stored) stored[i] = data;
    r.stored_bytes = stored[i].size();
    r.stored_sha256 = r.method == Method::stored ? r.sha256 : sha256(stored[i]);
  });
  for (const double s : seconds) w.encode_seconds += s;
  std::uint64_t data_bytes = 0;
  for (std::size_t i = 0; i < in.size(); ++i) {
    m.members[i].offset = data_bytes;
    data_bytes += (m.members[i].stored_bytes + 7) / 8 * 8;
  }
  const std::string text = write_manifest(m);
  if (text.size() > kMaxManifestBytes) throw Error(NVFX_ERROR_ARGUMENT, "the manifest is too large");
  const std::uint64_t total = kHeaderBytes + text.size() + data_bytes;
  w.file.assign(static_cast<std::size_t>(total), 0);
  std::uint8_t* p = w.file.data();
  std::memcpy(p, kMagic, 8);
  put32(p + 8, static_cast<std::uint32_t>(m.format));
  put32(p + 12, static_cast<std::uint32_t>(kHeaderBytes));
  put64(p + 16, text.size());
  put64(p + 24, data_bytes);
  put64(p + 32, total);
  const Digest md = sha256(std::span(reinterpret_cast<const std::uint8_t*>(text.data()), text.size()));
  std::memcpy(p + 40, md.data(), 32);
  std::memcpy(p + kHeaderBytes, text.data(), text.size());
  std::uint8_t* data = p + kHeaderBytes + text.size();
  for (std::size_t i = 0; i < in.size(); ++i)
    if (!stored[i].empty()) std::memcpy(data + m.members[i].offset, stored[i].data(), stored[i].size());
  return w;
}

Pack read(std::span<const std::uint8_t> file, const ReadOptions& o) {
  if (file.size() < 8 || std::memcmp(file.data(), kMagic, 8) != 0) throw Error(NVFX_ERROR_FORMAT, "not an effect pack (no NVFXPACK header)");
  if (file.size() < kHeaderBytes) throw Error(NVFX_ERROR_FORMAT, std::format("truncated: {} bytes, shorter than a pack's header", file.size()));
  const std::uint8_t* h = file.data();
  const std::uint32_t format = get32(h + 8), header = get32(h + 12);
  if (format > static_cast<std::uint32_t>(kFormat)) throw Error(NVFX_ERROR_UNSUPPORTED, std::format("pack format {} is newer than this library reads ({})", format, kFormat));
  if (format < 1 || header < kHeaderBytes) throw Error(NVFX_ERROR_FORMAT, "damaged header (format or header size)");
  const std::uint64_t manifest_bytes = get64(h + 16), data_bytes = get64(h + 24), total = get64(h + 32);
  if (manifest_bytes > kMaxManifestBytes || data_bytes > (std::uint64_t{1} << 40) || total != header + manifest_bytes + data_bytes)
    throw Error(NVFX_ERROR_FORMAT, "damaged header (its sizes do not add up)");
  if (file.size() != total)
    throw Error(NVFX_ERROR_FORMAT, file.size() < total ? std::format("truncated: {} of its {} bytes", file.size(), total)
                                                        : std::format("{} bytes, longer than the {} its header says", file.size(), total));
  const auto manifest = file.subspan(header, static_cast<std::size_t>(manifest_bytes));
  Digest want;
  std::memcpy(want.data(), h + 40, 32);
  if (sha256(manifest) != want) throw Error(NVFX_ERROR_FORMAT, "damaged manifest (its SHA-256 does not match the header's)");
  Pack pk;
  pk.file_bytes = file.size();
  pk.manifest_text.assign(reinterpret_cast<const char*>(manifest.data()), manifest.size());
  pk.manifest = parse_manifest(pk.manifest_text);
  const Manifest& m = pk.manifest;
  const auto data = file.subspan(static_cast<std::size_t>(header + manifest_bytes));
  for (const Record& r : m.members)
    if (r.offset > data.size() || r.stored_bytes > data.size() - r.offset)
      throw Error(NVFX_ERROR_FORMAT, std::format("member {} lies outside the pack ({} bytes at {} of {})", r.name, r.stored_bytes, r.offset, data.size()));
  check_records(m);
  pk.members.resize(m.members.size());
  parallel(m.members.size(), o.threads, [&](std::size_t i) {
    const Record& r = m.members[i];
    const auto s = data.subspan(static_cast<std::size_t>(r.offset), static_cast<std::size_t>(r.stored_bytes));
    if (sha256(s) != r.stored_sha256) throw Error(NVFX_ERROR_FORMAT, std::format("member {} is damaged (its stored bytes do not match their SHA-256)", r.name));
    Member& out = pk.members[i];
    out.rec = r;
    if (r.method == Method::stored) {
      out.buf.reserve(s.size() + 1);
      out.buf.assign(s.begin(), s.end());
    } else {
      auto u = cm::unpack_model(s);
      if (!u) throw Error(NVFX_ERROR_FORMAT, std::format("member {} does not decode: {}", r.name, u.error()));
      out.buf = std::move(*u);
      if (out.buf.size() != r.bytes || sha256(out.buf) != r.sha256)
        throw Error(NVFX_ERROR_FORMAT, std::format("member {} decodes to other bytes than were packed (size or SHA-256)", r.name));
    }
    out.buf.push_back(0);
  });
  return pk;
}

}  // namespace nfx::pack
