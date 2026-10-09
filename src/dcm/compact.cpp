// Inference-only released mixers (include/neuralfx/dcm/compact.hpp). CompactMixer is ported from the owner's
// CameraDetector, cabinlab/src/diffusion/compact.cpp; CompactValueNet and the flat binary form are new here.
#include <neuralfx/dcm/compact.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <format>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace nfx::dcm {

namespace {

constexpr std::string_view kMagic = "NVFXDCM1";
constexpr std::uint8_t kKindPaq = 1, kKindValue = 2;
constexpr int kKnots = AVM::kKnots;

std::vector<double> parse_values(const std::string& line) {
  std::vector<double> out;
  const char* p = line.c_str();
  while (*p != '\0') {
    char* end = nullptr;
    const double v = std::strtod(p, &end);
    if (end == p) throw std::runtime_error("dcm: compact mixer: bad value line");
    out.push_back(v);
    p = end;
    if (*p == ',') ++p;
  }
  return out;
}

// Little-endian writer and bounds-checked reader of the flat binary form.
struct Writer {
  std::vector<std::uint8_t> out;
  void u8(std::uint8_t v) { out.push_back(v); }
  void u16(std::uint16_t v) {
    for (int i = 0; i < 2; ++i) out.push_back(static_cast<std::uint8_t>((v >> (8 * i)) & 0xffu));
  }
  void u32(std::uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<std::uint8_t>((v >> (8 * i)) & 0xffu));
  }
  void u64(std::uint64_t v) {
    for (int i = 0; i < 8; ++i) out.push_back(static_cast<std::uint8_t>((v >> (8 * i)) & 0xffu));
  }
  void f64(double v) { u64(std::bit_cast<std::uint64_t>(v)); }
  template <class Real>
  void real(Real v) {
    if constexpr (std::is_same_v<Real, float>) {
      u32(std::bit_cast<std::uint32_t>(v));
    } else {
      u64(std::bit_cast<std::uint64_t>(v));
    }
  }
  void text(std::string_view s) { out.insert(out.end(), s.begin(), s.end()); }
  void count(std::size_t n) {
    if (n > 0xffffffffu) throw std::runtime_error("dcm: compact form: count too large");
    u32(static_cast<std::uint32_t>(n));
  }
};

struct Reader {
  std::span<const std::uint8_t> b;
  std::size_t at = 0;
  void need(std::size_t n) const {
    if (b.size() - at < n) throw std::runtime_error("dcm: compact form: truncated");
  }
  std::uint64_t le(int n) {
    need(static_cast<std::size_t>(n));
    std::uint64_t v = 0;
    for (int i = 0; i < n; ++i) v |= std::uint64_t{b[at + static_cast<std::size_t>(i)]} << (8 * i);
    at += static_cast<std::size_t>(n);
    return v;
  }
  std::uint8_t u8() { return static_cast<std::uint8_t>(le(1)); }
  std::uint16_t u16() { return static_cast<std::uint16_t>(le(2)); }
  std::uint32_t u32() { return static_cast<std::uint32_t>(le(4)); }
  double f64() { return std::bit_cast<double>(le(8)); }
  template <class Real>
  Real real() {
    if constexpr (std::is_same_v<Real, float>) {
      return std::bit_cast<float>(static_cast<std::uint32_t>(le(4)));
    } else {
      return std::bit_cast<double>(le(8));
    }
  }
  std::string text(std::size_t n) {
    need(n);
    std::string s(reinterpret_cast<const char*>(b.data() + at), n);
    at += n;
    return s;
  }
  // A count of Real values that must still fit in the remaining bytes (refuses absurd sizes before allocating).
  template <class Real>
  std::vector<Real> reals(std::size_t n) {
    if (n > (b.size() - at) / sizeof(Real)) throw std::runtime_error("dcm: compact form: truncated");
    std::vector<Real> v(n);
    for (auto& x : v) x = real<Real>();
    return v;
  }
};

bool is_version(std::string_view v) {
  return v.size() == 64 && std::ranges::all_of(v, [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}

template <class Real>
void write_header(Writer& w, std::uint8_t kind, const std::string& version) {
  w.text(kMagic);
  w.u8(kind);
  w.u8(static_cast<std::uint8_t>(sizeof(Real)));
  w.u16(0);
  w.text(version);
}

template <class Real>
std::string read_header(Reader& r, std::uint8_t kind) {
  if (r.text(kMagic.size()) != kMagic) throw std::runtime_error("dcm: compact form: not an NVFXDCM1 block");
  const auto k = r.u8();
  const auto size = r.u8();
  (void)r.u16();
  if (k != kind) throw std::runtime_error(std::format("dcm: compact form: kind {} where {} was expected", k, kind));
  if (size != sizeof(Real)) throw std::runtime_error(std::format("dcm: compact form: {}-byte weights, not {}", size, sizeof(Real)));
  std::string version = r.text(64);
  if (!is_version(version)) throw std::runtime_error("dcm: compact form: bad version");
  return version;
}

template <class Layer>
void write_layer(Writer& w, const Layer& l) {
  w.count(static_cast<std::size_t>(l.n));
  w.count(static_cast<std::size_t>(l.k));
  for (const auto v : l.w) w.real(v);
}

template <class Real, class Layer>
Layer read_layer(Reader& r) {
  Layer l;
  const auto n = r.u32(), k = r.u32();
  if (n == 0 || k == 0 || n > 0x7fffffffu || k > 0x7fffffffu) throw std::runtime_error("dcm: compact form: bad mixer shape");
  l.n = static_cast<int>(n);
  l.k = static_cast<int>(k);
  if (static_cast<std::uint64_t>(n) * k > (r.b.size() - r.at) / sizeof(Real)) throw std::runtime_error("dcm: compact form: truncated");
  l.w = r.reals<Real>(static_cast<std::size_t>(n) * k);
  return l;
}

// The weight set of `ctx` in a layer, or std::out_of_range.
template <class Layer>
auto weights_of(const Layer& l, int ctx) {
  if (ctx < 0 || ctx >= l.k) throw std::out_of_range(std::format("dcm: mixer context {} of {}", ctx, l.k));
  return l.w.data() + static_cast<std::size_t>(ctx) * static_cast<std::size_t>(l.n);
}

// One "mixer n k" block (header line already read) of a text serialisation.
template <class Layer>
Layer parse_layer(std::istringstream& is, const std::string& header) {
  Layer l;
  std::istringstream h(header);
  h >> l.n >> l.k;
  std::string values;
  if (!h || l.n <= 0 || l.k <= 0 || !std::getline(is, values)) throw std::runtime_error("dcm: compact mixer: bad mixer");
  const auto v = parse_values(values);
  if (v.size() != static_cast<std::size_t>(l.n) * static_cast<std::size_t>(l.k)) {
    throw std::runtime_error("dcm: compact mixer: weight count");
  }
  using Real = typename decltype(l.w)::value_type;
  l.w.resize(v.size());
  std::ranges::transform(v, l.w.begin(), [](double d) { return static_cast<Real>(d); });
  return l;
}

}  // namespace

// --- CompactMixer ---------------------------------------------------------------------------------------------------

template <class Real>
CompactMixer<Real>::CompactMixer(std::string_view serialised) : version_(sha256_hex(serialised)) {
  std::istringstream is{std::string(serialised)};
  std::string line;
  if (!std::getline(is, line) || line != "nvfx-paq-mixer v1") {
    throw std::runtime_error("dcm: compact mixer: not an nvfx-paq-mixer v1 serialisation");
  }
  std::vector<Layer> mixers;
  while (std::getline(is, line)) {
    if (line.starts_with("mixer ")) {
      mixers.push_back(parse_layer<Layer>(is, line.substr(6)));
    } else if (line.starts_with("apm_weight ")) {
      apm_weight_ = std::stod(line.substr(11));
    } else if (line.starts_with("apm ")) {
      apm_k_ = std::stoi(line.substr(4));
      std::string values;
      if (apm_k_ <= 0 || !std::getline(is, values)) throw std::runtime_error("dcm: compact mixer: bad apm");
      const auto v = parse_values(values);
      if (v.size() != static_cast<std::size_t>(apm_k_) * 33) throw std::runtime_error("dcm: compact mixer: apm size");
      apm_.resize(v.size());
      std::ranges::transform(v, apm_.begin(), [](double d) { return static_cast<Real>(d); });
    } else if (!line.empty()) {
      throw std::runtime_error(std::format("dcm: compact mixer: unexpected line '{}'", line.substr(0, 40)));
    }
  }
  if (mixers.size() < 2) throw std::runtime_error("dcm: compact mixer: needs first-layer mixers and a final mixer");
  final_ = std::move(mixers.back());
  mixers.pop_back();
  layer1_ = std::move(mixers);
  n_ = layer1_.front().n;
  if (apm_k_ == 0) apm_weight_ = 0.0;
  check();
}

template <class Real>
void CompactMixer<Real>::check() const {
  if (layer1_.empty()) throw std::runtime_error("dcm: compact mixer: needs first-layer mixers and a final mixer");
  for (const auto& l : layer1_) {
    if (l.n != n_) throw std::runtime_error("dcm: compact mixer: first-layer mixers differ in inputs");
  }
  if (final_.n != static_cast<int>(layer1_.size())) throw std::runtime_error("dcm: compact mixer: final mixer inputs");
  if (apm_.size() != static_cast<std::size_t>(apm_k_) * 33) throw std::runtime_error("dcm: compact mixer: apm size");
}

template <class Real>
std::vector<std::uint8_t> CompactMixer<Real>::to_bytes() const {
  Writer w;
  write_header<Real>(w, kKindPaq, version_);
  w.count(layer1_.size());
  for (const auto& l : layer1_) write_layer(w, l);
  write_layer(w, final_);
  w.f64(apm_weight_);
  w.count(static_cast<std::size_t>(apm_k_));
  for (const auto v : apm_) w.real(v);
  return std::move(w.out);
}

template <class Real>
CompactMixer<Real> CompactMixer<Real>::from_bytes(std::span<const std::uint8_t> bytes) {
  Reader r{bytes};
  CompactMixer m;
  m.version_ = read_header<Real>(r, kKindPaq);
  const auto layers = r.u32();
  if (layers == 0 || layers > bytes.size()) throw std::runtime_error("dcm: compact form: bad layer count");
  for (std::uint32_t i = 0; i < layers; ++i) m.layer1_.push_back(read_layer<Real, Layer>(r));
  m.final_ = read_layer<Real, Layer>(r);
  m.apm_weight_ = r.f64();
  const auto apm_k = r.u32();
  if (apm_k > 0x7fffffffu) throw std::runtime_error("dcm: compact form: bad apm");
  m.apm_k_ = static_cast<int>(apm_k);
  m.apm_ = r.reals<Real>(static_cast<std::size_t>(apm_k) * 33);
  if (r.at != bytes.size()) throw std::runtime_error("dcm: compact form: trailing bytes");
  m.n_ = m.layer1_.front().n;
  if (!(m.apm_weight_ >= 0.0 && m.apm_weight_ <= 1.0)) throw std::runtime_error("dcm: compact form: bad apm weight");
  m.check();
  return m;
}

template <class Real>
double CompactMixer<Real>::predict(std::span<const double> x, std::span<const int> contexts) const {
  const std::size_t m = layer1_.size();
  if (x.size() != static_cast<std::size_t>(n_) || contexts.size() != m + 2) {
    throw std::invalid_argument("dcm: compact mixer: input or context count");
  }
  Real h[64];
  std::vector<Real> heap;
  Real* hp = h;
  if (m > 64) {
    heap.resize(m);
    hp = heap.data();
  }
  for (std::size_t i = 0; i < m; ++i) {
    const Real* w = weights_of(layer1_[i], contexts[i]);
    Real dot = 0;
    for (int j = 0; j < n_; ++j) dot += w[j] * static_cast<Real>(x[static_cast<std::size_t>(j)]);
    hp[i] = std::clamp(dot, static_cast<Real>(-16), static_cast<Real>(16));
  }
  const Real* wf = weights_of(final_, contexts[m]);
  Real z = 0;
  for (std::size_t i = 0; i < m; ++i) z += wf[i] * hp[i];
  const double p = squash(static_cast<double>(z));
  if (apm_weight_ <= 0.0) return p;
  const int ctx = std::clamp(contexts[m + 1], 0, apm_k_ - 1);
  const double s = std::clamp(stretch(p), -7.999, 7.999);
  const double pos = (s + 8.0) * 2.0;
  const auto j = static_cast<std::size_t>(pos);
  const double frac = pos - static_cast<double>(j);
  const std::size_t lo = static_cast<std::size_t>(ctx) * 33 + j;
  const double pa = static_cast<double>(apm_[lo]) * (1.0 - frac) + static_cast<double>(apm_[lo + 1]) * frac;
  return (1.0 - apm_weight_) * p + apm_weight_ * pa;
}

template <class Real>
std::size_t CompactMixer<Real>::values() const noexcept {
  std::size_t n = final_.w.size() + apm_.size();
  for (const auto& l : layer1_) n += l.w.size();
  return n;
}

// --- CompactValueNet ------------------------------------------------------------------------------------------------

template <class Real>
CompactValueNet<Real>::CompactValueNet(std::string_view serialised) : version_(sha256_hex(serialised)) {
  std::istringstream is{std::string(serialised)};
  std::string line;
  if (!std::getline(is, line) || line != "nvfx-value-mixer v1") {
    throw std::runtime_error("dcm: compact value net: not an nvfx-value-mixer v1 serialisation");
  }
  std::vector<Layer> mixers;
  bool have_limit = false, have_scale = false;
  while (std::getline(is, line)) {
    if (line.starts_with("limit ")) {
      limit_ = std::stod(line.substr(6));
      have_limit = true;
    } else if (line.starts_with("mixer ")) {
      mixers.push_back(parse_layer<Layer>(is, line.substr(6)));
    } else if (line.starts_with("avm_weight ")) {
      avm_weight_ = std::stod(line.substr(11));
    } else if (line.starts_with("avm ")) {
      std::istringstream h(line.substr(4));
      h >> avm_k_ >> avm_lo_ >> avm_hi_;
      std::string values;
      if (!h || avm_k_ <= 0 || !std::getline(is, values)) throw std::runtime_error("dcm: compact value net: bad avm");
      const auto v = parse_values(values);
      if (v.size() != static_cast<std::size_t>(avm_k_) * kKnots) throw std::runtime_error("dcm: compact value net: avm size");
      avm_.resize(v.size());
      std::ranges::transform(v, avm_.begin(), [](double d) { return static_cast<Real>(d); });
    } else if (line.starts_with("scale ")) {
      std::istringstream h(line.substr(6));
      h >> scale_.n >> scale_.k >> log_b_min_ >> log_b_max_;
      if (!h) throw std::runtime_error("dcm: compact value net: bad scale net");
      scale_ = parse_layer<Layer>(is, std::format("{} {}", scale_.n, scale_.k));
      have_scale = true;
    } else if (!line.empty()) {
      throw std::runtime_error(std::format("dcm: compact value net: unexpected line '{}'", line.substr(0, 40)));
    }
  }
  if (!have_limit || !have_scale) throw std::runtime_error("dcm: compact value net: needs a limit and a scale net");
  if (mixers.size() < 2) throw std::runtime_error("dcm: compact value net: needs first-layer mixers and a final mixer");
  final_ = std::move(mixers.back());
  mixers.pop_back();
  layer1_ = std::move(mixers);
  n_ = layer1_.front().n;
  if (avm_k_ == 0) avm_weight_ = 0.0;
  check();
}

template <class Real>
void CompactValueNet<Real>::check() const {
  if (layer1_.empty()) throw std::runtime_error("dcm: compact value net: needs first-layer mixers and a final mixer");
  for (const auto& l : layer1_) {
    if (l.n != n_) throw std::runtime_error("dcm: compact value net: first-layer mixers differ in inputs");
  }
  if (final_.n != static_cast<int>(layer1_.size())) throw std::runtime_error("dcm: compact value net: final mixer inputs");
  if (avm_.size() != static_cast<std::size_t>(avm_k_) * kKnots) throw std::runtime_error("dcm: compact value net: avm size");
  if (avm_k_ > 0 && !(avm_hi_ > avm_lo_)) throw std::runtime_error("dcm: compact value net: avm range");
  if (!(limit_ > 0.0)) throw std::runtime_error("dcm: compact value net: limit");
  if (scale_.n < 1 || scale_.k < 1 || !(log_b_min_ < log_b_max_)) throw std::runtime_error("dcm: compact value net: scale net");
}

template <class Real>
std::vector<std::uint8_t> CompactValueNet<Real>::to_bytes() const {
  Writer w;
  write_header<Real>(w, kKindValue, version_);
  w.f64(limit_);
  w.count(layer1_.size());
  for (const auto& l : layer1_) write_layer(w, l);
  write_layer(w, final_);
  w.f64(avm_weight_);
  w.count(static_cast<std::size_t>(avm_k_));
  w.f64(avm_lo_);
  w.f64(avm_hi_);
  for (const auto v : avm_) w.real(v);
  w.count(static_cast<std::size_t>(scale_.n));
  w.count(static_cast<std::size_t>(scale_.k));
  w.f64(log_b_min_);
  w.f64(log_b_max_);
  for (const auto v : scale_.w) w.real(v);
  return std::move(w.out);
}

template <class Real>
CompactValueNet<Real> CompactValueNet<Real>::from_bytes(std::span<const std::uint8_t> bytes) {
  Reader r{bytes};
  CompactValueNet m;
  m.version_ = read_header<Real>(r, kKindValue);
  m.limit_ = r.f64();
  const auto layers = r.u32();
  if (layers == 0 || layers > bytes.size()) throw std::runtime_error("dcm: compact form: bad layer count");
  for (std::uint32_t i = 0; i < layers; ++i) m.layer1_.push_back(read_layer<Real, Layer>(r));
  m.final_ = read_layer<Real, Layer>(r);
  m.avm_weight_ = r.f64();
  const auto avm_k = r.u32();
  if (avm_k > 0x7fffffffu) throw std::runtime_error("dcm: compact form: bad avm");
  m.avm_k_ = static_cast<int>(avm_k);
  m.avm_lo_ = r.f64();
  m.avm_hi_ = r.f64();
  m.avm_ = r.reals<Real>(static_cast<std::size_t>(avm_k) * kKnots);
  // the scale net: the same layout as a mixer with its clamp range between the shape and the weights
  const auto n = r.u32(), k = r.u32();
  if (n == 0 || k == 0 || n > 0x7fffffffu || k > 0x7fffffffu) throw std::runtime_error("dcm: compact form: bad scale net");
  m.scale_.n = static_cast<int>(n);
  m.scale_.k = static_cast<int>(k);
  m.log_b_min_ = r.f64();
  m.log_b_max_ = r.f64();
  if (static_cast<std::uint64_t>(n) * k > (bytes.size() - r.at) / sizeof(Real)) throw std::runtime_error("dcm: compact form: truncated");
  m.scale_.w = r.reals<Real>(static_cast<std::size_t>(n) * k);
  if (r.at != bytes.size()) throw std::runtime_error("dcm: compact form: trailing bytes");
  m.n_ = m.layer1_.front().n;
  if (!(m.avm_weight_ >= 0.0 && m.avm_weight_ <= 1.0)) throw std::runtime_error("dcm: compact form: bad avm weight");
  m.check();
  return m;
}

template <class Real>
ValuePrediction CompactValueNet<Real>::predict(std::span<const double> x, std::span<const int> contexts,
                                               std::span<const double> z) const {
  const std::size_t m = layer1_.size();
  if (x.size() != static_cast<std::size_t>(n_) || contexts.size() != m + 3 ||
      z.size() != static_cast<std::size_t>(scale_.n - 1)) {
    throw std::invalid_argument("dcm: compact value net: input, context or scale-feature count");
  }
  Real h[64];
  std::vector<Real> heap;
  Real* hp = h;
  if (m > 64) {
    heap.resize(m);
    hp = heap.data();
  }
  const auto lim = static_cast<Real>(limit_);
  for (std::size_t i = 0; i < m; ++i) {
    const Real* w = weights_of(layer1_[i], contexts[i]);
    Real dot = 0;
    for (int j = 0; j < n_; ++j) dot += w[j] * static_cast<Real>(x[static_cast<std::size_t>(j)]);
    hp[i] = std::clamp(dot, -lim, lim);
  }
  const Real* wf = weights_of(final_, contexts[m]);
  Real zf = 0;
  for (std::size_t i = 0; i < m; ++i) zf += wf[i] * hp[i];
  const double mu_mix = std::clamp(static_cast<double>(zf), -limit_, limit_);
  double mu = mu_mix;
  if (avm_weight_ > 0.0) {  // as AVM::offset
    const int ctx = std::clamp(contexts[m + 1], 0, avm_k_ - 1);
    double pos = (mu_mix - avm_lo_) / (avm_hi_ - avm_lo_) * static_cast<double>(kKnots - 1);
    pos = std::isnan(pos) ? 0.0 : std::clamp(pos, 0.0, static_cast<double>(kKnots - 1) - 0.001);
    const auto j = static_cast<std::size_t>(pos);
    const double frac = pos - static_cast<double>(j);
    const std::size_t at = static_cast<std::size_t>(ctx) * kKnots + j;
    const double offset = static_cast<double>(avm_[at]) * (1.0 - frac) + static_cast<double>(avm_[at + 1]) * frac;
    mu = mu_mix + avm_weight_ * offset;
  }
  const Real* ws = weights_of(scale_, contexts[m + 2]);
  Real s = 0;
  s += ws[0] * static_cast<Real>(1);
  for (std::size_t j = 0; j < z.size(); ++j) s += ws[j + 1] * static_cast<Real>(z[j]);
  const double log_b = std::clamp(static_cast<double>(s), log_b_min_, log_b_max_);
  return {mu, std::exp(log_b)};
}

template <class Real>
std::size_t CompactValueNet<Real>::values() const noexcept {
  std::size_t n = final_.w.size() + avm_.size() + scale_.w.size();
  for (const auto& l : layer1_) n += l.w.size();
  return n;
}

template class CompactMixer<float>;
template class CompactMixer<double>;
template class CompactValueNet<float>;
template class CompactValueNet<double>;

}  // namespace nfx::dcm
