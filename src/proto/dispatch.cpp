#include <neuralfx/proto.hpp>

#include <format>
#include <stdexcept>

namespace nfx::proto {

namespace isa_base { std::unique_ptr<Model> create(const Spec& s); }
namespace isa_avx2 { std::unique_ptr<Model> create(const Spec& s); }
namespace isa_avx512 { std::unique_ptr<Model> create(const Spec& s); }

const char* isa_name(Isa isa) {
  switch (isa) {
    case Isa::base: return "base";
    case Isa::avx2: return "avx2";
    case Isa::avx512: return "avx512";
  }
  return "?";
}

bool isa_supported(Isa isa) {
#if defined(__GNUC__) && defined(__x86_64__)
  __builtin_cpu_init();
  switch (isa) {
    case Isa::base: return true;
    case Isa::avx2: return __builtin_cpu_supports("x86-64-v3");
    case Isa::avx512: return __builtin_cpu_supports("x86-64-v4");
  }
  return false;
#else
  return isa == Isa::base;
#endif
}

Isa best_isa() {
  if (isa_supported(Isa::avx512)) return Isa::avx512;
  if (isa_supported(Isa::avx2)) return Isa::avx2;
  return Isa::base;
}

bool parse_isa(std::string_view text, Isa& out) {
  for (const Isa i : {Isa::base, Isa::avx2, Isa::avx512}) {
    if (text == isa_name(i)) {
      out = i;
      return true;
    }
  }
  return false;
}

std::string Spec::describe() const {
  if (kind == "mlp_naive" || kind == "mlp_sep") return std::format("{} H{} L{} F{}", kind, hidden, layers, freqs);
  if (kind == "grid_mlp") return std::format("grid_mlp {}x{}x{} C{} H{} L{}", grid, grid, grid_t, channels, hidden, layers);
  if (kind == "hash_mlp") {
    return std::format("hash_mlp L{} F{} T2^{} H{} L{}", levels, features, log2_table, hidden, layers);
  }
  if (kind == "conv_dec") return std::format("conv_dec {}^2x{} {}-{}-{}-4", latent, grid_t, c0, c1, c2);
  return kind;
}

std::unique_ptr<Model> make_model(const Spec& spec, Isa isa) {
  if (!isa_supported(isa)) throw std::invalid_argument(std::format("this CPU cannot run ISA {}", isa_name(isa)));
  switch (isa) {
    case Isa::base: return isa_base::create(spec);
    case Isa::avx2: return isa_avx2::create(spec);
    case Isa::avx512: return isa_avx512::create(spec);
  }
  throw std::invalid_argument("unknown ISA");
}

}  // namespace nfx::proto
