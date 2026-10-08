// neuralfx_arch_bench: Phase 0 microbenchmark. Milliseconds per sprite frame on one pinned core for each candidate
// architecture family (docs/PLAN.md §8). The weights are random; only cost is measured here.
//
//   neuralfx_arch_bench [--isa base|avx2|avx512] [--sizes 64,128,256] [--frames N] [--core K] [--csv FILE]
//
// Prints one markdown table per size. Times are the median and 90th percentile over N frames (t sweeps [0, 1)),
// after 10% warm-up frames. Memory is the learnable parameters at 2 bytes each (fp16 storage) and at 4.
#include <neuralfx/proto.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif
#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

using nfx::proto::Spec;

namespace {

std::vector<Spec> sweep(int size) {
  std::vector<Spec> v;
  const auto add = [&](auto&& f) {
    Spec s;
    s.size = size;
    f(s);
    v.push_back(s);
  };
  for (const int h : {16, 32, 64}) add([&](Spec& s) { s.kind = "mlp_naive"; s.hidden = h; s.layers = 2; });
  for (const auto& [h, l] : {std::pair{16, 2}, {32, 2}, {32, 3}, {64, 2}, {64, 3}}) {
    add([&](Spec& s) { s.kind = "mlp_sep"; s.hidden = h; s.layers = l; });
  }
  add([](Spec& s) { s.kind = "grid_mlp"; s.grid = 32; s.grid_t = 16; s.channels = 8; s.hidden = 16; s.layers = 1; });
  add([](Spec& s) { s.kind = "grid_mlp"; s.grid = 32; s.grid_t = 16; s.channels = 8; s.hidden = 32; s.layers = 1; });
  add([](Spec& s) { s.kind = "grid_mlp"; s.grid = 32; s.grid_t = 32; s.channels = 16; s.hidden = 32; s.layers = 2; });
  add([](Spec& s) { s.kind = "grid_mlp"; s.grid = 64; s.grid_t = 16; s.channels = 8; s.hidden = 32; s.layers = 2; });
  add([](Spec& s) { s.kind = "hash_mlp"; s.levels = 8; s.features = 2; s.log2_table = 12; s.hidden = 16; s.layers = 1; });
  add([](Spec& s) { s.kind = "hash_mlp"; s.levels = 8; s.features = 2; s.log2_table = 14; s.hidden = 32; s.layers = 2; });
  add([](Spec& s) { s.kind = "hash_mlp"; s.levels = 16; s.features = 2; s.log2_table = 14; s.hidden = 64; s.layers = 2; });
  if (size % 8 == 0 && (size / 8) % 2 == 0) {
    const int lat = size / 8;
    add([&](Spec& s) { s.kind = "conv_dec"; s.latent = lat; s.grid_t = 16; s.c0 = 16; s.c1 = 8; s.c2 = 8; });
    add([&](Spec& s) { s.kind = "conv_dec"; s.latent = lat; s.grid_t = 16; s.c0 = 32; s.c1 = 16; s.c2 = 8; });
    add([&](Spec& s) { s.kind = "conv_dec"; s.latent = lat; s.grid_t = 16; s.c0 = 32; s.c1 = 32; s.c2 = 16; });
  }
  return v;
}

struct Timing {
  double median_ms = 0, p90_ms = 0;
  std::uint64_t checksum = 0;
};

template <class F>
Timing time_frames(int frames, F&& render) {
  const int warm = std::max(2, frames / 10);
  std::vector<double> ms;
  ms.reserve(frames);
  std::uint64_t sum = 0;
  for (int i = 0; i < warm + frames; ++i) {
    const float t = static_cast<float>(i % frames) / static_cast<float>(frames);
    const auto a = std::chrono::steady_clock::now();
    sum += render(t);
    const auto b = std::chrono::steady_clock::now();
    if (i >= warm) ms.push_back(std::chrono::duration<double, std::milli>(b - a).count());
  }
  std::sort(ms.begin(), ms.end());
  return {ms[ms.size() / 2], ms[std::min(ms.size() - 1, ms.size() * 9 / 10)], sum};
}

// Reference: playing back a 64-frame RGBA8 flipbook with linear blending between neighbouring frames.
Timing flipbook(int size, int frames) {
  const int nf = 64;
  const std::size_t px = static_cast<std::size_t>(size) * size;
  std::vector<std::uint8_t> atlas(px * 4 * nf), out(px * 4);
  for (std::size_t i = 0; i < atlas.size(); ++i) atlas[i] = static_cast<std::uint8_t>(i * 2654435761u >> 24);
  return time_frames(frames, [&](float t) -> std::uint64_t {
    const float f = t * (nf - 1);
    const int f0 = std::min(static_cast<int>(f), nf - 2);
    const auto w = static_cast<std::uint32_t>((f - static_cast<float>(f0)) * 256.f);
    const std::uint8_t* a = atlas.data() + px * 4 * f0;
    const std::uint8_t* b = a + px * 4;
    for (std::size_t i = 0; i < px * 4; ++i) out[i] = static_cast<std::uint8_t>((a[i] * (256 - w) + b[i] * w) >> 8);
    return out[px];
  });
}

std::vector<int> parse_sizes(const std::string& s) {
  std::vector<int> v;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) v.push_back(std::stoi(item));
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  nfx::proto::Isa isa = nfx::proto::best_isa();
  std::vector<int> sizes{128};
  int frames = 200, core = 0;
  std::string csv;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::cerr << a << " needs a value\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--isa") {
      if (!nfx::proto::parse_isa(next(), isa)) {
        std::cerr << "--isa: base, avx2 or avx512\n";
        return 2;
      }
    } else if (a == "--sizes") {
      sizes = parse_sizes(next());
    } else if (a == "--frames") {
      frames = std::stoi(next());
    } else if (a == "--core") {
      core = std::stoi(next());
    } else if (a == "--csv") {
      csv = next();
    } else {
      std::cerr << "usage: neuralfx_arch_bench [--isa base|avx2|avx512] [--sizes 64,128] [--frames N] [--core K] "
                   "[--csv FILE]\n";
      return 2;
    }
  }
  if (!nfx::proto::isa_supported(isa)) {
    std::cerr << "this CPU cannot run " << nfx::proto::isa_name(isa) << "\n";
    return 2;
  }
#if defined(__linux__)
  if (core >= 0) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) std::cerr << "warning: could not pin to core " << core << "\n";
  }
#endif
#if defined(__x86_64__)
  _mm_setcsr(_mm_getcsr() | 0x8040);  // flush denormals to zero (FTZ | DAZ), as a game runtime would
#endif

  std::ofstream out;
  if (!csv.empty()) {
    out.open(csv);
    out << "isa,size,model,params,kb_fp16,macs_per_px,median_ms,p90_ms,gmac_s\n";
  }
  std::cout << std::format("ISA {}, {} frames per model, pinned to core {}\n", nfx::proto::isa_name(isa), frames, core);
  for (const int size : sizes) {
    std::cout << std::format("\n### {0}x{0}\n\n", size);
    std::cout << "| Model | Params | KB fp16 | MAC/px | ms median | ms p90 | GMAC/s |\n";
    std::cout << "|---|---:|---:|---:|---:|---:|---:|\n";
    const Timing fb = flipbook(size, frames);
    std::cout << std::format("| flipbook 64 frames RGBA8, blended (reference) | - | {} | - | {:.3f} | {:.3f} | - |\n",
                             size * size * 4 * 64 / 1024, fb.median_ms, fb.p90_ms);
    for (const Spec& s : sweep(size)) {
      auto m = nfx::proto::make_model(s, isa);
      std::vector<std::uint8_t> rgba(static_cast<std::size_t>(size) * size * 4);
      const Timing t = time_frames(frames, [&](float tt) -> std::uint64_t {
        nfx::proto::Controls c;
        c.t = tt;
        m->render(c, rgba.data());
        return rgba[rgba.size() / 2];
      });
      const double kb = static_cast<double>(m->param_count()) * 2.0 / 1024.0;
      const double gmac = m->macs_per_pixel() * size * size / (t.median_ms * 1e6);
      std::cout << std::format("| {} | {} | {:.1f} | {:.0f} | {:.3f} | {:.3f} | {:.1f} |\n", s.describe(), m->param_count(), kb,
                               m->macs_per_pixel(), t.median_ms, t.p90_ms, gmac);
      std::cout.flush();
      if (out) {
        out << std::format("{},{},{},{},{:.1f},{:.0f},{:.4f},{:.4f},{:.2f}\n", nfx::proto::isa_name(isa), size, s.describe(),
                           m->param_count(), kb, m->macs_per_pixel(), t.median_ms, t.p90_ms, gmac);
      }
    }
  }
  return 0;
}
