// nvfx_scene_script: play a scene script of composed effects (docs/COMPOSE.md §4) to video, keyframes and a profile.
//
//   nvfx_scene_script --script FILE [--models DIR] [--out scene.mp4 | --no-video] [--keyframes DIR] [--sheet sheet.png]
//                     [--threads 2] [--isa avx2|avx512|baseline] [--frames N] [--profile profile.csv]
//                     [--verify frozen.csv] [--check] [--print] [--no-overlap] [--own-scratch]
//
// Each frame's picture is drawn on a thread of its own while the next frame's state is computed (Options::overlap,
// with 2 or more threads; --threads counts that thread); --no-overlap draws it after the state. The frames are the
// same. --own-scratch gives every module its own step working memory instead of one set per thread (Options::
// shared_scratch; the same frames). The profile has every frame's stages, its wall time and the checksum of its RGB (FNV-1a, 64 bits, as
// nvfx_fireball's: the hand-written and the scripted fireball can be compared frame by frame).
//
// --check parses and checks the script without loading any effect; --print writes it back in canonical form.
// --verify compares the SHA-256 of the keyframes written to --keyframes with the rows "*_keyframe,frame_NNN,...,sha256"
// of a CSV (results/experiments/v1_frozen.csv holds the hand-written fireball's): exit 1 on a mismatch, 77 (skipped)
// when the effects cannot be found. Effects named in the script are read from --models DIR (default:
// $NEURALVFX_DATA/experiments/models/d, else the script's folder). Heap allocations in the frame loop are counted
// (this program replaces operator new); a scene should make none.
#include "script.hpp"

#include <neuralfx/dcm/mixer.hpp>
#include <neuralfx/image_io.hpp>
#include <neuralfx/nvfx.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <new>
#include <print>
#include <sstream>
#include <string>
#include <vector>

namespace {
std::atomic<long> g_allocations{0};
std::atomic<bool> g_counting{false};
}  // namespace

void* operator new(std::size_t n) {
  if (g_counting.load(std::memory_order_relaxed)) g_allocations.fetch_add(1, std::memory_order_relaxed);
  if (void* p = std::malloc(n ? n : 1)) return p;
  throw std::bad_alloc();
}
void* operator new[](std::size_t n) { return operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

namespace {

using namespace nfx;
namespace sc = nfx::compose::script;
using Clock = std::chrono::steady_clock;

struct Args {
  std::filesystem::path script, models, out = "scene.mp4", keyframes, sheet, profile, verify;
  int threads = 2, frames = -1;
  std::string isa;
  bool video = true, check = false, print = false, overlap = true, own_scratch = false;
};

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    const auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::invalid_argument(k + " needs a value");
      return argv[++i];
    };
    if (k == "--script") a.script = next();
    else if (k == "--models") a.models = next();
    else if (k == "--out") a.out = next();
    else if (k == "--keyframes") a.keyframes = next();
    else if (k == "--sheet") a.sheet = next();
    else if (k == "--profile") a.profile = next();
    else if (k == "--verify") a.verify = next();
    else if (k == "--threads") a.threads = std::stoi(next());
    else if (k == "--frames") a.frames = std::stoi(next());
    else if (k == "--isa") a.isa = next();
    else if (k == "--no-video") a.video = false;
    else if (k == "--check") a.check = true;
    else if (k == "--print") a.print = true;
    else if (k == "--no-overlap") a.overlap = false;
    else if (k == "--own-scratch") a.own_scratch = true;
    else throw std::invalid_argument("unknown option " + k + " (see the source header)");
  }
  if (a.script.empty()) throw std::invalid_argument("--script FILE is needed");
  if (a.models.empty()) {
    if (const char* d = std::getenv("NEURALVFX_DATA")) a.models = std::filesystem::path(d) / "experiments" / "models" / "d";
    else a.models = a.script.parent_path();
  }
  if (!a.verify.empty() && a.keyframes.empty()) throw std::invalid_argument("--verify needs --keyframes DIR");
  return a;
}

std::string file_sha256(const std::filesystem::path& p) {
  std::ifstream f(p, std::ios::binary);
  std::stringstream ss;
  ss << f.rdbuf();
  return dcm::sha256_hex(ss.str());
}

}  // namespace

int main(int argc, char** argv) try {
  const Args A = parse(argc, argv);
  const sc::Script script = sc::parse_file(A.script);
  if (A.print) std::print("{}", sc::print(script));
  if (A.check) {
    sc::validate(script);
    std::println("{}: ok ({} statements)", A.script.string(), script.statements.size());
    return 0;
  }
  if (A.print) return 0;
  if (!A.isa.empty()) {
    const nvfx_isa want = A.isa == "avx512" ? NVFX_ISA_AVX512 : A.isa == "avx2" ? NVFX_ISA_AVX2 : NVFX_ISA_BASELINE;
    if (nvfx_set_isa(want) != NVFX_OK) throw std::runtime_error("this CPU cannot run --isa " + A.isa);
  }
  if (!A.verify.empty()) {  // a check against frozen keyframes is skipped, not failed, without the effects
    for (const sc::Statement& s : script.statements) {
      if (s.keyword != "effect") continue;
      const std::filesystem::path p = std::filesystem::path(s.kind).is_absolute() ? std::filesystem::path(s.kind) : A.models / s.kind;
      if (!std::filesystem::exists(p)) {
        std::println("nvfx_scene_script: {} not found: skipped", p.string());
        return 77;
      }
    }
  }
  const auto setup0 = Clock::now();
  sc::Options opt;
  opt.threads = A.threads;
  opt.isa = compose::best_isa();
  opt.overlap = A.overlap;
  opt.shared_scratch = !A.own_scratch;
  sc::Scene scene(script, sc::load_from(A.models), opt);
  const double setup_ms = std::chrono::duration<double, std::milli>(Clock::now() - setup0).count();
  const int W = scene.width(), H = scene.height();
  const int frames = A.frames >= 0 ? std::min(A.frames, scene.frames()) : scene.frames();
  std::println("nvfx_scene_script: {} at {}x{}, {} frames at {} fps ({} threads, {}{}); {} modules; setup {:.0f} ms", A.script.filename().string(), W, H, frames,
               scene.fps(), A.threads, compose::isa_name(opt.isa), A.overlap && A.threads > 1 ? ", overlapped" : "", scene.modules().size(), setup_ms);

  std::FILE* video = nullptr;
  if (A.video) {
    const std::string cmd = std::format("ffmpeg -loglevel error -y -f rawvideo -pix_fmt rgb24 -s {}x{} -r {} -i - -c:v libx264 -preset slow -crf 19 "
                                        "-pix_fmt yuv420p -movflags +faststart \"{}\"",
                                        W, H, scene.fps(), A.out.string());
    video = popen(cmd.c_str(), "w");
    if (!video) throw std::runtime_error("cannot start ffmpeg");
  }
  std::vector<std::uint8_t> rgb(static_cast<std::size_t>(W) * static_cast<std::size_t>(H) * 3);
  std::vector<int> key_frames;
  for (const float kt : scene.keyframes()) key_frames.push_back(static_cast<int>(std::lround(kt * scene.fps())));
  std::vector<Image> keys;
  std::vector<std::filesystem::path> key_files;
  std::vector<std::array<double, sc::Scene::kStages + 1>> prof(static_cast<std::size_t>(frames));
  std::vector<long> allocs(static_cast<std::size_t>(frames));
  std::vector<std::uint64_t> sums(static_cast<std::size_t>(frames));
  for (int f = 0; f < frames; ++f) {
    if (f == 1) g_counting.store(true);  // the first frame may still touch lazily sized buffers
    const long a0 = g_allocations.load();
    const auto c0 = Clock::now();
    scene.render(f, rgb);
    const double total = std::chrono::duration<double, std::milli>(Clock::now() - c0).count();
    allocs[static_cast<std::size_t>(f)] = g_allocations.load() - a0;
    g_counting.store(false);
    auto& P = prof[static_cast<std::size_t>(f)];
    std::ranges::copy(scene.stage_ms(), P.begin());
    P[sc::Scene::kStages] = total;
    std::uint64_t h = 0xcbf29ce484222325ull;
    for (const std::uint8_t v : rgb) h = (h ^ v) * 0x100000001b3ull;
    sums[static_cast<std::size_t>(f)] = h;
    if (video) std::fwrite(rgb.data(), 1, rgb.size(), video);
    if (std::ranges::find(key_frames, f) != key_frames.end()) {
      Image img;
      img.allocate(W, H);
      for (std::size_t i = 0, j = 0; i < rgb.size(); i += 3, j += 4) {
        img.rgba[j] = rgb[i];
        img.rgba[j + 1] = rgb[i + 1];
        img.rgba[j + 2] = rgb[i + 2];
        img.rgba[j + 3] = 255;
      }
      if (!A.keyframes.empty()) {
        std::filesystem::create_directories(A.keyframes);
        key_files.push_back(A.keyframes / std::format("frame_{:03d}.png", f));
        if (auto w = write_png(key_files.back(), img); !w) throw std::runtime_error(w.error());
      }
      keys.push_back(std::move(img));
    }
    if (f % 30 == 0) {
      std::println("  t {:4.1f} s: {:6.1f} ms ({} modules, {} particles)", static_cast<double>(f) / static_cast<double>(scene.fps()), total, scene.active_modules(),
                   scene.particles().alive());
      std::fflush(stdout);
    }
  }
  if (video && pclose(video) != 0) throw std::runtime_error("ffmpeg failed");

  // summary
  const auto median = [](std::vector<double> v) {
    std::ranges::sort(v);
    return v.empty() ? 0.0 : v[v.size() / 2];
  };
  std::println("\nstage              median ms per frame ({} frames)", frames);
  for (int s = 0; s <= sc::Scene::kStages; ++s) {
    std::vector<double> v;
    for (const auto& p : prof) v.push_back(p[static_cast<std::size_t>(s)]);
    std::println("{:<18} {:8.2f}", s == sc::Scene::kStages ? "total" : sc::Scene::stage_name(s), median(v));
  }
  long alloc_total = 0, alloc_frames = 0;
  for (int f = 1; f < frames; ++f) {
    alloc_total += allocs[static_cast<std::size_t>(f)];
    alloc_frames += allocs[static_cast<std::size_t>(f)] > 0;
  }
  std::println("allocations in the frame loop: {} in {} of {} frames", alloc_total, alloc_frames, std::max(0, frames - 1));
  std::println("module scratch {:.1f} MB", static_cast<double>(scene.scratch_bytes()) / 1048576.0);
  for (const auto& [name, t] : scene.rules_fired()) {
    if (std::isinf(t)) std::println("rule '{}' did not fire", name);
    else std::println("rule '{}' first fired at {:.2f} s", name, t);
  }
  if (!A.profile.empty()) {
    std::ofstream o(A.profile);
    o << "frame,allocations";
    for (int s = 0; s < sc::Scene::kStages; ++s) o << ',' << sc::Scene::stage_name(s);
    o << ",period,rgb_fnv\n";  // period: the frame's wall time (render() returns each frame done)
    for (int f = 0; f < frames; ++f) {
      o << f << ',' << allocs[static_cast<std::size_t>(f)];
      for (const double v : prof[static_cast<std::size_t>(f)]) o << std::format(",{:.3f}", v);
      o << std::format(",{:016x}\n", sums[static_cast<std::size_t>(f)]);
    }
  }
  if (!A.sheet.empty() && !keys.empty()) {  // keyframes, 4 per row, at a quarter size
    const int kw = W / 4, kh = H / 4, cols = 4, rows_n = (static_cast<int>(keys.size()) + cols - 1) / cols;
    Image sheet;
    sheet.allocate(cols * kw, rows_n * kh);
    for (std::size_t k = 0; k < keys.size(); ++k) {
      const int ox = static_cast<int>(k) % cols * kw, oy = static_cast<int>(k) / cols * kh;
      for (int y = 0; y < kh; ++y) {
        for (int x = 0; x < kw; ++x) {
          int acc[3] = {0, 0, 0};
          for (int dy = 0; dy < 4; ++dy)
            for (int dx = 0; dx < 4; ++dx)
              for (int c = 0; c < 3; ++c) acc[c] += keys[k].pixel(4 * x + dx, 4 * y + dy)[c];
          std::uint8_t* d = sheet.pixel(ox + x, oy + y);
          for (int c = 0; c < 3; ++c) d[c] = static_cast<std::uint8_t>(acc[c] / 16);
          d[3] = 255;
        }
      }
    }
    if (auto w = write_png(A.sheet, sheet); !w) std::println("sheet: {}", w.error());
  }
  if (!A.verify.empty()) {
    std::ifstream in(A.verify);
    if (!in) throw std::runtime_error("cannot read " + A.verify.string());
    std::map<std::string, std::string> want;  // frame_NNN -> sha256
    for (std::string line; std::getline(in, line);) {
      std::vector<std::string> cols;
      std::stringstream ls(line);
      for (std::string c; std::getline(ls, c, ',');) cols.push_back(c);
      if (cols.size() >= 5 && cols[0].ends_with("_keyframe")) want[cols[1]] = cols[4];
    }
    int same = 0, differ = 0;
    for (const auto& p : key_files) {
      const std::string name = p.stem().string();
      const auto it = want.find(name);
      if (it == want.end()) continue;
      const std::string got = file_sha256(p);
      const bool ok = got == it->second;
      (ok ? same : differ) += 1;
      std::println("  {} {} {}", name, got, ok ? "same" : "DIFFERENT (frozen " + it->second + ")");
    }
    std::println("verify: {} of {} keyframes bit-exact against {}", same, same + differ, A.verify.string());
    if (differ > 0 || same == 0) return 1;
  }
  return alloc_total == 0 ? 0 : 2;
} catch (const std::exception& e) {
  std::println(stderr, "nvfx_scene_script: {}", e.what());
  return 1;
}
