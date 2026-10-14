// nvfx_gallery: every effect of a library catalogue (docs/LIBRARY.md), played through the library's C API
// (nvfx_library.h): a labelled contact sheet, a video that walks through them, keyframes and their check against
// frozen SHA-256, and a check of the whole catalogue.
//
//   nvfx_gallery --library DIR --models DIR [--sheet sheet.png] [--video walk.mp4 [--clip S]] [--keyframes DIR]
//                [--verify frozen.csv] [--only a,b,c] [--threads 2] [--cell 320x240] [--columns 6]
//   nvfx_gallery --library DIR [--models DIR] --check [--require-models]
//   nvfx_gallery --library DIR --list [--markdown]
//   nvfx_gallery --library DIR --print NAME            the stage script of an effect alone (as the gallery plays it)
//   nvfx_gallery --starts FILE.nvfx                     a learned effect's start points and their controls
//   nvfx_gallery --library DIR --models DIR --shot 960x540 [--ground 470] --spawn NAME@X,Y[,SCALE[,AT[,VX,VY]]] ...
//                --frames N --png frame.png [--print-stage]
//   --pack FILE in place of --library and --models: the catalogue and its models from an effect pack (nvfx_pack.h;
//                exit 77 with --verify when the pack is not there)
//
// Each effect plays alone on a stage fitted to its box (its catalogue size, at scale 1): spawned at 0.2 s, its
// thumbnail and keyframe taken `show` seconds later, its clip in the video 4 s long (one-shots: their length; --clip S:
// S seconds each). A moving effect (`moving (vx, vy)`, a trail) is moved every frame (nvfx_stage_move) at that velocity
// through its box: half its `show` time before the anchor point at the spawn, half of it after at the show, and back
// and forth after that. The keyframes are PNG files named after the effects, with their rows for a frozen CSV in
// DIR/keyframes.csv; --verify compares their SHA-256 with the rows
// "library_keyframe,NAME,file,bytes,sha256" of a CSV (exit 1 on a difference, 77 when the effects' files are missing).
//
// --check opens the catalogue (every entry parsed and checked as a stage of its own) and plays every entry whose
// learned effects are in --models (procedural ones always): two instances, parameters set to their ends, one moved,
// one stopped and spawned again, with heap allocations counted in the frame loop and around those calls (this program
// replaces operator new); each must allocate nothing and draw something. --require-models: exit 77 when a file is
// missing (a test that needs the data).
//
// --shot plays a stage the way a game does (engines/godot's test does the same): created, its frame 0 drawn, the
// spawns given, then N frames stepped and drawn one at a time; the last frame is written as PNG. A spawn with VX, VY is
// moved before each frame to (X + VX s, Y + VY s), s the seconds since its spawn (world pixels a second).
#include "label_font.hpp"
#include "library.hpp"

#include <neuralfx/dcm/mixer.hpp>
#include <neuralfx/image_io.hpp>
#include <neuralfx/nvfx_library.h>
#include <neuralfx/nvfx_pack.h>

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

using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;

struct Spawn {  // (doubles: a moving spawn's places are computed in double and rounded to float, as GDScript does)
  std::string name;
  double x = 0, y = 0, scale = 1, at = -1, vx = 0, vy = 0;
};

struct Args {
  fs::path library, models, sheet, video, keyframes, verify, png, starts, pack;
  std::vector<std::string> only;
  std::string print;
  int threads = 2, columns = 6, cell_w = 320, cell_h = 240, frames = 45, shot_w = 0, shot_h = 0;
  float ground = 0.f, clip = 0.f;
  std::vector<Spawn> spawns;
  bool check = false, require_models = false, list = false, markdown = false, print_stage = false;
};

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  for (std::string x; std::getline(ss, x, sep);)
    if (!x.empty()) out.push_back(x);
  return out;
}

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    const auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::invalid_argument(k + " needs a value");
      return argv[++i];
    };
    const auto dims = [&](int& w, int& h) {
      const auto v = split(next(), 'x');
      if (v.size() != 2) throw std::invalid_argument(k + " takes W x H, as 320x240");
      w = std::stoi(v[0]);
      h = std::stoi(v[1]);
    };
    if (k == "--library") a.library = next();
    else if (k == "--pack") a.pack = next();
    else if (k == "--models") a.models = next();
    else if (k == "--sheet") a.sheet = next();
    else if (k == "--video") a.video = next();
    else if (k == "--keyframes") a.keyframes = next();
    else if (k == "--verify") a.verify = next();
    else if (k == "--png") a.png = next();
    else if (k == "--only") a.only = split(next(), ',');
    else if (k == "--print") a.print = next();
    else if (k == "--starts") a.starts = next();
    else if (k == "--threads") a.threads = std::stoi(next());
    else if (k == "--columns") a.columns = std::stoi(next());
    else if (k == "--cell") dims(a.cell_w, a.cell_h);
    else if (k == "--shot") dims(a.shot_w, a.shot_h);
    else if (k == "--ground") a.ground = std::stof(next());
    else if (k == "--clip") a.clip = std::stof(next());
    else if (k == "--frames") a.frames = std::stoi(next());
    else if (k == "--spawn") {
      const std::string v = next();
      const auto at = v.find('@');
      if (at == std::string::npos) throw std::invalid_argument("--spawn takes NAME@X,Y[,SCALE[,AT[,VX,VY]]]");
      Spawn s;
      s.name = v.substr(0, at);
      const auto n = split(v.substr(at + 1), ',');
      if (n.size() < 2 || n.size() == 5) throw std::invalid_argument("--spawn takes NAME@X,Y[,SCALE[,AT[,VX,VY]]]");
      s.x = std::stod(n[0]);
      s.y = std::stod(n[1]);
      if (n.size() > 2) s.scale = std::stod(n[2]);
      if (n.size() > 3) s.at = std::stod(n[3]);
      if (n.size() > 5) {
        s.vx = std::stod(n[4]);
        s.vy = std::stod(n[5]);
      }
      a.spawns.push_back(s);
    } else if (k == "--check") a.check = true;
    else if (k == "--require-models") a.require_models = true;
    else if (k == "--list") a.list = true;
    else if (k == "--markdown") a.markdown = true;
    else if (k == "--print-stage") a.print_stage = true;
    else throw std::invalid_argument("unknown option " + k + " (see the source header)");
  }
  if (a.library.empty() && a.starts.empty() && a.pack.empty()) throw std::invalid_argument("--library DIR (or --pack FILE) is needed");
  if (a.models.empty()) {
    if (const char* d = std::getenv("NEURALVFX_DATA")) a.models = fs::path(d) / "library" / "models";
  }
  if (!a.verify.empty() && a.keyframes.empty()) throw std::invalid_argument("--verify needs --keyframes DIR");
  return a;
}

struct Rgba {
  int w = 0, h = 0;
  std::vector<std::uint8_t> px;
  void allocate(int width, int height) {
    w = width;
    h = height;
    px.assign(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 4, 0);
  }
  std::uint8_t* at(int x, int y) { return px.data() + (static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)) * 4; }
  const std::uint8_t* at(int x, int y) const { return px.data() + (static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)) * 4; }
};

void fill(Rgba& img, int x0, int y0, int w, int h, std::array<std::uint8_t, 3> c) {
  for (int y = std::max(0, y0); y < std::min(img.h, y0 + h); ++y)
    for (int x = std::max(0, x0); x < std::min(img.w, x0 + w); ++x) {
      std::uint8_t* p = img.at(x, y);
      p[0] = c[0];
      p[1] = c[1];
      p[2] = c[2];
      p[3] = 255;
    }
}

// `src` into the rectangle (x0, y0, w, h) of `dst`, fitted (aspect kept, centred): each target pixel the mean of the
// source pixels whose centres fall in it (bilinear when enlarging).
void blit_fit(const Rgba& src, Rgba& dst, int x0, int y0, int w, int h) {
  const float k = std::min(static_cast<float>(w) / static_cast<float>(src.w), static_cast<float>(h) / static_cast<float>(src.h));
  const int tw = std::max(1, static_cast<int>(std::lround(static_cast<float>(src.w) * k))), th = std::max(1, static_cast<int>(std::lround(static_cast<float>(src.h) * k)));
  const int ox = x0 + (w - tw) / 2, oy = y0 + (h - th) / 2;
  for (int ty = 0; ty < th; ++ty) {
    const float sy0 = static_cast<float>(ty) / k, sy1 = static_cast<float>(ty + 1) / k;
    for (int tx = 0; tx < tw; ++tx) {
      const float sx0 = static_cast<float>(tx) / k, sx1 = static_cast<float>(tx + 1) / k;
      int ya = static_cast<int>(std::ceil(sy0 - 0.5f)), yb = static_cast<int>(std::ceil(sy1 - 0.5f));
      int xa = static_cast<int>(std::ceil(sx0 - 0.5f)), xb = static_cast<int>(std::ceil(sx1 - 0.5f));
      if (yb <= ya) yb = ya + 1;
      if (xb <= xa) xb = xa + 1;
      ya = std::clamp(ya, 0, src.h - 1);
      yb = std::clamp(yb, ya + 1, src.h);
      xa = std::clamp(xa, 0, src.w - 1);
      xb = std::clamp(xb, xa + 1, src.w);
      unsigned acc[3] = {0, 0, 0}, n = 0;
      for (int y = ya; y < yb; ++y)
        for (int x = xa; x < xb; ++x) {
          const std::uint8_t* p = src.at(x, y);
          for (int c = 0; c < 3; ++c) acc[c] += p[c];
          ++n;
        }
      const int dx = ox + tx, dy = oy + ty;
      if (dx < 0 || dy < 0 || dx >= dst.w || dy >= dst.h) continue;
      std::uint8_t* d = dst.at(dx, dy);
      for (int c = 0; c < 3; ++c) d[c] = static_cast<std::uint8_t>((acc[c] + n / 2) / n);
      d[3] = 255;
    }
  }
}

void text(Rgba& img, std::string_view s, int x, int y, int scale, std::array<std::uint8_t, 3> c) {
  nfx::tools::draw_text(s, x, y, scale, [&](int px, int py) {
    if (px < 0 || py < 0 || px >= img.w || py >= img.h) return;
    std::uint8_t* p = img.at(px, py);
    p[0] = c[0];
    p[1] = c[1];
    p[2] = c[2];
  });
}

std::string sha256_file(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  std::stringstream ss;
  ss << f.rdbuf();
  return nfx::dcm::sha256_hex(ss.str());
}

bool write(const fs::path& p, const Rgba& img) {
  nfx::Image out;
  out.allocate(img.w, img.h);
  std::ranges::copy(img.px, out.rgba.begin());
  if (auto w = nfx::write_png(p, out); !w) {
    std::println(stderr, "{}: {}", p.string(), w.error());
    return false;
  }
  return true;
}

[[noreturn]] void die(const std::string& m) {
  std::println(stderr, "nvfx_gallery: {}", m);
  std::exit(1);
}

bool g_from_pack = false;  // the library and its models come from a pack: nothing is missing

nvfx_library* open(const Args& A) {
  if (!A.pack.empty()) {
    nvfx_pack_desc pd;
    nvfx_pack_desc_init(&pd);
    const std::string path = A.pack.string();
    pd.path = path.c_str();
    nvfx_pack* pack = nullptr;
    nvfx_library* lib = nullptr;
    nvfx_scene_error err;
    if (nvfx_pack_open(&pd, &pack, &err) != NVFX_OK) die(err.message);
    const nvfx_status st = nvfx_library_open_pack(pack, &lib, &err);
    nvfx_pack_free(pack);  // the library keeps what it needs
    if (st != NVFX_OK) die(err.message);
    g_from_pack = true;
    return lib;
  }
  nvfx_library_desc d;
  nvfx_library_desc_init(&d);
  const std::string dir = A.library.string(), models = A.models.string();
  d.catalogue_dir = dir.c_str();
  d.models_dir = A.models.empty() ? nullptr : models.c_str();
  nvfx_library* lib = nullptr;
  nvfx_scene_error err;
  if (nvfx_library_open(&d, &lib, &err) != NVFX_OK) die(err.message);
  return lib;
}

// The files an effect needs that are not in the models' folder.
std::vector<std::string> missing(const nvfx_library* lib, int i, const fs::path& models) {
  std::vector<std::pair<std::string, std::string>> files;
  nvfx_library_list_models(lib, i, [](void* u, const char* n, const char* f) { static_cast<std::vector<std::pair<std::string, std::string>>*>(u)->emplace_back(n, f); },
                           &files);
  std::vector<std::string> out;
  if (g_from_pack) return out;
  for (const auto& [n, f] : files) {
    const fs::path p = fs::path(f).is_absolute() ? fs::path(f) : models / f;
    if (models.empty() || !fs::exists(p)) out.push_back(f);
  }
  return out;
}

// The stage an effect is shown on: fitted to its box, about 4:3; where it is spawned.
struct Shot {
  int w = 320, h = 240;
  float ground = 210.f, x = 160.f, y = 210.f;
};

int ceil8(float v) { return static_cast<int>(std::ceil(v / 8.f)) * 8; }

Shot shot_for(const nvfx_library_effect& e) {
  Shot s;
  s.w = ceil8(std::max(320.f, e.width * 1.12f + 24.f));
  s.h = ceil8(std::max(240.f, e.height * 1.16f + 24.f));
  if (static_cast<float>(s.w) < 4.f / 3.f * static_cast<float>(s.h)) s.w = ceil8(4.f / 3.f * static_cast<float>(s.h));
  else s.h = ceil8(0.75f * static_cast<float>(s.w));
  s.ground = std::round(0.9f * static_cast<float>(s.h));
  s.x = e.anchor == NVFX_ANCHOR_LEFT ? std::round(0.5f * (static_cast<float>(s.w) - e.width)) : 0.5f * static_cast<float>(s.w);
  s.y = e.anchor == NVFX_ANCHOR_BASE ? s.ground : std::round(s.ground - 0.5f * e.height - 0.05f * static_cast<float>(s.h));
  return s;
}

struct Stage {
  nvfx_stage* stage = nullptr;
  nvfx_scene* scene = nullptr;
  nvfx_scene_info info{};
  Rgba frame;
  ~Stage() {
    if (stage) nvfx_stage_free(stage);
  }
  void render() {
    if (nvfx_scene_render(scene, frame.px.data(), static_cast<std::size_t>(frame.w) * 4) != NVFX_OK) die("render failed");
  }
};

void make_stage(Stage& s, const nvfx_library* lib, int w, int h, float ground, const std::vector<nvfx_stage_room>& rooms, int threads) {
  nvfx_stage_desc d;
  nvfx_stage_desc_init(&d);
  d.width = w;
  d.height = h;
  d.ground = ground;
  d.rooms = rooms.data();
  d.n_rooms = static_cast<int>(rooms.size());
  d.threads = threads;
  nvfx_scene_error err;
  if (nvfx_stage_create(lib, &d, &s.stage, &err) != NVFX_OK) die(err.message);
  s.scene = nvfx_stage_scene(s.stage);
  nvfx_scene_get_info(s.scene, &s.info);
  s.frame.allocate(s.info.width, s.info.height);
}

constexpr float kSpawnAt = 0.2f;  // when the gallery spawns an effect (frame 6)

// Where the gallery has a moving effect `s` seconds after its spawn: through its box at its velocity, half its show time
// before the anchor point at the spawn and half after it at the show, then back and forth between the two.
std::array<float, 2> path(const nvfx_library_effect& e, float x, float y, float s) {
  const float T = std::max(e.show, 0.5f), half = 0.5f * T;
  float u = std::fmod(std::max(s, 0.f), 2.f * T);
  if (u > T) u = 2.f * T - u;
  return {x + e.move_x * (u - half), y + e.move_y * (u - half)};
}
bool moving(const nvfx_library_effect& e) { return e.move_x != 0.f || e.move_y != 0.f; }

// --- the gallery ---------------------------------------------------------------------------------------------------

struct Shown {
  int index = 0;
  nvfx_library_effect e{};
  Rgba thumb;  // the frame at `show`
  fs::path key;
  double ms = 0;
  long allocs = 0;
};

int gallery(const Args& A, nvfx_library* lib) {
  std::vector<int> order;  // by category, then catalogue order
  for (int c = 0; c < nvfx_library_category_count(lib); ++c) {
    const std::string cat = nvfx_library_category(lib, c);
    for (int i = 0; i < nvfx_library_count(lib); ++i) {
      nvfx_library_effect e;
      nvfx_library_get_effect(lib, i, &e);
      if (e.category != cat) continue;
      if (!A.only.empty() && std::ranges::find(A.only, std::string(e.name)) == A.only.end()) continue;
      order.push_back(i);
    }
  }
  for (const std::string& o : A.only)
    if (nvfx_library_find(lib, o.c_str()) < 0) die("no effect '" + o + "' in the catalogue");
  for (const int i : order)
    if (const auto m = missing(lib, i, A.models); !m.empty()) {
      nvfx_library_effect e;
      nvfx_library_get_effect(lib, i, &e);
      std::println("nvfx_gallery: {} needs {} (not in {}): skipped", e.name, m.front(), A.models.string());
      if (!A.verify.empty()) return 77;
      die("the effects' files are needed for the gallery (--models DIR)");
    }
  constexpr int kVW = 960, kVH = 540;
  std::FILE* video = nullptr;
  if (!A.video.empty()) {
    const std::string cmd = std::format("ffmpeg -loglevel error -y -f rawvideo -pix_fmt rgb24 -s {}x{} -r 30 -i - -c:v libx264 -preset slow -crf 20 -threads 2 "
                                        "-pix_fmt yuv420p -movflags +faststart \"{}\"",
                                        kVW, kVH, A.video.string());
    video = popen(cmd.c_str(), "w");
    if (!video) die("cannot start ffmpeg");
  }
  Rgba canvas;
  canvas.allocate(kVW, kVH);
  std::vector<std::uint8_t> rgb(static_cast<std::size_t>(kVW) * kVH * 3);
  std::vector<Shown> shown;
  if (!A.keyframes.empty()) fs::create_directories(A.keyframes);
  for (const int i : order) {
    Shown s;
    s.index = i;
    nvfx_library_get_effect(lib, i, &s.e);
    const Shot sh = shot_for(s.e);
    Stage st;
    make_stage(st, lib, sh.w, sh.h, sh.ground, {{s.e.name, 1, 0}}, A.threads);
    const float fps = st.info.fps;
    const int spawn_f = static_cast<int>(std::lround(kSpawnAt * fps));
    nvfx_fx fx = 0;
    const auto start = moving(s.e) ? path(s.e, sh.x, sh.y, 0.f) : std::array<float, 2>{sh.x, sh.y};
    if (nvfx_stage_spawn(st.stage, s.e.name, start[0], start[1], 1.f, static_cast<double>(static_cast<float>(spawn_f) / fps), &fx) != NVFX_OK) die("spawn failed");
    const int show_f = spawn_f + static_cast<int>(std::lround(s.e.show * fps));
    const float clip = A.clip > 0.f ? A.clip : s.e.looping ? 4.f : std::min(s.e.duration + 0.6f, 6.f);
    const int clip0 = spawn_f, clip1 = spawn_f + static_cast<int>(std::lround(clip * fps));
    const int last = video ? std::max(show_f, clip1) : show_f;
    const auto t0 = Clock::now();
    for (int f = 0; f <= last; ++f) {
      if (f == 2) g_counting.store(true);
      const long a0 = g_allocations.load();
      if (moving(s.e) && f > spawn_f) {  // where it is in the frame about to be computed
        const auto p = path(s.e, sh.x, sh.y, static_cast<float>(f - spawn_f) / fps);
        nvfx_stage_move(st.stage, fx, p[0], p[1]);
      }
      if (f > 0) nvfx_scene_step_frames(st.scene, 1);
      st.render();
      s.allocs += g_allocations.load() - a0;
      g_counting.store(false);
      if (f == show_f) s.thumb = st.frame;
      if (video && f >= clip0 && f < clip1) {
        fill(canvas, 0, 0, kVW, kVH, {12, 12, 14});
        blit_fit(st.frame, canvas, 0, 0, kVW, kVH);
        fill(canvas, 0, 0, kVW, 34, {0, 0, 0});
        text(canvas, s.e.name, 12, 9, 2, {255, 255, 255});
        text(canvas, s.e.category, 12 + nfx::tools::text_width(s.e.name, 2) + 20, 16, 1, {170, 170, 175});
        text(canvas, s.e.models[0] ? s.e.models : "procedural", kVW - 12 - nfx::tools::text_width(s.e.models[0] ? s.e.models : "procedural", 1), 16, 1,
             {170, 170, 175});
        for (std::size_t p = 0, q = 0; p < canvas.px.size(); p += 4, q += 3) {
          rgb[q] = canvas.px[p];
          rgb[q + 1] = canvas.px[p + 1];
          rgb[q + 2] = canvas.px[p + 2];
        }
        std::fwrite(rgb.data(), 1, rgb.size(), video);
      }
    }
    s.ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / (last + 1);
    if (!A.keyframes.empty()) {
      s.key = A.keyframes / (std::string(s.e.name) + ".png");
      if (!write(s.key, s.thumb)) return 1;
    }
    std::println("  {:<20} {:<10} {}x{}, {:.1f} ms a frame, {} allocations after frame 1", s.e.name, s.e.category, sh.w, sh.h, s.ms, s.allocs);
    std::fflush(stdout);
    shown.push_back(std::move(s));
  }
  if (video && pclose(video) != 0) die("ffmpeg failed");
  if (!A.keyframes.empty()) {  // the rows a frozen CSV takes (results/library/frozen.csv)
    std::ofstream csv(A.keyframes / "keyframes.csv");
    for (const Shown& s : shown)
      csv << std::format("library_keyframe,{},{},{},{}\n", s.e.name, s.key.filename().string(), fs::file_size(s.key), sha256_file(s.key));
    std::println("keyframes: {} (and their rows in keyframes.csv)", A.keyframes.string());
  }
  if (!A.sheet.empty()) {
    const int label = 34, cols = std::max(1, std::min(A.columns, static_cast<int>(shown.size())));
    const int rows = (static_cast<int>(shown.size()) + cols - 1) / cols;
    Rgba sheet;
    sheet.allocate(cols * A.cell_w, rows * (A.cell_h + label));
    fill(sheet, 0, 0, sheet.w, sheet.h, {20, 20, 23});
    for (std::size_t k = 0; k < shown.size(); ++k) {
      const int cx = static_cast<int>(k) % cols * A.cell_w, cy = static_cast<int>(k) / cols * (A.cell_h + label);
      fill(sheet, cx + 1, cy + 1, A.cell_w - 2, A.cell_h - 2, {8, 8, 10});
      blit_fit(shown[k].thumb, sheet, cx + 1, cy + 1, A.cell_w - 2, A.cell_h - 2);
      text(sheet, shown[k].e.name, cx + 6, cy + A.cell_h + 4, 2, {240, 240, 240});
      const std::string sub = std::format("{} - {}", shown[k].e.category, shown[k].e.models[0] ? shown[k].e.models : "procedural");
      text(sheet, sub, cx + 6, cy + A.cell_h + 22, 1, {150, 150, 158});
    }
    if (!write(A.sheet, sheet)) return 1;
    std::println("sheet: {} ({} effects)", A.sheet.string(), shown.size());
  }
  long allocs = 0;
  for (const Shown& s : shown) allocs += s.allocs;
  std::println("allocations in the frame loops: {}", allocs);
  if (!A.verify.empty()) {
    std::ifstream in(A.verify);
    if (!in) die("cannot read " + A.verify.string());
    std::map<std::string, std::string> want;
    for (std::string line; std::getline(in, line);) {
      const auto c = split(line, ',');
      if (c.size() >= 5 && c[0] == "library_keyframe") want[c[1]] = c[4];
    }
    int same = 0, differ = 0;
    for (const Shown& s : shown) {
      const auto it = want.find(s.e.name);
      if (it == want.end()) continue;
      const std::string got = sha256_file(s.key);
      const bool ok = got == it->second;
      (ok ? same : differ) += 1;
      std::println("  {} {} {}", s.e.name, got, ok ? "same" : "DIFFERENT (frozen " + it->second + ")");
    }
    std::println("verify: {} of {} keyframes bit-exact against {}", same, same + differ, A.verify.string());
    if (differ > 0 || same == 0) return 1;
  }
  return allocs == 0 ? 0 : 2;
}

// --- the check -----------------------------------------------------------------------------------------------------

int check(const Args& A, nvfx_library* lib) {
  int failures = 0, played = 0, unplayed = 0;
  std::vector<std::string> need;
  for (int i = 0; i < nvfx_library_count(lib); ++i)
    for (const std::string& m : missing(lib, i, A.models)) need.push_back(m);
  if (A.require_models && !need.empty()) {
    std::println("nvfx_gallery: {} not in {}: skipped", need.front(), A.models.empty() ? "(no --models)" : A.models.string());
    return 77;
  }
  for (int i = 0; i < nvfx_library_count(lib); ++i) {
    nvfx_library_effect e;
    nvfx_library_get_effect(lib, i, &e);
    if (!A.only.empty() && std::ranges::find(A.only, std::string(e.name)) == A.only.end()) continue;
    if (const auto m = missing(lib, i, A.models); !m.empty()) {
      std::println("  {:<20} parsed and checked; not played ({} missing)", e.name, m.front());
      ++unplayed;
      continue;
    }
    Stage st;
    make_stage(st, lib, 480, 360, 320.f, {{e.name, 2, 1}}, A.threads);
    const float fps = st.info.fps;
    const float y = e.anchor == NVFX_ANCHOR_BASE ? 320.f : std::max(0.5f * e.height, 320.f - 0.5f * e.height);
    const float life = e.looping ? 1.6f : std::min(e.duration, 3.f);
    const int last = static_cast<int>(std::lround((0.1f + life + 0.8f) * fps));
    const int before = static_cast<int>(std::lround(0.1f * fps)) - 1, show = static_cast<int>(std::lround((0.1f + std::min(e.show, life)) * fps));
    Rgba first;
    long allocs = 0;
    int changed = 0;
    nvfx_fx a = 0, b = 0;
    std::string why;
    for (int f = 0; f <= last && why.empty(); ++f) {
      if (f == 2) g_counting.store(true);
      const long a0 = g_allocations.load();
      if (f == 0) {
        if (nvfx_stage_spawn(st.stage, e.name, 240.f, y, 1.f, 0.1, &a) != NVFX_OK) why = "spawn failed";
        if (nvfx_stage_spawn(st.stage, e.name, 300.f, y, 0.7f, 0.4, &b) != NVFX_OK) why = "a second spawn failed";
      } else {
        if (moving(e)) {  // a moving effect moves (its trail, tiles taken and given back)
          const auto p = path(e, 240.f, y, static_cast<float>(f) / fps - 0.1f);
          nvfx_stage_move(st.stage, a, p[0], p[1]);
        }
        nvfx_scene_step_frames(st.scene, 1);
      }
      if (f == static_cast<int>(std::lround((0.1f + 0.5f * life) * fps))) {  // every parameter to its ends, the second moved
        for (int p = 0; p < e.n_params; ++p) {
          nvfx_library_param pr;
          nvfx_library_get_param(lib, i, p, &pr);
          if (nvfx_stage_set_param(st.stage, a, pr.name, pr.max) != NVFX_OK || nvfx_stage_set_param(st.stage, b, pr.name, pr.min) != NVFX_OK) why = "set_param failed";
        }
        if (nvfx_stage_move(st.stage, b, 200.f, y) != NVFX_OK) why = "move failed";
      }
      if (f == static_cast<int>(std::lround((0.1f + life) * fps))) {  // stopped, and spawned again in a room it frees
        nvfx_stage_fx_info info{};
        nvfx_stage_get_fx(st.stage, a, &info);
        if (info.state != NVFX_FX_PLAYING && e.looping) why = std::format("the first instance is not playing ({})", static_cast<int>(info.state));
        nvfx_stage_stop(st.stage, a, -1.0);
        nvfx_fx c = 0;
        if (nvfx_stage_spawn(st.stage, e.name, 260.f, y, 1.2f, -1.0, &c) != NVFX_OK) why = "a spawn in a room in use failed (steal)";
      }
      st.render();
      allocs += g_allocations.load() - a0;
      g_counting.store(false);
      if (f == before) first = st.frame;
      if (f == show) {  // the effect draws something: pixels changed since before its spawn
        for (std::size_t p = 0; p < first.px.size(); p += 4) {
          int d = 0;
          for (int c = 0; c < 3; ++c) d = std::max(d, std::abs(static_cast<int>(st.frame.px[p + static_cast<std::size_t>(c)]) - static_cast<int>(first.px[p + static_cast<std::size_t>(c)])));
          changed += d > 8 ? 1 : 0;
        }
      }
    }
    const double share = 100.0 * changed / (480.0 * 360.0);
    if (why.empty() && allocs != 0) why = std::format("{} allocations in the frame loop", allocs);
    if (why.empty() && share < 0.05) why = std::format("it draws almost nothing ({:.3f}% of the pixels changed)", share);
    if (why.empty()) {
      std::println("  {:<20} plays: {} frames, 0 allocations, {:.2f}% of the pixels changed at {:.1f} s", e.name, last + 1, share, std::min(e.show, life));
      ++played;
    } else {
      std::println("  {:<20} FAILED: {}", e.name, why);
      ++failures;
    }
    std::fflush(stdout);
  }
  std::println("check: {} effects, {} played, {} parsed and checked only, {} failed", nvfx_library_count(lib), played, unplayed, failures);
  return failures == 0 ? 0 : 1;
}

// --- a shot, as a game plays it ------------------------------------------------------------------------------------

int shot(const Args& A, nvfx_library* lib) {
  std::vector<nvfx_stage_room> rooms;
  for (const Spawn& s : A.spawns) {
    auto it = std::ranges::find_if(rooms, [&](const nvfx_stage_room& r) { return s.name == r.effect; });
    if (it == rooms.end()) rooms.push_back({s.name.c_str(), 1, 0});
    else ++it->count;
  }
  for (const Spawn& s : A.spawns) {
    const int i = nvfx_library_find(lib, s.name.c_str());
    if (i < 0) die("no effect '" + s.name + "'");
    if (const auto m = missing(lib, i, A.models); !m.empty()) {
      std::println("nvfx_gallery: {} not in {}: skipped", m.front(), A.models.string());
      return 77;
    }
  }
  Stage st;
  make_stage(st, lib, A.shot_w, A.shot_h, A.ground, rooms, A.threads);
  if (A.print_stage) {
    std::string text(nvfx_stage_script(st.stage, nullptr, 0) + 1, '\0');
    nvfx_stage_script(st.stage, text.data(), text.size());
    std::print("{}", text.c_str());
  }
  st.render();  // frame 0, as an engine draws it when the stage is made
  std::vector<nvfx_fx> fxs;
  for (const Spawn& s : A.spawns) {
    nvfx_fx fx = 0;
    if (nvfx_stage_spawn(st.stage, s.name.c_str(), static_cast<float>(s.x), static_cast<float>(s.y), static_cast<float>(s.scale), s.at, &fx) != NVFX_OK) {
      die("spawn of " + s.name + " failed");
    }
    fxs.push_back(fx);
  }
  for (int f = 1; f <= A.frames; ++f) {
    const double t = static_cast<double>(f) / static_cast<double>(st.info.fps);  // the time of the frame about to be computed
    for (std::size_t k = 0; k < A.spawns.size(); ++k) {
      const Spawn& s = A.spawns[k];
      if (s.vx == 0.0 && s.vy == 0.0) continue;
      const double since = std::max(0.0, t - std::max(0.0, s.at));
      nvfx_stage_move(st.stage, fxs[k], static_cast<float>(s.x + s.vx * since), static_cast<float>(s.y + s.vy * since));
    }
    nvfx_scene_step_frames(st.scene, 1);
    st.render();
  }
  if (!A.png.empty() && !write(A.png, st.frame)) return 1;
  std::println("shot: {} x {}, {} frames, {} spawns{}", A.shot_w, A.shot_h, A.frames, A.spawns.size(), A.png.empty() ? "" : ", " + A.png.string());
  return 0;
}

// --- listing -------------------------------------------------------------------------------------------------------

int list(const Args& A, nvfx_library* lib) {
  if (A.markdown) std::println("| effect | category | plays | size (world px), anchor | duration | parameters | what it is |\n|---|---|---|---|---|---|---|");
  for (int c = 0; c < nvfx_library_category_count(lib); ++c) {
    const std::string cat = nvfx_library_category(lib, c);
    for (int i = 0; i < nvfx_library_count(lib); ++i) {
      nvfx_library_effect e;
      nvfx_library_get_effect(lib, i, &e);
      if (e.category != cat) continue;
      std::string params;
      for (int p = 0; p < e.n_params; ++p) {
        nvfx_library_param pr;
        nvfx_library_get_param(lib, i, p, &pr);
        params += std::format("{}{} ({:g})", params.empty() ? "" : ", ", pr.name, pr.value);
      }
      const std::string plays = std::string(e.models) + (e.models[0] && e.elements[0] ? "; " : "") + e.elements;
      const std::string life = e.looping ? "loop" : std::format("once, {:g} s", e.duration);
      const std::string size = std::format("{:g} x {:g}, {}", e.width, e.height, e.anchor == NVFX_ANCHOR_BASE ? "base" : e.anchor == NVFX_ANCHOR_LEFT ? "left" : "centre");
      if (A.markdown) std::println("| `{}` | {} | {} | {} | {} | {} | {} |", e.name, e.category, plays, size, life, params.empty() ? "-" : params, e.description);
      else std::println("{:<20} {:<10} {:<34} {:<20} {:<12} {}", e.name, e.category, plays, size, life, params);
    }
  }
  return 0;
}

}  // namespace

// The start points of a learned effect's file, with their controls: what `start N` in an entry picks (choose one whose
// controls are near the ones the entry plays at).
int starts(const Args& A) {
  const nfx::rt::RolloutEffect e = nfx::compose::script::load_from(A.starts.parent_path())(A.starts.filename().string());
  const auto& m = e.m;
  std::string names;
  for (const std::string& n : m.control_names) names += (names.empty() ? "" : ", ") + n;
  std::println("{}: {} start points; controls ({})", A.starts.string(), m.starts.size(), names);
  for (std::size_t i = 0; i < m.starts.size(); ++i) {
    std::string c;
    for (const float v : m.starts[i].controls) c += std::format("{}{:.2f}", c.empty() ? "" : ", ", v);
    std::println("  start {:>2}: at {:.2f} s, controls ({})", i, m.starts[i].time, c);
  }
  return 0;
}

int main(int argc, char** argv) try {
  const Args A = parse(argc, argv);
  if (!A.starts.empty()) return starts(A);
  if (!A.print.empty()) {  // the stage of an effect alone, as the library's C++ side writes it
    const nfx::library::Catalogue c = nfx::library::load_dir(A.library);
    const int i = c.index(A.print);
    if (i < 0) die("no effect '" + A.print + "'");
    nfx::library::StageSpec spec;
    std::print("{}", nfx::library::single(c, i, spec).text());
    return 0;
  }
  if (!A.pack.empty() && !A.verify.empty() && !fs::exists(A.pack)) {
    std::println("nvfx_gallery: {} is not there: skipped", A.pack.string());
    return 77;
  }
  nvfx_library* lib = open(A);
  int status = 0;
  if (A.list) status = list(A, lib);
  else if (A.check) status = check(A, lib);
  else if (!A.spawns.empty() || A.shot_w > 0) status = shot(A, lib);
  else status = gallery(A, lib);
  nvfx_library_free(lib);
  return status;
} catch (const std::exception& e) {
  std::println(stderr, "nvfx_gallery: {}", e.what());
  return 1;
}
