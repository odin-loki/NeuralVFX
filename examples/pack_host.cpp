// nvfx_pack_host: an engine's start and frame loop with the C++ wrapper (include/neuralfx/nvfx.hpp,
// docs/INTEGRATION.md): one effect pack, the effect library on a stage and the effect manager.
//
//   nvfx_pack_host PACK [--out sheet.png] [--seconds 3] [--threads 2]
//
// Checks the library against the headers, opens the pack (every checksum checked), opens its catalogue as a library,
// spawns a few library effects on a stage and plays it, then plays a few copies of one model with the effect manager
// (its costs from the pack if it has them, else measured here), and writes a PNG sheet: six frames of the stage, then
// the manager's sprites blended (premultiplied) over a dark ground. Exit 77 when the pack is not there.
#include <neuralfx/nvfx.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

// --- a PNG writer without dependencies (stored deflate blocks: large files, no compression) ----------------------------

std::uint32_t crc32(const std::uint8_t* p, std::size_t n, std::uint32_t c = 0xffffffffu) {
  for (std::size_t i = 0; i < n; ++i) {
    c ^= p[i];
    for (int k = 0; k < 8; ++k) c = (c >> 1) ^ (0xedb88320u & (0u - (c & 1u)));
  }
  return c;
}

bool write_png(const std::string& path, int w, int h, const std::vector<std::uint8_t>& rgb) {
  std::vector<std::uint8_t> raw;  // filter byte 0 and the row
  for (int y = 0; y < h; ++y) {
    raw.push_back(0);
    raw.insert(raw.end(), rgb.begin() + static_cast<std::ptrdiff_t>(y) * w * 3, rgb.begin() + static_cast<std::ptrdiff_t>(y + 1) * w * 3);
  }
  std::vector<std::uint8_t> z = {0x78, 0x01};
  std::uint32_t a = 1, b = 0;
  for (std::size_t at = 0; at < raw.size() || at == 0;) {
    const std::size_t n = std::min<std::size_t>(65535, raw.size() - at);
    z.push_back(at + n == raw.size() ? 1 : 0);
    z.push_back(static_cast<std::uint8_t>(n));
    z.push_back(static_cast<std::uint8_t>(n >> 8));
    z.push_back(static_cast<std::uint8_t>(~n));
    z.push_back(static_cast<std::uint8_t>(~n >> 8));
    z.insert(z.end(), raw.begin() + static_cast<std::ptrdiff_t>(at), raw.begin() + static_cast<std::ptrdiff_t>(at + n));
    for (std::size_t i = at; i < at + n; ++i) {
      a = (a + raw[i]) % 65521u;
      b = (b + a) % 65521u;
    }
    at += n;
    if (n == 0) break;
  }
  for (const std::uint32_t v : {b >> 8, b, a >> 8, a}) z.push_back(static_cast<std::uint8_t>(v));
  std::ofstream f(path, std::ios::binary);
  const auto chunk = [&](const char* type, const std::vector<std::uint8_t>& data) {
    const std::uint32_t n = static_cast<std::uint32_t>(data.size());
    const std::uint8_t len[4] = {static_cast<std::uint8_t>(n >> 24), static_cast<std::uint8_t>(n >> 16), static_cast<std::uint8_t>(n >> 8), static_cast<std::uint8_t>(n)};
    f.write(reinterpret_cast<const char*>(len), 4);
    std::vector<std::uint8_t> body(type, type + 4);
    body.insert(body.end(), data.begin(), data.end());
    f.write(reinterpret_cast<const char*>(body.data()), static_cast<std::streamsize>(body.size()));
    const std::uint32_t c = ~crc32(body.data(), body.size());
    const std::uint8_t crc[4] = {static_cast<std::uint8_t>(c >> 24), static_cast<std::uint8_t>(c >> 16), static_cast<std::uint8_t>(c >> 8), static_cast<std::uint8_t>(c)};
    f.write(reinterpret_cast<const char*>(crc), 4);
  };
  f.write("\x89PNG\r\n\x1a\n", 8);
  const auto be = [](std::uint32_t v) { return std::vector<std::uint8_t>{static_cast<std::uint8_t>(v >> 24), static_cast<std::uint8_t>(v >> 16), static_cast<std::uint8_t>(v >> 8), static_cast<std::uint8_t>(v)}; };
  std::vector<std::uint8_t> ihdr = be(static_cast<std::uint32_t>(w));
  const auto hh = be(static_cast<std::uint32_t>(h));
  ihdr.insert(ihdr.end(), hh.begin(), hh.end());
  ihdr.insert(ihdr.end(), {8, 2, 0, 0, 0});  // 8 bits, RGB
  chunk("IHDR", ihdr);
  chunk("IDAT", z);
  chunk("IEND", {});
  return static_cast<bool>(f);
}

struct Sheet {
  int w, h;
  std::vector<std::uint8_t> rgb;
  Sheet(int width, int height) : w(width), h(height), rgb(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3, 16) {}
  // An opaque picture (a stage's) at (x0, y0), shrunk by `k` (box filter).
  void put(const std::uint8_t* rgba, int pw, int ph, int x0, int y0, int k) {
    for (int y = 0; y < ph / k; ++y)
      for (int x = 0; x < pw / k; ++x)
        for (int c = 0; c < 3; ++c) {
          int s = 0;
          for (int dy = 0; dy < k; ++dy)
            for (int dx = 0; dx < k; ++dx) s += rgba[(static_cast<std::size_t>(y * k + dy) * static_cast<std::size_t>(pw) + static_cast<std::size_t>(x * k + dx)) * 4 + static_cast<std::size_t>(c)];
          rgb[(static_cast<std::size_t>(y0 + y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x0 + x)) * 3 + static_cast<std::size_t>(c)] = static_cast<std::uint8_t>(s / (k * k));
        }
  }
  // A premultiplied sprite of `size` px stretched to cell x cell at (x0, y0): dst = src + dst * (1 - a).
  void blend(const nvfx::Picture& p, int x0, int y0, int cell) {
    for (int y = 0; y < cell; ++y)
      for (int x = 0; x < cell; ++x) {
        const std::uint8_t* s = p.rgba + static_cast<std::size_t>(y * p.size / cell) * p.stride + static_cast<std::size_t>(x * p.size / cell) * 4;
        std::uint8_t* d = &rgb[(static_cast<std::size_t>(y0 + y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x0 + x)) * 3];
        for (int c = 0; c < 3; ++c) d[c] = static_cast<std::uint8_t>(std::min(255, s[c] + d[c] * (255 - s[3]) / 255));
      }
  }
};

}  // namespace

int main(int argc, char** argv) {
  std::string pack_path, out = "pack_host_sheet.png";
  double seconds = 3.0;
  int threads = 2;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--out" && i + 1 < argc) out = argv[++i];
    else if (a == "--seconds" && i + 1 < argc) seconds = std::atof(argv[++i]);
    else if (a == "--threads" && i + 1 < argc) threads = std::atoi(argv[++i]);
    else if (pack_path.empty() && a[0] != '-') pack_path = a;
    else {
      std::fprintf(stderr, "nvfx_pack_host PACK [--out sheet.png] [--seconds 3] [--threads 2]\n");
      return 1;
    }
  }
  if (pack_path.empty() || !std::filesystem::exists(pack_path)) {
    std::printf("nvfx_pack_host: no pack (%s): skipped\n", pack_path.c_str());
    return 77;
  }

  // 1. The library is the one these headers describe.
  const nvfx::Version v = nvfx::version();
  if (!v.matches_headers()) {
    std::fprintf(stderr, "libnvfx is version %u, the headers %d\n", static_cast<unsigned>(v.version), NVFX_VERSION);
    return 1;
  }

  // 2. The pack: one file, every member checked.
  auto t0 = Clock::now();
  auto pack = nvfx::Pack::open(pack_path, threads);
  if (!pack) {
    std::fprintf(stderr, "%s\n", pack.error().message.c_str());
    return 1;
  }
  const nvfx_pack_info info = *pack->get_info();
  std::printf("%s: %d members (%d models, %d catalogue files), %zu bytes (%zu unpacked), opened in %.1f ms\n", pack_path.c_str(), info.n_members, info.n_models,
              info.n_catalogue, info.file_bytes, info.unpacked_bytes, ms_since(t0));

  // 3. The library from the pack, and a stage with room for a few of its effects.
  t0 = Clock::now();
  auto lib = nvfx::Library::open_pack(*pack);
  if (!lib) {
    std::fprintf(stderr, "%s\n", lib.error().message.c_str());
    return 1;
  }
  std::vector<std::string> names;
  for (const char* n : {"campfire", "steam_vent", "plasma_orb", "grenade"})
    if (lib->find(n) >= 0) names.emplace_back(n);
  for (int i = 0; names.size() < 3 && i < lib->count(); ++i) names.emplace_back(lib->get_effect(i)->name);
  nvfx::StageRooms rooms;
  for (const std::string& n : names) rooms.push_back({n.c_str(), 1, 1});
  nvfx_stage_desc sd = nvfx::stage_desc();
  sd.width = 640;
  sd.height = 360;
  sd.rooms = rooms.data();
  sd.n_rooms = static_cast<int>(rooms.size());
  sd.threads = threads;
  auto stage = nvfx::Stage::create(*lib, sd);
  if (!stage) {
    std::fprintf(stderr, "%s\n", stage.error().message.c_str());
    return 1;
  }
  std::printf("library: %d effects; stage of %zu rooms made in %.0f ms\n", lib->count(), rooms.size(), ms_since(t0));
  const float ground = 0.87f * 360.f;
  for (std::size_t i = 0; i < names.size(); ++i) {
    const nvfx_library_effect e = *lib->get_effect(lib->find(names[i]));
    const float x = 110.f + 140.f * static_cast<float>(i);
    const float y = e.anchor == NVFX_ANCHOR_BASE ? ground : ground - 0.5f * e.height * 0.5f;
    const auto fx = stage->spawn(names[i], x, y, 0.5f, e.looping ? 0.2 : 1.0);
    std::printf("  spawned %s at (%.0f, %.0f)%s\n", names[i].c_str(), x, y, fx ? "" : " FAILED");
  }

  // 4. The frame loop: step the scene by the game's dt, read fields for gameplay, draw.
  nvfx::SceneView scene = stage->scene();
  std::vector<std::uint8_t> frame(640 * 360 * 4);
  Sheet sheet(960, 360 + 160);
  const int frames = static_cast<int>(seconds * 60.0);
  double worst = 0, total = 0;
  int shot = 0;
  for (int f = 1; f <= frames; ++f) {
    const auto tf = Clock::now();
    scene.step(1.0 / 60.0);
    const auto heat = scene.sample(110.f, ground - 40.f);
    scene.render(frame);  // a new picture every other call at 60 Hz (the scene runs at 30 fps)
    const double ms = ms_since(tf);
    worst = std::max(worst, ms);
    total += ms;
    if (f * 6 / frames != (f - 1) * 6 / frames && shot < 6) {
      sheet.put(frame.data(), 640, 360, (shot % 3) * 320, (shot / 3) * 180, 2);
      ++shot;
    }
    if (f == frames && heat) std::printf("stage: %d frames, %.2f ms per call on average, %.2f worst; heat over the first effect %.2f\n", frames, total / frames, worst, heat->heat);
  }

  // 5. Many copies of one model with the effect manager: costs from the pack, or measured on this machine at start.
  const std::string model_name = pack->find("fire") >= 0 ? "fire" : pack->get_member(0)->name;
  auto model_fx = pack->load_effect(model_name);
  if (!model_fx) {
    std::fprintf(stderr, "%s\n", model_fx.error().message.c_str());
    return 1;
  }
  nvfx_manager_desc md = nvfx::manager_desc();
  md.threads = threads;
  md.budget_ms = 4.f;
  auto mgr = nvfx::Manager::create(md);
  if (!mgr) return 1;
  const auto model = mgr->add_model(*model_fx);  // the effect outlives the manager (declared first)
  if (!model) {
    std::fprintf(stderr, "%s\n", model.error().message.c_str());
    return 1;
  }
  const int from_pack = mgr->set_costs_from(*pack, *model, model_name).value_or(0);
  if (from_pack == 0) {
    t0 = Clock::now();
    mgr->calibrate(*model);
    std::printf("manager: calibrated in %.0f ms (step at 64 px %.3f ms)\n", ms_since(t0), mgr->get_cost(*model, 64)->step_ms);
  }
  std::vector<nvfx_handle> handles;
  const float sizes[6] = {160.f, 128.f, 96.f, 64.f, 48.f, 32.f};
  for (int i = 0; i < 6; ++i) {
    nvfx_spawn_desc s = nvfx::spawn_desc();
    s.model = *model;
    s.seed = 100 + static_cast<std::uint64_t>(i);
    s.screen_px = sizes[i];
    if (const auto h = mgr->spawn(s)) handles.push_back(*h);
  }
  for (int t = 0; t < static_cast<int>(seconds * 60.0); ++t) mgr->tick(1.0 / 60.0);
  const nvfx_manager_stats st = *mgr->get_stats();
  std::printf("manager: %d instances, %d simulations, last tick %.2f ms of jobs against %.2f planned, %.1f MB\n", st.instances, st.sims, st.jobs_ms, st.planned_ms,
              static_cast<double>(st.memory_bytes) / 1048576.0);
  for (std::size_t i = 0; i < handles.size(); ++i) {
    const auto p = mgr->get_picture(handles[i]);
    if (p && p->rgba) sheet.blend(*p, 8 + static_cast<int>(i) * 158, 360 + 8, 144);
  }

  if (!write_png(out, sheet.w, sheet.h, sheet.rgb)) {
    std::fprintf(stderr, "cannot write %s\n", out.c_str());
    return 1;
  }
  std::printf("sheet: %s\n", out.c_str());
  return 0;
}
