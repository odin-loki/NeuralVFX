// nvfx_viewer: play neural effects live with sliders for every control, and edit scene scripts live (docs/VIEWER.md).
//
//   nvfx_viewer effect.nvfx [more.nvfx ...] [--clip reference.nfxclip] [--size 128] [--scale 3]
//               [--screenshot out.png --frames 30]   [--demo]
//   nvfx_viewer --scene script.nvfxs [--effects DIR] [--stand-ins] [--threads 2] [--at SECONDS] [--paused]
//               [--screenshot out.png --frames 30 [--check-picture]] [--edit-replace FROM --with TO [--expect-error-line N]]
//               [--input NAME=V,...] [--trigger RULE,...] [--pin X,Y] [--overlay heat|soot] [--window WxH]
//
// Two modes, switched in the menu bar.
//
// Effects: side by side, the neural effect (rendered by the runtime, exactly as a game would), and optionally the
// reference clip it was trained on, a BC3 flipbook of that clip at about the same memory, and the fluid simulation
// running live at the same controls, each with its cost per frame. --demo builds a small untrained effect in memory so
// the viewer can be tried (and tested) without a model file.
//
// Scene: a scene script (docs/COMPOSE.md §4) in an editor next to the scene it plays through the scene C API, rebuilt
// as it is edited (scene_ui.hpp, scene_session.hpp). --stand-ins plays untrained stand-ins for every effect the script
// names, so the mode can be tried (and tested) without the models.
//
// Headless tests: --screenshot saves the window after --frames frames (in the scene mode, frames counted once the scene
// is built); --check-picture fails (exit 3) if the scene's picture on the screenshot is flat; --edit-replace makes an
// edit once the scene is built (the first FROM in the script becomes TO), and --expect-error-line fails unless that
// edit leaves an error at line N with the first scene still playing; --input and --trigger set inputs and fire rules
// once the scene is built, as the sliders and buttons do; --pin pins the field probe.
#include "../tools/args.hpp"
#include "scene_ui.hpp"
#include "texture.hpp"

#include <neuralfx/flipbook.hpp>
#include <neuralfx/image_io.hpp>
#include <neuralfx/model.hpp>
#include <neuralfx/nvfx.h>
#include <neuralfx/sim.hpp>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <memory>
#include <numeric>
#include <optional>
#include <print>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace nfx;
using viewer::Texture;

namespace {

using clock_type = std::chrono::steady_clock;

double ms_since(clock_type::time_point t0) { return std::chrono::duration<double, std::milli>(clock_type::now() - t0).count(); }

// A rolling mean of the last 60 timings.
class Rolling {
 public:
  void add(double v) {
    v_.push_back(v);
    if (v_.size() > 60) v_.pop_front();
  }
  double mean() const { return v_.empty() ? 0.0 : std::accumulate(v_.begin(), v_.end(), 0.0) / static_cast<double>(v_.size()); }

 private:
  std::deque<double> v_;
};

struct Effect {
  std::string name;
  nvfx_effect* fx = nullptr;
  nvfx_effect_info info{};
};

// A tiny untrained effect, so the viewer runs without a model file.
nvfx_effect* demo_effect() {
  Hyper h;
  h.arch = Arch::grid;
  h.size = 128;
  h.frames = 64;
  h.n_controls = 3;
  h.n_latent = 2;
  h.bases = 2;
  h.grid = 16;
  h.channels = 8;
  h.hidden = 16;
  h.grid_t = 8;
  Model m = init_model(h, 7);
  for (float& v : m.features) v *= 8.f;
  for (float& b : m.layers.back().b) b = 0.25f;
  m.effect = "demo (untrained)";
  m.control_names = {"intensity", "wind", "turbulence"};
  m.z_train = {{0.3f, -0.3f}, {-0.3f, 0.3f}};
  m.z_mean = {0.f, 0.f};
  m.z_std = {0.3f, 0.3f};
  std::ostringstream os;
  save_model(os, m);
  const std::string b = os.str();
  nvfx_effect* e = nullptr;
  nvfx_effect_load_memory(b.data(), b.size(), &e);
  return e;
}

// The scene mode's effects folder by default: $NEURALVFX_DATA (else ~/nvfx-data) /v2/models, else /experiments/models/d.
std::string default_effects_dir() {
  std::vector<std::filesystem::path> roots;
  if (const char* d = std::getenv("NEURALVFX_DATA"); d && *d) roots.emplace_back(d);
  if (const char* h = std::getenv("HOME"); h && *h) roots.emplace_back(std::filesystem::path(h) / "nvfx-data");
  for (const auto& root : roots) {
    for (const char* sub : {"v2/models", "experiments/models/d"}) {
      std::error_code ec;
      if (std::filesystem::is_directory(root / sub, ec)) return (root / sub).string();
    }
  }
  return roots.empty() ? std::string("models") : (roots.front() / "v2/models").string();
}

enum class Mode { effects, scene };

// The mean and spread of the luma inside a rectangle of an RGBA image: a flat (empty) picture has no spread.
std::pair<double, double> luma_stats(const Image& img, int x0, int y0, int x1, int y1) {
  x0 = std::clamp(x0, 0, img.width), x1 = std::clamp(x1, 0, img.width), y0 = std::clamp(y0, 0, img.height), y1 = std::clamp(y1, 0, img.height);
  double sum = 0.0, sum2 = 0.0;
  long n = 0;
  for (int y = y0; y < y1; ++y) {
    for (int x = x0; x < x1; ++x) {
      const std::uint8_t* p = img.rgba.data() + (static_cast<std::size_t>(y) * static_cast<std::size_t>(img.width) + static_cast<std::size_t>(x)) * 4;
      const double l = 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
      sum += l;
      sum2 += l * l;
      ++n;
    }
  }
  if (n == 0) return {0.0, 0.0};
  const double mean = sum / static_cast<double>(n);
  return {mean, std::sqrt(std::max(0.0, sum2 / static_cast<double>(n) - mean * mean))};
}

}  // namespace

int main(int argc, char** argv) try {
  const tools::Args a(argc, argv, {"demo", "help", "stand-ins", "paused", "check-picture"});
  if (a.flag("help")) {
    std::println("nvfx_viewer effect.nvfx [...] [--clip ref.nfxclip] [--size 128] [--scale 3] [--screenshot out.png --frames 30] [--demo]");
    std::println("nvfx_viewer --scene script.nvfxs [--effects DIR] [--stand-ins] [--threads 2] [--at SECONDS] [--paused]");
    std::println("            [--screenshot out.png --frames 30 [--check-picture]] [--edit-replace FROM --with TO [--expect-error-line N]]");
    std::println("            [--input NAME=V,...] [--trigger RULE,...] [--pin X,Y (the field probe, picture pixels)] [--overlay heat|soot] [--window WxH]");
    return 0;
  }
  std::vector<Effect> effects;
  for (const auto& p : a.positional()) {
    Effect e{std::filesystem::path(p).stem().string()};
    if (const auto s = nvfx_effect_load(p.c_str(), &e.fx); s != NVFX_OK) {
      std::println(stderr, "{}: {}", p, nvfx_status_string(s));
      continue;
    }
    nvfx_effect_get_info(e.fx, &e.info);
    effects.push_back(e);
  }
  if (a.flag("demo") || effects.empty()) {
    Effect e{"demo"};
    e.fx = demo_effect();
    nvfx_effect_get_info(e.fx, &e.info);
    effects.push_back(e);
  }
  std::optional<Clip> ref;
  if (a.has("clip")) {
    auto c = read_clip(a.str("clip"));
    if (!c) throw std::runtime_error(c.error());
    ref = std::move(*c);
  }
  const int scale = a.i("scale", 3);
  const std::string screenshot = a.str("screenshot");
  const int shot_frames = a.i("frames", 30);
  Mode mode = a.has("scene") ? Mode::scene : Mode::effects;
  viewer::SceneUi::Options scene_opt;
  scene_opt.script_path = a.str("scene");
  scene_opt.settings.effects_dir = a.str("effects", default_effects_dir());
  scene_opt.settings.stand_ins = a.flag("stand-ins") ? viewer::StandIns::all : viewer::StandIns::off;
  scene_opt.settings.threads = a.i("threads", 2);
  scene_opt.start_at = static_cast<double>(a.f("at", 0.f));
  scene_opt.paused = a.flag("paused");
  const std::string edit_from = a.str("edit-replace"), edit_to = a.str("with");
  const int expect_error_line = a.i("expect-error-line", 0);
  const bool check_picture = a.flag("check-picture");
  const std::vector<float> pin = tools::parse_floats(a.str("pin"));
  std::vector<std::pair<std::string, float>> set_inputs;  // --input name=value,...
  for (std::string rest = a.str("input"); !rest.empty();) {
    const std::size_t comma = rest.find(','), eq = rest.find('=');
    const std::string item = rest.substr(0, comma);
    if (eq == std::string::npos || eq > item.size()) throw std::invalid_argument("--input name=value[,name=value]");
    set_inputs.emplace_back(item.substr(0, eq), std::stof(item.substr(eq + 1)));
    rest = comma == std::string::npos ? std::string() : rest.substr(comma + 1);
  }
  std::vector<std::string> triggers;  // --trigger name,...
  for (std::string rest = a.str("trigger"); !rest.empty();) {
    const std::size_t comma = rest.find(',');
    triggers.push_back(rest.substr(0, comma));
    rest = comma == std::string::npos ? std::string() : rest.substr(comma + 1);
  }
  int win_w = mode == Mode::scene ? 1600 : 1500, win_h = mode == Mode::scene ? 900 : 720;
  if (const std::string ws = a.str("window"); !ws.empty()) {
    const std::size_t xpos = ws.find('x');
    if (xpos == std::string::npos) throw std::invalid_argument("--window WxH");
    win_w = std::stoi(ws.substr(0, xpos));
    win_h = std::stoi(ws.substr(xpos + 1));
  }

  if (!glfwInit()) throw std::runtime_error("GLFW initialisation failed (no display?)");
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  GLFWwindow* window = glfwCreateWindow(win_w, win_h, "NeuralVFX viewer", nullptr, nullptr);
  if (!window) throw std::runtime_error("cannot create a window");
  glfwMakeContextCurrent(window);
  glfwSwapInterval(screenshot.empty() ? 1 : 0);
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::GetIO().IniFilename = nullptr;
  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 130");

  // State
  int current = 0, size = a.i("size", effects[0].info.arch == 2 ? effects[0].info.native_size : 128);
  nvfx_instance* inst = nullptr;
  float controls[8] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
  int seed = 1, variation = -1, bg_index = 0;
  float drift = 6.f, speed = 1.f, hue = 0.f, brightness = 1.f;
  bool paused = false, show_ref = ref.has_value(), show_fb = ref.has_value(), show_sim = false;
  double time = 0.0;
  Rolling effect_ms, sim_ms;
  std::vector<std::uint8_t> frame, shown;
  Texture tex_effect, tex_ref, tex_fb, tex_sim;
  std::optional<Clip> fb_clip;
  std::unique_ptr<sim::Fluid> fluid;
  std::vector<std::uint8_t> sim_frame;
  float sim_controls[3] = {-1, -1, -1};
  std::unique_ptr<viewer::SceneUi> scene_ui;  // made when the scene mode is first shown (it starts a worker thread)
  if (mode == Mode::scene) {
    scene_ui = std::make_unique<viewer::SceneUi>(scene_opt);
    if (pin.size() == 2) scene_ui->pin(pin[0], pin[1]);
    scene_ui->set_overlay(a.str("overlay") == "heat" ? 1 : a.str("overlay") == "soot" ? 2 : 0);
  }

  const auto rebuild = [&] {
    if (inst) nvfx_instance_free(inst);
    inst = nullptr;
    const auto& info = effects[static_cast<std::size_t>(current)].info;
    if (info.arch == 2 && size != info.native_size && size * 2 != info.native_size && size * 4 != info.native_size) size = info.native_size;
    if (info.arch == 3) size = std::max(32, size / 32 * 32);  // rollout effects: a multiple of the 32-cell coarse grid
    if (nvfx_instance_create(effects[static_cast<std::size_t>(current)].fx, size, &inst) != NVFX_OK) throw std::runtime_error("instance failed");
    frame.assign(static_cast<std::size_t>(size) * size * 4, 0);
    shown.assign(frame.size(), 0);
    if (ref) {  // a BC3 flipbook of the reference clip at about the effect's stored size, for comparison
      const double frames_fit = static_cast<double>(info.stored_bytes) / (static_cast<double>(ref->size) * ref->size);
      const int kept = std::clamp(static_cast<int>(std::floor(frames_fit)), 2, ref->frames);
      fb_clip = flipbook::play(flipbook::build(*ref, {kept, ref->size, flipbook::Codec::bc3, 0}));
    }
  };
  rebuild();
  const Background bgs[] = {Background::black, Background::grey, Background::checker};
  const char* bg_names[] = {"black", "grey", "checker"};

  // The screenshot in the scene mode: wait for the scene, make the test's edit and wait for it, then count frames.
  enum class Shot { wait_scene, edit, wait_edit, count } shot = Shot::wait_scene;
  int shot_count = 0, first_frame = 0;
  std::uint64_t first_generation = 0;
  const auto shot_deadline = clock_type::now() + std::chrono::minutes(15);
  auto last_tick = clock_type::now();

  for (int frame_no = 0; !glfwWindowShouldClose(window); ++frame_no) {
    glfwPollEvents();
    const auto tick = clock_type::now();
    const double real_dt = std::min(0.1, std::chrono::duration<double>(tick - last_tick).count());
    last_tick = tick;
    const double dt = 1.0 / 60.0;
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    float top = 0.f;
    if (ImGui::BeginMainMenuBar()) {
      if (ImGui::MenuItem("Effects", nullptr, mode == Mode::effects)) mode = Mode::effects;
      if (ImGui::MenuItem("Scene", nullptr, mode == Mode::scene)) {
        if (mode != Mode::scene) {
          int ww = 0, wh = 0;
          glfwGetWindowSize(window, &ww, &wh);
          if (ww < 1600 || wh < 900) glfwSetWindowSize(window, std::max(ww, 1600), std::max(wh, 900));
        }
        mode = Mode::scene;
        if (!scene_ui) scene_ui = std::make_unique<viewer::SceneUi>(scene_opt);
      }
      ImGui::TextDisabled("  %s", mode == Mode::effects ? "effects: one effect, its controls and its cost" : "scene: a script edited live");
      top = ImGui::GetWindowSize().y;
      ImGui::EndMainMenuBar();
    }

    if (mode == Mode::scene) {
      const ImVec2 ds = ImGui::GetIO().DisplaySize;
      // the test's steps (Shot) play at a fixed 60 Hz; otherwise in real time
      const bool shooting = !screenshot.empty();
      scene_ui->frame(6.f, top + 6.f, ds.x - 12.f, ds.y - top - 12.f, shooting ? (shot == Shot::count ? dt : 0.0) : real_dt);
    } else {
      if (!paused) time += dt * speed;
      const auto& info = effects[static_cast<std::size_t>(current)].info;
      nvfx_instance_set_controls(inst, controls, info.n_controls);
      nvfx_instance_set_seed(inst, static_cast<std::uint64_t>(seed));
      nvfx_instance_set_variation(inst, variation);
      nvfx_instance_set_drift(inst, drift);
      nvfx_instance_set_colour(inst, hue, brightness);
      const auto t0 = clock_type::now();
      nvfx_render(inst, time, frame.data(), static_cast<std::size_t>(size) * 4);
      effect_ms.add(ms_since(t0));
      composite(frame, size, bgs[bg_index], shown);
      tex_effect.upload(shown, size);
      const int clip_frame = ref ? static_cast<int>(std::floor(time * ref->fps)) % ref->frames : 0;
      if (ref && show_ref) {
        std::vector<std::uint8_t> s(ref->frame_bytes());
        composite(ref->frame(clip_frame), ref->size, bgs[bg_index], s);
        tex_ref.upload(s, ref->size);
      }
      if (fb_clip && show_fb) {
        std::vector<std::uint8_t> s(fb_clip->frame_bytes());
        composite(fb_clip->frame(clip_frame), fb_clip->size, bgs[bg_index], s);
        tex_fb.upload(s, fb_clip->size);
      }
      if (show_sim) {  // the simulation itself, restarted when its controls change
        sim::Effect e{};
        const bool known = sim::parse_effect(info.name, e);
        if (!fluid || sim_controls[0] != controls[0] || sim_controls[1] != controls[1] || sim_controls[2] != controls[2]) {
          sim::Params p;
          p.effect = known ? e : sim::Effect::fire;
          p.intensity = controls[0];
          p.wind = controls[1];
          p.turbulence = controls[2];
          p.size = 128;
          p.seed = static_cast<std::uint64_t>(seed);
          fluid = std::make_unique<sim::Fluid>(p);
          std::copy_n(controls, 3, sim_controls);
          sim_frame.assign(128 * 128 * 4, 0);
        }
        const auto s0 = clock_type::now();
        fluid->step_frame();
        fluid->render(sim_frame);
        sim_ms.add(ms_since(s0));
        std::vector<std::uint8_t> s(sim_frame.size());
        composite(sim_frame, 128, bgs[bg_index], s);
        tex_sim.upload(s, 128);
      }

      ImGui::SetNextWindowPos({10, top + 6.f}, ImGuiCond_Always);
      ImGui::SetNextWindowSize({380, ImGui::GetIO().DisplaySize.y - top - 12.f}, ImGuiCond_Always);
      ImGui::Begin("Effect", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize);
      if (ImGui::BeginCombo("effect", effects[static_cast<std::size_t>(current)].name.c_str())) {
        for (int i = 0; i < static_cast<int>(effects.size()); ++i) {
          if (ImGui::Selectable(effects[static_cast<std::size_t>(i)].name.c_str(), i == current)) {
            current = i;
            variation = -1;
            rebuild();
          }
        }
        ImGui::EndCombo();
      }
      const bool rollout = info.arch == 3;
      ImGui::Text("%s%s, %d controls, %d %s", rollout ? "rollout, " : "", info.loops ? "looping" : "one-shot", info.n_controls, info.n_variations,
                  rollout ? "start points" : "variations");
      ImGui::SeparatorText("learned controls");
      for (int i = 0; i < info.n_controls; ++i) {
        const char* name = nvfx_effect_control_name(effects[static_cast<std::size_t>(current)].fx, i);
        ImGui::SliderFloat(name && *name ? name : "control", &controls[i], 0.f, 1.f);
      }
      ImGui::SeparatorText("variation");
      ImGui::InputInt("seed", &seed);
      if (ImGui::Button("new seed")) seed = static_cast<int>(clock_type::now().time_since_epoch().count() & 0x7fffffff);
      ImGui::SliderInt(rollout ? "start point" : "training variation", &variation, -1, std::max(-1, info.n_variations - 1), variation < 0 ? "seeded" : "%d");
      ImGui::SliderFloat(rollout ? "shard (s)" : "drift (s)", &drift, 0.f, 30.f);  // rollout: shard length, 0 = one run
      ImGui::SeparatorText("exact runtime controls");
      ImGui::SliderFloat("speed", &speed, 0.f, 3.f);
      ImGui::SliderAngle("hue", &hue, -180.f, 180.f);
      ImGui::SliderFloat("brightness", &brightness, 0.f, 2.f);
      ImGui::SeparatorText("view");
      int sz = size;
      if (ImGui::InputInt("size", &sz, rollout ? 32 : 16, 64) && sz != size) {
        size = rollout ? std::clamp(sz / 32 * 32, 32, 512) : std::clamp(sz / 16 * 16, 16, 512);
        rebuild();
      }
      ImGui::Combo("background", &bg_index, bg_names, 3);
      ImGui::Checkbox("paused", &paused);
      ImGui::SameLine();
      if (ImGui::Button("restart")) time = 0;
      if (ref) {
        ImGui::Checkbox("reference clip", &show_ref);
        ImGui::SameLine();
        ImGui::Checkbox("flipbook", &show_fb);
      }
      ImGui::Checkbox("live simulation", &show_sim);
      ImGui::SeparatorText("cost (this machine, this thread)");
      ImGui::Text("neural: %.3f ms per %dx%d frame", effect_ms.mean(), size, size);
      ImGui::Text("%.0f MAC/px, ISA %d", nvfx_instance_macs_per_pixel(inst), static_cast<int>(nvfx_get_isa()));
      ImGui::Text("weights: %.1f KB stored, %.1f KB resident", static_cast<double>(info.stored_bytes) / 1024.0, static_cast<double>(info.resident_bytes) / 1024.0);
      ImGui::Text("scratch: %.1f KB per instance", static_cast<double>(nvfx_instance_scratch_bytes(inst)) / 1024.0);
      if (show_sim) ImGui::Text("simulation: %.3f ms per 128x128 frame", sim_ms.mean());
      if (fb_clip) ImGui::Text("flipbook: BC3, same memory class as the effect");
      ImGui::End();

      ImGui::SetNextWindowPos({400, top + 6.f}, ImGuiCond_Always);
      ImGui::SetNextWindowSize({ImGui::GetIO().DisplaySize.x - 410.f, ImGui::GetIO().DisplaySize.y - top - 12.f}, ImGuiCond_Always);
      ImGui::Begin("View", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize);
      const float px = 128.f * static_cast<float>(scale) * 0.8f;
      const auto panel = [&](const char* title, const Texture& t) {
        ImGui::BeginGroup();
        ImGui::TextUnformatted(title);
        ImGui::Image(t.id(), {px, px});
        ImGui::EndGroup();
        ImGui::SameLine();
      };
      panel("neural (runtime)", tex_effect);
      if (ref && show_ref) panel("reference clip", tex_ref);
      if (fb_clip && show_fb) panel("flipbook (BC3)", tex_fb);
      if (show_sim) panel("simulation (live)", tex_sim);
      ImGui::NewLine();
      ImGui::Text("time %.2f s", time);
      ImGui::End();
    }

    ImGui::Render();
    int w, h;
    glfwGetFramebufferSize(window, &w, &h);
    glViewport(0, 0, w, h);
    glClearColor(0.08f, 0.08f, 0.09f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    bool take = false;
    if (!screenshot.empty() && mode == Mode::scene) {
      viewer::SceneSession& ss = scene_ui->session();
      const bool settled = !ss.busy() && !ss.edit_pending();
      if (clock_type::now() > shot_deadline) throw std::runtime_error("the scene did not settle in 15 minutes");
      switch (shot) {
        case Shot::wait_scene:
          if (settled && !ss.has_scene()) throw std::runtime_error("no scene: " + (ss.error() ? ss.error()->message : std::string("?")));
          if (settled) {
            first_generation = ss.generation();
            for (const auto& [name, value] : set_inputs) {  // --input, --trigger: as the sliders and buttons would
              bool found = false;
              for (viewer::InputValue& in : ss.inputs()) {
                if (in.name == name) {
                  in.value = value;
                  found = true;
                }
              }
              if (!found) throw std::runtime_error("--input: the script has no input '" + name + "'");
            }
            for (const std::string& r : triggers)
              if (!ss.trigger(r)) throw std::runtime_error("--trigger: '" + r + "' cannot be fired");
            if (!edit_from.empty()) {
              shot = Shot::edit;
            } else {
              shot = Shot::count;
              first_frame = ss.frame();
            }
          }
          break;
        case Shot::edit:
          if (!scene_ui->replace_text(edit_from, edit_to)) throw std::runtime_error("--edit-replace: '" + edit_from + "' is not in the script");
          shot = Shot::wait_edit;
          break;
        case Shot::wait_edit:
          if (settled) {
            shot = Shot::count;
            first_frame = ss.frame();
          }
          break;
        case Shot::count:
          take = ++shot_count >= shot_frames;
          break;
      }
      if (shot != Shot::count) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    } else if (!screenshot.empty()) {
      take = frame_no + 1 >= shot_frames;
    }
    if (take) {
      Image img;
      img.allocate(w, h);
      std::vector<std::uint8_t> raw(img.rgba.size());
      glPixelStorei(GL_PACK_ALIGNMENT, 1);
      glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, raw.data());
      for (int y = 0; y < h; ++y) {  // OpenGL rows are bottom-up
        std::copy_n(raw.begin() + static_cast<std::ptrdiff_t>(y) * w * 4, w * 4, img.rgba.begin() + static_cast<std::ptrdiff_t>(h - 1 - y) * w * 4);
      }
      if (auto r = write_png(screenshot, img); !r) throw std::runtime_error(r.error());
      if (mode != Mode::scene) {
        std::println("screenshot {} ({}x{}) after {} frames; neural {:.3f} ms per frame", screenshot, w, h, frame_no + 1, effect_ms.mean());
        break;
      }
      // the scene mode: what the test checks
      viewer::SceneSession& ss = scene_ui->session();
      bool ok = true;
      ImVec2 p0, p1;
      int ww = 1, wh = 1;
      glfwGetWindowSize(window, &ww, &wh);
      const float fx = static_cast<float>(w) / static_cast<float>(std::max(1, ww)), fy = static_cast<float>(h) / static_cast<float>(std::max(1, wh));
      const bool drawn = scene_ui->picture_rect(p0, p1);
      const auto [mean, spread] = drawn ? luma_stats(img, static_cast<int>(p0.x * fx), static_cast<int>(p0.y * fy), static_cast<int>(p1.x * fx), static_cast<int>(p1.y * fy))
                                        : std::pair{0.0, 0.0};
      std::println("screenshot {} ({}x{}): scene {} at {:.2f} s (frame {}, {} frames played), built in {:.0f} ms; picture luma mean {:.1f}, spread {:.1f}",
                   screenshot, w, h, ss.source_name(), ss.time(), ss.frame(), ss.frame() - first_frame, ss.build_ms(), mean, spread);
      if (ss.error()) std::println("error shown: {}", ss.error()->message);
      if (check_picture && (!drawn || spread < 4.0 || mean < 2.0)) {
        std::println(stderr, "nvfx_viewer: the scene's picture is empty or flat");
        ok = false;
      }
      if (expect_error_line > 0) {
        if (!ss.error() || ss.error()->line != expect_error_line) {
          std::println(stderr, "nvfx_viewer: expected an error at line {}", expect_error_line);
          ok = false;
        }
        if (!ss.scene() || ss.generation() != first_generation) {
          std::println(stderr, "nvfx_viewer: the last good scene should still be playing");
          ok = false;
        }
        if (!scene_opt.paused && ss.frame() == first_frame) {
          std::println(stderr, "nvfx_viewer: the last good scene did not play on");
          ok = false;
        }
      } else if (check_picture && ss.error()) {
        ok = false;
      }
      if (!ok) {
        scene_ui.reset();
        nvfx_instance_free(inst);
        for (auto& e : effects) nvfx_effect_free(e.fx);
        return 3;
      }
      break;
    }
    glfwSwapBuffers(window);
  }
  scene_ui.reset();
  nvfx_instance_free(inst);
  for (auto& e : effects) nvfx_effect_free(e.fx);
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
} catch (const std::exception& e) {
  std::println(stderr, "nvfx_viewer: {}", e.what());
  return 2;
}
