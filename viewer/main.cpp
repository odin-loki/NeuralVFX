// nvfx_viewer: play neural effects live with sliders for every control (docs/VIEWER.md).
//
//   nvfx_viewer effect.nvfx [more.nvfx ...] [--clip reference.nfxclip] [--size 128] [--scale 3]
//               [--screenshot out.png --frames 30]   [--demo]
//
// Side by side: the neural effect (rendered by the runtime, exactly as a game would), and optionally the reference
// clip it was trained on, a BC3 flipbook of that clip at about the same memory, and the fluid simulation running
// live at the same controls, each with its cost per frame. --demo builds a small untrained effect in memory so the
// viewer can be tried (and tested) without a model file.
#include "../tools/args.hpp"

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
#include <deque>
#include <filesystem>
#include <memory>
#include <numeric>
#include <optional>
#include <print>
#include <sstream>
#include <string>
#include <vector>

using namespace nfx;

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

class Texture {
 public:
  Texture() { glGenTextures(1, &id_); }
  ~Texture() { glDeleteTextures(1, &id_); }
  Texture(const Texture&) = delete;
  Texture& operator=(const Texture&) = delete;
  void upload(std::span<const std::uint8_t> rgba, int size) {
    glBindTexture(GL_TEXTURE_2D, id_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size, size, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
  }
  ImTextureID id() const { return static_cast<ImTextureID>(id_); }

 private:
  GLuint id_ = 0;
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

}  // namespace

int main(int argc, char** argv) try {
  const tools::Args a(argc, argv, {"demo", "help"});
  if (a.flag("help")) {
    std::println("nvfx_viewer effect.nvfx [...] [--clip ref.nfxclip] [--size 128] [--scale 3] [--screenshot out.png --frames 30] [--demo]");
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

  if (!glfwInit()) throw std::runtime_error("GLFW initialisation failed (no display?)");
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  GLFWwindow* window = glfwCreateWindow(1500, 720, "NeuralVFX viewer", nullptr, nullptr);
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
  float drift = 8.f, speed = 1.f, hue = 0.f, brightness = 1.f;
  bool paused = false, show_ref = ref.has_value(), show_fb = ref.has_value(), show_sim = false;
  double time = 0.0;
  Rolling effect_ms, sim_ms;
  std::vector<std::uint8_t> frame, shown;
  Texture tex_effect, tex_ref, tex_fb, tex_sim;
  std::optional<Clip> fb_clip;
  std::unique_ptr<sim::Fluid> fluid;
  std::vector<std::uint8_t> sim_frame;
  float sim_controls[3] = {-1, -1, -1};

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

  for (int frame_no = 0; !glfwWindowShouldClose(window); ++frame_no) {
    glfwPollEvents();
    const double dt = 1.0 / 60.0;
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

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGui::SetNextWindowPos({10, 10}, ImGuiCond_Always);
    ImGui::SetNextWindowSize({380, 700}, ImGuiCond_Always);
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
    if (!rollout) ImGui::SliderFloat("drift (s)", &drift, 0.f, 30.f);  // rollout effects never repeat anyway
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

    ImGui::SetNextWindowPos({400, 10}, ImGuiCond_Always);
    ImGui::SetNextWindowSize({1090, 700}, ImGuiCond_Always);
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

    ImGui::Render();
    int w, h;
    glfwGetFramebufferSize(window, &w, &h);
    glViewport(0, 0, w, h);
    glClearColor(0.08f, 0.08f, 0.09f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    if (!screenshot.empty() && frame_no + 1 >= shot_frames) {
      Image img;
      img.allocate(w, h);
      std::vector<std::uint8_t> raw(img.rgba.size());
      glPixelStorei(GL_PACK_ALIGNMENT, 1);
      glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, raw.data());
      for (int y = 0; y < h; ++y) {  // OpenGL rows are bottom-up
        std::copy_n(raw.begin() + static_cast<std::ptrdiff_t>(y) * w * 4, w * 4, img.rgba.begin() + static_cast<std::ptrdiff_t>(h - 1 - y) * w * 4);
      }
      if (auto r = write_png(screenshot, img); !r) throw std::runtime_error(r.error());
      std::println("screenshot {} ({}x{}) after {} frames; neural {:.3f} ms per frame", screenshot, w, h, frame_no + 1, effect_ms.mean());
      break;
    }
    glfwSwapBuffers(window);
  }
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
