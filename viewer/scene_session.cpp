// The viewer's scene mode apart from drawing: see scene_session.hpp.
#include "scene_session.hpp"

#include <neuralfx/rollout.hpp>

#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <sstream>
#include <utility>

namespace nfx::viewer {

namespace {

// The scene API's frame of a time (nvfx_scene_seek's rounding).
int frame_at(double seconds, float fps) {
  return static_cast<int>(std::clamp(std::floor(seconds * static_cast<double>(fps) + 1e-6), 0.0, 1e9));
}

double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

}  // namespace

std::string stand_in_bytes(std::string_view name) {
  std::uint64_t seed = 1469598103934665603ULL;  // FNV-1a of the name: the same stand-in for the same name
  for (const char c : name) {
    seed ^= static_cast<unsigned char>(c);
    seed *= 1099511628211ULL;
  }
  rollout::Hyper h;
  h.res = 32;  // the study D effects' grid, so module sizes and cell ranges written for them hold
  h.hidden = 8;
  h.memory = 2;
  h.jacobi = 12;
  h.render_hidden = 8;
  h.warmup = 4;
  rollout::Model m = rollout::init_model(h, seed);
  m.effect = std::string(name) + " (stand-in)";
  m.control_names = {"intensity", "wind", "turbulence"};
  m.scale = {0.2f, 0.2f, 0.4f, 0.3f};
  m.lo = {-2.f, -2.f, 0.f, 0.f};
  m.hi = {2.f, 2.f, 3.f, 3.f};
  m.detail.swirl_control = 2;
  const auto res = static_cast<std::size_t>(h.res);
  for (int k = 0; k < 12; ++k) {  // a plume over a hot base, larger from one start point to the next
    rollout::StartPoint sp;
    sp.controls = {0.3f + 0.05f * static_cast<float>(k), 0.5f, 0.5f};
    sp.seed = 100 + static_cast<std::uint64_t>(k);
    sp.coarse.resize(res * res * rollout::kPhys);
    const float spread = 2.5f + 0.5f * static_cast<float>(k);
    for (std::size_t y = 0; y < res; ++y) {  // rows from the bottom
      for (std::size_t x = 0; x < res; ++x) {
        const float dx = (static_cast<float>(x) - 15.5f) / spread, dy = static_cast<float>(y) / (2.f * spread);
        float* c = sp.coarse.data() + (y * res + x) * rollout::kPhys;
        c[0] = 0.04f * std::sin(0.5f * static_cast<float>(y) + static_cast<float>(k));
        c[1] = 0.3f * std::exp(-(dx * dx + (dy - 0.8f) * (dy - 0.8f)));
        c[2] = 1.6f * std::exp(-(dx * dx + dy * dy));
        c[3] = 0.9f * std::exp(-(0.6f * dx * dx + (dy - 1.2f) * (dy - 1.2f)));
      }
    }
    m.starts.push_back(std::move(sp));
  }
  std::ostringstream os;
  if (!rollout::save_model(os, m)) return {};
  return os.str();
}

SceneSession::SceneSession(SessionSettings s) : settings_(std::move(s)), debounce_(settings_.debounce) {
  worker_ = std::thread([this] { work(); });
}

SceneSession::~SceneSession() {
  {
    std::lock_guard lk(m_);
    stop_ = true;
  }
  cancel_ = true;
  cv_.notify_all();
  if (worker_.joinable()) worker_.join();
}

// --- the script ---------------------------------------------------------------------------------------------------

void SceneSession::edit(std::string text, Clock::time_point now) {
  if (text == text_) return;
  text_ = std::move(text);
  ++version_;
  debounce_.touch(now);
}

void SceneSession::open(std::string text, std::string source_name, double at) {
  text_ = std::move(text);
  source_ = source_name.empty() ? "script" : std::move(source_name);
  ++version_;
  debounce_.cancel();
  inputs_.clear();
  applied_.clear();
  check_and_build(std::max(0.0, at), false);
}

void SceneSession::set_settings(const SessionSettings& s) {
  const bool rebuild = s.effects_dir != settings_.effects_dir || s.stand_ins != settings_.stand_ins || s.threads != settings_.threads ||
                       s.overlap != settings_.overlap;
  settings_ = s;
  debounce_.set_delay(s.debounce);
  if (rebuild && !text_.empty()) {
    ++version_;
    check_and_build(time(), false);
  }
}

void SceneSession::reload_effects() {
  if (text_.empty()) return;
  ++version_;
  check_and_build(time(), true);
}

void SceneSession::check_and_build(double at, bool reload_effects) {
  nvfx_scene_error err{};
  if (nvfx_scene_check(text_.c_str(), source_.c_str(), &err) != NVFX_OK) {  // the scene plays on
    error_ = ScriptError{err.line, err.column, err.message, true};
    error_version_ = version_;
    return;
  }
  error_.reset();
  error_version_ = version_;
  Job j;
  j.kind = Kind::build;
  j.script = text_;
  j.source = source_;
  j.settings = settings_;
  j.inputs = inputs_;
  j.reload_effects = reload_effects || (want_build_ && want_build_->reload_effects);
  j.time = at;
  j.version = version_;
  want_build_ = std::move(j);
  if (running_ && running_kind_ == Kind::build) cancel_ = true;  // the older build is abandoned
}

// --- each frame ---------------------------------------------------------------------------------------------------

void SceneSession::update(Clock::time_point now) {
  if (debounce_.due(now)) check_and_build(settings_.restart_on_reload || !has_scene() ? 0.0 : time(), false);
  std::optional<Result> r;
  {
    std::lock_guard lk(m_);
    if (done_) {
      r = std::move(done_);
      done_.reset();
    }
  }
  if (r) {
    running_ = false;
    take(std::move(*r));
  }
  if (!running_) {
    std::lock_guard lk(m_);
    start_next();
  }
  apply_inputs();
}

bool SceneSession::settle(std::chrono::milliseconds timeout) {
  const auto end = Clock::now() + timeout;
  for (;;) {
    update(Clock::now());
    if (!busy() && !debounce_.pending()) return true;
    if (Clock::now() > end) return false;
    std::unique_lock lk(m_);
    done_cv_.wait_for(lk, std::chrono::milliseconds(5), [&] { return done_.has_value(); });
  }
}

void SceneSession::start_next() {
  if (want_seek_ && !live_) want_seek_.reset();  // nothing to seek yet
  Job job;
  if (want_seek_) {
    job.kind = want_seek_->restart ? Kind::restart : Kind::seek;
    job.time = want_seek_->time;
    job.scene = std::move(live_);
    seek_target_ = job.time;
    want_seek_.reset();
  } else if (want_build_) {
    job = std::move(*want_build_);
    want_build_.reset();
  } else {
    return;
  }
  cancel_ = false;
  running_ = true;
  running_kind_ = job.kind;
  job_ = std::move(job);
  cv_.notify_all();
}

void SceneSession::take(Result r) {
  if (r.kind != Kind::build) {  // a seek or a restart gives the scene back
    live_ = std::move(r.scene);
    drawn_frame_ = -1;
    return;
  }
  ++builds_;
  if (r.cancelled) return;  // a newer build follows
  effects_ = std::move(r.effects);
  const bool current = r.version >= error_version_;  // no newer edit has failed its check since
  if (!r.scene) {                                    // the last good scene plays on
    if (current) {
      error_ = std::move(r.error);
      error_version_ = r.version;
    }
    return;
  }
  if (live_) {
    std::lock_guard lk(m_);
    trash_.push_back(std::move(live_));
    cv_.notify_all();
  }
  live_ = std::move(r.scene);
  ++generation_;
  if (current) {
    error_.reset();
    error_version_ = r.version;
  }
  build_ms_ = r.ms;
  nvfx_scene_get_info(live_.get(), &info_);
  picture_.assign(static_cast<std::size_t>(info_.width) * static_cast<std::size_t>(info_.height) * 4, 0);
  drawn_frame_ = -1;
  // values set while it was building carry over, if the script's starting value is unchanged
  std::vector<InputValue> next = std::move(r.inputs);
  applied_.assign(next.size(), 0.f);
  for (std::size_t i = 0; i < next.size(); ++i) {
    applied_[i] = next[i].value;
    for (const InputValue& old : inputs_)
      if (old.name == next[i].name && old.script == next[i].script) next[i].value = old.value;
  }
  inputs_ = std::move(next);
  rules_.clear();
  for (int i = 0; i < info_.n_rules; ++i) {
    const char* n = nvfx_scene_rule_name(live_.get(), i);
    if (n && std::string_view(n).rfind("line ", 0) != 0) rules_.emplace_back(n);  // unnamed rules are "line N"
  }
}

void SceneSession::apply_inputs() {
  if (!live_) return;
  for (std::size_t i = 0; i < inputs_.size() && i < applied_.size(); ++i) {
    if (inputs_[i].value == applied_[i]) continue;
    nvfx_scene_set_input(live_.get(), inputs_[i].name.c_str(), inputs_[i].value);
    applied_[i] = inputs_[i].value;
  }
}

// --- the clock and the picture ----------------------------------------------------------------------------------------

void SceneSession::step(double dt, int max_frames) {
  nvfx_scene* s = live_.get();
  if (!s || !(dt > 0.0)) return;
  const auto t0 = Clock::now();
  const int f0 = nvfx_scene_frame(s);
  if (frame_at(nvfx_scene_time(s) + dt, info_.fps) - f0 > max_frames) {
    nvfx_scene_step_frames(s, max_frames);
  } else {
    nvfx_scene_step(s, dt, nullptr);
  }
  if (nvfx_scene_frame(s) != f0) step_ms_ = ms_since(t0);
}

void SceneSession::step_frames(int n) {
  if (live_ && n > 0) nvfx_scene_step_frames(live_.get(), n);
}

void SceneSession::seek(double seconds) {
  if (!has_scene() || !std::isfinite(seconds)) return;
  seconds = std::max(0.0, seconds);
  if (live_ && !want_seek_) {  // a frame or two forward: here
    const int f0 = nvfx_scene_frame(live_.get()), target = frame_at(seconds, info_.fps);
    if (target >= f0 && target - f0 <= 2) {
      nvfx_scene_seek(live_.get(), seconds);
      return;
    }
  }
  want_seek_ = Wish{seconds, false};
  if (running_ && running_kind_ == Kind::seek) cancel_ = true;  // it gives the scene back where it got to
}

void SceneSession::restart() {
  if (!has_scene()) return;
  want_seek_ = Wish{0.0, true};
  if (running_ && running_kind_ == Kind::seek) cancel_ = true;
}

double SceneSession::time() const {
  if (want_seek_) return want_seek_->time;
  if (running_ && running_kind_ != Kind::build) return seek_target_;
  return live_ ? nvfx_scene_time(live_.get()) : 0.0;
}

int SceneSession::frame() const { return live_ ? nvfx_scene_frame(live_.get()) : frame_at(time(), info_.fps); }

bool SceneSession::draw() {
  if (!live_) return false;
  const int f = nvfx_scene_frame(live_.get());
  if (drawn_generation_ == generation_ && drawn_frame_ == f) return false;
  const auto t0 = Clock::now();
  if (nvfx_scene_render(live_.get(), picture_.data(), static_cast<std::size_t>(info_.width) * 4) != NVFX_OK) return false;
  draw_ms_ = ms_since(t0);
  drawn_generation_ = generation_;
  drawn_frame_ = f;
  return true;
}

bool SceneSession::trigger(const std::string& rule) { return live_ && nvfx_scene_trigger(live_.get(), rule.c_str()) == NVFX_OK; }

bool SceneSession::probe(float px, float py, float& wx, float& wy, nvfx_scene_fields& out) const {
  if (!live_ || info_.width <= 0 || info_.height <= 0) return false;
  float cx = 0.f, cy = 0.f;
  nvfx_scene_camera(live_.get(), &cx, &cy);
  wx = cx + px * static_cast<float>(info_.scene_width) / static_cast<float>(info_.width);
  wy = cy + py * static_cast<float>(info_.scene_height) / static_cast<float>(info_.height);
  return nvfx_scene_sample(live_.get(), wx, wy, &out) == NVFX_OK;
}

// --- the worker -----------------------------------------------------------------------------------------------------

void SceneSession::work() {
  std::unique_lock lk(m_);
  for (;;) {
    cv_.wait(lk, [&] { return stop_ || job_.has_value() || !trash_.empty(); });
    if (!trash_.empty()) {  // scenes replaced: freed here, not on the frame
      std::vector<ScenePtr> t;
      t.swap(trash_);
      lk.unlock();
      t.clear();
      lk.lock();
      continue;
    }
    if (stop_) return;
    Job job = std::move(*job_);
    job_.reset();
    lk.unlock();
    Result r = run(job);
    lk.lock();
    done_ = std::move(r);
    done_cv_.notify_all();
  }
}

SceneSession::Result SceneSession::run(Job& job) {
  try {
    if (job.kind == Kind::build) return build(job);
    Result r;
    r.kind = job.kind;
    nvfx_scene* s = job.scene.get();
    if (job.kind == Kind::restart) {
      nvfx_scene_restart(s);
    } else {
      nvfx_scene_info info{};
      nvfx_scene_get_info(s, &info);
      const int target = frame_at(job.time, info.fps);
      if (target < nvfx_scene_frame(s)) nvfx_scene_restart(s);
      while (nvfx_scene_frame(s) < target && !cancel_) nvfx_scene_step_frames(s, 1);
      if (!cancel_) nvfx_scene_seek(s, job.time);
    }
    r.scene = std::move(job.scene);
    return r;
  } catch (const std::exception& e) {  // the scene API does not throw; the rest could, out of memory
    Result r;
    r.kind = job.kind;
    r.scene = std::move(job.scene);
    r.version = job.version;
    if (job.kind == Kind::build) r.error = ScriptError{0, 0, e.what(), false};
    return r;
  }
}

SceneSession::Result SceneSession::build(Job& job) {
  Result r;
  r.kind = Kind::build;
  r.version = job.version;
  const auto t0 = Clock::now();
  if (cancel_) {
    r.cancelled = true;
    return r;
  }
  if (job.reload_effects) files_.clear();
  // the effects the script names: read from the folder (kept while their files do not change), or stand-ins
  std::vector<std::pair<std::string, std::string>> named;
  nvfx_scene_list_effects(
      job.script.c_str(), job.source.c_str(),
      [](void* user, const char* name, const char* file) { static_cast<std::vector<std::pair<std::string, std::string>>*>(user)->emplace_back(name, file); },
      &named, nullptr);
  std::vector<nvfx_scene_effect> given;
  const std::filesystem::path dir(job.settings.effects_dir);
  for (const auto& [name, file] : named) {
    EffectUse u;
    u.name = name;
    u.file = file;
    const std::filesystem::path p = std::filesystem::path(file).is_absolute() ? std::filesystem::path(file) : dir / file;
    std::error_code ec;
    const bool exists = std::filesystem::is_regular_file(p, ec);
    const nvfx_effect* e = nullptr;
    if (exists && job.settings.stand_ins != StandIns::all) {
      u.from = EffectUse::From::file;  // if it cannot be loaded, the build says so at its line
      u.path = p.string();
      e = load_effect(u.path);
    } else if (job.settings.stand_ins != StandIns::off) {
      u.from = EffectUse::From::stand_in;
      e = stand_in(std::filesystem::path(file).stem().string());
    }
    if (e) given.push_back({file.c_str(), e});
    r.effects.push_back(std::move(u));
  }
  nvfx_scene_desc d;
  nvfx_scene_desc_init(&d);
  d.script = job.script.c_str();
  d.source_name = job.source.c_str();
  d.effects_dir = job.settings.effects_dir.empty() ? nullptr : job.settings.effects_dir.c_str();
  d.effects = given.data();
  d.n_effects = static_cast<int>(given.size());
  d.threads = job.settings.threads;
  d.overlap = job.settings.overlap;
  nvfx_scene* raw = nullptr;
  nvfx_scene_error err{};
  if (nvfx_scene_create(&d, &raw, &err) != NVFX_OK) {
    r.error = ScriptError{err.line, err.column, err.message, false};
    r.ms = ms_since(t0);
    return r;
  }
  ScenePtr scene(raw);
  nvfx_scene_info info{};
  nvfx_scene_get_info(raw, &info);
  // inputs: a value the user set is kept while the script's starting value is unchanged
  bool reset = false;
  for (int i = 0; i < info.n_inputs; ++i) {
    InputValue v;
    v.name = nvfx_scene_input_name(raw, i);
    nvfx_scene_get_input(raw, v.name.c_str(), &v.script);
    v.value = v.script;
    for (const InputValue& old : job.inputs) {
      if (old.name != v.name || old.script != v.script || old.value == v.script) continue;
      nvfx_scene_set_input(raw, v.name.c_str(), old.value);
      v.value = old.value;
      reset = true;
    }
    r.inputs.push_back(std::move(v));
  }
  if (reset) nvfx_scene_restart(raw);  // frame 0 again, with those values from the start
  // played to the time of the edit, frame by frame, abandoned for a newer edit
  const int target = frame_at(job.time, info.fps);
  while (nvfx_scene_frame(raw) < target) {
    if (cancel_) {
      r.cancelled = true;
      return r;  // the scene is freed here, on the worker
    }
    nvfx_scene_step_frames(raw, 1);
  }
  if (job.time > 0.0) nvfx_scene_seek(raw, job.time);
  r.scene = std::move(scene);
  r.ms = ms_since(t0);
  return r;
}

const nvfx_effect* SceneSession::load_effect(const std::string& path) {
  std::error_code ec;
  const auto mtime = std::filesystem::last_write_time(path, ec).time_since_epoch().count();
  const std::uintmax_t size = ec ? 0 : std::filesystem::file_size(path, ec);
  if (ec) return nullptr;
  if (const auto it = files_.find(path); it != files_.end() && it->second.mtime == mtime && it->second.size == size) return it->second.effect.get();
  files_.erase(path);
  nvfx_effect* e = nullptr;
  if (nvfx_effect_load(path.c_str(), &e) != NVFX_OK) return nullptr;
  files_[path] = Cached{static_cast<std::int64_t>(mtime), size, EffectPtr(e)};
  return e;
}

const nvfx_effect* SceneSession::stand_in(const std::string& name) {
  EffectPtr& e = stand_ins_[name];
  if (!e) {
    const std::string bytes = stand_in_bytes(name);
    nvfx_effect* p = nullptr;
    if (!bytes.empty() && nvfx_effect_load_memory(bytes.data(), bytes.size(), &p) == NVFX_OK) e.reset(p);
  }
  return e.get();
}

}  // namespace nfx::viewer
