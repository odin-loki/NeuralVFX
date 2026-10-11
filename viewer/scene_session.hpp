// The viewer's scene mode apart from drawing (docs/VIEWER.md): a scene script being edited and the scene it plays,
// with the slow work between them done off the frame.
//
// An edit is debounced, then checked on the frame (nvfx_scene_check: parse and names, no effects, well under a
// millisecond) and, if it passes, rebuilt on a worker thread (nvfx_scene_create, then played to the time of the edit)
// while the last good scene plays on. A script that fails, in the check or in the build, leaves that scene playing and
// says where (line and column). Seeks and restarts also run on the worker; the picture stays meanwhile. Everything goes
// through the scene C API (include/neuralfx/nvfx_scene.h), as a game would: stepping, drawing and the field reads
// allocate nothing, so neither does a frame here unless a build or a seek has just finished.
//
// Not thread-safe: one thread (the UI's) calls everything; the worker is internal.
#pragma once

#include <neuralfx/nvfx.h>
#include <neuralfx/nvfx_scene.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace nfx::viewer {

using Clock = std::chrono::steady_clock;

// Fires once when edits have been quiet for `delay`.
class Debouncer {
 public:
  explicit Debouncer(Clock::duration delay = std::chrono::milliseconds(300)) : delay_(delay) {}
  void touch(Clock::time_point now) { last_ = now; }  // an edit
  bool due(Clock::time_point now) {                   // true once, `delay` after the last edit
    if (!last_ || now - *last_ < delay_) return false;
    last_.reset();
    return true;
  }
  bool pending() const { return last_.has_value(); }
  void cancel() { last_.reset(); }
  Clock::duration delay() const { return delay_; }
  void set_delay(Clock::duration d) { delay_ = d; }

 private:
  Clock::duration delay_;
  std::optional<Clock::time_point> last_;
};

struct SceneDeleter {
  void operator()(nvfx_scene* s) const { nvfx_scene_free(s); }
};
using ScenePtr = std::unique_ptr<nvfx_scene, SceneDeleter>;
struct EffectDeleter {
  void operator()(nvfx_effect* e) const { nvfx_effect_free(e); }
};
using EffectPtr = std::unique_ptr<nvfx_effect, EffectDeleter>;

// Effects whose file the folder lacks: none (the build fails at the `effect` line), stand-ins for missing files, or
// stand-ins for every effect (tests, and trying a script's layout without the models).
enum class StandIns { off, missing, all };

// A stand-in for an effect file: an untrained rollout effect (32-cell grid, 12 start points of a plume, controls
// intensity, wind and turbulence), the same for the same name. Its .nvfx bytes.
std::string stand_in_bytes(std::string_view name);

struct ScriptError {
  int line = 0, column = 0;  // 1-based; 0: not about a line of the script
  std::string message;       // "name:line:column: message", as the scene API gives it
  bool from_check = false;   // found by nvfx_scene_check (else by the build)
};

struct InputValue {
  std::string name;
  float script = 0.f;  // the script's starting value
  float value = 0.f;   // the value wanted (a slider)
};

struct EffectUse {
  std::string name, file;  // as the script names them
  enum class From { file, stand_in, missing } from = From::missing;
  std::string path;        // the file read (From::file)
};

struct SessionSettings {
  std::string effects_dir;               // relative effect files are read from here
  StandIns stand_ins = StandIns::off;
  int threads = 2;                       // nvfx_scene_desc.threads
  int overlap = -1;                      // nvfx_scene_desc.overlap
  bool restart_on_reload = false;        // a rebuilt scene starts at 0 (else at the time of the edit)
  Clock::duration debounce = std::chrono::milliseconds(300);
};

class SceneSession {
 public:
  explicit SceneSession(SessionSettings s = {});
  ~SceneSession();
  SceneSession(const SceneSession&) = delete;
  SceneSession& operator=(const SceneSession&) = delete;

  // The script. edit(): typed; checked and rebuilt once edits pause. open(): a new script, built at once and played to
  // `at` seconds.
  void edit(std::string text, Clock::time_point now);
  void open(std::string text, std::string source_name, double at = 0.0);
  const std::string& text() const { return text_; }
  const std::string& source_name() const { return source_; }
  bool edit_pending() const { return debounce_.pending(); }

  // Changing the folder, the stand-ins, threads or overlap rebuilds the scene (at the current time).
  const SessionSettings& settings() const { return settings_; }
  void set_settings(const SessionSettings& s);
  void reload_effects();  // read the effects' files again, and rebuild

  // Once per UI frame: a debounced edit is checked and handed to the worker; what the worker finished is taken (a new
  // scene replaces the playing one, a failed build leaves it) and its next job started; inputs are applied.
  void update(Clock::time_point now);
  // Updates until nothing is pending or running (true), or the timeout (false): for tests and screenshots.
  bool settle(std::chrono::milliseconds timeout);

  // The clock. step(): forward by dt seconds, computing at most `max_frames` frames (a scene slower than real time
  // plays slower rather than falling behind). seek() and restart() run on the worker (the last picture stays), except
  // a seek forward by a frame or two, which runs here.
  void step(double dt, int max_frames = 2);
  void step_frames(int n);
  void seek(double seconds);
  void restart();
  double time() const;  // the scene's time (during a seek: its target)
  int frame() const;

  // The playing scene: nullptr before the first good build and while a seek or restart has it on the worker.
  nvfx_scene* scene() const { return live_.get(); }
  bool has_scene() const { return generation_ > 0; }
  const nvfx_scene_info& info() const { return info_; }  // of the last good scene
  std::uint64_t generation() const { return generation_; }  // +1 for every scene that replaces the playing one
  bool busy() const { return running_ || want_build_.has_value() || want_seek_.has_value(); }
  bool building() const { return (running_ && running_kind_ == Kind::build) || want_build_.has_value(); }
  bool seeking() const { return (running_ && running_kind_ != Kind::build) || want_seek_.has_value(); }

  // The picture: the current frame into picture() (info().width x height RGBA), when it is not there yet. True when
  // it changed.
  bool draw();
  const std::vector<std::uint8_t>& picture() const { return picture_; }
  double draw_ms() const { return draw_ms_; }
  double step_ms() const { return step_ms_; }

  // Inputs (set .value; applied at the next update) and named rules of the playing scene.
  std::vector<InputValue>& inputs() { return inputs_; }
  const std::vector<std::string>& rules() const { return rules_; }
  bool trigger(const std::string& rule);  // false: no scene, or a rule that cannot be triggered (a landing)

  // The fields under a point of the picture (pixels of info().width x height): its world position (the camera of the
  // last frame computed) and the bus there. False without a scene.
  bool probe(float px, float py, float& wx, float& wy, nvfx_scene_fields& out) const;

  // The last problem, the check's or the build's (cleared by a check that passes and by a good build).
  const std::optional<ScriptError>& error() const { return error_; }
  const std::vector<EffectUse>& effects() const { return effects_; }  // of the last build tried
  double build_ms() const { return build_ms_; }                       // of the last good build
  int builds() const { return builds_; }                               // builds finished, good or not

 private:
  enum class Kind { build, seek, restart };
  struct Job {
    Kind kind = Kind::build;
    std::string script, source;     // build
    SessionSettings settings;       // build
    std::vector<InputValue> inputs;  // build: the playing scene's (values set by the user carry over)
    bool reload_effects = false;    // build
    double time = 0;                // build: play to; seek: the target
    std::uint64_t version = 0;      // build: of the text
    ScenePtr scene;                 // seek, restart: the playing scene
  };
  struct Result {
    Kind kind = Kind::build;
    ScenePtr scene;
    std::optional<ScriptError> error;
    std::vector<InputValue> inputs;
    std::vector<EffectUse> effects;
    double ms = 0;
    std::uint64_t version = 0;
    bool cancelled = false;
  };
  struct Wish {
    double time = 0;
    bool restart = false;
  };

  // Checks the text (an error stays, the scene plays on) and, if it passes, asks for a build played to `at` seconds.
  void check_and_build(double at, bool reload_effects);
  void start_next();  // with m_ held
  void take(Result r);
  void apply_inputs();
  void work();
  Result run(Job& job);
  Result build(Job& job);
  const nvfx_effect* load_effect(const std::string& path);
  const nvfx_effect* stand_in(const std::string& name);

  SessionSettings settings_;
  Debouncer debounce_;
  std::string text_, source_ = "script";
  std::uint64_t version_ = 0;        // of the text and settings: +1 for every change
  std::uint64_t error_version_ = 0;  // the version error_ is about
  ScenePtr live_;
  nvfx_scene_info info_{};
  std::uint64_t generation_ = 0;
  std::vector<std::uint8_t> picture_;
  std::uint64_t drawn_generation_ = 0;
  int drawn_frame_ = -1;
  double draw_ms_ = 0, step_ms_ = 0;
  std::vector<InputValue> inputs_;
  std::vector<float> applied_;  // per input: the value the scene has
  std::vector<std::string> rules_;
  std::optional<ScriptError> error_;
  std::vector<EffectUse> effects_;
  double build_ms_ = 0;
  int builds_ = 0;
  std::optional<Job> want_build_;  // the next build (latest wins)
  std::optional<Wish> want_seek_;  // the next seek or restart (latest wins)
  double seek_target_ = 0;

  // The worker. m_ guards job_, done_, trash_ and stop_; running_ and running_kind_ are the UI thread's.
  std::mutex m_;
  std::condition_variable cv_, done_cv_;
  std::optional<Job> job_;
  std::optional<Result> done_;
  std::vector<ScenePtr> trash_;  // scenes replaced, freed on the worker
  bool stop_ = false;
  bool running_ = false;
  Kind running_kind_ = Kind::build;
  std::atomic<bool> cancel_{false};  // a newer request supersedes the running job
  // The worker's own: effects read from files (by path, with the file's time and size) and stand-ins (by name).
  struct Cached {
    std::int64_t mtime = 0;
    std::uintmax_t size = 0;
    EffectPtr effect;
  };
  std::map<std::string, Cached> files_;
  std::map<std::string, EffectPtr> stand_ins_;
  std::thread worker_;
};

}  // namespace nfx::viewer
