// Scene scripts: checking a parsed script and running it (script.hpp, docs/COMPOSE.md §4).
//
// compile() turns the statements into a Program: names resolved, expressions compiled to a small stack machine that
// computes in float with the same operations, in the same order, as the C++ they stand for (so a scripted scene can
// match a hand-written one to the bit), constants evaluated, and every context checked. Scene then loads the effects
// and creates every module, list and buffer, so that a frame allocates nothing.
//
// A frame (Scene::render) runs in this order:
//   1. rules on time, shocks and fields (at, when), in script order; their actions run at once;
//   2. emitters (emit), in script order;
//   3. settings that change over time: modules (controls, opacity, look, place), scorch marks, frame, light, camera;
//   4. every active module steps (in parallel);
//   5. tiles of a domain share their bands; the couplings of `every frame` (transfer, suppress, stop), in order;
//   6. the field bus is cleared and every active module publishes to it;
//   7. pushes of `every frame` (they read the bus), then the force fields, in order;
//   8. light; 9. fields on particles, particles move, rules on landings; then the water (water.hpp, its settings and a
//      step); 10. shading and the picture.
#include "script.hpp"
#include "water.hpp"
#include "weather.hpp"

#include "fx.hpp"

#include <neuralfx/noise.hpp>
#include <neuralfx/rollout.hpp>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <format>
#include <limits>
#include <map>
#include <set>

namespace nfx::compose::script {

namespace {

constexpr float kNever = std::numeric_limits<float>::infinity();
constexpr int kMaxStack = 32;
constexpr int kMaxShocks = 32;  // shocks and scorch marks a repeating rule can leave

float smooth01(float t) {
  t = std::clamp(t, 0.f, 1.f);
  return t * t * (3.f - 2.f * t);
}

// --- expressions as code -----------------------------------------------------------------------------------------------

enum class Op : std::uint8_t { num, slot, neg, not_, add, sub, mul, div, lt, le, gt, ge, eq, ne, jz, jnz, jmp, to_bool, fn1, fn2, fn3, rand, heat, soot, shock_r, reaches };
enum Fn : std::uint8_t { f_smooth, f_exp, f_sqrt, f_sin, f_cos, f_abs, f_floor, f_min, f_max, f_pow, f_hypot, f_clamp, f_lerp, f_noise };

struct Instr {
  Op op = Op::num;
  std::uint8_t fn = 0;
  int arg = 0;
  float value = 0.f;
};

struct Code {
  std::vector<Instr> ops;
  bool varying = false;  // depends on time, rules, modules, the bus, particles or rand
  bool present() const { return !ops.empty(); }
};

// Slots: values the runner keeps up to date and expressions read.
enum : int { s_t, s_length, s_fps, s_ground, s_land_x, s_land_temp, s_first };
enum : int { m_x, m_y, m_started, m_active, m_vx, m_vy, m_slots };  // per module (vx, vy: a trail's source)

struct Env {
  float* slots = nullptr;
  Particles* parts = nullptr;
  const FieldBus* bus = nullptr;
  const std::vector<Shock>* shocks = nullptr;
};

float fn1(std::uint8_t f, float a) {
  switch (f) {
    case f_smooth: return smooth01(a);
    case f_exp: return std::exp(a);
    case f_sqrt: return std::sqrt(a);
    case f_sin: return std::sin(a);
    case f_cos: return std::cos(a);
    case f_abs: return std::fabs(a);
    case f_floor: return std::floor(a);
    default: return 0.f;
  }
}

float fn2(std::uint8_t f, float a, float b) {
  switch (f) {
    case f_min: return std::min(a, b);
    case f_max: return std::max(a, b);
    case f_pow: return std::pow(a, b);
    case f_hypot: return std::hypot(a, b);
    default: return 0.f;
  }
}

float fn3(std::uint8_t f, float a, float b, float c) {
  switch (f) {
    case f_clamp: return std::clamp(a, b, c);
    case f_lerp: return a + c * (b - a);
    case f_noise: return value_noise(a, b, c, 0x5eed);
    default: return 0.f;
  }
}

// Every operation in float, one at a time, as the C++ it stands for.
float run(const Code& c, const Env& env) {
  float st[kMaxStack];
  int sp = 0;
  const int n = static_cast<int>(c.ops.size());
  for (int pc = 0; pc < n; ++pc) {
    const Instr& in = c.ops[zs(pc)];
    switch (in.op) {
      case Op::num: st[sp++] = in.value; break;
      case Op::slot: st[sp++] = env.slots[in.arg]; break;
      case Op::neg: st[sp - 1] = -st[sp - 1]; break;
      case Op::not_: st[sp - 1] = st[sp - 1] == 0.f ? 1.f : 0.f; break;
      case Op::to_bool: st[sp - 1] = st[sp - 1] != 0.f ? 1.f : 0.f; break;
      case Op::add: --sp; st[sp - 1] = st[sp - 1] + st[sp]; break;
      case Op::sub: --sp; st[sp - 1] = st[sp - 1] - st[sp]; break;
      case Op::mul: --sp; st[sp - 1] = st[sp - 1] * st[sp]; break;
      case Op::div: --sp; st[sp - 1] = st[sp - 1] / st[sp]; break;
      case Op::lt: --sp; st[sp - 1] = st[sp - 1] < st[sp] ? 1.f : 0.f; break;
      case Op::le: --sp; st[sp - 1] = st[sp - 1] <= st[sp] ? 1.f : 0.f; break;
      case Op::gt: --sp; st[sp - 1] = st[sp - 1] > st[sp] ? 1.f : 0.f; break;
      case Op::ge: --sp; st[sp - 1] = st[sp - 1] >= st[sp] ? 1.f : 0.f; break;
      case Op::eq: --sp; st[sp - 1] = st[sp - 1] == st[sp] ? 1.f : 0.f; break;
      case Op::ne: --sp; st[sp - 1] = st[sp - 1] != st[sp] ? 1.f : 0.f; break;
      case Op::jz:
        if (st[--sp] == 0.f) pc = in.arg - 1;
        break;
      case Op::jnz:
        if (st[--sp] != 0.f) pc = in.arg - 1;
        break;
      case Op::jmp: pc = in.arg - 1; break;
      case Op::fn1: st[sp - 1] = fn1(in.fn, st[sp - 1]); break;
      case Op::fn2: --sp; st[sp - 1] = fn2(in.fn, st[sp - 1], st[sp]); break;
      case Op::fn3: sp -= 2; st[sp - 1] = fn3(in.fn, st[sp - 1], st[sp], st[sp + 1]); break;
      case Op::rand: st[sp++] = env.parts ? env.parts->uniform() : 0.f; break;
      case Op::heat:
        --sp;
        st[sp - 1] = env.bus ? env.bus->at(st[sp - 1], st[sp]).heat : 0.f;
        break;
      case Op::soot:
        --sp;
        st[sp - 1] = env.bus ? env.bus->at(st[sp - 1], st[sp]).soot : 0.f;
        break;
      case Op::shock_r: {
        const int k = static_cast<int>(st[sp - 1]);
        st[sp - 1] = env.shocks && k >= 1 && k <= static_cast<int>(env.shocks->size()) ? (*env.shocks)[zs(k - 1)].radius(env.slots[s_t]) : 0.f;
        break;
      }
      case Op::reaches: {  // the shock front has reached (x, y): radius(t) >= hypot(x - shock x, y - shock y)
        sp -= 2;
        const float x = st[sp], y = st[sp + 1];
        bool hit = false;
        if (env.shocks && in.arg >= 1 && in.arg <= static_cast<int>(env.shocks->size())) {
          const Shock& s = (*env.shocks)[zs(in.arg - 1)];
          hit = s.radius(env.slots[s_t]) >= std::hypot(x - s.x, y - s.y);
        }
        st[sp++] = hit ? 1.f : 0.f;
        break;
      }
    }
  }
  return sp > 0 ? st[0] : 0.f;
}

// --- the program -------------------------------------------------------------------------------------------------------

enum P : int {
  p_x, p_y, p_x1, p_y1, p_w, p_h, p_fraction, p_top, p_heat, p_soot, p_gain, p_from, p_until, p_seed, p_speed, p_decay,
  p_amp, p_width, p_radius, p_glow, p_embers, p_debris, p_count, p_tries, p_chance, p_spread, p_level, p_strength,
  p_soft, p_damping, p_swirl, p_steam, p_amount, p_scale, p_rate, p_u, p_v, p_core, p_weight, p_if, p_start,
  p_cr, p_cg, p_cb, p_bright, p_jitter, p_branches, p_squash, p_sparks, p_dx, p_dy, p_cone,
  p_density, p_wu, p_wv, p_flowk, p_wash, p_melt, p_glowing, p_spatter, p_shards, p_grain, p_stain, p_floor, p_power, p_mist_amount, p_mist_heat,
  p_shutter, p_depth, kParams
};
using Params = std::array<Code, kParams>;

struct ModC {
  std::string name;
  Pos pos;
  int effect = -1;
  bool tiled = false;
  int cols = 1, rows = 1, band = 0, over = -1;
  int size = 0;
  float width = 0, sink = 0, feather = 0;
  Code width_code;  // a single module's width may change over time (a spawned effect's scale); tiles' is constant
  bool has_size = false, has_width = false;
  bool glows = false;
  std::array<float, 3> glow{};  // the colour of its light (Module::glows)
  float shards = 0.f;           // seconds per shard (Module::use_shards); 0: one continuous rollout
  Code ax, ay;  // where it stands (bottom centre of the tile or domain); varying: it moves
  bool has_at = false;
  int look_a = -1, look_b = -1;  // -1: the learned renderer
  Code look_w;
  std::vector<float> controls;  // constant controls (tuple)
  struct Named {
    std::string name;
    Pos pos;
    Code code;
  };
  std::vector<Named> named;  // per-control settings
  Code opacity;
  bool has_start = false, waiting = false, empty = false;
  int start = 0;
  std::uint64_t seed = 1;
  bool declared_controls = false;
  int slot = 0;
  // A trail (docs/COMPOSE.md §5.3): `tiles` tiles laid in the world along its source's path (ax, ay).
  bool trail = false;
  struct TrailC {
    int count = 8, band = 8, rows = 0;  // rows 0: any
    float age = 3.f, hold = 1.f, fade = 1.5f, jump = 0.f, base = 0.f;
    bool has_base = false, has_quiet = false;
    std::array<int, 4> quiet{};
    Code heat, soot, radius, kick, u, v, on;
  } tr;
};

struct ActC {
  std::string kind;
  Pos pos;
  std::vector<int> subjects, targets;
  Params p;
  std::vector<std::array<int, 2>> tiles;  // suppress: which tiles (empty: all)
  std::array<int, 4> cells{};             // suppress: x0, y0, x1, y1
  bool has_in = false, empty = false;
  int in_c = 0, in_r = 0, from = 0;  // start
  std::vector<Module*> buf;           // transfer: the active targets, filled per call
  int water = -1, mist = -1;          // splash: the pool it lands on, the module its mist feeds; wave: the pool
};

// Water (water.hpp, docs/EFFECTS_WATER.md): an element's settings, each an expression evaluated every frame.
enum WP : int {
  w_x0, w_y0, w_x1, w_y1, w_flow, w_vx, w_vy, w_speed, w_angle, w_spread, w_fan, w_pulse, w_breakup, w_gravity, w_wu, w_wv,
  w_rate, w_radius, w_depth, w_damping, w_drift, w_surface, w_swell, w_wavelength, w_period, w_density, w_sway, w_gust,
  w_floor, w_mist_amount, w_mist_heat, w_brightness, w_clarity, w_cr, w_cg, w_cb, w_from, w_until, w_if, w_threshold, w_churn,
  kWaterParams
};

struct WaterC {
  std::string name;
  Pos pos;
  WaterKind kind = WaterKind::pool;
  std::array<Code, kWaterParams> p;
  int strands = 24, nodes = 72, cells = 128, count = 256;  // what is fixed: WaterLayout
  int onto = -1, mist = -1;                                // a pool (a water element), a module
  std::uint64_t seed = 1;
};

struct RuleC {
  std::string name;  // `as NAME`, or "line N"
  Pos pos;
  std::string kind;  // time, shock, lands, condition
  Code cond;
  Kind particle = Kind::ember;
  int limit = 1;
  int slot = 0;
  std::vector<ActC> actions;
};

struct FieldC {
  std::string name;
  Pos pos;
  ForceField::Kind kind = ForceField::Kind::ceiling;
  Params p;
  std::vector<int> mods;
  bool particles = false;
};

struct EmitC {
  std::string kind;
  Pos pos;
  Params p;
  std::vector<int> mods;
  // the procedural elements of fx.hpp: which, and its values (fx::V)
  bool is_fx = false;
  fx::Emitter fx_kind = fx::Emitter::hail;
  std::array<Code, fx::kValues> fxv;
};

struct EffectC {
  std::string name, file;
  Pos pos;
  std::vector<std::pair<std::string, float>> detail;
  int precision = -1;  // `precision float|int8` on the effect: its modules' precision; -1: the scene's (Options::precision)
};

struct InputC {  // `input NAME = value`: a value the game sets while the scene plays
  std::string name;
  Pos pos;
  int slot = 0;
  float value = 0.f;  // the script's starting value
};

// The weather director (weather.hpp, docs/WEATHER.md): `weather NAME, seed S, regime R, over D, direction X [: schedule]`.
// Its members (NAME.rain, ...) are slots from `slot` on, written each frame before the rules.
struct WeatherC {
  std::string name;
  Pos pos;
  std::uint64_t seed = 1;
  float direction = 1.f;
  Code regime;      // where to head (a number: 0 clear ... 6 dust; regime names are values in it), each frame; none: clear
  Code over;        // how long a change takes, evaluated when it starts (default 60 s)
  struct Line {     // a line of the schedule: from `at` on, head for `regime` over `over` (NaN: the statement's)
    int regime = 0;
    float at = 0.f, over = 0.f;
  };
  std::vector<Line> schedule;
  int start = 0;    // the regime it starts in, with a schedule
  int slot = 0;
};

struct Program {
  int width = 1280, height = 720;
  float fps = 30.f, length = 10.f, ground = 600.f;
  float bus_x = 0, bus_y = 0, bus_w = 0, bus_h = 0, bus_cell = 8.f;
  bool has_bus = false;
  int capacity = 4096;
  Code flow, light_gain, exposure, fade, haze, bloom, bloom_threshold, cam_x, cam_y, daylight, overcast, visibility;
  std::array<Code, 3> flash;
  std::vector<float> keyframes;
  std::vector<EffectC> effects;
  std::vector<ShaderSpec> looks;
  std::vector<std::string> look_names;
  std::vector<ModC> modules;
  std::vector<FieldC> fields;
  std::vector<EmitC> emitters;
  std::vector<WaterC> waters;
  int splash_capacity = 0;  // splashes that may be under way at once (as shocks: each splash action's copies)
  std::vector<RuleC> rules;
  std::vector<ActC> every;
  std::vector<InputC> inputs;
  std::vector<WeatherC> weathers;
  int slots = s_first;
  int shock_capacity = 0, scorch_capacity = 0;
};

// --- the compiler ------------------------------------------------------------------------------------------------------

struct Ctx {
  bool landing = false;    // inside a `when ... lands` rule: x and temp are the landing's
  bool constant = false;   // must not change over time
  bool scene = false;      // the scene's own settings: ground, length and fps are not known yet
  bool regimes = false;    // a weather's `regime`: the regimes' names are values (clear 0 ... dust 6)
  std::string_view what;   // for messages
};

class Compiler {
 public:
  explicit Compiler(const Script& s) : s_(s) {}

  Program compile();

 private:
  [[noreturn]] void fail(Pos p, const std::string& m) const { throw Error(p, m, s_.source); }

  // Effects and looks have their own names; values, rules, modules and fields share one set (they meet in expressions).
  void declare(const std::string& name, Pos pos, std::string_view what) {
    if (name.empty()) return;
    const bool listed = what == "module";  // modules are named in lists, which end at a property's key
    if (listed ? is_reserved(name) : is_keyword(name)) fail(pos, std::format("'{}' is a word of the script language; choose another name for this {}", name, what));
    auto& names = what == "effect" ? effect_names_ : what == "look" ? look_names_ : names_;
    if (const auto it = names.find(name); it != names.end()) {
      const std::string& kind = it->second.first;
      fail(pos, std::format("'{}' is already the name of {} {} (line {})", name, kind == "effect" || kind == "input" ? "an" : "a", kind, it->second.second.line));
    }
    names[name] = {std::string(what), pos};
  }

  // expressions
  void emit(const Expr& e, const Ctx& c, Code& code, std::vector<std::string>& lets_open);
  Code code(const Expr& e, const Ctx& c) {
    Code out;
    std::vector<std::string> open;
    emit(e, c, out, open);
    // The stack's depth: each op's effect, along every path (jumps only go forward, so one pass sees every way into
    // an op before the op). Both sides of an if() or an `and` are not on the stack at once.
    std::vector<int> depth_in(out.ops.size() + 1, -1);
    depth_in[0] = 0;
    int hmax = 0;
    for (std::size_t pc = 0; pc < out.ops.size(); ++pc) {
      if (depth_in[pc] < 0) continue;  // nothing falls or jumps here
      const Instr& in = out.ops[pc];
      int h = depth_in[pc];
      switch (in.op) {
        case Op::num:
        case Op::slot:
        case Op::rand: ++h; break;
        case Op::add: case Op::sub: case Op::mul: case Op::div: case Op::lt: case Op::le: case Op::gt: case Op::ge:
        case Op::eq: case Op::ne: case Op::fn2: case Op::heat: case Op::soot: case Op::jz: case Op::jnz: --h; break;
        case Op::fn3: h -= 2; break;
        case Op::reaches: --h; break;
        default: break;
      }
      hmax = std::max(hmax, h + 1);
      const auto flow = [&](std::size_t to) {
        if (to < depth_in.size()) depth_in[to] = std::max(depth_in[to], h);
      };
      if (in.op == Op::jz || in.op == Op::jnz) {
        flow(zs(in.arg));
        flow(pc + 1);
      } else if (in.op == Op::jmp) {
        flow(zs(in.arg));
      } else {
        flow(pc + 1);
      }
    }
    if (hmax >= kMaxStack) fail(e.pos, "this expression is too deeply nested");
    if (c.constant && out.varying) fail(e.pos, std::format("{} must be a constant: it cannot depend on time, rules, modules, inputs or rand", c.what));
    return out;
  }
  float constant(const Expr& e, std::string_view what, bool scene = false) {
    Ctx c;
    c.constant = true;
    c.scene = scene;
    c.what = what;
    const Code k = code(e, c);
    float slots[s_first] = {0.f, prog_.length, prog_.fps, prog_.ground, 0.f, 0.f};
    Env env;
    env.slots = slots;
    return run(k, env);
  }
  int integer(const Expr& e, std::string_view what, int lo, int hi = INT_MAX) {
    const float v = constant(e, what);
    if (!(v == std::floor(v)) || v < static_cast<float>(lo) || v > static_cast<float>(hi)) {
      fail(e.pos, hi == INT_MAX ? std::format("{} must be a whole number of at least {}", what, lo) : std::format("{} must be a whole number from {} to {}", what, lo, hi));
    }
    return static_cast<int>(v);
  }
  // A property's expressions into params (a point fills two).
  void take(Params& p, const Statement& s, std::string_view key, P a, P b, const Ctx& c) {
    const Prop* pr = s.prop(key);
    if (!pr) return;
    Ctx cc = c;
    const std::string what = std::format("'{}'", key);
    cc.what = what;
    p[a] = code(pr->value.exprs[0], cc);
    if (b != kParams) p[b] = code(pr->value.exprs[1], cc);
  }
  int module_index(const std::string& name, Pos pos) const {
    std::vector<std::string> all;
    for (std::size_t i = 0; i < prog_.modules.size(); ++i) {
      if (prog_.modules[i].name == name) return static_cast<int>(i);
      all.push_back(prog_.modules[i].name);
    }
    if (const auto it = names_.find(name); it != names_.end()) fail(pos, std::format("'{}' is a {}, not a module", name, it->second.first));
    fail(pos, std::format("unknown module '{}'{}", name, did_you_mean(name, all)));
  }
  int look_index(const std::string& name, Pos pos) const {
    for (std::size_t i = 0; i < prog_.look_names.size(); ++i)
      if (prog_.look_names[i] == name) return static_cast<int>(i);
    const std::string hint = did_you_mean(name, prog_.look_names);
    fail(pos, hint.empty() ? std::format("unknown look '{}' (declare it with 'look {} = shader ...')", name, name) : std::format("unknown look '{}'{}", name, hint));
  }
  std::vector<std::string> value_names() const {  // what a name in an expression can be
    std::vector<std::string> out = {"t", "length", "fps", "ground", "rand", "infinity"};
    for (const auto& [n, what] : names_)
      if (what.first == "value" || what.first == "rule" || what.first == "input") out.push_back(n);
    return out;
  }
  void only(const Statement& s, std::initializer_list<std::string_view> keys, std::string_view what) {
    for (const Prop& p : s.props) {
      if (std::ranges::find(keys, p.key) == keys.end()) {
        std::string list;
        for (const auto k : keys) list += std::format("{}{}", list.empty() ? "" : ", ", k);
        fail(p.pos, std::format("'{}' does not apply to {} (it takes: {})", p.key, what, list));
      }
    }
  }

  void scene(const Statement& s);
  void module(const Statement& s, ModC& m);
  void trail(const Statement& s, ModC& m);
  void field(const Statement& s);
  void emitter(const Statement& s);
  void water(const Statement& s, WaterC& w);
  int water_index(const std::string& name, Pos pos, bool pool) const {
    std::vector<std::string> all;
    for (std::size_t i = 0; i < prog_.waters.size(); ++i) {
      const WaterC& w = prog_.waters[i];
      if (w.name == name) {
        if (pool && w.kind != WaterKind::pool) fail(pos, std::format("'{}' is water of kind {}, not a pool", name, water_kind_name(w.kind)));
        return static_cast<int>(i);
      }
      all.push_back(w.name);
    }
    if (const auto it = names_.find(name); it != names_.end()) fail(pos, std::format("'{}' is a {}, not water", name, it->second.first));
    fail(pos, std::format("unknown water '{}'{}", name, did_you_mean(name, all)));
  }
  const WeatherC* weather_of(const std::string& name) const {
    for (const WeatherC& w : prog_.weathers)
      if (w.name == name) return &w;
    return nullptr;
  }
  void weather(const Statement& s, WeatherC& w);
  void rule(const Statement& s, RuleC& r);
  ActC action(const Statement& a, const Ctx& c, bool every, int limit);
  std::vector<int> modules_of(const std::vector<std::string>& names, Pos pos, bool particles_ok, bool* particles = nullptr) const;

  const Script& s_;
  Program prog_;
  std::map<std::string, std::pair<std::string, Pos>> names_, effect_names_, look_names_;
  std::map<std::string, const Statement*> lets_;
  std::map<std::string, int> rule_slots_, input_slots_;
};

void Compiler::emit(const Expr& e, const Ctx& c, Code& code, std::vector<std::string>& open) {
  const auto op = [&](Op o, int arg = 0, float v = 0.f, std::uint8_t fn = 0) { code.ops.push_back({o, fn, arg, v}); };
  const auto slot = [&](int i, bool varying) {
    op(Op::slot, i);
    code.varying |= varying;
  };
  const auto module_of = [&](const std::string& name, Pos pos) -> const ModC& {
    const int i = module_index(name, pos);
    return prog_.modules[zs(i)];
  };
  switch (e.kind) {
    case Expr::Kind::number: op(Op::num, 0, e.number); return;
    case Expr::Kind::name: {
      const std::string& n = e.name;
      if (wx::Regime r{}; c.regimes && wx::regime_of(n, r)) return op(Op::num, 0, static_cast<float>(static_cast<int>(r)));
      if (n == "t") {
        if (c.scene) fail(e.pos, std::format("{} cannot depend on time", c.what));
        return slot(s_t, true);
      }
      if (n == "length" || n == "fps" || n == "ground") {
        if (c.scene) fail(e.pos, std::format("the scene's settings cannot use '{}'", n));
        return slot(n == "length" ? s_length : n == "fps" ? s_fps : s_ground, false);
      }
      if (n == "x" || n == "temp") {
        if (!c.landing) fail(e.pos, std::format("'{}' is the landing particle's; it can be used only in a rule 'when ember lands' (as in 'at ({}, ground)')", n, n));
        return slot(n == "x" ? s_land_x : s_land_temp, true);
      }
      if (n == "rand") {
        op(Op::rand);
        code.varying = true;
        return;
      }
      if (n == "infinity") return op(Op::num, 0, kNever);
      if (const auto it = lets_.find(n); it != lets_.end()) {
        if (std::ranges::find(open, n) != open.end()) fail(e.pos, std::format("'{}' is defined in terms of itself", n));
        open.push_back(n);
        emit(it->second->args[0], c, code, open);
        open.pop_back();
        return;
      }
      if (const auto it = rule_slots_.find(n); it != rule_slots_.end()) {
        if (c.scene) fail(e.pos, "the scene's settings cannot depend on rules");
        return slot(it->second, true);
      }
      if (const auto it = input_slots_.find(n); it != input_slots_.end()) {  // set by the game: it may change at any frame
        if (c.scene) fail(e.pos, "the scene's settings cannot depend on inputs");
        return slot(it->second, true);
      }
      if (const auto it = names_.find(n); it != names_.end()) {
        if (it->second.first == "module") fail(e.pos, std::format("a module is not a value: use {}.x, {}.y, {}.started, {}.age or {}.active", n, n, n, n, n));
        if (it->second.first == "weather") fail(e.pos, std::format("a weather is not a value: use {}.rain, {}.cloud, {}.wind, {}.visibility, ...", n, n, n, n));
        fail(e.pos, std::format("'{}' is a {}, not a value", n, it->second.first));
      }
      if (is_reserved(n)) fail(e.pos, std::format("'{}' is not a value here", n));
      if (c.regimes) {  // a weather's regime: most likely a misspelt regime
        std::vector<std::string> all = value_names();
        for (const std::string_view r : wx::regime_names()) all.emplace_back(r);
        fail(e.pos, std::format("unknown name '{}': a regime is clear, overcast, rain, storm, snow, fog or dust{}", n, did_you_mean(n, all)));
      }
      fail(e.pos, std::format("unknown name '{}'{}", n, did_you_mean(n, value_names())));
    }
    case Expr::Kind::member: {
      if (const WeatherC* w = weather_of(e.name)) {  // the weather director's state (weather.hpp)
        if (c.scene) fail(e.pos, "the scene's settings cannot depend on the weather");
        const int k = wx::member_of(e.member);
        if (k < 0) {
          std::vector<std::string> all;
          for (const std::string_view m : wx::member_names()) all.emplace_back(m);
          const std::string hint = did_you_mean(e.member, all);
          fail(e.pos, hint.empty() ? std::format("a weather has no '{}' (it has cloud, rain, snow, hail, wind, mean, gust, visibility, temp, lightning, dust, fog, "
                                                 "extinction, u, beaufort, lean, leaves, lift, light, flash, strike, bolt, since, regime and progress)", e.member)
                                   : std::format("a weather has no '{}'{}", e.member, hint));
        }
        return slot(w->slot + k, true);
      }
      if (c.scene) fail(e.pos, "the scene's settings cannot depend on modules");
      const ModC& m = module_of(e.name, e.pos);
      if (e.member == "x") return slot(m.slot + m_x, true);
      if (e.member == "y") return slot(m.slot + m_y, true);
      if (e.member == "started") return slot(m.slot + m_started, true);
      if (e.member == "active") return slot(m.slot + m_active, true);
      if (e.member == "vx" || e.member == "vy") {
        if (!m.trail) fail(e.pos, std::format("only a trail has a velocity ({}.{}): '{}' is a module", e.name, e.member, e.name));
        return slot(m.slot + (e.member == "vx" ? m_vx : m_vy), true);
      }
      if (e.member == "age") {
        slot(s_t, true);
        slot(m.slot + m_started, true);
        op(Op::sub);
        return;
      }
      fail(e.pos, std::format("a module has no '{}' (x, y, started, age or active; a trail also vx and vy)", e.member));
    }
    case Expr::Kind::unary:
      emit(e.args[0], c, code, open);
      op(e.name == "not" ? Op::not_ : Op::neg);
      return;
    case Expr::Kind::binary: {
      const std::string& o = e.name;
      if (o == "and" || o == "or") {  // short-circuit: the right side is not evaluated (nor its rand drawn) when not needed
        emit(e.args[0], c, code, open);
        const std::size_t j = code.ops.size();
        op(o == "and" ? Op::jz : Op::jnz);
        emit(e.args[1], c, code, open);
        op(Op::to_bool);
        const std::size_t j2 = code.ops.size();
        op(Op::jmp);
        code.ops[j].arg = static_cast<int>(code.ops.size());
        op(Op::num, 0, o == "and" ? 0.f : 1.f);
        code.ops[j2].arg = static_cast<int>(code.ops.size());
        return;
      }
      emit(e.args[0], c, code, open);
      emit(e.args[1], c, code, open);
      static constexpr std::pair<std::string_view, Op> ops[] = {{"+", Op::add}, {"-", Op::sub}, {"*", Op::mul}, {"/", Op::div}, {"<", Op::lt},
                                                                {"<=", Op::le}, {">", Op::gt},  {">=", Op::ge}, {"==", Op::eq}, {"!=", Op::ne}};
      for (const auto& [text, code_op] : ops) {
        if (text == o) return op(code_op);
      }
      fail(e.pos, std::format("unknown operator '{}'", o));
    }
    case Expr::Kind::call: {
      const std::string& f = e.name;
      const auto arity = [&](std::size_t n) {
        if (e.args.size() != n) fail(e.pos, std::format("{}() takes {} value{}, not {}", f, n, n == 1 ? "" : "s", e.args.size()));
      };
      const auto args = [&] {
        for (const Expr& a : e.args) emit(a, c, code, open);
      };
      static constexpr std::pair<std::string_view, Fn> one[] = {{"smooth", f_smooth}, {"exp", f_exp}, {"sqrt", f_sqrt}, {"sin", f_sin},
                                                                {"cos", f_cos},       {"abs", f_abs}, {"floor", f_floor}};
      static constexpr std::pair<std::string_view, Fn> two[] = {{"min", f_min}, {"max", f_max}, {"pow", f_pow}, {"hypot", f_hypot}};
      static constexpr std::pair<std::string_view, Fn> three[] = {{"clamp", f_clamp}, {"lerp", f_lerp}, {"noise", f_noise}};
      for (const auto& [name, fn] : one) {
        if (name == f) {
          arity(1);
          args();
          return op(Op::fn1, 0, 0.f, fn);
        }
      }
      for (const auto& [name, fn] : two) {
        if (name == f) {
          arity(2);
          args();
          return op(Op::fn2, 0, 0.f, fn);
        }
      }
      for (const auto& [name, fn] : three) {
        if (name == f) {
          arity(3);
          args();
          return op(Op::fn3, 0, 0.f, fn);
        }
      }
      if (f == "if") {  // if(c, a, b): only the side taken is evaluated
        arity(3);
        emit(e.args[0], c, code, open);
        const std::size_t j = code.ops.size();
        op(Op::jz);
        emit(e.args[1], c, code, open);
        const std::size_t j2 = code.ops.size();
        op(Op::jmp);
        code.ops[j].arg = static_cast<int>(code.ops.size());
        emit(e.args[2], c, code, open);
        code.ops[j2].arg = static_cast<int>(code.ops.size());
        return;
      }
      if (f == "rand") {
        arity(0);
        op(Op::rand);
        code.varying = true;
        return;
      }
      if (f == "heat" || f == "soot") {  // the field bus at a world point (last frame's)
        arity(2);
        if (c.scene || c.constant) fail(e.pos, std::format("{}() reads the scene as it plays; it cannot be used here", f));
        args();
        op(f == "heat" ? Op::heat : Op::soot);
        code.varying = true;
        return;
      }
      if (f == "shock") {  // shock(n): the radius of the n-th shock front now (0 before it)
        arity(1);
        args();
        op(Op::shock_r);
        code.varying = true;
        return;
      }
      if (f == "near") {  // near(M, x, d): M is active and its centre is within d of x
        arity(3);
        if (e.args[0].kind != Expr::Kind::name) fail(e.args[0].pos, "near() takes a module first: near(fire_a, x, 200)");
        const ModC& m = module_of(e.args[0].name, e.args[0].pos);
        slot(m.slot + m_active, true);
        const std::size_t j = code.ops.size();
        op(Op::jz);
        emit(e.args[1], c, code, open);
        slot(m.slot + m_x, true);
        op(Op::sub);
        op(Op::fn1, 0, 0.f, f_abs);
        emit(e.args[2], c, code, open);
        op(Op::lt);
        const std::size_t j2 = code.ops.size();
        op(Op::jmp);
        code.ops[j].arg = static_cast<int>(code.ops.size());
        op(Op::num, 0, 0.f);
        code.ops[j2].arg = static_cast<int>(code.ops.size());
        return;
      }
      static const std::vector<std::string> functions = {"smooth", "exp", "sqrt", "sin", "cos", "abs", "floor", "min", "max", "pow", "hypot", "clamp",
                                                         "lerp", "noise", "if", "rand", "heat", "soot", "shock", "near"};
      fail(e.pos, std::format("unknown function '{}'{}", f, did_you_mean(f, functions)));
    }
  }
}

constexpr std::string_view kModuleKeys[] = {"tiles", "band", "size",  "width", "at",    "sink",    "over", "feather", "look",
                                            "controls", "opacity", "start", "seed", "waiting", "empty", "glow", "shards"};
bool module_key(std::string_view k) { return std::ranges::find(kModuleKeys, k) != std::end(kModuleKeys); }

std::vector<int> Compiler::modules_of(const std::vector<std::string>& names, Pos pos, bool particles_ok, bool* particles) const {
  std::vector<int> out;
  for (const std::string& n : names) {
    if (n == "particles") {
      if (!particles_ok || !particles) fail(pos, "only a flow (wind, gust, vortex, ring, attract) can act on particles");
      *particles = true;
      continue;
    }
    const int i = module_index(n, pos);
    if (std::ranges::find(out, i) != out.end()) fail(pos, std::format("'{}' is listed twice", n));
    out.push_back(i);
  }
  return out;
}

void Compiler::scene(const Statement& s) {
  if (const Prop* p = s.prop("size")) {
    prog_.width = integer(p->value.exprs[0], "the scene's width", 16, 8192);
    prog_.height = integer(p->value.exprs[1], "the scene's height", 16, 8192);
  }
  if (const Prop* p = s.prop("fps")) {
    prog_.fps = constant(p->value.exprs[0], "fps", true);
    if (!(prog_.fps > 0.f)) fail(p->pos, "fps must be positive");
  }
  if (const Prop* p = s.prop("length")) {
    prog_.length = constant(p->value.exprs[0], "the length", true);
    if (!(prog_.length > 0.f)) fail(p->pos, "the length must be positive");
  }
  if (const Prop* p = s.prop("ground")) prog_.ground = constant(p->value.exprs[0], "the ground", true);
}

void Compiler::module(const Statement& s, ModC& m) {
  for (std::size_t i = 0; i < prog_.effects.size(); ++i)
    if (prog_.effects[i].name == s.kind) m.effect = static_cast<int>(i);
  if (m.effect < 0) {
    std::vector<std::string> effects;
    for (const EffectC& e : prog_.effects) effects.push_back(e.name);
    const std::string hint = did_you_mean(s.kind, effects);
    fail(s.pos, hint.empty() ? std::format("unknown effect '{}' (declare it with 'effect {} = \"FILE\"')", s.kind, s.kind) : std::format("unknown effect '{}'{}", s.kind, hint));
  }
  const int self = static_cast<int>(&m - prog_.modules.data());
  if (const Prop* p = s.prop("over")) {
    m.over = module_index(p->value.names[0], p->value.pos);
    if (m.over >= self) fail(p->value.pos, std::format("'{}' must be declared before '{}' to be under it", p->value.names[0], s.name));
    const ModC& o = prog_.modules[zs(m.over)];
    if (o.trail) fail(p->value.pos, std::format("'{}' is a trail: its tiles move with its source, so nothing can be over them", o.name));
    for (const char* k : {"tiles", "band", "width", "at", "sink", "size"})
      if (s.prop(k)) fail(s.prop(k)->pos, std::format("'{}' comes from '{}' (the module this one is over)", k, o.name));
    m.tiled = o.tiled;
    m.cols = o.cols;
    m.rows = o.rows;
    m.band = o.band;
    m.size = o.size;
    m.has_size = true;
    m.width = o.width;
    m.width_code = o.width_code;
    m.has_width = true;
    m.ax = o.ax;
    m.ay = o.ay;
    m.has_at = o.has_at;
    m.sink = o.sink;
  }
  if (const Prop* p = s.prop("tiles")) {
    m.tiled = true;
    m.cols = integer(p->value.exprs[0], "the number of tile columns", 1, 16);
    m.rows = integer(p->value.exprs[1], "the number of tile rows", 1, 16);
  }
  if (const Prop* p = s.prop("band")) {
    if (!m.tiled) fail(p->pos, "'band' is for tiles: give 'tiles C x R' too");
    m.band = integer(p->value.exprs[0], "the band", 1);
  }
  if (m.tiled && m.band == 0 && m.cols * m.rows > 1) fail(s.pos, "tiles share a band of cells with their neighbours: add 'band 8' (cells of the effect's grid)");
  if (const Prop* p = s.prop("size")) {
    m.size = integer(p->value.exprs[0], "the size", 8, 4096);
    m.has_size = true;
  }
  if (!m.has_size) fail(s.pos, std::format("module '{}' needs a size (pixels of its tile, as in 'size 192')", s.name));
  if (const Prop* p = s.prop("width")) {  // a single module's may change over time (it is then at least a world pixel)
    Ctx c;
    c.what = "the width of tiles";
    c.constant = m.tiled;
    m.width_code = code(p->value.exprs[0], c);
    if (!m.width_code.varying) {
      m.width = constant(p->value.exprs[0], "the width");
      if (!(m.width > 0.f)) fail(p->pos, "the width must be positive");
    }
    m.has_width = true;
  }
  if (!m.has_width) m.width = static_cast<float>(m.size);
  if (const Prop* p = s.prop("sink")) m.sink = constant(p->value.exprs[0], "the sink");
  if (const Prop* p = s.prop("feather")) m.feather = constant(p->value.exprs[0], "the feather");
  if (const Prop* p = s.prop("glow")) {
    if (p->value.exprs.size() != 3) fail(p->pos, "a glow has three values (red, green, blue)");
    for (int i = 0; i < 3; ++i) {
      m.glow[zs(i)] = constant(p->value.exprs[zs(i)], "a glow");
      if (!(m.glow[zs(i)] >= 0.f)) fail(p->value.exprs[zs(i)].pos, "a glow's colour cannot be negative");
    }
    m.glows = true;
  }
  if (const Prop* p = s.prop("shards")) {
    m.shards = constant(p->value.exprs[0], "the shards' length");
    if (m.shards != 0.f && !(m.shards >= 1.f)) fail(p->pos, "a shard lasts a second or more (0: none)");
    if (m.shards > 0.f && m.tiled) fail(p->pos, "tiles play one continuous rollout: shards are for single modules");
  }
  if (const Prop* p = s.prop("at")) {
    Ctx c;
    c.what = "the place of tiles";
    c.constant = m.tiled;
    m.ax = code(p->value.exprs[0], c);
    m.ay = code(p->value.exprs[1], c);
    m.has_at = true;
  }
  if (const Prop* p = s.prop("look")) {
    const Value& v = p->value;
    if (v.names[0] == "learned") {
      if (v.names.size() > 1) fail(v.pos, "the learned renderer cannot be blended with a shader look");
    } else {
      m.look_a = look_index(v.names[0], v.pos);
      if (v.names.size() > 1) {
        m.look_b = look_index(v.names[1], v.pos);
        Ctx c;
        c.what = "the look's blend";
        m.look_w = code(v.exprs[0], c);
      }
    }
  }
  if (const Prop* p = s.prop("controls")) {
    for (const Expr& e : p->value.exprs) m.controls.push_back(constant(e, "a control"));
    m.declared_controls = true;
  }
  if (const Prop* p = s.prop("opacity")) {
    Ctx c;
    c.what = "the opacity";
    m.opacity = code(p->value.exprs[0], c);
  }
  if (const Prop* p = s.prop("start")) {
    m.start = integer(p->value.exprs[0], "the start point", 0);
    m.has_start = true;
  }
  if (const Prop* p = s.prop("seed")) m.seed = static_cast<std::uint64_t>(integer(p->value.exprs[0], "the seed", 0, 1 << 25));
  if (const Prop* p = s.prop("empty")) {
    if (!m.has_start) fail(p->pos, "'empty' needs 'start N': the module starts with nothing in it, at that start point's age");
    m.empty = true;
  }
  if (const Prop* p = s.prop("waiting")) {
    if (!m.has_start) fail(p->pos, "'waiting' needs 'start N': the module is started now and woken later");
    if (m.tiled) fail(p->pos, "tiles cannot wait (wake works on single modules)");
    m.waiting = true;
  }
  for (const Prop& p : s.props) {
    if (module_key(p.key)) continue;
    Ctx c;
    c.what = "a control";
    m.named.push_back({p.key, p.pos, code(p.value.exprs[0], c)});
    m.declared_controls = true;
  }
}

constexpr std::string_view kTrailKeys[] = {"size",  "width", "band", "tiles",  "at",     "sink",     "base", "rows",  "look",
                                           "controls", "opacity", "seed", "age", "hold", "fade",     "heat",     "soot", "radius",
                                           "kick",  "velocity", "jump", "quiet", "feather", "glow", "if"};

void Compiler::trail(const Statement& s, ModC& m) {
  m.trail = true;
  for (std::size_t i = 0; i < prog_.effects.size(); ++i)
    if (prog_.effects[i].name == s.kind) m.effect = static_cast<int>(i);
  if (m.effect < 0) {
    std::vector<std::string> effects;
    for (const EffectC& e : prog_.effects) effects.push_back(e.name);
    const std::string hint = did_you_mean(s.kind, effects);
    fail(s.pos, hint.empty() ? std::format("unknown effect '{}' (declare it with 'effect {} = \"FILE\"')", s.kind, s.kind) : std::format("unknown effect '{}'{}", s.kind, hint));
  }
  ModC::TrailC& T = m.tr;
  const Prop* size = s.prop("size");
  if (!size) fail(s.pos, std::format("trail '{}' needs a size (pixels of its tiles, as in 'size 64')", s.name));
  m.size = integer(size->value.exprs[0], "the size", 8, 4096);
  m.has_size = true;
  m.width = static_cast<float>(m.size);
  if (const Prop* p = s.prop("width")) {
    m.width = constant(p->value.exprs[0], "the width of a trail's tiles");
    if (!(m.width > 0.f)) fail(p->pos, "the width must be positive");
  }
  m.has_width = true;
  if (const Prop* p = s.prop("band")) T.band = integer(p->value.exprs[0], "the band", 0);
  if (const Prop* p = s.prop("tiles")) T.count = integer(p->value.exprs[0], "the number of a trail's tiles", 1, 64);
  if (const Prop* p = s.prop("rows")) T.rows = integer(p->value.exprs[0], "the rows of tiles", 1);
  const Prop* at = s.prop("at");
  if (!at) fail(s.pos, std::format("trail '{}' needs its source: 'at (x, y)', which may move", s.name));
  {
    Ctx c;
    c.what = "the trail's source";
    m.ax = code(at->value.exprs[0], c);
    m.ay = code(at->value.exprs[1], c);
    m.has_at = true;
  }
  if (const Prop* p = s.prop("sink")) m.sink = constant(p->value.exprs[0], "the sink");
  if (const Prop* p = s.prop("base")) {
    T.base = constant(p->value.exprs[0], "the base");
    T.has_base = true;
  }
  if (const Prop* p = s.prop("feather")) m.feather = constant(p->value.exprs[0], "the feather");
  const auto seconds = [&](const char* key, float& v, bool positive) {
    if (const Prop* p = s.prop(key)) {
      v = constant(p->value.exprs[0], std::format("'{}'", key));
      if (positive ? !(v > 0.f) : !(v >= 0.f)) fail(p->pos, std::format("'{}' must be {}", key, positive ? "positive" : "zero or more"));
    }
  };
  seconds("age", T.age, false);
  seconds("hold", T.hold, false);
  seconds("fade", T.fade, true);
  seconds("jump", T.jump, true);
  if (const Prop* p = s.prop("quiet")) {
    for (int i = 0; i < 4; ++i) T.quiet[zs(i)] = integer(p->value.exprs[zs(i)], "a cell", 0, 4096);
    T.has_quiet = true;
  }
  if (const Prop* p = s.prop("glow")) {
    if (p->value.exprs.size() != 3) fail(p->pos, "a glow has three values (red, green, blue)");
    for (int i = 0; i < 3; ++i) {
      m.glow[zs(i)] = constant(p->value.exprs[zs(i)], "a glow");
      if (!(m.glow[zs(i)] >= 0.f)) fail(p->value.exprs[zs(i)].pos, "a glow's colour cannot be negative");
    }
    m.glows = true;
  }
  if (const Prop* p = s.prop("look")) {
    const Value& v = p->value;
    if (v.names[0] == "learned") {
      if (v.names.size() > 1) fail(v.pos, "the learned renderer cannot be blended with a shader look");
    } else {
      m.look_a = look_index(v.names[0], v.pos);
      if (v.names.size() > 1) {
        m.look_b = look_index(v.names[1], v.pos);
        Ctx c;
        c.what = "the look's blend";
        m.look_w = code(v.exprs[0], c);
      }
    }
  }
  if (const Prop* p = s.prop("controls")) {
    for (const Expr& e : p->value.exprs) m.controls.push_back(constant(e, "a control"));
    m.declared_controls = true;
  }
  if (const Prop* p = s.prop("opacity")) {
    Ctx c;
    c.what = "the opacity";
    m.opacity = code(p->value.exprs[0], c);
  }
  if (const Prop* p = s.prop("seed")) m.seed = static_cast<std::uint64_t>(integer(p->value.exprs[0], "the seed", 0, 1 << 25));
  Ctx c;
  const auto expr = [&](const char* key, Code& out) {
    if (const Prop* p = s.prop(key)) {
      const std::string what = std::format("'{}'", key);
      c.what = what;
      out = code(p->value.exprs[0], c);
    }
  };
  expr("heat", T.heat);
  expr("soot", T.soot);
  expr("radius", T.radius);
  expr("kick", T.kick);
  expr("if", T.on);
  if (const Prop* p = s.prop("velocity")) {
    c.what = "the velocity";
    T.u = code(p->value.exprs[0], c);
    T.v = code(p->value.exprs[1], c);
  }
  for (const Prop& p : s.props) {  // the effect's controls, by name
    if (std::ranges::find(kTrailKeys, p.key) != std::end(kTrailKeys)) continue;
    Ctx cc;
    cc.what = "a control";
    m.named.push_back({p.key, p.pos, code(p.value.exprs[0], cc)});
    m.declared_controls = true;
  }
}

void Compiler::field(const Statement& s) {
  FieldC f;
  f.name = s.name;
  f.pos = s.pos;
  using K = ForceField::Kind;
  static constexpr std::pair<std::string_view, K> kinds[] = {{"ceiling", K::ceiling}, {"vortex", K::vortex}, {"wind", K::wind}, {"gust", K::gust},
                                                             {"ring", K::ring},       {"attract", K::attract}, {"heat", K::heat}, {"cold", K::cold},
                                                             {"lamp", K::lamp}};
  for (const auto& [n, k] : kinds)
    if (n == s.kind) f.kind = k;
  const std::string what = std::format("a {} field", s.kind);
  switch (f.kind) {
    case K::ceiling: only(s, {"level", "soft", "damping", "on", "weight", "if", "from", "until"}, what); break;
    case K::vortex: only(s, {"at", "radius", "strength", "on", "weight", "if", "from", "until"}, what); break;
    case K::wind: only(s, {"velocity", "on", "weight", "if", "from", "until"}, what); break;
    case K::gust: only(s, {"velocity", "amount", "scale", "rate", "seed", "on", "weight", "if", "from", "until"}, what); break;
    case K::ring: only(s, {"at", "direction", "radius", "core", "strength", "on", "weight", "if", "from", "until"}, what); break;
    case K::attract: only(s, {"at", "radius", "strength", "swirl", "on", "weight", "if", "from", "until"}, what); break;
    case K::heat: only(s, {"at", "radius", "strength", "on", "weight", "if", "from", "until"}, what); break;
    case K::cold: only(s, {"at", "radius", "strength", "steam", "on", "weight", "if", "from", "until"}, what); break;
    case K::lamp: only(s, {"at", "radius", "strength", "colour", "weight", "if", "from", "until"}, what); break;
  }
  const auto need = [&](std::string_view key) {
    if (!s.prop(key)) fail(s.pos, std::format("{} needs '{}'", what, key));
  };
  switch (f.kind) {
    case K::ceiling: need("level"); break;
    case K::wind:
    case K::gust: need("velocity"); break;
    default:
      need("at");
      need("strength");
      break;
  }
  if (f.kind != K::lamp) need("on");  // a lamp lights the scene; it acts on no module
  Ctx c;
  take(f.p, s, "at", p_x, p_y, c);
  take(f.p, s, "level", p_level, kParams, c);
  take(f.p, s, "soft", p_soft, kParams, c);
  take(f.p, s, "damping", p_damping, kParams, c);
  take(f.p, s, "radius", p_radius, kParams, c);
  take(f.p, s, "strength", p_strength, kParams, c);
  take(f.p, s, "velocity", p_u, p_v, c);
  take(f.p, s, "direction", p_u, p_v, c);
  take(f.p, s, "amount", p_amount, kParams, c);
  take(f.p, s, "scale", p_scale, kParams, c);
  take(f.p, s, "rate", p_rate, kParams, c);
  take(f.p, s, "seed", p_seed, kParams, c);
  take(f.p, s, "core", p_core, kParams, c);
  take(f.p, s, "swirl", p_swirl, kParams, c);
  take(f.p, s, "steam", p_steam, kParams, c);
  take(f.p, s, "weight", p_weight, kParams, c);
  take(f.p, s, "if", p_if, kParams, c);
  take(f.p, s, "from", p_from, kParams, c);
  take(f.p, s, "until", p_until, kParams, c);
  if (const Prop* col = s.prop("colour")) {
    if (col->value.exprs.size() != 3) fail(col->pos, "a colour has three values (red, green, blue)");
    Ctx cc;
    cc.what = "a colour";
    f.p[p_cr] = code(col->value.exprs[0], cc);
    f.p[p_cg] = code(col->value.exprs[1], cc);
    f.p[p_cb] = code(col->value.exprs[2], cc);
  }
  const bool flow = f.kind == K::wind || f.kind == K::gust || f.kind == K::vortex || f.kind == K::ring || f.kind == K::attract;
  if (const Prop* on = s.prop("on")) f.mods = modules_of(on->value.names, on->value.pos, flow, &f.particles);
  prog_.fields.push_back(std::move(f));
}

void Compiler::emitter(const Statement& s) {
  EmitC e;
  e.kind = s.kind;
  e.pos = s.pos;
  const std::string what = std::format("emit {}", s.kind);
  const auto need = [&](std::string_view key) {
    if (!s.prop(key)) fail(s.pos, std::format("{} needs '{}'", what, key));
  };
  if (fx::emitter_of(s.kind, e.fx_kind)) {  // the procedural elements of fx_emit.cpp (docs/COMPOSE.md §4.10)
    e.is_fx = true;
    const auto allowed = fx::keys(e.fx_kind);
    for (const Prop& pr : s.props) {
      if (pr.key == "from" || pr.key == "until" || pr.key == "if" || std::ranges::find(allowed, pr.key) != allowed.end()) continue;
      std::string list;
      for (const std::string_view k : allowed) list += std::format("{}, ", k);
      fail(pr.pos, std::format("'{}' does not apply to {} (it takes: {}from, until, if)", pr.key, what, list));
    }
    for (const std::string_view k : fx::needs(e.fx_kind)) need(k);
    if (e.fx_kind == fx::Emitter::glints && !s.prop("at") && !(s.prop("in") && s.prop("size"))) fail(s.pos, "emit glints needs 'at' (and 'spread'), or 'in' and 'size'");
    if (e.fx_kind == fx::Emitter::glints && s.prop("at") && s.prop("in")) fail(s.prop("in")->pos, "emit glints takes 'at' (and 'spread') or 'in' and 'size', not both");
    for (const Prop& pr : s.props) {
      fx::Slot slot;
      if (!fx::slot_of(pr.key, slot)) continue;
      if (static_cast<int>(pr.value.exprs.size()) != slot.n) fail(pr.pos, std::format("'{}' has {} values", pr.key, slot.n));
      Ctx c;
      const std::string w = std::format("'{}'", pr.key);
      c.what = w;
      for (int i = 0; i < slot.n; ++i) e.fxv[zs(slot.first + i)] = code(pr.value.exprs[zs(i)], c);
    }
    Ctx c;
    take(e.p, s, "from", p_from, kParams, c);
    take(e.p, s, "until", p_until, kParams, c);
    take(e.p, s, "if", p_if, kParams, c);
    prog_.emitters.push_back(std::move(e));
    return;
  }
  if (s.kind == "sparks") {
    only(s, {"along", "from", "until", "count", "if"}, what);
    need("along");
    const Prop* a = s.prop("along");
    Ctx c;
    c.what = "'along'";
    e.p[p_x] = code(a->value.exprs[0], c);
    e.p[p_y] = code(a->value.exprs[1], c);
    e.p[p_x1] = code(a->value.exprs[2], c);
    e.p[p_y1] = code(a->value.exprs[3], c);
  } else if (s.kind == "embers") {
    only(s, {"on", "tries", "chance", "spread", "from", "until", "if"}, what);
    need("on");
    e.mods = modules_of(s.prop("on")->value.names, s.prop("on")->value.pos, false);
  } else if (s.kind == "flakes") {
    only(s, {"in", "size", "tries", "chance", "soot", "from", "until", "if"}, what);
    need("in");
    need("size");
  } else if (s.kind == "arc") {  // procedural: a lightning bolt or an electric arc between two points
    only(s, {"along", "colour", "width", "brightness", "jitter", "branches", "rate", "seed", "from", "until", "if"}, what);
    need("along");
    const Prop* a = s.prop("along");
    Ctx c;
    c.what = "'along'";
    e.p[p_x] = code(a->value.exprs[0], c);
    e.p[p_y] = code(a->value.exprs[1], c);
    e.p[p_x1] = code(a->value.exprs[2], c);
    e.p[p_y1] = code(a->value.exprs[3], c);
  } else if (s.kind == "rain" || s.kind == "snow" || s.kind == "ash" || s.kind == "leaves") {  // procedural weather
    // rain and snow also take the director's physics (docs/WEATHER.md §4.1): `rate` (mm/h) sizes, speeds and counts them
    if (s.kind == "rain") {
      only(s, {"in", "size", "density", "wind", "radius", "colour", "brightness", "flow", "wash", "seed", "from", "until", "if", "rate", "scale", "shutter", "depth"}, what);
    } else if (s.kind == "snow") {
      only(s, {"in", "size", "density", "wind", "radius", "colour", "brightness", "flow", "melt", "seed", "from", "until", "if", "rate", "scale", "depth"}, what);
    } else if (s.kind == "ash") {
      only(s, {"in", "size", "density", "wind", "radius", "colour", "brightness", "flow", "glowing", "seed", "from", "until", "if"}, what);
    } else {
      only(s, {"in", "size", "density", "wind", "radius", "colour", "brightness", "flow", "seed", "from", "until", "if"}, what);
    }
    need("in");
    need("size");
    Ctx c;
    take(e.p, s, "density", p_density, kParams, c);
    take(e.p, s, "wind", p_wu, p_wv, c);
    take(e.p, s, "flow", p_flowk, kParams, c);
    take(e.p, s, "wash", p_wash, kParams, c);
    take(e.p, s, "melt", p_melt, kParams, c);
    take(e.p, s, "glowing", p_glowing, kParams, c);
    take(e.p, s, "scale", p_scale, kParams, c);
    take(e.p, s, "shutter", p_shutter, kParams, c);
    take(e.p, s, "depth", p_depth, kParams, c);
  } else if (s.kind == "ring") {  // procedural: a glowing ring (an ellipse when squashed: a shock ring on the ground)
    only(s, {"at", "radius", "squash", "colour", "width", "brightness", "from", "until", "if"}, what);
    need("at");
    need("radius");
  } else {  // glow, procedural: a soft disc of light, or `count` of them wandering within `spread`
    only(s, {"at", "radius", "colour", "brightness", "count", "spread", "rate", "seed", "from", "until", "if"}, what);
    need("at");
    need("radius");
  }
  if (const Prop* col = s.prop("colour")) {
    if (col->value.exprs.size() != 3) fail(col->pos, "a colour has three values (red, green, blue)");
    Ctx cc;
    cc.what = "a colour";
    e.p[p_cr] = code(col->value.exprs[0], cc);
    e.p[p_cg] = code(col->value.exprs[1], cc);
    e.p[p_cb] = code(col->value.exprs[2], cc);
  }
  Ctx c;
  take(e.p, s, "from", p_from, kParams, c);
  take(e.p, s, "until", p_until, kParams, c);
  take(e.p, s, "count", p_count, kParams, c);
  take(e.p, s, "tries", p_tries, kParams, c);
  take(e.p, s, "chance", p_chance, kParams, c);
  take(e.p, s, "spread", p_spread, kParams, c);
  take(e.p, s, "in", p_x, p_y, c);
  take(e.p, s, "size", p_w, p_h, c);
  take(e.p, s, "soot", p_soot, kParams, c);
  take(e.p, s, "if", p_if, kParams, c);
  take(e.p, s, "at", p_x, p_y, c);
  take(e.p, s, "radius", p_radius, kParams, c);
  take(e.p, s, "squash", p_squash, kParams, c);
  take(e.p, s, "width", p_width, kParams, c);
  take(e.p, s, "brightness", p_bright, kParams, c);
  take(e.p, s, "jitter", p_jitter, kParams, c);
  take(e.p, s, "branches", p_branches, kParams, c);
  take(e.p, s, "rate", p_rate, kParams, c);
  take(e.p, s, "seed", p_seed, kParams, c);
  prog_.emitters.push_back(std::move(e));
}

// Water (docs/EFFECTS_WATER.md): what each kind takes and needs; counts and capacities are constants (they size it when
// the scene is made); `onto` names a pool, `mist` a module.
void Compiler::water(const Statement& s, WaterC& w) {
  const std::string what = std::format("water {}", s.kind);
  const auto need = [&](std::string_view key) {
    if (!s.prop(key)) fail(s.pos, std::format("{} needs '{}'", what, key));
  };
  switch (w.kind) {
    case WaterKind::pool:
      only(s, {"along", "depth", "cells", "speed", "damping", "drift", "rate", "surface", "swell", "wavelength", "period", "colour", "clarity", "churn",
               "brightness", "capacity", "seed", "from", "until", "if"}, what);
      need("along");
      break;
    case WaterKind::sheet:
      only(s, {"along", "flow", "velocity", "strands", "capacity", "breakup", "gravity", "wind", "floor", "onto", "mist", "mist_amount",
               "mist_heat", "colour", "brightness", "seed", "from", "until", "if"}, what);
      need("along");
      break;
    case WaterKind::jet:
      only(s, {"at", "speed", "angle", "spread", "fan", "jets", "pulse", "flow", "capacity", "breakup", "gravity", "wind", "floor", "onto",
               "mist", "mist_amount", "mist_heat", "colour", "brightness", "seed", "from", "until", "if"}, what);
      need("at");
      need("speed");
      break;
    case WaterKind::drips:
      only(s, {"along", "rate", "radius", "capacity", "gravity", "wind", "floor", "onto", "mist", "mist_amount", "mist_heat", "colour",
               "brightness", "seed", "from", "until", "if"}, what);
      need("along");
      break;
    case WaterKind::bubbles:
      only(s, {"in", "size", "rate", "radius", "speed", "capacity", "onto", "colour", "brightness", "seed", "from", "until", "if"}, what);
      need("in");
      need("size");
      break;
    case WaterKind::shaft:
      only(s, {"in", "size", "density", "speed", "wind", "sway", "gust", "capacity", "mist", "mist_amount", "mist_heat", "colour", "clarity",
               "brightness", "seed", "from", "until", "if"}, what);
      need("in");
      need("size");
      break;    case WaterKind::spray:  // drops thrown off a learned water effect's water (the bus's soot)
      only(s, {"in", "size", "density", "threshold", "speed", "radius", "capacity", "gravity", "wind", "floor", "onto", "mist", "mist_amount",
               "mist_heat", "colour", "brightness", "seed", "from", "until", "if"}, what);
      need("in");
      need("size");
      break;
  }
  if (const Prop* p = s.prop("strands")) w.strands = integer(p->value.exprs[0], "the strands", 1, 256);
  if (const Prop* p = s.prop("jets")) w.strands = integer(p->value.exprs[0], "the jets", 1, 64);
  else if (w.kind == WaterKind::jet) w.strands = 1;
  if (const Prop* p = s.prop("cells")) w.cells = integer(p->value.exprs[0], "the cells", 8, 4096);
  const int cap_default = w.kind == WaterKind::sheet || w.kind == WaterKind::jet ? 72 : w.kind == WaterKind::drips ? 64 : w.kind == WaterKind::shaft ? 2048 : 256;
  const int cap = s.prop("capacity") ? integer(s.prop("capacity")->value.exprs[0], "the capacity", 1, 65536) : cap_default;
  if (w.kind == WaterKind::sheet || w.kind == WaterKind::jet) {
    w.nodes = cap;
    if (w.nodes < 8 || w.nodes > 512) fail(s.prop("capacity")->pos, "a strand holds 8 to 512 nodes (frames in flight)");
  } else {
    w.count = cap;
  }
  if (const Prop* p = s.prop("seed")) w.seed = static_cast<std::uint64_t>(integer(p->value.exprs[0], "the seed", 0));
  if (const Prop* p = s.prop("onto")) w.onto = water_index(p->value.names[0], p->value.pos, true);
  if (const Prop* p = s.prop("mist")) {
    w.mist = module_index(p->value.names[0], p->value.pos);
    if (prog_.modules[zs(w.mist)].trail) fail(p->value.pos, std::format("'{}' is a trail: mist goes into a module", p->value.names[0]));
  }
  Ctx c;
  const auto take2 = [&](std::string_view key, WP a, WP b) {
    const Prop* pr = s.prop(key);
    if (!pr) return;
    Ctx cc = c;
    const std::string k = std::format("'{}'", key);
    cc.what = k;
    w.p[a] = code(pr->value.exprs[0], cc);
    if (b != kWaterParams) w.p[b] = code(pr->value.exprs[1], cc);
  };
  if (const Prop* a = s.prop("along")) {
    Ctx cc;
    cc.what = "'along'";
    for (int k = 0; k < 4; ++k) w.p[zs(w_x0 + k)] = code(a->value.exprs[zs(k)], cc);
  }
  take2("at", w_x0, w_y0);
  take2("in", w_x0, w_y0);
  take2("size", w_x1, w_y1);
  for (const auto& [key, a, b] : std::initializer_list<std::tuple<std::string_view, WP, WP>>{
           {"flow", w_flow, kWaterParams},       {"velocity", w_vx, w_vy},           {"speed", w_speed, kWaterParams},
           {"angle", w_angle, kWaterParams},     {"spread", w_spread, kWaterParams}, {"fan", w_fan, kWaterParams},
           {"pulse", w_pulse, kWaterParams},     {"breakup", w_breakup, kWaterParams}, {"gravity", w_gravity, kWaterParams},
           {"wind", w_wu, w_wv},                 {"rate", w_rate, kWaterParams},     {"radius", w_radius, kWaterParams},
           {"depth", w_depth, kWaterParams},     {"damping", w_damping, kWaterParams}, {"drift", w_drift, kWaterParams},
           {"surface", w_surface, kWaterParams}, {"swell", w_swell, kWaterParams},   {"wavelength", w_wavelength, kWaterParams},
           {"period", w_period, kWaterParams},   {"density", w_density, kWaterParams}, {"sway", w_sway, kWaterParams},
           {"gust", w_gust, kWaterParams},       {"floor", w_floor, kWaterParams},   {"mist_amount", w_mist_amount, kWaterParams},
           {"mist_heat", w_mist_heat, kWaterParams}, {"brightness", w_brightness, kWaterParams}, {"clarity", w_clarity, kWaterParams},
           {"from", w_from, kWaterParams},       {"until", w_until, kWaterParams},   {"if", w_if, kWaterParams},
           {"threshold", w_threshold, kWaterParams}, {"churn", w_churn, kWaterParams}}) {
    take2(key, a, b);
  }
  if (const Prop* col = s.prop("colour")) {
    if (col->value.exprs.size() != 3) fail(col->pos, "a colour has three values (red, green, blue)");
    Ctx cc;
    cc.what = "a colour";
    w.p[w_cr] = code(col->value.exprs[0], cc);
    w.p[w_cg] = code(col->value.exprs[1], cc);
    w.p[w_cb] = code(col->value.exprs[2], cc);
  }
}

ActC Compiler::action(const Statement& a, const Ctx& c, bool every, int limit) {
  ActC out;
  out.kind = a.keyword;
  out.pos = a.pos;
  const std::string& k = a.keyword;
  if (every && (k == "hand_over" || k == "start" || k == "wake" || k == "shock" || k == "scorch" || k == "burst" || k == "splash")) {
    fail(a.pos, std::format("'{}' happens once: put it in a rule (at ..., when ...), not under 'every frame'", k));
  }
  if (k == "wave") {  // its subject is a pool
    out.water = water_index(a.subjects[0], a.pos, true);
  } else {
    for (const std::string& n : a.subjects) out.subjects.push_back(module_index(n, a.pos));
  }
  out.targets = modules_of(a.targets, a.pos, false);
  const int copies = std::min(limit, kMaxShocks);
  if (k == "transfer") {
    for (const int t : out.targets)
      if (t == out.subjects[0]) fail(a.pos, "a module cannot transfer into itself");
  } else if (k == "hand_over") {
    const ModC& f = prog_.modules[zs(out.subjects[0])];
    const ModC& t = prog_.modules[zs(out.targets[0])];
    if (f.trail || t.trail) fail(a.pos, std::format("'{}' is a trail: its tiles cannot be handed over", f.trail ? f.name : t.name));
    if (f.tiled != t.tiled || f.cols != t.cols || f.rows != t.rows) fail(a.pos, std::format("'{}' and '{}' must have the same tiles to hand over", f.name, t.name));
  } else if (k == "start") {
    const ModC& m = prog_.modules[zs(out.subjects[0])];
    if (m.trail) fail(a.pos, std::format("'{}' is a trail: its tiles start by themselves where its source goes", m.name));
    if (const Prop* p = a.prop("from")) out.from = integer(p->value.exprs[0], "the start point", 0);
    if (const Prop* p = a.prop("in")) {
      if (!m.tiled) fail(p->pos, "'in (column, row)' picks the tile that starts from the start point: only for tiles");
      out.has_in = true;
      out.in_c = integer(p->value.exprs[0], "the tile's column", 0, m.cols - 1);
      out.in_r = integer(p->value.exprs[1], "the tile's row", 0, m.rows - 1);
    }
    if (a.prop("at") && m.tiled) fail(a.prop("at")->pos, "tiles stay where they are declared");
    out.empty = a.prop("empty") != nullptr;
    if (out.empty && out.has_in) fail(a.prop("in")->pos, "an empty start has no tile with the start point");
  } else if (k == "wake") {
    for (const int i : out.subjects) {
      const ModC& m = prog_.modules[zs(i)];
      if (!m.waiting) fail(a.pos, std::format("'{}' is not waiting: give it 'start N, waiting' to wake it later", m.name));
    }
  } else if (k == "suppress") {
    const ModC& m = prog_.modules[zs(out.subjects[0])];
    const Prop* cells = a.prop("cells");
    if (!cells) fail(a.pos, "suppress needs 'cells (x0, y0) to (x1, y1)' (cells of the effect's grid, y up)");
    for (int i = 0; i < 4; ++i) out.cells[zs(i)] = integer(cells->value.exprs[zs(i)], "a cell", 0, 4096);
    if (const Prop* p = a.prop("tiles")) {
      if (!m.tiled) fail(p->pos, "'tiles' picks tiles of a tiled module");
      for (std::size_t i = 0; i < p->value.exprs.size(); i += 2) {
        out.tiles.push_back({integer(p->value.exprs[i], "a tile's column", 0, m.cols - 1), integer(p->value.exprs[i + 1], "a tile's row", 0, m.rows - 1)});
      }
    }
  } else if (k == "shock" || k == "scorch" || k == "burst" || k == "splash" || k == "wave") {
    if (!a.prop("at")) fail(a.pos, std::format("{} needs 'at (x, y)'", k));
    if (k == "shock") prog_.shock_capacity += copies;
    if (k == "scorch") prog_.scorch_capacity += copies;
    if (k == "splash") {
      prog_.splash_capacity += copies;
      if (const Prop* p = a.prop("onto")) out.water = water_index(p->value.names[0], p->value.pos, true);
      if (const Prop* p = a.prop("mist")) {
        out.mist = module_index(p->value.names[0], p->value.pos);
        if (prog_.modules[zs(out.mist)].trail) fail(p->value.pos, std::format("'{}' is a trail: mist goes into a module", p->value.names[0]));
      }
    }
  }
  for (const auto& [key, pa, pb] : std::initializer_list<std::tuple<std::string_view, P, P>>{
           {"at", p_x, p_y},         {"fraction", p_fraction, kParams}, {"top", p_top, kParams},     {"heat", p_heat, kParams},
           {"soot", p_soot, kParams}, {"gain", p_gain, kParams},        {"seed", p_seed, kParams},   {"speed", p_speed, kParams},
           {"decay", p_decay, kParams}, {"amp", p_amp, kParams},        {"width", p_width, kParams}, {"radius", p_radius, kParams},
           {"glow", p_glow, kParams}, {"embers", p_embers, kParams},    {"debris", p_debris, kParams}, {"if", p_if, kParams},
           {"sparks", p_sparks, kParams}, {"direction", p_dx, p_dy},    {"cone", p_cone, kParams}, {"spatter", p_spatter, kParams},
           {"shards", p_shards, kParams}, {"grain", p_grain, kParams},  {"stain", p_stain, kParams},     {"floor", p_floor, kParams},
           {"power", p_power, kParams}, {"mist_amount", p_mist_amount, kParams}, {"mist_heat", p_mist_heat, kParams},
           {"amount", p_amount, kParams}}) {
    take(out.p, a, key, pa, pb, c);
  }
  if (const Prop* col = a.prop("colour")) {  // a burst's spatter and shards
    if (col->value.exprs.size() != 3) fail(col->pos, "a colour has three values (red, green, blue)");
    Ctx cc = c;
    cc.what = "a colour";
    out.p[p_cr] = code(col->value.exprs[0], cc);
    out.p[p_cg] = code(col->value.exprs[1], cc);
    out.p[p_cb] = code(col->value.exprs[2], cc);
  }
  return out;
}

// The weather director (weather.hpp, docs/WEATHER.md). Its seed and direction are constants. `regime` is where it heads:
// an expression (the game may drive it through an input), whose regime names are values (clear 0 ... dust 6); with a
// schedule, it is where the weather starts, a constant, and the schedule's lines say where it heads from when.
void Compiler::weather(const Statement& s, WeatherC& w) {
  if (const Prop* p = s.prop("seed")) w.seed = static_cast<std::uint64_t>(integer(p->value.exprs[0], "a weather's seed", 0, 1 << 25));
  if (const Prop* p = s.prop("direction")) {
    const float d = constant(p->value.exprs[0], "a weather's direction");
    if (d != 1.f && d != -1.f) fail(p->value.pos, "a weather's direction is 1 (the wind blows to the right, +x) or -1 (to the left)");
    w.direction = d;
  }
  const bool scheduled = s.kind == "schedule";
  const auto regime_number = [&](float v, Pos at) {
    if (!(v == std::floor(v)) || v < 0.f || v >= static_cast<float>(wx::kRegimes)) {
      fail(at, std::format("a regime is clear, overcast, rain, storm, snow, fog or dust (or its number, 0 to 6), not {}", v));
    }
    return static_cast<int>(v);
  };
  if (const Prop* p = s.prop("regime")) {
    Ctx c;
    c.regimes = true;
    c.constant = scheduled;
    c.what = scheduled ? "the regime a scheduled weather starts in" : "'regime'";
    w.regime = code(p->value.exprs[0], c);
    if (scheduled || !w.regime.varying) {
      float slots[s_first] = {0.f, prog_.length, prog_.fps, prog_.ground, 0.f, 0.f};
      Env env;
      env.slots = slots;
      w.start = regime_number(run(w.regime, env), p->value.pos);
    }
  }
  if (const Prop* p = s.prop("over")) {
    Ctx c;
    c.what = "'over'";
    w.over = code(p->value.exprs[0], c);
  }
  float last = -kNever;
  for (const Statement& line : s.body) {
    WeatherC::Line l;
    wx::Regime r{};
    wx::regime_of(line.kind, r);  // (the parser checked the name)
    l.regime = static_cast<int>(r);
    const Prop* at = line.prop("at");
    if (!at) fail(line.pos, std::format("say when the weather heads for {}: '{} at T' (seconds)", line.kind, line.kind));
    l.at = constant(at->value.exprs[0], "a schedule's time");
    if (!(l.at >= 0.f)) fail(at->value.pos, "a schedule's times are from 0 on");
    if (l.at < last) fail(at->value.pos, std::format("a schedule's lines come in the order of their times ({} is before the line above's {})", l.at, last));
    last = l.at;
    l.over = std::numeric_limits<float>::quiet_NaN();
    if (const Prop* o = line.prop("over")) {
      l.over = constant(o->value.exprs[0], "a schedule's 'over'");
      if (!(l.over >= 0.f)) fail(o->value.pos, "'over' is a time of at least 0 seconds");
    }
    w.schedule.push_back(l);
  }
}

void Compiler::rule(const Statement& s, RuleC& r) {
  r.pos = s.pos;
  r.kind = s.kind;
  r.limit = s.at_most ? integer(*s.at_most, "'at most'", 1) : s.repeat ? INT_MAX : 1;
  Ctx c;
  c.landing = s.kind == "lands";
  c.what = "the condition";
  std::vector<std::string> open;
  if (s.kind == "time") {  // at T: t >= T
    r.cond.ops.push_back({Op::slot, 0, s_t, 0.f});
    emit(s.args[0], c, r.cond, open);
    r.cond.ops.push_back({Op::ge, 0, 0, 0.f});
    r.cond.varying = true;
  } else if (s.kind == "condition") {
    r.cond = code(s.args[0], c);
  } else if (s.kind == "shock") {
    const int n = integer(s.args[0], "the shock's number", 1, kMaxShocks);
    if (!s.reach.empty()) {
      const ModC& m = prog_.modules[zs(module_index(s.reach, s.pos))];
      r.cond.ops.push_back({Op::slot, 0, m.slot + m_x, 0.f});
      r.cond.ops.push_back({Op::slot, 0, m.slot + m_y, 0.f});
    } else {
      emit(s.args[1], c, r.cond, open);
      emit(s.args[2], c, r.cond, open);
    }
    r.cond.ops.push_back({Op::reaches, 0, n, 0.f});
    r.cond.varying = true;
  } else {  // lands
    r.particle = s.subjects[0] == "debris" ? Kind::debris : Kind::ember;
    if (!s.args.empty()) r.cond = code(s.args[0], c);
  }
  for (const Statement& a : s.body) r.actions.push_back(action(a, c, false, r.limit));
}

Program Compiler::compile() {
  // 1. names, so that any statement can use a name declared after it
  std::vector<int> rule_slot;
  std::set<std::string> once;
  for (const Statement& st : s_.statements) {
    const std::string& w = st.keyword;
    if (w == "entry" || w == "param") fail(st.pos, std::format("'{}' belongs in a library catalogue (docs/LIBRARY.md), not in a scene", w));
    if (w == "scene" || w == "bus" || w == "light" || w == "particles" || w == "frame" || w == "camera" || w == "keyframes") {
      if (!once.insert(w).second) fail(st.pos, std::format("'{}' is given twice (put all its settings on one line)", w));
    } else if (w == "effect") {
      declare(st.name, st.name_pos, "effect");
      prog_.effects.push_back({st.name, st.kind, st.pos, {}});
    } else if (w == "look") {
      declare(st.name, st.name_pos, "look");
      prog_.look_names.push_back(st.name);
    } else if (w == "let") {
      declare(st.name, st.name_pos, "value");
      lets_[st.name] = &st;
    } else if (w == "input") {
      declare(st.name, st.name_pos, "input");
      input_slots_[st.name] = prog_.slots;
      prog_.inputs.push_back({st.name, st.pos, prog_.slots++, 0.f});
    } else if (w == "module" || w == "trail") {
      declare(st.name, st.name_pos, "module");
      ModC m;
      m.name = st.name;
      m.trail = w == "trail";  // (known before any expression is compiled: a trail's velocity)
      m.pos = st.pos;
      m.slot = prog_.slots;
      prog_.slots += m_slots;
      prog_.modules.push_back(std::move(m));
    } else if (w == "field") {
      declare(st.name, st.name_pos, "field");
    } else if (w == "weather") {  // (known before any expression is compiled: its members)
      declare(st.name, st.name_pos, "weather");
      WeatherC wc;
      wc.name = st.name;
      wc.pos = st.pos;
      wc.slot = prog_.slots;
      prog_.slots += wx::kMembers;
      prog_.weathers.push_back(std::move(wc));
    } else if (w == "water") {  // (its kind is known now: splashes and waves name pools)
      declare(st.name, st.name_pos, "water");
      WaterC wc;
      wc.name = st.name;
      wc.pos = st.pos;
      static constexpr std::pair<std::string_view, WaterKind> kinds[] = {{"pool", WaterKind::pool}, {"sheet", WaterKind::sheet}, {"jet", WaterKind::jet},
                                                                         {"drips", WaterKind::drips}, {"bubbles", WaterKind::bubbles}, {"shaft", WaterKind::shaft}, {"spray", WaterKind::spray}};
      for (const auto& [n, kk] : kinds)
        if (n == st.kind) wc.kind = kk;
      prog_.waters.push_back(std::move(wc));
    } else if (w == "at" || w == "when") {
      declare(st.name, st.name_pos, "rule");
      rule_slot.push_back(prog_.slots++);
      if (!st.name.empty()) rule_slots_[st.name] = rule_slot.back();
    }
  }
  // 2. the scene, then everything that only needs constants
  for (const Statement& st : s_.statements)
    if (st.keyword == "scene") scene(st);
  std::size_t input_i = 0;
  for (const Statement& st : s_.statements)
    if (st.keyword == "input") prog_.inputs[input_i++].value = constant(st.args[0], "an input's starting value");
  for (const Statement& st : s_.statements) {
    if (st.keyword == "let") {  // a let is checked where it is used; here only that it compiles at all
      Ctx c;
      c.what = "a value";
      c.landing = true;
      (void)code(st.args[0], c);
    }
  }
  std::size_t effect_i = 0, look_i = 0, module_i = 0;
  for (const Statement& st : s_.statements) {
    if (st.keyword == "effect") {
      EffectC& e = prog_.effects[effect_i++];
      for (const Prop& p : st.props) {
        if (p.key == "precision") {
          const std::string& v = p.value.names[0];
          if (v != "float" && v != "int8") fail(p.value.pos, std::format("an effect's precision is 'float' or 'int8', not '{}'", v));
          e.precision = v == "int8" ? 1 : 0;
          continue;
        }
        const float v = constant(p.value.exprs[0], std::format("'{}'", p.key));
        // study O (docs/REPORT.md §16): the detail layer in passes, a whole number of them (1 to 8)
        if (p.key == "passes" && !(v >= 1.f && v <= 8.f && v == std::floor(v)))
          fail(p.value.pos, std::format("an effect's passes are a whole number from 1 to 8, not {}", v));
        // study Q (docs/REPORT.md §21): the fine fields' softening (study D2's DetailSpec::soften), a share from 0 to 1
        if (p.key == "soften" && !(v >= 0.f && v <= 1.f)) fail(p.value.pos, std::format("an effect's soften is from 0 to 1, not {}", v));
        e.detail.push_back({p.key, v});
      }
    } else if (st.keyword == "look") {
      ShaderSpec spec;
      if (st.kind != "shader") {
        const int base = look_index(st.kind, st.pos);
        if (zs(base) >= look_i) fail(st.pos, std::format("look '{}' must be declared before '{}'", st.kind, st.name));
        spec = prog_.looks[zs(base)];
      }
      for (const Prop& p : st.props) {
        const std::string what = std::format("'{}'", p.key);
        if (p.key == "tint") {
          if (p.value.exprs.size() != 3) fail(p.pos, "a tint has three values (red, green, blue)");
          for (int i = 0; i < 3; ++i) spec.tint[zs(i)] = constant(p.value.exprs[zs(i)], "a tint");
          continue;
        }
        const float v = constant(p.value.exprs[0], what);
        if (p.key == "heat_scale") spec.heat_scale = v;
        else if (p.key == "emission") spec.emission = v;
        else if (p.key == "emission_power") spec.emission_power = v;
        else if (p.key == "soot_density") spec.soot_density = v;
        else if (p.key == "soot_albedo") spec.soot_albedo = v;
        else if (p.key == "sky") spec.sky = v;
        else if (p.key == "shadow") spec.shadow = v;
        else if (p.key == "scene_light") spec.scene_light = v;
        else if (p.key == "relief") spec.relief = v;
        else if (p.key == "hue") spec.hue = v;
      }
      if (!(spec.heat_scale > 0.f)) fail(st.pos, "heat_scale must be positive");
      prog_.looks.push_back(spec);
      ++look_i;
    } else if (st.keyword == "module") {
      module(st, prog_.modules[module_i++]);
    } else if (st.keyword == "trail") {
      trail(st, prog_.modules[module_i++]);
    }
  }
  // 3. settings that may change over time
  std::size_t water_i = 0, weather_i = 0;
  for (const Statement& st : s_.statements)
    if (st.keyword == "weather") weather(st, prog_.weathers[weather_i++]);
  for (const Statement& st : s_.statements) {
    Ctx c;
    const auto expr = [&](std::string_view key, Code& out) {
      if (const Prop* p = st.prop(key)) {
        const std::string what = std::format("'{}'", key);
        c.what = what;
        out = code(p->value.exprs[0], c);
      }
    };
    if (st.keyword == "bus") {
      prog_.has_bus = true;
      const Prop* at = st.prop("at");
      const Prop* size = st.prop("size");
      if (!at || !size) fail(st.pos, "the bus needs 'at (x, y)' (its top-left corner) and 'size W x H' (world pixels)");
      prog_.bus_x = constant(at->value.exprs[0], "the bus's corner");
      prog_.bus_y = constant(at->value.exprs[1], "the bus's corner");
      prog_.bus_w = constant(size->value.exprs[0], "the bus's size");
      prog_.bus_h = constant(size->value.exprs[1], "the bus's size");
      if (const Prop* p = st.prop("cell")) prog_.bus_cell = constant(p->value.exprs[0], "the bus's cell");
      if (!(prog_.bus_w > 0.f && prog_.bus_h > 0.f && prog_.bus_cell >= 1.f)) fail(st.pos, "the bus's size must be positive and its cell at least 1");
    } else if (st.keyword == "light") {
      expr("gain", prog_.light_gain);
      if (const Prop* p = st.prop("flash")) {
        if (p->value.exprs.size() != 3) fail(p->pos, "the flash has three values (red, green, blue)");
        c.what = "the flash";
        for (int i = 0; i < 3; ++i) prog_.flash[zs(i)] = code(p->value.exprs[zs(i)], c);
      }
    } else if (st.keyword == "particles") {
      if (const Prop* p = st.prop("capacity")) prog_.capacity = integer(p->value.exprs[0], "the capacity", 1, 1 << 22);
      expr("flow", prog_.flow);
    } else if (st.keyword == "frame") {
      expr("exposure", prog_.exposure);
      expr("fade", prog_.fade);
      expr("haze", prog_.haze);
      expr("bloom", prog_.bloom);
      expr("bloom_threshold", prog_.bloom_threshold);
      expr("daylight", prog_.daylight);  // the day (docs/WEATHER.md §4.6)
      expr("overcast", prog_.overcast);
      expr("visibility", prog_.visibility);
    } else if (st.keyword == "camera") {
      expr("x", prog_.cam_x);
      expr("y", prog_.cam_y);
    } else if (st.keyword == "keyframes") {
      for (const Expr& e : st.args) {
        const float v = constant(e, "a keyframe's time");
        if (v < 0.f || v > prog_.length) fail(e.pos, std::format("keyframe {} is outside the scene (0 to {} s)", v, prog_.length));
        prog_.keyframes.push_back(v);
      }
    } else if (st.keyword == "field") {
      field(st);
    } else if (st.keyword == "emit") {
      emitter(st);
    } else if (st.keyword == "water") {
      water(st, prog_.waters[water_i++]);
    } else if (st.keyword == "at" || st.keyword == "when") {
      RuleC r;
      r.name = st.name.empty() ? std::format("line {}", st.pos.line) : st.name;
      r.slot = rule_slot[prog_.rules.size()];
      rule(st, r);
      prog_.rules.push_back(std::move(r));
    } else if (st.keyword == "every") {
      Ctx ec;
      for (const Statement& a : st.body) prog_.every.push_back(action(a, ec, true, INT_MAX));
    }
  }
  return prog_;
}

}  // namespace

void validate(const Script& s) { (void)Compiler(s).compile(); }

EffectLoader load_from(const std::filesystem::path& dir) {
  return [dir](const std::string& file) {
    const std::filesystem::path p = std::filesystem::path(file).is_absolute() ? std::filesystem::path(file) : dir / file;
    auto m = rollout::load_model(p);
    if (!m) throw std::runtime_error(std::format("{}: {}", p.string(), m.error()));
    rt::RolloutEffect e;
    e.m = std::move(*m);
    e.stored_bytes = e.m.storage_bytes();
    return e;
  };
}

// --- the runner --------------------------------------------------------------------------------------------------------

namespace {

struct ModR {
  std::vector<Module*> tiles;  // row by row from the bottom, each row left to right (a trail: its pool of tiles)
  float W = 0, Wd = 0, stride = 0, ax = 0, ay = 0;
  int group = -1;
  int trail = -1;          // a trail: its index in Impl::trails (ax, ay: its source)
  std::vector<int> named;  // the control index of each named setting
};

using Clock = std::chrono::steady_clock;
double ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

}  // namespace

struct Scene::Impl {
  Program prog;
  std::string source;
  std::vector<std::unique_ptr<rt::RolloutEffect>> effects;
  std::vector<std::unique_ptr<Module>> owned;
  std::vector<Module*> all;
  std::vector<ModR> mods;
  std::unique_ptr<FieldBus> bus;
  std::unique_ptr<Light> light;
  std::unique_ptr<Particles> parts;
  std::unique_ptr<Water> water;  // water statements (water.hpp); none without them
  std::unique_ptr<Frame> frame;
  std::unique_ptr<Pool> pool;
  std::unique_ptr<PictureThread> picture;  // Options::overlap
  std::unique_ptr<StepScratch> step_scratch;  // Options::shared_scratch
  std::vector<std::unique_ptr<Trail>> trails;  // trail statements, in order
  std::vector<wx::Director> directors;    // weather statements, in order (weather.hpp)
  std::vector<char> emit_on;                   // per emitter: it emitted in the last frame (weather fills its region when it starts)
  std::vector<Deposit> deposits;               // what each trail's source leaves this frame
  std::vector<Shock> shocks;
  std::vector<std::array<float, 4>> scorch;
  std::vector<const Code*> scorch_glow;
  std::size_t next_shock = 0, next_scorch = 0;  // the oldest, once the lists are full
  std::vector<Lamp> lamps;                      // this frame's lamp fields
  std::vector<float> slots;
  std::vector<int> fired;
  std::vector<char> pending;  // per rule: triggered by the game, to fire at the start of the next frame
  std::vector<std::pair<std::string, float>> first;
  std::vector<Module*> active;
  std::vector<float> scratch;
  std::vector<ForceField> fstate;
  std::vector<float> fweight;  // per field, this frame (0: off)
  std::array<double, kStages> stage{};
  float light_gain = 0.14f, flow = 0.9f, bloom = 1.f, bloom_threshold = 1.f;
  std::array<float, 3> flash{};
  Env env;
  int drawn = -1;     // the last frame drawn
  int computed = -1;  // the last frame whose state (steps 1 to 9) is computed
  int shaded = -1;    // the frame the modules' images show

  [[noreturn]] void fail(Pos p, const std::string& m) const { throw Error(p, m, source); }
  float eval(const Code& c) { return run(c, env); }
  float eval_or(const Code& c, float d) { return c.present() ? run(c, env) : d; }
  Module* tile(int mi, int c, int r) { return mods[zs(mi)].tiles[zs(r * prog.modules[zs(mi)].cols + c)]; }

  void sync(int mi) {
    const ModR& r = mods[zs(mi)];
    const ModC& mc = prog.modules[zs(mi)];
    if (r.trail >= 0) {  // a trail: where its source is, and whether a tile plays
      const Trail& T = *trails[zs(r.trail)];
      slots[zs(mc.slot + m_x)] = r.ax;
      slots[zs(mc.slot + m_y)] = r.ay;
      slots[zs(mc.slot + m_active)] = T.active() > 0 ? 1.f : 0.f;
      slots[zs(mc.slot + m_vx)] = T.vx();
      slots[zs(mc.slot + m_vy)] = T.vy();
      return;
    }
    const Module* m0 = r.tiles[0];
    slots[zs(mc.slot + m_x)] = mc.tiled ? (r.ax - 0.5f * r.Wd) + 0.5f * r.Wd : m0->at.x + 0.5f * r.W;
    slots[zs(mc.slot + m_y)] = r.ay;
    slots[zs(mc.slot + m_active)] = m0->active ? 1.f : 0.f;
  }

  // A single module standing at (X, Y): its tile's bottom edge `sink` of its width below Y, centred on X.
  void place(int mi, float X, float Y) {
    ModR& r = mods[zs(mi)];
    const ModC& mc = prog.modules[zs(mi)];
    r.ax = X;
    r.ay = Y;
    if (r.trail >= 0) {  // a trail's source: its tiles stay where they are
      sync(mi);
      return;
    }
    if (mc.tiled) {
      const float left = X - 0.5f * r.Wd, bottom = Y + mc.sink * r.W - r.W;
      for (int row = 0; row < mc.rows; ++row) {
        for (int c = 0; c < mc.cols; ++c) {
          Module* m = tile(mi, c, row);
          m->at.x = left + r.stride * static_cast<float>(c);
          m->at.y = bottom - r.stride * static_cast<float>(row);
        }
      }
    } else {
      Module* m = r.tiles[0];
      m->at.x = X - 0.5f * r.W;
      m->at.y = Y + mc.sink * r.W - r.W;
    }
    sync(mi);
  }

  // Start a module from start point `from` (tiles: the one in (in_c, in_r), by default the bottom row's middle, takes
  // it; the others start empty at its age), or with nothing in it at all (empty).
  void start(int mi, int from, std::uint64_t seed, bool has_in, int in_c, int in_r, float t, Pos pos, bool empty = false) {
    const ModC& mc = prog.modules[zs(mi)];
    const rt::RolloutEffect& e = *effects[zs(mc.effect)];
    if (from >= static_cast<int>(e.m.starts.size())) fail(pos, std::format("effect '{}' has {} start points (0 to {})", prog.effects[zs(mc.effect)].name, e.m.starts.size(), e.m.starts.size() - 1));
    const rollout::StartPoint& sp = e.m.starts[zs(from)];
    const int ic = has_in ? in_c : mc.cols / 2, ir = has_in ? in_r : 0;
    for (int row = 0; row < mc.rows; ++row) {
      for (int c = 0; c < mc.cols; ++c) {
        Module* m = tile(mi, c, row);
        if (!mc.declared_controls && sp.controls.size() == m->controls.size()) std::ranges::copy(sp.controls, m->controls.begin());
        if (!empty && (!mc.tiled || (c == ic && row == ir))) m->start(from, seed);
        else m->start_empty(sp.time, mc.tiled ? seed + static_cast<std::uint64_t>(10 * row + c) : seed);
      }
    }
    slots[zs(mc.slot + m_started)] = t;
    sync(mi);
  }

  void set_active(int mi, bool on) {
    if (const int ti = mods[zs(mi)].trail; ti >= 0) {  // a trail is stopped by clearing it (and started by its source)
      if (!on) trails[zs(ti)]->clear();
      sync(mi);
      return;
    }
    for (Module* m : mods[zs(mi)].tiles) m->active = on;
    sync(mi);
  }

  // The bursts of the hand-written scene, to the bit: its spawn() arguments were evaluated right to left (GCC), so the
  // random numbers are drawn here in that order.
  void burst(float x, float y, float r, int embers, int debris, float speed, int sparks = 0, float dir_x = 0.f, float dir_y = -1.f, float cone = 360.f) {
    Particles& P = *parts;
    for (int i = 0; i < embers; ++i) {
      const float a = P.uniform() * 6.2831853f, d = std::sqrt(P.uniform()) * r;
      const float s = speed * (0.3f + 0.7f * P.uniform());
      const float life = 2.f + 3.f * P.uniform();
      const float size = 0.7f + 1.1f * P.uniform();
      const float temp = 0.8f + 0.4f * P.uniform();
      P.spawn(Kind::ember, x + d * std::cos(a), y + d * std::sin(a), s * std::cos(a), s * std::sin(a) - 0.45f * speed, temp, size, life);
    }
    for (int i = 0; i < debris; ++i) {
      const float a = 3.1415927f + P.uniform() * 3.1415927f, s = speed * (0.4f + 0.9f * P.uniform());
      const float size = 2.f + 3.f * P.uniform();
      const float temp = 0.5f + 0.4f * P.uniform();
      P.spawn(Kind::debris, x, y, s * std::cos(a), s * std::sin(a), temp, size, 6.f);
    }
    // sparks (after the hand-written scene's draws, which they leave as they were): thrown within `cone` degrees of
    // the direction, fast and short-lived
    const float aim = std::atan2(dir_y, dir_x), spread = cone * 3.1415927f / 180.f;
    for (int i = 0; i < sparks; ++i) {
      const float a = aim + (P.uniform() - 0.5f) * spread;
      const float s = speed * (0.4f + 0.8f * P.uniform());
      const float life = 0.3f + 0.8f * P.uniform();
      P.spawn(Kind::spark, x, y, s * std::cos(a), s * std::sin(a), 1.f, 1.2f, life);
    }
  }

  void act(ActC& a, float t) {
    const std::string& k = a.kind;
    if (a.p[p_if].present() && eval(a.p[p_if]) == 0.f) return;
    if (k == "transfer") {
      a.buf.clear();
      for (const int ti : a.targets)
        for (Module* m : mods[zs(ti)].tiles)
          if (m->active) a.buf.push_back(m);
      if (a.buf.empty()) return;
      const float fraction = eval_or(a.p[p_fraction], 1.f);
      const int top = a.p[p_top].present() ? static_cast<int>(eval(a.p[p_top])) : 0;
      const float hg = eval_or(a.p[p_heat], 1.f), sg = eval_or(a.p[p_soot], 1.f);
      for (Module* m : mods[zs(a.subjects[0])].tiles) transfer(*m, a.buf, fraction, top > 0 ? m->res() - top : 0, hg, sg);
    } else if (k == "push") {
      const float gain = eval_or(a.p[p_gain], 1.f);
      for (Module* m : mods[zs(a.subjects[0])].tiles) push(*m, *bus, gain);
    } else if (k == "suppress") {
      const int mi = a.subjects[0];
      const auto one = [&](Module* m) {
        if (m->active) suppress(*m, a.cells[0], a.cells[1], a.cells[2], a.cells[3]);
      };
      if (a.tiles.empty()) {
        for (Module* m : mods[zs(mi)].tiles) one(m);
      } else {
        for (const auto& [c, r] : a.tiles) one(tile(mi, c, r));
      }
    } else if (k == "hand_over") {
      const int from = a.subjects[0], to = a.targets[0];
      const ModC& mc = prog.modules[zs(to)];
      const bool seeded = a.p[p_seed].present();
      const std::uint64_t seed = seeded ? static_cast<std::uint64_t>(eval(a.p[p_seed])) : 0;
      for (int row = 0; row < mc.rows; ++row) {
        for (int c = 0; c < mc.cols; ++c) {
          Module* f = tile(from, c, row);
          Module* m = tile(to, c, row);
          if (seeded) m->seed = seed + static_cast<std::uint64_t>(10 * row + c);
          m->take_over(*f);
          f->active = false;
        }
      }
      slots[zs(mc.slot + m_started)] = t;
      sync(from);
      sync(to);
    } else if (k == "start") {
      const int mi = a.subjects[0];
      if (a.p[p_x].present()) place(mi, eval(a.p[p_x]), eval(a.p[p_y]));
      const std::uint64_t seed = a.p[p_seed].present() ? static_cast<std::uint64_t>(eval(a.p[p_seed])) : prog.modules[zs(mi)].seed;
      start(mi, a.from, seed, a.has_in, a.in_c, a.in_r, t, a.pos, a.empty);
    } else if (k == "wake") {
      for (const int mi : a.subjects) {
        if (mods[zs(mi)].tiles[0]->active) continue;
        if (a.p[p_x].present()) place(mi, eval(a.p[p_x]), eval(a.p[p_y]));
        slots[zs(prog.modules[zs(mi)].slot + m_started)] = t;
        set_active(mi, true);
        break;
      }
    } else if (k == "stop") {
      for (const int mi : a.subjects) set_active(mi, false);
    } else if (k == "shock") {
      Shock s;
      s.x = eval(a.p[p_x]);
      s.y = eval(a.p[p_y]);
      s.t0 = t;
      s.speed = eval_or(a.p[p_speed], s.speed);
      s.decay = eval_or(a.p[p_decay], s.decay);
      s.amp = eval_or(a.p[p_amp], s.amp);
      s.width = eval_or(a.p[p_width], s.width);
      if (shocks.size() < shocks.capacity()) {
        shocks.push_back(s);
      } else if (!shocks.empty()) {  // full (a rule that repeats): the oldest makes room
        shocks[next_shock % shocks.size()] = s;
        ++next_shock;
      }
    } else if (k == "scorch") {
      const float x = eval(a.p[p_x]), y = eval(a.p[p_y]);
      const float r = eval_or(a.p[p_radius], 100.f), g = eval_or(a.p[p_glow], 1.f);
      if (scorch.size() < scorch.capacity()) {
        scorch.push_back({x, y, r, g});
        scorch_glow.push_back(a.p[p_glow].present() ? &a.p[p_glow] : nullptr);
      } else if (!scorch.empty()) {  // full (a rule that repeats): the oldest mark makes room
        const std::size_t i = next_scorch++ % scorch.size();
        scorch[i] = {x, y, r, g};
        scorch_glow[i] = a.p[p_glow].present() ? &a.p[p_glow] : nullptr;
      }
    } else if (k == "burst") {
      const float x = eval(a.p[p_x]), y = eval(a.p[p_y]);
      const float r = eval_or(a.p[p_radius], 50.f), speed = eval_or(a.p[p_speed], 600.f);
      const int embers = std::max(0, static_cast<int>(eval_or(a.p[p_embers], 200.f))), debris = std::max(0, static_cast<int>(eval_or(a.p[p_debris], 0.f)));
      const int sparks = std::max(0, static_cast<int>(eval_or(a.p[p_sparks], 0.f)));
      const float dx = eval_or(a.p[p_dx], 0.f), dy = eval_or(a.p[p_dy], -1.f), cone = eval_or(a.p[p_cone], 360.f);
      burst(x, y, r, embers, debris, speed, sparks, dx, dy, cone);
      const int spatter = std::max(0, static_cast<int>(eval_or(a.p[p_spatter], 0.f))), shards = std::max(0, static_cast<int>(eval_or(a.p[p_shards], 0.f)));
      if (spatter > 0 || shards > 0) {  // spatter and shards (fx_emit.cpp), after the sparks: the draws before them stay as they were
        constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
        fx::Burst b;
        b.x = x;
        b.y = y;
        b.radius = r;
        b.speed = speed;
        b.dx = dx;
        b.dy = dy;
        b.cone = cone;
        b.spatter = spatter;
        b.shards = shards;
        b.colour = {eval_or(a.p[p_cr], kNaN), eval_or(a.p[p_cg], kNaN), eval_or(a.p[p_cb], kNaN)};
        b.grain = eval_or(a.p[p_grain], kNaN);
        b.stain = std::max(0.f, eval_or(a.p[p_stain], 0.f));
        b.floor = eval_or(a.p[p_floor], kNaN);
        fx::burst(b, *parts, prog.ground);
      }
    } else if (k == "splash") {  // water: a crown of drops where something hits (water.hpp)
      if (!water) return;
      const float x = eval(a.p[p_x]), y = eval(a.p[p_y]);
      const float power = eval_or(a.p[p_power], 1.f), amount = eval_or(a.p[p_mist_amount], 1.f), heat = eval_or(a.p[p_mist_heat], 0.2f);
      const auto seed = static_cast<std::uint64_t>(std::max(0.f, eval_or(a.p[p_seed], 1.f)));
      Module* mist = a.mist >= 0 ? mods[zs(a.mist)].tiles[0] : nullptr;
      water->splash(x, y, power, a.water, mist, amount, heat, seed, t, static_cast<int>(std::lround(t * prog.fps)));
    } else if (k == "wave") {  // water: waves on a pool (every frame along a path: a wake)
      if (!water) return;
      water->stir(a.water, eval(a.p[p_x]), eval_or(a.p[p_amount], 200.f), eval_or(a.p[p_width], 20.f), static_cast<int>(std::lround(t * prog.fps)));
    }
  }

  void fire(RuleC& r, std::size_t i, float t) {
    for (ActC& a : r.actions) act(a, t);
    ++fired[i];
    slots[zs(r.slot)] = t;
    if (first[i].second == kNever) first[i].second = t;
  }

  bool window(const Params& p, float t) {
    if (p[p_from].present() && t < eval(p[p_from])) return false;
    if (p[p_until].present() && !(t < eval(p[p_until]))) return false;
    return !p[p_if].present() || eval(p[p_if]) != 0.f;
  }

  // Emitters, each a transcription of the hand-written scene's (random numbers drawn in the order it drew them).
  void emit(const EmitC& e, std::size_t ei, float t) {
    Particles& P = *parts;
    if (e.kind == "sparks") {  // a fuse: a point running from one end to the other over [from, until)
      const float t0 = eval_or(e.p[p_from], 0.f), t1 = eval_or(e.p[p_until], prog.length);
      if (!(t >= t0 && t < t1)) return;
      if (e.p[p_if].present() && eval(e.p[p_if]) == 0.f) return;
      const float s = (t - t0) / (t1 - t0);
      const float ax = eval(e.p[p_x]), ay = eval(e.p[p_y]), bx = eval(e.p[p_x1]), by = eval(e.p[p_y1]);
      const float x = ax + (bx - ax) * s, y = ay + (by - ay) * s;
      const int n = static_cast<int>(eval_or(e.p[p_count], 22.f));
      for (int i = 0; i < n; ++i) {
        const float life = 0.2f + 0.35f * P.uniform();
        const float vy = -(60.f + 240.f * P.uniform());
        const float vx = (P.uniform() - 0.5f) * 160.f;
        P.spawn(Kind::spark, x, y, vx, vy, 1.f, 1.2f, life);
      }
      return;
    }
    if (!window(e.p, t)) {
      emit_on[ei] = 0;
      return;
    }
    if (e.is_fx) return fx_emit(e, ei, t);
    if (e.kind == "arc" || e.kind == "ring" || e.kind == "glow") return glows(e, t);
    if (e.kind == "rain" || e.kind == "snow" || e.kind == "ash" || e.kind == "leaves") return weather(e, ei, t);
    const int tries = static_cast<int>(eval_or(e.p[p_tries], e.kind == "embers" ? 2.f : 40.f));
    if (e.kind == "embers") {  // embers shed by burning modules, from their base
      const float chance = eval_or(e.p[p_chance], 0.6f), spread = eval_or(e.p[p_spread], 40.f);
      for (const int mi : e.mods) {
        const float y = mods[zs(mi)].ay - 20.f;
        for (Module* m : mods[zs(mi)].tiles) {
          if (!m->active) continue;
          const float cx = m->at.x + 0.5f * static_cast<float>(m->size()) * m->at.scale;
          for (int i = 0; i < tries; ++i) {
            if (P.uniform() < chance) {
              const float life = 1.5f + 1.5f * P.uniform();
              const float size = 0.6f + 0.5f * P.uniform();
              const float vy = -(60.f + 90.f * P.uniform());
              const float vx = (P.uniform() - 0.5f) * 30.f;
              const float x = cx + (P.uniform() - 0.5f) * spread;
              P.spawn(Kind::ember, x, y, vx, vy, 0.7f, size, life);
            }
          }
        }
      }
      return;
    }
    // flakes of soot falling out of the bus's thick smoke
    const float x0 = eval(e.p[p_x]), y0 = eval(e.p[p_y]), w = eval(e.p[p_w]), h = eval(e.p[p_h]);
    const float chance = eval_or(e.p[p_chance], 0.25f), soot = eval_or(e.p[p_soot], 0.35f);
    for (int i = 0; i < tries; ++i) {
      const float x = x0 + P.uniform() * w;
      const float y = y0 + P.uniform() * h;
      if (bus->at(x, y).soot > soot && P.uniform() < chance) P.spawn(Kind::flake, x, y, 0.f, 15.f, 0.f, 1.f + P.uniform(), 5.f);
    }
  }

  // The procedural elements of fx_emit.cpp (docs/COMPOSE.md §4.10): their values this frame (NaN: not given), then they
  // make their particles. Like the weather, the ones that fill a region do it when they start (or start again).
  void fx_emit(const EmitC& e, std::size_t ei, float t) {
    fx::Emit f;
    f.kind = e.fx_kind;
    for (std::size_t k = 0; k < f.v.size(); ++k) f.v[k] = e.fxv[k].present() ? eval(e.fxv[k]) : std::numeric_limits<float>::quiet_NaN();
    const bool fresh = emit_on[ei] == 0;
    emit_on[ei] = 1;
    fx::emit(f, *parts, t, prog.fps, fresh, static_cast<std::uint32_t>(ei));
  }

  // Procedural glows (not learned; Particles::spawn_glow): a lightning bolt or an electric arc, a ring, discs of light.
  // Each is made again every frame and lives for that frame. Their shapes come from a hash of the emitter's seed and
  // the strike, not from the particles' generator: a bolt keeps its shape until the next strike, and the rest of the
  // scene draws the same random numbers with or without them.
  void glows(const EmitC& e, float t) {
    Particles& P = *parts;
    const float life = 1.5f / prog.fps;
    const bool arc = e.kind == "arc", ring = e.kind == "ring";
    const std::array<float, 3> dc = arc ? std::array<float, 3>{0.55f, 0.7f, 1.f} : ring ? std::array<float, 3>{1.f, 0.75f, 0.45f} : std::array<float, 3>{1.f, 0.7f, 0.4f};
    const std::array<float, 3> col{eval_or(e.p[p_cr], dc[0]), eval_or(e.p[p_cg], dc[1]), eval_or(e.p[p_cb], dc[2])};
    const float bright = eval_or(e.p[p_bright], arc ? 2.5f : 1.5f);
    if (!(bright > 0.f)) return;
    if (arc) {
      const float ax = eval(e.p[p_x]), ay = eval(e.p[p_y]), bx = eval(e.p[p_x1]), by = eval(e.p[p_y1]);
      const float width = std::max(0.3f, eval_or(e.p[p_width], 1.5f)), jitter = eval_or(e.p[p_jitter], 0.2f);
      const int branches = std::clamp(static_cast<int>(eval_or(e.p[p_branches], 2.f)), 0, 8);
      const float rate = eval_or(e.p[p_rate], 0.f);
      const auto seed = static_cast<std::uint64_t>(std::max(0.f, eval_or(e.p[p_seed], 1.f)));
      const auto now = static_cast<std::uint32_t>(std::lround(t * prog.fps));  // the frame
      const std::uint32_t strike = rate > 0.f ? static_cast<std::uint32_t>(std::floor(t * rate)) : now;
      std::uint32_t h = hash_cell(static_cast<std::int32_t>(strike), 17, 0, seed);
      const auto rnd = [&h] {  // [0, 1)
        h = hash32(h + 0x9e3779b9U);
        return static_cast<float>(h >> 8) * (1.f / 16777216.f);
      };
      const float flicker = 0.7f + 0.3f * (0.5f + 0.5f * cell_value(static_cast<std::int32_t>(now), 3, 1, seed));
      bolt(ax, ay, bx, by, width, jitter, col, bright * flicker, life, rnd, 6);
      const float dx = bx - ax, dy = by - ay, len = std::sqrt(dx * dx + dy * dy);
      for (int b = 0; b < branches && len > 8.f; ++b) {  // forks: from a point of the main bolt, shorter and fainter
        const float u = 0.2f + 0.6f * rnd(), turn = (rnd() < 0.5f ? -1.f : 1.f) * (0.35f + 0.5f * rnd()), l = (0.2f + 0.3f * rnd()) * len;
        const float sx = ax + u * dx + (rnd() - 0.5f) * jitter * len, sy = ay + u * dy + (rnd() - 0.5f) * jitter * len;
        const float c = std::cos(turn), sn = std::sin(turn), ux = dx / len, uy = dy / len;
        bolt(sx, sy, sx + l * (c * ux - sn * uy), sy + l * (sn * ux + c * uy), 0.7f * width, jitter, col, 0.55f * bright * flicker, life, rnd, 5);
      }
      return;
    }
    const float x = eval(e.p[p_x]), y = eval(e.p[p_y]), radius = eval(e.p[p_radius]);
    if (!(radius > 0.f)) return;
    if (ring) {
      const float squash = eval_or(e.p[p_squash], 1.f), width = std::max(0.3f, eval_or(e.p[p_width], 3.f));
      const int n = std::clamp(static_cast<int>(6.2831853f * radius / 8.f), 12, 256);
      for (int pass = 0; pass < 2; ++pass) {  // the ring, then a wide faint halo around it
        const float w = pass == 0 ? width : 3.f * width, b = pass == 0 ? bright : 0.3f * bright;
        for (int i = 0; i < n; ++i) {
          const float a0 = 6.2831853f * static_cast<float>(i) / static_cast<float>(n), a1 = 6.2831853f * static_cast<float>(i + 1) / static_cast<float>(n);
          P.spawn_glow(x + radius * std::cos(a0), y + squash * radius * std::sin(a0), x + radius * std::cos(a1), y + squash * radius * std::sin(a1), w, col, b, life,
                       false, false);
        }
      }
      return;
    }
    const int count = std::clamp(static_cast<int>(eval_or(e.p[p_count], 1.f)), 1, 512);
    const float spread = eval_or(e.p[p_spread], 0.f), rate = eval_or(e.p[p_rate], 0.5f);
    const auto seed = static_cast<std::uint64_t>(std::max(0.f, eval_or(e.p[p_seed], 1.f)));
    if (count == 1 && spread == 0.f) {
      P.spawn_glow(x, y, x, y, radius, col, bright, life, true, true);
      return;
    }
    for (int i = 0; i < count; ++i) {  // motes wandering within `spread` of the point, each flickering
      const float k = static_cast<float>(i);
      const float ox = spread * value_noise(7.31f * k, rate * t, 0.5f, seed), oy = spread * value_noise(3.17f * k + 50.f, rate * t, 1.5f, seed);
      const float f = 0.55f + 0.45f * value_noise(2.17f * k, 3.f * rate * t, 9.5f, seed + 1);
      P.spawn_glow(x + ox, y + oy, x + ox, y + oy, radius, col, bright * f, life, true, true);
    }
  }

  // Weather (procedural, not learned): rain, snow, ash, leaves or petals falling through the region `in`, `size`, with
  // `density` of them in each 100 x 100 world pixels at once, drifting with the `wind` (world pixels a second) and the
  // bus's flow (`flow` of it), fluttering; the region is filled at once when the emitter starts, then fed from its top
  // edge (and from upwind of it). Their random numbers come from a hash of the emitter, its seed and the frame, not from
  // the particles' generator, so the rest of the scene draws the same numbers with or without them. Rain washes the
  // modules it falls on: `wash` of their soot a second (and half that of their heat).
  void weather(const EmitC& e, std::size_t ei, float t) {
    Particles& P = *parts;
    struct Def {
      Kind kind;
      float fall, size, density, alpha, flow, freq;
      std::array<float, 3> colour;
    };
    const Def d = e.kind == "rain"   ? Def{Kind::rain, 850.f, 0.8f, 3.f, 0.35f, 0.15f, 0.f, {0.55f, 0.6f, 0.7f}}
                  : e.kind == "snow" ? Def{Kind::snow, 55.f, 1.6f, 1.5f, 0.9f, 1.f, 2.f, {0.85f, 0.87f, 0.92f}}
                  : e.kind == "ash"  ? Def{Kind::ash, 32.f, 1.1f, 1.f, 0.85f, 1.f, 1.5f, {0.3f, 0.28f, 0.27f}}
                                     : Def{Kind::leaf, 45.f, 2.6f, 0.15f, 0.95f, 0.8f, 2.5f, {0.8f, 0.4f, 0.1f}};
    const float x0 = eval(e.p[p_x]), y0 = eval(e.p[p_y]), W = std::max(1.f, eval(e.p[p_w])), H = std::max(1.f, eval(e.p[p_h]));
    const float density = std::max(0.f, eval_or(e.p[p_density], d.density));
    const float wu = eval_or(e.p[p_wu], 0.f), wv = eval_or(e.p[p_wv], 0.f);
    const float size = std::max(0.25f, eval_or(e.p[p_radius], d.size)), bright = std::max(0.f, eval_or(e.p[p_bright], 1.f));
    const bool coloured = e.p[p_cr].present();
    const std::array<float, 3> col{eval_or(e.p[p_cr], d.colour[0]) * bright, eval_or(e.p[p_cg], d.colour[1]) * bright, eval_or(e.p[p_cb], d.colour[2]) * bright};
    const float flowk = eval_or(e.p[p_flowk], d.flow);
    const float melt = d.kind == Kind::snow ? std::max(0.f, eval_or(e.p[p_melt], 0.15f)) : 0.f;
    const float glowing = d.kind == Kind::ash ? std::clamp(eval_or(e.p[p_glowing], 0.08f), 0.f, 1.f) : 0.f;
    const auto seed = static_cast<std::uint64_t>(std::max(0.f, eval_or(e.p[p_seed], 1.f)));
    const float vy = std::max(5.f, d.fall + wv);                       // its fall, world pixels a second
    const float shift = wu * H / vy;                                   // how far the wind takes it across the region
    const float xa = x0 - std::max(0.f, shift), span = W + std::abs(shift);
    const auto now = static_cast<std::int32_t>(std::lround(t * prog.fps));
    std::uint32_t h = hash_cell(now, 29 + static_cast<std::int32_t>(ei), 3, seed);
    const auto rnd = [&h] {  // [0, 1)
      h = hash32(h + 0x9e3779b9U);
      return static_cast<float>(h >> 8) * (1.f / 16777216.f);
    };
    const bool fresh = emit_on[ei] == 0;
    emit_on[ei] = 1;
    const float live = density * span * H / 10000.f;  // at once in the region
    const float count = fresh ? live : live * vy / H / prog.fps;
    const bool physical = e.p[p_rate].present() && (d.kind == Kind::rain || d.kind == Kind::snow);
    const int n = physical ? 0 : std::min(8192, static_cast<int>(count) + (rnd() < count - std::floor(count) ? 1 : 0));
    if (physical) physical_weather(e, d.kind, fresh, x0, y0, W, H, wu, wv, col, d.alpha, flowk, melt, rnd);
    static constexpr std::array<std::array<float, 3>, 4> autumn{{{0.78f, 0.33f, 0.08f}, {0.86f, 0.56f, 0.1f}, {0.55f, 0.17f, 0.06f}, {0.62f, 0.47f, 0.13f}}};
    for (int i = 0; i < n; ++i) {
      const float x = xa + rnd() * span;
      const float y = fresh ? y0 + rnd() * H : y0 - rnd() * vy / prog.fps;
      const float phase = 6.2831853f * rnd(), freq = d.freq * (0.7f + 0.6f * rnd()), s = size * (0.7f + 0.6f * rnd());
      const float fall = vy * (0.85f + 0.3f * rnd());
      std::array<float, 3> c = col;
      if (d.kind == Kind::leaf && !coloured) c = autumn[std::min<std::size_t>(3, static_cast<std::size_t>(4.f * rnd()))];
      const float tone = 0.85f + 0.3f * rnd();
      for (float& v : c) v *= tone;
      const float aux = d.kind == Kind::snow ? melt : d.kind == Kind::ash && rnd() < glowing ? 0.5f + 0.5f * rnd() : 0.f;
      const float life = (y0 + H - y) / fall * 1.25f + 0.5f;
      P.spawn_weather(d.kind, x, y, wu, fall, s, life, phase, freq, c, d.alpha, flowk, aux);
    }
    const float washk = d.kind == Kind::rain ? std::max(0.f, eval_or(e.p[p_wash], 0.f)) : 0.f;
    if (washk > 0.f) {  // the rain washes the smoke out where it falls
      const float ks = std::max(0.f, 1.f - washk / prog.fps), kh = std::max(0.f, 1.f - 0.5f * washk / prog.fps);
      for (Module* m : all) {
        if (!m->active) continue;
        const float span_m = static_cast<float>(m->size()) * m->at.scale;
        if (m->at.x > xa + span || m->at.x + span_m < xa || m->at.y > y0 + H || m->at.y + span_m < y0) continue;
        compose::wash(*m, xa, y0, xa + span, y0 + H, ks, kh);
      }
    }
  }

  // Rain or snow by the weather director's physics (weather.hpp, docs/WEATHER.md §4.1), at `rate` mm/h: drops sized by
  // Marshall and Palmer (0.5 to 6 mm) and falling at Atlas, Srivastava and Sekhon's speeds, or flakes by Locatelli and
  // Hobbs; as many as the distribution holds in a slab `depth` metres deep (0.5) seen at `scale` world pixels a metre (600)
  // unless `density` says; each drop as wide as it is there (times `radius`, 1) and its streak the length it falls in the
  // `shutter` (half a frame: a 180-degree shutter). The region is filled from the distribution itself; what feeds it from
  // the top is weighted by fall speed (the faster drops cross it sooner), so the drops in it keep the distribution.
  template <class R>
  void physical_weather(const EmitC& e, Kind kind, bool fresh, float x0, float y0, float W, float H, float wu, float wv, std::array<float, 3> col, float alpha,
                        float flowk, float melt, R& rnd) {
    Particles& P = *parts;
    const bool rain = kind == Kind::rain;
    const float rate = eval(e.p[p_rate]);
    if (!(rate > 0.f)) return;
    const float scale = std::max(1.f, eval_or(e.p[p_scale], 600.f)), depth = std::max(0.f, eval_or(e.p[p_depth], 0.5f));
    const float shutter = std::max(1e-4f, eval_or(e.p[p_shutter], 0.5f / prog.fps)), mult = std::max(0.f, eval_or(e.p[p_radius], 1.f));
    const float k100 = 100.f / scale;
    const float number = rain ? wx::mp_number(rate) : wx::snow_number(rate);  // a cubic metre
    const float density = e.p[p_density].present() ? std::max(0.f, eval(e.p[p_density])) : number * depth * k100 * k100;
    const float vmean = std::max(5.f, (rain ? wx::mp_mean_speed(rate) : wx::flake_mean_speed()) * scale + wv);
    const float vmax = (rain ? wx::drop_speed(wx::kDropMax) : wx::flake_speed(wx::kFlakeMax)) * scale;
    const float shift = wu * H / vmean, xa = x0 - std::max(0.f, shift), span = W + std::abs(shift);
    const float live = density * span * H / 10000.f;
    const float count = fresh ? live : live * vmean / H / prog.fps;
    const int n = std::min(8192, static_cast<int>(count) + (rnd() < count - std::floor(count) ? 1 : 0));
    const auto size_of = [&](float u) { return rain ? wx::mp_sample(rate, u) : wx::flake_sample(u); };
    const auto speed_of = [&](float D) { return (rain ? wx::drop_speed(D) : wx::flake_speed(D)) * scale; };
    for (int i = 0; i < n; ++i) {
      float D = size_of(rnd());
      if (!fresh)  // fed from the top: a size kept with the chance of its speed over the fastest's
        for (int k = 0; k < 16 && rnd() * vmax > speed_of(D); ++k) D = size_of(rnd());
      const float fall = std::max(5.f, speed_of(D) + wv);
      const float x = xa + rnd() * span;
      const float y = fresh ? y0 + rnd() * H : y0 - rnd() * fall / prog.fps;
      const float phase = 6.2831853f * rnd(), freq = rain ? 0.f : 2.f * (0.7f + 0.6f * rnd());
      const float across = D * scale / 1000.f;  // its diameter there, world pixels
      std::array<float, 3> c = col;
      const float tone = 0.85f + 0.3f * rnd();
      for (float& v : c) v *= tone;
      const float life = (y0 + H - y) / fall * 1.25f + 0.5f;
      P.spawn_weather(kind, x, y, wu, fall, mult * (rain ? across : 0.5f * across), life, phase, freq, c, alpha, flowk, rain ? shutter * prog.fps : melt);
    }
  }

  // A bolt from (ax, ay) to (bx, by): its middle points moved across it by up to `jitter` of its length, then the halves'
  // middles by half as much, and so on, down to pieces of about 10 world pixels (at most 2^levels of them); drawn as a
  // core, and a faint halo two and a half times wider along every other point (smoother, so its pieces meet cleanly).
  template <class R>
  void bolt(float ax, float ay, float bx, float by, float width, float jitter, std::array<float, 3> col, float bright, float life, R& rnd, int levels) {
    std::array<std::array<float, 2>, 65> pt;
    const float dx = bx - ax, dy = by - ay, len = std::sqrt(dx * dx + dy * dy);
    if (!(len > 0.5f)) return;
    const int L = std::clamp(static_cast<int>(std::ceil(std::log2(len / 10.f))), 1, std::min(levels, 6));
    const float px = -dy / len, py = dx / len;
    pt[0] = {ax, ay};
    pt[1] = {bx, by};
    int n = 2;
    float amp = jitter * len;
    for (int l = 0; l < L; ++l) {
      for (int i = n - 1; i >= 0; --i) pt[zs(2 * i)] = pt[zs(i)];
      for (int i = 0; i + 1 < n; ++i) {
        const auto& p0 = pt[zs(2 * i)];
        const auto& p1 = pt[zs(2 * i + 2)];
        const float o = (rnd() - 0.5f) * 2.f * amp;
        pt[zs(2 * i + 1)] = {0.5f * (p0[0] + p1[0]) + o * px, 0.5f * (p0[1] + p1[1]) + o * py};
      }
      n = 2 * n - 1;
      amp *= 0.5f;
    }
    Particles& P = *parts;
    for (int i = 0; i + 1 < n; ++i) P.spawn_glow(pt[zs(i)][0], pt[zs(i)][1], pt[zs(i + 1)][0], pt[zs(i + 1)][1], width, col, bright, life, i == 0, i + 2 == n);
    const int step = n > 2 ? 2 : 1;  // (n - 1 is even when it is more than 1)
    for (int i = 0; i + step < n; i += step) {
      P.spawn_glow(pt[zs(i)][0], pt[zs(i)][1], pt[zs(i + step)][0], pt[zs(i + step)][1], 2.5f * width, col, 0.22f * bright, life, i == 0, i + step + 1 == n);
    }
  }

  void update_field(std::size_t fi, float t) {
    const FieldC& fc = prog.fields[fi];
    ForceField& F = fstate[fi];
    fweight[fi] = 0.f;
    if (!window(fc.p, t)) return;
    fweight[fi] = eval_or(fc.p[p_weight], 1.f);
    F.time = t;
    F.x = eval_or(fc.p[p_x], F.x);
    F.y = eval_or(fc.p[p_y], F.y);
    if (fc.p[p_level].present()) F.y = eval(fc.p[p_level]);
    F.soft = eval_or(fc.p[p_soft], F.soft);
    F.damping = eval_or(fc.p[p_damping], F.damping);
    F.radius = eval_or(fc.p[p_radius], F.radius);
    F.strength = eval_or(fc.p[p_strength], F.strength);
    F.u = eval_or(fc.p[p_u], F.u);
    F.v = eval_or(fc.p[p_v], F.v);
    F.amount = eval_or(fc.p[p_amount], F.amount);
    F.rate = eval_or(fc.p[p_rate], F.rate);
    if (fc.p[p_scale].present()) F.soft = eval(fc.p[p_scale]);
    if (fc.p[p_core].present()) F.soft = eval(fc.p[p_core]);
    if (fc.p[p_swirl].present()) F.damping = eval(fc.p[p_swirl]);
    if (fc.p[p_steam].present()) F.damping = eval(fc.p[p_steam]);
    if (fc.p[p_seed].present()) F.seed = static_cast<std::uint64_t>(eval(fc.p[p_seed]));
    if (fc.p[p_cr].present()) F.colour = {eval(fc.p[p_cr]), eval(fc.p[p_cg]), eval(fc.p[p_cb])};
  }

  // A flow field's velocity at a world point (world pixels per frame, y down), for particles.
  static std::array<float, 2> flow_at(const ForceField& f, float x, float y, float w) {
    switch (f.kind) {
      case ForceField::Kind::wind: return {w * f.u, w * f.v};
      case ForceField::Kind::vortex: {
        const float ex = x - f.x, ey = y - f.y, r2 = ex * ex + ey * ey, rr = f.radius * f.radius;
        const float s = w * f.strength * std::sqrt(r2) / f.radius * std::exp(0.5f * (1.f - r2 / rr)) / (std::sqrt(r2) + 1e-3f);
        return {-s * ey, s * ex};
      }
      case ForceField::Kind::attract: {
        auto uv = field_flow(f, x, y, w);
        const float ex = f.x - x, ey = f.y - y, r = std::sqrt(ex * ex + ey * ey) + 1e-3f, R = std::max(f.radius, 1e-3f);
        const float pull = w * f.strength * (r / R) * std::exp(0.5f * (1.f - r * r / (R * R))) / r;
        return {uv[0] + pull * ex, uv[1] + pull * ey};
      }
      default: return field_flow(f, x, y, w);
    }
  }

  void live(float t) {
    for (std::size_t mi = 0; mi < mods.size(); ++mi) {
      const ModC& mc = prog.modules[mi];
      ModR& r = mods[mi];
      if (mc.look_b >= 0 && mc.look_w.varying) {
        const float w = eval(mc.look_w);
        const ShaderSpec& a = prog.looks[zs(mc.look_a)];
        ShaderSpec mix = prog.looks[zs(mc.look_b)];
        const auto lerp = [w](float x, float y) { return x + w * (y - x); };
        mix.heat_scale = lerp(a.heat_scale, mix.heat_scale);
        mix.emission = lerp(a.emission, mix.emission);
        mix.emission_power = lerp(a.emission_power, mix.emission_power);
        mix.soot_density = lerp(a.soot_density, mix.soot_density);
        mix.soot_albedo = lerp(a.soot_albedo, mix.soot_albedo);
        mix.sky = lerp(a.sky, mix.sky);
        mix.shadow = lerp(a.shadow, mix.shadow);
        mix.scene_light = lerp(a.scene_light, mix.scene_light);
        mix.relief = lerp(a.relief, mix.relief);
        mix.hue = lerp(a.hue, mix.hue);
        for (int c = 0; c < 3; ++c) mix.tint[zs(c)] = lerp(a.tint[zs(c)], mix.tint[zs(c)]);
        for (Module* m : r.tiles) m->spec = mix;
      }
      for (std::size_t k = 0; k < mc.named.size(); ++k) {
        if (!mc.named[k].code.varying) continue;
        const float v = eval(mc.named[k].code);
        for (Module* m : r.tiles) m->controls[zs(r.named[k])] = v;
      }
      if (mc.opacity.varying) {
        const float v = eval(mc.opacity);
        for (Module* m : r.tiles) m->opacity = v;
      }
      bool resized = false;  // a single module whose width changes over time (a spawned effect's scale)
      if (!mc.tiled && mc.width_code.varying) {
        const float w = std::max(1.f, eval(mc.width_code));
        if (w != r.W) {
          r.W = r.Wd = w;
          for (Module* m : r.tiles) m->at.scale = w / static_cast<float>(mc.size);
          resized = true;
        }
      }
      if (mc.has_at && !mc.tiled && (mc.ax.varying || mc.ay.varying)) place(static_cast<int>(mi), eval(mc.ax), eval(mc.ay));
      else if (resized) place(static_cast<int>(mi), r.ax, r.ay);
    }
    for (std::size_t i = 0; i < scorch.size(); ++i)
      if (scorch_glow[i]) scorch[i][3] = eval(*scorch_glow[i]);
    frame->exposure = eval_or(prog.exposure, 1.f);
    frame->fade = eval_or(prog.fade, 1.f);
    frame->haze = eval_or(prog.haze, 1.f);
    frame->time = t;
    frame->daylight = eval_or(prog.daylight, 0.f);
    frame->overcast = eval_or(prog.overcast, 0.f);
    frame->visibility = std::max(1.f, eval_or(prog.visibility, kNever));  // metres (at least one)
    bloom = eval_or(prog.bloom, 1.f);
    bloom_threshold = eval_or(prog.bloom_threshold, 1.f);
    light_gain = eval_or(prog.light_gain, 0.14f);
    for (int c = 0; c < 3; ++c) flash[zs(c)] = eval_or(prog.flash[zs(c)], 0.f);
    flow = eval_or(prog.flow, 0.9f);
    frame->cam_x = eval_or(prog.cam_x, 0.f);
    frame->cam_y = eval_or(prog.cam_y, 0.f);
  }

  // A weather director at frame f (weather.hpp, docs/WEATHER.md): where it heads (its schedule's line whose time has
  // come, or its `regime`), a request when that changes, a step of a frame (frame 0: it starts, in its regime at t = 0),
  // then its members into their slots. Its random numbers are its own (a hash of its seed and the step).
  void weather_step(std::size_t wi, int f, float t) {
    const WeatherC& wc = prog.weathers[wi];
    wx::Director& d = directors[wi];
    int to = wc.start;
    float over = std::numeric_limits<float>::quiet_NaN();
    if (!wc.schedule.empty()) {
      for (const WeatherC::Line& l : wc.schedule) {
        if (!(t >= l.at)) break;
        to = l.regime;
        over = l.over;
      }
    } else if (wc.regime.varying) {
      const float v = eval(wc.regime);
      to = std::isfinite(v) ? static_cast<int>(std::lround(std::clamp(v, 0.f, static_cast<float>(wx::kRegimes - 1)))) : static_cast<int>(d.regime());
    }
    if (f == 0) d = wx::Director(wc.seed, static_cast<wx::Regime>(wc.schedule.empty() ? to : wc.start), wc.direction);
    if (static_cast<int>(d.regime()) != to) d.request(static_cast<wx::Regime>(to), std::isnan(over) ? eval_or(wc.over, 60.f) : over);
    if (f > 0) d.step(1.f / prog.fps);
    for (int k = 0; k < wx::kMembers; ++k) slots[zs(wc.slot + k)] = d.value(k);
  }

  // The water's settings for this frame, then its step (water.hpp).
  void water_step(int f, float t) {
    if (!water) return;
    for (std::size_t i = 0; i < prog.waters.size(); ++i) {
      const WaterC& w = prog.waters[i];
      WaterSpec& s = water->spec(static_cast<int>(i));
      const auto& P = w.p;
      const auto ev = [&](WP k, float d) { return P[zs(k)].present() ? eval(P[zs(k)]) : d; };
      s.on = !(P[w_from].present() && t < eval(P[w_from])) && !(P[w_until].present() && !(t < eval(P[w_until]))) &&
             !(P[w_if].present() && eval(P[w_if]) == 0.f);
      s.x0 = ev(w_x0, 0.f);
      s.y0 = ev(w_y0, 0.f);
      s.x1 = ev(w_x1, w.kind == WaterKind::jet ? s.x0 : 100.f);
      s.y1 = ev(w_y1, w.kind == WaterKind::jet ? s.y0 : 100.f);
      s.flow = ev(w_flow, 0.6f);
      s.vx = ev(w_vx, 0.f);
      s.vy = ev(w_vy, 40.f);
      s.speed = ev(w_speed, 0.f);
      s.angle = ev(w_angle, 0.f);
      s.spread = ev(w_spread, 3.f);
      s.fan = ev(w_fan, 0.f);
      s.pulse = ev(w_pulse, 0.f);
      s.breakup = ev(w_breakup, 1.f);
      s.gravity = ev(w_gravity, 700.f);
      s.wu = ev(w_wu, 0.f);
      s.wv = ev(w_wv, 0.f);
      s.rate = ev(w_rate, w.kind == WaterKind::pool ? 0.f : w.kind == WaterKind::drips ? 1.f : 4.f);
      s.radius = ev(w_radius, w.kind == WaterKind::bubbles ? 3.f : 2.f);
      s.depth = ev(w_depth, 60.f);
      s.damping = ev(w_damping, 1.2f);
      s.drift = ev(w_drift, 0.f);
      s.surface = ev(w_surface, 0.f);
      s.swell = ev(w_swell, 0.f);
      s.wavelength = ev(w_wavelength, 120.f);
      s.period = ev(w_period, 3.f);
      s.density = ev(w_density, w.kind == WaterKind::spray ? 200.f : 20.f);
      s.threshold = ev(w_threshold, 0.15f);
      s.churn = ev(w_churn, 0.f);
      s.sway = ev(w_sway, 0.f);
      s.gust = std::clamp(ev(w_gust, 0.6f), 0.f, 1.f);
      s.floor = ev(w_floor, prog.ground);
      s.mist_amount = ev(w_mist_amount, 1.f);
      s.mist_heat = ev(w_mist_heat, 0.2f);
      s.brightness = ev(w_brightness, 1.f);
      s.clarity = ev(w_clarity, w.kind == WaterKind::shaft ? 0.4f : 0.5f);
      const std::array<float, 3> dc = w.kind == WaterKind::shaft ? std::array<float, 3>{0.55f, 0.6f, 0.68f} : std::array<float, 3>{0.62f, 0.8f, 0.95f};
      s.colour = {ev(w_cr, dc[0]), ev(w_cg, dc[1]), ev(w_cb, dc[2])};
    }
    water->step(t, 1.f / prog.fps, f, bus.get());
  }

  void build(const EffectLoader& load, const Options& o);
  void render(int f, std::span<std::uint8_t> rgb);
  void advance(int f) {  // the states of the frames up to f, without pictures
    while (computed < f) state(++computed);
  }
  void shade_for(int f) {  // the modules' images of frame f (its state is the last computed)
    if (shaded == f) return;
    shade();
    shaded = f;
  }
  int script_module(std::string_view name) const {
    for (std::size_t i = 0; i < prog.modules.size(); ++i)
      if (prog.modules[i].name == name) return static_cast<int>(i);
    return -1;
  }
  void state(int f);  // steps 1 to 9 of frame f
  void shade();       // the active modules into their images
  void picture_ms();  // the picture's stages into stage[] (background and modules are one pass: draw)
};

void Scene::Impl::build(const EffectLoader& load, const Options& o) {
  // step()'s working memory, one set per thread that may step at once (the picture thread is one of o.threads)
  if (o.shared_scratch) step_scratch = std::make_unique<StepScratch>(std::max(1, o.threads));
  slots.assign(zs(prog.slots), 0.f);
  slots[s_length] = prog.length;
  slots[s_fps] = prog.fps;
  slots[s_ground] = prog.ground;
  for (const RuleC& r : prog.rules) slots[zs(r.slot)] = kNever;
  for (const InputC& in : prog.inputs) slots[zs(in.slot)] = in.value;
  for (const auto& [name, v] : o.inputs) {  // the game's starting values
    const auto it = std::ranges::find(prog.inputs, name, &InputC::name);
    if (it == prog.inputs.end()) {
      std::vector<std::string> names;
      for (const InputC& in : prog.inputs) names.push_back(in.name);
      throw std::invalid_argument(std::format("the script has no input '{}'{}", name, did_you_mean(name, names)));
    }
    slots[zs(it->slot)] = v;
  }
  env.slots = slots.data();
  env.shocks = &shocks;
  // effects, with the script's settings of their detail layers
  for (const EffectC& ec : prog.effects) {
    rt::RolloutEffect e;
    try {
      e = load(ec.file);
    } catch (const std::exception& ex) {
      fail(ec.pos, std::format("cannot load effect '{}': {}", ec.name, ex.what()));
    }
    for (const auto& [key, v] : ec.detail) {
      rollout::DetailSpec& d = e.m.detail;
      if (key == "swirl") d.swirl = v;
      else if (key == "swirl_scale") d.swirl_scale = v;
      else if (key == "swirl_rate") d.swirl_rate = v;
      else if (key == "swirl_ramp") d.swirl_ramp = v;
      else if (key == "contrast") d.contrast = v;
      else if (key == "grow") d.grow = v;
      else if (key == "passes") d.passes = static_cast<int>(v);  // study O: the lock every 1 / passes of a frame's flow
      else if (key == "soften") d.soften = v;  // study Q: the fine fields move this share of the way to their neighbours' mean each frame
    }
    if (e.m.starts.empty()) fail(ec.pos, std::format("effect '{}' has no start points", ec.name));
    effects.push_back(std::make_unique<rt::RolloutEffect>(std::move(e)));
  }
  // modules, in the order of the script (tiles row by row from the bottom)
  int groups = 0;
  for (std::size_t mi = 0; mi < prog.modules.size(); ++mi) {
    const ModC& mc = prog.modules[mi];
    const rt::RolloutEffect& e = *effects[zs(mc.effect)];
    const int res = e.m.h.res;
    if (mc.size % res != 0) fail(mc.pos, std::format("the size of '{}' must be a multiple of its effect's grid ({} cells)", mc.name, res));
    if (mc.tiled && 2 * mc.band >= res) fail(mc.pos, std::format("the band of '{}' must be less than half its effect's grid ({} cells)", mc.name, res));
    if (mc.trail && 2 * mc.tr.band >= res) fail(mc.pos, std::format("the band of trail '{}' must be less than half its effect's grid ({} cells)", mc.name, res));
    ModR r;
    r.W = mc.width_code.varying ? std::max(1.f, eval(mc.width_code)) : mc.width;
    if (mc.tiled) {
      r.stride = r.W * static_cast<float>(res - mc.band) / static_cast<float>(res);
      r.Wd = r.W + r.stride * static_cast<float>(mc.cols - 1);
    } else {
      r.Wd = r.W;
    }
    if (mc.over >= 0) {
      const ModC& oc = prog.modules[zs(mc.over)];
      if (effects[zs(oc.effect)]->m.h.res != res) fail(mc.pos, std::format("'{}' and '{}' must have effects of the same grid to share a place", mc.name, oc.name));
      r.group = mods[zs(mc.over)].group;
    } else {
      r.group = groups++;
    }
    const float scale = r.W / static_cast<float>(mc.size);
    // The module's precision: its effect's own (`precision float|int8`) or the scene's; with its own, the ISA the runtime
    // takes for that precision, unless the scene was given another ISA than the runtime's choice for its own precision.
    const int eprec = prog.effects[zs(mc.effect)].precision;
    const rt::Precision prec = eprec < 0 ? o.precision : eprec == 1 ? rt::Precision::int8 : rt::Precision::float32;
    const Isa isa = eprec < 0 || o.isa != best_isa(o.precision) ? o.isa : best_isa(prec);
    const int rows = mc.trail ? mc.tr.count : mc.rows;  // a trail: its pool of tiles, one per "row"
    for (int row = 0; row < rows; ++row) {
      for (int c = 0; c < mc.cols; ++c) {
        const std::string name = mc.tiled ? std::format("{}_r{}c{}", mc.name, row, c) : mc.trail ? std::format("{}_t{}", mc.name, row) : mc.name;
        owned.push_back(std::make_unique<Module>(name, e, mc.size, Placement{0.f, 0.f, scale}, isa, prec));
        Module* m = owned.back().get();
        if (step_scratch) m->share_scratch(*step_scratch);
        m->group = r.group;
        if (mc.tiled) m->band = {c > 0 ? mc.band : 0, c < mc.cols - 1 ? mc.band : 0, row > 0 ? mc.band : 0, row < mc.rows - 1 ? mc.band : 0};
        m->feather = mc.feather * static_cast<float>(mc.size);
        if (mc.shards > 0.f) {  // changes half a second long, the first after half a period or more (by the seed)
          const int period = std::max(2, static_cast<int>(std::lround(mc.shards * prog.fps)));
          m->use_shards(period, static_cast<int>(std::lround(0.5f * prog.fps)),
                        static_cast<int>((mc.seed * 2654435761ULL) % static_cast<std::uint64_t>(std::max(1, period / 2))));
        }
        m->glows = mc.glows;
        m->glow = mc.glow;
        if (mc.look_a >= 0) {
          m->look = Look::shader;
          m->spec = prog.looks[zs(mc.look_a)];
        }
        r.tiles.push_back(m);
        all.push_back(m);
      }
    }
    const int nc = e.m.h.n_controls;
    if (!mc.controls.empty()) {
      if (static_cast<int>(mc.controls.size()) != nc) fail(mc.pos, std::format("effect '{}' has {} controls, '{}' gives {}", prog.effects[zs(mc.effect)].name, nc, mc.name, mc.controls.size()));
      for (Module* m : r.tiles) std::ranges::copy(mc.controls, m->controls.begin());
    }
    for (const ModC::Named& n : mc.named) {
      const auto& names = e.m.control_names;
      const auto it = std::ranges::find(names, n.name);
      if (it == names.end()) {
        std::string list;
        for (const auto& s : names) list += (list.empty() ? "" : ", ") + s;
        std::vector<std::string> keys(names.begin(), names.end());
        for (const std::string_view k : kModuleKeys) keys.emplace_back(k);
        if (mc.trail) {
          keys.assign(names.begin(), names.end());
          for (const std::string_view k : kTrailKeys) keys.emplace_back(k);
        }
        const std::string hint = did_you_mean(n.name, keys);
        fail(n.pos, std::format("'{}' is not a property of a {} nor a control of effect '{}' (its controls: {}){}", n.name, mc.trail ? "trail" : "module",
                                prog.effects[zs(mc.effect)].name, list, hint));
      }
      r.named.push_back(static_cast<int>(it - names.begin()));
    }
    mods.push_back(std::move(r));
    ModR& rr = mods.back();
    for (std::size_t k = 0; k < mc.named.size(); ++k) {  // constant settings now; the others from the first frame
      if (mc.named[k].code.varying) continue;
      const float v = eval(mc.named[k].code);
      for (Module* m : rr.tiles) m->controls[zs(rr.named[k])] = v;
    }
    if (mc.opacity.present() && !mc.opacity.varying) {
      const float v = eval(mc.opacity);
      for (Module* m : rr.tiles) m->opacity = v;
    }
    if (mc.look_b >= 0 && !mc.look_w.varying) {
      const float w = eval(mc.look_w);
      const ShaderSpec& a = prog.looks[zs(mc.look_a)];
      ShaderSpec mix = prog.looks[zs(mc.look_b)];
      const auto lerp = [w](float x, float y) { return x + w * (y - x); };
      mix.heat_scale = lerp(a.heat_scale, mix.heat_scale);
      mix.emission = lerp(a.emission, mix.emission);
      mix.emission_power = lerp(a.emission_power, mix.emission_power);
      mix.soot_density = lerp(a.soot_density, mix.soot_density);
      mix.soot_albedo = lerp(a.soot_albedo, mix.soot_albedo);
      mix.sky = lerp(a.sky, mix.sky);
      mix.shadow = lerp(a.shadow, mix.shadow);
      mix.scene_light = lerp(a.scene_light, mix.scene_light);
      mix.relief = lerp(a.relief, mix.relief);
      mix.hue = lerp(a.hue, mix.hue);
      for (int c = 0; c < 3; ++c) mix.tint[zs(c)] = lerp(a.tint[zs(c)], mix.tint[zs(c)]);
      for (Module* m : rr.tiles) m->spec = mix;
    }
    const int i = static_cast<int>(mi);
    if (mc.trail) {  // its tiles lie on a grid whose row 0 stands on the base (by default the ground), as a module stands
      TrailSpec ts;
      ts.width = rr.W;
      ts.band = mc.tr.band;
      const float base = mc.tr.has_base ? mc.tr.base : prog.ground;
      ts.y0 = base + mc.sink * rr.W - rr.W;
      if (mc.tr.rows > 0) {
        ts.row_lo = 1 - mc.tr.rows;
        ts.row_hi = 0;
      }
      ts.age = mc.tr.age;
      ts.hold = mc.tr.hold;
      ts.fade = mc.tr.fade;
      ts.jump = mc.tr.jump > 0.f ? mc.tr.jump : 2.f * rr.W;
      ts.has_quiet = mc.tr.has_quiet;
      ts.quiet = mc.tr.quiet;
      ts.feather = mc.feather * static_cast<float>(mc.size);
      rr.trail = static_cast<int>(trails.size());
      trails.push_back(std::make_unique<Trail>(rr.tiles, ts, rr.group, mc.seed));
      deposits.emplace_back();
      slots[zs(mc.slot + m_started)] = 0.f;  // it plays from the start (its tiles start where its source goes)
      place(i, eval(mc.ax), eval(mc.ay));
      continue;
    }
    if (mc.has_at) place(i, eval(mc.ax), eval(mc.ay));
    else place(i, 0.5f * rr.Wd, rr.W - mc.sink * rr.W);
    slots[zs(mc.slot + m_started)] = kNever;
    if (mc.has_start) {
      start(i, mc.start, mc.seed, false, 0, 0, 0.f, mc.pos, mc.empty);  // the warm-up runs now, not in a frame
      if (mc.waiting) {
        set_active(i, false);
        slots[zs(mc.slot + m_started)] = kNever;
      }
    }
    sync(i);
  }
  // actions that would fail in a frame are checked now: start points that exist, hand-overs between equal tiles
  const auto check = [&](const ActC& a) {
    if (a.kind == "start") {
      const ModC& mc = prog.modules[zs(a.subjects[0])];
      const std::size_t n = effects[zs(mc.effect)]->m.starts.size();
      if (zs(a.from) >= n) fail(a.pos, std::format("effect '{}' has {} start points (0 to {})", prog.effects[zs(mc.effect)].name, n, n - 1));
    } else if (a.kind == "hand_over") {
      const Module* f = mods[zs(a.subjects[0])].tiles[0];
      const Module* t = mods[zs(a.targets[0])].tiles[0];
      if (f->size() != t->size() || f->res() != t->res()) fail(a.pos, "a hand-over needs tiles of the same size and grid on both sides");
    }
  };
  for (const RuleC& r : prog.rules)
    for (const ActC& a : r.actions) check(a);
  // the bus (by default the screen with a margin, and the sky above it), light, particles, the picture
  const float W = static_cast<float>(prog.width), H = static_cast<float>(prog.height);
  const float bx = prog.has_bus ? prog.bus_x : -0.125f * W, by = prog.has_bus ? prog.bus_y : -0.5f * H;
  const float bw = prog.has_bus ? prog.bus_w : 1.25f * W, bh = prog.has_bus ? prog.bus_h : 1.5f * H, cell = prog.bus_cell;
  bus = std::make_unique<FieldBus>(bx, by, static_cast<int>(std::ceil(bw / cell)), static_cast<int>(std::ceil(bh / cell)), cell, std::max(1, groups));
  light = std::make_unique<Light>(*bus);
  parts = std::make_unique<Particles>(prog.capacity);
  frame = std::make_unique<Frame>(prog.width, prog.height);
  pool = std::make_unique<Pool>(o.overlap && o.threads > 1 ? o.threads - 1 : o.threads);  // overlapped: the picture thread is one of them
  if (o.overlap && o.threads > 1) picture = std::make_unique<PictureThread>(*frame, *pool);
  frame->reserve(static_cast<int>(all.size()), std::max(1, prog.shock_capacity), std::max(1, prog.scorch_capacity), prog.capacity);
  frame->ground_y = prog.ground;
  if (!prog.waters.empty() || prog.splash_capacity > 0) {  // water (water.hpp): its elements, and room for what they throw
    std::vector<WaterLayout> layout;
    int drops = 256, rings = 64;
    for (const WaterC& w : prog.waters) {
      WaterLayout L;
      L.kind = w.kind;
      L.strands = w.strands;
      L.nodes = w.nodes;
      L.cells = w.cells;
      L.count = w.count;
      L.onto = w.onto;
      L.mist = w.mist >= 0 ? mods[zs(w.mist)].tiles[0] : nullptr;
      L.seed = w.seed;
      switch (w.kind) {
        case WaterKind::sheet:
        case WaterKind::jet: drops += 24 * w.strands; break;
        case WaterKind::pool:
          drops += 128;
          rings += 192;
          break;
        case WaterKind::drips: drops += 4 * w.count + 16; break;
        case WaterKind::bubbles: drops += w.count; break;
        case WaterKind::shaft: drops += w.count / 4; break;
        case WaterKind::spray: drops += 4 * w.count; break;
      }
      layout.push_back(L);
    }
    const int crowns = std::clamp(prog.splash_capacity, 1, 32);
    drops += 640 * std::min(crowns, std::max(0, prog.splash_capacity));
    water = std::make_unique<Water>(layout, drops, crowns, rings);
    frame->reserve_water(*water);
  }
  if (o.layers) {  // the engine layers: the frame's, and every module's beside its image
    frame->use_layers(true);
    for (Module* m : all) m->use_layers(true);
  }
  env.bus = bus.get();
  env.parts = parts.get();
  // every list a frame fills, at its largest
  active.reserve(all.size());
  shocks.reserve(zs(std::max(1, prog.shock_capacity)));
  scorch.reserve(zs(std::max(1, prog.scorch_capacity)));
  scorch_glow.reserve(scorch.capacity());
  const auto reserve = [&](ActC& a) {
    std::size_t n = 0;
    for (const int ti : a.targets) n += mods[zs(ti)].tiles.size();
    a.buf.reserve(n);
  };
  for (RuleC& r : prog.rules)
    for (ActC& a : r.actions) reserve(a);
  for (ActC& a : prog.every) reserve(a);
  std::size_t pull = 0;
  for (const FieldC& f : prog.fields) {
    ForceField F;
    F.kind = f.kind;
    if (f.kind == ForceField::Kind::attract || f.kind == ForceField::Kind::cold) F.damping = 0.f;  // swirl and steam: none unless given
    if (f.kind == ForceField::Kind::lamp) F.radius = 40.f;
    fstate.push_back(F);
    if (f.kind == ForceField::Kind::attract)
      for (const int mi : f.mods)
        for (const Module* m : mods[zs(mi)].tiles) pull = std::max(pull, pull_scratch(*m));
  }
  fweight.assign(prog.fields.size(), 0.f);
  lamps.reserve(prog.fields.size());
  scratch.assign(pull, 0.f);
  fired.assign(prog.rules.size(), 0);
  pending.assign(prog.rules.size(), 0);
  emit_on.assign(prog.emitters.size(), 0);
  directors.assign(prog.weathers.size(), wx::Director{});
  for (const RuleC& r : prog.rules) first.push_back({r.name, kNever});
}

void Scene::Impl::render(int f, std::span<std::uint8_t> rgb) {
  if (f <= drawn || f < computed) throw std::invalid_argument("Scene::render: frames must come in order, after the last one drawn or computed");
  if (rgb.size() < zs(prog.width) * zs(prog.height) * 3) throw std::invalid_argument("Scene::render: the buffer is too small");
  drawn = f;
  advance(f);  // frames skipped since the last picture (and frame f itself, unless the last call computed it)
  shade_for(f);
  if (!picture) {
    frame->capture(*light, scorch, active, *parts, shocks, *bus, water.get());
    frame->render(rgb, bloom_threshold, bloom, *pool);
    picture_ms();
    return;
  }
  // Overlapped: frame f's picture is drawn on the picture thread while the next frame's state is computed (and shaded)
  // here; the next call finds it done.
  frame->capture(*light, scorch, active, *parts, shocks, *bus, water.get());
  picture->start(rgb, bloom_threshold, bloom);
  state(++computed);
  picture->wait_images();
  shade_for(computed);
  picture->wait();
  picture_ms();
}

void Scene::Impl::picture_ms() {
  const auto& R = frame->render_ms();
  stage[kBackground] = 0.0;
  stage[kDraw] = R[Frame::kCompose];
  stage[kPartDraw] = R[Frame::kParticles];
  stage[kDistort] = R[Frame::kDistort];
  stage[kBloom] = R[Frame::kBloom];
  stage[kFinish] = R[Frame::kFinish];
}

void Scene::Impl::shade() {
  const auto c0 = Clock::now();
  pool->run(static_cast<int>(active.size()), [&](int i) { active[zs(i)]->shade(light.get()); });
  stage[kShade] = ms(c0, Clock::now());
}

void Scene::Impl::state(int f) {
  const float t = static_cast<float>(f) / prog.fps;
  slots[s_t] = t;
  const auto c0 = Clock::now();
  // Shock fronts that have died away (40 times their decay: their rings move pixels by less than a millionth) are
  // dropped, so a long game's repeating rules keep the list short; no scene of the examples lives that long.
  if (std::ranges::any_of(shocks, [t](const Shock& s) { return t - s.t0 > 40.f * s.decay; })) {
    std::erase_if(shocks, [t](const Shock& s) { return t - s.t0 > 40.f * s.decay; });
    next_shock = 0;
  }
  // 0. the weather: where each director heads now, its step, and its members for everything after (weather.hpp)
  for (std::size_t wi = 0; wi < prog.weathers.size(); ++wi) weather_step(wi, f, t);
  // 1. rules on time, shocks and fields (and rules the game triggered)
  for (std::size_t i = 0; i < prog.rules.size(); ++i) {
    RuleC& r = prog.rules[i];
    const bool triggered = pending[i] != 0;
    pending[i] = 0;
    if (r.kind == "lands" || fired[i] >= r.limit) continue;
    if (triggered || eval(r.cond) != 0.f) fire(r, i, t);
  }
  // 2. emitters; 3. settings that change over time
  for (std::size_t ei = 0; ei < prog.emitters.size(); ++ei) emit(prog.emitters[ei], ei, t);
  live(t);
  // 3b. trails: their sources' places; the tiles their paths touch switched on (before the step), faded ones off
  for (std::size_t mi = 0; mi < mods.size(); ++mi) {
    const ModR& r = mods[mi];
    if (r.trail < 0) continue;
    const ModC::TrailC& T = prog.modules[mi].tr;
    Deposit& d = deposits[zs(r.trail)];
    const bool on = !T.on.present() || eval(T.on) != 0.f;
    d.heat = eval_or(T.heat, 0.f);
    d.soot = eval_or(T.soot, 0.f);
    d.radius = std::max(0.5f, eval_or(T.radius, 0.05f * r.W));
    d.kick = eval_or(T.kick, 0.f);
    d.u = eval_or(T.u, 0.f);
    d.v = eval_or(T.v, 0.f);
    trails[zs(r.trail)]->move(r.ax, r.ay, on, d.radius, t, prog.fps);
    sync(static_cast<int>(mi));
  }
  const auto c1 = Clock::now();
  // 4. step every active module
  active.clear();
  for (Module* m : all)
    if (m->active) active.push_back(m);
  pool->run(static_cast<int>(active.size()), [&](int i) { active[zs(i)]->step(); });
  const auto c2 = Clock::now();
  // 5. tiles share their bands; couplings
  for (std::size_t mi = 0; mi < mods.size(); ++mi) {
    const ModC& mc = prog.modules[mi];
    if (!mc.tiled || !mods[mi].tiles[0]->active) continue;
    const int i = static_cast<int>(mi);
    for (int row = 0; row < mc.rows; ++row)
      for (int c = 0; c + 1 < mc.cols; ++c) blend_band(*tile(i, c, row), *tile(i, c + 1, row), Side::right, mc.band);
    for (int c = 0; c < mc.cols; ++c)
      for (int row = 0; row + 1 < mc.rows; ++row) blend_band(*tile(i, c, row), *tile(i, c, row + 1), Side::top, mc.band);
  }
  for (std::size_t mi = 0; mi < mods.size(); ++mi) {  // trails: shared bands, what their sources leave, fading
    if (const int ti = mods[mi].trail; ti >= 0) {
      trails[zs(ti)]->couple(deposits[zs(ti)], t, prog.fps);
      sync(static_cast<int>(mi));
    }
  }
  for (ActC& a : prog.every)
    if (a.kind != "push") act(a, t);
  const auto c3 = Clock::now();
  // 6. the bus; 7. pushes and force fields
  bus->clear();
  for (Module* m : active) bus->publish(*m);
  for (ActC& a : prog.every)
    if (a.kind == "push") act(a, t);
  for (std::size_t fi = 0; fi < prog.fields.size(); ++fi) {
    update_field(fi, t);
    const float w = fweight[fi];
    if (w <= 0.f) continue;
    const FieldC& fc = prog.fields[fi];
    for (const int mi : fc.mods) {
      for (Module* m : mods[zs(mi)].tiles) {
        apply(*m, fstate[fi], w);
        if (fc.kind == ForceField::Kind::attract) pull(*m, fstate[fi], w, scratch);
      }
    }
  }
  lamps.clear();  // lamps: lights that are not on the bus (procedural)
  for (std::size_t fi = 0; fi < prog.fields.size(); ++fi) {
    const ForceField& F = fstate[fi];
    if (F.kind == ForceField::Kind::lamp && fweight[fi] > 0.f) lamps.push_back({F.x, F.y, F.radius, fweight[fi] * F.strength, F.colour});
  }
  const auto c4 = Clock::now();
  // 8. light
  light->update(*bus, light_gain, flash, *pool, lamps);
  const auto c5 = Clock::now();
  // 9. particles: fields that act on them, the flow of the bus, landings
  const float dt = 1.f / prog.fps;
  for (std::size_t fi = 0; fi < prog.fields.size(); ++fi) {
    if (!prog.fields[fi].particles || fweight[fi] <= 0.f) continue;
    const ForceField& F = fstate[fi];
    const float w = fweight[fi], k = prog.fps * std::min(1.f, 1.1f * dt);  // per frame to per second, with a drag of 1.1/s
    parts->each([&](float x, float y, float& vx, float& vy) {
      const auto uv = flow_at(F, x, y, w);
      vx += k * uv[0];
      vy += k * uv[1];
    });
  }
  parts->update(dt, bus.get(), flow, prog.ground);
  for (const auto& l : parts->landings()) {
    for (std::size_t i = 0; i < prog.rules.size(); ++i) {
      RuleC& r = prog.rules[i];
      if (r.kind != "lands" || r.particle != l.kind || fired[i] >= r.limit) continue;
      slots[s_land_x] = l.x;
      slots[s_land_temp] = l.temp;
      if (!r.cond.present() || eval(r.cond) != 0.f) fire(r, i, t);
    }
  }
  water_step(f, t);  // the water: its settings and a step (its time is counted with the particles')
  const auto c6 = Clock::now();
  // 10. the picture: shade(), then Frame::capture() and render() (render())
  stage[kScript] = ms(c0, c1);
  stage[kStep] = ms(c1, c2);
  stage[kCouple] = ms(c2, c3);
  stage[kBus] = ms(c3, c4);
  stage[kLight] = ms(c4, c5);
  stage[kParticles] = ms(c5, c6);
}

Scene::Scene(const Script& s, const EffectLoader& load, Options o) : impl_(std::make_unique<Impl>()) {
  impl_->prog = Compiler(s).compile();
  impl_->source = s.source;
  impl_->build(load, o);
}

Scene::~Scene() = default;

int Scene::width() const { return impl_->prog.width; }
int Scene::height() const { return impl_->prog.height; }
float Scene::fps() const { return impl_->prog.fps; }
int Scene::frames() const { return static_cast<int>(std::lround(impl_->prog.length * impl_->prog.fps)); }
float Scene::length() const { return impl_->prog.length; }
std::span<const float> Scene::keyframes() const { return impl_->prog.keyframes; }
void Scene::render(int f, std::span<std::uint8_t> rgb) { impl_->render(f, rgb); }
void Scene::advance(int f) { impl_->advance(f); }
int Scene::computed() const { return impl_->computed; }

int Scene::inputs() const { return static_cast<int>(impl_->prog.inputs.size()); }
const std::string& Scene::input_name(int i) const { return impl_->prog.inputs.at(zs(i)).name; }
int Scene::input_index(std::string_view name) const {
  const auto& in = impl_->prog.inputs;
  for (std::size_t i = 0; i < in.size(); ++i)
    if (in[i].name == name) return static_cast<int>(i);
  return -1;
}
float Scene::input(int i) const { return impl_->slots[zs(impl_->prog.inputs.at(zs(i)).slot)]; }
void Scene::set_input(int i, float v) { impl_->slots[zs(impl_->prog.inputs.at(zs(i)).slot)] = v; }

bool Scene::weather_value(std::string_view name, float& out) const {
  const std::size_t dot = name.find('.');
  if (dot == std::string_view::npos) return false;
  const int k = wx::member_of(name.substr(dot + 1));
  if (k < 0) return false;
  for (const WeatherC& w : impl_->prog.weathers) {
    if (w.name != name.substr(0, dot)) continue;
    out = impl_->slots[zs(w.slot + k)];
    return true;
  }
  return false;
}
int Scene::weathers() const { return static_cast<int>(impl_->prog.weathers.size()); }
const wx::Director& Scene::director(int i) const { return impl_->directors.at(zs(i)); }

bool Scene::trigger(std::string_view rule) {
  for (std::size_t i = 0; i < impl_->prog.rules.size(); ++i) {
    if (impl_->prog.rules[i].name != rule) continue;
    if (impl_->prog.rules[i].kind == "lands") return false;
    impl_->pending[i] = 1;
    return true;
  }
  return false;
}

bool Scene::place(std::string_view module, float x, float y) {
  const int mi = impl_->script_module(module);
  if (mi < 0 || impl_->prog.modules[zs(mi)].tiled) return false;
  impl_->place(mi, x, y);
  impl_->shaded = -1;  // a module's shading samples the light where it stands: an image shaded ahead is shaded again
  return true;
}

bool Scene::set_control(std::string_view module, std::string_view control, float v) {
  const int mi = impl_->script_module(module);
  if (mi < 0) return false;
  const auto& names = impl_->effects[zs(impl_->prog.modules[zs(mi)].effect)]->m.control_names;
  for (std::size_t c = 0; c < names.size(); ++c) {
    if (names[c] != control) continue;
    for (Module* m : impl_->mods[zs(mi)].tiles) m->controls[c] = v;
    return true;
  }
  return false;
}

int Scene::script_modules() const { return static_cast<int>(impl_->prog.modules.size()); }
Scene::ModuleInfo Scene::module_info(int i) const {
  const ModC& mc = impl_->prog.modules.at(zs(i));
  const ModR& r = impl_->mods[zs(i)];
  ModuleInfo out;
  out.name = mc.name;
  out.effect = impl_->prog.effects[zs(mc.effect)].name;
  out.tiles = static_cast<int>(r.tiles.size());
  out.x = impl_->slots[zs(mc.slot + m_x)];
  out.y = impl_->slots[zs(mc.slot + m_y)];
  out.width = r.Wd;
  out.started = impl_->slots[zs(mc.slot + m_started)];
  out.active = r.trail >= 0 ? impl_->trails[zs(r.trail)]->active() > 0 : r.tiles[0]->active;
  out.tiled = mc.tiled;
  out.controls = std::span<const float>(r.tiles[0]->controls);
  out.control_names = std::span<const std::string>(impl_->effects[zs(mc.effect)]->m.control_names);
  return out;
}
int Scene::rules() const { return static_cast<int>(impl_->prog.rules.size()); }
const std::string& Scene::rule_name(int i) const { return impl_->prog.rules.at(zs(i)).name; }
const char* Scene::stage_name(int s) {
  static constexpr const char* names[kStages] = {"script", "step", "couple", "bus", "light", "particles", "shade", "background", "draw", "particles_draw", "distort", "bloom", "finish"};
  return s >= 0 && s < kStages ? names[s] : "?";
}
const std::array<double, Scene::kStages>& Scene::stage_ms() const { return impl_->stage; }
std::span<Module* const> Scene::modules() const { return impl_->all; }
Module* Scene::module(std::string_view name, int tile) const {
  for (std::size_t i = 0; i < impl_->prog.modules.size(); ++i) {
    if (impl_->prog.modules[i].name != name) continue;
    const auto& t = impl_->mods[i].tiles;
    return tile >= 0 && zs(tile) < t.size() ? t[zs(tile)] : nullptr;
  }
  return nullptr;
}
int Scene::rule_count(std::string_view name) const {
  for (std::size_t i = 0; i < impl_->prog.rules.size(); ++i)
    if (impl_->prog.rules[i].name == name) return impl_->fired[i];
  return -1;
}
float Scene::rule_time(std::string_view name) const {
  for (const RuleC& r : impl_->prog.rules)
    if (r.name == name) return impl_->slots[zs(r.slot)];
  return kNever;
}
std::span<const std::pair<std::string, float>> Scene::rules_fired() const { return impl_->first; }
const FieldBus& Scene::bus() const { return *impl_->bus; }
Particles& Scene::particles() { return *impl_->parts; }
const Water* Scene::water() const { return impl_->water.get(); }
const Frame& Scene::frame() const { return *impl_->frame; }
int Scene::active_modules() const { return static_cast<int>(impl_->active.size()); }
std::size_t Scene::scratch_bytes() const {
  std::size_t n = impl_->step_scratch ? impl_->step_scratch->bytes() : 0;
  for (const Module* m : impl_->all) n += m->runner().scratch_bytes() + m->shard_bytes();
  return n;
}

}  // namespace nfx::compose::script
