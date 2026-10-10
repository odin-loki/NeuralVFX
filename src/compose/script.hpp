// Scene scripts for composed effects (docs/COMPOSE.md §4): a small text format an artist edits, its parser (written by
// hand, errors with line and column), a printer that writes a script back in a canonical form, and a runner that builds
// the compose objects of compose.hpp up front and then plays the scene frame by frame without allocating.
//
//   const script::Script s = script::parse(text, "fireball.nvfxs");   // throws script::Error
//   script::Scene scene(s, script::load_from(models_dir), {.threads = 2});
//   for (int f = 0; f < scene.frames(); ++f) scene.render(f, rgb);
//
// The grammar is in docs/COMPOSE.md §4 and in the comments of script.cpp.
#pragma once

#include "compose.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace nfx::compose::script {

struct Pos {
  int line = 0, col = 0;  // 1-based; 0: unknown
};

// A problem in a script: what() is "name:line:col: message".
class Error : public std::runtime_error {
 public:
  Error(Pos p, const std::string& message, std::string_view source = "script");
  Pos pos;
  std::string message;  // without the position
};

// --- the script as parsed ----------------------------------------------------------------------------------------------

struct Expr {
  enum class Kind { number, name, member, unary, binary, call };
  Kind kind = Kind::number;
  float number = 0.f;
  std::string name;         // name; member: the module; call: the function; unary and binary: the operator
  std::string member;       // member: x, y, started, age, active
  std::vector<Expr> args;   // unary: 1, binary: 2, call: its arguments
  Pos pos;
};

// A property's value. Which fields are used depends on the type.
struct Value {
  enum class Type { expr, point, tuple, dims, name, names, points, range, look, flag };
  Type type = Type::flag;
  std::vector<Expr> exprs;          // expr 1, point 2, tuple n, dims 2, points 2n, range 4, look 0 or 1 (the weight)
  std::vector<std::string> names;   // name 1, names n, look 1 or 2
  Pos pos;
};

struct Prop {
  std::string key;
  Value value;
  Pos pos;
};

// One statement: a declaration, a setting, a rule (with its actions in body) or an action.
struct Statement {
  std::string keyword;                // scene, module, when, transfer, ...
  Pos pos;
  std::string name;                   // declarations: NAME = ...; rules: `as NAME`
  Pos name_pos;
  std::string kind;                   // module: its effect; field and emit: the kind; look: "shader" or the look it is like;
                                      // effect: the file; when: "time", "shock", "lands" or "condition"
  std::vector<std::string> subjects;  // actions: the modules acted on; when ... lands: the particle kind
  std::vector<std::string> targets;   // actions: the modules after "->"
  std::vector<Expr> args;             // let: the value; at: the time; when: the condition (shock: its number, then the
                                      // point reached; lands: the `where` condition, if any); keyframes: the times
  std::string reach;                  // when shock N reaches MODULE
  std::vector<Prop> props;
  bool repeat = false;                // rules: fire every time the condition holds
  std::optional<Expr> at_most;        // rules: fire at most this many times
  std::vector<Statement> body;        // rules and `every frame`: the actions
  const Prop* prop(std::string_view key) const;
};

struct Script {
  std::string source = "script";  // a name for error messages
  std::vector<Statement> statements;
};

// Parse a script. Throws Error at the first problem (syntax, unknown statements and properties, values of the wrong
// shape). Names and the meaning of values are checked by validate() and Scene.
Script parse(std::string_view text, std::string_view source = "script");
Script parse_file(const std::filesystem::path& path);
// The script in canonical form: parse(print(s)) gives a script that prints the same.
std::string print(const Script& s);
std::string print(const Expr& e);

// Check names, references, contexts and constants without loading any effect. Throws Error.
void validate(const Script& s);
// Words of the grammar, which cannot name anything (is_keyword), and those with property names, which cannot name a
// module, a field, an effect or a look (they would be ambiguous in lists of names).
bool is_keyword(std::string_view word);
bool is_reserved(std::string_view word);
// " (did you mean 'x'?)" when exactly one candidate is the closest within two edits of `word`, else "".
std::string did_you_mean(std::string_view word, std::span<const std::string> candidates);

// --- the runner --------------------------------------------------------------------------------------------------------

// Loads an effect named in an `effect` statement (the file as written in the script).
using EffectLoader = std::function<rt::RolloutEffect(const std::string& file)>;
// Files relative to `dir` (absolute paths as they are), with load_model().
EffectLoader load_from(const std::filesystem::path& dir);

struct Options {
  int threads = 2;
  Isa isa = best_isa();
  // Draw each frame's picture on a thread of its own (one of `threads`) while the next frame's state is computed: the
  // same frames, faster. render(f) then returns with frame f + 1's state already computed (rules, particles and modules
  // are a frame ahead of the picture it returned).
  bool overlap = false;
};

// A scene built from a script: every module, buffer and list is created here, so render() allocates nothing.
class Scene {
 public:
  Scene(const Script& s, const EffectLoader& load, Options o = {});
  ~Scene();
  Scene(const Scene&) = delete;
  Scene& operator=(const Scene&) = delete;

  int width() const;
  int height() const;
  float fps() const;
  int frames() const;                     // length * fps, rounded
  std::span<const float> keyframes() const;  // times from the `keyframes` statement
  // Frame `f` into rgb (width * height * 3). Frames must come in order from 0.
  void render(int f, std::span<std::uint8_t> rgb);

  // Stages of the last frame, ms (overlapped: the picture's are frame f's, the others frame f + 1's; background and
  // modules are drawn in one pass, under kDraw).
  enum Stage { kScript, kStep, kCouple, kBus, kLight, kParticles, kShade, kBackground, kDraw, kPartDraw, kDistort, kBloom, kFinish, kStages };
  static const char* stage_name(int s);
  const std::array<double, kStages>& stage_ms() const;

  // For tests and tools.
  std::span<Module* const> modules() const;      // every module (tiles one by one), in declaration order
  Module* module(std::string_view name, int tile = 0) const;  // nullptr if unknown
  int rule_count(std::string_view name) const;   // times a named rule fired (-1: no such rule)
  float rule_time(std::string_view name) const;  // when it last fired (infinity: not yet)
  std::span<const std::pair<std::string, float>> rules_fired() const;  // every rule: its name (or line) and when it first fired
  const FieldBus& bus() const;
  Particles& particles();
  const Frame& frame() const;
  int active_modules() const;  // stepped in the last frame
  std::size_t scratch_bytes() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace nfx::compose::script
