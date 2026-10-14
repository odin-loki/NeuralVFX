// The effect library: catalogues and the stages that play them (library.hpp, docs/LIBRARY.md).
//
// A stage is one scene script. Each instance's room is the entry's statements, renamed with the room's prefix ("s3_")
// and rewritten so that they play from a spawn:
//   - inputs: _X, _Y, _scale, _variant, _at (when the game wants it to spawn), _end (when it wants it to stop), and the
//     entry's parameters; a marker rule _go records the spawn (it fires once each time t reaches _at);
//   - t becomes the time since the spawn (t - _go), a rule's time and a module's `started` likewise; `length` is the
//     entry's duration;
//   - _alive (a let) holds from the spawn until a stop or the end of a one-shot; every rule, emitter, field, coupling,
//     flash and shake of the entry is gated by it, and a rule that fires once fires once per spawn;
//   - modules with a start point are started and put to sleep when the stage is made (their warm-up runs then), woken
//     at the first spawn and started afresh at later ones; when the instance ends, they stop.
// The stage's flash and camera shake are the sums of its instances'.
#include "library.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>

namespace nfx::library {

namespace {

using sc::Error;
using sc::Expr;
using sc::Pos;
using sc::Prop;
using sc::Statement;
using sc::Value;

constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr float kShards = 8.f;  // seconds per shard of a looping entry's learned effects

// --- building expressions and statements ---------------------------------------------------------------------------

Expr num(float v, Pos p = {}) {
  Expr e;
  e.pos = p;
  if (std::isinf(v)) {
    e.kind = Expr::Kind::name;
    e.name = "infinity";
    if (v < 0.f) {
      Expr n;
      n.kind = Expr::Kind::unary;
      n.name = "-";
      n.pos = p;
      n.args.push_back(std::move(e));
      return n;
    }
    return e;
  }
  e.kind = Expr::Kind::number;
  e.number = v;
  return e;
}

Expr name(std::string n, Pos p = {}) {
  Expr e;
  e.kind = Expr::Kind::name;
  e.name = std::move(n);
  e.pos = p;
  return e;
}

Expr bin(std::string op, Expr a, Expr b, Pos p = {}) {
  Expr e;
  e.kind = Expr::Kind::binary;
  e.name = std::move(op);
  e.pos = p;
  e.args.push_back(std::move(a));
  e.args.push_back(std::move(b));
  return e;
}

Expr not_(Expr a, Pos p = {}) {
  Expr e;
  e.kind = Expr::Kind::unary;
  e.name = "not";
  e.pos = p;
  e.args.push_back(std::move(a));
  return e;
}

Expr member(std::string m, std::string field, Pos p = {}) {
  Expr e;
  e.kind = Expr::Kind::member;
  e.name = std::move(m);
  e.member = std::move(field);
  e.pos = p;
  return e;
}

Expr call(std::string f, std::vector<Expr> args, Pos p = {}) {
  Expr e;
  e.kind = Expr::Kind::call;
  e.name = std::move(f);
  e.args = std::move(args);
  e.pos = p;
  return e;
}

Value vexpr(Expr e) {
  Value v;
  v.type = Value::Type::expr;
  v.pos = e.pos;
  v.exprs.push_back(std::move(e));
  return v;
}

Value vpoint(Expr a, Expr b) {
  Value v;
  v.type = Value::Type::point;
  v.exprs.push_back(std::move(a));
  v.exprs.push_back(std::move(b));
  return v;
}

Value vtuple(std::vector<Expr> e) {
  Value v;
  v.type = Value::Type::tuple;
  v.exprs = std::move(e);
  return v;
}

Value vdims(Expr a, Expr b) {
  Value v;
  v.type = Value::Type::dims;
  v.exprs.push_back(std::move(a));
  v.exprs.push_back(std::move(b));
  return v;
}

Value vflag() {
  Value v;
  v.type = Value::Type::flag;
  return v;
}

Prop prop(std::string key, Value v) {
  Prop p;
  p.key = std::move(key);
  p.pos = v.pos;
  p.value = std::move(v);
  return p;
}

Statement statement(std::string keyword, std::string nm = {}, std::string kind = {}) {
  Statement s;
  s.keyword = std::move(keyword);
  s.name = std::move(nm);
  s.kind = std::move(kind);
  return s;
}

Prop* find_prop(Statement& s, std::string_view key) {
  for (Prop& p : s.props)
    if (p.key == key) return &p;
  return nullptr;
}

// A constant: numbers, infinity and + - * / (an entry's header and parameters).
float constant(const Expr& e, std::string_view what, std::string_view source) {
  switch (e.kind) {
    case Expr::Kind::number: return e.number;
    case Expr::Kind::name:
      if (e.name == "infinity") return kInf;
      break;
    case Expr::Kind::unary:
      if (e.name == "-") return -constant(e.args[0], what, source);
      break;
    case Expr::Kind::binary: {
      const float a = constant(e.args[0], what, source), b = constant(e.args[1], what, source);
      if (e.name == "+") return a + b;
      if (e.name == "-") return a - b;
      if (e.name == "*") return a * b;
      if (e.name == "/") return a / b;
      break;
    }
    default: break;
  }
  throw Error(e.pos, std::format("{} must be a number", what), source);
}

// The instance's own values in an entry: its anchor, scale and variant (inputs), and alive (1 while it plays).
bool builtin(std::string_view n) { return n == "X" || n == "Y" || n == "scale" || n == "variant" || n == "alive"; }
// What a stage names after an instance's prefix and '_': its own inputs and rules, then its parameters.
bool own(std::string_view n) { return builtin(n) || n == "at" || n == "end" || n == "go" || n == "alive" || n == "stop"; }

// --- the catalogue -------------------------------------------------------------------------------------------------

Entry header(const Statement& st, const std::string& file) {
  Entry e;
  e.name = st.name;
  e.file = file;
  e.pos = st.pos;
  const auto fail = [&](Pos p, const std::string& m) { throw Error(p, m, file); };
  if (e.name.empty() || e.name.front() == '_') fail(st.name_pos, "an entry's name starts with a letter");
  const Prop* cat = st.prop("category");
  if (!cat) fail(st.pos, std::format("entry '{}' needs a category (fire, smoke, explosions, steam, magic, ambient, sparks, light, ...)", e.name));
  e.category = cat->value.names[0];
  const Prop* size = st.prop("size");
  if (!size) fail(st.pos, std::format("entry '{}' needs its size, the box it covers at scale 1 (as in 'size 300 x 380', world pixels)", e.name));
  e.width = constant(size->value.exprs[0], "the size", file);
  e.height = constant(size->value.exprs[1], "the size", file);
  if (!(e.width > 0.f && e.height > 0.f && std::isfinite(e.width) && std::isfinite(e.height))) fail(size->pos, "the size must be positive");
  if (const Prop* a = st.prop("anchor")) {
    const std::string& v = a->value.names[0];
    if (v == "base") e.anchor = Anchor::base;
    else if (v == "centre") e.anchor = Anchor::centre;
    else fail(a->value.pos, std::format("an anchor is 'base' (X, Y at the bottom centre of its box) or 'centre', not '{}'", v));
  }
  const Prop* loop = st.prop("loop");
  const Prop* once = st.prop("once");
  if (!loop == !once) fail(st.pos, std::format("entry '{}' plays either until stopped ('loop') or once for a time ('once 2.5 s')", e.name));
  e.loop = loop != nullptr;
  if (once) {
    e.duration = constant(once->value.exprs[0], "a one-shot's length", file);
    if (!(e.duration > 0.f && std::isfinite(e.duration))) fail(once->pos, "a one-shot's length must be positive");
  }
  e.show = e.loop ? 2.f : 0.4f * e.duration;
  if (const Prop* show = st.prop("show")) {
    e.show = constant(show->value.exprs[0], "the moment to show", file);
    if (!(e.show >= 0.f) || (!e.loop && e.show > e.duration)) fail(show->pos, "the moment to show is a time in the effect's life");
  }
  const Prop* about = st.prop("about");
  if (!about || about->value.names[0].empty()) fail(st.pos, std::format("entry '{}' needs a line about it (about \"...\")", e.name));
  e.about = about->value.names[0];
  return e;
}

Param param(const Statement& st, const std::string& file) {
  Param p;
  p.name = st.name;
  p.pos = st.pos;
  p.value = constant(st.args[0], "a parameter's default", file);
  if (const Prop* r = st.prop("range")) {
    if (r->value.exprs.size() != 2) throw Error(r->pos, "a range has two values (lowest, highest)", file);
    p.lo = constant(r->value.exprs[0], "a range", file);
    p.hi = constant(r->value.exprs[1], "a range", file);
    if (!(p.lo <= p.hi)) throw Error(r->pos, "a range goes from its lowest value to its highest", file);
  }
  if (!(p.value >= p.lo && p.value <= p.hi)) throw Error(st.args[0].pos, std::format("the default of '{}' is outside its range ({} to {})", p.name, p.lo, p.hi), file);
  const Prop* about = st.prop("about");
  if (!about || about->value.names[0].empty()) throw Error(st.pos, std::format("parameter '{}' needs a line about it (about \"...\")", p.name), file);
  p.about = about->value.names[0];
  return p;
}

void add_unique(std::vector<std::string>& v, std::string s) {
  if (std::ranges::find(v, s) == v.end()) v.push_back(std::move(s));
}

// What an entry plays: its learned effects and its procedural elements.
void inventory(Entry& e) {
  const auto action = [&](const Statement& a) {
    if (a.keyword == "shock" || a.keyword == "scorch" || a.keyword == "burst") add_unique(e.elements, a.keyword);
  };
  for (const Statement& st : e.body) {
    if (st.keyword == "module") {
      add_unique(e.effects, st.kind);
      if (const Prop* l = st.prop("look")) {
        for (const std::string& n : l->value.names)
          if (n != "learned") add_unique(e.elements, "look " + n);
      }
    } else if (st.keyword == "emit") {
      add_unique(e.elements, st.kind);
    } else if (st.keyword == "field") {
      add_unique(e.elements, st.kind);
    } else if (st.keyword == "light") {
      add_unique(e.elements, "flash");
    } else if (st.keyword == "camera") {
      add_unique(e.elements, "shake");
    }
    for (const Statement& a : st.body) action(a);
  }
}

}  // namespace

const Entry* Catalogue::find(std::string_view n) const {
  const int i = index(n);
  return i < 0 ? nullptr : &entries[static_cast<std::size_t>(i)];
}

int Catalogue::index(std::string_view n) const {
  for (std::size_t i = 0; i < entries.size(); ++i)
    if (entries[i].name == n) return static_cast<int>(i);
  return -1;
}

std::vector<std::string> Catalogue::categories() const {
  std::vector<std::string> out;
  for (const char* c : {"fire", "smoke", "explosions", "steam", "magic", "ambient", "sparks", "light"})
    if (std::ranges::any_of(entries, [&](const Entry& e) { return e.category == c; })) out.emplace_back(c);
  for (const Entry& e : entries) add_unique(out, e.category);
  return out;
}

std::string Catalogue::effect_file(std::string_view n) const {
  for (const Statement& s : effects)
    if (s.name == n) return s.kind;
  return {};
}

Catalogue parse(std::span<const std::pair<std::string, std::string>> files) {
  Catalogue c;
  std::map<std::string, std::string> effect_where, look_where;  // name -> the file that declares it
  for (const auto& [file, text] : files) {
    const sc::Script s = sc::parse(text, file);
    int cur = -1;  // the entry being read
    for (const Statement& st : s.statements) {
      const auto fail = [&](Pos p, const std::string& m) { throw Error(p, m, file); };
      if (st.keyword == "entry") {
        if (c.find(st.name)) fail(st.name_pos, std::format("'{}' is already an entry (in {})", st.name, c.find(st.name)->file));
        c.entries.push_back(header(st, file));
        cur = static_cast<int>(c.entries.size()) - 1;
        continue;
      }
      if (st.keyword == "effect") {
        if (cur >= 0) fail(st.pos, "declare effects before the first entry of a file: they are shared by every entry");
        if (const auto it = effect_where.find(st.name); it != effect_where.end()) {
          const Statement& prev = *std::ranges::find(c.effects, st.name, &Statement::name);
          if (sc::print(sc::Script{"", {prev}}) != sc::print(sc::Script{"", {st}})) {
            fail(st.name_pos, std::format("effect '{}' is declared differently in {}", st.name, it->second));
          }
          continue;
        }
        effect_where[st.name] = file;
        c.effects.push_back(st);
        continue;
      }
      if (cur < 0) {
        if (st.keyword != "look") fail(st.pos, std::format("'{}' belongs in an entry: start one with 'entry NAME, category ..., size W x H, ...' (only effects and looks come before the first)", st.keyword));
        if (const auto it = look_where.find(st.name); it != look_where.end()) fail(st.name_pos, std::format("look '{}' is already declared in {}", st.name, it->second));
        look_where[st.name] = file;
        c.looks.push_back(st);
        continue;
      }
      Entry& e = c.entries[static_cast<std::size_t>(cur)];
      if (st.keyword == "param") {
        if (own(st.name) || st.name.front() == '_') fail(st.name_pos, std::format("'{}' is the instance's own (X, Y, scale, variant, alive, at, end, go, stop) or starts with '_': choose another name", st.name));
        for (const Param& p : e.params)
          if (p.name == st.name) fail(st.name_pos, std::format("'{}' is already a parameter of '{}'", st.name, e.name));
        e.params.push_back(param(st, file));
        continue;
      }
      e.body.push_back(st);
    }
  }
  // the shared looks in an order where each comes after the look it is like
  std::vector<Statement> ordered;
  while (ordered.size() < c.looks.size()) {
    const std::size_t before = ordered.size();
    for (const Statement& l : c.looks) {
      const bool done = std::ranges::any_of(ordered, [&](const Statement& o) { return o.name == l.name; });
      const bool ready = l.kind == "shader" || std::ranges::any_of(ordered, [&](const Statement& o) { return o.name == l.kind; });
      if (!done && ready) ordered.push_back(l);
    }
    if (ordered.size() == before) {  // a look like one that is not declared (or a loop)
      for (const Statement& l : c.looks) {
        if (std::ranges::any_of(ordered, [&](const Statement& o) { return o.name == l.name; })) continue;
        throw Error(l.pos, std::format("look '{}' is like '{}', which is not one of the catalogue's looks", l.name, l.kind), look_where[l.name]);
      }
    }
  }
  c.looks = std::move(ordered);
  // each shared effect and look checked in its own file (a stage holds them all, and says nothing of where they are)
  for (const Statement& e : c.effects) sc::validate(sc::Script{effect_where[e.name], {e}});
  for (std::size_t i = 0; i < c.looks.size(); ++i) {
    sc::Script s{look_where[c.looks[i].name], {}};
    s.statements.assign(c.looks.begin(), c.looks.begin() + static_cast<std::ptrdiff_t>(i) + 1);
    sc::validate(s);
  }
  for (Entry& e : c.entries) inventory(e);
  StageSpec spec;  // each entry, alone in a stage: names, contexts and constants, without effects
  for (std::size_t i = 0; i < c.entries.size(); ++i) {
    for (const std::string& f : c.entries[i].effects) {
      if (c.effect_file(f).empty()) {
        const Statement* m = nullptr;
        for (const Statement& st : c.entries[i].body)
          if (st.keyword == "module" && st.kind == f && !m) m = &st;
        throw Error(m ? m->pos : c.entries[i].pos, std::format("unknown effect '{}' (declare it before the entries: effect {} = \"FILE\")", f, f), c.entries[i].file);
      }
    }
    sc::validate(single(c, static_cast<int>(i), spec).script);
  }
  return c;
}

std::vector<std::pair<std::string, std::string>> read_dir(const std::filesystem::path& dir) {
  std::vector<std::filesystem::path> paths;
  for (const auto& de : std::filesystem::directory_iterator(dir))
    if (de.is_regular_file() && de.path().extension() == ".nvfxl") paths.push_back(de.path());
  std::ranges::sort(paths);
  std::vector<std::pair<std::string, std::string>> out;
  for (const auto& p : paths) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + p.string());
    std::stringstream ss;
    ss << f.rdbuf();
    out.emplace_back(p.filename().string(), ss.str());
  }
  if (out.empty()) throw std::runtime_error("no catalogue files (*.nvfxl) in " + dir.string());
  return out;
}

Catalogue load_dir(const std::filesystem::path& dir) {
  const auto files = read_dir(dir);
  return parse(files);
}

// --- stages --------------------------------------------------------------------------------------------------------

namespace {

// One instance of an entry, rewritten into a stage's statements (see the top of the file).
class Instance {
 public:
  Instance(const Catalogue& c, const Entry& e, std::string prefix, std::int64_t seed_offset) : e_(e), p_(std::move(prefix)), seed_offset_(seed_offset) {
    go_ = p_ + "_go";
    alive_ = p_ + "_alive";
    for (const Statement& st : e.body) {
      const auto local = [&](std::set<std::string>& into, const std::string& n, Pos at) {
        if (n.empty()) return;
        if (builtin(n) || n.front() == '_') fail(at, std::format("'{}' is the instance's own (X, Y, scale, variant, alive) or starts with '_': choose another name", n));
        into.insert(n);
      };
      if (st.keyword == "module") local(modules_, st.name, st.name_pos);
      else if (st.keyword == "let") local(values_, st.name, st.name_pos);
      else if (st.keyword == "field") local(fields_, st.name, st.name_pos);
      else if (st.keyword == "look") {
        local(looks_, st.name, st.name_pos);
        if (std::ranges::any_of(c.looks, [&](const Statement& g) { return g.name == st.name; })) {
          fail(st.name_pos, std::format("look '{}' is already one of the catalogue's: choose another name", st.name));
        }
      } else if (st.keyword == "at" || st.keyword == "when") {
        local(rules_, st.name, st.name_pos);
      }
    }
    for (const Param& pa : e.params) {
      for (const auto* set : {&modules_, &values_, &fields_, &rules_})
        if (set->contains(pa.name)) fail(pa.pos, std::format("'{}' is a parameter and also declared in the entry: choose another name", pa.name));
      params_.insert(pa.name);
    }
  }

  const std::string& go() const { return go_; }
  const std::string& alive() const { return alive_; }
  std::string in(std::string_view n) const { return p_ + "_" + std::string(n); }

  // The entry's statements, rewritten, after the instance's own (its inputs, its spawn and its end).
  void emit(std::vector<Statement>& out, std::vector<std::array<Expr, 3>>& flash, std::vector<std::array<Expr, 2>>& shake) {
    for (const char* n : {"X", "Y", "scale", "variant", "at", "end"}) {
      Statement s = statement("input", in(n));
      s.args.push_back(num(std::string_view(n) == "scale" ? 1.f : std::string_view(n) == "at" || std::string_view(n) == "end" ? kInf : 0.f));
      out.push_back(std::move(s));
    }
    for (const Param& pa : e_.params) {
      Statement s = statement("input", in(pa.name));
      s.pos = pa.pos;
      s.args.push_back(num(pa.value, pa.pos));
      out.push_back(std::move(s));
    }
    // alive: from the spawn until a stop asked for after it, or the end of a one-shot
    Expr ended = bin("and", bin(">=", name("t"), name(in("end"))), bin(">=", name(in("end")), name(go_)));
    if (!e_.loop) ended = bin("or", std::move(ended), bin(">=", name("t"), bin("+", name(go_), num(e_.duration))));
    Statement alive = statement("let", alive_);
    alive.args.push_back(bin("and", bin(">=", name("t"), name(go_)), not_(std::move(ended))));
    out.push_back(std::move(alive));
    // the spawn: once each time t reaches _at
    Statement spawn = statement("when", go_, "condition");
    spawn.repeat = true;
    spawn.args.push_back(bin("and", bin(">=", name("t"), name(in("at"))),
                             bin("or", bin("<", name(go_), name(in("at"))), bin("==", name(go_), name("infinity")))));
    std::vector<std::string> mods;
    for (const Statement& st : e_.body) {
      if (st.keyword != "module") continue;
      mods.push_back(p_ + st.name);
      const Prop* start = st.prop("start");
      if (!start) continue;
      const std::string m = p_ + st.name;
      Statement a = statement("start");
      a.pos = st.pos;
      a.subjects.push_back(m);
      a.props.push_back(prop("from", vexpr(start->value.exprs[0])));
      const Prop* seed = st.prop("seed");
      Expr base = seed ? seed->value.exprs[0] : num(1.f);
      a.props.push_back(prop("seed", vexpr(bin("+", std::move(base), call("floor", {bin("*", name(in("variant")), num(9973.f))})))));
      if (st.prop("empty")) a.props.push_back(prop("empty", vflag()));
      a.props.push_back(prop("if", vexpr(bin("<", member(m, "started"), name("infinity")))));
      spawn.body.push_back(std::move(a));
      if (st.prop("waiting")) {  // woken by the entry's own rules: fresh but asleep
        Statement z = statement("stop");
        z.subjects.push_back(m);
        spawn.body.push_back(std::move(z));
      } else {  // the first spawn wakes the module started when the stage was made
        Statement w = statement("wake");
        w.subjects.push_back(m);
        w.props.push_back(prop("if", vexpr(bin("==", member(m, "started"), name("infinity")))));
        spawn.body.push_back(std::move(w));
      }
    }
    out.push_back(std::move(spawn));
    if (!mods.empty()) {  // the end: its modules stop
      Expr any;
      for (std::size_t i = 0; i < mods.size(); ++i) any = i == 0 ? member(mods[i], "active") : bin("or", std::move(any), member(mods[i], "active"));
      Statement end = statement("when", in("stop"), "condition");
      end.repeat = true;
      end.args.push_back(bin("and", not_(name(alive_)), std::move(any)));
      Statement z = statement("stop");
      z.subjects = mods;
      end.body.push_back(std::move(z));
      out.push_back(std::move(end));
    }
    for (const Statement& st : e_.body) statement_of(st, out, flash, shake);
  }

 private:
  [[noreturn]] void fail(Pos p, const std::string& m) const { throw Error(p, m, e_.file); }

  std::string module(const std::string& n) const { return modules_.contains(n) ? p_ + n : n; }
  std::string look(const std::string& n) const { return looks_.contains(n) ? p_ + n : n; }

  // An expression of the entry as the stage computes it.
  Expr expr(const Expr& e) const {
    Expr o = e;
    switch (e.kind) {
      case Expr::Kind::number: return o;
      case Expr::Kind::name: {
        const std::string& n = e.name;
        if (n == "t") return bin("-", name("t", e.pos), name(go_, e.pos), e.pos);
        if (n == "length") return num(e_.loop ? kInf : e_.duration, e.pos);
        if (builtin(n) || params_.contains(n)) return name(in(n), e.pos);
        if (values_.contains(n)) return name(p_ + n, e.pos);
        if (rules_.contains(n)) return bin("-", name(p_ + n, e.pos), name(go_, e.pos), e.pos);
        if (modules_.contains(n) || fields_.contains(n)) return name(p_ + n, e.pos);  // the checker says what it is
        return o;
      }
      case Expr::Kind::member:
        o.name = module(e.name);
        if (e.member == "started") return bin("-", std::move(o), name(go_, e.pos), e.pos);
        return o;
      case Expr::Kind::unary:
      case Expr::Kind::binary:
        for (Expr& a : o.args) a = expr(a);
        return o;
      case Expr::Kind::call:
        if (e.name == "shock") fail(e.pos, "shock fronts are numbered in the whole stage: an entry cannot read them");
        for (std::size_t i = 0; i < o.args.size(); ++i) {
          if (i == 0 && e.name == "near" && e.args[0].kind == Expr::Kind::name) o.args[0].name = module(e.args[0].name);
          else o.args[i] = expr(e.args[i]);
        }
        return o;
    }
    return o;
  }

  Value value(const Value& v, std::string_view key) const {
    Value o = v;
    for (Expr& e : o.exprs) e = expr(e);
    if (v.type == Value::Type::names || (v.type == Value::Type::name && key == "over")) {
      for (std::string& n : o.names)
        if (n != "particles") n = module(n);
    } else if (v.type == Value::Type::look) {
      for (std::string& n : o.names)
        if (n != "learned") n = look(n);
    }
    return o;
  }

  void props(Statement& s) const {
    for (Prop& p : s.props) p.value = value(p.value, p.key);
  }

  Expr gate(Pos at) const { return name(alive_, at); }

  // `if`: gated by _alive (and the statement's own condition); `from` and `until`: times after the spawn.
  void gated(Statement& s, bool times, bool sparks = false) const {
    if (Prop* p = find_prop(s, "if")) p->value.exprs[0] = bin("and", gate(s.pos), p->value.exprs[0], p->pos);
    else s.props.push_back(prop("if", vexpr(gate(s.pos))));
    if (!times) return;
    Prop* from = find_prop(s, "from");
    Prop* until = find_prop(s, "until");
    if (from) from->value.exprs[0] = bin("+", name(go_), from->value.exprs[0], from->pos);
    if (until) until->value.exprs[0] = bin("+", name(go_), until->value.exprs[0], until->pos);
    if (sparks) {  // the fuse runs over the given times, by default the entry's life
      if (!from) s.props.push_back(prop("from", vexpr(name(go_))));
      if (!until) s.props.push_back(prop("until", vexpr(bin("+", name(go_), num(e_.loop ? kInf : e_.duration)))));
    }
  }

  Statement action(const Statement& a) const {
    Statement o = a;
    for (std::string& n : o.subjects) n = module(n);
    for (std::string& n : o.targets) n = module(n);
    props(o);
    return o;
  }

  void statement_of(const Statement& st, std::vector<Statement>& out, std::vector<std::array<Expr, 3>>& flash, std::vector<std::array<Expr, 2>>& shake) const {
    const std::string& w = st.keyword;
    Statement s = st;
    if (w == "module") {
      if (st.prop("tiles") || st.prop("band")) fail(st.pos, "tiles stay where they are declared, so they cannot be spawned: use single modules side by side");
      if (!st.prop("at") && !st.prop("over")) fail(st.pos, std::format("module '{}' stands relative to the anchor: give it 'at (X, Y)' (or an offset from it)", st.name));
      s.name = p_ + st.name;
      props(s);
      if (Prop* seed = find_prop(s, "seed"); seed && seed_offset_ != 0) seed->value.exprs[0] = bin("+", seed->value.exprs[0], num(static_cast<float>(seed_offset_)));
      if (st.prop("start") && !st.prop("waiting")) s.props.push_back(prop("waiting", vflag()));
      // a looping entry's learned effects play for good in shards (Module::use_shards), unless it says otherwise
      if (e_.loop && st.prop("start") && !st.prop("empty") && !st.prop("shards")) s.props.push_back(prop("shards", vexpr(num(kShards))));
      out.push_back(std::move(s));
    } else if (w == "let" || w == "look") {
      s.name = w == "let" ? p_ + st.name : look(st.name);
      if (w == "look" && s.kind != "shader") s.kind = look(s.kind);
      for (Expr& e : s.args) e = expr(e);
      props(s);
      out.push_back(std::move(s));
    } else if (w == "field") {
      s.name = p_ + st.name;
      props(s);
      gated(s, true);
      out.push_back(std::move(s));
    } else if (w == "emit") {
      props(s);
      gated(s, true, st.kind == "sparks");
      out.push_back(std::move(s));
    } else if (w == "at" || w == "when") {
      if (st.kind == "shock") fail(st.pos, "shock fronts are numbered in the whole stage: an entry cannot wait for one (use the time, or heat(x, y))");
      if (st.at_most) fail(st.at_most->pos, "a rule of an entry fires once per spawn, or every time with 'repeat'");
      Statement r = statement("when", st.name.empty() ? std::format("{}_r{}", p_, st.pos.line) : p_ + st.name, st.kind == "lands" ? "lands" : "condition");
      r.pos = st.pos;
      r.name_pos = st.name_pos;
      r.subjects = st.subjects;
      r.repeat = true;
      Expr guard = gate(st.pos);
      if (!st.repeat) guard = bin("and", std::move(guard), bin("or", bin("<", name(r.name), name(go_)), bin("==", name(r.name), name("infinity"))));
      if (w == "at") {
        r.args.push_back(bin("and", std::move(guard), bin(">=", bin("-", name("t"), name(go_)), expr(st.args[0]), st.pos), st.pos));
      } else if (st.kind == "lands") {
        r.args.push_back(st.args.empty() ? std::move(guard) : bin("and", std::move(guard), expr(st.args[0]), st.pos));
      } else {
        r.args.push_back(bin("and", std::move(guard), expr(st.args[0]), st.pos));
      }
      for (const Statement& a : st.body) r.body.push_back(action(a));
      out.push_back(std::move(r));
    } else if (w == "every") {
      Statement ev = statement("every");
      ev.pos = st.pos;
      for (const Statement& a : st.body) {
        Statement o = action(a);
        gated(o, false);
        ev.body.push_back(std::move(o));
      }
      out.push_back(std::move(ev));
    } else if (w == "light") {
      for (const Prop& p : st.props) {
        if (p.key != "flash") fail(p.pos, "an entry's light is its flash (light flash (r, g, b)); the stage sets the rest");
        if (p.value.exprs.size() != 3) fail(p.pos, "the flash has three values (red, green, blue)");
        std::array<Expr, 3> f;
        for (int c = 0; c < 3; ++c) f[static_cast<std::size_t>(c)] = call("if", {gate(p.pos), expr(p.value.exprs[static_cast<std::size_t>(c)]), num(0.f)}, p.pos);
        flash.push_back(std::move(f));
      }
    } else if (w == "camera") {
      std::array<Expr, 2> sh{num(0.f), num(0.f)};
      for (const Prop& p : st.props) sh[p.key == "x" ? 0 : 1] = call("if", {gate(p.pos), expr(p.value.exprs[0]), num(0.f)}, p.pos);
      shake.push_back(std::move(sh));
    } else if (w == "effect") {
      fail(st.pos, "declare effects before the first entry of a file: they are shared by every entry");
    } else if (w == "input") {
      fail(st.pos, "an entry's inputs are its parameters: write 'param NAME = default, range (lo, hi), about \"...\"'");
    } else {
      fail(st.pos, std::format("'{}' is the stage's (its picture, clock and bus): an entry cannot set it", w));
    }
  }

  const Entry& e_;
  std::string p_, go_, alive_;
  std::int64_t seed_offset_;
  std::set<std::string> modules_, values_, params_, fields_, looks_, rules_;
};

Expr sum(std::vector<Expr> terms, Expr base) {
  Expr out = std::move(base);
  bool first = out.kind == Expr::Kind::number && out.number == 0.f;
  for (Expr& t : terms) {
    out = first ? std::move(t) : bin("+", std::move(out), std::move(t));
    first = false;
  }
  return out;
}

Stage make(const Catalogue& c, const StageSpec& spec, const std::vector<std::pair<int, std::string>>& instances, const std::vector<std::array<int, 2>>& rooms_k) {
  Stage st;
  st.script.source = "stage";
  auto& out = st.script.statements;
  {
    Statement s = statement("scene");
    s.props.push_back(prop("size", vdims(num(static_cast<float>(spec.width)), num(static_cast<float>(spec.height)))));
    s.props.push_back(prop("fps", vexpr(num(spec.fps))));
    s.props.push_back(prop("length", vexpr(num(86400.f))));  // a stage plays on: a day
    s.props.push_back(prop("ground", vexpr(num(spec.ground))));
    out.push_back(std::move(s));
  }
  if (spec.has_bus) {
    Statement s = statement("bus");
    s.props.push_back(prop("at", vpoint(num(spec.bus_x), num(spec.bus_y))));
    s.props.push_back(prop("size", vdims(num(spec.bus_w), num(spec.bus_h))));
    s.props.push_back(prop("cell", vexpr(num(spec.bus_cell))));
    out.push_back(std::move(s));
  }
  {
    Statement s = statement("particles");
    s.props.push_back(prop("capacity", vexpr(num(static_cast<float>(spec.particles)))));
    out.push_back(std::move(s));
  }
  std::vector<std::string> used;
  for (const auto& inst : instances)
    for (const std::string& f : c.entries[static_cast<std::size_t>(inst.first)].effects) add_unique(used, f);
  for (const Statement& s : c.effects) {
    if (std::ranges::find(used, s.name) == used.end()) continue;
    out.push_back(s);
    add_unique(st.files, s.kind);
  }
  for (const Statement& s : c.looks) out.push_back(s);
  for (const char* n : {"_camera_x", "_camera_y"}) {
    Statement s = statement("input", n);
    s.args.push_back(num(0.f));
    out.push_back(std::move(s));
  }
  std::vector<Statement> body;
  std::vector<std::array<Expr, 3>> flash;
  std::vector<std::array<Expr, 2>> shake;
  for (std::size_t i = 0; i < instances.size(); ++i) {
    const auto& [e, prefix] = instances[i];
    const Entry& entry = c.entries[static_cast<std::size_t>(e)];
    Instance inst(c, entry, prefix, 7919 * static_cast<std::int64_t>(rooms_k[i][1]));
    inst.emit(body, flash, shake);
    Slot slot;
    slot.entry = e;
    slot.room = rooms_k[i][0];
    slot.k = rooms_k[i][1];
    slot.prefix = prefix;
    slot.X = inst.in("X");
    slot.Y = inst.in("Y");
    slot.scale = inst.in("scale");
    slot.variant = inst.in("variant");
    slot.at = inst.in("at");
    slot.end = inst.in("end");
    slot.go = inst.go();
    for (const Param& p : entry.params) slot.params.push_back(inst.in(p.name));
    st.slots.push_back(std::move(slot));
  }
  {
    Statement s = statement("light");
    s.props.push_back(prop("gain", vexpr(num(spec.light_gain))));
    if (!flash.empty()) {
      std::vector<Expr> rgb;
      for (int ch = 0; ch < 3; ++ch) {
        std::vector<Expr> terms;
        for (auto& f : flash) terms.push_back(std::move(f[static_cast<std::size_t>(ch)]));
        rgb.push_back(sum(std::move(terms), num(0.f)));
      }
      s.props.push_back(prop("flash", vtuple(std::move(rgb))));
    }
    out.push_back(std::move(s));
  }
  {
    Statement s = statement("frame");
    s.props.push_back(prop("exposure", vexpr(num(spec.exposure))));
    s.props.push_back(prop("haze", vexpr(num(spec.haze))));
    s.props.push_back(prop("bloom", vexpr(num(spec.bloom))));
    out.push_back(std::move(s));
  }
  {
    Statement s = statement("camera");
    for (int ax = 0; ax < 2; ++ax) {
      std::vector<Expr> terms;
      for (auto& sh : shake) terms.push_back(std::move(sh[static_cast<std::size_t>(ax)]));
      s.props.push_back(prop(ax == 0 ? "x" : "y", vexpr(sum(std::move(terms), name(ax == 0 ? "_camera_x" : "_camera_y")))));
    }
    out.push_back(std::move(s));
  }
  for (Statement& s : body) out.push_back(std::move(s));
  return st;
}

}  // namespace

std::string Stage::text() const { return sc::print(script); }

Stage build_stage(const Catalogue& c, const StageSpec& spec) {
  std::vector<std::pair<int, std::string>> instances;
  std::vector<std::array<int, 2>> rk;
  for (std::size_t r = 0; r < spec.rooms.size(); ++r) {
    const Room& room = spec.rooms[r];
    if (room.entry < 0 || static_cast<std::size_t>(room.entry) >= c.entries.size()) throw std::invalid_argument("a room of an effect the catalogue does not have");
    if (room.count < 1 || room.count > 4096) throw std::invalid_argument("a room holds 1 to 4096 instances");
    for (int k = 0; k < room.count; ++k) {
      instances.emplace_back(room.entry, std::format("s{}_", instances.size()));
      rk.push_back({static_cast<int>(r), k});
    }
  }
  return make(c, spec, instances, rk);
}

Stage single(const Catalogue& c, int entry, const StageSpec& spec) {
  if (entry < 0 || static_cast<std::size_t>(entry) >= c.entries.size()) throw std::invalid_argument("no such entry");
  Stage st = make(c, spec, {{entry, std::string()}}, {{0, 0}});
  st.script.source = c.entries[static_cast<std::size_t>(entry)].file;
  return st;
}

}  // namespace nfx::library
