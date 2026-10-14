// Scene scripts: the lexer, the parser and the printer (script.hpp, docs/COMPOSE.md §4).
//
// A script is a list of lines. `#` starts a comment. A line that ends inside parentheses, or with a comma, an operator,
// `and`, `or` or `not`, continues on the next line. A line that ends with `:` opens a block (a rule or `every frame`):
// the indented lines after it are its actions; actions may also follow the colon, separated by `;`.
//
//   statement   := setting | declaration | emitter | rule | 'every' 'frame' ':' actions
//   setting     := ('scene' | 'bus' | 'light' | 'particles' | 'frame' | 'camera') props | 'keyframes' expr {',' expr}
//   declaration := 'effect' NAME '=' STRING props | 'look' NAME '=' ('shader' | 'like' NAME) props
//                | 'let' NAME '=' expr | 'input' NAME '=' expr | 'module' NAME '=' EFFECT props | 'field' NAME '=' KIND props
//   emitter     := 'emit' ('sparks' | 'embers' | 'flakes') props
//   rule        := ('at' expr ['s'] | 'when' condition) {modifier} ':' [action {';' action}]
//   condition   := 'shock' expr 'reaches' (point | NAME) | ('ember' | 'debris') 'lands' ['where' expr] | expr
//   modifier    := 'as' NAME | 'repeat' | 'at' 'most' expr
//   action      := 'transfer' NAME '->' names props | 'push' NAME props | 'suppress' NAME props
//                | 'hand_over' NAME '->' NAME props | 'start' NAME props | 'wake' names props | 'stop' names
//                | ('shock' | 'scorch' | 'burst') props
//   props       := {[','] KEY value}          (the keys, and the shape of each value, depend on the statement)
//   value       := expr | point | '(' expr {',' expr} ')' | expr 'x' expr | NAME | names | point 'to' point
//   point       := '(' expr ',' expr ')'
//   names       := NAME {',' NAME}
//   expr        := or;  or := and {'or' and};  and := not {'and' not};  not := 'not' not | cmp
//   cmp         := sum [('<' | '<=' | '>' | '>=' | '==' | '!=') sum];  sum := prod {('+' | '-') prod}
//   prod        := unary {('*' | '/') unary};  unary := '-' unary | primary
//   primary     := NUMBER | NAME | NAME '.' NAME | NAME '(' [expr {',' expr}] ')' | '(' expr ')'
#include "script.hpp"

#include <algorithm>
#include <charconv>
#include <format>
#include <fstream>
#include <sstream>

namespace nfx::compose::script {

Error::Error(Pos p, const std::string& m, std::string_view source)
    : std::runtime_error(std::format("{}:{}:{}: {}", source, p.line, p.col, m)), pos(p), message(m) {}

const Prop* Statement::prop(std::string_view key) const {
  for (const Prop& p : props)
    if (p.key == key) return &p;
  return nullptr;
}

namespace {

// --- the vocabulary ----------------------------------------------------------------------------------------------------

enum class VT { expr, time, point, tuple, dims, name, names, points, range, look, flag };
struct Key {
  std::string_view key;
  VT type;
};

constexpr Key kScene[] = {{"size", VT::dims}, {"fps", VT::expr}, {"length", VT::time}, {"ground", VT::expr}};
constexpr Key kBus[] = {{"at", VT::point}, {"size", VT::dims}, {"cell", VT::expr}};
constexpr Key kLight[] = {{"gain", VT::expr}, {"flash", VT::tuple}};
constexpr Key kParticles[] = {{"capacity", VT::expr}, {"flow", VT::expr}};
constexpr Key kFrame[] = {{"exposure", VT::expr}, {"fade", VT::expr}, {"haze", VT::expr}, {"bloom", VT::expr}, {"bloom_threshold", VT::expr}};
constexpr Key kCamera[] = {{"x", VT::expr}, {"y", VT::expr}};
constexpr Key kEffect[] = {{"swirl", VT::expr}, {"swirl_scale", VT::expr}, {"swirl_rate", VT::expr}, {"swirl_ramp", VT::expr},
                           {"contrast", VT::expr}, {"grow", VT::expr}};
constexpr Key kLook[] = {{"heat_scale", VT::expr}, {"emission", VT::expr}, {"emission_power", VT::expr}, {"soot_density", VT::expr},
                         {"soot_albedo", VT::expr}, {"sky", VT::expr},      {"shadow", VT::expr},         {"scene_light", VT::expr},
                         {"relief", VT::expr},     {"tint", VT::tuple}};
constexpr Key kModule[] = {{"tiles", VT::dims},   {"band", VT::expr},   {"size", VT::expr},       {"width", VT::expr},
                           {"at", VT::point},     {"sink", VT::expr},   {"over", VT::name},       {"feather", VT::expr},
                           {"look", VT::look},    {"controls", VT::tuple}, {"opacity", VT::expr}, {"start", VT::expr},
                           {"seed", VT::expr},    {"waiting", VT::flag}, {"empty", VT::flag},   {"glow", VT::tuple}};
constexpr Key kField[] = {{"at", VT::point},     {"level", VT::expr},    {"radius", VT::expr},    {"strength", VT::expr},
                          {"soft", VT::expr},    {"damping", VT::expr},  {"swirl", VT::expr},     {"steam", VT::expr},
                          {"amount", VT::expr},  {"scale", VT::expr},    {"rate", VT::expr},      {"seed", VT::expr},
                          {"velocity", VT::point}, {"direction", VT::point}, {"core", VT::expr},  {"on", VT::names},
                          {"weight", VT::expr},  {"if", VT::expr},       {"from", VT::time},      {"until", VT::time}};
constexpr Key kEmit[] = {{"along", VT::range}, {"from", VT::time}, {"until", VT::time}, {"count", VT::expr}, {"tries", VT::expr},
                         {"chance", VT::expr}, {"spread", VT::expr}, {"in", VT::point}, {"size", VT::dims}, {"soot", VT::expr},
                         {"on", VT::names},    {"if", VT::expr}};
constexpr Key kTransfer[] = {{"fraction", VT::expr}, {"top", VT::expr}, {"heat", VT::expr}, {"soot", VT::expr}, {"if", VT::expr}};
constexpr Key kPush[] = {{"gain", VT::expr}, {"if", VT::expr}};
constexpr Key kSuppress[] = {{"tiles", VT::points}, {"cells", VT::range}, {"if", VT::expr}};
constexpr Key kHandOver[] = {{"seed", VT::expr}};
constexpr Key kStart[] = {{"from", VT::expr}, {"in", VT::point}, {"seed", VT::expr}, {"at", VT::point}, {"empty", VT::flag}};
constexpr Key kWake[] = {{"at", VT::point}};
constexpr Key kStop[] = {{"if", VT::expr}};
constexpr Key kShock[] = {{"at", VT::point}, {"speed", VT::expr}, {"decay", VT::expr}, {"amp", VT::expr}, {"width", VT::expr}};
constexpr Key kScorch[] = {{"at", VT::point}, {"radius", VT::expr}, {"glow", VT::expr}};
constexpr Key kBurst[] = {{"at", VT::point}, {"radius", VT::expr}, {"embers", VT::expr}, {"debris", VT::expr}, {"speed", VT::expr}};

struct Schema {
  std::string_view keyword;
  std::span<const Key> keys;
};
constexpr Schema kSchemas[] = {
    {"scene", kScene},     {"bus", kBus},         {"light", kLight},      {"particles", kParticles}, {"frame", kFrame},
    {"camera", kCamera},   {"effect", kEffect},   {"look", kLook},        {"module", kModule},       {"field", kField},
    {"emit", kEmit},       {"transfer", kTransfer}, {"push", kPush},      {"suppress", kSuppress},   {"hand_over", kHandOver},
    {"start", kStart},     {"wake", kWake},       {"stop", kStop},        {"shock", kShock},         {"scorch", kScorch},
    {"burst", kBurst},     {"let", {}},           {"keyframes", {}},      {"every", {}},             {"at", {}},
    {"when", {}},          {"input", {}}};

std::span<const Key> keys_of(std::string_view keyword) {
  for (const Schema& s : kSchemas)
    if (s.keyword == keyword) return s.keys;
  return {};
}

const Key* find_key(std::span<const Key> keys, std::string_view k) {
  for (const Key& key : keys)
    if (key.key == k) return &key;
  return nullptr;
}

constexpr std::string_view kActions[] = {"transfer", "push", "suppress", "hand_over", "start", "wake", "stop", "shock", "scorch", "burst"};
constexpr std::string_view kTop[] = {"scene", "bus", "light", "particles", "frame", "camera", "keyframes", "effect", "look",
                                     "let", "input", "module", "field", "emit", "every", "at", "when"};
constexpr std::string_view kFieldKinds[] = {"ceiling", "vortex", "wind", "gust", "ring", "attract", "heat", "cold"};
constexpr std::string_view kEmitKinds[] = {"sparks", "embers", "flakes"};
constexpr std::string_view kWords[] = {
    // expressions
    "and", "or", "not", "if", "t", "x", "y", "temp", "rand", "length", "fps", "ground", "smooth", "exp", "sqrt", "sin",
    "cos", "abs", "floor", "min", "max", "clamp", "hypot", "pow", "lerp", "noise", "heat", "soot", "shock", "near",
    // the rest of the grammar
    "to", "by", "as", "most", "repeat", "lands", "reaches", "where", "learned", "shader", "like", "s", "ember", "debris",
    "spark", "flake", "particles", "frame", "infinity"};

template <std::size_t N>
bool in(const std::string_view (&list)[N], std::string_view w) {
  return std::ranges::find(list, w) != std::end(list);
}

bool is_action(std::string_view w) { return in(kActions, w); }

}  // namespace

bool is_keyword(std::string_view w) { return in(kActions, w) || in(kTop, w) || in(kFieldKinds, w) || in(kEmitKinds, w) || in(kWords, w); }

// A list of names ends at the first word that is a key of its statement; these statements have lists of modules.
bool is_reserved(std::string_view w) {
  if (is_keyword(w)) return true;
  for (const std::string_view k : {"field", "emit", "transfer", "wake", "stop"})
    if (find_key(keys_of(k), w)) return true;
  return false;
}

std::string did_you_mean(std::string_view w, std::span<const std::string> candidates) {
  const auto distance = [](std::string_view a, std::string_view b) {  // Levenshtein, small strings
    std::vector<std::size_t> row(b.size() + 1);
    for (std::size_t j = 0; j <= b.size(); ++j) row[j] = j;
    for (std::size_t i = 1; i <= a.size(); ++i) {
      std::size_t diag = row[0];
      row[0] = i;
      for (std::size_t j = 1; j <= b.size(); ++j) {
        const std::size_t up = row[j];
        row[j] = std::min({row[j] + 1, row[j - 1] + 1, diag + (a[i - 1] == b[j - 1] ? 0 : 1)});
        diag = up;
      }
    }
    return row[b.size()];
  };
  std::size_t best = 3;
  const std::string* pick = nullptr;
  bool tie = false;
  for (const std::string& c : candidates) {
    if (c == w) continue;
    const std::size_t d = distance(w, c);
    if (d < best) {
      best = d;
      pick = &c;
      tie = false;
    } else if (d == best) {
      tie = true;
    }
  }
  return pick && !tie && best < w.size() ? std::format(" (did you mean '{}'?)", *pick) : std::string{};
}

namespace {
template <std::size_t N>
std::vector<std::string> strings(const std::string_view (&list)[N]) {
  return {std::begin(list), std::end(list)};
}
std::vector<std::string> strings(std::span<const Key> keys) {
  std::vector<std::string> out;
  for (const Key& k : keys) out.emplace_back(k.key);
  return out;
}

// Whether a name list goes on with the word after a comma: not a key of the statement, nor a word of the grammar.
bool list_goes_on(std::string_view w, std::span<const Key> keys) { return w == "particles" || (!is_keyword(w) && !find_key(keys, w)); }
}  // namespace

namespace {

// --- the lexer ---------------------------------------------------------------------------------------------------------

struct Tok {
  enum Type { ident, number, string, punct } type = punct;
  std::string text;
  float num = 0.f;
  Pos pos;
};

struct LogicalLine {
  std::vector<Tok> toks;
  bool indented = false;
};

bool continues(const std::vector<Tok>& toks, int depth) {
  if (depth > 0) return true;
  if (toks.empty()) return false;
  const Tok& t = toks.back();
  if (t.type == Tok::ident) return t.text == "and" || t.text == "or" || t.text == "not";
  if (t.type != Tok::punct) return false;
  static constexpr std::string_view open[] = {",", "+", "-", "*", "/", "<", "<=", ">", ">=", "==", "!=", "->", "=", "("};
  return in(open, t.text);
}

std::vector<LogicalLine> lex(std::string_view src, std::string_view source) {
  std::vector<LogicalLine> out;
  bool open = false;  // the last logical line continues
  int depth = 0;
  std::vector<Pos> parens;  // where the open '(' are
  int line_no = 0;
  std::size_t at = 0;
  while (at <= src.size()) {
    const std::size_t nl = src.find('\n', at);
    std::string_view line = src.substr(at, nl == std::string_view::npos ? std::string_view::npos : nl - at);
    at = nl == std::string_view::npos ? src.size() + 1 : nl + 1;
    ++line_no;
    if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
    std::vector<Tok> toks;
    std::size_t i = 0;
    while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) ++i;
    const bool indented = i > 0;
    while (i < line.size()) {
      const char c = line[i];
      const Pos pos{line_no, static_cast<int>(i) + 1};
      if (c == ' ' || c == '\t') {
        ++i;
        continue;
      }
      if (c == '#') break;
      if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
        std::size_t j = i;
        while (j < line.size() && (std::isalnum(static_cast<unsigned char>(line[j])) || line[j] == '_')) ++j;
        toks.push_back({Tok::ident, std::string(line.substr(i, j - i)), 0.f, pos});
        i = j;
        continue;
      }
      if (std::isdigit(static_cast<unsigned char>(c)) || (c == '.' && i + 1 < line.size() && std::isdigit(static_cast<unsigned char>(line[i + 1])))) {
        std::size_t j = i;
        while (j < line.size() && std::isdigit(static_cast<unsigned char>(line[j]))) ++j;
        if (j < line.size() && line[j] == '.') {
          ++j;
          while (j < line.size() && std::isdigit(static_cast<unsigned char>(line[j]))) ++j;
        }
        if (j < line.size() && (line[j] == 'e' || line[j] == 'E')) {
          std::size_t k = j + 1;
          if (k < line.size() && (line[k] == '+' || line[k] == '-')) ++k;
          if (k < line.size() && std::isdigit(static_cast<unsigned char>(line[k]))) {
            j = k;
            while (j < line.size() && std::isdigit(static_cast<unsigned char>(line[j]))) ++j;
          }
        }
        Tok t{Tok::number, std::string(line.substr(i, j - i)), 0.f, pos};
        const auto r = std::from_chars(t.text.data(), t.text.data() + t.text.size(), t.num);  // correctly rounded, as a C++ float literal
        if (r.ec != std::errc{} || r.ptr != t.text.data() + t.text.size()) throw Error(pos, std::format("'{}' is not a number", t.text), source);
        toks.push_back(std::move(t));
        i = j;
        if (i + 1 < line.size() && line[i] == 'x' && std::isdigit(static_cast<unsigned char>(line[i + 1]))) {  // 3x2: 3 x 2
          toks.push_back({Tok::ident, "x", 0.f, Pos{line_no, static_cast<int>(i) + 1}});
          ++i;
        }
        continue;
      }
      if (c == '"') {
        std::size_t j = i + 1;
        std::string s;
        while (j < line.size() && line[j] != '"') {
          if (line[j] == '\\' && j + 1 < line.size()) ++j;
          s += line[j++];
        }
        if (j >= line.size()) throw Error(pos, "this string has no closing quote", source);
        toks.push_back({Tok::string, std::move(s), 0.f, pos});
        i = j + 1;
        continue;
      }
      static constexpr std::string_view two[] = {"->", "<=", ">=", "==", "!="};
      if (i + 1 < line.size() && in(two, line.substr(i, 2))) {
        toks.push_back({Tok::punct, std::string(line.substr(i, 2)), 0.f, pos});
        i += 2;
        continue;
      }
      if (std::string_view("(),:;=+-*/<>.").find(c) != std::string_view::npos) {
        toks.push_back({Tok::punct, std::string(1, c), 0.f, pos});
        if (c == '(') {
          ++depth;
          parens.push_back(pos);
        }
        if (c == ')' && depth > 0) {
          --depth;
          parens.pop_back();
        }
        ++i;
        continue;
      }
      // a character outside ASCII: quote all of its UTF-8 bytes, not the first alone
      const auto lead = static_cast<unsigned char>(c);
      const std::size_t n = lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 : lead >= 0xC0 ? 2 : 1;
      throw Error(pos, std::format("unexpected character '{}'", line.substr(i, std::min(n, line.size() - i))), source);
    }
    if (toks.empty()) continue;
    if (open && !out.empty()) {
      auto& prev = out.back().toks;
      prev.insert(prev.end(), toks.begin(), toks.end());
    } else {
      out.push_back({std::move(toks), indented});
    }
    open = continues(out.back().toks, depth);
  }
  if (depth > 0) throw Error(parens.back(), "this '(' is not closed", source);
  return out;
}

// --- the parser --------------------------------------------------------------------------------------------------------

class Parser;
bool list_item(const Parser& p, std::span<const Key> keys);

class Parser {
 public:
  Parser(const std::vector<Tok>& toks, std::string_view source) : t_(toks), src_(source) {}

  bool done() const { return i_ >= t_.size(); }
  const Tok* peek(std::size_t k = 0) const { return i_ + k < t_.size() ? &t_[i_ + k] : nullptr; }
  bool punct(std::string_view p, std::size_t k = 0) const {
    const Tok* t = peek(k);
    return t && t->type == Tok::punct && t->text == p;
  }
  bool word(std::string_view w, std::size_t k = 0) const {
    const Tok* t = peek(k);
    return t && t->type == Tok::ident && t->text == w;
  }
  Pos pos() const { return done() ? end_pos() : t_[i_].pos; }
  Pos end_pos() const {
    if (t_.empty()) return {};
    const Tok& b = t_.back();
    return {b.pos.line, b.pos.col + static_cast<int>(b.text.size())};
  }
  const Tok& next() {
    if (done()) fail(end_pos(), "the line ends too early");
    return t_[i_++];
  }
  [[noreturn]] void fail(Pos p, const std::string& m) const { throw Error(p, m, src_); }
  std::string found() const {
    if (done()) return "the end of the line";
    const Tok& t = t_[i_];
    return t.type == Tok::string ? "a string" : std::format("'{}'", t.text);
  }
  void expect(std::string_view p, std::string_view what) {
    if (!punct(p)) fail(pos(), std::format("expected '{}' {}, found {}", p, what, found()));
    ++i_;
  }
  void expect_word(std::string_view w, std::string_view what) {
    if (!word(w)) fail(pos(), std::format("expected '{}' {}, found {}", w, what, found()));
    ++i_;
  }
  std::string name(std::string_view what) {
    const Tok* t = peek();
    if (!t || t->type != Tok::ident) fail(pos(), std::format("expected {}, found {}", what, found()));
    ++i_;
    return t->text;
  }

  // expressions
  Expr expr() { return or_(); }

 private:
  Expr binary(std::string op, Pos p, Expr a, Expr b) {
    Expr e;
    e.kind = Expr::Kind::binary;
    e.name = std::move(op);
    e.pos = p;
    e.args.push_back(std::move(a));
    e.args.push_back(std::move(b));
    return e;
  }
  Expr or_() {
    Expr a = and_();
    while (word("or")) {
      const Pos p = next().pos;
      a = binary("or", p, std::move(a), and_());
    }
    return a;
  }
  Expr and_() {
    Expr a = not_();
    while (word("and")) {
      const Pos p = next().pos;
      a = binary("and", p, std::move(a), not_());
    }
    return a;
  }
  Expr not_() {
    if (word("not")) {
      Expr e;
      e.kind = Expr::Kind::unary;
      e.name = "not";
      e.pos = next().pos;
      e.args.push_back(not_());
      return e;
    }
    return cmp();
  }
  Expr cmp() {
    Expr a = sum();
    for (const char* op : {"<", "<=", ">", ">=", "==", "!="}) {
      if (punct(op)) {
        const Pos p = next().pos;
        Expr e = binary(op, p, std::move(a), sum());
        for (const char* op2 : {"<", "<=", ">", ">=", "==", "!="})
          if (punct(op2)) fail(pos(), "comparisons do not chain: write 'a < b and b < c'");
        return e;
      }
    }
    return a;
  }
  Expr sum() {
    Expr a = prod();
    while (punct("+") || punct("-")) {
      const Tok& op = next();
      a = binary(op.text, op.pos, std::move(a), prod());
    }
    return a;
  }
  Expr prod() {
    Expr a = unary();
    while (punct("*") || punct("/")) {
      const Tok& op = next();
      a = binary(op.text, op.pos, std::move(a), unary());
    }
    return a;
  }
  Expr unary() {
    if (punct("-")) {
      Expr e;
      e.kind = Expr::Kind::unary;
      e.name = "-";
      e.pos = next().pos;
      e.args.push_back(unary());
      return e;
    }
    return primary();
  }
  Expr primary() {
    if (done()) fail(pos(), "expected a value (a number, a name or '('), found the end of the line");
    const Tok& t = next();
    Expr e;
    e.pos = t.pos;
    if (t.type == Tok::number) {
      e.kind = Expr::Kind::number;
      e.number = t.num;
      return e;
    }
    if (t.type == Tok::punct && t.text == "(") {
      Expr inner = expr();
      if (punct(",")) fail(pos(), "a list of values cannot be used here (expected one value)");
      expect(")", "to close the '('");
      return inner;
    }
    if (t.type != Tok::ident) fail(t.pos, std::format("expected a value (a number, a name or '('), found {}", t.type == Tok::string ? "a string" : "'" + t.text + "'"));
    if (punct("(")) {
      next();
      e.kind = Expr::Kind::call;
      e.name = t.text;
      if (!punct(")")) {
        e.args.push_back(expr());
        while (punct(",")) {
          next();
          e.args.push_back(expr());
        }
      }
      expect(")", std::format("to close the arguments of {}", t.text));
      return e;
    }
    if (punct(".")) {
      next();
      e.kind = Expr::Kind::member;
      e.name = t.text;
      e.member = name("a property after '.' (x, y, started, age or active)");
      return e;
    }
    e.kind = Expr::Kind::name;
    e.name = t.text;
    return e;
  }

  const std::vector<Tok>& t_;
  std::size_t i_ = 0;
  std::string_view src_;
};

// After a comma in a list of names: the next word is one more name when it is not a key or a word of the grammar and
// what follows it could follow a name (a comma, the end, ':' or ';', or the next property). So in `on a, strenght 3`
// the misspelled key is reported as such instead of being read as a module.
bool list_item(const Parser& p, std::span<const Key> keys) {
  if (!p.punct(",") || !p.peek(1) || p.peek(1)->type != Tok::ident || !list_goes_on(p.peek(1)->text, keys)) return false;
  const Tok* after = p.peek(2);
  return !after || after->type == Tok::ident || p.punct(",", 2) || p.punct(":", 2) || p.punct(";", 2);
}

Value parse_value(Parser& p, VT type, std::span<const Key> keys) {
  Value v;
  v.pos = p.pos();
  const auto point = [&](std::vector<Expr>& out) {
    p.expect("(", "to start a point (x, y)");
    out.push_back(p.expr());
    p.expect(",", "between the two coordinates of a point");
    out.push_back(p.expr());
    p.expect(")", "to close a point (x, y)");
  };
  switch (type) {
    case VT::expr:
    case VT::time:
      v.type = Value::Type::expr;
      v.exprs.push_back(p.expr());
      if (type == VT::time && p.word("s")) p.next();
      break;
    case VT::point:
      v.type = Value::Type::point;
      point(v.exprs);
      break;
    case VT::tuple:
      v.type = Value::Type::tuple;
      p.expect("(", "to start a list of values");
      v.exprs.push_back(p.expr());
      while (p.punct(",")) {
        p.next();
        v.exprs.push_back(p.expr());
      }
      p.expect(")", "to close the list of values");
      break;
    case VT::dims:
      v.type = Value::Type::dims;
      v.exprs.push_back(p.expr());
      p.expect_word("x", "between the two sizes (as in 3 x 2)");
      v.exprs.push_back(p.expr());
      break;
    case VT::name:
      v.type = Value::Type::name;
      v.names.push_back(p.name("a name"));
      break;
    case VT::names:
      v.type = Value::Type::names;
      v.names.push_back(p.name("a name"));
      while (list_item(p, keys)) {
        p.next();
        v.names.push_back(p.next().text);
      }
      break;
    case VT::points:
      v.type = Value::Type::points;
      point(v.exprs);
      while (p.punct(",") && p.punct("(", 1)) {
        p.next();
        point(v.exprs);
      }
      break;
    case VT::range:
      v.type = Value::Type::range;
      point(v.exprs);
      p.expect_word("to", "between the two corners (as in (0, 0) to (8, 4))");
      point(v.exprs);
      break;
    case VT::look:
      v.type = Value::Type::look;
      v.names.push_back(p.name("a look's name (or 'learned')"));
      if (p.word("to")) {
        p.next();
        v.names.push_back(p.name("the look to blend to"));
        p.expect_word("by", "and the blend's weight (as in look a to b by smooth(t - 2))");
        v.exprs.push_back(p.expr());
      }
      break;
    case VT::flag: v.type = Value::Type::flag; break;
  }
  return v;
}

// Properties until the end of the line (or ':' or ';'). Keys not in the schema are an error, except for a module,
// whose other keys name its effect's controls (checked when the effect is loaded).
void parse_props(Parser& p, Statement& s, bool controls_allowed) {
  const std::span<const Key> keys = keys_of(s.keyword);
  while (!p.done()) {
    if (p.punct(",")) {
      p.next();
      continue;
    }
    if (p.punct(":") || p.punct(";")) break;
    const Tok& k = p.next();
    if (k.type != Tok::ident) {
      p.fail(k.pos, std::format("expected a property name, found {}", k.type == Tok::string ? "a string" : "'" + k.text + "'"));
    }
    const Key* key = find_key(keys, k.text);
    VT type = VT::expr;
    if (key) {
      type = key->type;
    } else if (!controls_allowed) {
      std::string list;
      for (const Key& kk : keys) list += std::format("{}{}", list.empty() ? "" : ", ", kk.key);
      const std::string hint = did_you_mean(k.text, strings(keys));
      p.fail(k.pos, hint.empty() ? std::format("'{}' is not a property of {}{}", k.text, s.keyword, list.empty() ? std::string(" (it takes none)") : " (it takes: " + list + ")")
                                 : std::format("'{}' is not a property of {}{}", k.text, s.keyword, hint));
    }
    if (s.prop(k.text)) p.fail(k.pos, std::format("'{}' is given twice", k.text));
    Value v = parse_value(p, type, keys);
    s.props.push_back({k.text, std::move(v), k.pos});
  }
}

Statement parse_action(Parser& p) {
  Statement s;
  const Tok& k = p.next();
  s.keyword = k.text;
  s.pos = k.pos;
  if (k.type != Tok::ident || !is_action(k.text)) {
    const std::string hint = did_you_mean(k.text, strings(kActions));
    p.fail(k.pos, hint.empty() ? std::format("expected an action (transfer, push, suppress, hand_over, start, wake, stop, shock, scorch, burst), found '{}'", k.text)
                               : std::format("expected an action, found '{}'{}", k.text, hint));
  }
  const auto names = [&](std::vector<std::string>& out, std::string_view what) {
    out.push_back(p.name(what));
    while (list_item(p, keys_of(s.keyword))) {
      p.next();
      out.push_back(p.next().text);
    }
  };
  if (s.keyword == "transfer" || s.keyword == "hand_over") {
    s.subjects.push_back(p.name("the module material comes from"));
    p.expect("->", "between the module and where its material goes");
    if (s.keyword == "transfer") names(s.targets, "a module to receive the material");
    else s.targets.push_back(p.name("the module that takes over"));
  } else if (s.keyword == "push" || s.keyword == "suppress" || s.keyword == "start") {
    s.subjects.push_back(p.name("a module"));
  } else if (s.keyword == "wake" || s.keyword == "stop") {
    names(s.subjects, "a module");
  }
  parse_props(p, s, false);
  return s;
}

void parse_modifiers(Parser& p, Statement& s) {
  while (!p.done() && !p.punct(":")) {
    if (p.punct(",")) {
      p.next();
    } else if (p.word("as")) {
      const Pos at = p.next().pos;
      if (!s.name.empty()) p.fail(at, "the rule is named twice");
      s.name_pos = p.pos();
      s.name = p.name("the rule's name after 'as'");
    } else if (p.word("repeat")) {
      p.next();
      s.repeat = true;
    } else if (p.word("at") && p.word("most", 1)) {
      p.next();
      p.next();
      s.at_most = p.expr();
    } else {
      p.fail(p.pos(), std::format("expected ':' to end the rule (or 'as NAME', 'repeat', 'at most N'), found {}", p.found()));
    }
  }
  p.expect(":", "to end the rule");
}

void parse_body_inline(Parser& p, Statement& s) {
  while (!p.done()) {
    s.body.push_back(parse_action(p));
    if (p.done()) break;
    p.expect(";", "between two actions on one line");
  }
}

Statement parse_statement(Parser& p, std::string_view source) {
  Statement s;
  const Tok& k = p.next();
  s.pos = k.pos;
  s.keyword = k.text;
  if (k.type != Tok::ident) p.fail(k.pos, std::format("a line starts with a statement (module, field, at, when, ...), found '{}'", k.text));
  const auto declared = [&](std::string_view what) {
    s.name_pos = p.pos();
    s.name = p.name(std::format("the {}'s name", what));
    p.expect("=", std::format("after the {}'s name", what));
  };
  const std::string& w = s.keyword;
  if (w == "scene" || w == "bus" || w == "light" || w == "particles" || w == "frame" || w == "camera") {
    parse_props(p, s, false);
  } else if (w == "keyframes") {
    s.args.push_back(p.expr());
    while (p.punct(",")) {
      p.next();
      s.args.push_back(p.expr());
    }
  } else if (w == "effect") {
    declared("effect");
    const Tok* f = p.peek();
    if (!f || f->type != Tok::string) p.fail(p.pos(), std::format("expected the effect's file in quotes, found {}", p.found()));
    s.kind = p.next().text;
    parse_props(p, s, false);
  } else if (w == "look") {
    declared("look");
    if (p.word("shader")) {
      p.next();
      s.kind = "shader";
    } else if (p.word("like")) {
      p.next();
      s.kind = p.name("the look it starts from");
      if (s.kind == "shader") p.fail(p.pos(), "write 'look NAME = shader' to start from the defaults");
    } else {
      p.fail(p.pos(), std::format("expected 'shader' or 'like LOOK', found {}", p.found()));
    }
    parse_props(p, s, false);
  } else if (w == "let") {
    declared("value");
    s.args.push_back(p.expr());
  } else if (w == "input") {
    declared("input");
    s.args.push_back(p.expr());
  } else if (w == "module") {
    declared("module");
    s.kind = p.name("the module's effect");
    parse_props(p, s, true);
  } else if (w == "field") {
    declared("field");
    const Pos at = p.pos();
    s.kind = p.name("the field's kind");
    if (!in(kFieldKinds, s.kind)) p.fail(at, std::format("'{}' is not a kind of field (ceiling, vortex, wind, gust, ring, attract, heat, cold){}", s.kind, did_you_mean(s.kind, strings(kFieldKinds))));
    parse_props(p, s, false);
  } else if (w == "emit") {
    const Pos at = p.pos();
    s.kind = p.name("what to emit (sparks, embers or flakes)");
    if (!in(kEmitKinds, s.kind)) p.fail(at, std::format("'{}' cannot be emitted (sparks, embers or flakes)", s.kind));
    parse_props(p, s, false);
  } else if (w == "every") {
    p.expect_word("frame", "after 'every'");
    p.expect(":", "after 'every frame'");
    parse_body_inline(p, s);
  } else if (w == "at") {
    s.kind = "time";
    s.args.push_back(p.expr());
    if (p.word("s")) p.next();
    parse_modifiers(p, s);
    parse_body_inline(p, s);
  } else if (w == "when") {
    if (p.word("shock")) {
      p.next();
      s.kind = "shock";
      s.args.push_back(p.expr());
      p.expect_word("reaches", "after the shock's number");
      if (p.punct("(")) {
        p.next();
        s.args.push_back(p.expr());
        p.expect(",", "between the two coordinates of a point");
        s.args.push_back(p.expr());
        p.expect(")", "to close a point (x, y)");
      } else {
        s.reach = p.name("a point (x, y) or a module");
      }
    } else if (p.peek() && p.peek()->type == Tok::ident && p.word("lands", 1)) {
      const Tok& kind = p.next();
      p.next();
      if (kind.text != "ember" && kind.text != "debris") p.fail(kind.pos, "only an ember or debris lands (when ember lands ...)");
      s.kind = "lands";
      s.subjects.push_back(kind.text);
      if (p.word("where")) {
        p.next();
        s.args.push_back(p.expr());
      }
    } else {
      s.kind = "condition";
      s.args.push_back(p.expr());
    }
    parse_modifiers(p, s);
    parse_body_inline(p, s);
  } else if (is_action(w)) {
    p.fail(k.pos, std::format("'{}' is an action: put it in a rule (at ..., when ...) or under 'every frame:', indented", w));
  } else {
    const std::string hint = did_you_mean(w, strings(kTop));
    p.fail(k.pos, hint.empty() ? std::format("unknown statement '{}' (expected scene, bus, light, particles, frame, camera, keyframes, effect, look, let, input, module, field, emit, every, at or when)", w)
                               : std::format("unknown statement '{}'{}", w, hint));
  }
  if (!p.done()) p.fail(p.pos(), std::format("unexpected {} at the end of the {} statement", p.found(), w));
  (void)source;
  return s;
}

bool is_block(const Statement& s) { return s.keyword == "every" || s.keyword == "at" || s.keyword == "when"; }

}  // namespace

Script parse(std::string_view text, std::string_view source) {
  Script out;
  out.source = std::string(source);
  Statement* block = nullptr;
  for (const LogicalLine& l : lex(text, source)) {
    Parser p(l.toks, source);
    if (l.indented) {
      if (!block) p.fail(l.toks.front().pos, "this line is indented, but no rule or 'every frame:' above it is waiting for actions");
      block->body.push_back(parse_action(p));
      while (!p.done()) {
        p.expect(";", "between two actions on one line");
        block->body.push_back(parse_action(p));
      }
      continue;
    }
    out.statements.push_back(parse_statement(p, source));
    block = is_block(out.statements.back()) ? &out.statements.back() : nullptr;
  }
  for (const Statement& s : out.statements) {
    if (is_block(s) && s.body.empty()) throw Error(s.pos, std::format("this {} has no actions (indent them under it)", s.keyword == "every" ? "'every frame:'" : "rule"), source);
  }
  return out;
}

Script parse_file(const std::filesystem::path& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + path.string());
  std::stringstream ss;
  ss << f.rdbuf();
  return parse(ss.str(), path.filename().string());
}

// --- the printer -------------------------------------------------------------------------------------------------------

namespace {

int precedence(const Expr& e) {
  switch (e.kind) {
    case Expr::Kind::binary:
      if (e.name == "or") return 1;
      if (e.name == "and") return 2;
      if (e.name == "+" || e.name == "-") return 5;
      if (e.name == "*" || e.name == "/") return 6;
      return 4;  // comparisons
    case Expr::Kind::unary: return e.name == "not" ? 3 : 7;
    default: return 8;
  }
}

std::string number_text(float v) { return std::format("{}", v); }  // the shortest text that reads back as the same float

void print_expr(std::string& o, const Expr& e, int min_prec) {
  const int p = precedence(e);
  const bool paren = p < min_prec;
  if (paren) o += '(';
  switch (e.kind) {
    case Expr::Kind::number: o += number_text(e.number); break;
    case Expr::Kind::name: o += e.name; break;
    case Expr::Kind::member: o += e.name + "." + e.member; break;
    case Expr::Kind::call:
      o += e.name + "(";
      for (std::size_t i = 0; i < e.args.size(); ++i) {
        if (i) o += ", ";
        print_expr(o, e.args[i], 0);
      }
      o += ')';
      break;
    case Expr::Kind::unary:
      o += e.name == "not" ? "not " : "-";
      print_expr(o, e.args[0], p);
      break;
    case Expr::Kind::binary:
      // left-associative: the right operand needs parentheses at equal precedence; comparisons do not chain
      print_expr(o, e.args[0], p == 4 ? p + 1 : p);
      o += " " + e.name + " ";
      print_expr(o, e.args[1], p + 1);
      break;
  }
  if (paren) o += ')';
}

std::string ex(const Expr& e) {
  std::string o;
  print_expr(o, e, 0);
  return o;
}

std::string join_names(const std::vector<std::string>& v) {
  std::string o;
  for (std::size_t i = 0; i < v.size(); ++i) o += (i ? ", " : "") + v[i];
  return o;
}

std::string value_text(const Value& v) {
  const auto pt = [&](std::size_t i) { return std::format("({}, {})", ex(v.exprs[i]), ex(v.exprs[i + 1])); };
  switch (v.type) {
    case Value::Type::expr: return ex(v.exprs[0]);
    case Value::Type::point: return pt(0);
    case Value::Type::tuple: {
      std::string o = "(";
      for (std::size_t i = 0; i < v.exprs.size(); ++i) o += (i ? ", " : "") + ex(v.exprs[i]);
      return o + ")";
    }
    case Value::Type::dims: return std::format("{} x {}", ex(v.exprs[0]), ex(v.exprs[1]));
    case Value::Type::name: return v.names[0];
    case Value::Type::names: return join_names(v.names);
    case Value::Type::points: {
      std::string o;
      for (std::size_t i = 0; i < v.exprs.size(); i += 2) o += (i ? ", " : "") + pt(i);
      return o;
    }
    case Value::Type::range: return pt(0) + " to " + pt(2);
    case Value::Type::look: return v.names.size() == 1 ? v.names[0] : std::format("{} to {} by {}", v.names[0], v.names[1], ex(v.exprs[0]));
    case Value::Type::flag: return {};
  }
  return {};
}

// The properties, after a space (settings) or after a comma (statements with a head: a name, a kind, a list).
std::string props_text(const Statement& s, bool after_head = true) {
  std::string o;
  for (std::size_t i = 0; i < s.props.size(); ++i) {
    const Prop& p = s.props[i];
    o += (i || after_head ? ", " : " ") + p.key;
    const std::string v = value_text(p.value);
    if (!v.empty()) o += " " + v;
  }
  return o;
}

std::string statement_text(const Statement& s) {
  const std::string& w = s.keyword;
  std::string o = w;
  if (w == "keyframes") {
    for (std::size_t i = 0; i < s.args.size(); ++i) o += (i ? ", " : " ") + ex(s.args[i]);
  } else if (w == "effect") {
    std::string file;
    for (const char c : s.kind) file += (c == '"' || c == '\\') ? std::string("\\") + c : std::string(1, c);
    o += std::format(" {} = \"{}\"", s.name, file) + props_text(s);
  } else if (w == "look") {
    o += std::format(" {} = {}", s.name, s.kind == "shader" ? std::string("shader") : "like " + s.kind) + props_text(s);
  } else if (w == "let" || w == "input") {
    o += std::format(" {} = {}", s.name, ex(s.args[0]));
  } else if (w == "module" || w == "field") {
    o += std::format(" {} = {}", s.name, s.kind) + props_text(s);
  } else if (w == "emit") {
    o += " " + s.kind + props_text(s);
  } else if (w == "every") {
    o += " frame:";
  } else if (w == "at" || w == "when") {
    if (w == "at") {
      o += " " + ex(s.args[0]);
    } else if (s.kind == "shock") {
      o += " shock " + ex(s.args[0]) + " reaches " + (s.reach.empty() ? std::format("({}, {})", ex(s.args[1]), ex(s.args[2])) : s.reach);
    } else if (s.kind == "lands") {
      o += " " + s.subjects[0] + " lands";
      if (!s.args.empty()) o += " where " + ex(s.args[0]);
    } else {
      o += " " + ex(s.args[0]);
    }
    if (!s.name.empty()) o += " as " + s.name;
    if (s.repeat) o += ", repeat";
    if (s.at_most) o += ", at most " + ex(*s.at_most);
    o += ":";
  } else if (w == "transfer" || w == "hand_over") {
    o += " " + s.subjects[0] + " -> " + join_names(s.targets) + props_text(s);
  } else if (w == "push" || w == "suppress" || w == "start" || w == "wake" || w == "stop") {
    o += " " + join_names(s.subjects) + props_text(s);
  } else {
    o += props_text(s, false);
  }
  return o;
}

}  // namespace

std::string print(const Expr& e) { return ex(e); }

std::string print(const Script& s) {
  std::string o;
  for (const Statement& st : s.statements) {
    o += statement_text(st) + "\n";
    for (const Statement& a : st.body) o += "  " + statement_text(a) + "\n";
  }
  return o;
}

}  // namespace nfx::compose::script
