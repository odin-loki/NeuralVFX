// The effect library (docs/LIBRARY.md): a catalogue of named, ready-to-use effects written in the scene-script language,
// and the stages that play them: one scene with room for a number of instances of chosen effects, which a game spawns,
// moves, drives and stops while the scene plays.
//
// A catalogue is a set of files (.nvfxl). Each holds, first, the learned effects and the field-shader looks its entries
// use (`effect`, `look`, as in a scene), then entries: an `entry` line, its `param` lines and the statements of a small
// scene that starts when the effect is spawned, at an anchor (X, Y) and a scale:
//
//   effect fire = "fire.nvfx"
//   entry campfire, category fire, size 300 x 380, anchor base, loop, show 2, about "A campfire ..."
//   param intensity = 0.7, range (0, 1), about "how big it burns"
//   module flame = fire, size 128, width 200 * scale, at (X, Y), sink 0.08, start 2, seed 11, intensity intensity
//   emit embers, on flame
//
// In an entry, t is the time since the spawn, X and Y the anchor, scale the size it was spawned at and variant a number
// in [0, 1) that differs from spawn to spawn. A stage instantiates each entry once per room it has, renaming what the
// entry declares, and turns the entry's rules into rules that fire once per spawn; a spawn and a stop are inputs of the
// scene, so everything stays a scene script (Stage::text) that plays without allocating.
#pragma once

#include "script.hpp"

#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace nfx::library {

namespace sc = nfx::compose::script;

struct Param {
  std::string name, about;
  float value = 0.f, lo = 0.f, hi = 1.f;  // its default and its range
  sc::Pos pos;
};

// (X, Y) is the bottom centre of the effect's box, its centre, or the middle of its left edge (a jet's nozzle)
enum class Anchor { base, centre, left };

struct Entry {
  std::string name, category, about, file;  // file: the catalogue file it is in
  sc::Pos pos;
  float width = 0.f, height = 0.f;  // its box, world pixels at scale 1
  Anchor anchor = Anchor::base;
  bool loop = true;                 // plays until stopped; else a one-shot of `duration` seconds
  float duration = 0.f;
  float show = 1.f;                 // a moment that shows it well, seconds after the spawn (the gallery's)
  std::vector<Param> params;
  std::vector<std::string> effects;   // the learned effects it plays (names of `effect` statements), in order
  std::vector<std::string> elements;  // the procedural elements it uses (embers, lamp, arc, shock, ...), in order
  std::vector<sc::Statement> body;    // its statements, as parsed
};

struct Catalogue {
  std::vector<sc::Statement> effects;  // `effect` statements (one per name)
  std::vector<sc::Statement> looks;    // looks shared by the entries
  std::vector<Entry> entries;          // in file order
  const Entry* find(std::string_view name) const;
  int index(std::string_view name) const;  // -1: none
  // The categories, the usual ones first (fire, smoke, explosions, steam, magic, ambient, sparks, light), then others
  // in order of appearance.
  std::vector<std::string> categories() const;
  // The file of a learned effect (`effect NAME = "FILE"`), or "" if none is declared.
  std::string effect_file(std::string_view name) const;
};

// Parse catalogue files (name, text), in this order; every entry is checked as a stage of its own (names, contexts,
// constants), without loading any effect. Throws sc::Error with the file's name and the line and column.
Catalogue parse(std::span<const std::pair<std::string, std::string>> files);
// Every *.nvfxl file of a folder, in name order.
Catalogue load_dir(const std::filesystem::path& dir);
std::vector<std::pair<std::string, std::string>> read_dir(const std::filesystem::path& dir);

struct Room {
  int entry = 0;   // index in the catalogue
  int count = 1;   // instances that may play at once
};

struct StageSpec {
  int width = 960, height = 540;
  float fps = 30.f;
  float ground = 470.f;              // world y of the ground
  bool has_bus = false;              // the field bus's rectangle (default: the scene's: the picture with margins)
  float bus_x = 0, bus_y = 0, bus_w = 0, bus_h = 0, bus_cell = 8.f;
  int particles = 8192;
  float light_gain = 0.14f, exposure = 1.2f, haze = 1.f, bloom = 1.f;
  std::vector<Room> rooms;
};

// One instance's room in a stage: the names of its inputs and rules in the stage's script.
struct Slot {
  int entry = 0;        // in the catalogue
  int room = 0, k = 0;  // which room, and which of its instances
  std::string prefix;   // what the stage's script puts before the entry's names ("s3_")
  // inputs: X, Y, scale, variant, at (the spawn's time), end (a stop's time); rules: go (the spawn), whose time is when
  // it last spawned; params: the entry's parameters, in order
  std::string X, Y, scale, variant, at, end, go;
  std::vector<std::string> params;
};

struct Stage {
  sc::Script script;          // the stage's scene
  std::vector<Slot> slots;    // rooms in order, each room's instances in order
  std::vector<std::string> files;  // the files of the learned effects it plays
  std::string text() const;   // the script as text (sc::print)
};

// The stage's scene script: `count` instances of each room's entry. Throws sc::Error.
Stage build_stage(const Catalogue& c, const StageSpec& spec);
// An entry alone (one instance, its names as written, in a stage of the given size): what parse() checks, and what a
// gallery or a viewer plays. Throws sc::Error.
Stage single(const Catalogue& c, int entry, const StageSpec& spec);

}  // namespace nfx::library
