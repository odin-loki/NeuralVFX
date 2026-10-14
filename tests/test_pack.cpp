// Effect packs (include/neuralfx/nvfx_pack.h, src/pack/pack.hpp, docs/INTEGRATION.md §5): SHA-256 against its known
// answers; packs of the tests' small rollout effects, a catalogue, a scene script and a licence, made with every coder,
// whose members come back byte for byte, opened from a file and from memory; a library and a scene from a pack playing
// the frames they play from loose files; damage found (a member's bytes, the manifest, the header, truncation, a
// member an entry needs that is not there, a newer format). With the library's eleven trained models (skipped without
// them): the full library packed, every member back byte for byte, and stages from the pack drawing the same frames
// as from the folder.
#include "compose_scene.hpp"
#include "pack.hpp"
#include "sha256.hpp"

#include <neuralfx/nvfx_manager.h>
#include <neuralfx/nvfx_pack.h>
#include <neuralfx/rollout.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
namespace pk = nfx::pack;
using Bytes = std::vector<std::uint8_t>;

const fs::path kLibraryDir = fs::path(NEURALFX_SOURCE_DIR) / "library";
const fs::path kModelsDir = NEURALFX_MODELS_LIBRARY;

Bytes bytes_of(std::string_view s) { return {s.begin(), s.end()}; }

Bytes tiny_model(std::uint64_t seed) {
  const nfx::rt::RolloutEffect t = nfx::compose::testing::tiny_effect(seed);
  std::ostringstream os;
  EXPECT_TRUE(nfx::rollout::save_model(os, t.m));
  return bytes_of(os.str());
}

Bytes read_file(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), {}};
}

void write_file(const fs::path& p, std::span<const std::uint8_t> b) {
  fs::create_directories(p.parent_path());
  std::ofstream(p, std::ios::binary).write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
}

struct TempDir {
  fs::path path;
  TempDir() {
    std::random_device rd;
    path = fs::temp_directory_path() / std::format("nvfx_pack_test_{:08x}{:08x}", rd(), rd());
    fs::create_directories(path);
  }
  ~TempDir() {
    std::error_code ec;
    fs::remove_all(path, ec);
  }
};

constexpr const char* kCatalogue = R"(
effect tiny = "tiny.nvfx"
look hot = shader, heat_scale 1, emission 2

entry blob, category fire, size 64 x 64, anchor base, loop, show 0.5, about "a tiny looping effect"
param intensity = 0.5, range (0, 1), about "its control"
module m = tiny, size 32, width 48 * scale, at (X, Y), look hot, start 1, seed 7, controls (0.5, 0.5, 0.5),
           intensity intensity
emit embers, on m, tries 1, chance 0.5, spread 10

entry pop, category explosions, size 64 x 64, anchor centre, once 0.5, about "a tiny one-shot"
module m = tiny, size 32, width 40 * scale, at (X, Y + 20 * scale), start 0, seed 3
at 0: shock at (X, Y), speed 300
)";

constexpr const char* kCatalogue2 = R"(
effect other = "fx/other.nvfx"

entry zap, category sparks, size 80 x 40, anchor left, loop, about "a procedural arc"
emit arc, along (X - 30, Y - 10) to (X + 30, Y - 10), rate 10, seed 2
field shine = lamp, at (X, Y - 10), radius 20, strength 0.5

entry puff, category smoke, size 64 x 64, anchor base, loop, about "the other tiny effect"
module m = other, size 32, width 48 * scale, at (X, Y), start 0, seed 9
)";

constexpr const char* kScript = R"(
scene size 160 x 90, fps 30, length 2, ground 80
effect tiny = "tiny.nvfx"
effect other = "other"
input spot = 40
module fire = tiny, size 32, at (spot, ground), start 1, seed 5
module torch = other, size 32, at (120, ground), start 0, seed 6
when 0 as boom, repeat:
  shock at (80, 50), speed 600
)";

// The members of the tests' pack, in the order the tool writes them: models, catalogue, script, licence.
std::vector<pk::Input> tiny_members() {
  return {{"tiny.nvfx", pk::Kind::model, tiny_model(3)},
          {"fx/other.nvfx", pk::Kind::model, tiny_model(4)},
          {"a.nvfxl", pk::Kind::catalogue, bytes_of(kCatalogue)},
          {"b.nvfxl", pk::Kind::catalogue, bytes_of(kCatalogue2)},
          {"camp.nvfxs", pk::Kind::script, bytes_of(kScript)},
          {"LICENCE.txt", pk::Kind::licence, bytes_of("Effects: CC-BY 4.0, the NeuralVFX authors.\n")}};
}

std::vector<pk::Need> tiny_needs() { return {{"blob", {"tiny.nvfx"}}, {"pop", {"tiny.nvfx"}}, {"zap", {}}, {"puff", {"fx/other.nvfx"}}}; }

std::vector<pk::Cost> tiny_costs() { return {{"tiny.nvfx", 32, 0.25f, 0.05f}, {"tiny.nvfx", 64, 0.26f, 0.125f}}; }

Bytes tiny_pack(pk::Coder coder = pk::Coder::light) {
  pk::PackOptions o;
  o.coder = coder;
  return pk::write(tiny_members(), tiny_needs(), tiny_costs(), o).file;
}

struct PackHandle {
  nvfx_pack* p = nullptr;
  nvfx_scene_error err{};
  nvfx_status status = NVFX_OK;
  explicit PackHandle(std::span<const std::uint8_t> file, int threads = 1, const char* name = "t.nvfxp") {
    nvfx_pack_desc d;
    nvfx_pack_desc_init(&d);
    d.data = file.data();
    d.bytes = file.size();
    d.threads = threads;
    d.name = name;
    status = nvfx_pack_open(&d, &p, &err);
  }
  ~PackHandle() { nvfx_pack_free(p); }
};

std::uint64_t fnv(std::span<const std::uint8_t> b, std::uint64_t h = 1469598103934665603ull) {
  for (const std::uint8_t c : b) h = (h ^ c) * 1099511628211ull;
  return h;
}

// A stage of `lib` with a room for each effect named, each spawned at 0.1 s, played `frames` frames: the hash of every
// frame's pixels, one after another.
std::uint64_t play_stage(const nvfx_library* lib, const std::vector<const char*>& effects, int w, int h, int frames, int threads = 2) {
  std::vector<nvfx_stage_room> rooms;
  for (const char* e : effects) rooms.push_back({e, 1, 0});
  nvfx_stage_desc d;
  nvfx_stage_desc_init(&d);
  d.width = w;
  d.height = h;
  d.rooms = rooms.data();
  d.n_rooms = static_cast<int>(rooms.size());
  d.threads = threads;
  nvfx_stage* s = nullptr;
  nvfx_scene_error err;
  EXPECT_EQ(nvfx_stage_create(lib, &d, &s, &err), NVFX_OK) << err.message;
  if (!s) return 0;
  nvfx_scene* scene = nvfx_stage_scene(s);
  for (std::size_t i = 0; i < effects.size(); ++i) {
    nvfx_fx fx = 0;
    const float x = static_cast<float>(w) * (0.2f + 0.6f * static_cast<float>(i) / static_cast<float>(std::max<std::size_t>(1, effects.size() - 1)));
    EXPECT_EQ(nvfx_stage_spawn(s, effects[i], x, 0.8f * static_cast<float>(h), 0.6f, 0.1, &fx), NVFX_OK) << effects[i];
  }
  Bytes rgba(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 4);
  std::uint64_t hash = 1469598103934665603ull;
  for (int f = 0; f < frames; ++f) {
    if (f > 0) nvfx_scene_step_frames(scene, 1);
    nvfx_scene_render(scene, rgba.data(), static_cast<std::size_t>(w) * 4);
    hash = fnv(rgba, hash);
  }
  std::uint8_t lo = 255, hi = 0;  // the last frame shows something: not one flat colour
  for (std::size_t i = 0; i < rgba.size(); i += 4) {
    lo = std::min(lo, rgba[i]);
    hi = std::max(hi, rgba[i]);
  }
  EXPECT_GT(hi - lo, 40) << "a flat frame";
  nvfx_stage_free(s);
  return hash;
}

// The pack with its manifest replaced (the header's sizes and checksum made to match): a pack whose writer erred.
Bytes with_manifest(const Bytes& file, const std::function<std::string(std::string)>& edit) {
  const auto get64 = [&](std::size_t at) {
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= std::uint64_t{file[at + static_cast<std::size_t>(i)]} << (8 * i);
    return v;
  };
  const std::size_t mbytes = static_cast<std::size_t>(get64(16)), dbytes = static_cast<std::size_t>(get64(24));
  const std::string manifest = edit(std::string(file.begin() + pk::kHeaderBytes, file.begin() + static_cast<std::ptrdiff_t>(pk::kHeaderBytes + mbytes)));
  Bytes out(file.begin(), file.begin() + pk::kHeaderBytes);
  out.insert(out.end(), manifest.begin(), manifest.end());
  out.insert(out.end(), file.end() - static_cast<std::ptrdiff_t>(dbytes), file.end());
  const auto put64 = [&](std::size_t at, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) out[at + static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(v >> (8 * i));
  };
  put64(16, manifest.size());
  put64(32, out.size());
  const pk::Digest d = pk::sha256(bytes_of(manifest));
  std::memcpy(out.data() + 40, d.data(), 32);
  return out;
}

std::string replace(std::string s, const std::string& from, const std::string& to) {
  const std::size_t at = s.find(from);
  EXPECT_NE(at, std::string::npos) << from;
  if (at != std::string::npos) s.replace(at, from.size(), to);
  return s;
}

}  // namespace

TEST(Pack, Sha256GivesTheKnownAnswers) {
  EXPECT_EQ(pk::hex(pk::sha256({})), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  EXPECT_EQ(pk::hex(pk::sha256(bytes_of("abc"))), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  EXPECT_EQ(pk::hex(pk::sha256(bytes_of("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"))),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  pk::Sha256 s;  // a million 'a', in pieces that straddle the blocks
  const Bytes a(1000, 'a');
  for (std::size_t done = 0, k = 1; done < 1000000; k = k * 7 % 999 + 1) {
    const std::size_t n = std::min<std::size_t>(k, 1000000 - done);
    s.update(std::span(a.data(), n));
    done += n;
  }
  EXPECT_EQ(pk::hex(s.finish()), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
  pk::Digest d{};
  EXPECT_TRUE(pk::parse_hex("CDC76E5C9914FB9281A1C7E284D73E67F1809A48A497200E046D39CCC7112CD0", d));
  EXPECT_EQ(pk::hex(d), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
  EXPECT_FALSE(pk::parse_hex("xyz", d));
}

TEST(Pack, EveryMemberComesBackByteForByteWithEveryCoder) {
  const auto members = tiny_members();
  for (const pk::Coder coder : {pk::Coder::store, pk::Coder::fast, pk::Coder::light, pk::Coder::full}) {
    pk::PackOptions o;
    o.coder = coder;
    o.threads = 2;
    const pk::Written w = pk::write(members, tiny_needs(), tiny_costs(), o);
    EXPECT_EQ(w.file, pk::write(members, tiny_needs(), tiny_costs(), o).file) << "the same pack twice";
    for (const int threads : {1, 3}) {
      const pk::Pack p = pk::read(w.file, {threads});
      ASSERT_EQ(p.members.size(), members.size());
      for (std::size_t i = 0; i < members.size(); ++i) {
        EXPECT_EQ(p.members[i].rec.name, members[i].name);
        EXPECT_EQ(p.members[i].rec.kind, members[i].kind);
        EXPECT_TRUE(std::ranges::equal(p.members[i].bytes(), members[i].data)) << members[i].name << " with " << pk::coder_name(coder);
        EXPECT_EQ(p.members[i].buf.back(), 0) << "a NUL after every member";
        EXPECT_EQ(p.members[i].rec.sha256, pk::sha256(members[i].data));
        if (coder == pk::Coder::store) {
          EXPECT_EQ(p.members[i].rec.method, pk::Method::stored);
        }
      }
      EXPECT_EQ(p.manifest.needs.size(), 4u);
      EXPECT_EQ(p.manifest.costs.size(), 2u);
    }
    if (coder != pk::Coder::store) {  // the models are worth coding; and coding pays on the whole
      EXPECT_EQ(w.manifest.members[0].method, pk::Method::cm) << pk::coder_name(coder);
      std::size_t loose = 0;
      for (const auto& m : members) loose += m.data.size();
      EXPECT_LT(w.file.size(), loose) << pk::coder_name(coder);
    }
  }
}

TEST(Pack, TheManifestSaysWhatThePackHolds) {
  const Bytes file = tiny_pack();
  const pk::Pack p = pk::read(file);
  EXPECT_TRUE(p.manifest_text.starts_with("nvfx-pack 1\nmember tiny.nvfx model cm ")) << p.manifest_text;
  EXPECT_NE(p.manifest_text.find("\nneeds blob tiny.nvfx\n"), std::string::npos);
  EXPECT_NE(p.manifest_text.find("\nneeds zap\n"), std::string::npos);
  EXPECT_NE(p.manifest_text.find("\nneeds puff fx/other.nvfx\n"), std::string::npos);
  EXPECT_NE(p.manifest_text.find("\ncost tiny.nvfx 64 0.26 0.125\n"), std::string::npos);
  EXPECT_EQ(pk::parse_manifest(p.manifest_text).members.size(), 6u);
  // a reader ignores records it does not know, and comments
  const pk::Manifest m = pk::parse_manifest("nvfx-pack 1\n# a comment\nlater something new\nneeds x\n");
  EXPECT_EQ(m.needs.size(), 1u);
  // names the writer refuses
  for (const char* bad : {"", "/abs.nvfx", "a b.nvfx", "../up.nvfx", "a/./b", "c:\\x", "a//b"}) EXPECT_FALSE(pk::valid_name(bad)) << bad;
  EXPECT_TRUE(pk::valid_name("fx/fire.nvfx"));
  auto members = tiny_members();
  members[1].name = "tiny.nvfx";
  EXPECT_THROW(pk::write(members, {}, {}, {}), pk::Error) << "two members of one name";
  EXPECT_THROW(pk::write(tiny_members(), {{"blob", {"missing.nvfx"}}}, {}, {}), pk::Error) << "an entry needs a model that is not there";
  EXPECT_THROW(pk::write(tiny_members(), {{"blob", {"a.nvfxl"}}}, {}, {}), pk::Error) << "an entry needs a member that is not a model";
  EXPECT_THROW(pk::write(tiny_members(), {}, {{"tiny.nvfx", 48, 1.f, 1.f}}, {}), pk::Error) << "a cost for a size the manager does not have";
}

TEST(Pack, OpensFromAFileAndFromMemory) {
  const TempDir dir;
  const Bytes file = tiny_pack();
  write_file(dir.path / "t.nvfxp", file);
  nvfx_pack_desc d;
  nvfx_pack_desc_init(&d);
  EXPECT_EQ(d.threads, 1);
  const std::string path = (dir.path / "t.nvfxp").string();
  d.path = path.c_str();
  nvfx_pack* p = nullptr;
  nvfx_scene_error err;
  ASSERT_EQ(nvfx_pack_open(&d, &p, &err), NVFX_OK) << err.message;
  const PackHandle mem(file, 2);
  ASSERT_EQ(mem.status, NVFX_OK) << mem.err.message;
  for (const nvfx_pack* q : {static_cast<const nvfx_pack*>(p), static_cast<const nvfx_pack*>(mem.p)}) {
    nvfx_pack_info info;
    ASSERT_EQ(nvfx_pack_get_info(q, &info), NVFX_OK);
    EXPECT_EQ(info.format, 1);
    EXPECT_EQ(info.n_members, 6);
    EXPECT_EQ(info.n_models, 2);
    EXPECT_EQ(info.n_catalogue, 2);
    EXPECT_EQ(info.n_entries, 4);
    EXPECT_EQ(info.n_costs, 2);
    EXPECT_EQ(info.file_bytes, file.size());
    EXPECT_STREQ(info.licence, "Effects: CC-BY 4.0, the NeuralVFX authors.\n");
    std::size_t unpacked = 0;
    const auto members = tiny_members();
    for (int i = 0; i < info.n_members; ++i) {
      nvfx_pack_member m;
      ASSERT_EQ(nvfx_pack_get_member(q, i, &m), NVFX_OK);
      EXPECT_STREQ(m.name, members[static_cast<std::size_t>(i)].name.c_str());
      EXPECT_EQ(m.kind, static_cast<nvfx_pack_kind>(members[static_cast<std::size_t>(i)].kind));
      EXPECT_TRUE(std::ranges::equal(std::span(m.data, m.bytes), members[static_cast<std::size_t>(i)].data));
      EXPECT_EQ(m.data[m.bytes], 0);
      EXPECT_EQ(std::string(m.sha256), pk::hex(pk::sha256(members[static_cast<std::size_t>(i)].data)));
      EXPECT_LE(m.stored_bytes, m.bytes);
      unpacked += m.bytes;
    }
    EXPECT_EQ(info.unpacked_bytes, unpacked);
    nvfx_pack_member m;
    EXPECT_EQ(nvfx_pack_get_member(q, 6, &m), NVFX_ERROR_ARGUMENT);
    EXPECT_EQ(nvfx_pack_find(q, "tiny.nvfx"), 0);
    EXPECT_EQ(nvfx_pack_find(q, "tiny"), 0);
    EXPECT_EQ(nvfx_pack_find(q, "other"), 1);
    EXPECT_EQ(nvfx_pack_find(q, "fx/other.nvfx"), 1);
    EXPECT_EQ(nvfx_pack_find(q, "camp.nvfxs"), 4);
    EXPECT_EQ(nvfx_pack_find(q, "camp"), -1) << "short names are for models";
    EXPECT_EQ(nvfx_pack_find(q, "nothing"), -1);
    // entries and the models they need
    EXPECT_STREQ(nvfx_pack_entry_name(q, 0), "blob");
    EXPECT_STREQ(nvfx_pack_entry_name(q, 3), "puff");
    EXPECT_EQ(nvfx_pack_entry_name(q, 4), nullptr);
    std::vector<std::string> got;
    ASSERT_EQ(nvfx_pack_entry_models(q, 3, [](void* u, const char* n, const char* f) { static_cast<std::vector<std::string>*>(u)->push_back(std::string(n) + "=" + f); }, &got),
              NVFX_OK);
    EXPECT_EQ(got, std::vector<std::string>{"other=fx/other.nvfx"});
    EXPECT_EQ(nvfx_pack_entry_models(q, 9, [](void*, const char*, const char*) {}, nullptr), NVFX_ERROR_ARGUMENT);
    // models, and their costs
    nvfx_effect* e = nullptr;
    ASSERT_EQ(nvfx_pack_load_effect(q, "tiny", &e), NVFX_OK);
    nvfx_effect_info ei;
    nvfx_effect_get_info(e, &ei);
    EXPECT_EQ(ei.arch, 3);
    nvfx_effect_free(e);
    EXPECT_EQ(nvfx_pack_load_effect(q, "nothing", &e), NVFX_ERROR_IO);
    EXPECT_EQ(e, nullptr);
    EXPECT_EQ(nvfx_pack_load_effect(q, "a.nvfxl", &e), NVFX_ERROR_IO) << "not a model";
    float step = 0, render = 0;
    ASSERT_EQ(nvfx_pack_model_cost(q, "tiny", 64, &step, &render), NVFX_OK);
    EXPECT_EQ(step, 0.26f);
    EXPECT_EQ(render, 0.125f);
    EXPECT_EQ(nvfx_pack_model_cost(q, "tiny.nvfx", 128, &step, &render), NVFX_ERROR_ARGUMENT);
    EXPECT_EQ(nvfx_pack_model_cost(q, "other", 32, &step, &render), NVFX_ERROR_ARGUMENT);
  }
  nvfx_pack_free(p);
  // errors
  d.path = "/nonexistent/x.nvfxp";
  EXPECT_EQ(nvfx_pack_open(&d, &p, &err), NVFX_ERROR_IO);
  EXPECT_EQ(p, nullptr);
  EXPECT_STREQ(err.message, "x.nvfxp: cannot read /nonexistent/x.nvfxp");
  const std::string folder = dir.path.string();
  d.path = folder.c_str();
  EXPECT_EQ(nvfx_pack_open(&d, &p, &err), NVFX_ERROR_IO) << "a folder";
  nvfx_pack_desc_init(&d);
  EXPECT_EQ(nvfx_pack_open(&d, &p, &err), NVFX_ERROR_ARGUMENT);
  EXPECT_EQ(nvfx_pack_open(nullptr, &p, &err), NVFX_ERROR_ARGUMENT);
  EXPECT_EQ(nvfx_pack_get_info(nullptr, nullptr), NVFX_ERROR_ARGUMENT);
  nvfx_pack_free(nullptr);
}

TEST(Pack, ALibraryFromAPackPlaysTheFramesItPlaysFromLooseFiles) {
  const TempDir dir;  // the loose files, as a folder
  for (const pk::Input& m : tiny_members()) write_file(dir.path / m.name, m.data);
  nvfx_library_desc ld;
  nvfx_library_desc_init(&ld);
  const std::string folder = dir.path.string();
  ld.catalogue_dir = folder.c_str();
  nvfx_library* loose = nullptr;
  nvfx_scene_error err;
  ASSERT_EQ(nvfx_library_open(&ld, &loose, &err), NVFX_OK) << err.message;
  nvfx_library* packed = nullptr;
  {
    const PackHandle p(tiny_pack());
    ASSERT_EQ(p.status, NVFX_OK) << p.err.message;
    ASSERT_EQ(nvfx_library_open_pack(p.p, &packed, &err), NVFX_OK) << err.message;
  }  // the pack is freed: the library keeps what it needs
  ASSERT_EQ(nvfx_library_count(packed), nvfx_library_count(loose));
  for (int i = 0; i < nvfx_library_count(loose); ++i) {
    nvfx_library_effect a, b;
    nvfx_library_get_effect(loose, i, &a);
    nvfx_library_get_effect(packed, i, &b);
    EXPECT_STREQ(a.name, b.name);
    EXPECT_STREQ(a.file, b.file);
    EXPECT_STREQ(a.models, b.models);
  }
  const std::uint64_t want = play_stage(loose, {"blob", "pop", "zap", "puff"}, 192, 96, 40);
  EXPECT_EQ(play_stage(packed, {"blob", "pop", "zap", "puff"}, 192, 96, 40), want);
  EXPECT_EQ(play_stage(packed, {"blob", "pop", "zap", "puff"}, 192, 96, 40, 1), want) << "on one thread";
  nvfx_library_free(loose);
  nvfx_library_free(packed);
  // a pack without a catalogue
  pk::PackOptions o;
  const Bytes models_only = pk::write({tiny_members()[0]}, {}, {}, o).file;
  const PackHandle q(models_only);
  ASSERT_EQ(q.status, NVFX_OK);
  EXPECT_EQ(nvfx_library_open_pack(q.p, &packed, &err), NVFX_ERROR_ARGUMENT);
  EXPECT_STREQ(err.message, "t.nvfxp: the pack has no catalogue");
}

TEST(Pack, ASceneAndTheManagerTakeTheirModelsFromAPack) {
  const PackHandle p(tiny_pack());
  ASSERT_EQ(p.status, NVFX_OK);
  // the scene, from the script member, against the script with its effects passed in
  nvfx_scene_desc d;
  nvfx_scene_desc_init(&d);
  d.script_path = "camp.nvfxs";
  d.threads = 1;
  nvfx_scene* a = nullptr;
  nvfx_scene_error err;
  ASSERT_EQ(nvfx_scene_create_from_pack(p.p, &d, &a, &err), NVFX_OK) << err.message;
  nvfx_effect *tiny = nullptr, *other = nullptr;
  ASSERT_EQ(nvfx_pack_load_effect(p.p, "tiny", &tiny), NVFX_OK);
  ASSERT_EQ(nvfx_pack_load_effect(p.p, "other", &other), NVFX_OK);
  const nvfx_scene_effect fx[2] = {{"tiny.nvfx", tiny}, {"other", other}};
  nvfx_scene_desc_init(&d);
  d.script = kScript;
  d.effects = fx;
  d.n_effects = 2;
  d.threads = 1;
  nvfx_scene* b = nullptr;
  ASSERT_EQ(nvfx_scene_create(&d, &b, &err), NVFX_OK) << err.message;
  Bytes fa(160 * 90 * 4), fb(fa.size());
  for (int f = 0; f < 20; ++f) {
    nvfx_scene_step_frames(a, 1);
    nvfx_scene_step_frames(b, 1);
    if (f == 5) {
      nvfx_scene_trigger(a, "boom");
      nvfx_scene_trigger(b, "boom");
    }
    nvfx_scene_render(a, fa.data(), 160 * 4);
    nvfx_scene_render(b, fb.data(), 160 * 4);
    ASSERT_EQ(fa, fb) << "frame " << f;
  }
  nvfx_scene_free(a);
  nvfx_scene_free(b);
  // an error in a script from the pack names the member
  auto members = tiny_members();
  members[4].data = bytes_of("scene size 160 x 90\neffect tiny = \"tiny.nvfx\"\nmodule m = tiny, size 32, at (40, 80), strat 0\n");
  const PackHandle broken(pk::write(members, tiny_needs(), {}, {}).file);
  nvfx_scene_desc_init(&d);
  d.script_path = "camp.nvfxs";
  EXPECT_EQ(nvfx_scene_create_from_pack(broken.p, &d, &a, &err), NVFX_ERROR_SCRIPT);
  EXPECT_EQ(std::string(err.message).rfind("camp.nvfxs:3:", 0), 0u) << err.message;
  EXPECT_EQ(err.line, 3);
  // the manager: a model from the pack, with the pack's costs
  nvfx_manager_desc md;
  nvfx_manager_desc_init(&md);
  md.max_instances = 8;
  md.max_size = 64;
  nvfx_manager* m = nullptr;
  ASSERT_EQ(nvfx_manager_create(&md, &m), NVFX_OK);
  nvfx_manager_model_desc mm;
  nvfx_manager_model_desc_init(&mm);
  mm.max_sims = 2;
  int model = -1;
  ASSERT_EQ(nvfx_manager_add_model(m, tiny, &mm, &model), NVFX_OK);
  for (const int size : {32, 64, 128, 256}) {
    float step = 0, render = 0;
    if (nvfx_pack_model_cost(p.p, "tiny", size, &step, &render) == NVFX_OK) {
      EXPECT_EQ(nvfx_manager_set_cost(m, model, size, step, render), NVFX_OK);
    }
  }
  float step = 0, render = 0;
  nvfx_manager_get_cost(m, model, 64, &step, &render);
  EXPECT_EQ(step, 0.26f);
  nvfx_spawn_desc s;
  nvfx_spawn_desc_init(&s);
  s.model = model;
  s.screen_px = 48;
  nvfx_handle h = 0;
  ASSERT_EQ(nvfx_manager_spawn(m, &s, &h), NVFX_OK);
  nvfx_manager_picture pic{};
  for (int t = 0; t < 12 && !pic.rgba; ++t) {
    nvfx_manager_tick(m, 1.0 / 30.0);
    nvfx_manager_get_picture(m, h, &pic);
  }
  EXPECT_NE(pic.rgba, nullptr);
  nvfx_manager_free(m);
  nvfx_effect_free(tiny);
  nvfx_effect_free(other);
}

TEST(Pack, DamageIsFoundAndNamed) {
  const Bytes good = tiny_pack();
  const pk::Pack ref = pk::read(good);
  const auto open = [](const Bytes& file, nvfx_scene_error& err) {
    const PackHandle p(file);
    err = p.err;
    return p.status;
  };
  nvfx_scene_error err;
  const std::size_t data_at = good.size() - [&] {
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= std::uint64_t{good[24 + static_cast<std::size_t>(i)]} << (8 * i);
    return static_cast<std::size_t>(v);
  }();
  // a byte of each member's stored bytes
  for (const pk::Member& m : ref.members) {
    Bytes bad = good;
    bad[data_at + static_cast<std::size_t>(m.rec.offset + m.rec.stored_bytes / 2)] ^= 0x10;
    EXPECT_EQ(open(bad, err), NVFX_ERROR_FORMAT) << m.rec.name;
    EXPECT_EQ(std::string(err.message), std::format("t.nvfxp: member {} is damaged (its stored bytes do not match their SHA-256)", m.rec.name));
  }
  // the manifest, the header's checksum of it, its sizes
  Bytes bad = good;
  bad[pk::kHeaderBytes + 20] ^= 1;
  EXPECT_EQ(open(bad, err), NVFX_ERROR_FORMAT);
  EXPECT_STREQ(err.message, "t.nvfxp: damaged manifest (its SHA-256 does not match the header's)");
  bad = good;
  bad[50] ^= 1;
  EXPECT_EQ(open(bad, err), NVFX_ERROR_FORMAT);
  EXPECT_STREQ(err.message, "t.nvfxp: damaged manifest (its SHA-256 does not match the header's)");
  bad = good;
  bad[17] ^= 1;
  EXPECT_EQ(open(bad, err), NVFX_ERROR_FORMAT);
  EXPECT_STREQ(err.message, "t.nvfxp: damaged header (its sizes do not add up)");
  // truncated, extended, not a pack, a newer format
  for (const std::size_t keep : {good.size() - 1, good.size() / 2, std::size_t{200}}) {
    EXPECT_EQ(open(Bytes(good.begin(), good.begin() + static_cast<std::ptrdiff_t>(keep)), err), NVFX_ERROR_FORMAT);
    EXPECT_EQ(std::string(err.message), std::format("t.nvfxp: truncated: {} of its {} bytes", keep, good.size()));
  }
  EXPECT_EQ(open(Bytes(good.begin(), good.begin() + 40), err), NVFX_ERROR_FORMAT);
  EXPECT_STREQ(err.message, "t.nvfxp: truncated: 40 bytes, shorter than a pack's header");
  bad = good;
  bad.push_back(0);
  EXPECT_EQ(open(bad, err), NVFX_ERROR_FORMAT);
  EXPECT_EQ(std::string(err.message), std::format("t.nvfxp: {} bytes, longer than the {} its header says", bad.size(), good.size()));
  EXPECT_EQ(open(tiny_model(3), err), NVFX_ERROR_FORMAT);
  EXPECT_STREQ(err.message, "t.nvfxp: not an effect pack (no NVFXPACK header)");
  bad = good;
  bad[8] = 2;
  EXPECT_EQ(open(bad, err), NVFX_ERROR_UNSUPPORTED);
  EXPECT_STREQ(err.message, "t.nvfxp: pack format 2 is newer than this library reads (1)");
  // manifests a writer got wrong: a member an entry needs is not there, a line that does not parse (with its line), a
  // coded member whose decoded checksum differs, a newer format in the manifest, a member outside the data
  EXPECT_EQ(open(with_manifest(good, [](std::string m) { return m + "needs ghost phantom.nvfx\n"; }), err), NVFX_ERROR_FORMAT);
  EXPECT_STREQ(err.message, "t.nvfxp: entry 'ghost' needs phantom.nvfx, which is not a model of the pack");
  EXPECT_EQ(open(with_manifest(good, [](std::string m) { return replace(m, "needs pop tiny.nvfx", "needs pop LICENCE.txt"); }), err), NVFX_ERROR_FORMAT);
  EXPECT_STREQ(err.message, "t.nvfxp: entry 'pop' needs LICENCE.txt, which is not a model of the pack");
  EXPECT_EQ(open(with_manifest(good, [](std::string m) { return replace(m, "member a.nvfxl catalogue", "member a.nvfxl catalogue 12"); }), err), NVFX_ERROR_FORMAT);
  EXPECT_EQ(err.line, 4);
  EXPECT_STREQ(err.message, "t.nvfxp: manifest line 4: a member takes NAME KIND METHOD BYTES STORED OFFSET SHA256 STORED_SHA256");
  const std::string sha = pk::hex(ref.members[0].rec.sha256);
  EXPECT_EQ(open(with_manifest(good, [&](std::string m) { return replace(m, sha, std::string(64, '0')); }), err), NVFX_ERROR_FORMAT);
  EXPECT_STREQ(err.message, "t.nvfxp: member tiny.nvfx decodes to other bytes than were packed (size or SHA-256)");
  EXPECT_EQ(open(with_manifest(good, [](std::string m) { return replace(m, "nvfx-pack 1", "nvfx-pack 7"); }), err), NVFX_ERROR_UNSUPPORTED);
  EXPECT_STREQ(err.message, "t.nvfxp: pack format 7 is newer than this library reads (1)");
  EXPECT_EQ(open(with_manifest(good, [](std::string m) { return replace(m, " cm ", " zstd "); }), err), NVFX_ERROR_UNSUPPORTED);
  EXPECT_EQ(err.line, 2);
  // a member an entry needs is missing although the manifest does not say so: found when the library is opened
  auto members = tiny_members();
  members.erase(members.begin() + 1);  // fx/other.nvfx, which puff plays
  const PackHandle p(pk::write(members, {}, {}, {}).file);
  ASSERT_EQ(p.status, NVFX_OK) << p.err.message;
  nvfx_library* lib = nullptr;
  EXPECT_EQ(nvfx_library_open_pack(p.p, &lib, &err), NVFX_ERROR_FORMAT);
  EXPECT_EQ(lib, nullptr);
  EXPECT_STREQ(err.message, "t.nvfxp: entry 'puff' needs fx/other.nvfx, which the pack does not hold");
  // and a catalogue error from a pack says where, as from a folder
  members = tiny_members();
  members[2].data = bytes_of("effect tiny = \"tiny.nvfx\"\nentry blob, category fire, size 64 x 64, loop, about \"x\"\nmodul m = tiny, size 32\n");
  const PackHandle q(pk::write(members, {}, {}, {}).file);
  ASSERT_EQ(q.status, NVFX_OK);
  EXPECT_EQ(nvfx_library_open_pack(q.p, &lib, &err), NVFX_ERROR_SCRIPT);
  EXPECT_EQ(std::string(err.message).rfind("a.nvfxl:3:", 0), 0u) << err.message;
}

// The full library (the repository's catalogue and the eleven trained models; skipped without them): packed with the
// default coder, every member back byte for byte, a pack smaller than the loose files, and stages from the pack drawing
// the frames they draw from the folder.
TEST(Pack, TheFullLibraryPlaysTheSameFramesFromAPack) {
  if (!fs::exists(kModelsDir / "fire.nvfx")) GTEST_SKIP() << "the library's models are not in " << kModelsDir;
  std::vector<pk::Input> members;
  std::vector<std::string> models;
  for (const auto& de : fs::directory_iterator(kModelsDir))
    if (de.path().extension() == ".nvfx") models.push_back(de.path().filename().string());
  std::ranges::sort(models);
  for (const std::string& m : models) members.push_back({m, pk::Kind::model, read_file(kModelsDir / m)});
  std::vector<std::string> cats;
  for (const auto& de : fs::directory_iterator(kLibraryDir))
    if (de.path().extension() == ".nvfxl") cats.push_back(de.path().filename().string());
  std::ranges::sort(cats);
  for (const std::string& c : cats) members.push_back({c, pk::Kind::catalogue, read_file(kLibraryDir / c)});
  std::size_t loose = 0;
  for (const auto& m : members) loose += m.data.size();
  pk::PackOptions o;
  o.threads = 3;
  const Bytes file = pk::write(members, {}, {}, o).file;
  EXPECT_LT(file.size(), loose * 3 / 4);
  const PackHandle p(file, 3, "library.nvfxp");
  ASSERT_EQ(p.status, NVFX_OK) << p.err.message;
  for (std::size_t i = 0; i < members.size(); ++i) {
    nvfx_pack_member m;
    ASSERT_EQ(nvfx_pack_get_member(p.p, static_cast<int>(i), &m), NVFX_OK);
    EXPECT_TRUE(std::ranges::equal(std::span(m.data, m.bytes), members[i].data)) << members[i].name;
  }
  nvfx_library* packed = nullptr;
  nvfx_scene_error err;
  ASSERT_EQ(nvfx_library_open_pack(p.p, &packed, &err), NVFX_OK) << err.message;
  nvfx_library_desc ld;
  nvfx_library_desc_init(&ld);
  const std::string dir = kLibraryDir.string(), mdir = kModelsDir.string();
  ld.catalogue_dir = dir.c_str();
  ld.models_dir = mdir.c_str();
  nvfx_library* loose_lib = nullptr;
  ASSERT_EQ(nvfx_library_open(&ld, &loose_lib, &err), NVFX_OK) << err.message;
  EXPECT_EQ(nvfx_library_count(packed), nvfx_library_count(loose_lib));
  const std::vector<const char*> fx = {"campfire", "grenade", "steam_vent", "plasma_orb"};
  EXPECT_EQ(play_stage(packed, fx, 320, 200, 24), play_stage(loose_lib, fx, 320, 200, 24));
  nvfx_library_free(packed);
  nvfx_library_free(loose_lib);
}
