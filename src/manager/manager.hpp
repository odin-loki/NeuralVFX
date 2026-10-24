// The effect manager (include/neuralfx/nvfx_manager.h, docs/MANAGER.md): many rollout effects within a CPU budget.
//
// Instances are what the engine plays; simulations (Sim) are what runs. A simulation is one shard timeline of a rollout
// effect, exactly as nvfx_render plays it (src/runtime/nvfx.cpp: shards of 6 s from start points chosen by the seed,
// rolled ahead and crossfaded), but advanced only to keyframes: every `div` frames (30, 15, 10 or 7.5 Hz) the state is
// stepped to the next keyframe and drawn at the simulation's size (32 to 256 px). Frames between keyframes are drawn by
// interpolation (interp.hpp). The machinery at one size (two runners, keyframe pictures) is a Slot from a pool per
// effect and size; a level change moves the state to a slot of another size (RolloutRunner::transfer_from: the coarse
// state exactly, the fine fields resampled). Instances with the same effect, controls and seed view one simulation;
// copies offset in time and proxies view a ring of past keyframes kept at copy size.
//
// A tick: begin() plans on the calling thread (instances to simulations, levels against the budget from a cost model,
// which updates run now, the jobs); the jobs (one per simulation with work) run on any threads, each with its own step
// scratch, and touch only their simulation and the instances that view it; end() publishes the pictures. Every
// decision is a function of the calls made and the cost model, so the pictures do not depend on the threads.
//
// Drawn at half size where it pays (docs/MANAGER.md §2.10): a simulation's level is its picture's, chosen as before; a
// simulation large on screen (nvfx_manager_desc.half_px, half_share) is drawn at the level below (its slot is there), and
// its picture is the runtime's bilinear 2x of the drawing (rt_upscale.hpp, as nvfx_instance_set_upscale draws it) or the
// drawing itself (NVFX_UPSCALE_OFF). The planner works on picture levels; the draw level is the picture level minus one
// when `half` is set, and the cost model counts the drawing's step and pictures and the 2x.
#pragma once

#include <neuralfx/nvfx.h>
#include <neuralfx/nvfx_manager.h>

#include "interp.hpp"
#include "../runtime/rt_common.hpp"
#include "../runtime/rt_upscale.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string_view>
#include <thread>
#include <vector>

namespace nfx::mgr {

inline constexpr int kLevels = 4;
inline constexpr std::array<int, kLevels> kSizes{32, 64, 128, 256};
inline constexpr int kMaxDiv = 4;       // frames per keyframe: 1 to 4 (30 to 7.5 Hz at 30 fps)
inline constexpr int kMaxControls = 8;
inline constexpr std::int64_t kBlend = 15;  // shard crossfade frames, as nvfx.cpp
// Drawing at half size (nvfx_manager_desc.half_px): below 0, the default, each effect's measured default by its name
// (docs/REPORT.md §22.11): on from kHalfOnPx on screen (the threshold measured) for the effects that passed both of the
// rule's conditions (fire, smoke, steam, explosion), off for magic (it cost 0.54 of full size, above the 0.5 allowed) and
// for every effect not measured; 0, off for every effect; above 0, that threshold for every effect.
inline constexpr float kDefaultHalfPx = -1.f;
inline constexpr float kHalfOnPx = 256.f;
bool half_default_on(std::string_view effect);

int level_of_size(int size);  // -1 if not one of kSizes

struct Track {
  std::unique_ptr<rt::RolloutRunner> r;
  std::int64_t shard = -1, frame = 0;
  std::uint64_t seed = 0;
};

// A simulation's machinery at one size.
struct Slot {
  int level = 0, size = 0;
  bool proxy_pool = false;  // reserved for proxy simulations
  std::array<Track, 2> t;
  std::vector<std::uint8_t> mix, out;         // the shard crossfade's second picture; the composed picture
  std::array<std::vector<std::uint8_t>, 2> key;  // the two latest keyframes
  std::vector<std::uint8_t> up;  // drawn at half size: the composed picture's bilinear 2x (2 size x 2 size; empty if never)
  bool used = false;
};

// Past keyframes at copy size, for copies offset in time and proxies.
struct Ring {
  int size = 0, cap = 0, res = 0;
  std::vector<std::uint8_t> px;
  std::vector<std::int8_t> d;   // displacement since the previous keyframe, res * res * 2, quantised
  std::vector<float> scale;     // per entry: cells per step of d
  std::vector<std::int64_t> frame;
  int head = 0, count = 0;      // entries head - count .. head - 1 (mod cap), oldest first
  int slot(int k) const { return ((head - count + k) % cap + cap) % cap; }  // k-th oldest
};

struct Model;

struct Sim {
  bool used = false;
  int model = 0, index = 0;
  std::uint64_t serial = 0;  // order of creation, for phases
  std::array<float, kMaxControls> controls{};
  std::uint64_t seed = 0;
  bool shareable = false;
  int proxy_cell = -1;  // >= 0: a proxy simulation
  // the clock: seconds of the effect shown (frame = floor(clock * fps)); runs while ready and not paused
  double clock = 0.0, subframe = 0.0;
  bool started = false, ready = false;
  std::int64_t preroll = 0;  // frames to keyframe ahead before the first picture (time-offset copies, proxies)
  int phase = 1;             // the first interval's length (staggers updates)
  int refs = 0;  // instances attached (viewers, and one waiting for it to be ready)
  // level
  Slot* slot = nullptr;
  Slot* next_slot = nullptr;  // a level change, carried out by this tick's job
  Slot* old_slot = nullptr;   // the slot it left in this tick's job, released by end()
  int div = 1, next_div = 1;
  bool half = false, next_half = false;  // drawn at half size: the picture level is the slot's (next slot's) plus one
  double last_up = -1e30;
  // keyframes: slot->key[ka] at frame fa, slot->key[ka ^ 1] at frame fb (the state is at fb)
  int ka = 0;
  std::int64_t fa = 0, fb = 0;
  std::vector<float> acc, dab;  // displacement accumulated since fb, and from fa to fb (cells, res * res * 2)
  Ring* ring = nullptr;          // past keyframes (proxies always; shared simulations with copies offset in time)
  // the composed picture for viewers without an offset
  const std::uint8_t* pic = nullptr;
  int pic_size = 0;
  std::int64_t shown = -1;
  bool pic_interp = false;
  // planning (begin)
  int viewers = 0, visible = 0, first_viewer = -1;
  float px = 0.f, importance = 0.f;
  bool paused = false, parked = false, locked = false, all_proxy = true;
  double paused_s = 0.0, idle_s = 0.0;
  int want_level = 0, want_div = 1, plan_level = -1, plan_div = 1;  // picture levels
  bool want_half = false, plan_half = false;
  bool plan_proxy = false;
  int proxy_target = -1;  // the proxy cell the plan sends its viewers to
  std::int64_t max_offset = 0;
  // this tick
  bool due = false, run = false, compose = false, transfer_job = false;
  std::int64_t show = 0;  // the display frame
  int deferred = 0;
  float cost = 0.f;
  bool updated = false, held = false, composed = false;
  float compose_cost = 0.f;
  int steps = 0, renders = 0;
};

struct Inst {
  bool used = false, alive = false;
  std::uint16_t gen = 0;
  int model = 0, nc = 0;
  std::array<float, kMaxControls> controls{};
  std::uint64_t seed = 0;
  bool share = true, allow_proxy = true;
  float offset_s = 0.f, priority = 1.f;
  std::array<float, 3> pos{};
  float world_size = 1.f, screen_px = 0.f;
  bool screen_mode = false, screen_visible = true;
  int lock_level = -1, lock_div = 0;
  // attachment
  int sim = -1;            // the simulation it views (index in its model)
  int own_wait = -1;       // a simulation of its own warming up while it still views a proxy
  bool proxied = false, controls_dirty = false, culled = false;
  std::int64_t offset = 0;  // frames behind its simulation
  double switched_at = -1e30;
  int next_viewer = -1;    // the next instance viewing the same simulation
  // this tick
  float px = 0.f, importance = 0.f;
  bool visible = false, finished = false;
  // pictures: its own buffer (offset copies, proxies, fades), the fade's snapshot
  std::vector<std::uint8_t> buf, snap;
  float fade = 0.f;  // seconds left of a fade from snap
  bool use_buf = false, buf_new = false, buf_interp = false;
  std::int64_t buf_frame = -1;  // the frame its buffer holds (-1: none, or a fade in progress)
  const std::uint8_t* pic = nullptr;
  int pic_size = 0, pic_flags = 0;
  double pic_time = 0.0;
};

struct Model {
  const nvfx_effect* fx = nullptr;
  const rt::RolloutEffect* re = nullptr;
  int res = 32, nc = 0, frames = 0, history = 0, proxy_steps = 3, proxy_level = 1, proxy_div = 4;
  float fps = 30.f, advect = 1.f;
  bool loop = true;
  nvfx_interp interp = NVFX_INTERP_CROSSFADE;  // drawing between keyframes (the manager's, or the model's own)
  rt::Precision precision = rt::Precision::float32;  // its simulations' (study K)
  int isa = 0;                 // its runners' ISA as an index into Work::up (0 baseline, 1 AVX2, 2 AVX-512)
  float half_px = -1.f;        // nvfx_manager_model_desc.half_px: < 0 the manager's thresholds, 0 never, > 0 its own
  bool half_default = false;   // its name's measured default is on (half_default_on; the manager's half_px below 0)
  bool half_bufs = false;      // its slots below the top level hold a 2x picture (the policy may draw it at half size)
  std::int64_t shard = 0;      // frames per shard (0: one continuous rollout)
  float step_factor = 1.f;     // steps per frame on average (the next shard's warm-up and pre-roll)
  float render_factor = 1.f;   // pictures per keyframe on average (the shard crossfades)
  std::array<int, kLevels> slots_n{};
  std::array<float, kLevels> step_ms{}, render_ms{};
  std::vector<std::unique_ptr<Slot>> slots;
  std::array<std::vector<Slot*>, kLevels> free;
  std::vector<Sim> sims;              // own_sims simulations, then proxy_max proxies
  std::vector<int> free_sims, free_proxy;
  int own_sims = 0, proxy_max = 0;
  std::vector<Slot*> proxy_free;      // slots at proxy_level reserved for proxies
  std::vector<std::unique_ptr<Ring>> rings;  // one per proxy (by index), then a pool for offset copies
  std::vector<Ring*> free_rings;
  std::vector<int> proxy_sim;        // per proxy cell
  std::vector<std::uint8_t> planned;  // per proxy cell: counted in this tick's plan
  std::vector<int> waiting;           // per proxy cell: its instances shown from another cell's proxy
  int min_level = 0, max_level = 0;
};

class Manager {
 public:
  explicit Manager(const nvfx_manager_desc& d);
  ~Manager();
  Manager(const Manager&) = delete;
  Manager& operator=(const Manager&) = delete;

  nvfx_status add_model(const nvfx_effect* fx, const nvfx_manager_model_desc& d, int* id);
  nvfx_status set_cost(int model, int size, float step_ms, float render_ms);
  nvfx_status get_cost(int model, int size, float* step_ms, float* render_ms) const;
  nvfx_status calibrate(int model, int frames);
  nvfx_status set_camera(const float* pos, const float* fwd, const float* up, float fov, float w, float h);
  void set_budget(float ms) { desc_.budget_ms = ms; }

  nvfx_status spawn(const nvfx_spawn_desc& d, nvfx_handle* out);
  Inst* find(nvfx_handle h);
  const Inst* find(nvfx_handle h) const;
  nvfx_status stop(nvfx_handle h);
  nvfx_status set_controls(nvfx_handle h, const float* c, int n);
  nvfx_status set_position(nvfx_handle h, const float* p, float size);
  nvfx_status set_screen(nvfx_handle h, float px, int visible);
  nvfx_status set_priority(nvfx_handle h, float p);
  nvfx_status set_lock(nvfx_handle h, int size, int div);

  nvfx_status tick(double dt);
  nvfx_status begin(double dt, int* jobs);
  void run_jobs(int slot);
  nvfx_status end();
  nvfx_status picture(nvfx_handle h, nvfx_manager_picture* out) const;
  void stats(nvfx_manager_stats* out) const { *out = stats_; }
  int job_slots() const { return static_cast<int>(work_.size()); }

 private:
  struct Work {  // per thread that runs jobs: its step scratch and temporaries
    std::unique_ptr<rt::RolloutScratch> scratch;
    std::vector<float> uv;
    std::vector<std::uint8_t> px;
    // the bilinear 2x of a drawing at half size, per ISA (Model::isa) and draw level (made by add_model)
    std::array<std::array<std::unique_ptr<rt::Upscaler>, kLevels>, 3> up;
    double ms = 0.0;
    char pad[64] = {};
  };

  // instances and simulations
  Sim* sim_of(const Inst& i) { return i.sim < 0 ? nullptr : &models_[static_cast<std::size_t>(i.model)].sims[static_cast<std::size_t>(i.sim)]; }
  int new_sim(Model& M, int model, bool proxy = false);
  void release_sim(Model& M, Sim& s);
  void free_slot(Model& M, Slot*& slot);
  void attach(Inst& in, int model, int sim, std::int64_t offset);
  void detach(Inst& in);
  int find_shared(const Model& M, const Inst& in) const;
  int proxy_cell(const Model& M, const Inst& in) const;
  int ensure_proxy(int model, int cell);
  std::int64_t proxy_offset(const Model& M, const Inst& in, int p) const;
  int nearest_proxy(const Model& M, int cell) const;
  bool proxy_ok(const Model& M, const Sim& s) const;
  bool evict(Model& M);
  void start_fade(Inst& in);
  bool fork(Inst& in);

  // planning
  void view(Inst& in);
  void want(Model& M, Sim& s);
  float half_threshold(const Model& M) const;  // pixels on screen from which a simulation is drawn at half size (0: never)
  bool half_fits(const Model& M, int level) const;  // a picture at `level` can be drawn at the level below
  float cost(const Model& M, int level, int div, bool half = false) const;
  float compose_ms(const Model& M, int size) const;
  float up_ms(const Model& M, int level, bool half) const;  // the 2x of a picture at `level` (0 unless half and bilinear)
  std::array<float, 3> rate_px() const { return {desc_.rate_px[0], desc_.rate_px[1], desc_.rate_px[2]}; }
  void plan();
  void apply();
  void schedule(double dt);

  // jobs
  void process(Sim& s, Work& w);
  void transfer(Model& M, Sim& s);
  int update(Model& M, Sim& s, Work& w);
  bool advance(Model& M, Sim& s, Slot& L, std::int64_t f, int limit, int& steps);
  bool bring(Model& M, Sim& s, Track& t, bool shown, std::int64_t k, std::int64_t f, int limit, int& steps);
  void draw(Model& M, Sim& s, Slot& L, std::int64_t f, std::uint8_t* out);
  void push_ring(Sim& s, const std::uint8_t* px, int size, std::int64_t frame, const float* d);
  void compose_main(Model& M, Sim& s, Work& w);
  void compose_drawing(Model& M, Sim& s);
  void compose_view(Model& M, Sim& s, Inst& in, Work& w);
  int ring_view(Model& M, Sim& s, std::int64_t f, std::uint8_t* out, Work& w);
  void interpolate(Model& M, const std::uint8_t* a, const std::uint8_t* b, int size, float wt, const float* uv, std::uint8_t* out) const;

  void fit_work(const rt::RolloutRunner& r, int res);
  void worker(int index);

  nvfx_manager_desc desc_{};
  nvfx_interp interp_ = NVFX_INTERP_MOTION;
  std::vector<Model> models_;
  std::vector<Inst> insts_;
  std::vector<int> free_insts_;
  std::vector<Sim*> jobs_, order_;
  std::vector<Inst*> order_inst_;
  std::vector<Work> work_;
  std::atomic<int> next_job_{0};
  int n_jobs_ = 0;
  bool in_tick_ = false;
  double now_ = 0.0, hz_ = 60.0, dt_ = 0.0;
  std::uint64_t serial_ = 0, tick_ = 0;
  // camera
  bool camera_ = false;
  std::array<float, 3> cam_pos_{}, cam_fwd_{0, 0, 1}, cam_up_{0, 1, 0}, cam_right_{1, 0, 0};
  float focal_ = 1.f, view_w_ = 1920.f, view_h_ = 1080.f;
  // threads
  std::vector<std::thread> threads_;
  std::atomic<std::uint32_t> epoch_{0};
  std::atomic<int> active_{0};
  std::atomic<bool> stop_{false};
  // the tick
  double t_begin_ = 0.0, plan_wall_ = 0.0;
  nvfx_manager_stats stats_{};
  float planned_ = 0.f, plan_total_ = 0.f;
  std::size_t memory_ = 0;
  std::array<float, kLevels> fade_ms_{}, motion_ms_{};  // composing one picture: crossfade, motion-compensated
  std::array<float, kLevels> up_ms_{};  // the bilinear 2x into a picture of that level (from the level below)
  float proxy_cut_ = 0.f;
  int split_model_ = -1, split_cell_ = -1;  // a proxy the plan adds for instances shown from another cell's  // the importance below which new instances start as proxies (the budget is short)
};

}  // namespace nfx::mgr
