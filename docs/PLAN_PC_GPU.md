# Plan: high-resolution 3D effects, each made in hours on the PC's GPU from the best simulators

**Status:** the owner's target for stage 2 (24 October 2026): **every catalogue effect that has a GPU teacher in this
plan**, through [PLAN_PC_GPU.md](PLAN_PC_GPU.md) §8 (P0 to P7) and `nvfx_pc effect <name>`, not a one-off hres demo.
Nothing in §8 has started on the PC yet (`nvfx_pc`, `nvfx_gpu_sim`, `nvfx_gpu_train`, `nvfx_gpu_play` are not built).
Study hres on this PC is the **laboratory before P0**: fire and smoke at 32 x 48 x 32 are done; **smoke_r2 at hres** (R2's
teacher, spread and edge losses, calibrated bounds) is **running** (`tools/study_3d/hres_smoke_r2.sh` on
`$NEURALVFX_DATA/hres_3d/smoke_r2`) and must finish before we treat hres as having answered the smoke question for
§6.3 and §8 P3. Then **P0** (PC set-up, §2) and the GPU stack, then **P3's gate** (fire and smoke at 64 x 96 x 64 in
hours), then **P4–P7** for the rest of the gas catalogue, liquids, grains and weather rows ([WAVE1_CHECKLIST.md](WAVE1_CHECKLIST.md)).

Written 21 October 2026; revised the same day for the hours target. It refines stage 2 of [TODO.md](../TODO.md) (the
catalogue, the measuring rules and the seeds stay as they are there). The accelerated variant
([GENERATOR.md](GENERATOR.md): a small seeded generator per effect) stays a bet to test in parallel, not a substitute
for this plan unless its pilots beat §6.4 on the first effects.

**What "high resolution" means here.** TODO's stage 2 sizes:
- a **64 x 96 x 64 learned state** (64³ for effects that aren't tall): twice study hres's 32 x 48 x 32 in each
  direction, and 4 times study N's;
- **256 x 384 x 256 detail**, 4 times today's in each direction;
- **teachers at 256 x 384 x 256**, checked for convergence at 512.

**What "in hours" means.** Each effect goes from simulation to a shipped, tested model in **5 to 9 hours of the PC,
unattended** (one a night). Each family pays once, in days, for checking its teacher and training its family model.
All 33 of today's effects take about **4 to 6 weeks** of the PC (§7.4).

Four things make that possible:
1. **The GPU simulates.** Our own solvers, ported to the GPU and batched, make 100 high-resolution runs in about an
   hour. No paid VFX package is in the loop: teachers are **`nvfx_gpu_sim` / `nvfx_gpu_flip` / `nvfx_gpu_mpm`**
   (from today's `sim3d` and new GPU ports). Each effect is checked against **one** scientific reference (§3.3.1).
   Shipped models run in **FUSE** on the Desktop repo ([FUSE.md](FUSE.md)); that is integration and timing, not a
   second reference.
2. **The GPU trains.** One model per family is trained once. Each new effect is a fine-tune of it, coarse to fine
   (§6).
3. **One command per effect.** It simulates, trains, measures by the house rules and ships, unattended, and logs every
   step's time (§5).
4. **The GPU plays them.** At this resolution a step is far too slow for a CPU, so the high-resolution effects play on
   the engine's GPU. Today's CPU 3D and 2D effects stay as the lower levels of detail (§9).

Audience: the owner, and whoever runs it on the PC.

## Contents

0. [In short](#0-in-short)
1. [Why change the plan](#1-why-change-the-plan)
2. [What I need from you](#2-what-i-need-from-you)
3. [The teachers](#3-the-teachers)
4. [The PC](#4-the-pc)
5. [Software on the PC](#5-software-on-the-pc)
6. [The models](#6-the-models)
7. [An effect in hours](#7-an-effect-in-hours)
8. [Order of work, with gates](#8-order-of-work-with-gates)
9. [Playing them: the GPU runtime](#9-playing-them-the-gpu-runtime)
10. [Risks](#10-risks)
11. [What goes in git, and seeds](#11-what-goes-in-git-and-seeds)

## 0. In short

- **Per effect, 5 to 9 hours, unattended,** with one command (`nvfx_pc effect <name>`):

  | step | where | hours |
  |---|---|---|
  | simulate 100 runs of 10 s at 256 x 384 x 256 | GPU | 0.5 to 2 |
  | fine-tune the family model, first at 32 x 48 x 32, then at 64 x 96 x 64 | GPU | 3 to 5 |
  | measure: null probe, validation, freeze, the one test | GPU and CPU | about 1 |
  | ship: videos, library entry | | under 0.5 |

  The simulation and the training share the GPU, so they take turns. Liquids and grains need GPU solvers of our own
  first (§3.2).
- **Once per family:**
  - check the teacher against measurements: 1 to 2 days;
  - train the family model on its first three effects: 2 to 3 days.
- **Teachers:**
  - **For gases, our 3D solver on the GPU** (`nvfx_gpu_sim`: today's `sim3d`, whose `smoke_r` passes the plume
    checks; R2's momentum fix for smoke-like presets).
  - **For liquids and grains, GPU solvers of our own** (FLIP and MPM), because film-scale CPU solvers take days at
    this resolution and are not in the budget.
  - **One reference per effect** (§3.3.1): the best single anchor for that row—usually a correlation, intercomparison
    dataset, or one research-code case—not several stacked.
  - **FUSE** plays and budgets shipped effects (owner source: `Desktop/FUSE`, git); the repo adapter is `engines/fuse/`.
- **Every teacher must pass its family's physical checks before any training data is made:** flame height, puffing
  frequency, plume spread, front speed, angle of repose (§3.4)—the measurable side of **range and spread** (§1.3).
- **Model:** one network per family (gas 25 effects, free surface 7, granular), conditioned on effect and controls,
  with the stage 1 design at high resolution. Each effect's fine-tune is the shipped `.nvfx3`, played on the GPU, with
  a small CPU version (16 x 24 x 16, study N2's size) for lower levels of detail.
- **Training on the GPU** uses LibTorch, PyTorch's C++ library: no Python, and no pretrained networks. It sits in a
  separate build option, off by default, so this repository's CPU build and CI do not change.
- **Each effect has a gate,** run under the house rules: the rule written before validation, held-out settings and
  seeds, one test, paired bootstrap intervals. The rule comes from a template committed once per family, and its
  numbers (margins and run counts from a null probe) are committed by the pipeline before validation, as study Q2
  does.
- **The first gate** is a high-resolution 3D fire and smoke that pass where studies N2 and R failed, each made in
  under 9 hours and played within a GPU budget measured in FUSE.

## 1. Why change the plan

### 1.1 What study hres's two shipped effects teach (this PC, October 2026)

These are the **two worked examples** the owner wants carried into PLAN_PC_GPU before the full catalogue roll-out.
They are not the destination (64 x 96 x 64 learned state, GPU sim, train, play, §0); they are the proof of what must be
in every effect's recipe and gate.

| | **hres fire** (`fire_hres.nvfx3`) | **hres smoke** (`smoke_hres.nvfx3`, study N teacher) |
|---|---|---|
| **Outcome** | Shipped: passes validation, the one test, and the **edge check with calibrated bounds** ([STUDY_3D_HRES.md](STUDY_3D_HRES.md) §7; `results/3d/hres/fire/edgecheck_test.md`) | Shipped as a **labelled demo**: fails the edge check (heat builds at **x = 0** from wind ≥ 0.50; §8) |
| **Teacher** | Study N fire preset on the finer grid | Study N smoke preset (weak buoyancy; not R2's `smoke_r2`) |
| **Training** | N2's four stages on the **CUDA** trainer (spread/edge losses not in that build) | Same recipe, CUDA; **no `--spread` / `--edge`** |
| **Lesson for PLAN_PC_GPU** | Finer state + N2 recipe + **four-run edge calibration** (`edge_bounds.csv`) can pass where N2 only almost did | Finer state **without** R2's losses repeats N/R's **upwind wall** failure; **calibrated bounds** matter (single-run teacher at 0.50 looked like 0.01) |
| **In flight** | Done | **`smoke_r2` at hres** (CPU train with `--spread 2 --edge 10`, R2 teacher, same bounds recipe): answers whether the smoke gate is fixable **before** P3 spends GPU weeks |

**Carry into §6.3 and `nvfx_pc`:** ambient edges and long rollouts (N2); **spread and edge losses** on smoke-like effects;
**edge bounds from four teacher runs per wind** before any candidate is judged; stage 4 burn 48–360 frames; rule committed
before validation (study Q2's shape). **Carry into P1:** extend the CUDA trainer with spread/edge (today it **refuses**
`--device cuda` with them) so hres-scale experiments are not CPU-weeks on the PC. **Carry into P3:** fire and smoke must
beat N2 C4, hres demos, and R's C4 on §6.4 with **no 60 s runaway**—the hres smoke failure is exactly what P3 must not
ship.

### 1.2 Earlier studies (container)

| study | state, detail | what it found | what fixes it |
|---|---|---|---|
| N (fire) | 16 x 24 x 16, 64 x 96 x 64 | tracks a held-out fire better than the coarse simulation; heat builds at the upwind edge and grows a second fire | ambient edges, long rollouts (N2) |
| N2 (fire) | the same | no runaway over 30 s, but too calm (motion 0.68 of the real runs'); an edge bound 3% too tight | bigger model, longer windows with gradient, bounds from the teacher's spread |
| hres (fire) | 32 x 48 x 32, 64 x 96 x 64 | **passes** hres rule and test edge check; ~64 ms a step on four CPU threads (GPU required at 64³) | GPU for train and play (this plan) |
| R (smoke) | 16 x 24 x 16, 64 x 96 x 64 | a teacher that is a plume by MTT's measures; the learned smoke runs away at an edge and spreads too far (mean picture -2.52 dB) | spread/edge losses, R2 teacher (R2 + hres smoke_r2) |
| R2 (smoke) | the same | teacher momentum fixed; 16³ retrain under way on the container | hres smoke_r2 on the PC tests the same losses at 32³ |

### 1.3 Dynamism: the real range and spread of an effect

The goal is not to match one clip. It is an effect that **feels like the real phenomenon** because it has the same
**dynamism**: variability over seeds and controls, lateral spread, puffing, entrainment, edge behaviour, and the
long-tail statistics of play (60 s, many winds). That is a **distribution**, not a frame.

Real footage works against that:
- it is **hard to get** and rarely licensed for training;
- one take is **one sample** from a huge space—it **narrows** what the pipeline optimizes to (lighting, camera, luck);
- it hides the **interior state** you need to judge spread and runaway (study R's wide cone looked fine at 1 s in
  pictures but failed on spread and edges).

Sim teachers do the opposite: **~100 runs** per effect, held-out settings and seeds, bounds from **several** teacher
runs' spread (N2, R2), spread and edge losses where plumes lie, endless statistics in the test rule. The one
`reference.md` anchor (FDS, MTT, Basilisk, …) checks physics; **§6.4 and the edge check** check that the **learned**
player reproduces the teacher ensemble's range—not a video plate. FUSE is where that ensemble is **played** in the game
with the same seeded variability (principle 4).

The limits are the same each time:
- small models: 16 to 32 channels, 14,000 to 60,000 weights;
- little data: 40 to 100 runs at 64 x 96 x 64;
- short training windows: gradients through 12 to 24 frames, on a CPU.

High resolution makes each of them about 64 times worse on a CPU: the teacher has 64 times the cells, and the state
has 8 times the cells at twice the width. The CPU can no longer simulate, train or play it. Study hres's state is
already 64 ms a step on four threads; one at 64 x 96 x 64 and 64 channels would be about 30 times that. So everything
at high resolution moves to the GPU: the teachers, the training and the playing.

The time per effect then comes from four choices:
- batched GPU teachers;
- about 100 runs of 10 s per effect;
- a family model that each effect fine-tunes, coarse to fine;
- one unattended command.

The first version of this plan, on CPU teachers and per-effect training, took weeks per effect. Its second version
reached hours by dropping to a 32³ state and 128³ teachers. This one keeps the high resolution and gets the hours from
the GPU.

## 2. What I need from you

| # | question | why it matters | default if you don't say |
|---|---|---|---|
| 1 | GPU model and its memory (VRAM) | the hours in §7 assume a 24 to 32 GB card of the current generation; a smaller card means fewer runs at once and longer fine-tunes | an NVIDIA card with 24 to 32 GB |
| 2 | CPU cores, RAM, disk | ingest, referees and CPU checks run beside the GPU; storage is 2 to 3 TB in all (§7.2) | 16+ cores, 64 to 128 GB RAM, 4 TB NVMe |
| 3 | Operating system | LibTorch and the GPU stack run on Windows and Linux; ROCm and some research codes are Linux-first | Linux, or Windows with WSL2 for the research codes |
| 4 | FUSE engine path | Adapter links against owner FUSE on the Desktop (or stand-in demo here for CI) | **`Desktop/FUSE`** (git); NeuralVFX `engines/fuse/` |
| 5 | Is a GPU dependency acceptable: LibTorch and CUDA for training, and a GPU step in FUSE for playing? | the repository has been CPU-only; high resolution can't play on a CPU (§9) | yes: training optional and off by default; the GPU step a FUSE path beside the CPU one |
| 6 | What GPU time can one high-resolution effect take in a frame of your game? | sets the shipped stepper's width (§9) | 1 to 3 ms for a hero effect on a current card |
| 7 | Real footage as a reference? | study J compared pictures to video; sims give fields and full state, not one camera | **no** — sim, theory and literature only (§3.3.1) |

### 2.1 Recorded answers (this PC, 24 October 2026)

P0 uses this block; change a row here when hardware or licences change.

| # | answer | notes |
|---|---|---|
| 1 | **NVIDIA GeForce RTX 3090, 24 GB VRAM** (driver 610.88) | Within §7's 24–32 GB assumption; CUDA 12.0 (`nvcc` in WSL). `build-cuda/` and `NEURALFX_CUDA` already build `rollout3d_train` and hres smoke trained on GPU; spread/edge losses still CPU-only until P1. |
| 2 | **64 threads** (1× Xeon Gold 6242 @ 2.80 GHz), **48 GiB RAM**; **~5.4 TiB free on `D:`** (HDD, 24 October 2026); **~730 GiB free** on the 1 TiB WSL root (`/`, ext4) | Cores and RAM match the plan. **Storage:** §7.2 needs **~2.5 TB** for all 33 effects' clips—the **D:** volume clears that with room for raw output in flight (plan's 4 TB NVMe target is met on capacity; HDD is slower than NVMe for batched GPU sim ingest, acceptable for archive and large sequential writes). **Recommended `NEURALVFX_DATA` for PLAN_PC_GPU:** `D:\nvfx-data` in Windows, **`/mnt/d/nvfx-data`** in WSL (`mkdir` once; point `NEURALVFX_DATA` there before P1's big sim batches). **Today:** hres and studies still use `/home/odin/nvfx-data` (~1 TiB ext4); migrate or symlink when moving smoke_r2 checkpoints and starting P1, or keep hres on ext4 and put only 64³ clip trees on `D:`. Repo stays on OneDrive under `C:`; build in WSL. |
| 3 | **Windows host + WSL2** (`Linux … microsoft-standard-WSL2`, x86_64) | LibTorch/CUDA and `nvfx_3d` run in WSL today. Research-code referees stay Linux-first. FUSE budget timing on **Windows** (`Desktop/FUSE`) or the stand-in demo in WSL. |
| 4 | **No paid VFX simulators** (owner, 24 October 2026) | Teachers are **our GPU solvers only** (§3.2). **FUSE** on Desktop for play/budget ([FUSE.md](FUSE.md)). |
| 5 | **Yes** — GPU dependency accepted for stage 2 | Owner target is full PLAN_PC_GPU (§8). Training: optional CMake flag, off in CI. Playing: GPU step in FUSE for 64³ effects (§9); CPU runtime unchanged for today's levels of detail. |
| 6 | **1–3 ms per hero high-res effect per frame** (plan default) | Sets stepper width in §9 until measured in FUSE during P3. Revisit if the game's budget is tighter or looser. |
| 7 | **No real footage** (owner, 24 October 2026) | Footage is costly and **collapses the statistical range** of an effect to one sample; we want **dynamism**—spread, variability, and interior fields—from sim ensembles (§1.3). No study J, no `reference.md` video. One sim/theory anchor per effect plus §6.4 on held-out **sim** runs. |

**P0 gate with this table:** CPU build + ctest on the PC (done on the working tree: 602 tests, 0 failed, 32 skipped);
§3.5 licence register covers research codes only (no commercial sim outputs).

## 3. The teachers

### 3.1 What "best" means here

- **Teacher:** our GPU solver for that family (§3.2)—fast enough for hundreds of runs.
- **Reference (one per effect):** the single most scientific anchor for that catalogue row—correlations and
  intercomparison data where they exist, otherwise **one** matched case from a research code (FDS, Basilisk, PALM, …).
  Slow codes never make training data; they only score the teacher and the shipped model on the case named in
  `reference.md`. **Not real footage** (§1.3): one clip cannot represent the effect's range; teachers supply an
  ensemble and field diagnostics (MTT audit, edge bounds from spread, R2's spread loss).

No paid DCC sim. **FUSE** ([FUSE.md](FUSE.md), `Desktop/FUSE`) is where effects ship; it does not count as the
scientific reference unless the row is designed-only (consistency).

Our teachers are tuned for the catalogue, not validated as research codes. No teacher makes training data until §3.4
passes on the family setup. Weather stays outside the hours target, as in TODO.

### 3.2 Speed: GPU teachers, batched

At 256 x 384 x 256 (25 million cells) a teacher has to make a frame in a fraction of a second for 100 runs to take
an hour or two. The hours therefore need **our** GPU solvers; the plan builds three, each checked against §3.4 and the
family's default reference case before any effect in that family trains:

| solver | for | from | estimated frame at 256 x 384 x 256 | 100 runs of 300 frames |
|---|---|---|---|---|
| `nvfx_gpu_sim` | gases: fire, smoke, blasts, dense flows, designed | today's `sim3d` (fire, `smoke_r`, R2's momentum fix) in LibTorch tensors: advection by trilinear sampling, pressure by multigrid | 0.05 to 0.15 s | 0.5 to 1.5 h |
| `nvfx_gpu_flip` | liquids: waterfall, fountain, splash, geyser, rain shaft, sea spray, lava | new: FLIP particles on the GPU, a level set for the surface, multigrid pressure; temperature-dependent viscosity for lava | 0.1 to 0.3 s | 1 to 2.5 h |
| `nvfx_gpu_mpm` | snow, soil, grains: avalanche, and the grains in dust, eruption and haboob | new: moving-least-squares MPM on the GPU, with snow and granular (Drucker-Prager) materials | 0.1 to 0.3 s | 1 to 2.5 h |

- Each is batched: as many runs at once as the card's memory holds (about 1 to 2 GB a run at this size).
- Each is headless, deterministic, licence-free, and built only with `-DNEURALFX_GPU=ON`.
- Each has a parity test against its CPU reference on a small case. `nvfx_gpu_sim`'s reference is `sim3d`; the new
  ones each get a small CPU version for this.
- The frame times above are estimates; P1, P5 and P6 measure them.

**`smoke_r` already passes the MTT plume checks.** The fire does not yet: it puffs at 0.5 to 0.65 Hz where a fire its
size puffs near 2.7 Hz (STUDY_3D.md §13). Its check at high resolution is part of P2, and a finer grid is the first
thing to try.

The other choices that keep the data in hours:
- **10 s runs** (300 frames at 30 fps) after the warm-up. Endless play is still tested over 60 s, and the model's own
  long rollouts are what training stage 3 uses (§6.3).
- **About 100 runs an effect** (training 60, validation 20, the test 20). The null probe sizes validation and the test
  as study M3 does. If it says 20 settings lack power, the pipeline runs more that night and says so.

### 3.3 Per family

Today's 33 2D effects, by family:
- **gas (25):** candle, cloud, coals, crown fire, dust, dust explosion, eruption, explosion, fire, fog, haboob, heavy
  gas, ink, jet, launch, magic, mushroom, orb, puffs, smoke, smoke ring, steam, vortex, wildfire, window fire;
- **free surface (7):** fountain, geyser, lava, rain shaft, sea spray, splash, waterfall;
- **granular (1, and the grains in 3 more):** avalanche, plus the grains in dust, eruption and haboob.

| family | effects | teacher | default reference (one per effect; §3.3.1) |
|---|---|---|---|
| gas: fire and combustion | fire, candle, wildfire, crown fire, window fire, coals, dust explosion | `nvfx_gpu_sim` with combustion (§3.2) | **FDS** (NIST; pool/jet fire case matched to the effect's scale) |
| gas: smoke, plumes, puffs | smoke, smoke ring, puffs, steam, jet, launch, ink | `nvfx_gpu_sim` (`smoke_r`) | **MTT plume theory** (§3.4; study R's audit). Jet/launch: same unless `reference.md` names one FDS plume case |
| gas: blasts | explosion, mushroom, eruption (plume) | `nvfx_gpu_sim` | **Mass-scaling and scaled-distance correlations** (§3.4). Eruption plume column: MTT |
| gas: dense and laden | heavy gas, haboob, dust | `nvfx_gpu_sim` with density | **FDS** heavy-gas release, or gravity-current theory where no case is run |
| gas: designed | magic, orb, vortex | `nvfx_gpu_sim` with designed forces | **Temporal consistency** (no external sim) |
| gas: weather | cloud, fog | PALM or MicroHH (outside the hours target) | **BOMEX or DYCOMS-II** intercomparison statistics |
| free surface | waterfall, fountain, splash, geyser, rain shaft, sea spray | `nvfx_gpu_flip` | **Basilisk** (one matched case per effect: sheet, jet, or drop impact) |
| viscous and hot | lava | `nvfx_gpu_flip` with temperature-dependent viscosity | **Published lava flow-rate measurements** |
| granular | avalanche; the grains in dust, eruption and haboob | `nvfx_gpu_mpm` | **Chute and avalanche runout measurements** (§3.4) |

### 3.3.1 One reference per effect

Before validation, `results/3d/<effect>/reference.md` commits **one** line: what the anchor is and which case (for
example `FDS McCaffrey plume B`, `Basilisk drop Re=2000`, `MTT buoyant plume α=0.14`). The pipeline compares teacher
and shipped model to that anchor only—not FDS plus Basilisk plus keyframes plus a FUSE capture for the same gate.

**Pick order:** (1) published correlations or intercomparison statistics already in §3.4; (2) otherwise one research-code
run at matched nondimensional numbers; (3) designed rows: consistency only. Override the family default only when the
effect's mechanism requires it, and say why in `reference.md`.

`nvfx_pc effect <name>` creates or checks `reference.md` from the table above if the file is missing.

Spray, foam, droplets, sparks, embers and grains stay seeded random (principle 1): the teachers' particles give their
statistics, not training targets frame by frame.

**Not used in any teacher:** another product's ML upres, or any learned node. The project trains no model on another
model's output, and uses no pretrained networks.

### 3.4 The checks every teacher passes first (once per family)

Each check is written in `results/3d/<family>/validation.md` before the runs, as TODO's pipeline asks. The setup is
run at 128 x 192 x 128, 256 x 384 x 256 and 512 x 768 x 512 (the last for a few runs, on the GPU or overnight). It must
converge by 256, then pass within the measurements' scatter:
- **fires:** flame height against Heskestad's correlation; centreline temperature and velocity against McCaffrey's
  profiles; puffing frequency near 1.5/√D Hz (D in metres);
- **plumes and smoke:** Morton-Taylor-Turner: buoyancy flux conserved, entrainment α 0.10 to 0.16, centreline
  exponents near -1/3 and -5/3, and momentum near its buoyancy (study R's audit, R2's fix);
- **blasts:** fireball diameter and duration against mass-scaling correlations, and the shock's arrival against
  scaled-distance curves where the early shock shows;
- **dense flows (heavy gas, haboob):** front speed against the gravity current's Froude number;
- **liquids:** ballistic jet trajectories, breakup against Rayleigh-Plateau, splash crowns against drop-impact
  measurements, falling sheets against free fall;
- **granular:** angle of repose, runout against chute and avalanche measurements, front speed;
- **designed effects:** none (say so); consistency only.

**FUSE (ship check, not a reference).** P3 measures GPU step time and manager budget in the owner's FUSE build
(`Desktop/FUSE` with this repo's adapter). That is separate from the one scientific reference in `reference.md`.

Bounds that later tests use (the edge check, endless statistics) come from several teacher runs' spread, never one
(study N2's lesson). Each effect's setup is a variation of its family's checked setup, within physical parameters
(fuel, size, wind, viscosity), so it needs no new check of its own. An effect that needs a new mechanism gets a check
first.

### 3.5 Licences

Confirm before the first production run, and record in the licence register (`docs/DATA.md`; extend `licences.tsv` to
simulator outputs, with `--licence commercial --licence-note` naming the agreement):
- **Paid DCC simulators:** not used; no SideFX or similar outputs in training data.
- **Research codes:** FDS is public domain; PeleLMeX is open source; Basilisk, PALM and MicroHH are GPL. Their outputs
  are ours to use. GPL applies to their code, which stays outside this repository; we keep only case files and
  readers.
- **Our own solvers:** ours.

## 4. The PC

Measure it once (TODO's step 0) into `results/3d/pc.md`: CPU, cores, RAM, GPU, VRAM, driver, CUDA or ROCm version,
disks and their speeds, and the operating system.

The GPU does the simulating, the training and the measuring, so it is the bottleneck. The pipeline keeps it busy all
the time. While an effect trains, the CPU ingests the next effect's runs, runs the referees and makes the videos.

## 5. Software on the PC

- **This repository, as today:** the CPU build, the tests, the CPU runtime and the tools. CI does not change.
- **`nvfx_pc`, new: the pipeline in one command.**
  - `nvfx_pc effect <name>` runs the whole chain unattended: simulate, ingest, fine-tune, null probe, commit the rule's
    numbers, validate, freeze, test once, ship if it passes, videos.
  - `nvfx_pc family <name>` runs the once-per-family steps: the teacher's checks, the look match and the family
    model's training.
  - Each step writes a marker, so a stopped chain resumes where it was.
  - It logs every step's time to `results/3d/<effect>/time.csv`, so the hours are measured, not hoped.
- **Built only with `-DNEURALFX_GPU=ON`** (off by default, not in CI), on LibTorch's C++ API (BSD licence) on CUDA or
  ROCm. No Python, and no pretrained weights: every model starts from our own initialisation.
  - **`nvfx_gpu_train`.** It uses LibTorch for 3D convolutions, 16-bit mixed precision, activation checkpointing,
    CUDA graphs and the optimiser. It reads the repository's 3D clip format and writes `.nvfx3`.
  - **`nvfx_gpu_sim`, `nvfx_gpu_flip`, `nvfx_gpu_mpm`:** the GPU teachers (§3.2).
  - **`nvfx_gpu_play`:** the GPU step of a `.nvfx3`, the same arithmetic FUSE's will use (§9), for measuring and
    videos.
  - **A differentiable volume renderer** that matches `render3d` (Planck's law, CIE 1931, soot absorption, single
    scattering), for picture losses from several cameras and for measuring.
- **Parity tests, run on the PC:**
  - the GPU step and the CPU runtime agree within float tolerance on the same weights and inputs;
  - the trainer's gradients agree with our hand-written ones on small cases (the finite-difference checks);
  - each GPU solver agrees with its CPU reference on a small case;
  - the GPU renderer agrees with `render3d`.
- **Teacher drivers, as text case files** under `sim3d/<solver>/<row>/` for our solvers; research codes take their own
  input files. GPU teachers write the clip format directly (§5, `nvfx_ingest3d` for referee output only).
- **`nvfx_ingest3d`, new:** reads OpenVDB (open-source library), AMReX plotfiles, FDS output and NetCDF into the clip
  format as each run is written. Our GPU solvers write the clip format directly. It keeps the 64 x 96 x 64 coarse
  grid and 256 x 384 x 256 crops at 16 bits, then deletes the raw output.

## 6. The models

### 6.1 One model per family, for all its effects

Each family has one network. Its input carries an **effect embedding** (one learned vector per effect) and the
effect's physical controls (wind, fuel rate, size, temperature, viscosity...) through FiLM, as stage 1's controls do.
The structure is stage 1's (STUDY_3D.md §4), at high resolution:
- a learned coarse state with memory channels;
- a stepper (3 x 3 x 3 convolutions, U-Net style) with advection and pressure projection;
- seeded 3D detail;
- a learned renderer.

The state per family:
- gas: velocity, temperature, the group's scalars (soot, fuel, vapour, dust) and memory;
- free surface: velocity, liquid fraction or level set, and memory;
- granular: density, velocity and memory.

Why one model per family: effects of a kind share their physics (buoyancy, entrainment, vortices, fronts). A model that
already knows them needs far less of each new effect's data and training, which is what lets 100 runs and 3 to 5 hours
of fine-tuning be enough. Whether it is enough is measured, effect by effect (§6.4).

### 6.2 Sizes

| | high resolution (GPU, ships) | the CPU version (lower levels of detail) |
|---|---|---|
| coarse state | 64 x 96 x 64 (64³ for effects that aren't tall) | 16 x 24 x 16, study N2's size (9 ms a step on four threads); 32 x 48 x 32 (study hres's, 64 ms) only where the CPU budget allows |
| stepper width | 32 to 64 channels, chosen by the GPU budget (§9) | 16 to 32 |
| memory channels | 8 to 16 | 4 to 8 |
| detail | 256 x 384 x 256, sparse bricks, seeded | 64 x 96 x 64, as today |
| plays on | the engine's GPU | the CPU runtime, as today |
| made by | the effect's fine-tune, coarse to fine | a short fine-tune of its own, from the same data |

Coarse to fine: each fine-tune runs first at 32 x 48 x 32, which is 8 times cheaper a window at the same width, then at
64 x 96 x 64, starting from it. The CPU version is a short fine-tune at its own size, which costs little at 16 x 24 x
16.

### 6.3 The training recipe

From studies D, D2, L, N2 and hres, staged, with a snapshot after every stage. The family model runs all four stages
over its first effects. Each effect's fine-tune runs stages 2 to 4 from the family model, first at 32 x 48 x 32 and
then at 64 x 96 x 64:
1. one-step targets from the teacher's states (teacher forcing);
2. unrolled windows of 16, then 32, then 64 frames, with gradients through the whole window (activation
   checkpointing per step keeps memory flat; §7.3);
3. long free-running rollouts of hundreds of frames (no gradient through the burn-in), at the controls' extremes and
   with moving controls, against the endless statistics (study D2's loss);
4. picture losses through the differentiable renderer from four or more cameras.

Throughout, ambient air comes in at open edges in the model's own advection (N2's fix), and `nvfx_3d edge` checks
every candidate. Candidates are chosen on endless play as well as on tracking (STUDY_3D.md §13, point 1).

### 6.4 The questions each effect answers, by a committed rule

1. Does the high-resolution model beat:
   - (a) the coarse simulation at 64 x 96 x 64 with the same seeded detail;
   - (b) its own CPU version;
   - (c) today's best 3D model of that effect (N2's C4, hres, R's C4)?

   It must win on tracking at 1 s and 2 s and on every endless statistic, with no edge runaway over 60 s.
2. Does the CPU version beat the coarse simulation at its own size and today's model on the same measures? It ships
   as the lower level of detail only if it does.
3. Against reality: the model's own runs pass the teacher's physical checks (§3.4) as the teacher did.
4. Once per family, on its first three effects: does the family model's fine-tune beat a model trained on that effect
   alone, with the same data and the same hours? If not, per-effect training is used for that family, and the hours
   are measured again.

## 7. An effect in hours

### 7.1 The budget

Times are for one effect on a 24 to 32 GB card of the current generation and a 16-core CPU. The pipeline logs every
step's real time to `results/3d/<effect>/time.csv`, and the pilots replace these numbers.

| step | where | estimate | how |
|---|---|---|---|
| simulate 100 runs at 256 x 384 x 256 | GPU | 0.5 to 2 h | our GPU solver, batched (§3.2) |
| ingest | CPU | overlapped | as each run is written |
| fine-tune at 32 x 48 x 32 (the warm start) | GPU | 0.3 to 1 h | 10,000 windows at 0.1 to 0.4 s, by width |
| fine-tune at 64 x 96 x 64 (the high-resolution version) | GPU | 2.5 to 4 h | 6,000 windows of 32 frames and 1,500 of 64, about 1 to 1.7 s and 2 to 3.4 s each (§7.3) |
| fine-tune the CPU version at 16 x 24 x 16 | GPU | under 0.5 h | small: study N2's size |
| null probe, validation, freeze | GPU and CPU | about 0.5 h | `nvfx_gpu_play` for the high-resolution version, the CPU runtime on 16 cores for the CPU version |
| the one test | GPU and CPU | under 0.5 h | 20 held-out settings, fresh seeds |
| ship: videos, library entry, `models/3d` | CPU and GPU | under 0.5 h | |
| **total** | | **5 to 9 h** | one effect a night |

Once per family:

| step | estimate |
|---|---|
| build and check the family's GPU solver (gases: port `sim3d`; liquids and grains: new) | engineering, before the PC's hours: about a week each for FLIP and MPM, less for the gas port |
| the teacher's checks (§3.4) at three resolutions and the family's default reference case (once) | 1 to 2 days |
| the family model: its first three effects' data, stages 1 to 4, mostly at 32 x 48 x 32 and finished at 64 x 96 x 64 | 2 to 3 days of GPU |

### 7.2 Data and storage

- **Kept a run:**
  - the 64 x 96 x 64 coarse grid of 6 fields at 16 bits is 4.7 MB a frame, about 1.6 MB with sparse bricks, so
    0.5 GB for 300 frames;
  - 256 x 384 x 256 crops of two fields every 10th frame, a quarter of the volume, sparse, add about 0.25 GB;
  - so **about 0.75 GB a run**.
- **An effect:** 100 runs, about 75 GB.
- **All 33 effects:** about 2.5 TB, so a 4 TB NVMe holds everything with room for raw output in flight. Raw output is
  deleted once ingested.

### 7.3 GPU training time

- **At 64 x 96 x 64 with 64 channels:** a 3 x 3 x 3 layer is 64 x 64 x 27 x 393,216 multiply-adds, about 87 GFLOP.
  Ten layers are 0.87 TFLOP forward and about 2.6 TFLOP with the backward pass. A 32-frame window is about 84 TFLOP:
  **1 to 1.7 s** on a 24 to 32 GB card at a realistic 50 to 80 TFLOP/s in 16-bit, and a 64-frame window twice that.
- **At 32 x 48 x 32:** 8 times less a step at the same width. A 64-frame window is about 0.1 s at 32 channels and
  0.25 to 0.4 s at 64, so the warm start is short.
- **Memory:** one layer's activations at 64 x 96 x 64 are 64 x 393,216 x 2 bytes, 50 MB. With checkpointing per step,
  64-frame windows in batches of 2 to 4 fit in 24 GB.
- **Width:** at 32 channels instead of 64 everything is 4 times cheaper. The width is fixed by the GPU budget in the
  game (§9), so the hours may well be less.

### 7.4 All of today's effects

The GPU is shared by simulating and training, so effects go one after another, one a night:

| family | its effects | time on the PC |
|---|---|---|
| gas | 23 with a GPU teacher (cloud and fog wait for the weather, P7) | about 3 to 4 days once, then 23 nights: about 4 weeks |
| free surface | 7 | the GPU FLIP built first; then about 3 days once and 7 nights: about 1.5 weeks |
| granular | the avalanche, and the grains of dust, eruption and haboob | the GPU MPM built first; then about 3 days once and 4 nights: about 1 week |

That is **4 to 6 weeks** of the PC in all, with liquids and grains overlapping the gases once their solvers exist. It
is unattended: the owner looks at each morning's result. Weather and the rest of TODO's catalogue follow, row by row:
in hours where a GPU teacher exists, slower where only a research code does.

## 8. Order of work, with gates

Each phase ends with a gate. A phase that fails its gate stops, is written up with its numbers, and the next step is
decided then. Every effect's time is a gate too: an effect that takes more than 12 hours stops the roll-out until the
slow step is found and fixed.

| phase | what | how long | gate (the rule committed before validation) |
|---|---|---|---|
| **P0** | Set up the PC, measure it, confirm the licences (§3.5) | a day | the CPU build and ctest pass on the PC; licences recorded |
| **P1** | `nvfx_gpu_train`, `nvfx_gpu_sim`, `nvfx_gpu_play` and their parity tests; retrain N2's fire, hres's fire and R's smoke on the GPU from their own data | 2 to 3 days | parity within tolerance; the GPU-trained models tie or beat the CPU-trained ones on their own held-out tests; `nvfx_gpu_sim` makes a 256 x 384 x 256 frame in 0.15 s or less |
| **P2** | The gas teacher at three resolutions: `nvfx_gpu_sim`'s fire and smoke; family references (FDS for fire, MTT for plumes) on one case each | 1 to 2 days | §3.4 passes by 256; default `reference.md` templates for fire and smoke rows; teacher settings frozen |
| **P3** | The gas family model on fire, smoke and one more; then fire and smoke through `nvfx_pc effect`; their GPU step put into FUSE and timed | 3 to 4 days | §6.4 for fire and smoke: better than the coarse simulation, their CPU versions, N2's C4, hres and R's C4; no runaway over 60 s; playing on as well as tracking; each effect's chain in 9 hours or less; the step within the GPU budget in FUSE. This is where N2 and R failed. |
| **P4** | The other 20 gas effects, one a night (cloud and fog wait for P7) | about 3 weeks | per effect, §6.4; ship only what passes |
| **P5** | `nvfx_gpu_flip`, one Basilisk case per liquid family; then the 7 liquids (each with `reference.md`) | a week to build and check, then 1.5 weeks | as P4 |
| **P6** | `nvfx_gpu_mpm`, runout measurements as reference; avalanche and grain rows | a week to build and check, then a week | as P4 |
| **P7** | The rest of TODO's catalogue and the weather system (TODO's waves 1 to 3) | row by row | as P4, per row |

P5 and P6 build their solvers while P4 runs on the PC.

A passing effect ships as TODO's step 6 describes:
- its high-resolution `.nvfx3` and its CPU version in `models/3d/`, with `SHA256SUMS`;
- frozen hashes: pinned to float for the CPU version; for the GPU step, as §9 says;
- demo videos;
- library entries.

Today's 2D effect stays as the lightest option. An effect that fails is written up with its numbers and kept off the
default; the next step for it (more runs, a fixed teacher, a wider stepper) is decided then.

## 9. Playing them: the GPU runtime

At high resolution the CPU can't play an effect. Study hres's 32 x 48 x 32 state already takes 64 ms a step on four
threads, and the 256 x 384 x 256 detail is 64 times today's. So TODO's principle 5 ("the engine's GPU draws; the
library's CPU steps") changes for these effects only: **the engine's GPU steps and draws them.**

- **In FUSE:** the stepper, the advection, the pressure projection and the seeded detail run as compute shaders on the
  effect's sparse bricks, and the fields stay on the GPU for FUSE's volumetric renderer. Nothing is copied back to
  the CPU each frame.
- **The budget sets the width.** Dense at 64 x 96 x 64 and 64 channels, a step is 0.87 TFLOP, too much for a frame.
  - Sparse bricks (30 to 40% filled) cut it to a third.
  - 8-bit arithmetic doubles the rate.
  - 32 channels instead of 64 cut it to a quarter.

  So a hero effect is about **1 to 3 ms** on a current card at 32 to 48 channels. P3 measures it in FUSE, and the
  owner's budget (§2, question 6) fixes the width before the family model is trained.
- **Determinism.** The same seed must give the same frames on every player's machine (principle 4). On the GPU that
  needs integer arithmetic (8-bit weights, 32-bit integer sums, fixed-point fields) throughout the step. Its frozen
  hashes are then the same on every GPU. Float on the GPU differs between vendors and drivers. Where frames may differ
  between players (effects that don't touch gameplay), float is allowed and labelled so.
- **Levels of detail:**
  - near: the high-resolution effect on the GPU;
  - middle: its CPU version (16 x 24 x 16, today's runtime);
  - far: today's 2D effect.

  The effect manager (budget, sharing, proxies) extends to choosing among them.
- **The CPU runtime is unchanged:** the volume API (`nvfx_volume.h`), sparse bricks, int8, threads and shards, no
  allocation per frame. A CPU reference of the GPU step exists for tests, on a few frames.

GPU training need not be bit-reproducible: every shipped file is frozen by its SHA-256, and its training log is kept.

## 10. Risks

| risk | what happens then |
|---|---|
| Our GPU solvers fail the physical checks (the fire's puffing is the known one) | fix the solver first, as R2 fixes R's momentum; delay that sub-family's `nvfx_pc` roll-out until checks pass |
| FUSE step over budget but §6.4 passes | narrow stepper width (§9) or ship CPU LOD until GPU path fits |
| An effect has no published check | `reference.md` names **consistency** or the nearest sim case; extend §3.4 before inventing a video check |
| The GPU FLIP or MPM takes longer to build than a week | the gases go on meanwhile; the liquids and grains move later |
| 256 is not converged for an effect | that effect's teacher moves up (320 or 384 across) and its hours are measured again |
| 100 runs lack power for the test | the null probe says so before validation; the pipeline adds runs, and the extra hours are logged |
| A fine-tune fails at high resolution | its CPU version may still pass and ship as the middle level; the high-resolution version is written up and retried with more runs or a wider stepper |
| The GPU budget allows only 32 channels or less | the family model is trained at that width; the hours fall; whether the looks hold is what the test measures |
| Integer arithmetic on the GPU costs too much accuracy | float on the GPU, labelled as not bit-identical between players |
| The family model doesn't beat per-effect training (§6.4, question 4) | per-effect training for that family; the hours are measured again |
| AMD GPU | LibTorch on ROCm, Linux only; FUSE's compute shaders are vendor-neutral; parity tests as for CUDA |
| Learned models copy the teacher's flaws | the referee and the §3.4 checks on the model's own runs catch them; the teacher is fixed first |

## 11. What goes in git, and seeds

**In git:**
- the GPU trainer's, the GPU solvers', the GPU player's and the pipeline's code;
- the teachers' case files (text);
- each family's rule template;
- each effect's committed rule numbers, results tables and `time.csv`;
- figures;
- the shipped models (`models/3d/`: the CPU versions under 1 MB each; the high-resolution versions about 1 to 5 MB at
  32 to 64 channels);
- demo videos (each 5 MB at most).

**Out of git:** simulator output, ingested clips, training runs and checkpoints, unless the owner decides otherwise.
The family models, a few MB each, may go in `models/3d/families/` if the owner wants them kept.

**Seeds:** TODO's stage 2 ranges. Catalogue row # owns 10,000,000 + 100,000 x #. Each effect's training, validation and
test seeds come from its row, and the tests use held-out settings and fresh seeds from the same row. The family
models' seeds come from their first effect's row.

**Queue (24 October 2026).** The owner has said go on **this plan for all effects** (not hres alone). Order: (1) finish
**hres smoke_r2** and write up pass/fail in [STUDY_3D_HRES.md](STUDY_3D_HRES.md) §10; (2) **P0** once §2's PC details
and licences are recorded; (3) **P1–P3** until fire and smoke pass §6.4 at 64 x 96 x 64 in ≤9 h each; (4) **P4–P7**
row by row in [WAVE1_CHECKLIST.md](WAVE1_CHECKLIST.md). P0 is not blocked on smoke_r2 finishing, but **P3 must not
repeat smoke_hres's mistake** (no spread/edge, wrong teacher). Wave 1 ship (`COMMIT_PLAN.md`) is separate: it lands the
hres scripts and checklist; it does not replace building `nvfx_pc`.
