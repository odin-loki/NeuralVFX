# Plan: every effect in 3D, trained on the PC's GPU from the best simulators

**Status:** a plan, written 21 October 2026. Nothing in it has started. It refines stage 2 of [TODO.md](../TODO.md)
(the catalogue, the pipeline, the measuring rules and the seeds stay as they are there) in three ways:

1. **The GPU trains.** Training moves from this container's CPU trainer to the PC's GPU, so the models can be far
   bigger, train on far more data and learn from long rollouts with gradients through them. The shipped effects still
   step on the CPU, as today, unless the owner later chooses a GPU path (§9).
2. **The teachers are the best simulators there are, checked against measurements.** For most effects that is
   Houdini, the film industry's standard, with research codes that are validated against experiments as referees and
   as teachers where Houdini fails its checks (§3).
3. **One model for all the effects of a kind.** One gas model for the 25 gas effects, one free-surface model for the 7
   liquids, and one granular model. Each is conditioned on the effect and its controls, then distilled into a small,
   cheap model per effect for the CPU runtime (§6).

Audience: the owner, and whoever runs it on the PC.

## Contents

0. [In short](#0-in-short)
1. [Why change the plan](#1-why-change-the-plan)
2. [What I need from you](#2-what-i-need-from-you)
3. [The teachers](#3-the-teachers)
4. [The PC](#4-the-pc)
5. [Software on the PC](#5-software-on-the-pc)
6. [The models](#6-the-models)
7. [Time, data and storage](#7-time-data-and-storage)
8. [Order of work, with gates](#8-order-of-work-with-gates)
9. [Runtime and engines](#9-runtime-and-engines)
10. [Risks](#10-risks)
11. [What goes in git, and seeds](#11-what-goes-in-git-and-seeds)

## 0. In short

- **Teacher:** Houdini 21 (Pyro for gases, FLIP for liquids, MPM for snow, mud and grains) for the effects games use.
  Validated research codes (FDS for fire and smoke, Basilisk for splashes and jets, PeleLMeX for flames and blasts,
  PALM or MicroHH for clouds and fog) check it on a few cases per family, and teach where it fails.
- **Before any training data,** every teacher setup must pass its family's physical checks (flame height, puffing
  frequency, plume spread, front speed, angle of repose and so on), within the scatter of the measurements.
- **Model:** one network per family, conditioned on effect and controls. It keeps stage 1's design: a learned coarse
  state with memory, a stepper with advection and projection, seeded detail and a learned renderer. It is trained on
  the GPU at 64³ and 64 to 128 channels, then distilled into one model per effect at 32³ and at most 64 channels for
  the CPU runtime.
- **Training on the GPU** uses LibTorch, PyTorch's C++ library: no Python, and no pretrained networks. It sits in a
  separate build option, off by default, so this repository's CPU build and CI do not change. It exports the
  repository's own `.nvfx3` files, which play in today's runtime.
- **Each phase has a gate,** run under the house rules: the rule written before validation, held-out settings and
  seeds, one test, paired bootstrap intervals. The first gate is a 3D fire and smoke that pass where studies N2 and R
  failed: no edge runaway, and they play on as well as they track.
- **Cost, roughly, until the pilots measure it:**
  - about 1 GB of kept data a run and 150 to 300 GB an effect;
  - about a second of GPU time per training window, so 1 to 3 weeks of GPU per family model and about a day per
    effect to distil;
  - simulation time depends most on Houdini's GPU pyro speed on your card, which the first pilot measures (§7).

## 1. Why change the plan

What this container's 3D studies found:

| study | what it found | what fixes it |
|---|---|---|
| N (fire) | tracks a held-out fire better than the coarse simulation; heat builds at the upwind edge and grows a second fire | ambient edges, long rollouts (N2) |
| N2 (fire) | no runaway over 30 s, but too calm (motion 0.68 of the real runs'); an edge bound 3% too tight | bigger model, longer windows with gradient, bounds from the teacher's spread |
| hres (fire) | 32 x 48 x 32 state, finer flames; not yet tested | the same, at scale |
| R (smoke) | a teacher that is a plume by MTT's measures; the learned smoke runs away at an edge and spreads too far (mean picture -2.52 dB) | more data, longer windows with gradient, more capacity |
| R2 (smoke) | under way: the teacher's momentum fixed (confinement made momentum-neutral) | — |

The limits are the same each time: a small model (16 to 32 channels, 14,000 to 60,000 weights) trained on 40 to 100
runs, with gradients through windows of 12 to 24 frames on a CPU. A GPU removes all three limits at once. A model 100
times bigger trains in hours. Windows of 64 to 128 frames with gradient fit in its memory (§7). Data comes from as
many runs as the simulator can make. And the teacher's physics is the ceiling (STUDY_3D.md §13 point 5), so the
teacher has to be the best available.

## 2. What I need from you

| # | question | why it matters | default if you don't say |
|---|---|---|---|
| 1 | GPU model and its memory (VRAM) | decides window length, batch and model size; NVIDIA uses CUDA, AMD uses ROCm (Linux only) | an NVIDIA card with 24 to 32 GB |
| 2 | CPU cores, RAM, disk (NVMe and archive) | CPU simulators run beside GPU training; disk is the binding limit (§7) | 16+ cores, 128 GB RAM, 4 TB NVMe plus an archive disk |
| 3 | Operating system | Houdini and LibTorch run on Windows and Linux; ROCm and some research codes are Linux-first | Linux, or Windows with WSL2 for the research codes |
| 4 | Houdini licence: Indie, Core/FX, or none yet | Indie allows commercial use under US$100K revenue and US$1M funding, and its files can't share a pipeline with commercial Houdini; Apprentice and Education can't be used | Houdini Indie |
| 5 | Is a training-only GPU dependency (LibTorch, CUDA) acceptable in an optional build target? | the repository has been CPU-only; the runtime stays CPU-only | yes, optional and off by default |
| 6 | Teacher choice: Houdini first, with research codes as referees, or research codes first | Houdini gives the look games want and covers every effect in one tool; research codes are closer to measured physics but slower and narrower | Houdini first, as §3 explains |
| 7 | Do you have real footage of any effect? | the only check on looks where nothing can be measured (study J) | none, and say so per effect |

## 3. The teachers

### 3.1 What "best" means here

There are two kinds of best simulator, and the plan uses both:
- **Best for the look:** the production simulators film VFX uses. Houdini is the standard: one tool for gases (Pyro,
  with a GPU sparse solver in Houdini 21), liquids (FLIP), snow, mud and soil (its MPM solver, production-ready in
  Houdini 21), particles and cloth. EmberGen (JangaFX) is a real-time GPU gas simulator, very fast, but gases only, and
  without a documented batch or command-line mode it is hard to automate. It is a fallback for bulk gas data.
- **Best for the physics:** research codes validated against experiments. FDS (NIST) for fire and smoke, PeleLMeX for
  flames and blasts (C++, runs on GPUs through AMReX), Basilisk for splashes, jets and drops, PALM or MicroHH for clouds,
  fog and boundary layers, and CM1 for storms. These are the simulators already in TODO's stage 2.

Production simulators are tuned by artists, not validated. So Houdini teaches only after its setup passes the same
checks a research code would (§3.3), and a research code referees each family on a few cases. Where Houdini cannot
pass, the research code teaches that effect. Weather (clouds, storms, regional systems) stays with the research
codes, as TODO plans: Houdini has no cloud microphysics.

### 3.2 Per family

Today's 33 2D effects, by family:
- **gas (25):** candle, cloud, coals, crown fire, dust, dust explosion, eruption, explosion, fire, fog, haboob, heavy
  gas, ink, jet, launch, magic, mushroom, orb, puffs, smoke, smoke ring, steam, vortex, wildfire, window fire;
- **free surface (7):** fountain, geyser, lava, rain shaft, sea spray, splash, waterfall;
- **granular (1, and the grains in 3 more):** avalanche, plus the grains in dust, eruption and haboob.

| family | effects | teacher | referee (a few cases) | GPU in the simulator |
|---|---|---|---|---|
| gas: fire and combustion | fire, candle, wildfire, crown fire, window fire, coals, dust explosion | Houdini Pyro (combustion), sparse GPU solver | FDS (fire), PeleLMeX (jet and small flames) | Houdini 21 sparse pyro; PeleLMeX |
| gas: smoke, plumes, puffs | smoke, smoke ring, puffs, steam, jet, launch, ink | Houdini Pyro | FDS; our smoke_r (the MTT plume) | Houdini |
| gas: blasts | explosion, mushroom, eruption (plume) | Houdini Pyro | PeleC (compressible, early shock) | Houdini; PeleC |
| gas: dense and laden | heavy gas, haboob, dust | Houdini Pyro with density | FDS; gravity-current theory | Houdini |
| gas: designed | magic, orb, vortex | Houdini Pyro with designed forces | none: consistency only | Houdini |
| gas: weather | cloud, fog | PALM or MicroHH | measured cloud statistics | MicroHH (CUDA, to confirm) |
| free surface | waterfall, fountain, splash, geyser, rain shaft, sea spray | Houdini FLIP | Basilisk | partly (OpenCL pressure) |
| viscous and hot | lava | Houdini FLIP with temperature-dependent viscosity, or MPM | measured lava flow rates | partly |
| granular | avalanche; the grains in dust, eruption and haboob | Houdini MPM (snow, soil) | angle of repose, runout, front speed | to confirm |

Spray, foam, droplets, sparks, embers and grains stay seeded random (principle 1): the teachers' particles give their
statistics, not training targets frame by frame.

**Not used in any teacher:** Houdini's ML-based smoke upres, or any learned node. The project trains no model on
another model's output, and uses no pretrained networks.

### 3.3 The checks every teacher setup passes first

Each check is written in `results/3d/<family>/validation.md` before the runs, as TODO's pipeline asks. The setup is
run at three resolutions, each twice as fine as the last. It must converge, then pass within the measurements' scatter:
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

Bounds that later tests use (the edge check, endless statistics) come from several teacher runs' spread, never one
(study N2's lesson).

### 3.4 Licences

Confirm before the first production run, and record in the licence register (`docs/DATA.md`; extend `licences.tsv` to
simulator outputs, with `--licence commercial --licence-note` naming the agreement):
- **SideFX (Houdini):** the licence agreement's web page (revision 13 October 2021) says nothing on machine learning or
  on who owns outputs. Indie is for entities under US$100K revenue and US$1M funding, and Indie files can't share a
  pipeline with commercial Houdini. Apprentice is non-commercial. Ask SideFX in writing whether training a model on
  Indie outputs is allowed, and keep the answer.
- **JangaFX (EmberGen):** the licence agreement (19 July 2025) says nothing on machine learning or outputs. Indie is
  for revenue or budget under US$1M. Trial and Educational exports can't be used commercially. Ask in writing too.
- **Research codes:** FDS is public domain; PeleLMeX is open source; Basilisk, PALM and MicroHH are GPL. Their outputs
  are ours to use. GPL applies to their code, which stays outside this repository; we keep only case files and
  readers.

## 4. The PC

Measure it once (TODO's step 0) into `results/3d/pc.md`: CPU, cores, RAM, GPU, VRAM, driver, CUDA or ROCm version,
disks and their speeds, and the operating system. The GPU then trains while the CPU runs CPU-only referees (FDS,
Basilisk) and ingests. Houdini's GPU pyro and the trainer share the GPU, so they are scheduled one after the other.

## 5. Software on the PC

- **This repository, as today:** the CPU build, the tests, the runtime and the tools. CI does not change.
- **`nvfx_gpu_train`, new.** It is built only with `-DNEURALFX_GPU=ON`: off by default, and not in CI.
  - It uses LibTorch's C++ API (BSD licence) on CUDA or ROCm, for its 3D convolutions, mixed precision, activation
    checkpointing and optimiser. No Python, and no pretrained weights: every model starts from our own initialisation.
  - It reads the repository's 3D clip format, trains, and writes `.nvfx3` (and `.nvfx` for 2D).
  - **The parity test, run on the PC:** for the same weights and inputs, LibTorch's forward pass and the CPU runtime's
    agree within float tolerance. Its gradients agree with our hand-written ones on small cases (the trainer's
    finite-difference checks). A model exported from it plays in the CPU runtime with frozen hashes as today.
- **A differentiable volume renderer in the GPU trainer** that matches `render3d` (Planck's law, CIE 1931, soot
  absorption, single scattering), tested against it. Picture losses then reach the stepper and the renderer from
  several cameras.
- **Teacher drivers, as text case files** under `sim3d/houdini/<row>/` and `sim3d/<code>/<row>/`. Houdini runs headless
  through `hbatch` with HScript, so there is no Python: one file builds the network, sets the row's settings and seed,
  and writes OpenVDB files. Research codes take their own input files.
- **`nvfx_ingest3d`, new:** reads OpenVDB (open-source library), AMReX plotfiles, FDS output and NetCDF into the clip
  format, as each run is written. It keeps the coarse grid, fine crops and 16-bit fields, then deletes the raw output
  (§7).

## 6. The models

### 6.1 One model per family, for all its effects

Each family has one network. Its input carries an **effect embedding** (one learned vector per effect) and the
effect's physical controls (wind, fuel rate, size, temperature, viscosity...) through FiLM, as stage 1's controls do.
The structure is stage 1's (STUDY_3D.md §4): a learned coarse state with memory channels, a stepper (3 x 3 x 3
convolutions, U-Net style), advection and pressure projection, seeded 3D detail, and a learned renderer. The gas state
holds velocity, temperature, the group's scalars (soot, fuel, vapour, dust) and memory. The free-surface state holds
velocity, liquid fraction or level set, and memory. The granular state holds density, velocity and memory.

Why one model per family: effects of a kind share their physics (buoyancy, entrainment, vortices, fronts), so what one
effect's data teaches helps the others. That can cut the runs each effect needs, which matters most when simulations
are the cost (§7). Whether it does is a measured question (§6.4).

### 6.2 Two sizes

| | teacher size (GPU, trains here) | runtime size (CPU, ships) |
|---|---|---|
| coarse state | 64³ (or 64 x 96 x 64 for tall effects) | 32³ (or 32 x 48 x 32), from study hres |
| stepper width | 64 to 128 channels | at most 64 (the int8 path's limit), chosen by measured cost |
| memory channels | 8 to 16 | 8 |
| detail | 256³, sparse bricks | 128³ to 256³, sparse bricks, seeded |
| effects | every effect of the family | one per file, distilled from the family model |
| where it plays | on the PC, for measuring | the shipped runtime, `.nvfx3` |

**Distillation:** each runtime model learns from the family model's rollouts and the teacher's data together, at the
runtime's grid, with the same staged recipe. Its size is the largest that fits the cost budget in TODO's "What it
costs" (a 32³ step about 10 to 16 ms on one core, 3 to 4 ms on four, to be measured).

### 6.3 The training recipe

From studies D, D2, L, N2 and hres, staged, with a snapshot after every stage:
1. one-step targets from the teacher's states (teacher forcing);
2. unrolled windows of 16, then 32, then 64 frames, with gradients through the whole window (activation
   checkpointing per step keeps memory flat; §7);
3. long free-running rollouts of hundreds of frames (no gradient through the burn-in), at the controls' extremes and
   with moving controls, against the endless statistics (study D2's loss);
4. picture losses through the differentiable renderer from four or more cameras;
5. ambient air at open edges in the model's own advection (N2's fix), with `nvfx_3d edge` checking every candidate.

Candidates are chosen on endless play as well as on tracking (STUDY_3D.md §13, point 1).

### 6.4 The questions each family answers, by a committed rule

1. Does each distilled runtime model beat (a) the coarse simulation with the same seeded detail and (b) today's best
   3D model of that effect (N2's C4, hres, R's C4) on tracking at 1 s and 2 s and on every endless statistic, with no
   edge runaway over 60 s?
2. Does the family model, distilled, beat a model trained on that effect alone at the same runtime size? If not,
   per-effect training wins for that effect, and the plan says so.
3. Against reality: the model's own runs pass the teacher's physical checks (§3.3) as the teacher did.

## 7. Time, data and storage

These are estimates from the formulas below. Each family's pilot replaces them with measured numbers in
`results/3d/<family>/pilot.md`.

**Kept data a run** (20 s at 30 fps, 600 frames):
- coarse 64³, 6 fields at 16 bits: 3.1 MB a frame, about 1 MB with sparse bricks (30% filled), so about 0.6 GB a run;
- fine 256³ crops for the detail layer and the renderer, two fields, every 10th frame, sparse: about 0.4 GB a run;
- so **about 1 GB a run**, and 150 to 300 runs an effect (training, validation and the test, sized by a null probe as
  study M3 does): **150 to 300 GB an effect**. All 33 effects at once would need about 5 to 10 TB. Family by family,
  archiving each family's kept data after its models ship, needs 2 to 4 TB of NVMe.

**GPU training at teacher size** (64³, 64 channels, about 10 full-resolution convolution layers):
- a layer is 64 x 64 x 27 x 262,144 multiply-adds, about 58 GFLOP; a step about 0.6 TFLOP forward and 1.7 TFLOP with
  its backward pass;
- a 32-frame window about 56 TFLOP: about **1 s on a 24 to 32 GB card** at a realistic 50 to 80 TFLOP/s in 16-bit;
- 100,000 to 200,000 windows a stage gives 1 to 3 days a stage, so **1 to 3 weeks of GPU per family model**, and
  about a day per effect to distil;
- memory: one layer's activations are 64 x 262,144 x 2 bytes, 34 MB. Checkpointing each step keeps only the states
  (about 17 MB a step), so 64 to 128-frame windows fit in 24 GB.

**Simulation time** is the unknown, and the first pilot measures it. With Houdini's GPU sparse pyro at an effective
256³, a frame takes somewhere between 0.1 and 10 s on a current card. That makes a 600-frame run 1 minute to 1.7
hours, and 300 runs anywhere from 5 hours to 3 weeks an effect. The pilot sets the grid and the run count to fit. If
it takes too long, the family model (§6.1) is what lets each effect use fewer runs. FLIP, MPM and the research codes
run mostly on the CPU, beside training.

**Order of magnitude:** the gas family, the first and largest, is a few months of the PC including its
simulations. The other two families are shorter. Weather and the rest of TODO's catalogue follow, as TODO's waves
plan.

## 8. Order of work, with gates

Each phase ends with a gate. A phase that fails its gate stops, is written up with its numbers, and the next step is
decided then.

| phase | what | gate (the rule committed before validation) |
|---|---|---|
| **P0** | Set up the PC, measure it (`results/3d/pc.md`), confirm the licences (§3.4) | the repository's CPU build and ctest pass on the PC; licences recorded |
| **P1** | `nvfx_gpu_train` and the parity test; retrain N2's fire and R's smoke on the GPU from their own data | parity within tolerance; the GPU-trained models tie or beat the CPU-trained ones on their own held-out tests; their files play in the CPU runtime |
| **P2** | The first teacher: Houdini Pyro fire and smoke, three resolutions; FDS and smoke_r as referees | the checks of §3.3 pass within the measurements' scatter, at the converged resolution |
| **P3** | Fire and smoke at teacher size, then distilled to runtime size | the runtime models pass §6.4's questions 1 and 3: better than the coarse simulation, N2's C4 and R's C4, no runaway over 60 s, playing on as well as tracking. This is where N2 and R failed. |
| **P4** | The gas family: the other 23 gas effects into the one gas model (by sub-family, in §3.2's order); distilled per effect | per effect, §6.4's questions 1 to 3; ship only what passes |
| **P5** | Free surface: Houdini FLIP and Basilisk referees; the 7 liquids | as P4 |
| **P6** | Granular: Houdini MPM; the avalanche, then the grains of dust, eruption and haboob | as P4 |
| **P7** | The rest of TODO's catalogue and the weather system (TODO's waves 1 to 3), with the research codes for weather | as P4, per row |

A passing effect ships as TODO's step 6 describes:
- its `.nvfx3` in `models/3d/` with `SHA256SUMS`;
- frozen hashes, pinned to float;
- demo videos;
- library entries.

Today's 2D effect stays as the light option.

## 9. Runtime and engines

Unchanged from TODO: the CPU steps, the engine's GPU draws, through the volume API (`nvfx_volume.h`), with sparse
bricks, level of detail, int8, threads and shards, and no allocation per frame. Determinism to the bit (principle 4)
is the runtime's, not the trainer's. GPU training need not be bit-reproducible; every shipped file is frozen by its
SHA-256, and its training log is kept.

**A decision for later:** stepping on the GPU inside FUSE (a compute shader) would allow the teacher-size models in
the game. Bit-identical frames on every player's GPU would need integer (int8) arithmetic throughout. Decide after P3,
from the measured CPU costs.

## 10. Risks

| risk | what happens then |
|---|---|
| A licence forbids training on outputs | that teacher is not used; the research codes teach that family (slower) |
| Houdini setups fail the physical checks | tune within physical parameters only; if it still fails, the research code teaches that effect |
| Simulations are too slow | coarser effective grids, fewer runs per effect through the family model, or EmberGen for bulk gas data (if its licence and automation allow) |
| Disk fills | family by family, ingest-then-delete, archive kept data after shipping (§7) |
| The family model doesn't beat per-effect models | per-effect models are trained instead; §6.4 question 2 says so, per effect |
| Distillation loses what the big model learned | the runtime size grows, within the cost budget, or that effect ships at a bigger size with a measured cost |
| AMD GPU | LibTorch on ROCm, Linux only; parity test as for CUDA |
| Learned models copy the teacher's flaws | the referee and the §3.3 checks on the model's own runs catch them; the teacher is fixed first (as R2 fixes R's momentum) |

## 11. What goes in git, and seeds

**In git:**
- the GPU trainer's code;
- the teachers' case files (text);
- small result tables and figures;
- the shipped runtime models (`models/3d/`, a few MB each, as today);
- demo videos (each 5 MB at most).

**Out of git:** simulator output, ingested clips, training runs and the teacher-size models (tens to hundreds of MB),
unless the owner decides otherwise.

**Seeds:** TODO's stage 2 ranges. Catalogue row # owns 10,000,000 + 100,000 x #. The family models take their
training seeds from each effect's row, and the tests use held-out settings and fresh seeds from the same row.

Nothing here is queued. P0 starts when the owner gives the PC's details (§2) and says go.
