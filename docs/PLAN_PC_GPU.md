# Plan: every effect in 3D, trained on the PC's GPU from the best simulators, each in hours

**Status:** a plan, written 21 October 2026 and revised the same day so that each effect takes hours, not weeks.
Nothing in it has started. It refines stage 2 of [TODO.md](../TODO.md) (the catalogue, the measuring rules and the
seeds stay as they are there) in four ways:

1. **An effect in hours.** Each effect goes from simulation to a shipped, tested model in **4 to 8 hours of the PC,
   unattended**: simulated on the GPU, trained on the GPU, measured by the house rules, shipped if it passes (§7).
   Each family's first effect also pays once for checking the teacher and for training the family model (1 to 2
   days). All 33 of today's effects take about **2 to 3 weeks** of the PC in all, not months.
2. **The GPU simulates and trains.** Teachers that run on the GPU are batched (§3.2), and training moves from this
   container's CPU trainer to the GPU. Models can be bigger, learn from far more data and learn from long rollouts
   with gradients through them. The shipped effects still step on the CPU, as today, unless the owner later chooses a
   GPU path (§9).
3. **The teachers are the best simulators there are, checked against measurements.** For most effects that is
   Houdini, the film industry's standard, run on the GPU where it can be. Research codes that are validated against
   experiments are referees, and our own solver, ported to the GPU, is the fast fallback (§3).
4. **One model for all the effects of a kind.** One gas model for the 25 gas effects, one free-surface model for the
   7 liquids, and one granular model. Each is trained once, conditioned on the effect and its controls. Each new
   effect is a fine-tune of it, which is what makes an effect take hours (§6).

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
9. [Runtime and engines](#9-runtime-and-engines)
10. [Risks](#10-risks)
11. [What goes in git, and seeds](#11-what-goes-in-git-and-seeds)

## 0. In short

- **Per effect, 4 to 8 hours, unattended**, with one command (`nvfx_pc effect <name>`):

  | step | hours | what |
  |---|---|---|
  | simulate | 1 to 2 | about 100 runs of 10 s at 128³, batched on the GPU |
  | fine-tune | 1 to 3 | the family model, fine-tuned for the effect at runtime size |
  | measure | about 1 | null probe, validation, freeze, the one test |
  | ship | under 0.5 | videos and the library entry |

  Ingest runs as the simulations write; the runs for the next effect simulate while this one trains.
- **Once per family, 1 to 2 days:** check the teacher against measurements, and train the family model on its first
  few effects.
- **Teacher:** Houdini 21 for the effects games use. Its GPU sparse pyro does gases, FLIP does liquids, and MPM does
  snow, mud and grains. Research codes validated against experiments referee each family on a few cases: FDS for fire
  and smoke, Basilisk for splashes, PeleLMeX for flames and blasts. Our own 3D solver, ported to the GPU, is the
  fallback if Houdini is too slow or its licence forbids training (§3.2).
- **Every teacher setup must pass its family's physical checks before any training data is made:** flame height,
  puffing frequency, plume spread, front speed, angle of repose (§3.3).
- **Model:** one network per family, at runtime size (a 32³ state, at most 64 channels), conditioned on effect and
  controls. Each effect fine-tunes it; the result is the shipped `.nvfx3` for today's CPU runtime. A bigger 64³ model
  is an option kept for an effect that fails at 32³ (§6.2).
- **Training on the GPU** uses LibTorch, PyTorch's C++ library: no Python, and no pretrained networks. It sits in a
  separate build option, off by default, so this repository's CPU build and CI do not change.
- **Each effect has a gate,** run under the house rules: the rule written before validation, held-out settings and
  seeds, one test, paired bootstrap intervals. The rule comes from a template committed once per family, and its
  numbers (margins and run counts from a null probe) are committed by the pipeline before validation, as study Q2
  does. The first gate is a 3D fire and smoke that pass where studies N2 and R failed, each in under 8 hours.

## 1. Why change the plan

What this container's 3D studies found:

| study | what it found | what fixes it |
|---|---|---|
| N (fire) | tracks a held-out fire better than the coarse simulation; heat builds at the upwind edge and grows a second fire | ambient edges, long rollouts (N2) |
| N2 (fire) | no runaway over 30 s, but too calm (motion 0.68 of the real runs'); an edge bound 3% too tight | bigger model, longer windows with gradient, bounds from the teacher's spread |
| hres (fire) | 32 x 48 x 32 state, finer flames; not yet tested | the same, at scale |
| R (smoke) | a teacher that is a plume by MTT's measures; the learned smoke runs away at an edge and spreads too far (mean picture -2.52 dB) | more data, longer windows with gradient, more capacity |
| R2 (smoke) | under way: the teacher's momentum fixed (confinement made momentum-neutral) | — |

The limits are the same each time:
- small models: 16 to 32 channels, 14,000 to 60,000 weights;
- little data: 40 to 100 runs;
- short training windows: gradients through 12 to 24 frames, on a CPU.

A GPU removes all three. Windows of 64 to 128 frames with gradient fit in its memory (§7.3). Data comes from as many
runs as a batched GPU simulator can make. And the teacher's physics is the ceiling (STUDY_3D.md §13, point 5), so the
teacher has to be the best available.

The first version of this plan took about 1 GB a run, 150 to 300 runs an effect at 256³, a 64³ teacher model and a
distillation per effect: weeks per effect, months in all. Its sizes came from the 256³ detail the TODO plan named.
They are not needed for what is learned. The learned state is 32³, and the seeded detail layer makes the fine scale
(principle 1: model what's ordered, leave random to random). So a 128³ teacher, half the runs, runtime-size models
and per-effect fine-tunes give the same tests in hours (§7).

## 2. What I need from you

| # | question | why it matters | default if you don't say |
|---|---|---|---|
| 1 | GPU model and its memory (VRAM) | sets batch, window length and how many simulations run at once; NVIDIA uses CUDA, AMD uses ROCm (Linux only) | an NVIDIA card with 24 to 32 GB |
| 2 | CPU cores, RAM, disk | liquids and referees run on the CPU beside the GPU; 16 cores runs 8 liquid runs at once | 16+ cores, 64 to 128 GB RAM, 2 TB NVMe |
| 3 | Operating system | Houdini and LibTorch run on Windows and Linux; ROCm and some research codes are Linux-first | Linux, or Windows with WSL2 for the research codes |
| 4 | Houdini licence: Indie, Core/FX, or none yet | Indie allows commercial use under US$100K revenue and US$1M funding, and its files can't share a pipeline with commercial Houdini; Apprentice and Education can't be used | Houdini Indie |
| 5 | Is a training-only GPU dependency (LibTorch, CUDA) acceptable in an optional build target? | the repository has been CPU-only; the runtime stays CPU-only | yes, optional and off by default |
| 6 | Teacher choice: Houdini first, or our own GPU solver first | Houdini gives the look and every family; ours is fastest, needs no licence, but has only gases and must pass the same checks | Houdini first for the look, ours as the fallback |
| 7 | Do you have real footage of any effect? | the only check on looks where nothing can be measured (study J) | none, and say so per effect |

## 3. The teachers

### 3.1 What "best" means here

There are two kinds of best simulator, and the plan uses both:
- **Best for the look:** the production simulators film VFX uses. Houdini is the standard: one tool for gases (Pyro,
  with a GPU sparse solver in Houdini 21), liquids (FLIP), snow, mud and soil (MPM, production-ready in Houdini 21),
  particles and cloth. EmberGen (JangaFX) is a real-time GPU gas simulator, very fast, but gases only, and it has no
  documented batch or command-line mode, so it can't run unattended.
- **Best for the physics:** research codes validated against experiments. FDS (NIST) for fire and smoke, PeleLMeX for
  flames and blasts (C++, runs on GPUs through AMReX), Basilisk for splashes, jets and drops, PALM or MicroHH for clouds,
  fog and boundary layers, and CM1 for storms. These are the simulators already in TODO's stage 2. They are too slow
  to make every effect's training data in hours, so they referee: a few runs per family, once.

Production simulators are tuned by artists, not validated. So Houdini teaches only after its setup passes the same
checks a research code would (§3.3). Weather (clouds, storms, regional systems) stays with the research codes, as
TODO plans, outside the hours target: Houdini has no cloud microphysics.

### 3.2 Speed: what makes the data take an hour or two

- **Resolution 128³** (or 128 x 192 x 128 for tall effects), not 256³. The learned state is 32³, so 128³ is 4 cells of
  teacher per learned cell. The detail layer is seeded, and its statistics come from a few 256³ runs per family, made
  once.
- **10 s runs** (300 frames at 30 fps) after the warm-up, not 20 to 60 s. Endless play is still tested over 60 s; the
  model's own long rollouts are what training stage 3 uses (§6.3).
- **About 100 runs an effect** (training 60, validation 20, the test 20). The null probe sizes validation and the test
  as study M3 does. If it says 20 settings lack power, the pipeline runs more that night and says so.
- **Batched on the GPU:**
  - Houdini's sparse pyro runs several simulations at once on one card, as separate `hbatch` processes.
  - Our own GPU solver runs 16 to 64 simulations as one batched tensor.
- **Liquids and grains:** Houdini FLIP and MPM run mostly on the CPU. The pipeline runs 8 at once on 16 cores, at 96³
  to 128³ with 6 to 10 s runs.

**Our own solver on the GPU (`nvfx_gpu_sim`), the fallback.**
- It is today's `sim3d` (fire, `smoke_r` and R2's momentum fix) written with LibTorch tensors: advection by trilinear
  sampling, pressure by multigrid, all batched.
- It is headless and deterministic, and it needs no licence.
- `smoke_r` already passes the MTT plume checks. The fire must first pass §3.3, which it does not yet (it puffs at 0.5
  to 0.65 Hz where a fire its size puffs near 2.7 Hz; STUDY_3D.md §13). Its check at 128³ is part of P2.
- At 128³ a step should take a few milliseconds a simulation in a batch, so 100 runs take well under an hour. The
  pilot measures it.

### 3.3 Per family

Today's 33 2D effects, by family:
- **gas (25):** candle, cloud, coals, crown fire, dust, dust explosion, eruption, explosion, fire, fog, haboob, heavy
  gas, ink, jet, launch, magic, mushroom, orb, puffs, smoke, smoke ring, steam, vortex, wildfire, window fire;
- **free surface (7):** fountain, geyser, lava, rain shaft, sea spray, splash, waterfall;
- **granular (1, and the grains in 3 more):** avalanche, plus the grains in dust, eruption and haboob.

| family | effects | teacher | referee (a few cases, once) | runs on |
|---|---|---|---|---|
| gas: fire and combustion | fire, candle, wildfire, crown fire, window fire, coals, dust explosion | Houdini sparse pyro (combustion) | FDS (fire), PeleLMeX (jet and small flames) | GPU |
| gas: smoke, plumes, puffs | smoke, smoke ring, puffs, steam, jet, launch, ink | Houdini sparse pyro, or our `smoke_r` on the GPU | FDS; MTT plume theory | GPU |
| gas: blasts | explosion, mushroom, eruption (plume) | Houdini sparse pyro | PeleC (compressible, early shock) | GPU |
| gas: dense and laden | heavy gas, haboob, dust | Houdini sparse pyro with density | FDS; gravity-current theory | GPU |
| gas: designed | magic, orb, vortex | Houdini sparse pyro with designed forces | none: consistency only | GPU |
| gas: weather | cloud, fog | PALM or MicroHH (outside the hours target) | measured cloud statistics | CPU or GPU |
| free surface | waterfall, fountain, splash, geyser, rain shaft, sea spray | Houdini FLIP | Basilisk | CPU, 8 at once |
| viscous and hot | lava | Houdini FLIP with temperature-dependent viscosity, or MPM | measured lava flow rates | CPU, 8 at once |
| granular | avalanche; the grains in dust, eruption and haboob | Houdini MPM (snow, soil) | angle of repose, runout, front speed | CPU, 8 at once |

Spray, foam, droplets, sparks, embers and grains stay seeded random (principle 1): the teachers' particles give their
statistics, not training targets frame by frame.

**Not used in any teacher:** Houdini's ML-based smoke upres, or any learned node. The project trains no model on
another model's output, and uses no pretrained networks.

### 3.4 The checks every teacher setup passes first (once per family)

Each check is written in `results/3d/<family>/validation.md` before the runs, as TODO's pipeline asks. The setup is
run at 64³, 128³ and 256³. It must converge by 128³ (or the production grid moves up), then pass within the
measurements' scatter:
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
(study N2's lesson). Each effect's setup is a variation of its family's checked setup, within physical parameters
(fuel, size, wind, viscosity), so it needs no new check of its own. An effect that needs a new mechanism gets a check
first.

### 3.5 Licences

Confirm before the first production run, and record in the licence register (`docs/DATA.md`; extend `licences.tsv` to
simulator outputs, with `--licence commercial --licence-note` naming the agreement):
- **SideFX (Houdini):** the licence agreement's web page (revision 13 October 2021) says nothing on machine learning or
  on who owns outputs. Indie is for entities under US$100K revenue and US$1M funding, and Indie files can't share a
  pipeline with commercial Houdini. Apprentice is non-commercial. Ask SideFX in writing whether training a model on
  Indie outputs is allowed, and keep the answer.
- **JangaFX (EmberGen):** the licence agreement (19 July 2025) says nothing on machine learning or outputs. Indie is
  for revenue or budget under US$1M. Not in the hours plan, as it can't run unattended.
- **Research codes:** FDS is public domain; PeleLMeX is open source; Basilisk, PALM and MicroHH are GPL. Their outputs
  are ours to use. GPL applies to their code, which stays outside this repository; we keep only case files and
  readers.
- **Our own solver:** ours.

## 4. The PC

Measure it once (TODO's step 0) into `results/3d/pc.md`: CPU, cores, RAM, GPU, VRAM, driver, CUDA or ROCm version,
disks and their speeds, and the operating system.

The pipeline keeps both halves busy:
- the GPU simulates one effect's gases while it trains the previous one, as two queues, with simulation taking turns
  with training batches;
- the CPU ingests, runs liquids and grains, and runs the CPU runtime's validation and tests.

## 5. Software on the PC

- **This repository, as today:** the CPU build, the tests, the runtime and the tools. CI does not change.
- **`nvfx_pc`, new: the pipeline in one command.**
  - `nvfx_pc effect <name>` runs the whole chain unattended: simulate, ingest, fine-tune, null probe, commit the
    rule's numbers, validate, freeze, test once, ship if it passes, videos.
  - `nvfx_pc family <name>` runs the once-per-family steps: the teacher's checks and the family model's training.
  - Each step writes a marker, so a stopped chain resumes where it was.
  - It logs the time of every step to `results/3d/<effect>/time.csv`, so the hours target is measured, not hoped.
- **`nvfx_gpu_train`, new.** It is built only with `-DNEURALFX_GPU=ON`: off by default, and not in CI.
  - It uses LibTorch's C++ API (BSD licence) on CUDA or ROCm, for its 3D convolutions, 16-bit mixed precision,
    activation checkpointing, CUDA graphs and optimiser. No Python, and no pretrained weights: every model starts
    from our own initialisation.
  - It reads the repository's 3D clip format, trains, and writes `.nvfx3` (and `.nvfx` for 2D).
  - **The parity test, run on the PC:** for the same weights and inputs, LibTorch's forward pass and the CPU runtime's
    agree within float tolerance. Its gradients agree with our hand-written ones on small cases (the trainer's
    finite-difference checks). A model exported from it plays in the CPU runtime with frozen hashes as today.
- **`nvfx_gpu_sim`, new (the fallback teacher, §3.2):** `sim3d`'s gases on the GPU, batched, behind the same option.
  Its parity test: on the same small case, its fields match `sim3d`'s within float tolerance.
- **A differentiable volume renderer in the GPU trainer** that matches `render3d` (Planck's law, CIE 1931, soot
  absorption, single scattering), tested against it. Picture losses then reach the stepper and the renderer from
  several cameras.
- **Teacher drivers, as text case files** under `sim3d/houdini/<row>/` and `sim3d/<code>/<row>/`. Houdini runs headless
  through `hbatch` with HScript, so there is no Python: one file builds the network, sets the row's settings and seed,
  and writes OpenVDB files. Research codes take their own input files.
- **`nvfx_ingest3d`, new:** reads OpenVDB (open-source library), AMReX plotfiles, FDS output and NetCDF into the clip
  format as each run is written. It keeps the 32³ coarse grid and 128³ crops at 16 bits, then deletes the raw output.

## 6. The models

### 6.1 One model per family, for all its effects

Each family has one network. Its input carries an **effect embedding** (one learned vector per effect) and the
effect's physical controls (wind, fuel rate, size, temperature, viscosity...) through FiLM, as stage 1's controls do.
The structure is stage 1's (STUDY_3D.md §4): a learned coarse state with memory channels, a stepper (3 x 3 x 3
convolutions, U-Net style), advection and pressure projection, seeded 3D detail, and a learned renderer. The gas state
holds velocity, temperature, the group's scalars (soot, fuel, vapour, dust) and memory. The free-surface state holds
velocity, liquid fraction or level set, and memory. The granular state holds density, velocity and memory.

Why one model per family: effects of a kind share their physics (buoyancy, entrainment, vortices, fronts). A model that
already knows them needs far less of each new effect's data and training. That is what lets 100 runs and 1 to 3
hours of fine-tuning be enough. Whether it is enough is measured, effect by effect (§6.4).

### 6.2 One size, by default

| | the family model, and each effect's fine-tune (default) | the bigger option (only if an effect fails at 32³) |
|---|---|---|
| coarse state | 32³ (or 32 x 48 x 32 for tall effects), from study hres | 64³, distilled back to 32³ |
| stepper width | 64 channels (the int8 path's limit) | 64 to 128 |
| memory channels | 8 | 8 to 16 |
| detail | 128³, sparse bricks, seeded | 256³ |
| training | family: 1 to 2 days once; effect: 1 to 3 hours | 1 to 2 weeks per family |
| ships | the fine-tune itself, as `.nvfx3` | the distilled 32³ model |

The default trains at the size it ships, so no distillation step costs hours per effect. The bigger option is kept for
an effect whose fine-tune fails its test at 32³; it is the slow path, and the plan says so where it is used.

### 6.3 The training recipe

From studies D, D2, L, N2 and hres, staged, with a snapshot after every stage. The family model runs all four stages
over its first effects. Each effect's fine-tune runs stages 2 to 4, shorter, from the family model:
1. one-step targets from the teacher's states (teacher forcing);
2. unrolled windows of 16, then 32, then 64 frames, with gradients through the whole window (activation
   checkpointing per step keeps memory flat; §7.3);
3. long free-running rollouts of hundreds of frames (no gradient through the burn-in), at the controls' extremes and
   with moving controls, against the endless statistics (study D2's loss);
4. picture losses through the differentiable renderer from four or more cameras.

Throughout, ambient air comes in at open edges in the model's own advection (N2's fix), and `nvfx_3d edge` checks
every candidate. Candidates are chosen on endless play as well as on tracking (STUDY_3D.md §13, point 1).

### 6.4 The questions each effect answers, by a committed rule

1. Does the effect's model beat (a) the coarse simulation with the same seeded detail and (b) today's best 3D model of
   that effect (N2's C4, hres, R's C4) on tracking at 1 s and 2 s and on every endless statistic, with no edge
   runaway over 60 s?
2. Against reality: the model's own runs pass the teacher's physical checks (§3.4) as the teacher did.
3. Once per family, on its first three effects: does the family model's fine-tune beat a model trained on that effect
   alone, with the same data and the same hours? If not, per-effect training is used for that family, and the hours
   are measured again.

## 7. An effect in hours

### 7.1 The budget

Times are for one effect on a 24 to 32 GB NVIDIA card and a 16-core CPU. The pipeline logs every step's real time to
`results/3d/<effect>/time.csv`, and the pilots replace these numbers.

| step | where | estimate | how |
|---|---|---|---|
| simulate 100 runs (gas) | GPU | 1 to 2 h | 128³ sparse pyro at 0.1 to 0.2 s a frame, 300 frames, several at once; or our GPU solver, under 1 h |
| simulate 100 runs (liquid, granular) | CPU | 1 to 3 h | 96³ to 128³ FLIP or MPM, 6 to 10 s runs, 8 at once on 16 cores |
| ingest | CPU | overlapped | as each run is written |
| fine-tune | GPU | 1 to 3 h | 10,000 to 30,000 windows from the family model (§7.3) |
| null probe, validation, freeze | CPU and GPU | about 0.5 h | the CPU runtime on 16 cores; the rule's numbers committed before validation |
| the one test | CPU | under 0.5 h | 20 held-out settings, fresh seeds |
| ship: videos, library entry, `models/3d` | CPU | under 0.5 h | |
| **total** | | **4 to 8 h** | the next effect's simulations run during this one's training |

Once per family:

| step | estimate |
|---|---|
| the teacher's checks (§3.4) at 64³, 128³ and 256³, and the referee's few runs | half a day to a day (the referee on the CPU, beside other work) |
| the family model: its first three effects' data, stages 1 to 4 | 1 to 2 days of GPU |

**All of today's 33 effects:**
- the 23 gases with a GPU teacher at about 3 to 4 a day, after 2 days for the family: about 8 to 10 days (cloud and
  fog go with the weather, in P7);
- the 7 liquids at 2 to 3 a day, after 2 days: about 5 days;
- the avalanche and the grains of 3 gas effects: about 3 days.

In all, **2 to 3 weeks** of the PC, unattended. Weather and the rest of TODO's catalogue follow, row by row, in the
same hours where a GPU teacher exists, and slower where only a research code does.

### 7.2 Data and storage

- **Kept a run:** the 32³ coarse grid of 6 fields at 16 bits is 0.4 MB a frame, 0.12 GB for 300 frames. 128³ crops of
  two fields every 10th frame, sparse, add about 0.1 GB. So **about 0.2 GB a run**.
- **An effect:** 100 runs, about 20 GB. **All 33 effects:** about 0.7 TB, so 2 TB of NVMe holds everything with room
  for raw output in flight. Raw output is deleted once ingested.

### 7.3 GPU training time

- **At 32³ with 64 channels:** a 3 x 3 x 3 layer is 64 x 64 x 27 x 32,768 multiply-adds, about 7 GFLOP. Ten layers are
  0.07 TFLOP forward and about 0.22 TFLOP with the backward pass. A 64-frame window is about 14 TFLOP: about **0.2 to
  0.3 s** on a 24 to 32 GB card at a realistic 50 to 80 TFLOP/s in 16-bit.
- **A fine-tune** of 10,000 to 30,000 windows is therefore **1 to 3 hours**. The family model's 200,000 to 400,000
  windows take **1 to 2 days**.
- **Memory:** one layer's activations are 64 x 32,768 x 2 bytes, 4 MB. With checkpointing per step, 128-frame windows
  and batches of 8 to 16 fit in 24 GB.
- **The bigger option (64³)** costs 8 times as much per window, which is why it is the slow path (§6.2).

## 8. Order of work, with gates

Each phase ends with a gate. A phase that fails its gate stops, is written up with its numbers, and the next step is
decided then. Every effect's time is a gate too: an effect that takes more than 12 hours stops the roll-out until the
slow step is found and fixed.

| phase | what | how long | gate (the rule committed before validation) |
|---|---|---|---|
| **P0** | Set up the PC, measure it, confirm the licences (§3.5) | a day | the CPU build and ctest pass on the PC; licences recorded |
| **P1** | `nvfx_gpu_train`, `nvfx_gpu_sim` and their parity tests; retrain N2's fire and R's smoke on the GPU from their own data | 1 to 2 days | parity within tolerance; the GPU-trained models tie or beat the CPU-trained ones on their own held-out tests, in hours, not the CPU's days |
| **P2** | The gas teacher: Houdini sparse pyro fire and smoke, at 64³, 128³ and 256³; FDS and `smoke_r` as referees; time a frame | a day | §3.4's checks pass by 128³; a frame takes 0.2 s or less at 128³ (or our GPU solver takes over for the effects it passes for) |
| **P3** | The gas family model on fire, smoke and one more; then fire and smoke through `nvfx_pc effect` | 2 to 3 days | §6.4 for fire and smoke: better than the coarse simulation, N2's C4 and R's C4, no runaway over 60 s, playing on as well as tracking; each effect's chain in 8 hours or less. This is where N2 and R failed. |
| **P4** | The other 20 gas effects, 3 to 4 a day (cloud and fog wait for P7's weather teachers) | about a week | per effect, §6.4; ship only what passes |
| **P5** | Free surface: FLIP checked against Basilisk; the 7 liquids | about 5 days | as P4 |
| **P6** | Granular: MPM checked; the avalanche, then the grains of dust, eruption and haboob | about 3 days | as P4 |
| **P7** | The rest of TODO's catalogue and the weather system (TODO's waves 1 to 3) | row by row | as P4, per row |

A passing effect ships as TODO's step 6 describes:
- its `.nvfx3` in `models/3d/` with `SHA256SUMS`;
- frozen hashes, pinned to float;
- demo videos;
- library entries.

Today's 2D effect stays as the light option. An effect that fails is written up with its numbers and kept off the
default; the next step for it (the bigger option, more runs, or a fixed teacher) is decided then.

## 9. Runtime and engines

Unchanged from TODO: the CPU steps, the engine's GPU draws, through the volume API (`nvfx_volume.h`), with sparse
bricks, level of detail, int8, threads and shards, and no allocation per frame. Determinism to the bit (principle 4)
is the runtime's, not the trainer's. GPU training need not be bit-reproducible; every shipped file is frozen by its
SHA-256, and its training log is kept.

**A decision for later:** stepping on the GPU inside FUSE (a compute shader) would allow bigger models in the game.
Bit-identical frames on every player's GPU would need integer (int8) arithmetic throughout. Decide after P3, from the
measured CPU costs.

## 10. Risks

| risk | what happens then |
|---|---|
| Houdini's GPU pyro is slower than 0.2 s a frame at 128³ on your card | fewer runs at once, or our GPU solver for the effects it passes for; the hours are measured, not assumed |
| A licence forbids training on outputs | our GPU solver teaches the gases it passes the checks for; research codes teach the rest, slower |
| Houdini setups fail the physical checks | tune within physical parameters only; if it still fails, our solver or the research code teaches that effect |
| 128³ is not converged for an effect | that effect's grid moves up (160³ or 192³) and its hours are measured again |
| 100 runs lack power for the test | the null probe says so before validation; the pipeline adds runs, and the extra hours are logged |
| A fine-tune fails at 32³ | the bigger option for that effect (§6.2), labelled as the slow path |
| The family model doesn't beat per-effect training (§6.4, question 3) | per-effect training for that family; the hours are measured again |
| Liquids are slower than 3 hours | lower grid or shorter runs within the checks, or more CPU runs at once |
| AMD GPU | LibTorch on ROCm, Linux only; parity test as for CUDA |
| Learned models copy the teacher's flaws | the referee and the §3.4 checks on the model's own runs catch them; the teacher is fixed first (as R2 fixes R's momentum) |

## 11. What goes in git, and seeds

**In git:**
- the GPU trainer's, the GPU solver's and the pipeline's code;
- the teachers' case files (text);
- each family's rule template;
- each effect's committed rule numbers, results tables and `time.csv`;
- figures;
- the shipped runtime models (`models/3d/`, a few MB each, as today);
- demo videos (each 5 MB at most).

**Out of git:** simulator output, ingested clips, training runs and checkpoints, unless the owner decides otherwise.
The family models are a few MB at 32³ and may go in `models/3d/families/` if the owner wants them kept.

**Seeds:** TODO's stage 2 ranges. Catalogue row # owns 10,000,000 + 100,000 x #. Each effect's training, validation and
test seeds come from its row, and the tests use held-out settings and fresh seeds from the same row. The family
models' seeds come from their first effect's row.

Nothing here is queued. P0 starts when the owner gives the PC's details (§2) and says go.
