# To do on a PC: 3D effects learned from an accurate 3D simulation

**Status: planned, not started.** Nothing in this file has been run. It is meant for a desktop PC, much later. On that
PC run time does not matter: a simulation may take days and training may take weeks.

## The goal

Go straight to 3D:
- simulate one effect in 3D as accurately as the PC can, however long it takes;
- train a 3D rollout model on those runs, as large and as high resolution as the PC allows;
- play it in 3D: a volume that moves, burns and responds the same from any camera, for FUSE.

The effect plays like today's fire, from start points with a learned stepper driven by seeded noise. It is
pseudorandom but deterministic, for as long as it plays and in a vast range of settings, only with a 3D state.

Today's effects are 2D sprites learned from `src/sim`, a 2D solver whose look was tuned by eye. They play close to that
teacher (studies D, D2, K and M), so the teacher is the ceiling on realism. A 3D teacher that is checked against
measured fires raises the ceiling. A 3D model removes the sprite's limit: it can be seen from the side, from above, or
walked around.

## Decisions (proposed; settle them before starting)

1. **Fire first: a buoyant pool fire with its smoke.** Fire is the most used effect, and the most is known about it
   from experiments, so the simulation can be checked against measured flames, not by eye. Smoke, steam and dust use
   the same machinery with other sources. Explosions need a compressible solver for the blast, which is a separate job.
2. **Simulator: large-eddy simulation (LES) of reacting, low-Mach, variable-density flow, with combustion, soot and
   radiation.** The proposal is NIST's Fire Dynamics Simulator (FDS), run as an external program.
   - FDS is public domain and validated against decades of fire experiments. It runs on CPUs, with MPI across cores.
   - Nothing of FDS enters the repository: only a C++ reader for its output.
   - The alternative is to write our own 3D solver. That gives full control, but it means validating a new code from
     scratch.
3. **The model is 3D from the start.**
   - The state: a coarse 3D grid of velocity (u, v, w), temperature, soot and memory channels.
   - The stepper: learned 3 x 3 x 3 convolutions with a 3D pressure projection.
   - The forcing: seeded 3D noise.
   - Fine detail: a seeded 3D detail layer carried by the coarse flow. This is study M's split: model what's ordered,
     leave random to random.
   - Pictures: a physically based ray marcher, the same light model the teacher's pictures are drawn with.
4. **What the runtime gives an engine: both of these.**
   - **The volume itself**, as 3D textures (density, temperature or emission, optionally velocity), for the engine's
     own volumetric renderer. This is the natural fit for FUSE: the GPU draws it, lit with the scene, and the CPU only
     steps it.
   - **A picture for a camera**, ray-marched on the CPU, for engines without volumetrics, for tests and for videos.
5. **Resolution: as high as the PC allows**, chosen from a measured pilot and a grid-convergence check, for the
   simulation and the model alike.

## What it costs to play (arithmetic from today's 2D numbers; to be measured)

The 2D stepper works on 32 x 32 cells with 3 x 3 convolutions. Its network stage takes about 0.17 ms per frame on one
core at float, and 0.10 ms at int8 with VNNI (REPORT.md §13.8).

**A 32³ stepper** has 32 times the cells and 3 times the taps, so about 100 times the work:
- about 16 ms per frame on one core at float, and 10 at int8;
- about 3 to 4 ms on four cores.

That is playable for one hero effect, or for a few with sharing and level of detail (the effect manager).

**A 64³ stepper** is 8 times that again, about 130 ms on one core. That is for baking, cutscenes or hero shots on many
cores, unless the int8 path and threading bring it down. Measure it.

**Other costs:**
- the 3D pressure projection on the coarse grid: a few ms;
- the detail layer: it scales with the fine grid's size;
- ray-marching a picture on the CPU: about 6 million samples at 256 x 256 px and 96 steps a ray.

With the engine's GPU drawing the volume, the CPU pays only the step.

## Steps

### 0. Measure the PC

Record the PC's cores, RAM, free disk and CPU features (AVX2, AVX-512, VNNI) in `results/3d/pc.md`. Every size below
scales from these numbers. The library stays C++ on the CPU, as the rest of the repository is. The engine's GPU only
draws the volume it is given.

### 1. Pilot: prove the simulation is right before training anything

- **One burner first.** A square pool burner of propane or methane, about 0.3 to 1 m across.
- **Resolution** is measured as D*/dx, the plume's characteristic diameter over the cell size:
  - D* = (Q / (rho_inf c_p T_inf sqrt(g)))^(2/5), where Q is the heat release rate;
  - FDS's own guidance treats D*/dx of 4 to 16 as coarse to moderate;
  - aim for at least 20, and as high as RAM allows.
- **Grid convergence.** Run three resolutions, each twice as fine as the last. The validation quantities below must
  stop moving between the two finest grids. If they don't, the production grid must be finer still.
- **Check against measured fires.** Write the acceptance bands in `results/3d/validation.md` before the runs:
  - mean flame height against Heskestad's correlation, L = 0.235 Q^(2/5) - 1.02 D (Q in kW, L and D in m);
  - puffing frequency against f = 1.5 / sqrt(D) Hz (D in m);
  - centreline temperature and velocity against McCaffrey's plume correlations, in the continuous, intermittent and
    plume regions;
  - pass if each falls within the experiments' own scatter. If one fails, fix the simulation before anything else.
- **Cost.** Measure the memory per cell and the time per cell-step. The production grid is the largest that fits in
  80% of RAM at an acceptable time per run.

### 2. Production runs

- **Settings that matter in a game:** burner size, heat release rate, cross-wind (speed and direction), and the fuel's
  soot yield (a clean flame to a sooty one). Perhaps also an obstacle the fire wraps around.
  - Every setting gets several seeds: the inflow or initial perturbation differs, so the runs are different fires with
    the same statistics.
- **Each run:**
  - a warm-up, discarded until the puffing is statistically steady;
  - then 30 to 60 s at 30 frames per second.
- **The held-out split is written in `results/3d/split.md` before the first production run.**
  - It holds settings never trained on and fresh seeds.
  - Seed bases start at 9,000,000. The 7,000,000s are water's and the 8,000,000s are study M's.
- **Output per frame:** temperature, soot density, velocity (u, v, w) and heat release rate per unit volume, as 3D
  fields.
- **Disk.**
  - A 256³ grid with six float32 quantities is 402 MB a frame, so 362 GB for 30 s at 30 fps.
  - Ingest each output chunk as it is written, then delete it. Keep the coarse grids for the stepper, the fine fields
    the detail layer and the start points need, and 16-bit storage.
  - Keep full-resolution fields for a few runs only, for the pictures and the checks.
  - The pilot gives the real numbers.
- **None of it goes in git.** Runs live under the data root.

### 3. New code in the repository (C++, CMake-wired, tests in CI)

- **`nvfx_ingest3d`** reads FDS's 3D output (3D slices or Plot3D files) into a 3D clip format. It stores the fields,
  the frame times, the setting and the seed. It writes block averages for the coarse grid and fine fields at the
  detail layer's size.
- **A physically based volume renderer** (`src/render3d`), used for the teacher's pictures and the model's alike:
  - emission from temperature by Planck's law, integrated against a camera's RGB response;
  - absorption by soot, in proportion to its volume fraction;
  - optional single scattering of a sky and of the fire's own light;
  - one fixed exposure and tone curve per effect;
  - SIMD, threaded, and no allocation per frame;
  - limit: the blue base of a clean flame (chemiluminescence) is not in a soot-and-temperature model; say so wherever
    the pictures are shown.
- **The 3D rollout model and its trainer** (`include/neuralfx/rollout3d.hpp`, `src/train/rollout3d_train.cpp`). This is
  study D's model with one more dimension:
  - start points: 3D states taken from the runs;
  - the stepper: 3D convolutions with FiLM on the controls, plus memory channels;
  - advection: semi-Lagrangian, or MacCormack as study L found for fast swirls;
  - a 3D pressure projection;
  - seeded 3D noise forcing;
  - the detail layer in 3D;
  - training: hand-written gradients, checked against finite differences as today's are; the statistics loss (study
    D2) and long windows (study L).
- **The 3D runtime** (`include/neuralfx/nvfx_volume.h`), a new C API beside the effect and scene APIs:
  - create an instance from a 3D effect file, with a seed, controls and a variation;
  - step it to a time;
  - read the volume as 3D textures (R8, R16F or R32F; density, temperature or emission, velocity), or draw it for a
    camera given as view and projection matrices into RGBA;
  - per-ISA SIMD for the 3 x 3 x 3 convolutions and the int8 path;
  - threads, no allocation per frame, and shards so it plays forever;
  - the ABI test, `INTEGRATION.md`'s sizes and `NVFX_VERSION` updated;
  - a Godot demo (a FogVolume or a 3D texture) and an example host.
- **Tests:**
  - ingest round trip on a small synthetic file;
  - the renderer against analytic cases: a uniform absorbing slab must obey Beer-Lambert, and a thick uniform hot slab
    must tend to the blackbody;
  - the 3D stepper's gradients against finite differences;
  - runtime parity with the trainer's reference on every ISA, float and int8;
  - the same frames on any number of threads;
  - no allocation per frame.

### 4. The model, as big as it earns

Start at a 32³ coarse grid, with a stepper 24 to 32 wide, 4 memory channels, and fine detail at 128³. Then climb, and
keep each step only if it wins on validation:
- a 48³ or 64³ coarse grid;
- a stepper 48 to 64 wide, and 8 memory channels;
- fine detail at 256³.

Keep the stepper at most 64 channels wide, or extend the int8 path, which takes at most 64 (`src/runtime/rt_rollout.hpp`).
Training runs for days to weeks, with checkpoints, so it can stop and resume.

### 5. Measure

Follow the repository's rules:
- write the rules down before the test, and run the test once;
- use held-out settings and seeds;
- use paired bootstrap 95% intervals with 10,000 resamples;
- report ties as ties.

What to measure:
- **Against the simulation, in 3D:**
  - tracking from the true state, as field errors and as pictures from several cameras, at 8 frames, 1 s and 2 s;
  - endless statistics, as in studies D2 and K: the 3D detail spectrum, motion, coverage, emission and the mean volume.
- **Against reality.** The model's own fires are measured exactly as the simulation's were:
  - flame height and puffing frequency against Heskestad's correlation and 1.5 / sqrt(D);
  - centreline temperature and velocity against McCaffrey's correlations.

  A 3D model allows this, where a sprite does not. If real fire footage is supplied, also compare pictures from a
  matching camera with study J's statistics. If there is none, say so.
- **Against the 2D fire, from the front:** the v2 fire sprite beside the 3D fire drawn from the same side, on the
  checks above.
- **Cost:** milliseconds per frame for the step and for the picture, separately, at each coarse grid size, on 1 and 4
  threads, AVX2 and AVX-512, float and int8. Also the memory per instance.

### 6. Ship it, if it wins

If it wins, ship:
- the 3D fire in the effect pack;
- the volume API documented in `ENGINES.md` and `INTEGRATION.md`, with how FUSE draws the 3D textures;
- frozen keyframes, pinned to float;
- demo videos in `videos/`: the fire walked around, in a cross-wind, and at several heat release rates.

Bringing 3D modules into scene scripts (the field bus, couplings and the library) is a later job. If it does not win,
report the result with its numbers, as every other study does.

## What goes in git

Code, docs, small result tables, figures and the demo videos. Never 3D runs, frames, volumes, renders or model weights.
