# To do on a PC: a 3D reference simulation and a bigger model

**Status: planned, not started.** Nothing in this file has been run. It is meant for a desktop PC, much later. On that
PC run time does not matter: a simulation may take days and training may take weeks.

## The goal

Every learned effect in this repository learns from `src/sim`, a 2D stable-fluids solver whose look was tuned by eye.
The models now play close to that teacher: studies D, D2, K and M measure them against the simulation, not against
real fire. So the realism ceiling is the teacher, not the network.

This plan replaces the teacher for one effect with the most accurate 3D simulation the PC can run, however long it
takes. A larger, higher-resolution model is then trained on it.

The shipped runtime must stay CPU-light. Realism comes from the data. The cost is chosen at the end, from measured
numbers.

## Decisions (proposed; settle them before starting)

1. **Fire first: a buoyant pool fire with its smoke.** Fire is the most used effect, and the most is known about it from
   experiments. The simulation can therefore be checked against measured flames, not by eye. Smoke and steam can follow
   from the same kind of runs.
   - Explosions need a compressible solver for the blast, which is a separate job.
2. **Simulator: large-eddy simulation (LES) of reacting, low-Mach, variable-density flow, with combustion, soot and
   radiation.** The proposal is NIST's Fire Dynamics Simulator (FDS), run as an external program.
   - FDS is public domain and validated against decades of fire experiments. It runs on CPUs, with MPI across cores.
   - Nothing of FDS enters the repository: only a C++ reader for its output.
   - The alternative is to extend `src/sim` to 3D ourselves. That gives full control, but it means validating a new
     code from scratch. Do that only if FDS cannot give something we need.
3. **What the model learns: two stages.**
   - **(a) The runtime stays 2D.** The 3D runs are rendered from many directions with a physically based volume
     renderer. The existing rollout pipeline (start points, stepper, detail layer, renderer) learns from the projected
     fields and the rendered pictures. The runtime, the API, the library and the manager stay as they are; only the
     teacher is better.
   - **(b) Later, and only if (a) shows the new teacher was worth it: a true 3D rollout model.** It has a coarse 3D
     state and a volumetric render, for effects the camera moves around. This is new runtime work, and it costs far
     more per frame.
4. **Resolution: as high as the PC allows.** The size is chosen from a measured pilot and a grid-convergence check, not
   guessed.

## Steps

### 0. Measure the PC

Record the PC's cores, RAM, free disk and CPU features (AVX2, AVX-512, VNNI) in `results/3d/pc.md`. Every size below
scales from these numbers. A GPU, if there is one, stays unused: the code stays C++ on the CPU, as the rest of the
repository is.

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

- **Settings that matter in a game:** burner size, heat release rate, cross-wind, and the fuel's soot yield (a clean
  flame to a sooty one). Perhaps also an obstacle the fire wraps around.
  - Every setting gets several seeds: the inflow or initial perturbation differs, so the runs are different fires with
    the same statistics.
- **Each run:**
  - a warm-up, discarded until the puffing is statistically steady;
  - then 30 to 60 s at 30 frames per second.
- **The held-out split is written in `results/3d/split.md` before the first production run.**
  - It holds settings never trained on and fresh seeds.
  - Seed bases start at 9,000,000. The 7,000,000s are water's and the 8,000,000s are study M's.
  - Splits are by run, never by view or frame.
- **Output per frame:** temperature, soot density, velocity (u, v, w) and heat release rate per unit volume, as 3D
  fields.
- **Disk.** A 256³ grid with six float32 quantities is 402 MB a frame, so 362 GB for 30 s at 30 fps. Keep the full 3D
  fields for a few runs only. For the rest, render and project each output chunk as it is written, then delete it. The
  pilot gives the real numbers.
- **None of it goes in git.** Runs live under the data root.

### 3. New code in the repository (C++, CMake-wired, tests in CI)

- **`nvfx_ingest3d`** reads FDS's 3D output (3D slices or Plot3D files) into a 3D clip format: fields, frame times,
  the setting and the seed.
- **A physically based volume renderer** (`src/render3d`):
  - emission from temperature by Planck's law, integrated against a camera's RGB response;
  - absorption by soot, in proportion to its volume fraction;
  - optional single scattering of a sky and of the fire's own light;
  - one fixed exposure and tone curve per effect;
  - limit: the blue base of a clean flame (chemiluminescence) is not in a soot-and-temperature model; say so wherever
    the pictures are shown.
- **Projections for the 2D model.** For each view (azimuths around the vertical, a few elevations), compute the state
  the rollout model needs:
  - column-integrated soot;
  - emission-weighted temperature;
  - density-weighted velocity in the image plane;
  - the rendered picture.

  Each view of each run becomes a training clip. Views of one run are not independent, which is why splits are by run.
- **Tests:**
  - ingest round trip on a small synthetic file;
  - the renderer against analytic cases: a uniform absorbing slab must obey Beer-Lambert, and a thick uniform hot slab
    must tend to the blackbody;
  - projections of a known field.

### 4. A bigger model

Today's fire model has:
- a 32 x 32 coarse grid;
- a stepper 24 wide, with 4 memory channels;
- a renderer of 16 units;
- pictures of 128 to 256 px.

It trains in hours. A proposed ladder follows; keep each step only if it wins on validation:

- a 64 x 64 coarse grid, a stepper 48 to 64 wide, 8 memory channels, a renderer of 32 units, pictures of 256 and
  512 px;
- what the later studies found helps: 30-frame windows (study L), the statistics loss (study D2) and MacCormack
  advection (study L);
- training for days to weeks, with checkpoints, so it can stop and resume.

Keep the stepper at most 64 channels wide, or extend the int8 path, which takes at most 64 (`src/runtime/rt_rollout.hpp`).

### 5. Measure

Follow the repository's rules:
- write the rules down before the test, and run the test once;
- use held-out settings and seeds;
- use paired bootstrap 95% intervals with 10,000 resamples;
- report ties as ties.

What to measure:
- **Against the 3D teacher's renders:**
  - tracking PSNR at 8 frames, 1 s and 2 s;
  - endless statistics (detail spectrum, motion, coverage, emission, mean frame), as in studies D2 and K.
- **Is it more real than the shipped v2 fire?** The 3D teacher cannot referee that, since it favours its own student.
  Use quantities measured on the models' pictures that real fires fix:
  - flame height and puffing frequency against the correlations of step 1;
  - and, if real fire footage is supplied, study J's statistics against it.

  If there is no real footage, say so.
- **Cost:** milliseconds per frame at 128, 256 and 512 px, on one core, AVX2 and AVX-512, float and int8. Time it beside
  the v2 fire in the same session.

### 6. Ship it, if it wins

If it wins, ship:
- the new fire model in `library/models.nvfxl` and the effect pack;
- frozen keyframes, pinned to float, as the others are;
- demo videos in `videos/`.

The v2 fire stays in the library as the light option. If it does not win, report the result with its numbers, as
every other study does.

## What goes in git

Code, docs, small result tables, figures and the demo videos. Never 3D runs, frames, renders or model weights.
