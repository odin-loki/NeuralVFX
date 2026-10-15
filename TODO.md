# To do on a PC: every effect in 3D, learned from accurate 3D simulations

**Status: planned, not started.** Nothing in this file has been run. It is meant for a desktop PC, much later. On that
PC run time does not matter: a simulation may take days and training may take weeks.

## The goal

Go straight to 3D, for every effect in the repository:
- simulate each effect in 3D as accurately as the PC can, however long it takes;
- train a 3D rollout model on those runs, as large and as high resolution as the PC allows;
- play it in 3D: a volume that moves and responds the same from any camera, for FUSE;
- then rebuild the scenes and the library on top of the 3D effects.

Every effect plays as today's fire does: from start points, with a learned stepper driven by seeded noise. It is
pseudorandom but deterministic, plays for as long as it is wanted, and covers a vast range of settings, only with a 3D
state. What is ordered is learned. What is random stays seeded random: spray, sparks, foam, droplets and fine flicker.

Today's effects are 2D sprites learned from `src/sim`, a 2D solver whose look was tuned by eye. They play close to that
teacher (studies D, D2, K and M), so the teacher is the ceiling on realism. 3D teachers that are checked against
measured physics raise the ceiling. 3D models remove the sprite's limit: the effect can be walked around.

## The effects

The 21 learned effects, plus the water system (procedural today), grouped by the physics that makes them, since each
group shares one simulator and one kind of check. The vortex becomes two 3D effects: a fire whirl and a dust devil.

| group | effects | simulator (run outside the repository) |
|---|---|---|
| A. Fire and smoke | fire, smoke, wildfire, coals, puffs, jet (flamethrower), vortex (fire whirl) | NIST's Fire Dynamics Simulator (FDS): LES, low-Mach, combustion, soot, radiation, vegetation |
| B. Blasts and rockets | explosion, mushroom cloud, launch (rocket exhaust on the ground) | PeleC (compressible reacting flow with adaptive mesh refinement), handing over to PeleLM (low-Mach) once the shocks have gone |
| C. Air and water vapour | cloud, fog, steam, rainshaft | an atmospheric LES with moist thermodynamics and microphysics: PALM or MicroHH |
| D. Particles in air | dust, vortex (dust devil) | the same LES with Lagrangian particles: settling, two-way coupling |
| E. Liquids | waterfall, fountain, splash, ink, and the water system's pieces (sheets, jets, drips, bubbles, pools) | Basilisk: adaptive octree, volume of fluid, surface tension |
| F. Designed (no real counterpart) | magic, orb | a real 3D fluid (Basilisk single-phase, or our own solver) with designed sources and forces, as the 2D presets are |

The compositor's procedural parts are not learned, and they move to 3D as they are:
- particles: sparks, embers, debris, flakes, rain, snow, ash, leaves, hail, spatter, shards and streaks;
- emitters: hail, aurora, rainbow, meteors, beams, shields and glints;
- force fields.

They are covered in "3D scenes and the 3D library" below.

## Decisions (proposed; settle them before starting)

1. **Accuracy first, checked against measured physics.**
   - Each group uses an established, validated simulator, run as an external program. Nothing of the simulators
     enters the repository: only our case files (inputs) and C++ readers for their output.
   - Every simulation is checked against measurements before any training.
   - Writing our own solvers is the fallback, for when an established one cannot give what we need.
2. **One 3D rollout model for every group.**
   - The state: a coarse 3D grid of velocity (u, v, w), the group's own scalars (below) and memory channels.
   - The stepper: learned 3 x 3 x 3 convolutions with FiLM on the controls.
   - Advection: semi-Lagrangian or MacCormack (study L).
   - A 3D pressure projection.
   - The forcing: seeded 3D noise.
   - Fine detail: a seeded 3D detail layer carried by the coarse flow, as study M splits ordered from random.
   - Grids need not be cubes: fog is flat, a waterfall tall and a wildfire wide.
3. **The groups' scalars.**
   - A and F: temperature and soot.
   - B: temperature, soot and pressure (the blast, while it lasts).
   - C: temperature, water vapour, cloud water and rain.
   - D: temperature and dust.
   - E: the liquid fraction (or a level set), plus foam and spray densities.
4. **What the runtime gives an engine: both of these.**
   - **The volume itself**, for the engine's own renderer, which is the natural fit for FUSE (the GPU draws and
     lights it; the CPU only steps it):
     - gases as 3D textures: extinction, emission, scattering albedo and optionally velocity;
     - liquids as a level-set volume (to mesh with marching cubes or to ray-march with refraction), plus a list of
       spray and foam particles.
   - **A picture for a camera**, drawn on the CPU, for engines without volumetrics, for tests and for videos.
5. **Resolution: as high as the PC allows**, chosen per group from a measured pilot and a grid-convergence check, for
   the simulations and the models alike.

## The pipeline every effect goes through

### 0. Measure the PC (once)

Record the PC's cores, RAM, free disk and CPU features (AVX2, AVX-512, VNNI) in `results/3d/pc.md`. Every size scales
from these numbers. The library stays C++ on the CPU. The engine's GPU only draws the volumes it is given.

### 1. Pilot: prove the simulation is right before training anything

- **One case** of the effect, at three resolutions, each twice as fine as the last.
- **Grid convergence.** The group's checks (in the group's section below) must stop moving between the two finest
  grids, or the production grid must be finer still.
- **Acceptance bands are written in `results/3d/<group>/validation.md` before the runs.** Pass if each check falls
  within the measurements' own scatter. If one fails, fix the simulation before anything else.
- **Cost.** Measure the memory per cell and the time per cell-step. The production grid is the largest that fits in
  80% of RAM at an acceptable time per run.

### 2. Production runs

- **Settings that matter in a game**, per effect (in the group's section below), and several seeds per setting.
- **Each run:**
  - looping effects: a warm-up, discarded until the statistics are steady, then 30 to 60 s at 30 fps;
  - one-shots: from ignition to the end.
- **Held-out split.** Each effect's split goes in `results/3d/<effect>/split.md` before its first production run.
  - It holds settings never trained on and fresh seeds.
  - Seed bases are 9,000,000 + 100,000 x the effect's number. The 7,000,000s are water's and the 8,000,000s are
    study M's.
  - The numbers:

    | group | effects (number) |
    |---|---|
    | A | fire 0, smoke 1, wildfire 2, coals 3, puffs 4, jet 5, fire whirl 6 |
    | B | explosion 7, mushroom 8, launch 9 |
    | C | cloud 10, fog 11, steam 12, rainshaft 13 |
    | D | dust 14, dust devil 15 |
    | E | splash 16, fountain 17, waterfall 18, ink 19, water system 20 |
    | F | magic 21, orb 22 |
  - Splits are by run, never by frame.
- **Disk.**
  - A 256³ grid with six float32 quantities is 402 MB a frame, so 362 GB for 30 s at 30 fps.
  - Ingest each output chunk as it is written, then delete it. Keep the coarse grids, the fine fields the detail layer
    and the start points need, and 16-bit storage.
  - Keep full-resolution fields for a few runs only, for pictures and checks.
- **None of it goes in git.** Runs live under the data root.

### 3. Ingest

There is one reader per simulator, all into one 3D clip format: fields, frame times, the setting and the seed. The
format has block averages for the coarse grid and fine fields at the detail layer's size.

| simulator | its output |
|---|---|
| FDS | 3D slice or Plot3D files |
| PeleC and PeleLM | AMReX plotfiles |
| PALM and MicroHH | NetCDF, through libnetcdf's C API |
| Basilisk | a small output routine of ours in the case file, writing our raw format |

### 4. Train

- Start at a 32³ coarse grid, or the same number of cells in the effect's shape. Use a stepper 24 to 32 wide, 4 memory
  channels and fine detail at 128³.
- Then climb, keeping each step only if it wins on validation: a 48³ or 64³ grid, a stepper 48 to 64 wide, 8 memory
  channels, fine detail at 256³.
- Keep the stepper at most 64 channels wide, or extend the int8 path, which takes at most 64
  (`src/runtime/rt_rollout.hpp`).
- Use the statistics loss (study D2) and long windows (study L).
- Training runs for days to weeks, with checkpoints, so it can stop and resume.

### 5. Measure

Follow the repository's rules:
- write the rules down before the test, and run the test once;
- use held-out settings and seeds;
- use paired bootstrap 95% intervals with 10,000 resamples;
- report ties as ties.

What to measure:
- **Against the simulation, in 3D:**
  - tracking from the true state, as field errors and as pictures from several cameras, at 8 frames, 1 s and 2 s;
  - endless statistics, as in studies D2 and K: the 3D detail spectrum, motion, coverage, emission or extinction, and
    the mean volume.
- **Against reality.** The model's own runs are measured with the group's physical checks, exactly as the simulation's
  were. A 3D model allows this, where a sprite does not. If real footage is supplied, also compare pictures from a
  matching camera with study J's statistics. If there is none, say so.
- **Against the 2D effect from the front:** today's sprite beside the 3D effect drawn from the same side.
- **Cost:** milliseconds per frame for the step and for the picture, separately, at each grid size, on 1 and 4
  threads, AVX2 and AVX-512, float and int8. Also the memory per instance.

### 6. Ship it, if it wins

If it wins, ship:
- the 3D effect in the effect pack;
- frozen keyframes, pinned to float;
- demo videos in `videos/`.

If it does not win, report the result with its numbers, as every other study does.

## The groups

### A. Fire and smoke (FDS) — first

FDS is public domain, validated against decades of fire experiments, and runs on CPUs with MPI. Its resolution is
measured as D*/dx, the plume's characteristic diameter over the cell size:
- D* = (Q / (rho_inf c_p T_inf sqrt(g)))^(2/5), where Q is the heat release rate;
- FDS's own guidance treats D*/dx of 4 to 16 as coarse to moderate;
- aim for at least 20, and as high as RAM allows.

| effect | the case | settings | checked against |
|---|---|---|---|
| fire | a square pool burner of propane or methane, 0.3 to 1 m across | burner size, heat release rate, cross-wind (speed and direction), soot yield (clean to sooty) | mean flame height against Heskestad's L = 0.235 Q^(2/5) - 1.02 D (Q in kW, L and D in m); puffing frequency against f = 1.5 / sqrt(D) Hz; centreline temperature and velocity against McCaffrey's plume correlations |
| smoke | a smouldering or smoke-pot source, and a fire's plume far above its flames | source strength, heat, wind, soot yield | plume rise and spread against plume theory (Morton, Taylor and Turner); the light it blocks from soot's mass extinction coefficient, about 8.7 m²/g for flaming soot (Mulholland and Croarkin) |
| wildfire | a grass fire spreading across a strip of vegetation (FDS's vegetation and level-set models) | wind, fuel load, moisture, slope | rate of spread and fire-front depth against the CSIRO grassland fire experiments, as the physics-based grassland fire simulations of Mell and colleagues were checked |
| coals | a bed of charring fuel: the solid's char oxidation, glowing surface, low flames and hot air above | bed size, air flow (a draught or a bellows), how far it has burned down | surface temperatures and their flicker against infrared measurements of glowing ember beds; heat release against the bed's burning rate |
| puffs | hot gas from a chimney or an exhaust, released in pulses | pulse rate, strength, wind | puff (thermal) rise and growth against Scorer's thermal experiments and thermal theory |
| jet | a flamethrower: a burning stream of liquid fuel (Lagrangian droplets) | flow rate, aim, wind | turbulent jet flame length against Delichatsios's correlations; a gas jet flame against the Sandia piloted flame D data (TNF workshop) before the liquid stream |
| vortex (fire whirl) | a fire in a rotating flow (corner gaps or rotating screens) | circulation, heat release rate | flame height against circulation, from laboratory fire-whirl experiments |

The light: emission from temperature by Planck's law, integrated against a camera's RGB response, and absorption by
soot in proportion to its volume fraction. The blue base of a clean flame (chemiluminescence) is not in a
soot-and-temperature model; say so wherever the pictures are shown.

### B. Blasts and rockets (PeleC, then PeleLM)

| effect | the case | settings | checked against |
|---|---|---|---|
| explosion | a charge or a fuel-air cloud igniting near the ground | energy, height, fuel (clean to sooty) | blast overpressure and impulse against distance by the Kingery-Bulmash charts (TNT equivalent); fireball size and duration against fireball scaling |
| mushroom cloud | a large burst's fireball rising through a stratified atmosphere (compressible for the first fraction of a second, then low-Mach) | energy, stratification, wind | stem and cap rise against buoyant-thermal theory in a stratified atmosphere (Morton, Taylor and Turner); the toroidal cap's roll-up |
| launch | an under-expanded rocket exhaust hitting a pad or the ground and billowing both ways (afterburning; optionally a water deluge's steam) | thrust, nozzle pressure ratio, height above the ground | Mach disk location against x/D = 0.67 sqrt(p0 / pa) (Crist, Sherman and Glass); wall pressure of impinging jets against measurements |

The hand-over from compressible to low-Mach maps the fields at a fixed time, after the shock has left the domain. The
learned model sees only the low-Mach part. The blast's shock ring, a few frames long, is drawn by the compositor, as
today's shock fronts are.

### C. Air and water vapour (PALM or MicroHH)

| effect | the case | settings | checked against |
|---|---|---|---|
| cloud | shallow cumulus over a warm sea (the BOMEX case), then single clouds at a finer grid | surface heat and moisture fluxes, wind shear | cloud fraction, liquid water and cloud base and top against the BOMEX LES intercomparison (Siebesma and colleagues) |
| fog | radiation fog forming in a calm, cooling night, and advection fog rolling in | cooling rate, wind, humidity | liquid water and fog depth against a measured radiation-fog case, as published LES fog studies were checked |
| steam | a warm, moist plume condensing as it rises and evaporating at its edges (a kettle, a vent, a cooling tower, a hot spring) | heat, moisture, air temperature and humidity, wind | plume rise against plume theory; where the visible plume ends against saturation by mixing |
| rainshaft | rain falling from a cloud base, its evaporation, and the cold outflow it drives | rain rate, air humidity, wind | drop sizes against Marshall-Palmer; fall speeds against Gunn and Kinzer; the outflow's front speed against gravity-current theory (Benjamin) |

The light: scattering by droplets, with a Mie phase function and multiple scattering, which matters for clouds and fog,
plus absorption.

### D. Particles in air (the same LES, with Lagrangian particles)

| effect | the case | settings | checked against |
|---|---|---|---|
| dust | a burst of dust from an impact or a vehicle, spreading along the ground and settling | energy, particle size, wind | settling against measured terminal velocities; the spreading front against particle-laden gravity-current experiments |
| vortex (dust devil) | dust devils forming in a convective boundary layer, as LES finds them (Raasch and Franke), lifting dust | surface heating, background wind | core radius, tangential velocity profile and pressure drop against field measurements of dust devils |

### E. Liquids (Basilisk)

Basilisk resolves splashes, jets and breaking water with an adaptive octree, volume of fluid and surface tension. It
has been validated on drop impacts, jets and breaking waves.

| effect | the case | settings | checked against |
|---|---|---|---|
| splash | a drop or an object hitting water or a wet surface: crown, jets, droplets | impact speed, size, depth | splashing threshold against Mundo, Sommerfeld and Tropea's K = Oh Re^1.25; crown and Worthington jet heights against drop-impact experiments |
| fountain | a jet thrown up and breaking into drops, falling into a pool | flow rate, aim, nozzle | trajectory against ballistics with drag; break-up length against jet break-up measurements (Rayleigh-Plateau) |
| waterfall | a sheet or column falling, breaking up, entraining air and plunging into a pool with spray and mist | flow rate, height, width | break-up length of falling jets; plunge-pool air entrainment and spray against measurements |
| ink | a slightly dense plume in still or moving water (ink, blood, silt); high Schmidt number | density difference, release rate, current | spreading and entrainment against plume theory; Rayleigh-Taylor mixing growth against measurements (the alpha group, Dimonte and colleagues) |
| water system | sheets, jets, drips, bubbles and pools (the composer's water kinds today) | as each | as splash and fountain; bubble rise against measured terminal velocities |

**The learned part:** the liquid fraction or a level set on the coarse grid, with velocity.

**The seeded random part:** spray droplets, foam and bubbles as particles, which study M's split suits.

**The light:** surfaces need reflection and refraction, and the engine draws them. For tests and videos, our renderer
sphere-traces the level set with Fresnel reflection and refraction against an environment map. A learned stepper may
smear a sharp surface, so measure that first, on the splash.

### F. Designed effects (magic, orb)

There is no real counterpart to measure.
- The 3D versions are a real fluid with designed sources and forces, as the 2D presets are: magic's swirling portal
  and the orb's boiling, pulsing ball.
- They are checked against grid convergence and the design's own intent only, and they say so.
- They come last. Study L showed fast, regular swirls are hard for a learned stepper, so learn from groups A to E
  first.

## Order of work

1. **Shared code.** The 3D clip format and FDS reader, the 3D rollout model and trainer, and the volume renderer for
   emission and absorption. Then the volume API (`include/neuralfx/nvfx_volume.h`), a new C API beside the effect and
   scene APIs:
   - create an instance from a 3D effect file, with a seed, controls and a variation;
   - step it to a time;
   - read the volume as 3D textures (R8, R16F or R32F), or draw it for a camera given as view and projection matrices
     into RGBA;
   - per-ISA SIMD for the 3 x 3 x 3 convolutions and the int8 path;
   - threads, no allocation per frame, and shards so it plays forever;
   - the ABI test, `INTEGRATION.md`'s sizes and `NVFX_VERSION` updated;
   - a Godot demo (a FogVolume or a 3D texture) and an example host.

   The tests:
   - ingest round trips;
   - the renderer against Beer-Lambert and blackbody limits;
   - the 3D stepper's gradients against finite differences;
   - runtime parity with the trainer's reference on every ISA, float and int8;
   - the same frames on any number of threads;
   - no allocation per frame.
2. **Group A**, fire first. It proves the whole pipeline, with the most validation data behind it.
3. **Group C**, then **D**. These need the atmospheric LES, its NetCDF reader, and scattering in the renderer.
4. **Group B.** This needs PeleC and PeleLM, the AMReX reader, and the hand-over.
5. **Group E.** This needs Basilisk, the level-set state, particles for spray and foam, and the surface renderer.
6. **Group F.**
7. **3D scenes and the 3D library** (below), started once groups A and C have shipped effects to compose.

## 3D scenes and the 3D library

- **The scene language in 3D:**
  - positions, sizes and wind as 3D vectors, and a camera;
  - modules as volumes placed in the world;
  - the field bus as a 3D grid, with couplings (pushes, transfers, hand-overs, shocks) in 3D;
  - force fields in 3D.
- **The procedural parts in 3D:**
  - particles in 3D: sparks, embers, debris, flakes, rain, snow, ash, leaves, hail, spatter, shards and streaks;
  - emitters: aurora, rainbow, meteors, beams, shields and glints as 3D shapes;
  - the water system's strands, drops, crowns, pool waves, foam, rings, bubbles and shafts in 3D.
- **The library rebuilt in 3D:** today's 149 entries, plus round 7's and the water entries, made again from the 3D
  effects, each checked from several cameras.
- **The effect manager for volumes:** budget, level of detail by grid size and step rate, sharing and proxies.
- **A 3D scene C API** for engines, as `nvfx_scene.h` is for 2D. FUSE gets volumes, particles and lights per frame.

## What it costs to play (arithmetic from today's 2D numbers; to be measured)

The 2D stepper works on 32 x 32 cells with 3 x 3 convolutions. Its network stage takes about 0.17 ms per frame on one
core at float, and 0.10 ms at int8 with VNNI (REPORT.md §13.8).

**A 32³ stepper** has 32 times the cells and 3 times the taps, so about 100 times the work:
- about 16 ms per frame on one core at float, and 10 at int8;
- about 3 to 4 ms on four cores.

That is playable for one hero effect, or for a few with sharing and level of detail.

**A 64³ stepper** is 8 times that again, about 130 ms on one core: for baking, cutscenes or hero shots, unless int8 and
threading bring it down.

**Other costs:**
- the 3D pressure projection: a few ms;
- the detail layer: it scales with the fine grid's size;
- ray-marching a picture on the CPU: about 6 million samples at 256 x 256 px and 96 steps a ray.

With the engine's GPU drawing the volumes, the CPU pays only the step. Many distant effects can share instances, or
fall back to today's 2D sprites, which stay in the library as the light option.

## What goes in git

These go in:
- code;
- the simulators' case files (small text inputs) under `sim3d/<simulator>/<effect>/`, so every run can be repeated;
- docs, small result tables and figures;
- the demo videos.

These never go in: the simulators themselves, 3D runs, frames, volumes, renders or model weights.
