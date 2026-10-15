# Study N: a small 3D demo

Status: **design and rule committed before validation** (15 October 2026). Audience: dev, research, the owner. The
results, the cost and the videos are added below the rule as they come; nothing above §6 changes after validation.

The owner's plan ([TODO.md](../TODO.md)) is to take every effect to 3D on a big PC later: accurate 3D simulations and
large 3D rollout models. The owner: "we will do a small model here. First we demo and then we go big." This study is
the demo: a small but real, honestly measured 3D learned effect (a fire), playable from any camera, with demo videos.
Everything here runs on the CPU of this shared 4-core cloud machine.

## 1. The question

Does study D's recipe (start points, a learned stepper driven by the simulation's own noise, detail carried by the
learned flow; [REPORT.md](REPORT.md) §6) work with one more dimension? Concretely: does a small 3D rollout model,
trained in a few CPU-hours, follow a held-out 3D fire better than simply running the same simulation on its coarse grid
with the same seeded detail (study D0's baseline), and does it play as realistically?

## 2. The 3D simulator (`include/neuralfx/sim3d.hpp`, `src/sim3d/sim3d.cpp`)

**It is our own solver, tuned by eye. It is not a validated fire code**: accurate 3D teachers checked against measured
fires are the PC plan's job (TODO.md, group A). It is the 3D analogue of `src/sim`: a collocated grid with a one-cell
border; semi-Lagrangian advection (trilinear) for the velocity and a clamped MacCormack correction for temperature and
soot; buoyancy from temperature against soot weight; vorticity confinement (3D: the force is epsilon N x omega); a
seeded 3D curl-noise force (the curl of a vector potential of 4D value noise, `include/neuralfx/noise4.hpp`, shared
with the learned model); combustion expansion (the projection targets a divergence of 1.5 + 1.5 intensity per second
where the gas is hot); pressure projection by red-black Gauss-Seidel with over-relaxation (omega 1.7, 30 sweeps,
warm-started, one projection per step, two steps per frame). A closed floor (no flow through it, slip along it, Neumann
pressure) and an open top and sides (p = 0, outflow). The effect is a fire: a round burner on the floor, its heat, soot
and upward push broken up by a 4D flicker noise; three controls in [0, 1] as the 2D effects have (intensity: the
burner's size and strength; wind: along x, 0.5 none; turbulence: confinement and curl noise). Every loop writes per
cell, so the fields are the same on any number of threads (tested). A smoke preset exists in the code but was not
trained (no time).

### 2.1 The pilot (choosing the grid)

`nvfx_3d pilot`: the fire at controls (0.6, 0.5, 0.5), two seeds (9,000,900 and 9,000,901), at 48, 64 and 80 cells
across in a box 1.5 times as tall, 12 s each (puffing: 20 s each at 48 and 64); 3 threads on the shared machine
(`results/3d/pilot.csv`, `results/3d/pilot_puff.csv`; pilot.csv's puffing columns are a first measure, the heat in a fixed
band of heights, dropped because its peak moved between 0.3 and 1.2 Hz from one half of a run to the other):

| grid | CPU s per frame (3 threads) | plume top at 2 s (of the height) | heat centroid (of the height, 3 to 12 s) | puffing peak (Hz), first / second half / whole |
|---|---:|---:|---:|---|
| 48 x 72 x 48 | 0.11 to 0.15 | 0.50, 0.54 | 0.31 to 0.35 | 0.65 / 0.50 / 0.60; 0.50 / 0.55 / 0.50 |
| 64 x 96 x 64 | 0.21 to 0.34 | 0.53, 0.57 | 0.33 to 0.37 | 0.65 / 0.50 / 0.55; 0.50 / 0.55 / 0.50 |
| 80 x 120 x 80 | 0.45 to 0.52 | 0.54, 0.58 | 0.35 to 0.39 | (not run) |

What was checked, and what it shows:
- **The plume rises**: its top reaches about 55% of the box's height in 2 s on every grid (a 1.5:1 box), and the
  hot gas moves up (a test checks the heat centroid rises and the heat-weighted vertical velocity is positive). Between
  64 and 80 cells the plume top at 2 s moves by 0.01 to 0.02 of the height and the heat centroid by 0.02: close enough
  for a demo, not a converged solution.
- **It puffs at a steady rate**: the visible flame's volume (cells hotter than 0.5, about 1,400 K in the renderer) has
  a clear spectral peak (16 to 18 times the mean power between 0.5 and 6 Hz, Welch with 4 s windows) at 0.50 to 0.65 Hz,
  the same on both seeds, both halves of each run and both grids. That is at the low end of what 4 s windows resolve,
  and it is slow for a real fire: a pool fire 0.3 m across (this burner, if the box is a metre wide) puffs at about
  1.5 / sqrt(0.3 m) = 2.7 Hz. Our fire's puffing is a property of our solver and its forcing, not of real fire.
- **The grid: 64 x 96 x 64**, the expected order, at 0.21 to 0.34 CPU-seconds per frame on this machine; 80 cells
  cost twice as much for little change, 48 leave a 12-cell coarse grid for the model.

## 3. The volume renderer (`include/neuralfx/render3d.hpp`, `src/render3d/render3d.cpp`)

An emission-absorption ray marcher for fields of temperature and soot, used for the simulator's pictures, the learned
model's pictures and the volume C API:
- **Emission** by the hot gas and soot as a grey body at the local temperature: Planck's law integrated over 360 to
  830 nm against the CIE 1931 2-degree colour matching functions (Wyman, Sloan and Shirley's multi-lobe fit, JCGT 2013)
  to XYZ, then to linear sRGB (D65), tabulated every 5 K up to 8,000 K. The emission coefficient equals the absorption
  coefficient (Kirchhoff), so a thick isothermal slab shows the blackbody's colour (tested). The fields' temperature
  maps to 900 + 1,000 T kelvin (the burner's gas, T of 1 to 1.5, is 1,900 to 2,400 K; the flame's visible edge, about
  1,400 K, is T of 0.5). Absorption is soot's (30 per world unit per unit of soot) plus the flame's hot soot (14 per
  world unit times smoothstep(0.3, 1, T)).
- **Single scattering** by soot (albedo 0.5, isotropic) of a fixed key light from above, shadowed by a sweep through
  the volume towards the light, plus an ambient term and the fire's own light (its emitted power as a point light at
  the emission's centre, unshadowed).
- **One fixed exposure and tone curve**: 1 - exp(-L) per channel, then the sRGB transfer curve, for every picture of
  the study and every video.
- Output: premultiplied RGBA (rgb the light the volume adds, alpha one minus its transmittance). A look-at perspective
  camera (or any view and projection matrices), so a camera can orbit the effect. Each pixel is marched alone (steps of
  0.6 cells; empty 4-cell blocks skipped; the last step clipped at the box), so the same pictures come out on any number
  of threads (tested), and nothing is allocated per frame.

## 4. The 3D rollout model (`include/neuralfx/rollout3d.hpp`, `src/core/rollout3d.cpp`, `src/train/rollout3d_train.cpp`)

Study D's model with one more dimension:
- **The coarse state**: 16 x 24 x 16 cells (4 x 4 x 4 blocks of the 64 x 96 x 64 simulation) of velocity (u, v, w in
  coarse cells per frame), temperature and soot, plus 4 memory channels (bounded by tanh).
- **The stepper**, each frame: two 3 x 3 x 3 convolutions (16 channels) with FiLM on the three controls see the state,
  the simulation's own seeded noise at each cell (the three components of the curl force and the flicker, with the run's
  seed), and the position; a 1 x 1 output layer gives the change of every channel and a divergence source. Then a
  pressure projection (40 Jacobi iterations, warm-started from the last frame; closed floor, open top and sides) and
  semi-Lagrangian advection (trilinear) of the state by the projected flow; the physical channels are kept in the range
  seen in training. 14,218 weights (28 KB at fp16).
- **The detail layer** (64 x 96 x 64 fields of temperature and soot): carried by the coarse flow up-sampled
  (trilinear) plus a seeded sub-grid swirl (3D curl noise, 12 units long in a 128-unit width), with clamped MacCormack
  advection; locked to the coarse state block by block (a block holding too much is scaled down; too little is scaled
  up by at most 1.5 and the rest added as new material, broken up by the flicker noise). **Fine detail is seeded, not
  learned**: what is ordered at the coarse scale (the plume, its lean, its puffing) is learned; what is random at the
  scales the coarse grid cannot hold is the seeded noise's (study M's principle, "model what's ordered, leave random to
  random").
- **No learned renderer**: the picture is the volume renderer's (§3) on the fine fields.
- **Start points**: 8 coarse states from training runs (frame 150), spread over the control space by farthest-point
  sampling, stored at 8 bits per value; the fine fields grow from them in a 24-frame warm-up.
- **Training** (CPU only, hand-written gradients checked against finite differences, `tests/test_3d.cpp`): study D's
  three stages: backpropagation through time over windows of up to 12 frames from true states with noise on the inputs
  (900 iterations, the unroll growing over the first half); windows that start after up to 24 frames of the stepper's
  own rollout, with a loss on the heat's and soot's profiles along each axis (350 iterations); the same with a loss on
  how much each channel changes per frame (250 iterations); batch 8, Adam, cosine learning rates. Checkpoints every
  20 iterations; a resumed run is the same to the bit as one that never stopped (tested). (The run began on a schedule
  of 1,200, 500 and 300 iterations and was cut to this one at iteration 140, before anything was evaluated, because
  the shared machine gave it under one core: the stage-1 unroll ramp and learning rate changed from iteration 140 on.)
- **Data**: 40 runs of 8 s (240 frames) from an empty grid at random controls in [0, 1]^3, seeds 9,000,000 + i (5.3 minutes of
  simulated fire).
- **The runtime**: `include/neuralfx/nvfx_volume.h` (§9) runs this reference on buffers made at creation, on its own
  threads, with an AVX2 convolution where the CPU has it; at the baseline ISA its fields equal the reference's to the
  bit, with AVX2 within float rounding (tested).

## 5. The study's protocol (fixed before validation)

**Seeds** (this study owns 9,000,000 to 9,999,999): training 9,000,000 + i; validation tracking 9,400,000 + 100 s + k,
validation real runs 9,410,000 + 100 s + k, validation plays 9,420,000 + 100 s + k; test tracking 9,700,000 + 100 s + k,
test real runs 9,710,000 + 100 s + k, the test's floor (other real runs) 9,710,050 + 100 s + k, test plays
9,720,000 + 100 s + k (s the setting, k the run); the pilot 9,000,900 + k.

**Held-out settings** (intensity, wind, turbulence), never a training run's (training controls are random):
- validation: (0.35, 0.30, 0.40), (0.65, 0.70, 0.60), (0.50, 0.15, 0.75), (0.80, 0.55, 0.25);
- test: (0.30, 0.60, 0.55), (0.45, 0.25, 0.30), (0.55, 0.80, 0.70), (0.70, 0.40, 0.85), (0.85, 0.65, 0.45),
  (0.60, 0.50, 0.20).

**The methods compared**, each with the same detail layer and constants and the same renderer:
- **learned**: the 3D rollout model;
- **coarse simulation + detail** (the baseline, study D0's): the 3D simulator itself run on the coarse grid
  (16 x 24 x 16, its lengths and forces scaled to the grid as the simulator scales them), its velocity, heat and soot
  driving the same seeded detail layer;
- for reference only: **holding the first frame** (tracking) and **another set of real runs** (the floor of the endless
  statistics, test only).

**Tracking from the true state.** A run at a held-out setting is simulated at 64 x 96 x 64 from an empty grid for 100
frames; its state (the coarse state and the full fine fields) is the start of every method, which then plays one
continuous rollout with the run's own noise seed for 60 frames. At 8 frames, 1 s (30) and 2 s (60) against the
simulation's own continuation:
- **field PSNR** of temperature (peak 1.5) and of soot (peak 0.2) over the fine grid, active cells only (either field
  above 1% of the peak);
- **picture PSNR** from four cameras around the effect (azimuths 0, 90, 180 and 270 degrees, 2.2 world units away, a
  42-degree vertical field of view, aimed at 0.63 of the box's height; 128 x 128 pixels; active pixels, as study D's
  active PSNR), the mean over the four.
Validation: 3 runs at each of the 4 settings (12); test: 3 at each of the 6 (18). Paired over runs.

**Endless statistics** (study D2's pooled protocol). Per setting R real runs (simulated from an empty grid, 150 frames
of warm-up, then 240 frames) and R plays of each method (from the start point nearest the setting's controls, with the
play's seed: the learned model as `start` does it, a 24-frame warm-up of the fine fields, then 240 frames in one
rollout; the coarse simulation from the same start state with the same warm-up of the same detail layer). Every frame is
drawn by the four cameras at 128 x 128. Per run: per camera the coverage (mean alpha) and emission (mean max(0, rgb -
alpha)) per frame, the motion (mean absolute difference between consecutive frames) and the mean picture; and every 4th
frame the 3D power spectrum of the fine heat and of the fine soot over the lowest 64 x 64 x 64 cells, radially averaged
(32 bins). Pooled per setting over the R runs (curves, motions, mean pictures and spectra averaged), then against the
real runs' pooled statistics:
- **3D spectrum distance**: the mean over the bins of |difference of log10 power|, the mean of heat's and soot's;
- **|ln motion ratio|**: the motion summed over the cameras, method over real;
- **coverage and emission distances**: the mean |difference| of the curves, the mean over the cameras;
- **mean picture PSNR**: of the mean pictures, the mean over the cameras.
Validation: R = 2 at each of the 4 settings; test: R = 3 at each of the 6. Paired over settings.

**Intervals**: paired bootstrap of the mean difference, 10,000 resamples, seed 2026, 95% percentile intervals
(`metrics::paired_bootstrap`). An interval covering zero is a tie.

**Candidates** (built on training data only): **A**, the stepper after stages 1 and 2; **B**, after stage 3 (the
activity loss). The same start points, detail constants (contrast 1, swirl 0.5, swirl length 12, grow 1.5) and
renderer.

**Validation (the choice).** Both candidates and the baseline are scored on validation (§5's tracking and endless
statistics). The candidate with the higher mean picture PSNR at 1 s in tracking is chosen; if the two are within
0.1 dB, the one with the lower mean 3D spectrum distance. Its file is frozen by SHA-256 in `results/3d/frozen.csv`
before the test; `nvfx_3d` refuses to run the test's learned plays on any other file. Validation's comparison with the
baseline is reported but chooses nothing: the test runs whatever validation shows.

**The rule (the test runs once).** The chosen model minus the coarse simulation with the same detail, on the test:
1. **Tracking**: it *follows a real run better* if, at 1 s, the temperature field PSNR and the picture PSNR (four
   cameras) both have intervals entirely above zero; it *follows worse* if both are entirely below zero; otherwise it
   is a tie (or mixed) at tracking, reported as such.
2. **Look**: it *plays as realistically* if no endless statistic is worse: no interval entirely above zero for the 3D
   spectrum distance, |ln motion ratio|, the coverage distance and the emission distance, and none entirely below zero
   for the mean picture PSNR.
3. **The verdict**: "**the learned 3D fire beats running the cheap simulation**" if 1 (better) and 2 hold; "it follows
   real runs better but looks worse in ..." if 1 (better) holds and 2 fails (naming the statistics); otherwise "**it does
   not beat running the cheap simulation**", with every number. The soot field, 8 frames and 2 s, holding the first
   frame and the floor are reported beside, not judged.

**Cost** (reported, not judged; provisional on the shared machine): milliseconds per frame for a step and for a
picture at 256 x 256 and 512 x 512, through the volume C API, on 1 thread and on 2 to 4.

## 6. Validation

(To come.)

## 7. The test

(To come.)

## 8. Cost

(To come.)

## 9. The volume C API

`include/neuralfx/nvfx_volume.h`, in libnvfx (static and shared; `NVFX_VERSION` 6, feature bit `NVFX_FEATURE_VOLUME`):
load a 3D effect (`nvfx_volume_effect_load`, `_load_memory`), create a volume with its own threads, set the seed, the
controls, the variation (a start point) and the shard length, step to a time, read the fields (temperature, soot,
extinction, velocity) as float or half 3D textures, and draw it for a camera (view and projection matrices, or a
look-at description) into premultiplied RGBA8. How an engine uses it: [ENGINES.md](ENGINES.md) §12; struct sizes:
[INTEGRATION.md](INTEGRATION.md) §8. What is tested (`tests/test_3d.cpp`, `tests/alloc_test.cpp`, `tests/abi_*`):
- parity with the reference (`src/core/rollout3d.cpp`): equal to the bit at the baseline ISA, within float rounding
  with the AVX2 convolution;
- the same fields and pictures on 1 and 3 threads, through a shard change and its crossfade;
- every frame a function of the time (stepping back in time and forward again gives the same fields);
- no allocation per frame on 1 and 3 threads, through shard changes, new seeds, variations, restarts, every field and
  texel format and a moving camera (`nvfx_alloc_test`);
- the structs' sizes from C99 and C++ without exceptions (`nvfx_abi_test`), the error paths.

## 10. Demo videos

(To come.)

## 11. Limits

- **The teacher is our own solver, tuned by eye.** Nothing here is checked against a real fire: the plume rises and
  puffs at a steady rate, but at 0.5 to 0.65 Hz, several times slower than a real pool fire of this size would. The
  learned model can only be as real as its teacher. The PC plan's validated teachers (FDS) are what would change that.
- **One effect, one small model.** A fire, with a 16 x 24 x 16 learned state, 16 channels and 14,218 weights, trained
  for about two thousand iterations on 40 runs (5.3 minutes of simulated fire). The smoke preset exists but was not
  trained. No hyperparameter was searched.
- **Small samples.** The test pairs 18 tracked runs and 6 settings (3 runs pooled per setting); study D2 pooled 8.
  Intervals over 6 settings are wide.
- **The picture is a fixed physical model, not a learned one**: emission and absorption with single scattering of one
  fixed key light, an ambient light and the fire's own light without shadows. No multiple scattering, no blue
  chemiluminescent base (a soot-and-temperature model has none), one exposure for every picture.
- **The measures see the fields at 64 x 96 x 64 and pictures at 128 x 128 from four fixed cameras**; the 3D spectrum
  covers the lowest 64 cells of the box only.
- **Cost is provisional**: measured on a shared 4-core machine at a load of 8 to 14, where nice-15 jobs get under a
  core.

## 12. Reproduce

Data under `NEURALVFX_DATA/n_3d` (default `/root/nvfx-data/n_3d`); every step resumes from what it has written (the
job scripts used are in the data root's `jobs/`). Build as the main README says, then:

```sh
B=build/nvfx_3d
$B pilot --widths 48,64,80 --seconds 12 --seeds 2 --threads 3       # §2.1 (results/3d/pilot.csv)
$B pilot --widths 64,48 --seconds 20 --seeds 2 --threads 3          # §2.1's puffing (results/3d/pilot_puff.csv)
$B record --runs 40 --frames 240 --threads 3                        # the training runs (seeds 9,000,000 + i)
$B train --name base --iterations 900 --finetune 350 --activity 250 --batch 8 --unroll 12 --burn 24 --threads 4
$B finish --ckpt base_stage2 --out candA && $B finish --ckpt base --out candB
$B truth --split val && $B real --split val                         # the validation's simulations
for m in A B; do $B track --split val --method $m --model $NEURALVFX_DATA/n_3d/models/cand$m.nvfx3; done
$B track --split val --method coarse --model .../candA.nvfx3 ; $B track --split val --method hold --model .../candA.nvfx3
for m in A B; do $B play --split val --method $m --model .../cand$m.nvfx3; done; $B play --split val --method coarse --model .../candA.nvfx3
$B compare --split val --a A --b coarse                             # results/3d/compare_val_*.md
$B freeze --model .../chosen.nvfx3                                  # results/3d/frozen.csv, before the test
$B truth --split test && $B real --split test                       # the test's simulations (and the floor)
$B track --split test --method learned --model .../chosen.nvfx3 ; $B track --split test --method coarse --model .../chosen.nvfx3
$B play --split test --method learned --model .../chosen.nvfx3 ; $B play --split test --method coarse --model .../chosen.nvfx3
$B compare --split test --a learned --b coarse                      # once
$B timing --threads 1,2,4                                           # §8 (results/3d/timing.csv)
$B video --kind orbit --out 3d_fire_orbit.mp4 --seconds 12          # §10
```

## 13. What it means for the PC plan

(Completed after the test.)
