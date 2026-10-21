# Study N: a small 3D demo

Status: **done** (15 October 2026). Audience: dev, research, the owner. The design and the rule (§1 to §5) were
committed before validation and did not change; the results, the cost and the videos follow them (§6 to §10).

**In short.** A 3D fire from our own 3D solver (not a validated fire code), learned as a 16 x 24 x 16 state that drives
seeded 64 x 96 x 64 detail, drawn by a physically based volume renderer and played through a volume C API. On the
held-out test, run once, it **follows a real run better than running the simulation on the coarse grid with the same
detail** (at 1 s: temperature +1.08 dB [+0.74, +1.39], picture from four cameras +2.20 dB [+1.09, +3.12]) **but looks
worse when played on**, in how much it moves (|ln motion ratio| +0.229 [+0.063, +0.402]) and how much it covers (+0.0049
[+0.0006, +0.0116]); so by the rule it does not beat the cheap simulation outright. Holding the first frame still beats
it after a second. When the wind blows towards -x (0.3 or below) it runs away (§11), which is most of what it does
worse. A step costs about 70 ms and a 256 x 256 picture about 50 ms on one busy core (provisional), most of it the
detail layer (study N2's model, which the demos play, re-timed on a quiet machine with the detail layer on vectors: 21
ms and 56 ms, §14.12). Smoke was trained as a demo only (§10), not validated or tested.

**Study N2** (§14) fixed the runaway: the model's advection now brings in ambient air at the open sides and top, and the
same small model retrained with it and with long rollouts has no runaway over 30 s at any wind, and on a new held-out
test it tracks better than study N's model (+1.95 dB in pictures at 1 s) and plays better on four of five endless
statistics.
By its rule, written before validation, it still did not pass (its edge check failed on validation by 3% on one side),
so study N's model and videos remain the demo.

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

**A check before validation (probe split, not validation data).** A probe of the stage-1 stepper (iteration 920) looked
dimmer than the simulation (smaller flames): emission is so steep in temperature that block averages lose the hottest
spots. Whether other detail constants would help was checked on a probe split of its own (the controls of training
runs 0 to 3, seeds 9,300,000 + 100 s + k, 2 runs each, tracking only; `results/3d/probe_detail.csv`), means over the 8
runs:

| variant | temperature field, 8 frames / 1 s / 2 s (dB) | picture, 8 frames / 1 s / 2 s (dB) |
|---|---|---|
| stage-1 stepper, the registered constants (grow 1.5, contrast 1) | 22.93 / 20.47 / 19.82 | 23.30 / 20.05 / 18.91 |
| the same with grow 3 | 22.04 / 19.50 / 18.73 | 21.07 / 18.32 / 17.09 |
| the same with grow 3 and contrast 0.5 | 22.06 / 19.50 / 18.69 | 21.11 / 18.33 / 16.98 |
| coarse simulation + detail (registered constants) | 20.36 / 18.75 / 17.90 | 17.67 / 16.24 / 15.80 |
| holding the first frame | 22.24 / 20.61 / 20.49 | 21.53 / 19.43 / 19.47 |

The registered constants tracked best, so nothing in §5 changed. (Already on the probe the stepper led the coarse
simulation and was close to holding the first frame.)

**Candidate A had to be made again.** The trainer saved its stage-2 snapshot only at checkpoint iterations (every 20),
and stage 2 ended at iteration 1,250, which is not one, so A's file was never written (fixed: the hook now also runs at
the end of every stage). Training is deterministic and resumes to the bit (tested), so iterations 920 to 1,250 were
run again from the iteration-920 checkpoint with the same options; the losses logged at 920, 930, 940 and 950 equal
the original run's to every printed digit (`logs/train_regen.log` against `logs/train_base.log` in the data root).

**Validation** (`results/3d/compare_val_*.md`, per-run rows in `results/3d/val_track/`): 12 tracking runs, 4 settings
with 2 real runs and 2 plays each pooled. Means, and differences with 95% intervals:

| measure | A | B | coarse simulation + detail | holding the first frame | A - B | A - coarse |
|---|---:|---:|---:|---:|---|---|
| tracking, temperature field, 1 s (dB) | 19.57 | 19.31 | 18.31 | 20.38 | +0.26 [+0.12, +0.39] | +1.26 [+0.33, +2.14] |
| tracking, soot field, 1 s (dB) | 25.03 | 24.84 | 19.56 | 24.86 | +0.19 [+0.07, +0.31] | +5.47 [+4.71, +6.21] |
| tracking, picture, 8 frames (dB) | 21.32 | 20.81 | 18.45 | 19.65 | +0.51 [+0.29, +0.76] | +2.87 [+2.43, +3.28] |
| **tracking, picture, 1 s (dB)** | **17.37** | **17.08** | 14.78 | 18.06 | **+0.29 [+0.01, +0.58]** | +2.60 [+1.02, +4.14] |
| tracking, picture, 2 s (dB) | 17.20 | 17.51 | 14.67 | 18.00 | -0.31 [-0.84, +0.10] | +2.53 [+1.38, +3.63] |
| endless: 3D spectrum distance | 0.842 | 0.358 | 0.778 | | +0.484 [-0.009, +0.977] | +0.064 [-0.454, +0.582] |
| endless: \|ln motion ratio\| (motion ratio) | 0.890 (1.73) | 0.491 (0.83) | 0.233 (1.10) | | +0.399 [+0.058, +0.740] | +0.657 [+0.527, +0.786] |
| endless: coverage distance | 0.0262 | 0.0054 | 0.0037 | | +0.0208 [-0.0000, +0.0417] | +0.0225 [+0.0011, +0.0440] |
| endless: emission distance | 0.0270 | 0.0049 | 0.0136 | | +0.0221 [+0.0002, +0.0439] | +0.0133 [-0.0115, +0.0382] |
| endless: mean picture PSNR (dB) | 22.62 | 27.50 | 23.55 | | -4.87 [-10.09, +0.34] | -0.93 [-6.46, +4.61] |

- **The choice, by the rule: A** (the stepper after stage 2), with the higher picture PSNR at 1 s (17.37 against 17.08
  dB, more than 0.1 dB apart). Its file was frozen before the test (`results/3d/frozen.csv`: 274,756 bytes, SHA-256
  ac94e05b189d17a8673044799a18e3494bb6347d438cf76d80115b9e182443e6) and is `models/chosen.nvfx3` in the data root.
- **The rule chose the candidate that plays worse.** B (stage 3's activity loss) plays much more like the simulation:
  its spectrum distance is less than half A's, its motion 0.83 of the real runs' where A's is 1.73, and its mean picture
  4.9 dB closer. A moves too much and spreads too wide when it plays on; the per-setting rows (added to the
  comparison tables after the test, for the write-up) put all of that at the two lowest winds, where A runs away
  (§11), and show A and B alike at the other two. The rule looked only at tracking at 1 s, where A leads by 0.29 dB
  with an interval that just clears zero. Choosing on tracking was a poor rule for a model that will mostly be played
  endlessly; it stands, and the test runs A.
- **Against the baseline on validation**, both candidates follow a real run better than the coarse simulation at every
  horizon, by 1 to 6.6 dB. Played endlessly, A is worse in motion and coverage and ties the rest; B is better in the
  spectrum, emission and mean picture, worse in motion and coverage (by small amounts: 0.0017 coverage).
- **Holding the first frame beats both candidates at 1 s and 2 s** on validation (picture 18.06 against 17.37 dB at
  1 s, a tie for A; temperature at 2 s worse for A by 1.1 dB): over a second the fire changes less than the learned
  rollout's errors grow. Holding is a reference, not a method (it does not move).

## 7. The test

Run once, on the frozen file, after validation (`results/3d/compare_test_*.md`, per-run rows in `results/3d/test_track/`):
18 tracking runs at the 6 held-out settings, and 3 real runs, 3 plays of each method and 3 other real runs (the floor)
pooled per setting. Means, and the chosen model minus the coarse simulation with 95% intervals:

| measure | learned (A) | coarse simulation + detail | holding the first frame | floor (other real runs) | learned - coarse |
|---|---:|---:|---:|---:|---|
| tracking, temperature field, 8 frames (dB) | 22.57 | 20.58 | 21.01 | | +1.99 [+1.73, +2.27] |
| **tracking, temperature field, 1 s (dB)** | **19.31** | **18.23** | 20.09 | | **+1.08 [+0.74, +1.39]** |
| tracking, temperature field, 2 s (dB) | 18.86 | 17.77 | 19.83 | | +1.09 [+0.67, +1.49] |
| tracking, soot field, 8 frames / 1 s / 2 s (dB) | 27.35 / 24.77 / 24.22 | 24.78 / 19.22 / 17.79 | 25.27 / 24.92 / 24.71 | | +2.57 [+2.26, +2.92] / +5.55 [+4.84, +6.21] / +6.43 [+5.92, +6.92] |
| tracking, picture, 8 frames (dB) | 21.15 | 17.90 | 19.32 | | +3.25 [+2.69, +3.81] |
| **tracking, picture, 1 s (dB)** | **16.98** | **14.78** | 17.68 | | **+2.20 [+1.09, +3.12]** |
| tracking, picture, 2 s (dB) | 17.00 | 14.89 | 18.04 | | +2.11 [+1.25, +2.98] |
| **endless: 3D spectrum distance** | 0.451 | 0.769 | | 0.034 | -0.318 [-0.477, -0.053] (better) |
| **endless: \|ln motion ratio\|** | 0.409 | 0.181 | | 0.064 | **+0.229 [+0.063, +0.402] (worse)** |
| endless: motion ratio (context) | 1.063 | 1.071 | | 1.015 | -0.008 [-0.323, +0.421] |
| **endless: coverage distance** | 0.0089 | 0.0040 | | 0.0016 | **+0.0049 [+0.0006, +0.0116] (worse)** |
| **endless: emission distance** | 0.0101 | 0.0159 | | 0.0011 | -0.0059 [-0.0128, +0.0037] (tie) |
| **endless: mean picture PSNR (dB)** | 27.18 | 23.91 | | 42.52 | +3.27 [+0.54, +5.20] (better) |

**By the rule:**
1. **Tracking: it follows a real run better.** At 1 s the temperature field is 1.08 dB [+0.74, +1.39] and the picture
   from four cameras 2.20 dB [+1.09, +3.12] closer to the simulation's own continuation than the coarse simulation's
   with the same detail; both intervals are above zero. It is also better at 8 frames and 2 s, and in soot at every
   horizon.
2. **Look: it fails.** Two endless statistics are worse: |ln motion ratio| (+0.229 [+0.063, +0.402]) and the coverage
   distance (+0.0049 [+0.0006, +0.0116]). Two are better (the 3D spectrum distance, -0.318 [-0.477, -0.053], and the
   mean picture, +3.27 dB [+0.54, +5.20]) and the emission distance ties.
3. **The verdict: the learned 3D fire follows real runs better than running the cheap simulation, but looks worse in
   how much it moves (|ln motion ratio|) and in how much of the picture it covers (the coverage distance).** On
   average over the settings its motion is right (a ratio of 1.06 against the coarse simulation's 1.07), but it is off
   in both directions from setting to setting, where the coarse simulation is off by less: at the lowest wind it runs
   away and moves 2.4 times as much as the real runs, at the others it is calmer than they are (§11 has the
   per-setting rows).

Beside the rule (not judged):
- **Holding the first frame beats the learned model after a second**, as on validation: in temperature at 1 s by
  0.78 dB [0.29, 1.27] and at 2 s by 0.97 dB [0.20, 1.76], in the picture at 2 s by 1.04 dB [0.15, 2.06] (the picture
  at 1 s and the soot tie). Only at 8 frames does the learned model lead (+1.56 dB in temperature, +1.83 dB in the
  picture). At this grid a held-out fire changes so little in a second or two, against how fast the learned rollout's
  errors grow, that not moving at all scores better. Holding the first frame is not an effect (it does not move);
  it marks how far the learned rollout is from following a run.
- **The floor** (other real runs at the same setting against the real runs) is far better than both methods on every
  endless statistic: the 3D spectrum distance 0.034 against 0.451, the mean picture 42.5 dB against 27.2. Neither the learned
  model nor the coarse simulation plays like the 64 x 96 x 64 simulation.
- Validation's ordering held: the learned model leads the coarse simulation in tracking by about the same margins
  (picture at 1 s: +2.60 on validation, +2.20 on the test), and on the test it also wins the spectrum and the mean
  picture, which tied on validation.

## 8. Cost (provisional)

(Since 21 October 2026 the volume runtime's detail layer runs on vectors, with the same fields to the bit: about a fifth
of the time below on a quiet machine, and a step about 0.4; REPORT.md §19. Study N's model below was not re-timed;
study N2's, the one the demos play, was, on a quiet machine with the vector layer: a step 21 ms on one thread and 9 on
four, a 256 x 256 picture 56 and 14 ms, §14.12.)

`nvfx_3d timing` (`results/3d/timing.csv`): the chosen model through the volume C API at controls (0.7, 0.5, 0.6), 5
starts of 60 frames each, and 4 pictures of each size per start from one camera; then where a step's time goes (the
reference on one thread); then the simulator itself for scale. Milliseconds per frame, median (minimum):

| what | 1 thread | 2 threads | 4 threads |
|---|---:|---:|---:|
| a step (one frame) | 68 (60) | 79 (59) | 67 (33) |
| a picture, 256 x 256 | 49 (43) | 53 (36) | 55 (34) |
| a picture, 512 x 512 | 190 (170) | 194 (156) | 187 (141) |
| a start (the start point and the 24-frame warm-up of the fine fields) | 1,666 | 1,999 | 1,824 |

Where a step's time goes, on one thread (median ms): the noise inputs 2.7, the coarse step (convolutions, projection,
advection) 14.9, **the detail layer 59.1**, the renderer's preparation of the volume 19.2, and a 256 x 256 picture 43.8.
For scale, the simulator's frame costs 162 ms at 64 x 96 x 64 and 5.7 ms at the coarse 16 x 24 x 16, both on one
thread.

- **These numbers are provisional and say little about threads.** They were measured at nice 15 on the shared 4-core
  machine at a load of 7.5 to 8.1 (four other agents training), where extra threads find no free core: the medians do
  not fall with threads; the minimums, the frames that found free cores, do (a step 60 to 33 ms, a 512 x 512 picture
  170 to 141 ms). On an idle machine expect a step and a picture to scale with cores, as every loop splits over cells
  or rows.
- **The learned network is the cheap part.** The coarse step is 15 ms of a step's 77; the 64 x 96 x 64 detail layer
  (advection, the lock, the swirl) is three quarters of it. A step costs about 2.4 times less than the simulator's frame
  at full resolution, but about the same as the baseline (the coarse simulation's 5.7 ms plus the same detail layer):
  the learned model wins on quality (§7), not on cost.
- **The CPU picture is for tools and baking.** At 50 ms for 256 x 256 it is too slow for a game frame; an engine reads
  the fields as 3D textures (`nvfx_volume_read`) and ray-marches them on its GPU.
- **Memory**: a volume needs 31 MB (two shards' fine fields, one set of step buffers, the renderer's prepared volume);
  the effect file is 268 KB.

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

In `videos/` ([videos/README.md](../videos/README.md) §9), made by `nvfx_3d video` with the chosen model (A) at
1280 x 720, 30 fps, H.264 (CRF 24, yuv420p, +faststart, no audio), each over a dark floor grid and sky drawn by the tool.
The learned volume is played and drawn through the volume C API (the runtime's default playback: 8 s shards crossfaded
over 0.5 s); the real-against-learned video runs the reference model in one continuous rollout beside the simulator.
Demo seeds are 9,900,000 and up (no measured run uses them). Every frame is the volume renderer's (§3): the fire's
light, colour and smoke are physics-based pictures of the fields, not learned.

- **`3d_fire_orbit`** (12 s): the camera circles the fire once at controls (0.7, 0.5, 0.6), seed 9,900,001. A small
  flame under a column of grey smoke, seen from every side. The flame's base often breaks into bright, blocky spots
  (the 4-cell blocks of the detail layer's lock show through), it is whiter and shorter than the simulation's orange
  tongue, and the plume leans for a few seconds in still air before the 8 s shard change straightens it.
- **`3d_real_vs_learned`** (10 s): a held-out run (controls 0.65, 0.45, 0.55, seed 9,900,000) simulated for 100 frames,
  then the simulation (left) and the learned model from that same state with the same noise seed (right), one camera
  swinging a quarter turn around both. Captioned that they part after a second or two, which they do: by 2 s they are
  different fires with the same smoke column's height and lean; the learned flame is whiter at its base and broken into
  hot spots where the simulation's is one orange tongue.
- **`3d_wind`** (12 s): the side (left) and the top (right) at once, controls (0.7, *wind*, 0.5), seed 9,900,002,
  while the wind sweeps from none (0.5) to full towards -x (0) by 4.8 s, to full towards +x (1) by 9 s and back to none.
  The plume follows the control slowly: as the wind drops to full towards -x it barely turns (from above, the smoke
  still trails a little towards +x, as it did at the start), though held at a wind of 0.05 the model does lean towards
  -x as the simulation does; after the 8 s shard change, which starts from a stored state near the controls of that
  moment, it leans hard towards +x in the full +x wind and straightens as the wind falls back. **From 5 s, after the wind has blown hard towards -x, a second fire grows on the floor at the box's
  upwind edge and fills a corner of the box with a bright mass by 7.5 s**, until the shard change at 8 s clears it: the
  runaway of §11. A second render with the next seed (9,900,003) did the same, and so do constant winds of 0.05 and
  0.25, so the planned render is kept, as it is.
- **`3d_smoke_orbit`** (12 s), **a demo only**: §2 says the smoke preset was not trained; once the test was done there
  was time, so it was, on the same recipe and a shorter schedule (40 runs of 8 s, seeds 9,100,000 + i; 600, 250 and
  150 iterations; the model at the end of stage 3, which played better than stage 2's for the fire on validation;
  0.63 CPU-hours to simulate and 1.6 to train, on the shared machine). It was not validated or tested: no number in
  this study is about it. The camera circles it once at controls (0.7, 0.5, 0.5), seed 9,900,005. A thick grey column
  that billows and rises out of the frame, seen from every side; **small orange specks glow at its base, which the
  simulated smoke does not have** (its gas is too cool to glow; the detail layer's lock concentrates the learned heat
  into a few hot cells). Its sides do not heat up at winds of 0.05 or 0.5 in 8 s (`results/3d/edge.csv`; §11).

## 11. Limits

- **The teacher is our own solver, tuned by eye.** Nothing here is checked against a real fire: the plume rises and
  puffs at a steady rate, but at 0.5 to 0.65 Hz, several times slower than a real pool fire of this size would. The
  learned model can only be as real as its teacher. The PC plan's validated teachers (FDS) are what would change that.
- **One effect, one small model.** A fire, with a 16 x 24 x 16 learned state, 16 channels and 14,218 weights, trained
  for 1,500 iterations on 40 runs (5.3 minutes of simulated fire). The smoke was trained after the test as a demo
  only (§10): not validated, not tested. No hyperparameter was searched.
- **Small samples.** The test pairs 18 tracked runs and 6 settings (3 runs pooled per setting); study D2 pooled 8.
  Intervals over 6 settings are wide.
- **The picture is a fixed physical model, not a learned one**: emission and absorption with single scattering of one
  fixed key light, an ambient light and the fire's own light without shadows. No multiple scattering, no blue
  chemiluminescent base (a soot-and-temperature model has none), one exposure for every picture.
- **The measures see the fields at 64 x 96 x 64 and pictures at 128 x 128 from four fixed cameras**; the 3D spectrum
  covers the lowest 64 cells of the box only.
- **The chosen model runs away when the wind blows towards -x** (a wind control of 0.3 or below). A second fire
  starts on the floor at the box's upwind edge after 2 to 5 s and grows into a bright mass that can fill the box, until
  the next shard change (8 s) clears it ([figures/n_wind_runaway.png](figures/n_wind_runaway.png): constant controls
  0.7, 0.05, 0.5, seed 9,900,004, at 2, 3.5, 5 and 6.5 s). It was seen at winds of 0.05 and 0.25 in renders, and the
  endless statistics show it at 0.15, 0.25 and 0.30 (the per-setting rows below); not at 0.4 to 0.8. The wind video
  shows it (§10).

  **Where it grows** (`nvfx_3d edge`, `results/3d/edge.csv`: the reference model played for 8 s from the start point
  nearest the controls (0.7, *wind*, 0.5), seed 9,900,004, the coarse heat summed over each side of the box every
  15 frames): the heat grows at the box's sides, not at the burner, and grows exponentially, doubling about every half
  second once it starts. At a wind of 0.05 the upwind side (x = max) holds 0.003 at the start, 0.5 after 2 s, 64 after
  5 s; the two sides along the wind (z = 0 and z = max) follow, and the box's total heat goes from 37 to 1,848 by
  6.5 s. **With no wind (0.5) the same growth starts later**: the z = max side holds 0.001 at the start, 0.09 at
  4.5 s and 35 at 8 s, the box's total heat goes from 19 to 105 between 4.5 and 8 s, and the hottest side cell reaches
  0.67 (about 1,570 K in the renderer) just as the shard ends. At 0.95 it is slower again (4.5 at z = max after
  8 s). So the 8 s shards are what keeps the effect playable at most settings. **Candidate B** (stage 3's activity
  loss, passed over by the rule) grows far less: at 0.05 its side heat peaks at 23 and the box's total at 95 after
  5 s, then falls back to 60; without wind nothing grows in 8 s. The smoke (stage 3 too) grows at neither wind. The
  activity loss, which matches how much the state changes per frame, holds most of it back.

  **The most likely cause** is the coarse advection: it samples clamped to the grid, so at a side where air flows in
  (upwind, and wherever the plume draws air in from its sides) a cell traces back to itself and keeps its heat every
  frame; whatever the stepper adds there feeds itself, and only the range clamp bounds it. The simulator's sides have
  no heat to keep. Training windows (at most 24 frames of burn-in and 12 of loss) are far shorter than the 4 to 5 s
  the growth takes to show at most winds. Why the upwind side starts at once only when the wind blows towards -x is
  not known.
- **It is most of what the rule found worse.** Per setting (`compare_test_learned_vs_coarse.md`; context, not judged):
  at the test's lowest wind, (0.45, 0.25, 0.30), the learned fire moves 2.4 times as much as the real runs (the coarse
  simulation 1.44 times), its coverage distance is 0.022 (coarse 0.001) and its mean picture 2.9 dB worse than the
  coarse simulation's. At the five other settings it is calmer than the real runs (motion ratios 0.53 to 0.82, and
  1.15) and better than the coarse simulation in the 3D spectrum, the emission and the mean picture at every one; its
  coverage distance is still larger at four of the five, by less (0.0013 to 0.0051). On validation the same holds at
  winds 0.30 and 0.15 (motion 3.1 and 2.8 times the real runs'), which is why A played so much worse than B there;
  at the other two validation settings A and B are alike.
- **Controls changed live are followed slowly.** Training runs held their controls fixed; in the wind video the plume
  barely turns within three seconds of the wind dropping to full towards -x, and much of its later lean comes from the
  shard change's fresh start (§10). Within a shard a control change acts only through the stepper's FiLM.
- **The rule chose the candidate that plays worse** (§6): stage 3's model moves and spreads more like the simulation
  when it plays on, but tracked 0.29 dB worse at 1 s.
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
$B video --kind orbit --out 3d_fire_orbit.mp4 --seconds 12 --controls 0.7,0.5,0.6 --seed 9900001                  # §10
$B video --kind real-vs-learned --out 3d_real_vs_learned.mp4 --seconds 10 --controls 0.65,0.45,0.55 --seed 9900000
$B video --kind wind --out 3d_wind.mp4 --seconds 12 --controls 0.7,0.5,0.5 --seed 9900002
for w in 0.05 0.5 0.95; do $B edge --controls 0.7,$w,0.5; done                 # §11 (results/3d/edge.csv); also
for w in 0.05 0.5; do $B edge --model .../candB.nvfx3 --controls 0.7,$w,0.5; done  # with candB and smoke3d
# the smoke, a demo only (§10)
$B record --effect smoke --runs-dir smoke --seed-base 9100000 --runs 40 --frames 240 --threads 3
$B train --name smoke --runs-dir smoke --iterations 600 --finetune 250 --activity 150 --batch 8 --unroll 12 --burn 24
$B finish --runs-dir smoke --ckpt smoke --out smoke3d
$B video --kind orbit --model .../smoke3d.nvfx3 --title "LEARNED 3D SMOKE" --out 3d_smoke_orbit.mp4 --seconds 12 --controls 0.7,0.5,0.5 --seed 9900005
```

The per-run tracking rows (`results/3d/val_track/*.csv`, `results/3d/test_track/*.csv`) have no header: the setting,
the run, then for each horizon (8, 30 and 60 frames) the horizon, the temperature field PSNR, the soot field PSNR, the
picture PSNR (the mean over the cameras) and each camera's picture PSNR (azimuths 0, 90, 180 and 270 degrees).

## 13. What it means for the PC plan

The plan's stage 1 asked three questions ([TODO.md](../TODO.md)). The answers, from this one small fire:

- **Does a learned 3D rollout work at all?** Yes, as machinery: the 3D solver, the volume renderer, the model, its
  trainer with checked gradients, a resumable deterministic training run, the runtime and its C API all work end to
  end on a CPU, and the model follows a held-out fire for 2 s from any of four cameras. As an effect it is a start:
  small blocky flames, and a runaway when the wind blows towards -x.
- **Does it beat simply running a cheap 3D simulation?** At following a real run, clearly (+1.1 dB in temperature and
  +2.2 dB in the picture at 1 s, every horizon better); at playing on, not by the rule (worse in motion and coverage,
  better in the spectrum and the mean picture). And holding the first frame still beats it after a second, so the
  learned rollout's error growth, not the baseline, is what to beat next.
- **What does it cost to play?** About 70 ms a step and 50 ms a 256 x 256 picture on one busy core, three quarters of
  the step in the 64 x 96 x 64 detail layer, not the network; about as much as the cheap simulation with the same detail.

What to do differently on the PC:
1. **Choose on how it plays, not only on how it tracks.** The rule here chose on tracking at 1 s, and the candidate it
   chose plays worse than the one it passed over (§6) and runs away where that one mostly does not (§11). For effects that are played endlessly the choice should require
   the endless statistics not to be worse, then rank by tracking.
2. **Let nothing flow in at an open edge, and train on long rollouts.** The runaway (§11) grows at the box's sides at
   every wind and most likely comes from advection that samples clamped to the grid, so a side where air flows in
   keeps and feeds whatever heat reaches it; open boundaries should bring in ambient air (zero heat, soot and memory),
   and `nvfx_3d edge` is the check. And a slow build-up like it is invisible to
   windows of 12 frames after at most 24 of burn-in: train with burn-in from the stepper's own long rollouts (hundreds of
   frames), at the controls' extremes, and with controls that move.
3. **Spend the cost where it is.** The detail layer is the cost: SIMD and sparse bricks for it first (only blocks with
   material), and on an engine the picture belongs to the GPU. A bigger network (32 to 64 channels, 32³ to 64³ states)
   is affordable next to it.
4. **A finer learned state, or a smoother lock.** The 4-cell blocks show in the flame; emission is so steep in
   temperature that block averages lose the hottest spots, so learned flames are whiter and shorter. A 32³ state, or a
   lock that blends between blocks, or a sub-grid temperature variance channel, is the next step.
5. **A validated teacher.** This solver puffs at 0.5 to 0.65 Hz where a fire this size puffs near 2.7 Hz; the learned
   model copies its teacher faithfully, so the teacher's physics is the ceiling (FDS, as the plan says).
6. **Budget for training.** The fire's 40 training runs took 0.65 CPU-hours to simulate and its 1,500 iterations 2.3
   CPU-hours (2.4 hours of wall time, at under one core of the shared machine). The trainer splits a batch's windows
   over threads with the same result on any count, so a PC's cores shorten it directly.
7. **Snapshot every stage**, as the trainer now does: a stage that ended between checkpoints lost its model here (§6).

## 14. Study N2: ambient air at the open edges

Status: **done** (15 October 2026). Study N's demo has one fault that spoils it: the runaway at the box's sides (§11).
Study N2 fixes it in the model's advection, retrains the same small model on data that holds long rollouts, the
controls' extremes and controls that change, chooses on tracking *and* endless play (study N's first lesson, §13), and
tests once. §14.1 to §14.5 (the fix, the data, the training, the seeds, the settings and the rule) were written and
committed before any validation number; the results follow them (§14.6 to §14.11).

**In short.** The fix works and the retrained model (C4, after a new stage of long rollouts) has **no runaway**: over
30 s at five constant winds its upwind side and the sides along the wind hold at most 0.2 (study N's model: about 600,
the box white), on validation's seeds and the test's. On the test, run once, it **tracks better than study N's model**
(at 1 s: temperature +1.24 dB [+0.89, +1.56], picture +1.95 dB [+1.30, +2.64]) and **plays better** (better on four
of the five endless statistics, a tie on motion), it beats holding the first frame at every horizon, and against the
coarse simulation it follows real runs better but moves less than the real runs (|ln motion ratio| worse, +0.165
[+0.007, +0.329]). **But by the rule it does not pass**: on validation every candidate failed the edge check, C4 by
3% on one side (its downwind side at a wind of 0.75 peaked at 5.29 where the bound was 5.12), and the rule declared
that the chosen model then fails the test's first condition. So **study N's model and videos stay**; the numbers are
below, and whether to adopt C4 anyway is the owner's call (§14.11).

### 14.1 The fault and the fix

**The fault** (§11): the coarse advection traces each cell back along the flow and samples the state there; where the
trace left the grid it was clamped to the edge cell. At a side where air flows in, a cell then traces back to itself
and keeps its heat every frame, and whatever the stepper adds there feeds itself.

**The fix** (`src/core/rollout3d.cpp`, `trilinear_ambient`; the trainer's `advect` and `advect_back` in
`src/train/rollout3d_train.cpp`; the volume runtime runs the reference, so it has it too): beyond the open sides (x and
z) and the top lies ambient air, one ghost cell deep, with **zero heat, zero soot and zero memory, and still**; a trace
that leaves the grid there samples between the edge cell and that air. The floor is closed and still clamps.

- **"The wind's velocity where the wind is."** In our simulator the wind drags only material
  (`sim3d.cpp`, `add_forces`: u moves towards the wind's speed in proportion to min(1, 2 (heat + soot))), so air with
  nothing in it has no wind of its own. Study N's 40 training runs show it: the mean velocity across a side (x = 0 or
  x = max) never exceeds 0.019 coarse cells per frame in any frame from 4 s on, where the full wind is 0.375. So the
  model's ambient air is still.
- **The edge mode belongs to the model** (`Model::edges`): an `.nvfx3` file of version 2 stores it; a version 1 file
  (study N's) loads with clamped edges and plays exactly as before (study N's `edge` rows of §11 come out the same to
  every printed digit), and a model with clamped edges is still saved as version 1.
- **Tests** (`tests/test_3d.cpp`, 28 in all, 7 new): what flows in at each side and through the top, and that the floor
  clamps; the trainer's forward pass equals the reference's with air coming in through every side and the top, with
  either edge mode; **gradients match finite differences** with inflow everywhere, with clamped edges and with controls
  that change during the window; the runtime equals the reference to the bit with either edge mode; a steady schedule
  of controls is the fixed run to the bit; training with the new stage resumes to the bit.
- **A first look, before anything was trained** (study N's chosen model, its §11 seed 9,900,004 and 8 s, the reference
  played): with the fix alone and no wind, nothing grows (the box's heat 18 to 24, the sides under 0.02, where with
  clamped edges it reaches 105 and a side 35); at a wind of 0.05 the box's heat reaches 107 instead of 1,848, but the
  sides along the wind (z = 0, z = max) still collect heat (16 and 4 after 8 s). The stepper was trained with clamped
  edges, so the fix alone is not enough for study N's model: retrain.

### 14.2 The simulator's own boundaries

They do not have the fault. The simulator's heat and soot have a border of zero (ambient) that their advection samples,
so a cell at an inflow side takes in empty air, never its own heat. One small leak: the sample position is clamped half
a cell beyond the edge, so a trace more than half a cell outside (an inflow faster than 30 cells per second at 64
cells) takes half the edge cell's value instead of nothing; that halves every substep, it never feeds. Its velocity
border copies the edge cell (outflow), as usual. In study N's 40 runs the upwind side holds no heat at all (the
x = max side at winds below 0.35 and the x = 0 side above 0.65: at most 0.0003 summed over a side's 384 coarse cells,
from 4 s on). **The simulator is unchanged**, so study N's training runs are still the teacher's and are used again.
The edge check (§14.5) runs the simulation beside the model as the reference.

### 14.3 Data and training

- **Data**: study N's 40 runs (8 s each at random controls, seeds 9,000,000 + i) and **24 new runs of 18 s** (540
  frames; seeds 9,200,000 + i), by i mod 3: 8 with **the wind at an extreme** (0 to 0.15 for even i / 3, 0.85 to 1 for
  odd), the other controls random; 8 with **controls that change during the run** (segments of 2 to 4 s, each with new
  targets reached by a smooth 1 s ramp; the wind's target at an extreme six times in ten, else anywhere; intensity and
  turbulence anywhere); 8 at random controls held, as study N's but longer. 12.5 minutes of simulated fire in all.
- **The model**: study N's size (16 x 24 x 16, two 3 x 3 x 3 convolutions of 16 channels, 4 memory channels, 14,218
  weights) with ambient edges, initialised with seed 9,200,777; the detail layer's constants as study N's (contrast 1,
  swirl 0.5, swirl length 12, grow 1.5); 8 start points at frame 150 by farthest-point sampling over the controls (at
  that frame) of all 64 runs.
- **Training**: windows drawn in proportion to the runs' frames; every step takes the controls of its own frame. Stages
  1 to 3 as study N's (900, 350 and 250 iterations; batch 8; unroll 12; in stages 2 and 3 half the windows after up to
  24 frames of the stepper's own rollout, with the profile loss; stage 3 with the activity loss). **Stage 4, long
  rollouts** (new): 250 iterations in which every window starts after 48 to 240 frames of the stepper's own rollout
  (no gradient through it), from the runs long enough to hold it (the 24 new ones), with stage 3's losses and learning
  rate (7e-4, cosine). Trainer seed 9,200,000. **A snapshot at the end of every stage.**
- **A finer state (24 x 36 x 24) is not tried**: 24 does not divide the 64-cell detail grid, so it would need new
  training runs at another resolution, which this machine's time does not allow.

### 14.4 Seeds and held-out settings

**Seeds** (unused blocks of study N's range): training 9,200,000 + i; validation tracking 9,500,000 + 100 s + k, real
runs 9,510,000 + 100 s + k, plays 9,520,000 + 100 s + k; test tracking 9,800,000 + 100 s + k, real runs
9,810,000 + 100 s + k, the floor 9,810,050 + 100 s + k, plays 9,820,000 + 100 s + k (s the setting, k the run); the
edge check's plays 9,520,900 + j (validation) and 9,820,900 + j (test) and its simulations 9,510,900 + j (j the
wind's index); demo videos 9,950,000 and up.

**Held-out settings** (intensity, wind, turbulence), new, none a training run's (training controls are random or
scheduled), with a strong wind towards -x and towards +x in each split:
- validation: (0.60, 0.08, 0.50), (0.40, 0.92, 0.65), (0.75, 0.30, 0.35), (0.50, 0.50, 0.80), (0.30, 0.65, 0.25);
- test: (0.70, 0.05, 0.40), (0.55, 0.95, 0.55), (0.45, 0.20, 0.70), (0.80, 0.50, 0.30), (0.35, 0.75, 0.45),
  (0.65, 0.40, 0.90).

Validation: 3 tracking runs per setting (15), 2 real runs and 2 plays per method pooled per setting. Test: 3 tracking
runs per setting (18), 3 real runs, 3 plays and 3 other real runs (the floor) per setting.

### 14.5 Measures, the edge check and the rule

**Methods**: the candidates; **study N's chosen model as it shipped** (A, clamped edges; "N"); **the coarse simulation
with the same detail** (§5, from the new model's start points); for reference only, holding the first frame, study N's
model with the edge fix alone ("N + fix", validation only) and the floor (test only).

**Measures**, as §5: tracking from the true state (temperature and soot field PSNR, picture PSNR from four cameras; at
8 frames, 1 s and 2 s); endless statistics (3D spectrum distance, |ln motion ratio|, coverage distance, emission
distance, mean picture PSNR; 240 frames after the warm-up, as §5). Differences: **candidate minus study N's model** and
**candidate minus the coarse simulation**; paired bootstrap of the mean difference, 10,000 resamples, seed 2026, 95%
percentile intervals; tracking paired over runs, endless statistics over settings.

**The edge check** (`nvfx_3d edge`, `nvfx_3d edgecheck`): the model played in one rollout from the start point nearest
(0.7, w, 0.5), its 24-frame warm-up and then **30 s**, at constant winds **w = 0.05, 0.25, 0.5 (no wind), 0.75 and
0.95** (the coarse state: the detail layer never feeds back into it); and the simulation itself at the same controls
(64 x 96 x 64 from an empty grid, block averages, 30 s from 5 s on, as a start point is 5 s into its run). Every
0.5 s, the coarse heat summed over each side slab (x = 0, x = max, z = 0, z = max) and over the top layer. A side
**runs away** if (a) its mean over 20 to 30 s is more than twice its mean over 5 to 15 s plus 0.5 (it grows), or (b)
its largest value over 5 to 30 s is more than twice the simulation's largest on that side plus 1 (it holds far more
heat than the teacher ever does there). **A model passes if no side runs away at any wind.**

**Candidates**: **C2** (the stepper after stage 2), **C3** (after stage 3), **C4** (after stage 4), with the same start
points, detail constants and renderer.

**The choice (validation)**, on tracking and endless play:
1. eligible: the candidates that pass the edge check;
2. **tracking, non-inferiority**: of those, the ones whose mean picture PSNR at 1 s and mean temperature field PSNR at
   1 s are each within 0.5 dB of the best eligible candidate's;
3. **endless play, improvement**: of those, the one with the lowest mean rank over the five endless statistics (each the
   mean over the five settings; tied values share a rank); a tie goes to the higher picture PSNR at 1 s.

If no candidate passes the edge check, steps 2 and 3 choose among all three and the chosen one fails the test's first
condition. `nvfx_3d choose` applies the rule. The chosen file is frozen by SHA-256 in `results/3d/n2/frozen.csv` before
the test; the tool refuses the test's learned runs on any file not listed there.

**The test (it runs once)**, the chosen model on the test's settings and seeds:
1. **It fixes study N's fault** if it passes the edge check with the test's seeds (9,820,900 + j).
2. **Against study N's model**: its tracking is *non-inferior* if at 1 s the interval of the temperature field PSNR's
   difference lies above -0.5 dB and the picture PSNR's above -1 dB; its endless play is *not worse* if no endless
   statistic's interval lies entirely on the worse side, and *better* if, in addition, at least one lies entirely on
   the better side.
3. **It passes** if 1 holds, its tracking is non-inferior and its endless play is not worse: then it becomes the demo
   model and the videos are made again. "**It plays better than study N's model**" if it passes and its endless play is
   better. If it does not pass, study N's model and videos stay, and the numbers are reported.
4. **Against the coarse simulation**, study N's rule unchanged (§5): "beats running the cheap simulation", "follows
   real runs better but looks worse in ...", or "does not beat running the cheap simulation".

Soot, 8 frames and 2 s, holding the first frame and the floor are reported beside, not judged. **Cost**, as §5:
provisional.

### 14.6 Training and the candidates

As planned (§14.3). The 24 new runs took 0.82 CPU-hours to simulate (two jobs, 20 minutes); none of the held-out
settings is a training run's (the nearest training control to any of them, over every frame of the runs whose controls
change, is 0.03 to 0.17 away in the control cube). The stepper trained in 3.2 hours of wall time and 4.55 CPU-hours on
three threads: stage 1 1.2 CPU-hours, stages 2 and 3 1.2, **stage 4 (the long rollouts) 2.2**, as much as the other
three together. Every stage was snapshot (`ckpt/n2_stage1..4.ckpt` in the data root); the candidates' files are
`models/C2.nvfx3`, `C3.nvfx3` and `C4.nvfx3` (274,760 bytes each, `.nvfx3` version 2, ambient edges), with the same
8 start points.

### 14.7 Validation

**The edge check** (`results/3d/n2/edge_val.csv`, `edgecheck_val.md`): the largest coarse heat over 5 to 30 s summed
over a side's 384 cells (the upwind side: x = max when the wind blows towards -x, x = 0 towards +x; the box: all of it):

| model | upwind side | downwind side | x sides, no wind | z sides | top | the box | the check |
|---|---:|---:|---:|---:|---:|---:|---|
| the simulation (64 x 96 x 64) | 0.00 | 8.50 | 0.01 | 0.46 | 3.31 | 19 to 59 | (the reference) |
| study N's model (clamped edges) | 630 | 642 | 635 | 686 | 337 | 5 to 9,573 | runs away at every wind |
| study N's model with the fix alone | 14.6 | 6.8 | 6.9 | 27.0 | 5.1 | 4 to 132 | fails at 0.05, 0.25, 0.5 |
| C2 (after stage 2) | 3.6 | 17.8 | 5.4 | 4.8 | 4.3 | 2 to 62 | fails at every wind |
| C3 (after stage 3) | 1.7 | 4.9 | 1.2 | 0.2 | 0.2 | 0.5 to 42 | fails at 0.5 and 0.75, narrowly |
| **C4 (after stage 4)** | **0.09** | 8.5 | 0.16 | 0.17 | 0.95 | 15 to 45 | **fails at 0.75 only, by 3%** |

- **Study N's model fills the box at every wind within 30 s** played in one rollout (its heat reaches the range
  clamp); its 8 s shards are what hid this at most winds in study N's 8 s check (§11). Without shards it whites out
  ([figures/n2_edges.png](figures/n2_edges.png), top row).
- **The fix alone helps but does not cure it**: the box's heat stays under 132, but second and third fires still grow
  at the sides at winds of 0.05 to 0.5 (the figure's middle row). Its tracking is study N's (within 0.03 dB) and its
  endless play is better (spectrum, coverage and emission distances and the mean picture). It was trained with clamped
  edges, so its stepper still adds heat at the sides.
- **Retraining with the fix** removes most of it, and **the long-rollout stage removes the rest**: C4 keeps the upwind
  side and the sides along the wind under 0.2 at every wind and its box's heat within the simulation's range. **It
  fails the check on one count**: at a constant wind of 0.75 its downwind side (where the plume leans, x = max)
  peaked at 5.29 where the bound, twice the simulation's largest there (2.06) plus 1, was 5.12. That side does not grow
  (mean 3.4 over 5 to 15 s, 4.4 over 20 to 30 s); the plume leans against it a little harder than the one simulated run
  did. C3 fails as narrowly (an x side reaching 1.2 to 1.7 at 0.5 and 0.75 where the bound is about 1) and its fire
  fades out at a wind of 0.25 after 25 s (the box's heat 0.5); C2 sends puffs of heat to sides where the simulation has
  none.

**The rest of validation** (`compare_val_*.md`; 15 tracking runs at 5 settings, 2 real runs and 2 plays pooled per
setting), means:

| | C2 | C3 | **C4** | study N's model | N + fix | coarse simulation | holding the first frame |
|---|---:|---:|---:|---:|---:|---:|---:|
| tracking, picture, 1 s (dB) | 17.74 | 17.28 | **19.21** | 17.81 | 17.81 | 13.61 | 18.54 |
| tracking, temperature, 1 s (dB) | 19.22 | 18.88 | **20.56** | 19.41 | 19.38 | 17.42 | 20.36 |
| endless: 3D spectrum distance | 0.316 | 0.301 | **0.262** | 0.774 | 0.353 | 0.772 | |
| endless: \|ln motion ratio\| (motion ratio) | 0.424 (0.72) | 0.485 (0.65) | **0.373** (0.75) | 0.440 (1.59) | 0.379 (0.79) | 0.316 (1.34) | |
| endless: coverage distance | 0.0044 | 0.0046 | **0.0028** | 0.0285 | 0.0040 | 0.0035 | |
| endless: emission distance | 0.0045 | 0.0043 | **0.0040** | 0.0271 | 0.0047 | 0.0173 | |
| endless: mean picture PSNR (dB) | 28.84 | 28.91 | **31.56** | 24.15 | 28.05 | 23.62 | |

Differences with 95% intervals, C4 minus study N's model and minus the coarse simulation:

| measure | C4 - study N's model | C4 - coarse simulation |
|---|---|---|
| tracking, temperature, 1 s / 2 s (dB) | +1.16 [+0.72, +1.59] / +1.80 [+1.31, +2.25] | +3.14 [+2.43, +3.80] / +3.39 [+2.87, +3.96] |
| tracking, picture, 8 frames (dB) | +0.17 [-0.31, +0.65] | +5.09 [+3.76, +6.60] |
| tracking, picture, 1 s / 2 s (dB) | +1.40 [+0.49, +2.26] / +1.59 [+1.11, +2.05] | +5.60 [+4.33, +6.75] / +4.69 [+3.85, +5.58] |
| endless: 3D spectrum distance | -0.512 [-0.948, -0.109] (better) | -0.510 [-0.596, -0.428] (better) |
| endless: \|ln motion ratio\| | -0.067 [-0.543, +0.382] (tie) | +0.057 [-0.238, +0.372] (tie) |
| endless: coverage distance | -0.0257 [-0.0570, -0.0026] (better) | -0.0007 [-0.0012, -0.0002] (better) |
| endless: emission distance | -0.0231 [-0.0518, -0.0027] (better) | -0.0132 [-0.0192, -0.0063] (better) |
| endless: mean picture PSNR (dB) | +7.41 [+2.59, +13.60] (better) | +7.94 [+5.96, +10.48] (better) |

- **The choice, by the rule** (`choice_val.md`): **no candidate passes the edge check**, so steps 2 and 3 chose among
  all three: only C4 tracks within 0.5 dB of the best (it is the best, by 1.5 dB in the picture at 1 s), so **C4**,
  which also ranks first on every endless statistic. Its file was frozen before the test (`results/3d/n2/frozen.csv`:
  274,760 bytes, SHA-256 5a22b5c1722eae2fff07417daea6ba60f7a3c602e41517d199ed4d6fabd11b80, with study N's file for its
  own test runs). By the rule it already fails the test's first condition.
- **Stage 4 is what made C4**: against C3, the same model before the long rollouts, it tracks 1.9 dB better in the
  picture at 1 s and its means are better on every endless statistic. Stages 2 and 3 alone (C2, C3) track like study N's model and play
  better than it (no runaway at the validation settings' winds), but run off at the sides over 30 s.
- C4 is **calmer than the real runs** (motion 0.75 of theirs; study N's 1.59, the coarse simulation's 1.34): every N2
  candidate is (0.65 to 0.75), so the long rollouts are not the only cause.

### 14.8 The test

Run once, on the frozen C4 (`compare_test_*.md`, `edgecheck_test.md`, per-run rows in `results/3d/n2/test_track/`): 18
tracking runs at the 6 held-out settings; 3 real runs, 3 plays of each method and 3 other real runs (the floor) pooled
per setting; the edge check with the test's seeds.

| measure | C4 | study N's model | coarse simulation | holding | floor | C4 - study N's model | C4 - coarse simulation |
|---|---:|---:|---:|---:|---:|---|---|
| tracking, temperature, 8 frames (dB) | 21.90 | 21.81 | 19.73 | 20.41 | | +0.09 [-0.16, +0.32] | +2.17 [+1.83, +2.52] |
| **tracking, temperature, 1 s (dB)** | **20.80** | 19.56 | 18.00 | 19.94 | | **+1.24 [+0.89, +1.56]** | **+2.79 [+2.20, +3.36]** |
| tracking, temperature, 2 s (dB) | 21.11 | 18.95 | 17.55 | 19.72 | | +2.16 [+1.77, +2.57] | +3.56 [+3.06, +4.09] |
| tracking, soot, 1 s (dB) | 25.67 | 24.70 | 19.29 | 24.55 | | +0.97 [+0.67, +1.26] | +6.38 [+5.58, +7.18] |
| tracking, picture, 8 frames (dB) | 21.19 | 20.83 | 16.89 | 19.67 | | +0.36 [-0.04, +0.74] | +4.30 [+3.21, +5.47] |
| **tracking, picture, 1 s (dB)** | **19.40** | 17.45 | 14.24 | 18.08 | | **+1.95 [+1.30, +2.64]** | **+5.15 [+3.88, +6.38]** |
| tracking, picture, 2 s (dB) | 19.93 | 17.57 | 14.80 | 18.38 | | +2.36 [+1.85, +2.92] | +5.13 [+4.31, +5.89] |
| **endless: 3D spectrum distance** | 0.258 | 0.626 | 0.750 | | 0.044 | -0.368 [-0.751, -0.098] (better) | -0.492 [-0.550, -0.424] (better) |
| **endless: \|ln motion ratio\|** | 0.392 | 0.433 | 0.227 | | 0.099 | -0.041 [-0.254, +0.171] (tie) | **+0.165 [+0.007, +0.329] (worse)** |
| endless: motion ratio (context) | 0.682 | 1.480 | 1.171 | | 0.925 | -0.798 [-1.358, -0.314] | -0.489 [-0.716, -0.264] |
| **endless: coverage distance** | 0.0032 | 0.0272 | 0.0036 | | 0.0019 | -0.0240 [-0.0600, -0.0027] (better) | -0.0004 [-0.0019, +0.0010] (tie) |
| **endless: emission distance** | 0.0049 | 0.0238 | 0.0189 | | 0.0012 | -0.0189 [-0.0460, -0.0032] (better) | -0.0140 [-0.0217, -0.0071] (better) |
| **endless: mean picture PSNR (dB)** | 31.89 | 24.41 | 22.81 | | 41.51 | +7.49 [+3.86, +12.56] (better) | +9.08 [+7.85, +10.59] (better) |

**By the rule** (§14.5):
1. **It fixes study N's fault: no, by the rule.** No candidate passed the edge check on validation, so the rule had
   already declared this condition failed. With the test's seeds C4 *does* pass it: nothing runs away at any wind (its
   upwind side at most 0.18, the sides along the wind 0.14, the top 0.91, the box 13 to 53; study N's model fills the
   box again, `edgecheck_test.md`). That does not change the verdict; it is reported beside it.
2. **Against study N's model**: tracking is non-inferior, and better (temperature at 1 s +1.24 dB [+0.89, +1.56] with
   a lower bound far above -0.5; picture +1.95 dB [+1.30, +2.64], far above -1); endless play is **not worse and
   better** (the spectrum, coverage, emission and mean picture better; the motion a tie). At the strong -x wind, where
   study N's model ran away, its mean picture is 31.7 dB against 12.7.
3. **It does not pass**, because condition 1 failed. Study N's model stays the demo model and its videos stay. (Had the
   edge check held on validation, it would have passed and "played better than study N's model".)
4. **Against the coarse simulation**, study N's rule: tracking better (temperature +2.79 dB [+2.20, +3.36], picture
   +5.15 dB [+3.88, +6.38] at 1 s); look: worse in one statistic, **how much it moves** (|ln motion ratio| +0.165
   [+0.007, +0.329]): it moves 0.68 times as much as the real runs where the coarse simulation moves 1.17 times; the
   coverage ties, and the spectrum, emission and mean picture are better. **The verdict: it follows real runs better
   than running the cheap simulation, but looks worse in how much it moves** (calmer than the real fire, at every
   setting: motion ratios 0.54 to 0.76). Study N's model was worse there in motion and coverage, mostly by running
   away; C4 is worse in motion only, by being too calm.

Beside the rule: **C4 beats holding the first frame at every horizon** (temperature at 1 s +0.85 dB [+0.12, +1.58], at
2 s +1.39 [+0.60, +2.20]; picture at 1 s +1.32 [+0.43, +2.28]), which study N's model did not (it lost to it after a
second, §7). The floor (other real runs) is still far closer to the real runs than any method (spectrum 0.044, mean
picture 41.5 dB). Validation's ordering held on the test.

### 14.9 The smoke (optional)

Study N's smoke demo (§10) was retrained with the fix on study N's 40 smoke runs (study N's schedule of 600, 250 and
150 iterations plus a 150-iteration long stage with burn-ins of 48 to 200 frames; trainer seed 9,200,100; 2.9
CPU-hours; `models/smoke2.nvfx3` in the data root), and both were put through the edge check against the smoke
simulation (seeds 9,950,100 + j and 9,950,200 + j; `results/3d/n2/edge_smoke.csv`, `edgecheck_smoke.md`). **Study N's
smoke runs away too, given time**: over 30 s its box's heat climbs from 13 to 1,114 at a wind of 0.05 and to 3,128 at
0.25 (the smoke simulation's stays 6 to 20); study N checked it for 8 s only, where it held, and its 8 s shards hide
it. **The retrained smoke does not run away** (its box 8 to 40), but it fails the same strict check: a side along the
wind collects up to 5.2 where the simulation has 0.1. Neither is validated or tested; the smoke video stays study N's.

### 14.10 Videos, the figure and the demo

**The demo videos are not re-rendered**: by the rule C4 did not pass, so `videos/3d_*` stay study N's
([videos/README.md](../videos/README.md) §9). No `3d_wind_fixed` clip was added either: **the fix alone does not
visibly remove the runaway**. Study N's model with the fix, rendered as `3d_wind` with seed 9,950,002, still grows a
small second flame at the box's side at 6.5 to 7 s (where study N's own model grows three), and played without shards
at a held wind of 0.05 it grows second and third fires.

**[figures/n2_edges.png](figures/n2_edges.png)**: the wind held at 0.05, one continuous rollout (no shards), seed
9,950,010, from the side and from above at 3, 8, 14 and 20 s. Top: study N's model, white by 8 s. Middle: the same
with the fix alone, three fires by 14 s. Bottom: C4, one fire leaning with the wind for the whole 20 s.

**C4's demo renders, for the owner to look at** (not in git; in the data root's `videos/`, seeds 9,950,000 to
9,950,002, the same kinds and controls as §10): the orbit shows one small fire under a straight grey column (study N's
leaned in still air for seconds); the flame base is still broken into bright blocky spots (the detail layer's lock, as
in study N). Real against learned: they part after a second or two, the learned flame shorter and more broken than the
simulation's orange tongue. The wind sweep: **one fire throughout, no second fire anywhere**; the plume still follows
the control slowly: smoke blown towards -x keeps trailing that way well into the +x half of the sweep.

**The owner's decision (15 October 2026).** Shown the comparison, the owner chose to make C4 the demo. Since then
`videos/3d_fire_orbit.mp4`, `3d_real_vs_learned.mp4` and `3d_wind.mp4` are C4's renders above, labelled in
[videos/README.md](../videos/README.md) §9 as shown by that decision and not by the rule; study N's renders are in the git
history. The rule's verdict above stands as written: C4 did not pass N2's edge check on validation (by 3%), and nothing
in the study changed.

### 14.11 What it means

- **The runaway's cause is confirmed and fixed.** Clamped edges were the cause: with ambient air at the open sides
  and top the side cells cannot feed themselves, and the same model trained with it does not run away over 30 s at any
  wind. The simulator never had the fault. The fix is in the reference, the trainer and the runtime, and old files play
  as before.
- **Long rollouts in training are what make it hold over time.** Stages 2 and 3 with the fix (windows after at most 24
  frames of the stepper's own rollout) still let heat reach the sides over 30 s; 250 iterations with burn-ins of 48 to
  240 frames (half the training's cost) removed that and also improved tracking (+1.9 dB in the picture at 1 s against
  C3) and every endless statistic's mean.
- **The rule's edge bound was too tight for its reference.** The "holds too much" bound used one simulated run per wind
  as the teacher's range; the downwind side's heat varies from run to run (C4's largest there was 5.29 on validation's
  seed and 2.37 on the test's). Study M2 learned the same about margins set tighter than the measures' spread. Next
  time: calibrate such bounds on several simulated runs (the teacher's own spread) before writing them into the rule.
  The rule stands as written: C4 does not pass, and study N's model stays the demo.
- **Whether to adopt C4 anyway is the owner's call.** Everything measured favours it over study N's model (tracking,
  four endless statistics, no runaway on either seed set, beating holding the first frame), and its file and demo
  renders are ready in the data root. Adopting it would set aside a rule written before the numbers; this study does
  not.
- **What still falls short**: it is calmer than the real fire (motion 0.68 to 0.75 of the real runs'), it follows a
  live wind change slowly, its flame base is blocky, and the teacher is still our own solver. Study N's smoke model
  also runs away given 30 s.

### 14.12 Cost

**Re-timed on a quiet machine** (round 7 of the quiet re-timing, 21 October 2026,
[results/quiet/README.md](../results/quiet/README.md); nothing else running): `nvfx_3d timing` on C4 as below, six
rounds, each with study hres's fire and study R's smoke in turn, on 1 and 4 threads; the median of the rounds' medians
and its 95% interval from order statistics ([volume_summary.csv](../results/quiet/round7/volume_summary.csv)). The
volume API's step has used the vectorised detail layer since 21 October (REPORT.md §19), which is most of the change
from the first measurement (in brackets where it differs by more than 15%: `results/3d/n2/timing.csv`, at nice 15 on the
shared machine at a load of 1.0 to 1.7, the scalar detail layer). Milliseconds per frame:

| what | 1 thread | 4 threads |
|---|---:|---:|
| a step (one frame) | **21.0** [18.8, 22.2] (was 70) | **9.0** [8.5, 9.3] (was 23) |
| a picture, 256 x 256 | 55.9 [51.0, 57.4] | 14.1 [13.5, 14.7] |
| a picture, 512 x 512 | 200 [190, 215] | 51.6 [49.6, 53.7] |
| a start (the start point and the 24-frame warm-up) | 546 [517, 584] (was 1,866) | 224 [215, 241] (was 602) |

(Two threads were not re-timed; the first measurement gave a step 37 ms, pictures 30 and 110 ms and a start 953 ms.) On
one thread a step's stages with the reference detail layer (`rollout3d::detail_step`, which the volume API no longer
runs) are: the noise inputs 2.2 ms, the coarse step 10.5, the reference detail layer 50.2 (the vector one: about 10,
REPORT.md §19.2), the renderer's preparation 14.9, and a 256 x 256 picture 43.1; the simulator's frame is 154 ms at 64
x 96 x 64 and 5.6 ms at 16 x 24 x 16. The ambient edges cost nothing measurable (the coarse step 10.5 ms, 12 in the
first measurement, study N's 15 on a busier machine). **Threads scale**: a step is 2.3 times faster on four threads than
on one and a picture 4 times (where the step scales less was not taken apart). Training cost is in §14.6.

### 14.13 Reproduce

Data under `NEURALVFX_DATA/n2_3d` (default `/root/nvfx-data/n2_3d`); every step resumes from what it has written (the
job scripts used are in the data root's `jobs/`). Study N's training runs are read from its own root (`--also`).

```sh
B=build/nvfx_3d; N2="--study n2"; ALSO="--also /root/nvfx-data/n_3d/runs/train"; R=results/3d/n2; W="--winds 0.05,0.25,0.5,0.75,0.95 --seconds 30"
$B record $N2 --runs 24 --frames 540 --threads 2               # the 24 new runs (--from 12 for a second job)
$B train $N2 --name n2 $ALSO --iterations 900 --finetune 350 --activity 250 --long 250 --batch 8 --unroll 12 --burn 24 \
   --burn-long-min 48 --burn-long-max 240 --seed 9200000 --threads 3            # snapshots ckpt/n2_stage1..4.ckpt
for k in 2 3 4; do $B finish $N2 --ckpt n2_stage$k --out C$k $ALSO; done
cp /root/nvfx-data/n_3d/models/chosen.nvfx3 models/N.nvfx3; $B reedge --model models/N.nvfx3 --edges ambient --out models/Nfix.nvfx3
$B truth $N2 --split val; $B real $N2 --split val
$B edge $N2 --sim $W --out results/edge_sim.csv                  # the simulation's sides (seeds 9,510,900 + j)
for m in N Nfix C2 C3 C4; do $B edge $N2 --model models/$m.nvfx3 --label $m $W --out results/edge_val_$m.csv; done
for m in N Nfix C2 C3 C4; do $B track $N2 --split val --method $m --model models/$m.nvfx3; $B play $N2 --split val --method $m --model models/$m.nvfx3; done
$B track $N2 --split val --method coarse --model models/C4.nvfx3; $B play $N2 --split val --method coarse --model models/C4.nvfx3
$B track $N2 --split val --method hold --model models/N.nvfx3
$B edgecheck --csv $R/edge_val.csv                               # edge_sim.csv and the models' rows in one file
$B compare $N2 --split val --a C4 --b N --results $R             # and the other pairs (jobs/val_cands.sh)
$B choose $N2 --split val --cands C2,C3,C4 --edge $R/edge_val.csv
cp models/C4.nvfx3 models/chosen.nvfx3; $B freeze --model models/chosen.nvfx3 --frozen $R/frozen.csv; $B freeze --model models/N.nvfx3 --frozen $R/frozen.csv
$B truth $N2 --split test; $B real $N2 --split test              # the test (jobs/test.sh), once:
$B edge $N2 --model models/chosen.nvfx3 --label learned --seed-base 9820900 $W --out results/edge_test_learned.csv
$B track $N2 --split test --method learned --model models/chosen.nvfx3 --frozen $R/frozen.csv   # and play; N, coarse, hold
$B compare $N2 --split test --a learned --b N --results $R; $B compare $N2 --split test --a learned --b coarse --results $R
$B timing $N2 --model models/chosen.nvfx3 --threads 1,2,4 --out $R/timing.csv
$B video --kind wind --wind-fixed 0.05 --shard 0 --seconds 20 --width 640 --height 360 --controls 0.7,0.05,0.5 --seed 9950010 --model models/C4.nvfx3 --out fig_C4.mp4   # the figure (and N, Nfix)
```
