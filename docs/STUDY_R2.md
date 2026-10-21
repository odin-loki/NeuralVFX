# Study R2: the 3D smoke again, on a teacher with its momentum fixed

Status: **under way** (21 October 2026). Audience: dev, research, the owner. A second try at study R's learned 3D smoke
([STUDY_R.md](STUDY_R.md)): first the teacher's momentum deficit (study R §1.4: the plume gains only 0.78 of the
momentum its buoyancy gives), found and fixed and the plume measured against Morton, Taylor and Turner again (§1); then
the learned smoke retrained on the fixed teacher with study N2's recipe and two losses aimed at study R's two failures
(the edge check and the over-spreading), under a new rule whose margins are calibrated on the teacher's own runs,
committed before validation and tested once (§2 onwards).

## 0. Where the study stands (a handoff, kept current)

As of 21 October 2026, 18:20 UTC: the code is written and unit-tested (commit 962c538: the momentum audit, the velocity
in flux form and `smoke_r2`, the trainer's spread and edge losses, the rule's margins and verdict); the teacher's
diagnosis (§1.2) waits for the quiet machine (another agent's timing round), as the brief asked.
- Data root `$NEURALVFX_DATA/r2_smoke3d` (default `/root/nvfx-data/r2_smoke3d`): job scripts in `jobs/`, logs in
  `logs/`, scratch in `scratch/`. Every job resumes from what it has written.
- Next: §1.2's runs (`jobs/diag.sh`, started by `jobs/wait_quiet.sh` when `quiet7/QUIET_DONE` appears or at 20:40 UTC).

## 1. The teacher's momentum

### 1.1 What study R found

Study R's teacher, `smoke_r` (study N's smoke without its sinks, heat and soot carried in flux form), is a plume by
MTT's conserved flux and its spreading rate, but **gains only 0.78 [0.74, 0.83] of the vertical momentum its buoyancy
gives** (dM/dz against the plane's buoyancy, heights 24 to 92, at every source strength), its flux-balance parameter
stays at 2.5 to 3.7 (a lazy plume) and its centreline velocity is about a third below MTT's for its flux. Study R put it
down to the semi-Lagrangian velocity step's numerical dissipation; MacCormack velocity, which would have helped, was
unstable at 64 x 96 x 64 (STUDY_R.md §1.2). Nothing measured which operator loses the momentum.

### 1.2 Where the momentum goes: a budget per operator

In a steady plume the vertical momentum of a slab between two heights is fixed: what the advection carries in through
its bottom and out through its top (the momentum flux M = the plane's sum of w^2, including the turbulent part), what
the buoyancy adds (the plane's sum of g' over the slab), what the pressure takes (the plane's sum of p at the top minus
at the bottom), and what the other forces add (vorticity confinement, the curl noise; no damping in `smoke_r`). So
dM/dz = B only if the pressure's part is small **and the advection conserves momentum**. `sim3d::MomentumAudit` sums each
operator's change of the slab's vertical momentum as the solver runs (it only reads: every effect's fields are the same
with it; tested), `nvfx_3d plume` records it with each plane's pressure, and `plumefit --momentum` gives each term as a
fraction of what the buoyancy adds, with intervals by resampling the runs.

**A first look at half the resolution** (21 October, 18:30 UTC, before any run at 64 x 96 x 64): one exploration seed
(34,000,500) on 32 x 48 x 32, controls (0.5, 0.5, 0.5), 4 s of spin-up and 6 s averaged, the slab from 12 to 36 cells
(`scratch/x` in the data root; one run each, so no intervals). Each term as a fraction of what the buoyancy adds:

| solver | dM/dz over B | advection | over the planes' net flux | projection (pressure) | confinement | curl noise | stored |
|---|---:|---:|---:|---:|---:|---:|---:|
| `smoke_r` (semi-Lagrangian velocity) | 0.66 | -0.77 | 1.18 | +0.16 | **-0.30** | -0.07 | +0.02 |
| velocity in flux form | 0.73 | -0.75 | 1.00 | +0.18 | **-0.32** | -0.06 | +0.05 |
| flux form, confinement momentum-neutral | **1.02** | -1.04 | 1.00 | +0.22 | 0.00 | -0.06 | +0.12 |
| flux form, no confinement | 1.06 | -1.08 | 1.00 | +0.24 | 0 | -0.06 | +0.10 |

**Two operators take the momentum, and the larger is not the one study R suspected.**
- **Vorticity confinement pushes the plume down**, by about 0.3 of what its buoyancy gives. Its force, epsilon N x
  omega with N the unit vector up the gradient of |omega|, speeds the inner side of the plume's shear layer and slows
  the outer side; but N is a unit vector whatever the vorticity's size, so the whole outer annulus, much larger in area
  than the inner disc, pulls down at full strength. For a Gaussian plume (w = W exp(-r^2 / b^2), omega = 2 W r exp(-r^2
  / b^2) / b^2, peaking at r = b / sqrt 2) the plane's net force is epsilon times 4 pi W b times (0.087 - 0.356), a
  third of a typical buoyancy here. It is an artefact: a turbulent stress changes a plane's momentum flux only through
  the flux itself (it is a divergence), never as a net body force. Study R saw its trace without knowing it: the
  momentum lost rose with the turbulence control (0.85 at 0, 0.78 at 0.5, 0.72 at 1), which sets the confinement.
- **The semi-Lagrangian step is not conservative**: its change of the slab's momentum is 1.18 times what the planes'
  momentum flux carries out (in the flux form, 1.00 to rounding).
- The pressure gives some back (+0.16 to +0.24): P, the plane's sum of the pressure, falls with height, as it must where
  the plume draws the air in at its sides; it is part of the physical momentum flux (M + P), not an error.

So the candidate of §1.3 was revised before any run at full resolution: **the velocity in flux form and the confinement
momentum-neutral** (`Params::confinement` 1): the same force at every cell, less its horizontal Gaussian spread (8
world units, 4 cells at 64 across; each cell's force spread over its plane with weights that sum to one), so that it
still sharpens the vortices it sees and adds no net momentum to any plane (tested: its term in the audit is zero to
rounding). The criterion of §1.3 is unchanged and applies to the revised `smoke_r2` at 64 x 96 x 64.

**At full resolution**: two runs each of `smoke_r`, of the velocity in flux form alone (`smoke_r2` with `--confinement
0`) and of `smoke_r2` (exploration seeds 34,000,000 and 34,000,001, intensity 0.5, wind 0.5, turbulence 0.5, 6 s of
spin-up and 16 s averaged, as study R's). (Results: to come.)

### 1.3 The fix, and the criterion written before its answer

**The candidate**: the velocity carried in **flux form** (`Params::velocity` 2), each component exactly as the scalars'
flux form carries heat (second-order upwind with the monotonised central limiter, one sweep per axis in alternating
order, the face velocity the mean of its two cells', substeps where a sweep's Courant number would pass 0.95), by the
velocity after the forces, frozen for the step; the border is the velocity's own (outflow at the sides and the top, the
floor reflecting v, so no momentum crosses it). What leaves a cell enters its neighbour: the advection conserves
momentum (tested: its change of a slab's momentum is its faces' fluxes to float rounding). The limiter keeps the sweeps
total-variation diminishing, which is where MacCormack's correction (not conservative, and only clamped) went unstable.
It is a new preset, **`smoke_r2`**: `smoke_r` with this velocity step (and, after §1.2's first look, the confinement
momentum-neutral) and nothing else changed. The fire, study N's
smoke and `smoke_r` are unchanged (their recorded runs compared byte for byte, §1.5).

**The criterion** (written 21 October, 18:20 UTC, before any run of the fix): `smoke_r2` becomes study R2's teacher if,
on §1.2's two exploration seeds, (a) its momentum budget closes where `smoke_r`'s does not: dM/dz over B within 0.90 to
1.10 (study R's criterion), **or**, if the audit shows the pressure takes a part, d(M + P)/dz over B within 0.90 to 1.10
with the advection's change within 5% of the momentum flux its planes carry; (b) its buoyancy flux is still conserved
(the planes' flux within 10% of its mean from 24 to 92 cells); and (c) it is stable: finite, with no speed growing
over 30 s, at four extreme settings (intensity 1, turbulence 1, no wind; intensity 0.8 with the wind at 0 and at 1;
intensity 0.2 with no turbulence). If it fails, the audit says which operator to look at next (the projection, or finer
time steps); every such try is reported. The final measurement (§1.6) uses seeds of its own, 34,010,000 + 100 c + k.

### 1.4 Stability

(To come.)

### 1.5 The other effects, unchanged

`jobs/bitcheck.sh`: the fire, study N's smoke and study R's `smoke_r` (one run each), and study N2's two kinds of fire
run (controls held and controls that change), 2 s each at 64 x 96 x 64 from an empty grid, recorded by main's tool
(fc3d5f6, built clean from that commit) and by this branch's (3cdee7c, with the audit, the flux-form velocity and the
momentum-neutral confinement compiled in): **all five run files identical, byte for byte** (3,686,464 and 3,687,192
bytes). The new code paths run only for `smoke_r2` or when asked for (`Params::velocity` 2, `Params::confinement` 1 or
2, an audit given).

### 1.6 The plume against Morton, Taylor and Turner, again

(To come: study R §1.3's measurement and fits on `smoke_r2`, seeds 34,010,000 + 100 c + k.)

## 2. The learned smoke: design and rule (written and committed before validation)

Everything in this section is fixed, and committed, before any validation number exists; the edge bounds (§2.7) and the
margins (§2.7) are computed from the teacher's runs alone and committed with it (`results/3d/r2/edge_bounds.csv`,
`margins.csv`). The results follow in §3 onwards.

### 2.1 The question

Study R's question again, on the fixed teacher: does study N2's recipe, with two losses aimed at study R's failures,
give a 3D smoke that on held-out settings and fresh seeds follows its teacher better than running the same simulation
on the coarse grid with the same seeded detail, plays **as realistically within margins set by the teacher's own
run-to-run spread**, and never runs away at the box's open sides over 30 s at any wind?

### 2.2 The teacher's runs (data)

`smoke_r2` (§1) on 64 x 96 x 64, recorded as studies N2 and R record (block averages on the 16 x 24 x 16 coarse grid,
every frame, fp16): **24 runs of 8 s** at random controls held (seeds 34,100,000 + i, `runs/short`) and **24 runs of
18 s** of N2's three kinds by i mod 3 (the wind at an extreme, controls that change, random controls held; seeds
34,200,000 + i, `runs/long`): 9.6 minutes of simulated smoke, as study R had (the shared disk allows no more).

### 2.3 The model

Study N2's model unchanged in size and form, as study R's: a 16 x 24 x 16 learned state (velocity, heat, soot and 4
memory channels), two 3 x 3 x 3 convolutions of 16 channels with FiLM on the three controls, 14,218 weights, a projection
and advection with ambient air at the open sides and top, driving seeded 64 x 96 x 64 detail with N2's constants
(contrast 1, swirl 0.5, swirl length 12, grow 1.5). Initialised with seed 34,100,777; 8 start points at frame 150 by
farthest-point sampling over the controls of all 48 runs; effect `smoke3d`, warm smoke's optics.

### 2.4 Training, and the two losses aimed at study R's failures

Study N2's four stages, as study R ran them, with three changes:
- **Longer edge-aware rollouts**: stage 4's windows start after **48 to 480 frames** (16 s) of the stepper's own rollout
  (study R: 48 to 240), from the 18 s runs. Study R's C4 failed its edge check by a slow build-up at one side over 30 s
  (0.46 over 5 to 15 s, 0.83 over 20 to 30 s); windows after up to 16 s of its own rollout let the stepper see more of it.
- **The edge loss** (`TrainOptions::edge`, weight 10, stages 2 to 4): the heat and soot in the coarse cells of the four
  open sides beyond the truth's, squared, in channel scales, over the sides' cells (a hinge: only an excess counts; the
  top is left out, since the plume leaves by it). The mean squared error already counts these cells, at the weight of
  every other cell; this makes an excess at the sides about 17 times as costly, where study R's bound sat.
- **The spread loss** (`TrainOptions::spread`, weight 2, stages 2 to 4): for heat and soot, each row's mass-weighted
  spread about its own centroid (cells^2), the model's against the truth's, relative to the truth's plus one cell^2,
  squared, the rows weighted by the truth's mass in them. A plume that meanders is not punished for being elsewhere (the
  spread is about the row's own centre), and a broad haze is. Study R's learned smoke spread into a broad, diffuse cone
  (coverage, light and the mean picture worse); a squared error against one chaotic run, after a long burn-in, rewards
  the smooth average of its possible futures, which is wider than any one of them. The spread loss is a statistic of
  each frame, so it rewards the narrow instantaneous plume instead (the idea of studies D2 and M3's statistics in 2D).

**How wide study R's learned plume grows** (a probe on study R's own training runs, before any R2 training; `nvfx_3d
widthprobe`, the 16 of its 18 s runs whose controls are held, from 2 s on, the coarse state alone): each row's spread
about its centroid over the truth's, weighted by the truth's mass in the row, and the coarse state's mass over the
truth's:

| frames played on | 12 | 30 (1 s) | 90 | 180 | 300 | 450 (15 s) |
|---|---:|---:|---:|---:|---:|---:|
| study R's C4: spread over the truth's (soot) | 1.74 | 2.47 | 3.08 | 3.06 | 3.78 | 5.10 |
| study R's C4: mass over the truth's (soot) | 0.79 | 0.69 | 0.73 | 1.17 | 1.50 | 1.70 |
| the coarse simulation: spread over the truth's | 1.04 | 1.01 | 1.07 | 1.26 | 1.15 | 1.10 |
| the coarse simulation: mass over the truth's | 0.96 | 0.94 | 0.84 | 0.77 | 0.77 | 0.74 |

So the over-spreading is in the learned coarse state, and it grows the longer the model plays: half again as wide
within 12 frames, two and a half times within a second, five times after 15 s, while a haze builds up (the mass 1.7
times the truth's). The coarse simulation, the teacher's own equations, keeps the plume's width. This is what the
spread loss is for, and its weight follows from it: at R's error (a relative spread error near 1 at 1 s) it is about 2
per window, twice the rest of the loss; as the width comes right it fades.

Otherwise as study R: stage 1, 900 iterations of windows from true states (the unroll growing to 12); stage 2, 350
iterations with half the windows after up to 24 frames of the stepper's own rollout and the profile loss; stage 3, 250
with the activity loss; stage 4, 250 of long rollouts. Batch 8, Adam, learning rates 2e-3 and 7e-4 (cosine), trainer
seed 34,200,000, a checkpoint every 20 iterations and a snapshot at the end of every stage. **Candidates**: C2, C3 and
C4, the stepper after stages 2, 3 and 4, with the same start points and detail constants. Both losses' gradients are
checked against finite differences (`Rollout3d.GradientsMatchFiniteDifferencesWithTheSpreadLoss`, `...EdgeLoss`,
`...EveryLoss`).

### 2.5 Seeds and held-out settings

Seeds (this study owns 34,000,000 to 34,999,999): the plume measurement 34,000,000 to 34,010,899 (§1; exploration
34,000,000 + 100 c + k, the final measurement 34,010,000 + 100 c + k); training 34,100,000 + i and 34,200,000 + i; the
edge bounds' teacher runs 34,300,000 + 100 j + k (j the wind, k the run); the margins' teacher runs 34,500,000 + 100 s +
k (s the calibration setting, k = 0 to 8); validation tracking 34,400,000 + 100 s + k, real runs 34,410,000 + 100 s + k,
plays 34,420,000 + 100 s + k, the edge check's plays 34,430,000 + j; test tracking 34,700,000 + 100 s + k, real runs
34,710,000 + 100 s + k, the floor 34,710,050 + 100 s + k, plays 34,720,000 + 100 s + k, the edge check's plays
34,730,000 + j; demo videos 34,900,000 and up; unit tests 34,099,000 to 34,099,999.

Held-out settings (intensity, wind, turbulence), none a training run's or study R's, with a strong wind towards -x and
towards +x in each split:
- validation (6): (0.50, 0.08, 0.55), (0.40, 0.90, 0.35), (0.65, 0.30, 0.70), (0.30, 0.60, 0.45), (0.75, 0.45, 0.25),
  (0.55, 0.75, 0.85);
- test (8): (0.45, 0.04, 0.40), (0.60, 0.96, 0.60), (0.35, 0.20, 0.25), (0.70, 0.55, 0.80), (0.80, 0.80, 0.35),
  (0.25, 0.40, 0.65), (0.55, 0.65, 0.10), (0.90, 0.30, 0.50);
- calibration of the margins (5, teacher runs only): (0.50, 0.15, 0.50), (0.40, 0.85, 0.70), (0.70, 0.50, 0.30),
  (0.30, 0.35, 0.85), (0.85, 0.65, 0.60).

Validation: 3 tracking runs per setting (18), 2 real runs and 2 plays of each method pooled per setting. Test: 3
tracking runs per setting (24), 3 real runs, 3 plays of each method and 3 other real runs (the floor) per setting. The
test has two settings more than study R's, so that its intervals are narrower.

### 2.6 Methods and measures

As study R (STUDY_R.md §2.6), unchanged: the candidates (the chosen one on the test: "learned"); **the coarse simulation
with the same detail** (`smoke_r2` itself on 16 x 24 x 16, its lengths and forces scaled, driving the same seeded detail
layer); for reference only, holding the first frame (tracking) and other real runs (the floor, test only). Tracking from
the true state at 8 frames, 1 s and 2 s (the heat field PSNR, peak 0.5; the soot field PSNR, peak 1.2; the picture PSNR
from four cameras); the five endless statistics (the 3D spectrum distance, |ln motion ratio|, the coverage distance,
the light distance, the mean picture PSNR). Differences: method minus the coarse simulation, paired bootstrap of the mean
difference, 10,000 resamples, seed 2026, 95% percentile intervals; tracking paired over runs, the endless statistics
over settings.

### 2.7 The margins and the edge bounds, calibrated on the teacher's own runs

**The margins of the look** (`nvfx_3d real --split cal`, `nullcal`). Study R judged the look strictly: a statistic was
worse if its interval lay entirely on the worse side, however small the difference. A difference smaller than what
drawing new seeds of the teacher itself does is one no one could tell from a re-seed (study M3's reasoning, REPORT
§14.9.1). So at each of the 5 calibration settings, **9 teacher runs** (three sets of three, the test's pooling), each
set in turn the reference and the other two compared with it as if they were two methods: their difference in each
endless statistic is what chance alone moves a paired difference by at one setting when both "methods" are as good as
the teacher. **Each statistic's margin is the RMS of those differences** over the 5 settings and 3 rotations (15
values), written from the teacher's runs alone before any model is validated (`results/3d/r2/margins.csv`; the numbers
in §2.10). A statistic is **worse beyond its margin** if its interval's worse end lies beyond the margin: for the
spectrum, motion, coverage and light distances an upper end above +margin, for the mean picture a lower end below
-margin. Study R's strict reading is reported beside it and does not decide.

**The edge check** (study R's, STUDY_R.md §2.7, its form unchanged): the model played in one rollout from the start
point nearest (0.7, w, 0.5), its warm-up and then 30 s, at constant winds 0.05, 0.25, 0.5, 0.75 and 0.95, the coarse
heat summed over each side slab and the top every 0.5 s; **four teacher runs per wind** (seeds 34,300,000 + 100 j + k);
per wind and side the model *holds too much* if its largest over 5 to 30 s exceeds 2 x the teacher runs' largest + B,
and *grows* if its mean over 20 to 30 s exceeds 2 x its mean over 5 to 15 s + C; B = 1 and C = 0.5, raised to 1.5 times
the smallest value that clears every teacher run checked against the other three (leave one out). Bounds in
`results/3d/r2/edge_bounds.csv`. **A model passes if no side runs away at any wind.**

### 2.8 The choice (validation)

`nvfx_3d choose --study r2`:
1. **Eligible**: the candidates that pass the edge check with validation's seeds; if none does, all three, and the
   chosen one fails the test's first condition.
2. Of those, **the ones whose look is within the margins** against the coarse simulation on validation (no endless
   statistic worse beyond its margin); if none is, the ones with the fewest statistics beyond their margins.
3. Of those, **the highest tracking picture PSNR at 1 s** (the mean over the 18 runs); within 0.1 dB, the higher heat
   field PSNR at 1 s.

The chosen file is frozen by SHA-256 in `results/3d/r2/frozen.csv` before the test; the tool refuses the test's learned
runs on any other file.

### 2.9 The test and the verdict (it runs once)

`nvfx_3d verdict --study r2`: the chosen model on the test's settings and seeds against the coarse simulation with the
same detail:
1. **No runaway**: it passed the edge check on validation (step 1 of the choice) and passes it with the test's seeds
   (34,730,000 + j).
2. **Tracking**: it *follows a real run better* if, at 1 s, the heat field PSNR's and the picture PSNR's differences
   both have intervals entirely above zero; *worse* if both entirely below; otherwise a *tie* (or mixed).
3. **Look**: it *plays as realistically* if no endless statistic is worse beyond its margin (§2.7).

**The verdict**: "**passes: the learned 3D smoke beats running the cheap simulation**" if 1, 2 (better) and 3 hold;
otherwise "**fails**", naming each condition that does not hold, with every number. Study R's strict reading of the
look, soot, 8 frames and 2 s, holding the first frame and the floor are reported beside it, not judged. **If it passes,
it ships as `models/3d/smoke.nvfx3`**, replacing study R's demo (`models/layout.txt`, `SHA256SUMS`, the data root's
`r2_smoke3d/models/chosen.nvfx3`, `neuralfx.Smoke3dShippedModel` checking it); if it fails, study R's demo stays and the
numbers are reported.

**Cost** (reported, not judged; provisional on the shared machine), as study R.

### 2.10 The numbers the rule uses

(To come, from the teacher's runs, before validation: the margins and the edge bounds.)

## 7. Reproduce

(To come.)
