# Study R2: the 3D smoke again, on a teacher with its momentum fixed

Status: **under way, resumable** (22 October 2026; §0 says how to resume). Audience: dev, research, the owner. A second try at study R's learned 3D smoke
([STUDY_R.md](STUDY_R.md)): first the teacher's momentum deficit (study R §1.4: the plume gains only 0.78 of the
momentum its buoyancy gives), found and fixed and the plume measured against Morton, Taylor and Turner again (§1); then
the learned smoke retrained on the fixed teacher with study N2's recipe and two losses aimed at study R's two failures
(the edge check and the over-spreading), under a new rule whose margins are calibrated on the teacher's own runs,
committed before validation and tested once (§2 onwards).

## 0. Where the study stands (a handoff, kept current)

As of 22 October 2026, 03:25 UTC: **under way, resumable; the session ended here** (the owner's usage), with the jobs
left running unattended. **Resume after any restart with `/root/nvfx-data/r2_smoke3d/jobs/resume.sh`** (it restarts only
what is not done and leaves running jobs alone).
- **Done**: the teacher fixed and chosen (§1.2 to §1.4: `smoke_r2`, dM/dz over B 1.088 [1.080, 1.095], from `smoke_r`'s
  0.80); the old effects byte for byte (§1.5); the 48 training runs (`runs/`, 1.1 GB); the edge bounds' 20 teacher runs
  (`results/edge_teacher_w*.csv`).
- **Running**: training (stage 4, from its checkpoint at iteration 1520 of 1750; `logs/train_r2.log`; it then writes C2
  to C4 and `markers/candidates_done`); the margins' last 9 of 45 teacher runs and the plume measurement's 17 of 18
  (`jobs/runq.sh jobs/teacher_tasks.txt`).
- **Two container restarts** (about 02:49 and 03:10 UTC) killed every job; every output is written whole (a run's rows
  at once, units, profiles and checkpoints through a temporary file renamed), so nothing partial is used; training
  resumed from its checkpoints and reproduced its logged losses exactly (iterations 1500 and 1510: 9.439541, 1.363902).
  It now checkpoints every 10 iterations.
- **Left, in order** (`resume.sh` runs each stage when the one before is done, the hand steps marked): (1) the rule's
  numbers, `jobs/rule_numbers.sh` (run by `resume.sh` once the teacher's runs are whole: `results/3d/r2/edge_bounds.csv`,
  `margins.csv`); by hand, §2.10 written, committed and pushed **before validation**, then `touch
  markers/rule_committed`; (2) validation, `jobs/val.sh` (by `resume.sh`); by hand, `nvfx_3d choose --study r2`
  (results/3d/r2/choice_val.md), `cp models/<chosen>.nvfx3 models/chosen.nvfx3`, `echo <chosen> > models/chosen.txt`,
  `nvfx_3d freeze --model ... --frozen results/3d/r2/frozen.csv`, committed and pushed; (3) the test once,
  `jobs/test.sh` (by `resume.sh`), its verdict in `results/3d/r2/verdict_test.md`; ship only if it passes (§2.9);
  (4) §1.6's fits (`plumefit`, §7), the videos and the write-up.
- Data root `$NEURALVFX_DATA/r2_smoke3d` (default `/root/nvfx-data/r2_smoke3d`): job scripts in `jobs/`, logs in
  `logs/`, binaries in `bin/` (training and the teacher's runs on cb31ccb, validation and the test on a5637ed: the same
  solver and trainer), markers in `markers/`. Each job checks the disk first (`jobs/disk.sh`: R2 stays under 2.5 GB).
- **Nothing shipped changed**: `models/` and every frozen hash are main's; study R's demo stays `models/3d/smoke.nvfx3`.

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

**At full resolution** (`jobs/diag.sh`, 20:40 to 21:30 UTC; `results/3d/r2/diag_fit.md`): two runs each of `smoke_r`,
of the velocity in flux form alone (`smoke_r2 --confinement 0`) and of `smoke_r2` (exploration seeds 34,000,000 and
34,000,001, intensity 0.5, wind 0.5, turbulence 0.5, 6 s of spin-up and 16 s averaged, as study R's), the slab from 24
to 92 cells; 95% intervals by resampling the two runs (narrow: two runs understate the spread):

| solver | dM/dz over B | d(M + P)/dz over B | advection over the planes' flux | projection | confinement | curl noise | stored | turbulent share of M |
|---|---|---|---|---:|---:|---:|---:|---:|
| `smoke_r` | **0.80** [0.78, 0.82] | 0.76 | 1.12 | +0.04 | **-0.18** | -0.01 | -0.05 | 0.39 |
| flux form alone | 1.03 [1.00, 1.06] | 0.91 | 1.02 | +0.12 | **-0.17** | -0.01 | -0.13 | 0.42 |
| **`smoke_r2`** | 1.12 [1.10, 1.14] | **1.035** [1.027, 1.043] | 1.01 [0.97, 1.04] | +0.08 | **0.00** | -0.01 | -0.05 | 0.35 |

- `smoke_r` repeats study R's deficit on fresh seeds (0.80 against R's 0.78 [0.74, 0.81]). At this resolution the
  confinement takes 0.18 of what the buoyancy gives and the semi-Lagrangian step's non-conservation about 0.1 (its
  change of the slab's momentum is 1.12 times the planes' net flux).
- The flux form alone comes out near 1 by a balance of errors: the confinement still pushes down by 0.17, and the
  pressure (+0.12) and the slab's loss of stored momentum over the averaging (0.13) make it up.
- **`smoke_r2`** has no net push from its confinement, its advection carries what its faces carry (1.000 to rounding by
  the faces' own fluxes, 1.01 by the cell-centred estimate of the planes' w^2), and its momentum flux including the
  pressure grows by what the buoyancy gives: **d(M + P)/dz over B = 1.035**. Without the pressure, 1.12: the pressure
  adds 0.08 (it falls with height, as a plume drawing in air at its sides requires) and the slab gave up 0.05 of stored
  momentum while it was averaged.
- About a third of M is the turbulent part (w'^2; 0.35 to 0.42): the solver's plume is much more turbulent than a
  laboratory plume's 10 to 20%, so its mean centreline velocity sits lower for the same momentum flux (§1.6).
- Its buoyancy flux stays within +3.3% and -1.7% of its mean from 24 to 92 cells (`smoke_r` +3.3 / -2.3%).
- **Its flux-balance parameter now falls with height**, from 2.68 at 24 to 28 cells to 1.84 at 88 to 92, as a lazy
  plume's must on its way to a pure plume (Gamma = 1); `smoke_r`'s rose (3.26 to 3.52), a plume losing momentum.

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

### 1.4 Stability, a third revision, and the decision

`nvfx_3d stability` (`jobs/diag.sh`): `smoke_r2` from an empty grid for 30 s at the four extreme settings of §1.3, once a
second the largest speed and the box's kinetic energy per cell (`logs/job_diag.log` in the data root). All four stayed
finite and no largest speed grew (the largest over 1 to 10 s against 21 to 30 s: 150 and 132, 129 and 111, 136 and 150,
164 and 125 cells per second). **But at intensity 0.2 with no turbulence the box's kinetic energy rose without
levelling**: about 11 per cell from 5 to 20 s, 29 at 30 s, 52 at 45 s, where `smoke_r`'s stayed at 6 to 10. The mean
vertical velocity in the side cells had turned downward (-0.5 to -1.6 cells per second): a recirculation, up in the
plume and down at the sides, growing. Its cause is the border: the velocity's flux form took what flows in at an open
side from the edge cell (outflow's zero gradient), so a downflow at the side came back in with the air the plume draws
in and fed itself; the semi-Lagrangian step's dissipation had hidden this. The physical far field is still air: **the
entrained air brings no velocity along the face it comes through** (the component through the face is the inflow
itself). With that (`smoke_r2` from cb31ccb on), the same run's kinetic energy stays at 10 to 19 per cell to 45 s and
the side cells' mean vertical velocity at 0.0 to 0.07. The budget and the stability were then run again on this
version (`jobs/diag2.sh`, 22:20 to 22:48 UTC; `results/3d/r2/diag_fit.md`, row `d_r2b_i50`), the same two exploration
seeds:

| `smoke_r2` (cb31ccb) | dM/dz over B | d(M + P)/dz over B | advection over the planes' flux | projection | confinement | stored | turbulent share of M |
|---|---|---|---|---:|---:|---:|---:|
| 2 runs | **1.088** [1.080, 1.095] | **1.005** [1.002, 1.007] | 1.002 [1.001, 1.002] | +0.07 | 0.000 | -0.01 | 0.35 |

Its buoyancy flux stays within +3.1% and -1.6% of its mean from 24 to 92 cells; top-hat alpha 0.157 [0.150, 0.163];
the flux-balance parameter falls from 2.77 to 1.96 up the fitted range. The four stability runs stay finite with no
largest speed growing (over 1 to 10 s against after 20 s: 150 and 172, 142 and 115, 158 and 150, 164 and 143 cells per
second) and the kinetic energy bounded (45 and 46, 28 and 26, 27 and 27, and at intensity 0.2 with no turbulence 11
and at most 22, levelling at 10 to 19 to 45 s).

**By the criterion of §1.3, `smoke_r2` (as of cb31ccb) is study R2's teacher**: (a) its momentum budget closes where
`smoke_r`'s does not (dM/dz over B = 1.088, within 0.90 to 1.10; with the pressure 1.005; the advection within 0.2% of
the momentum flux its planes carry); (b) its buoyancy flux is conserved (within 3.1% of its mean); (c) it is stable.
Decided at 22:49 UTC, before any training run or the final measurement. (The version before the still inflow, 3cdee7c,
met (a) only by the criterion's second branch, 1.12 and 1.035, and grew the recirculation above; it was not used.)
The three changes from `smoke_r`, each found by the audit or the stability check and each tested: the velocity in flux
form, the confinement momentum-neutral, still ambient inflow for the velocity. Its cost: about 0.4 CPU-seconds a frame
at 64 x 96 x 64 on one thread on this machine, against `smoke_r`'s 0.32.

### 1.5 The other effects, unchanged

`jobs/bitcheck.sh`: the fire, study N's smoke and study R's `smoke_r` (one run each), and study N2's two kinds of fire
run (controls held and controls that change), 2 s each at 64 x 96 x 64 from an empty grid, recorded by main's tool
(fc3d5f6, built clean from that commit) and by this branch's (3cdee7c, with the audit, the flux-form velocity and the
momentum-neutral confinement compiled in): **all five run files identical, byte for byte** (3,686,464 and 3,687,192
bytes). The new code paths run only for `smoke_r2` or when asked for (`Params::velocity` 2, `Params::confinement` 1 or
2, an audit given).

### 1.6 The plume against Morton, Taylor and Turner, again

Study R §1.3's measurement and fits, unchanged, on `smoke_r2` with seeds of its own (34,010,000 + 100 c + k): intensity
0.5, 0.2 and 0.8 at turbulence 0.5 (4 seeds each), turbulence 0 and 1 at intensity 0.5 (3 seeds each; study R had 5
and 4, cut for the machine's load as §2.4 says); 6 s of spin-up
and 16 s averaged; the fits over **24 to 92 cells**, study R's range, fixed before these runs: its box study (the same
box and source, a box two thirds taller and one half as wide again) found the top mattering only in the last 10 cells
and the sides narrowing the plume by about 5%, and the solver's change does not touch either, so the larger boxes are
not run again. The wind-bent plume is not run again either (study R §1.6: its departure from Briggs's law is the wind's
model, a drag on the smoke, which this study does not change). (Results: to come.)

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
- **Longer edge-aware rollouts**: stage 4's windows start after **48 to 360 frames** (12 s) of the stepper's own rollout
  (study R: 48 to 240), from the 18 s runs. Study R's C4 failed its edge check by a slow build-up at one side over 30 s
  (0.46 over 5 to 15 s, 0.83 over 20 to 30 s); windows after up to 12 s of its own rollout let the stepper see more of it.
  (16 s was planned; cut to 12 s before training began, with the plume measurement's seeds (§1.6) and the test's floor
  (§2.5), because the shared machine ran at a load of 15 to 20 from 23:00 UTC and the plan's 19 CPU-hours would not have
  fitted.)
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
tracking runs per setting (24), 3 real runs and 3 plays of each method pooled per setting. The test has two settings more
than study R's, so that its intervals are narrower; it runs **no floor** (other real runs, which studies N, N2 and R
reported beside their verdicts and never judged), to save its cost on the busy machine: the margins (§2.7) play the
floor's part, the teacher against itself.

### 2.6 Methods and measures

As study R (STUDY_R.md §2.6), unchanged: the candidates (the chosen one on the test: "learned"); **the coarse simulation
with the same detail** (`smoke_r2` itself on 16 x 24 x 16, its lengths and forces scaled, driving the same seeded detail
layer); for reference only, holding the first frame (tracking). Tracking from
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
look, soot, 8 frames and 2 s and holding the first frame are reported beside it, not judged. **If it passes,
it ships as `models/3d/smoke.nvfx3`**, replacing study R's demo (`models/layout.txt`, `SHA256SUMS`, the data root's
`r2_smoke3d/models/chosen.nvfx3`, `neuralfx.Smoke3dShippedModel` checking it); if it fails, study R's demo stays and the
numbers are reported.

**Cost** (reported, not judged; provisional on the shared machine), as study R.

### 2.10 The numbers the rule uses

(To come, from the teacher's runs, before validation: the margins and the edge bounds.)

## 7. Reproduce

Data under `NEURALVFX_DATA/r2_smoke3d` (default `/root/nvfx-data/r2_smoke3d`); every step resumes from what it has
written; the job scripts used are in its `jobs/` (each takes the tool as `B=...`). Build as the main README says, then:

```sh
B=build/nvfx_3d; R=results/3d/r2; D=$NEURALVFX_DATA/r2_smoke3d; W="--winds 0.05,0.25,0.5,0.75,0.95 --seconds 30"
# §1.2: the momentum audit (exploration seeds), smoke_r, the flux form alone and smoke_r2; the stability (jobs/diag2.sh)
P="$B plume --study r2 --spin 6 --seconds 16"
$P --label d_r_i50 --effect smoke_r --seeds 2 --seed-base 34000000; $P --label d_flux_i50 --effect smoke_r2 --confinement 0 --seeds 2 --seed-base 34000000
$P --label d_r2b_i50 --effect smoke_r2 --seeds 2 --seed-base 34000000
$B plumefit --study r2 --configs d_r_i50,d_flux_i50,d_r2b_i50 --momentum d_r_i50,d_flux_i50,d_r2b_i50 --fit-lo 24 --fit-hi 92 --out $R --md diag_fit.md --csv /dev/null
$B stability --study r2 --seconds 45 --intensity 0.2 --wind 0.5 --turbulence 0 --seed 34000903   # and the other three extremes
# §1.5: the old effects byte for byte against main's tool (jobs/bitcheck.sh)
# §1.6: the plume against MTT (seeds 34,010,000 + 100 c + k; jobs/teacher_lanes.sh), then the fits
$P --label v_i50 --seeds 4 --seed-base 34010000; $P --label v_i20 --intensity 0.2 --seeds 4 --seed-base 34010100
$P --label v_i80 --intensity 0.8 --seeds 4 --seed-base 34010200; $P --label v_t00 --turbulence 0 --seeds 3 --seed-base 34010300
$P --label v_t100 --turbulence 1 --seeds 3 --seed-base 34010400
$B plumefit --study r2 --configs v_i20,v_i50,v_i80,v_t00,v_t100 --strengths v_i20,v_i50,v_i80 --momentum v_i20,v_i50,v_i80,v_t00,v_t100 \
   --budget v_i50,v_t100 --fit-lo 24 --fit-hi 92 --out $R
# §2: the training runs (jobs/records.sh), the rule's numbers (jobs/teacher_lanes.sh, jobs/rule_numbers.sh), training
$B record --study r2 --runs-dir short --runs 24; $B record --study r2 --runs-dir long --runs 24
for j in 0 1 2 3 4; do $B edge --study r2 --sim --runs 4 --winds <wind j> --seconds 30 --seed-base $((34300000 + 100 * j)) --out $D/results/edge_teacher_w$j.csv; done
$B edgecal --study r2 --csv $R/edge_teacher.csv --out $R/edge_bounds.csv          # the five files concatenated
$B real --study r2 --split cal; $B nullcal --study r2 --out $R/margins.csv
RUNS="--runs-dir long --also $D/runs/short"
$B train --study r2 --name r2 $RUNS --iterations 900 --finetune 350 --activity 250 --long 250 --batch 8 --unroll 12 --burn 24 \
   --burn-long-min 48 --burn-long-max 360 --spread 2 --edge 10 --seed 34200000 --threads 3        # snapshots ckpt/r2_stage1..4.ckpt
for k in 2 3 4; do $B finish --study r2 --ckpt r2_stage$k --out C$k $RUNS; done
# §3: validation and the choice (jobs/val.sh), then the freeze
$B choose --study r2 --split val --cands C2,C3,C4 --edge $R/edge_val.csv --bounds $R/edge_bounds.csv --margins $R/margins.csv
cp $D/models/<chosen>.nvfx3 $D/models/chosen.nvfx3; $B freeze --model $D/models/chosen.nvfx3 --frozen $R/frozen.csv
# §4: the test, once (jobs/test.sh), and the verdict
$B verdict --study r2 --split test --margins $R/margins.csv --bounds $R/edge_bounds.csv --edge-val $R/edge_val.csv --val-label <chosen> --edge-test $R/edge_test.csv
```
