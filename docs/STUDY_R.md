# Study R: a validated 3D smoke

Status: **in progress**. Audience: dev, research, the owner. A second 3D effect after study N's fire
([STUDY_3D.md](STUDY_3D.md)), done properly this time: the teacher (our 3D solver's smoke) is checked against
established plume physics first (§1), and the learned smoke is then trained with study N2's recipe and tested once
under a rule committed before validation (§2 onwards).

## 1. The teacher against plume physics

### 1.1 The theory and the measurements it is checked against

**Morton, Taylor and Turner (1956)**, "Turbulent gravitational convection from maintained and instantaneous sources",
*Proc. R. Soc. Lond. A* 234, 1-23 (MTT). A plume from a point source of buoyancy flux F (volume flux times reduced
gravity g' = g dT / T, conserved with height in a uniform, still environment) entrains ambient fluid at a rate
proportional to its own vertical velocity (the entrainment coefficient alpha). Its similarity solution, with z measured
from a virtual origin z0:
- the radius grows linearly, **b = (6/5) alpha (z - z0)**;
- the centreline vertical velocity falls as **w ~ F^(1/3) (z - z0)^(-1/3)**;
- the centreline buoyancy falls as **g' ~ F^(2/3) (z - z0)^(-5/3)**, and so does the concentration of a passive tracer
  released with the buoyancy (our soot);
- the buoyancy flux F itself does not change with height.

With top-hat profiles (uniform across radius b_T) the constants are w_T = (5 / (6 alpha_T)) (9 alpha_T F / (10 pi))^(1/3)
z^(-1/3). With Gaussian profiles of equal width (w = W exp(-r^2 / b^2), the 1/e radius b, as MTT and most measurements
use), b = (6/5) alpha_G z, W = (25 F / (12 pi alpha_G^2))^(1/3) z^(-1/3) and G = 2 F / (pi b^2 W); the two descriptions
are the same plume when b_T = sqrt(2) b_G and alpha_T = sqrt(2) alpha_G.

**The entrainment coefficient, measured.** **List (1982)**, "Turbulent jets and plumes", *Annu. Rev. Fluid Mech.* 14,
189-212 (doi:10.1146/annurev.fl.14.010182.001201), reviews the laboratory plumes on which the constant-alpha model rests;
**Carazzo, Kaminski and Tait (2006)**, "The route to self-similarity in turbulent jets and plumes", *J. Fluid Mech.*
547, 137-148, compiles the published data. In van Reeuwijk and Craske's (2015, *J. Fluid Mech.* 782, 333-355) summary of
that compilation, "typical values for the 'top-hat' entrainment coefficient are in the range 0.065 < alpha < 0.080 in
jets and 0.10 < alpha < 0.16 in pure plumes (Fischer et al. 1979; Carazzo, Kaminski & Tait 2006)". This study's
brief gave alpha of about 0.08 to 0.12 for a top-hat plume; we report alpha in both conventions and compare it with
the compiled band 0.10 to 0.16 (top-hat), 0.071 to 0.113 (Gaussian). Carazzo et al. also show that plumes approach
self-similarity only some distance from the source, which matters here: our box is short (§1.4).

**The wind-bent plume.** Far downwind a buoyant plume in a uniform crossflow U is bent over and rises as a line
thermal: **Briggs's two-thirds law**, z = 1.6 F_B^(1/3) x^(2/3) / U (Briggs 1969, *Plume Rise*, USAEC TID-25075;
Briggs 1975, "Plume rise predictions", in *Lectures on Air Pollution and Environmental Impact Analyses*, AMS, 59-111),
with Briggs's flux F_B = F / pi and x the distance downwind.

## 2. The learned smoke: design and rule (written and committed before validation)

Everything in this section was fixed, and committed, before any validation number existed; the edge check's bounds
(§2.7) were computed from the teacher's runs alone and committed with it (`results/3d/r/edge_bounds.csv`). The results
follow in §3 onwards.

### 2.1 The question

Does study N2's recipe give a 3D smoke that, on held-out settings and fresh seeds, follows its (now validated) teacher
better than simply running the same simulation on the coarse grid with the same seeded detail, plays as realistically
when played on endlessly, and never runs away at the box's open sides over 30 s at any wind?

### 2.2 The teacher's runs (data)

`smoke_r` (§1) on 64 x 96 x 64, recorded as study N's runs are (block averages on the 16 x 24 x 16 coarse grid, every
frame, fp16):
- **24 runs of 8 s** (240 frames) at random controls held, seeds 29,100,000 + i (`runs/short`);
- **24 runs of 18 s** (540 frames) of study N2's three kinds by i mod 3, seeds 29,200,000 + i (`runs/long`): the wind at
  an extreme (0 to 0.15 or 0.85 to 1), controls that change during the run (segments of 2 to 4 s with 1 s ramps), and
  random controls held.

9.6 minutes of simulated smoke in all (N2 had 12.5 minutes of fire; fewer runs here because the shared disk is short).

### 2.3 The model

Study N2's model unchanged in size and form: a 16 x 24 x 16 learned state (velocity, heat, soot and 4 memory channels),
two 3 x 3 x 3 convolutions of 16 channels with FiLM on the three controls, 14,218 weights, a projection and advection
with **ambient air at the open sides and top** (still, with no heat, soot or memory), driving seeded 64 x 96 x 64
detail with N2's constants (contrast 1, swirl 0.5, swirl length 12, grow 1.5). Initialised with seed 29,100,777. 8
start points at frame 150 by farthest-point sampling over the controls (at that frame) of all 48 runs. Effect name
`smoke3d`, drawn with warm smoke's optics (`render3d::optics_for`).

### 2.4 Training

Study N2's four stages and losses, unchanged: stage 1, 900 iterations of windows from true states (the unroll growing
to 12); stage 2, 350 iterations with half the windows after up to 24 frames of the stepper's own rollout and the
profile loss; stage 3, 250 iterations adding the activity loss; **stage 4, 250 iterations of long rollouts**, every
window after 48 to 240 frames of the stepper's own rollout. Batch 8, Adam, learning rates 2e-3 and 7e-4 (cosine),
trainer seed 29,200,000, a checkpoint every 20 iterations and a snapshot at the end of every stage. **Candidates**: C2,
C3 and C4, the stepper after stages 2, 3 and 4, with the same start points and detail constants.

### 2.5 Seeds and held-out settings

Seeds (this study owns 29,000,000 to 29,999,999): the plume measurement 29,000,000 to 29,010,899 (§1); training
29,100,000 + i and 29,200,000 + i; the edge bounds' teacher runs 29,300,000 + 100 j + k (j the wind, k the run);
validation tracking 29,400,000 + 100 s + k, real runs 29,410,000 + 100 s + k, plays 29,420,000 + 100 s + k, the edge
check's plays 29,430,000 + j; test tracking 29,700,000 + 100 s + k, real runs 29,710,000 + 100 s + k, the floor
29,710,050 + 100 s + k, plays 29,720,000 + 100 s + k, the edge check's plays 29,730,000 + j; demo videos 29,900,000 and
up; unit tests 29,099,000 to 29,099,999.

Held-out settings (intensity, wind, turbulence), none a training run's (training controls are random or scheduled),
with a strong wind towards -x and towards +x in each split:
- validation: (0.55, 0.10, 0.45), (0.35, 0.85, 0.60), (0.70, 0.35, 0.75), (0.45, 0.55, 0.30), (0.80, 0.70, 0.55);
- test: (0.60, 0.06, 0.65), (0.40, 0.94, 0.40), (0.75, 0.25, 0.30), (0.30, 0.45, 0.80), (0.50, 0.70, 0.15),
  (0.85, 0.60, 0.50).

Validation: 3 tracking runs per setting (15), 2 real runs and 2 plays of each method pooled per setting. Test: 3
tracking runs per setting (18), 3 real runs, 3 plays of each method and 3 other real runs (the floor) per setting.

### 2.6 Methods and measures

**Methods**: the candidates (the chosen one on the test: "learned"); **the coarse simulation with the same detail**
(`smoke_r` itself run on 16 x 24 x 16, its lengths and forces scaled as the simulator scales them, driving the same
seeded detail layer; study D0's baseline, as N and N2); for reference only, **holding the first frame** (tracking) and
**other real runs** (the floor of the endless statistics, test only). No earlier smoke model is compared: study N's and
N2's smoke demos were trained on the old preset, a different teacher.

**Tracking from the true state** (as N): a run at a held-out setting simulated at 64 x 96 x 64 from an empty grid for
100 frames; every method starts from its state and plays 60 frames with the run's own noise seed; at 8 frames, 1 s and
2 s against the simulation's own continuation: the field PSNR of heat (peak 0.5, the source's largest is 0.45) and of
soot (peak 1.2, the source's largest), over the active cells (either field above 1% of its peak), and the picture PSNR
from four cameras (as N: azimuths 0, 90, 180, 270 degrees, 128 x 128, active pixels), the mean over the four.

**Endless statistics** (as N): per setting R real runs (150 frames of warm-up, then 240) and R plays of each method
(from the start point nearest the setting, its 24-frame warm-up, then 240 frames in one rollout), every frame drawn by
the four cameras; pooled per setting, then against the real runs': the **3D spectrum distance** (heat and soot), **|ln
motion ratio|**, the **coverage distance**, the **light distance** (the mean |difference| of the per-frame mean light
the smoke adds, rgb; warm smoke emits nothing, so this replaces N's emission distance) and the **mean picture PSNR**.

Differences: method minus the coarse simulation (and minus holding, reported beside), paired bootstrap of the mean
difference, 10,000 resamples, seed 2026, 95% percentile intervals (`metrics::paired_bootstrap`); tracking paired over
runs, endless statistics over settings. An interval that covers zero is a tie.

### 2.7 The edge check, its bounds calibrated on the teacher's spread

`nvfx_3d edge`: the model played in one rollout from the start point nearest (0.7, w, 0.5), its 24-frame warm-up and
then **30 s**, at the constant winds **w = 0.05, 0.25, 0.5, 0.75 and 0.95**, the coarse state only; every 0.5 s the
coarse heat summed over each side slab (x = 0, x = max, z = 0, z = max) and over the top layer. Per wind and side the
mean over 5 to 15 s (early), the mean over 20 to 30 s (late) and the largest over 5 to 30 s.

**The teacher's spread** (study N2's lesson: its bound, set from one simulated run per wind, was 3% too tight): **four
teacher runs per wind** (`smoke_r` at 64 x 96 x 64 from an empty grid, 5 s, then 30 s, seeds 29,300,000 + 100 j + k).
Per wind and side:
- **holds too much**: the model's largest exceeds **A x the largest of the four teacher runs + B**;
- **grows**: the model's late mean exceeds **2 x its early mean + C**;
- with N2's constants **A = 2, B = 1, C = 0.5**, then **checked on the teacher itself, leaving one out**: each teacher
  run is checked against the bound made from the other three, and its own growth against C. Where any teacher run
  fails, B (or C) is raised to 1.5 times the smallest value that clears every teacher run, at that wind and side
  (`nvfx_3d edgecal`). The resulting bounds are written to `results/3d/r/edge_bounds.csv` and committed before
  validation.

**A model passes the edge check if no side runs away (holds too much or grows) at any wind.** This is also the 30 s
no-runaway check at every wind.

The only fixed margins in this rule are these bounds; every other decision below is made by 95% intervals or by ranks,
so there is nothing else to calibrate.

### 2.8 The choice (validation)

1. **Eligible**: the candidates that pass the edge check with validation's seeds.
2. Of those, **the lowest mean rank over seven measures**: the tracking picture PSNR and heat field PSNR at 1 s (higher
   better) and the five endless statistics (each the mean over the validation settings; lower better but for the mean
   picture PSNR); tied values share a rank; a tie in mean rank goes to the higher picture PSNR at 1 s.
3. If no candidate passes the edge check, step 2 chooses among all three, and the chosen one fails the test's first
   condition.

`nvfx_3d choose --study r` applies it. The chosen file is frozen by SHA-256 in `results/3d/r/frozen.csv` before the
test; the tool refuses the test's learned runs on any other file.

### 2.9 The test and the verdict (it runs once)

The chosen model, on the test's settings and seeds, against the coarse simulation with the same detail:
1. **No runaway**: it passed the edge check on validation (step 1 of the choice) and passes it with the test's seeds
   (29,730,000 + j).
2. **Tracking**: it *follows a real run better* if, at 1 s, the heat field PSNR's and the picture PSNR's differences
   both have intervals entirely above zero; *worse* if both entirely below; otherwise a *tie* (or mixed), reported as
   such.
3. **Look**: it *plays as realistically* if no endless statistic is worse: no interval entirely above zero for the 3D
   spectrum, |ln motion ratio|, coverage or light distance, and none entirely below zero for the mean picture PSNR.

**The verdict**: "**passes: the learned 3D smoke beats running the cheap simulation**" if 1, 2 (better) and 3 hold;
otherwise "**fails**", naming each condition that does not hold, with every number ("follows real runs better but
looks worse in ...", "ties at tracking", "runs away at ..."). Soot, 8 frames and 2 s, holding the first frame and the
floor are reported beside it, not judged. If it passes it ships as the 3D smoke (the data root's `r_smoke3d/models`,
the orbit video re-rendered); if it fails, the best candidate is shipped only as a labelled demo.

**Cost** (reported, not judged; provisional on the shared machine), as N2.
