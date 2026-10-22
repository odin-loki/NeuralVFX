# Study R: a validated 3D smoke

Status: **finished** (21 October 2026): the teacher is a plume by Morton, Taylor and Turner's measures (§1) but a third
too slow for its buoyancy; **the learned smoke fails its test** (it tracks real runs better than the coarse
simulation, +1.59 dB in pictures at 1 s, but missed an edge bound by 7% on validation and looks worse played on) and
ships as a labelled demo (§4, §5). Audience: dev, research, the owner. A second 3D effect after study N's fire
([STUDY_3D.md](STUDY_3D.md)), done properly this time: the teacher (our 3D solver's smoke) is checked against
established plume physics first (§1), and the learned smoke is then trained with study N2's recipe and tested once
under a rule committed before validation (§2 onwards).

## 0. Where the study stands (a handoff, kept current)

As of 21 October 2026, 14:45 UTC: **finished**; nothing is running.
- The teacher fixed and measured against plume theory (§1, with the buoyancy budget that closed §1.4's open check); the
  rule committed before validation (§2, f269659) and its edge bounds (3c91ff2); training (§3.1); validation and the
  choice of C4 by the rule (§3, aae5dd6), frozen before the test; **the test, run once** (§4): **fails** (the edge
  check on validation; worse in coverage, light and the mean picture), tracking better; C4 ships as a labelled demo
  (`models/3d/smoke.nvfx3`, §5) with four demo videos (videos/README.md §9c); REPORT §20, TODO; origin/main merged
  (6c87b29) and the full checks run on the merge (REPORT §20's last paragraph).
- Data root `$NEURALVFX_DATA/r_smoke3d`: the runs, evaluations, candidates and logs stay there; the job scripts used
  are in its `jobs/`. Nothing in this study is left to run; the next steps are §6.3's.

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

### 1.2 Study N's preset, and what was wrong with it

Study N's smoke (`Effect::smoke`, kept unchanged so its runs can be repeated) has the fire's solver with smoke's
constants: warm gas (0.30 + 0.15 intensity at the source) with strong buoyancy and no combustion. First measurements
(two seeds, 12 s each, `plume/x_n_i50` in the data root) showed it **clearly wrong as a plume**: its buoyancy flux
through a horizontal plane fell by a factor of three from the source to the top of the box (39,000 to 12,400 cells^4 /
s^3), because the preset cools its gas (heat times exp(-0.3 t)), fades its soot (the same rate) and damps every velocity
(exp(-0.1 t)). None of these belongs in a plume of warm smoke a few tens of kelvin above the air: over the seconds it
takes to cross a metre-sized box, such smoke neither cools nor fades, and nothing drags on a free plume.

**Removing the three sinks was not enough.** The same preset without them (two seeds) had a buoyancy flux that *grew*
by 75% from the source to the top: the solver made heat and soot. Two checks found where. (1) The projection is
converged: 150 Gauss-Seidel sweeps instead of 30 changed the fluxes in the third digit. (2) The residual divergence of the
collocated projection (sum of soot x divergence over each plane) accounts for about a tenth of the gain. The rest is the
scalars' advection: semi-Lagrangian with the clamped MacCormack correction, which is not conservative, gained about 5%
of the soot per second inside the plume.

**The fix, `Effect::smoke_r`** (`src/sim3d/sim3d.cpp`): study N's source, controls and forces with **no cooling, no
soot decay and no velocity damping**, and **heat and soot carried in flux form** (`Params::scalars`, default per effect):
a second-order upwind scheme with the monotonised central limiter, one sweep per axis in alternating order, the face
velocity the mean of its two cells', substeps wherever a sweep's Courant number would pass 0.95. What leaves a cell
enters its neighbour, so the solver neither makes nor loses heat or soot (tested: a blob carried by a swirling flow keeps
its total to 2e-6; `Sim3d.FluxFormKeepsTheSootItCarries`). The fire and study N's smoke keep MacCormack and are
unchanged to the bit (their recorded runs compared byte for byte). With the fix one seed's buoyancy flux stayed within
about 7% of its mean at every height, with no trend. Everything below is about `smoke_r`.

**A second question, written down before its answer** (21 October, 07:50 UTC, before the test below finished and
before any training): the first five seeds of `smoke_r` gained only about 0.80 of the vertical momentum their buoyancy
should give (dM/dz against the plane's integral of g', heights 24 to 72) and their flux-balance parameter stayed at 2 to
3 (a pure plume has 1), which points at numerical dissipation in the semi-Lagrangian velocity step. `Params::velocity`
adds the clamped MacCormack correction to that step. **The criterion**: on the same two seeds (29,010,000 and 29,010,001),
if MacCormack velocity brings the momentum ratio within 0.90 to 1.10 *and* moves Gamma at both ends of the fitted range
closer to 1, `smoke_r` adopts it and every `smoke_r` run (training, the edge bounds' teacher runs, the plume
measurements) is made again; otherwise `smoke_r` keeps the semi-Lagrangian step and the deficit is reported as a limit.
**The answer: `smoke_r` keeps the semi-Lagrangian velocity step.** MacCormack velocity did raise the momentum early
on (a 32 x 48 x 32 box, 2 s averaged after 2 s: the momentum flux 30 to 60% higher at the plume's heights, the buoyancy flux within 20%),
but at 64 x 96 x 64 it is unstable: on seed 29,010,000 the momentum flux after 9 s was 10 to 20 times what it was after
4 to 6 s near the floor (460,000 against 30,000 to 38,000 cells^4 / s^2 at 4 cells up), and the full 22 s run never
finished (its velocities grew until the flux form's sweeps needed ever more substeps; they are now capped at 32). An
unstable step is no fix, so the criterion was not met; the momentum deficit stays and is reported as a limit (§1.4).
Decided at 08:20 UTC, before training began and before any training result.

### 1.3 The measurement

`nvfx_3d plume` (`tools/plume3d.cpp`): `smoke_r` in still air (wind 0.5) from an empty box, 6 s of spin-up, then 16 s
averaged every frame (480 frames), one run per seed. Per run: time means in radial bins one cell wide about the
burner's axis at every height (vertical velocity w, buoyancy g' = 210 heat - 0.3 soot in cells per second squared at 64
cells across, soot, and the products w g', w^2, w soot, so the turbulent fluxes are included); the buoyancy flux
through every horizontal plane, F(z) = the plane's sum of the time mean of w g'; a slice through the axis; the soot
summed over depth (for the bent plume). Per configuration the runs' time means are pooled, and at each height a
Gaussian is fitted to the core of each radial profile (ln f against r^2 by least squares, out to where f first falls
below e^-2 of the axis's): the centreline W, G and soot concentration and their 1/e radii b_w, b_g, b_c.

**The fits** (`nvfx_3d plumefit`), over a fixed range of heights:
- the velocity width, b_w = beta (z - z0) by least squares: the spreading rate beta, **alpha_G = beta / 1.2**, alpha_T
  = sqrt(2) alpha_G and **the virtual origin z0** (where the linear width extrapolates to zero);
- with that z0, the exponents of W, G, the soot and F against (z - z0) by least squares in logs (MTT: -1/3, -5/3, -5/3,
  0);
- the origin checked two other ways: W^-3 and G^-3/5 are linear in z for an MTT plume, and their zeros give z0 again;
  and an exponent fitted together with its own origin (a grid search), to show how far the two trade off over a short
  range;
- the constants W (z - z0)^(1/3) / F^(1/3) and G (z - z0)^(5/3) / F^(2/3) against MTT's for Gaussian profiles at the
  fitted alpha_G, and lambda = b_g / b_w (MTT's simple form assumes 1; laboratory plumes have about 1.1 to 1.2);
- **the source strengths**: the slope of ln (W (z - z0)^(1/3)) against ln F across intensities 0.2, 0.5 and 0.8
  (MTT: 1/3).

**Intervals**: 95% percentile intervals from 10,000 resamples of the runs (seeds) with replacement, each resample
pooled and fitted again (seed 2026). With 4 or 5 runs per configuration these understate the uncertainty somewhat.

**The heights fitted** (fixed before the final runs, from the two exploratory seeds): from **24 cells** above the floor
(below about 20 cells the plume is still contracting and accelerating off its broad, lazy source: b_w falls from the
source's width to 3.6 cells by 8 cells up and regains it only by about 20) up to **the highest height to which the
study's box agrees with larger ones**: the largest H such that at every height up to H the 95% intervals of the ratios
of W and of b_w (the study's box over the larger box, pooled, the runs resampled) reach into [0.95, 1.05], for both
larger boxes: one half as wide again (96 x 96 x 96 at the same cells per unit, span 192: 16 cells more on each side)
and one two thirds taller (64 x 160 x 64), each at intensity 0.5 with 4 seeds; H rounded down to a multiple of 4
cells. The local exponents at every height are shown as well (the figure).

**Configurations** (seeds 29,010,000 + 100 c + k): intensity 0.5, 0.2 and 0.8 at turbulence 0.5 (5 seeds each);
turbulence 0 and 1 at intensity 0.5 (4 seeds each; the control's range); the tall and the wide box (4 seeds each); and
the wind-bent plume (§1.6). 1 world unit is taken as a metre for the optics only (warm smoke at 293 K + 98 K per unit
of heat, the buoyancy read as g dT / T); MTT's laws need no units.

### 1.4 Results: the plume in still air

All fits over **24 to 92 cells** above the floor: by §1.3's rule, both larger boxes agree with the study's box within
5% (the intervals of the ratios of W and b_w) up to the top of the box (95.5 cells, rounded down to 92; §1.5). Full
tables: `results/3d/r/plume_fit.md` (and `plume_profiles.csv`, every pooled profile); 10,000 resamples of the runs.

| | intensity 0.2 | intensity 0.5 | intensity 0.8 | MTT, or the measured band |
|---|---|---|---|---|
| runs (seeds) | 5 | 5 | 5 | |
| buoyancy flux F (cells^4/s^3) | 28,700 [26,500, 30,900] | 48,200 [42,500, 52,500] | 76,200 [74,200, 78,100] | |
| **F's exponent with height** | **+0.03 [-0.03, +0.10]** | **+0.02 [-0.03, +0.08]** | **-0.00 [-0.06, +0.05]** | **0 (conserved)** |
| spreading rate beta = db_w/dz | 0.128 [0.118, 0.138] | 0.126 [0.113, 0.147] | 0.120 [0.109, 0.133] | (6/5) alpha |
| **alpha, top-hat** | **0.151 [0.139, 0.162]** | **0.149 [0.133, 0.173]** | **0.141 [0.129, 0.157]** | **0.10 to 0.16** (Carazzo et al.; the brief: 0.08 to 0.12) |
| alpha, Gaussian | 0.107 [0.099, 0.115] | 0.105 [0.094, 0.122] | 0.100 [0.091, 0.111] | 0.071 to 0.113 |
| virtual origin z0 (cells; the burner's top is at 4.7) | -7.3 [-10.0, -5.0] | -10.6 [-16.5, -3.7] | -17.6 [-21.5, -13.8] | |
| **centreline velocity's exponent** | **-0.36 [-0.40, -0.32]** | **-0.45 [-0.51, -0.37]** | **-0.41 [-0.43, -0.39]** | **-1/3** |
| **centreline buoyancy's exponent** (the soot's is the same) | **-1.58 [-1.71, -1.45]** | **-1.51 [-1.62, -1.42]** | **-1.67 [-1.74, -1.60]** | **-5/3** |
| z0 from W^-3 / from G^-3/5 (cells) | -4.7 / -8.4 | +4.6 / -15.6 | -3.6 / -16.1 | both = z0 |
| lambda = b_g / b_w | 1.02 | 0.98 | 1.01 | 1 in MTT's form; 1.1 to 1.2 measured |
| W (z - z0)^(1/3) / F^(1/3), against MTT's at the fitted alpha | 2.56 (MTT 3.88) | 2.67 (3.91) | 2.75 (4.05) | |
| G (z - z0)^(5/3) / F^(2/3), against MTT's | 8.4 (10.0) | 8.8 (10.2) | 9.2 (10.9) | |
| flux-balance Gamma, lowest / highest heights | 3.0 / 3.7 | 2.5 / 3.7 | 2.8 / 3.5 | 1 for a pure plume |
| momentum budget dM/dz over the plane's buoyancy | 0.78 [0.74, 0.83] | 0.78 [0.74, 0.81] | 0.79 [0.74, 0.84] | 1 |

**Across the source strengths**, the slope of ln (W (z - z0)^(1/3)) against ln F is **0.41 [0.33, 0.48]**: MTT's 1/3
lies just inside the interval.

What it says, plainly:
- **The buoyancy flux is conserved** up the plume at every source strength (the exponent within 0.03 of 0, every
  interval covering 0). This is the property study N's preset broke by a factor of three, and what makes the rest of
  the comparison meaningful.
- **The plume spreads linearly at the measured rate for pure plumes**: top-hat alpha 0.14 to 0.15, inside Carazzo et
  al.'s band of 0.10 to 0.16 and above the 0.08 to 0.12 the brief quoted. The spreading slows a little with source
  strength (0.151 to 0.141), within the intervals.
- **The power laws hold roughly, not exactly.** The centreline velocity falls as (z - z0)^-0.36 to -0.45 (MTT: -1/3; the
  interval excludes it at intensities 0.5 and 0.8), the centreline buoyancy and the soot as -1.51 to -1.67 (MTT: -5/3;
  the interval excludes it at 0.5 only). The three ways of finding the virtual origin disagree by 10 to 20 cells (from
  the width about -7 to -18; from W^-3 about 0; from G^-3/5 about -8 to -16), which a self-similar plume would not do;
  and with the origin fitted freely the velocity's exponent and origin trade off over a range this short (at intensity
  0.5, -0.38 with an interval from -2.8 to -0.2, `plume_fit.md`).
- **The velocity is too slow for its buoyancy flux, by about a third.** At the fitted alpha MTT's centreline velocity
  constant is 3.9 to 4.1; ours is 2.6 to 2.75. The flux-balance parameter stays at 2.5 to 3.7 instead of approaching 1 (a
  "lazy" plume, short of momentum), and **the plume gains only 0.78 to 0.79 of the vertical momentum its buoyancy should
  give it**. The likeliest cause is the semi-Lagrangian velocity step's numerical dissipation. MacCormack velocity, which
  would have reduced it, was unstable (§1.2). This is the solver's main departure from MTT, and it is not fixed.
- **The box is short for a plume.** The fits span (z - z0) from about 30 to 110 cells, 2 to 6 burner diameters, where
  laboratory plumes are still on their way to self-similarity (Carazzo et al.). Some of the departure is that; the
  momentum budget is not: it is wrong at every height.

**The turbulence control spans the measured band.** At intensity 0.5 (4 seeds each, the same heights):

| turbulence | top-hat alpha | velocity's exponent | buoyancy's exponent | F's exponent | momentum over buoyancy | W constant (MTT's) |
|---:|---|---|---|---|---|---|
| 0 | 0.110 [0.099, 0.120] | -0.53 [-0.64, -0.43] | -1.62 [-1.69, -1.56] | -0.02 [-0.06, +0.02] | 0.85 [0.81, 0.89] | 3.48 (4.77) |
| 0.5 | 0.149 [0.133, 0.173] | -0.45 [-0.51, -0.37] | -1.51 [-1.62, -1.43] | +0.02 [-0.03, +0.08] | 0.78 [0.74, 0.81] | 2.67 (3.91) |
| 1 | 0.164 [0.153, 0.175] | -0.37 [-0.42, -0.30] | -1.74 [-1.83, -1.67] | -0.03 [-0.05, -0.02] | 0.72 [0.69, 0.74] | 2.32 (3.66) |

The control's confinement and curl noise set the entrainment: alpha runs from 0.110 (inside the brief's 0.08 to 0.12
and the compiled band's lower end) to 0.164 (just above the compiled band's 0.16). The stronger the turbulence, the more
momentum is lost (0.85 to 0.72) and the slower the plume for its flux; at turbulence 1 the buoyancy flux also falls a
little with height (-0.03, the interval clear of 0; the budget below finds why). Turbulence 0 is the most MTT-like in
its constants and buoyancy law, though its velocity falls fastest.

**The buoyancy budget: where the flux goes** (§1.4's open check, measured after the fits above). The same seeds run
again with two more sums (`v_t100b`, the four of turbulence 1; `v_i50b`, four of turbulence 0.5's five; the planes'
fluxes repeat the first runs' to the byte): the buoyancy carried out through the box's four sides in each row, and each
plane's buoyancy at the first and the last averaged frame (what the box stores over the 16 s). Between the planes at 24
and 92 cells, as fractions of the flux at 24 (`results/3d/r/plume_budget.md`, 10,000 resamples of the runs):

| turbulence | the planes' flux lost, 24 to 92 | out through the sides | stored in the box | the rest (the numerics) |
|---:|---|---|---|---|
| 0.5 | -0.023 [-0.085, +0.020] | 0.005 [0.000, 0.013] | -0.026 [-0.086, +0.011] | -0.003 [-0.004, -0.000] |
| 1 | 0.034 [0.014, 0.053] | 0.003 [0.001, 0.004] | 0.034 [0.015, 0.053] | -0.003 [-0.003, -0.003] |

**Heat does not leave through the sides** (at most half a percent of the flux) and **the solver neither makes nor
loses buoyancy above the source** to 0.3%. At turbulence 1 the flux's fall with height is **the box still filling**:
the warm smoke it spreads sideways is still accumulating between 24 and 92 cells after 6 s of spin-up, in every run (the
interval clear of 0), and that is the 3.4% the planes lose; at turbulence 0.5 the stored amount ties with zero, and so
does the flux's change. A longer spin-up would remove it; the plume's laws above are unaffected beyond that 3%.

### 1.5 Where the box matters

The study's box (64 x 96 x 64) against one two thirds taller (64 x 160 x 64) and one half as wide again (96 x 96 x 96,
the same cells per unit), intensity 0.5, 4 seeds each (`results/3d/r/plume_box_*.csv`); ratios of the pooled profiles,
the study's box over the larger:
- **The top does not matter below about 86 cells.** W agrees within 6% and b_w within 4% up to 82 cells (point
  estimates; the intervals are about +-15%, from the seeds' spread of F), then the study's plume slows and narrows by 4 to
  7% in its last 10 cells. In the taller box the exponents move towards MTT's (velocity -0.36 [-0.41, -0.32], buoyancy -1.54), alpha
  0.148.
- **The sides matter a little at every height.** In the wider box the study's plume is 4 to 6% narrower from 14 cells
  up (b_w ratio 0.95 to 0.97, its interval clear of 1 at 26 to 35 cells), its centreline 2 to 8% faster and its
  centreline buoyancy 5 to 18% higher (the intervals reaching 1): a 32-cell half-width constrains the air the plume
  draws in. In the wider box the buoyancy's exponent is -1.74 [-1.82, -1.67], within 0.01 of MTT's -5/3, and alpha
  0.144 [0.137, 0.153].
- By the rule fixed before (§1.3: the intervals of W and b_w within reach of 5%), the box agrees to the top, so the fits
  above span 24 to 92 cells. The point estimates say more precisely: **trust the profiles to about 85 cells, and expect
  the width and the centreline buoyancy to be off by about 5 and 10 to 15% for the sides.**

### 1.6 The wind-bent plume

`smoke_r` at winds 0.75 and 1.0 (towards +x), intensity 0.5, in the wider box with the burner a quarter of the way
across (72 cells of room downwind), 3 seeds of 12 s each; the plume's path is the height of the soot's centroid in
each column (summed over depth), fitted from 8 to 56 cells downwind against **Briggs's two-thirds law**, z = 1.6 (F /
pi)^(1/3) x^(2/3) / U, with F from the still plume (48,200) and U the preset's wind speed:

| wind | U (cells/s) | the plume's mean horizontal speed | exponent (Briggs: 2/3) | coefficient (Briggs: 1.6) |
|---:|---:|---:|---|---|
| 0.75 | 17.5 | 7.8 | 0.54 [0.52, 0.57] | 4.3 [4.0, 4.4] |
| 1.0 | 35.0 | 13.7 | 0.88 [0.84, 0.95] | 5.3 [4.9, 5.6] |

**The two-thirds law does not hold, and for a reason in the solver's model of wind, not in its numerics.** In our solver
the wind is a drag on smoke towards the wind's speed (`add_forces`: u relaxes to U at a rate of 2 per second times
min(1, 2 (heat + soot))); the air itself is still. Briggs's plume is carried by moving air and entrains air that already
moves with U, so it is bent over within a few source diameters. Ours moves at well under half the nominal wind (7.8 and
13.7 against 17.5 and 35 cells per second: diluted smoke feels less drag, and the air it entrains has no wind in it),
so it leans rather than bends: at full wind its path is nearly straight (exponent 0.88), and at 0.75 it reaches the top
of the box by 40 cells downwind, which flattens the fitted path (0.54). Measured against the plume's own speed instead
of U the coefficient comes to 1.9 and 2.1, nearer Briggs's 1.6. Making the wind a moving ambient (an inflow boundary,
wind in the air as well as on the smoke) would change the learned models' open edges (study N2's ambient air is still)
and every 3D effect; it is not done here.

### 1.7 The teacher, in short

`smoke_r` is a plume as Morton, Taylor and Turner describe one in what conserves and how it spreads: its buoyancy flux
is the same at every height, it widens linearly at a top-hat alpha of 0.11 to 0.16 across its turbulence control (the
measured band for pure plumes is 0.10 to 0.16), its centreline velocity and buoyancy fall within about 0.1 of MTT's
exponents, and its velocity scales with the source's flux about as MTT says. It falls short in one clear way: **it is
about a third too slow for its buoyancy flux**, because the solver loses a fifth of the momentum the buoyancy gives (the
likeliest cause is numerical dissipation; the less dissipative step tried was unstable). The box is short (2 to 6 source
diameters of plume), trustworthy to about 85 cells, and its sides narrow the plume by about 5%. **Its wind is not a
crossflow** (a drag on the smoke through still air), so the wind-bent plume does not follow Briggs's law. This is the
teacher the learned smoke below copies; it is a much better smoke than study N's, and still our own solver.

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

**The bounds, as calibrated and committed before validation** (`results/3d/r/edge_calibration.md`, `edge_bounds.csv`;
the teacher's rows `edge_teacher.csv`; 20 teacher runs, 21 October 07:20 to 09:32 UTC). The teacher passed its own
check at every wind and side but one: at a wind of 0.95 one of the four runs sent heat to the z = max side (2.07 at its
peak, growing from 0.002 to 0.52) where the other three held at most 0.09. Left out, that run would have failed a bound
made from the other three (2 x 0.09 + 1 = 1.19) and its growth would have tripped C = 0.5, so the rule raised B there to
2.82 and C to 0.78: that side's bound is 2 x 2.07 + 2.82 = 6.95. This is study N2's lesson in action: one teacher run per
wind would have set a bound the teacher itself breaks. Every other bound is A = 2 times the four runs' largest plus 1:
from 1.00 (a side the smoke never reaches) to 9.35 (the downwind side at a wind of 0.05).

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

## 3. Training and validation

### 3.1 Training and the candidates

As planned (§2.4), in one run without a restart: 1,750 iterations in 5.5 hours of wall time and **4.3 CPU-hours** on
three threads of a machine shared with study Q (load 4 to 18): stage 1 1.06 CPU-hours, stage 2 0.62, stage 3 0.48,
**stage 4 (the long rollouts) 2.14**, half the total (as in N2). Every stage was snapshot (`ckpt/r_stage1..4.ckpt` in
the data root); the candidates `models/C2.nvfx3`, `C3.nvfx3`, `C4.nvfx3` are 274,760 bytes each (`.nvfx3` version 2,
ambient edges, effect `smoke3d`, 8 start points).

### 3.2 The edge check (validation's seeds)

`results/3d/r/edge_val.csv`, `edgecheck_val.md`: each candidate played in one rollout for 30 s at five winds; per side,
the largest coarse heat over 5 to 30 s against its bound (§2.7), and the late mean against twice the early mean plus C.

| candidate | the check | where it fails |
|---|---|---|
| C2 | fails | wind 0.5, x = 0: holds 1.52 against a bound of 1.01 (it does not grow) |
| C3 | fails | **wind 0.25: runs away** on three sides (x = 0 15.37 against 7.15, z = max 5.85 against 1.85, the top 7.12 against 4.69, each growing) |
| **C4** | **fails, on one count** | wind 0.5, x = 0: **holds 1.08 against a bound of 1.01** (its mean 0.46 over 5 to 15 s, 0.83 over 20 to 30 s: it rises, but not past 2 x early + C) |

C4 holds every other side at every wind within its bound, most with room: its largest downwind side 4.64 (bound 8.14,
wind 0.95), its upwind sides 0.49 to 0.74 where the bounds are 1.00 to 1.04, its top 1.24 to 1.58 (bounds 1.50 to 5.56). The
one failure is a side the teacher never reaches: with no wind (0.5) the four teacher runs put at most 0.005 of heat on
x = 0, so the bound there is A x 0.005 + B = 1.01, and C4's smoke leans onto that side a little in its 30 s (peak 1.08).
The bound was calibrated on the teacher's spread and committed before validation; it stands as written.

### 3.3 The rest of validation

`compare_val_*.md`: 15 tracking runs at the 5 settings; 2 real runs and 2 plays of each method pooled per setting.
Means:

| | C2 | C3 | **C4** | coarse simulation | holding the first frame |
|---|---:|---:|---:|---:|---:|
| tracking, picture, 1 s (dB) | 17.65 | 17.63 | **17.83** | 16.29 | 10.31 |
| tracking, heat field, 1 s (dB) | 18.22 | 18.13 | **19.30** | 17.54 | 17.23 |
| endless: 3D spectrum distance | 0.511 | 0.535 | 0.512 | 0.710 | |
| endless: \|ln motion ratio\| (motion ratio) | 0.674 (0.53) | 0.565 (0.60) | **0.531** (0.60) | 0.605 (0.56) | |
| endless: coverage distance | 0.0291 | 0.0240 | **0.0167** | 0.0130 | |
| endless: light distance | 0.0247 | **0.0214** | 0.0304 | 0.0121 | |
| endless: mean picture PSNR (dB) | 22.03 | 22.66 | **22.80** | 23.57 | |

C4 against the coarse simulation, differences with 95% intervals (paired bootstrap, 10,000 resamples):

| measure | 8 frames | 1 s | 2 s |
|---|---|---|---|
| tracking, heat field (dB) | +1.46 [+1.10, +1.80] | +1.76 [+1.40, +2.08] | +2.81 [+2.15, +3.51] |
| tracking, picture (dB) | +1.40 [+0.89, +1.91] | +1.54 [+0.96, +2.15] | +1.49 [+1.00, +1.99] |

| endless statistic | C4 - coarse simulation |
|---|---|
| 3D spectrum distance | -0.198 [-0.278, -0.123] (better) |
| \|ln motion ratio\| | -0.074 [-0.311, +0.119] (tie) |
| coverage distance | +0.0037 [+0.0002, +0.0072] (worse) |
| light distance | +0.0183 [+0.0090, +0.0275] (worse) |
| mean picture PSNR (dB) | -0.77 [-3.47, +1.94] (tie) |

- **Every candidate follows a real run better than the coarse simulation**, at every horizon, and C4 most (the heat
  field +1.76 dB at 1 s, C2 and C3 +0.6 to +0.7). Holding the first frame is far behind in pictures (a rising smoke
  column moves its sharp edges by several cells in 8 frames).
- **Played on, every candidate is worse than the coarse simulation in the light the smoke adds** (C4 +0.018, the
  worst of the three), C2 and C4 in coverage, and C2 and C3 in the mean picture (C4 ties): the learned smoke covers and darkens the cameras' pictures differently
  from the real runs, where the coarse simulation, which is the teacher's own equations on a coarse grid, gets the
  plume's bulk right. The 3D spectrum is better for every candidate (the seeded detail on a moving learned state).
  Every model is calmer than the real smoke (motion ratios 0.53 to 0.60; the coarse simulation 0.56).
- **Stage 4 made C4 again** (as in N2): against C3, the same model before the long rollouts, it tracks 1.2 dB better
  in the heat field at 1 s, and its coverage and motion are better; its light is worse.

### 3.4 The choice, by the rule

`choice_val.md`: **no candidate passes the edge check**, so (§2.8, step 3) the lowest mean rank over the seven measures
chose among all three: **C4** (mean rank 1.43; C2 and C3 2.29 each). Its file was frozen before the test
(`results/3d/r/frozen.csv`: `chosen.nvfx3`, 274,760 bytes, SHA-256
7a53e3e23a6cdbb667e2c96044a00ad74b83b255e89108c7b4af70bf3ac19988). **By the rule it already fails the test's first
condition**; the test ran once all the same, and everything it measured is reported (§4). (The choice table's
"temperature" column is the heat field and its "emission" column the light distance: the tool's labels from study N2.)

## 4. The test (run once)

Run once, 14:05 to 14:20 UTC on 21 October, on the frozen C4 (`jobs/test.sh`; `results/3d/r/compare_test_*.md`,
`edgecheck_test.md`, per-run rows in `test_track/`): 18 tracking runs at the 6 held-out settings; 3 real runs, 3 plays
of each method and 3 other real runs (the floor) pooled per setting; the edge check with the test's seeds. Differences
are C4 minus the coarse simulation with the same detail, paired bootstrap of the mean, 10,000 resamples, 95%
intervals.

| measure | C4 | coarse simulation | holding | floor | C4 - coarse simulation |
|---|---:|---:|---:|---:|---|
| tracking, heat field, 8 frames (dB) | 21.31 | 19.80 | 18.75 | | +1.51 [+1.29, +1.74] |
| **tracking, heat field, 1 s (dB)** | **20.02** | 18.17 | 17.78 | | **+1.85 [+1.45, +2.27]** |
| tracking, heat field, 2 s (dB) | 20.62 | 17.74 | 18.33 | | +2.88 [+2.38, +3.39] |
| tracking, soot field, 1 s (dB) | 19.56 | 17.67 | 17.29 | | +1.89 [+1.47, +2.32] |
| tracking, picture, 8 frames (dB) | 22.74 | 21.13 | 14.56 | | +1.62 [+1.16, +2.10] |
| **tracking, picture, 1 s (dB)** | **17.87** | 16.28 | 10.46 | | **+1.59 [+1.02, +2.20]** |
| tracking, picture, 2 s (dB) | 16.39 | 14.78 | 10.75 | | +1.62 [+1.08, +2.19] |
| **endless: 3D spectrum distance** | 0.504 | 0.686 | | 0.038 | -0.182 [-0.230, -0.140] (better) |
| **endless: \|ln motion ratio\|** | 0.552 | 0.602 | | 0.065 | -0.051 [-0.182, +0.084] (tie) |
| endless: motion ratio (context) | 0.582 | 0.549 | | 1.017 | +0.033 [-0.043, +0.107] |
| **endless: coverage distance** | 0.0201 | 0.0152 | | 0.0058 | **+0.0049 [+0.0002, +0.0095] (worse)** |
| **endless: light distance** | 0.0363 | 0.0131 | | 0.0039 | **+0.0232 [+0.0126, +0.0326] (worse)** |
| **endless: mean picture PSNR (dB)** | 21.70 | 24.22 | | 34.07 | **-2.52 [-3.79, -1.29] (worse)** |

**By the rule** (§2.9):
1. **No runaway: fails.** No candidate passed the edge check on validation, so the rule had already declared this
   condition failed (C4: the x = 0 side at a wind of 0.5, 1.08 against a bound of 1.01). With the test's seeds C4
   *passes* the check (`edgecheck_test.md`: every side within its bound at every wind, nothing grows; that same side
   reaches 1.00 against 1.01, its downwind sides at most 4.56 against 9.35, its top at most 1.49). That does not change
   the verdict; it is reported beside it, and it shows how close to C4's own seed-to-seed spread the bound sits.
2. **Tracking: it follows a real run better.** At 1 s both differences are entirely above zero: the heat field
   +1.85 dB [+1.45, +2.27] and the picture +1.59 dB [+1.02, +2.20]; so are the soot and the 8-frame and 2 s horizons.
3. **Look: it does not play as realistically.** Three endless statistics are worse: the coverage distance
   (+0.0049 [+0.0002, +0.0095]), the light distance (+0.0232 [+0.0126, +0.0326]) and the mean picture
   (-2.52 dB [-3.79, -1.29]); the 3D spectrum is better (-0.182 [-0.230, -0.140]) and the motion ties.

**The verdict: fails.** The learned 3D smoke follows real runs better than running the cheap simulation (+1.59 dB in
pictures and +1.85 dB in the heat field at 1 s), but by the rule it runs away (validation's edge check, one side, 7%
over its bound; it passes with the test's seeds), and it looks worse played on in coverage, in the light it adds and
in the mean picture. By §2.9 it is shipped only as a labelled demo (§5).

Beside the rule:
- **The look's shortfall is everywhere, not one setting**: C4's mean picture is below the coarse simulation's at all six
  test settings (by 0.6 to 4.9 dB), its light distance is larger at five (nearly equal at the sixth, 0.0191 against 0.0186),
  its coverage distance larger at four. The learned smoke spreads into a broad, diffuse cone and fills more of the
  box than the real plume ([videos/README.md](../videos/README.md) §9c, real against learned); the coarse simulation
  runs the teacher's own equations, so it gets the plume's bulk, its spread, right, and loses only in detail (its 3D
  spectrum distance is the worse). Validation said the same in coverage and light, the mean picture tied there.
- **Both are calmer than the real smoke** (motion ratio 0.58 for C4, 0.55 for the coarse simulation, 1.02 for other
  real runs): the seeded detail layer they share moves less than the simulation's own detail.
- **Holding the first frame is far behind** in pictures (C4 +7.42 dB [+7.08, +7.74] at 1 s) and in the heat field
  (+2.24 [+1.91, +2.58]): a rising smoke column does not stand still the way a flame's base does. Other real runs (the
  floor) are still far closer to the real runs than any method (spectrum 0.038, mean picture 34.1 dB).

**Cost**, re-timed on a quiet machine (round 7 of the quiet re-timing, 21 October 2026,
[results/quiet/README.md](../results/quiet/README.md): `nvfx_3d timing`, six rounds with N2's fire and study hres's in
turn, nothing else running; the median of the rounds' medians and its 95% interval,
[volume_summary.csv](../results/quiet/round7/volume_summary.csv)); in brackets the first measurement where it differs by
more than 15% (`results/3d/r/timing.csv`, nice 15, a quiet moment of the shared machine, load 1.4 to 1.6). Milliseconds
per frame:

| what | 1 thread | 4 threads |
|---|---:|---:|
| a step through the volume API | **21.4** [20.0, 24.8] (was 26.6) | **9.2** [8.7, 9.9] |
| a picture, 256 x 256 | 85.7 [79.8, 94.6] | 22.7 [21.7, 23.9] |
| a picture, 512 x 512 | 317 [300, 338] | 83.4 [78.7, 87.4] |
| a start (the start point and the 24-frame warm-up) | 554 [465, 615] (was 683) | 226 [211, 230] |

(Two threads were not re-timed; the first measurement gave 15.1, 47, 170 and 387 ms.) The volume API's step uses the
vectorised detail layer (REPORT §19); on one thread the stages with the reference detail layer are: the noise inputs
2.2 ms, the coarse step 9.9, the detail layer 51.3 (which the vector kernel replaces), the renderer's preparation 14.4,
a 256 x 256 picture 69.2. The simulation's own frame is 139 ms at 64 x 96 x 64 and 5.8 ms at 16 x 24 x 16. A smoke
picture costs more than the fire's (N2's: 56 ms at 256 x 256 on one thread in the same rounds), most likely because the
soot fills more of the box, so more of each ray is marched (not measured).


## 5. What ships

- **`models/3d/smoke.nvfx3`**: C4, the frozen file (SHA-256 7a53e3e2...19988), **as a labelled demo**: it failed its
  rule (§4). `models/layout.txt` puts it at `r_smoke3d/models/chosen.nvfx3` in the data root, where
  `neuralfx.Smoke3dShippedModel` (`nvfx_3d check`) finds it: the file is the one frozen for the test, it plays through
  the volume C API, its heat is finite and not negative, it is visible, and over 10 s at three winds its sides stay
  within the calibrated bounds (skipped, exit 77, without the file).
- `models/3d/smoke_demo.nvfx3` **stays N2's smoke2** (study N's smoke preset, never tested): study hres's smoke
  scripts copy it from `n2_3d/models/smoke2.nvfx3` as their baseline, and study R compared no earlier smoke (§2.6: a
  different teacher).
- **Videos** ([videos/README.md](../videos/README.md) §9c), each labelled a demo: the teacher against plume theory
  (`3d_plume_theory.mp4`), the learned smoke walked around (`3d_smoke_r_orbit.mp4`), real against learned
  (`3d_smoke_r_real_vs_learned.mp4`) and the fire and smoke coupled (`3d_fire_smoke_coupled.mp4`). Study N's smoke
  video stays: as in N2, a model that fails its rule does not replace the demo, which is the owner's call.
- **In the solver**: `Effect::smoke_r` (the validated teacher) and the flux-form scalars, the plume measurement and its
  fits; neither the library nor the FUSE demo plays the smoke.

## 6. Beside the study

### 6.1 Fire and smoke coupled (a demo, not measured)

`nvfx_3d video --kind coupled` stacks two learned volumes: N2's 3D fire in the lower box and study R's smoke in the box
above it, drawn as one 64 x 192 x 64 volume with the fire's optics, so the flame lights its smoke. The fire's soot
leaving the top of its box (the soot times the upward velocity over its top four layers, smoothed over a second,
against its running mean) sets the smoke's intensity, so the smoke above follows how much the fire sends up. Making
it found a fault in the demo's own code, fixed before the video: the volume API's textures run z slowest, the
renderer's fields y slowest, and the two boxes were stacked in z and drawn scrambled into stripes (and the flux was
read off the wrong cells). It is a demo of coupling through the public API, not a coupled simulation: the boxes do not
exchange heat or momentum, and the seam between them shows as a flat base under the upper smoke.

### 6.2 Study R and study hres

The owner's study hres ([STUDY_3D_HRES.md](STUDY_3D_HRES.md), on main since this branch began) and study R change
different things and share no data, seeds or files:
- **hres changes the model**: a coarse state of 32 x 48 x 32 (fine 2, a stepper 32 wide, 8 memory channels) driving
  the same 64 x 96 x 64 detail, trained with study N2's recipe; its fire is trained and shipped
  (`models/3d/fire_hres.nvfx3`), not yet validated. **Study R changes the teacher**: N2's model unchanged in size
  (16 x 24 x 16), trained on `smoke_r`, the smoke checked against plume theory first (§1), then validated and tested
  once under a rule committed before validation.
- **hres's smoke** (`tools/study_3d/hres_smoke.sh`, `hres_smoke_validate.sh`; STUDY_3D_HRES.md §6) retrains N2's smoke2
  lineage at the finer grid on `--effect smoke`: **study N's preset, the teacher §1.2 found wrong as a plume** (its
  buoyancy flux falls threefold up the box, because it cools, fades and damps). Its check is the edge check and a demo
  video, with no held-out test. So the two smokes overlap in name only: hres's is a finer model of the old teacher, study
  R's a validated teacher with N2's model.
- Seeds: hres's smoke uses blocks from 9,610,000 to 9,867,999 (inside study N's range), its data root `hres_3d/smoke`; study R uses
  29,000,000 to 29,999,999 and `r_smoke3d`. The merged `nvfx_3d` keeps both (`--study hres`, `--study r`; the merge
  resolved study R's and hres's additions to the same switches side by side and changed nothing of hres's behaviour).
- **The natural next step joins them**: hres's finer model trained on `smoke_r`. `--study hres --effect smoke_r` runs
  today, but hres's seed blocks treat only `--effect smoke` as smoke (`hres_smoke()`), so such a run would take the
  fire's seed blocks; it needs a seed block of its own and its own rule, and it is not done here.

### 6.3 What it means

- **The teacher is now a plume.** Study N's smoke lost two thirds of its buoyancy flux on the way up; `smoke_r`
  conserves it, spreads at the measured entrainment rate for pure plumes and follows MTT's power laws to about 0.1 in
  the exponents, with one clear fault: it is about a third too slow for its flux (it gains 0.78 of the momentum its
  buoyancy gives), and its wind is a drag on the smoke rather than moving air.
- **N2's recipe tracks this teacher well and plays it badly.** As for the fire, the learned smoke follows a real run
  better than the coarse simulation with the same detail, by about 1.6 dB in pictures and 1.9 dB in the heat field at 1
  s. Played on, it spreads too wide and fills the box: smoke is seen by its soot everywhere it goes, where a flame is
  seen by its emission at the source, so a plume's spread is what the pictures measure, and the coarse simulation,
  which runs the teacher's own equations, gets that right.
- **The edge check failed by a hair, at the tightest kind of bound.** At no wind the teacher puts no heat on the x = 0
  side, so the bound there is the fixed margin alone (B = 1); C4 holds about 1 (1.08 on validation's seed, 1.00 on the
  test's). The calibration on four teacher runs did what N2's lesson asked for wherever the teacher reaches a side; for
  a side it never reaches, the check asks the model for no more than one unit there over 30 s. The rule stands as
  written.
- **Next**: train against the spread (the profile and activity losses see means and motion, not the plume's width), more
  teacher time (9.6 minutes here, N2 had 12.5), and hres's finer state on this teacher (§6.2). A moving ambient wind in
  the solver would make the bent plume Briggs's, at the cost of changing every 3D effect's open edges.

## 7. Reproduce

Data under `NEURALVFX_DATA/r_smoke3d` (default `/root/nvfx-data/r_smoke3d`); every step resumes from what it has
written (the job scripts used are in the data root's `jobs/`). Build as the main README says, then:

```sh
B=build/nvfx_3d; R=results/3d/r; D=$NEURALVFX_DATA/r_smoke3d; W="--winds 0.05,0.25,0.5,0.75,0.95 --seconds 30"
# §1: the plume (seeds 29,010,000 + 100 c + k), then the fits
P="$B plume --study r --spin 6 --seconds 16"
$P --label v_i50 --intensity 0.5 --seeds 5 --seed-base 29010000; $P --label v_i20 --intensity 0.2 --seeds 5 --seed-base 29010100
$P --label v_i80 --intensity 0.8 --seeds 5 --seed-base 29010200; $P --label v_t00 --turbulence 0 --seeds 4 --seed-base 29010300
$P --label v_t100 --turbulence 1 --seeds 4 --seed-base 29010400; $P --label v_tall --seeds 4 --seed-base 29010500 --height 160
$P --label v_wide --seeds 4 --seed-base 29010600 --width 96 --span 192
$B plume --study r --spin 6 --seconds 12 --label v_w75 --wind 0.75 --seeds 3 --seed-base 29010700 --width 96 --span 192 --source-x 0.25 --wind-speed 17.5
$B plume --study r --spin 6 --seconds 12 --label v_w100 --wind 1.0 --seeds 3 --seed-base 29010800 --width 96 --span 192 --source-x 0.25 --wind-speed 35
$B plumefit --study r --configs v_i20,v_i50,v_i80,v_t00,v_t100,v_tall,v_wide,v_w75,v_w100 --strengths v_i20,v_i50,v_i80 \
   --box v_i50:v_tall,v_i50:v_wide --bent v_w75,v_w100 --still v_i50 --bent-lo 8 --bent-hi 56 --fit-lo 24 --fit-hi 92 --out $R
$B plumevideo --study r --configs v_i50,v_i20,v_i80 --fit-lo 24 --fit-hi 92 --seconds 16 --seed 29900100 --average-from 4 --out 3d_plume_theory.mp4
# §1.4's buoyancy budget: the same seeds again with the sides' and the store's sums
$P --label v_t100b --turbulence 1 --seeds 4 --seed-base 29010400; $P --label v_i50b --seeds 4 --seed-base 29010000
$B plumefit --study r --configs v_i50b,v_t100b --budget v_i50b,v_t100b --fit-lo 24 --fit-hi 92 --out $R --md plume_budget.md --csv /dev/null
# §2: the training runs, the edge bounds, training and the candidates
$B record --study r --runs-dir short --runs 24; $B record --study r --runs-dir long --runs 24
$B edge --study r --sim --runs 4 $W --out $D/results/edge_teacher.csv; $B edgecal --study r --csv $D/results/edge_teacher.csv --out $R/edge_bounds.csv
RUNS="--runs-dir long --also $D/runs/short"
$B train --study r --name r $RUNS --iterations 900 --finetune 350 --activity 250 --long 250 --batch 8 --unroll 12 --burn 24 \
   --burn-long-min 48 --burn-long-max 240 --seed 29200000 --threads 3                # snapshots ckpt/r_stage1..4.ckpt
for k in 2 3 4; do $B finish --study r --ckpt r_stage$k --out C$k $RUNS; done
# §3: validation (jobs/val_staged.sh, jobs/val_report.sh)
$B truth --study r --split val; $B real --study r --split val
for m in C2 C3 C4; do $B edge --study r --model $D/models/$m.nvfx3 --label $m $W --out $D/results/edge_val.csv
  $B track --study r --split val --method $m --model $D/models/$m.nvfx3; $B play --study r --split val --method $m --model $D/models/$m.nvfx3; done
$B track --study r --split val --method coarse --model $D/models/C2.nvfx3; $B play --study r --split val --method coarse --model $D/models/C2.nvfx3
$B track --study r --split val --method hold --model $D/models/C2.nvfx3
$B edgecheck --study r --csv $R/edge_val.csv --bounds $R/edge_bounds.csv
$B choose --study r --split val --cands C2,C3,C4 --edge $R/edge_val.csv --bounds $R/edge_bounds.csv
cp $D/models/<chosen>.nvfx3 $D/models/chosen.nvfx3; $B freeze --model $D/models/chosen.nvfx3 --frozen $R/frozen.csv
# §4: the test, once (jobs/test.sh, jobs/test_report.sh)
$B truth --study r --split test; $B real --study r --split test
$B edge --study r --model $D/models/chosen.nvfx3 --label learned --seed-base 29730000 $W --out $D/results/edge_test.csv
$B track --study r --split test --method learned --model $D/models/chosen.nvfx3 --frozen $R/frozen.csv   # and play; coarse; hold
$B compare --study r --split test --a learned --b coarse --results $R; $B compare --study r --split test --a learned --b hold --results $R
$B compare --study r --split test --a hold --b coarse --results $R; $B edgecheck --study r --csv $R/edge_test.csv --bounds $R/edge_bounds.csv
# the cost (§4), the shipped file (§5), the videos (videos/README.md §9c)
$B timing --study r --model $D/models/chosen.nvfx3 --threads 1,2,4 --out $R/timing.csv
cp $D/models/chosen.nvfx3 models/3d/smoke.nvfx3; (cd models && find 2d 3d prior upscale -type f | sort | xargs sha256sum > SHA256SUMS)
```
