# Study P: large effects drawn at half size and upscaled

Status: **done** (20 October 2026). Audience: dev, research, the owner. The design and the plan (§1 to §4.4, commit
`0108862`) were committed before any validation or test number; the rule's numbers (§4.5: the margins from the null
probe, the choice, the number of test settings; commit `1dcf55e`, with the mushroom's note in `858f301`) before the
test, which ran once (§6).

**In short.** Drawing a large effect at half size and bringing it back with the small learned 2x and seeded
micro-detail **did not pass its rule**: on 20 held-out settings at 512 px it beat the bilinear 2x against native on
PSNR, SSIM and the high band together for **none of the six effects** (it was a little closer than bilinear on PSNR
for three, by +0.09 to +0.11 dB, further for two; its high band never closer to native's), and it cost at most 0.55 of
native for **four** of them (0.45 to 0.59; five needed; a busy machine). Against the simulator's frames it was within
the null's margins of native for all six. What pays is drawing at half size itself: the half-size instance costs
**0.29 to 0.34 of native at 512 px**, and with the bilinear 2x (free on a GPU) it is within the null's margins of
native against the simulator's frames on every measure for all six effects at 512 px, and closer than native on
tracking and the spectrum. What it loses is native's finest filaments, which native's detail layer draws at 512 px and
which neither a 2x of the half-size frame nor the simulator (at its 128 cells) has. **The opt-in path ships, labelled
experimental**: `nvfx_instance_set_upscale` with `NVFX_UPSCALE_BILINEAR` (no file) or `NVFX_UPSCALE_LEARNED` (an
attached `.nvfxup`), NVFX_VERSION 7; ENGINES.md §5.

The fine fields never feed back into the coarse state, so a half-size instance shares the native one's motion to the
bit (§2): the comparison against native on the same seed is a paired one.

## 1. The question

A rollout effect costs a fixed coarse step (the stepper on its 32 x 32 grid, about 0.3 ms) plus a detail layer and a
learned renderer that run per pixel. An effect drawn large (an explosion filling the screen, a waterfall, a wall of
fire) is drawn at 512 px or more, where the per-pixel parts are most of the frame: about 4.5 ms of one busy core at
512 px against 1.4 ms at 256 px. The idea: draw such an effect at half size, then bring it to full size with

- a small learned 2x upscaler (a few small convolutions on the half-size RGBA frame), and
- seeded micro-detail: fine-scale noise whose amplitude the network gives from the frame, deterministic from the
  instance's seed and the time, as everything here is.

**Does that look as good as drawing at native size, for clearly less time?** Three arms per effect at a target size S
(256 and 512 px), all through the runtime as it ships on this CPU (its default precision, int8 on AVX-512 with VNNI):

| arm | what is drawn |
|---|---|
| **N** (native) | an instance created at S |
| **B** (half + bilinear) | an instance at S / 2, upscaled by the bilinear 2x: what any engine's GPU sampler gives for free |
| **L** (half + learned) | the same S / 2 instance, upscaled by the learned 2x with seeded micro-detail |

## 2. How the size enters, and what N and the half-size instance share

From the code (`src/core/rollout.cpp`, `src/runtime/rt_rollout.hpp`, `src/runtime/nvfx.cpp`):

- **The coarse step does not see the size.** `coarse_step` reads the coarse state, the noise inputs (on the 32 x 32
  grid, from the seed and the time) and the condition; it writes the next coarse state and the flow. Nothing of the fine
  fields enters it.
- **The detail layer and the renderer run at the instance's size.** The fine heat and soot are S x S; they are carried
  by the coarse flow (bilinear to the pixels) plus a sub-grid swirl, both in the coordinates of a 128-pixel frame, and
  locked to the coarse state per block of S / 32 pixels; new material is broken up by the flicker noise, also in
  128-pixel-frame coordinates. The renderer reads fine and coarse fields per pixel.
- **The fine fields never feed back into the coarse state**, and the instance's timeline (which start point and seed
  each shard takes, when it is rolled ahead and crossfaded) depends on the seed, the controls and the time, not on the
  size.

So **N and the half-size instance share the macro motion for the same seed**: `nvfx_study_p check` confirms that the
coarse states of runners at 512 and 256 px are the same to the bit over 120 frames for all six effects (at the default
int8 precision on AVX-512), and that the simulator's dynamics do not depend on the size it draws at. A paired,
frame-by-frame comparison of B and L against N is therefore meaningful.

**But the fine fields are not shared.** The detail layer at S / 2 is a coarser simulation of the same detail (more
numerical diffusion in its MacCormack advection, coarser sampling of the swirl and the flicker), and its differences
grow: the half-size instance's picture against N's averaged to half size is only 25 dB (smoke) to 44 dB (wildfire) in
active PSNR, while N averaged and brought back by the bilinear 2x is 31 to 48 dB from N (`nvfx_study_p anatomy`, one
play per effect at a training setting):

| effect | B against N, 256 / 512 px | the S / 2 instance against N averaged to S / 2 | N averaged to S / 2, bilinear 2x, against N |
|---|---|---|---|
| explosion | 36.7 / 39.9 dB | 38.7 / 41.2 dB | 40.3 / 46.5 dB |
| mushroom | 32.5 / 37.4 | 33.6 / 38.3 | 36.3 / 43.2 |
| wildfire | 38.8 / 43.1 | 40.5 / 44.2 | 41.9 / 48.5 |
| launch | 27.8 / 31.3 | 29.2 / 32.7 | 30.1 / 34.5 |
| smoke | 24.7 / 24.9 | 25.2 / 25.3 | 31.0 / 34.7 |
| waterfall | 30.6 / 33.1 | 32.0 / 34.4 | 33.3 / 36.8 |

What a 2x upscaler of the half-size frame can recover is bounded by this: N's finest filaments are not in the half-size
frame, not even blurred, because the half-size detail layer drew its own. Against N on the same seed, the conditional
mean of N given the half-size frame is close to the bilinear 2x, and any detail added that is not N's own costs PSNR.

## 3. Design

### 3.1 The effects

Six large ones from the 21 models the library shipped when the study began (before study O and round 9), copied to
the data root's `p_upscale/models` and frozen by SHA-256 in [p_models.csv](../results/experiments/upscale/p_models.csv):
**explosion** (study M3's), **mushroom**, **wildfire**, **launch**, **smoke** and **waterfall**.

**One change from the suggested set: launch instead of cloud.** How much of its frame each shipped effect fills, at 8
training settings, 128 px ([p_coverage.txt](../results/experiments/upscale/p_coverage.txt), `nvfx_study_p info`):
wildfire draws 71% of its pixels, launch 48%, waterfall 42%, explosion 25%, smoke 22% and mushroom 20%, but cloud only
12% (mean alpha 0.04), 16th of 21; launch is second (mean alpha 0.31, the highest). The explosion, the mushroom and
smoke are kept as suggested: they grow to fill their frames and are the effects a game draws largest.

### 3.2 The upscaler (`include/neuralfx/upscale.hpp`, `src/core/upscale.cpp`)

On the half-size frame (premultiplied RGBA8, n x n), per half-size pixel:

- a 3 x 3 convolution, 4 -> c1, ReLU; a k2 x k2 convolution, c1 -> c2, ReLU; a 1 x 1 output layer with 20 values: the
  residuals of the pixel's 2 x 2 full-size pixels (16) and the micro-detail's amplitude per channel (4, through a ReLU);
- out = bilinear 2x + residual + gain x amplitude x eta, then the instance's colour matrix, clamped, to bytes.

**eta** is seeded noise at the full size: white noise (one hash per pixel from the instance's seed and a time slice)
minus its 3 x 3 mean, scaled to unit variance, so its energy sits at the finest scales; slices change `rate` times a
second and are blended by a smoothstep, with the variance kept at 1. It depends on the seed, the pixel and the frame's
time only. **Pixels where the bilinear 2x is empty in every channel stay exactly transparent**, as a rollout effect's
empty pixels are. The file (`.nvfxup`, magic NVFXUPS1) holds c1, c2, k2, the gain, the rate, a tag and the weights:
2.2 KB for the shipped size.

**The size, from cost** (training data only). Held-out training tiles, PSNR against N, residual only:

| c1, c2, k2 | multiply-adds per full-size pixel | held-out tiles (bilinear 30.08 dB) |
|---|---:|---:|
| 8, 8, 1 | 128 | 30.15 dB |
| 16, 8, 1 | 216 | 30.19 |
| 16, 16, 1 | 288 | 30.21 |
| 8, 8, 3 | 256 | 30.25 |
| 16, 16, 3 | 800 | 30.31 |

Every size gains little over the bilinear 2x (§2 says why), and the larger ones cost what the plan cannot pay: with
8, 8, 1 the runtime's upscaler takes about 0.6 to 0.8 ms of a busy core at 512 px, which with the half-size instance
(0.30 to 0.35 of N) puts L at about 0.45 to 0.56 of N (a probe, `nvfx_study_p cost`, 8 rounds); the 8, 8, 3 net would
double the network's part. **The study uses 8, 8, 1.** One net serves both sizes: a net trained on 512 px tiles alone
did no better on them than one trained on both (`tile-eval`).

### 3.3 Training (`nvfx_study_p tiles`, `train`; training settings and seeds only)

- **Data**: per effect, the controls of its salt-1 training runs 0 to 15, one play each (seeds 3,600,000 + 100 s), N at
  S and the S / 2 instance with the same seed, at S = 256 and 512; every 15th frame (one-shots: 20 frames over their
  run), 8 tiles of 24 x 24 half-size pixels per frame where the half-size frame has something, with 3 pixels of context
  (zero outside the frame, as the convolutions pad). 31,488 tiles in all.
- **Loss**: squared error of bilinear + residual against N's pixels where the bilinear 2x is not empty and either frame
  is visible; from the middle of training, also the amplitudes against the RMS of the high-passed remaining error over
  each 2 x 2 block (what the residual leaves out, at the scales the noise draws). Adam, cosine schedule from 3e-3, 4,000
  steps of 32 tiles, hand-written gradients (`tile_pass`).
- **The candidates**: one **shared** net, trained on all six effects' tiles (each effect drawn equally often), and **one
  per effect**, trained on its own tiles, the same size and steps.
- **The micro-detail's gain and rate** (`nvfx_study_p calib`, the dev split: training runs 16 on, seeds 3,650,000):
  the gain is the largest of 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75 and 1 whose PSNR and SSIM against N each keep at least
  half of the residual's own gain over B (gain 0), means over the effects and sizes; if the residual gains nothing on
  either measure, there is no micro-detail. The rate is the one of 4, 8, 15 and 30 slices a second whose motion is
  closest to N's. A high-band loss that made the residual itself sharpen (restoring N's high-pass energy from 21% to
  87% of N's) was tried and dropped: it cost 0.6 dB of PSNR against N on held-out tiles, which the rule's first
  condition does not allow.

### 3.4 The runtime (`src/runtime/rt_upscale*.{hpp,cpp}`, `nvfx_effect_attach_upscaler`, `nvfx_instance_set_upscale`)

An opt-in path, mirroring how the prior against drift is attached (NVFX_VERSION 7):

```c
nvfx_effect_attach_upscaler(effect, "explosion.nvfxup");     /* once, before sharing the effect */
nvfx_instance_set_upscale(instance, NVFX_UPSCALE_LEARNED);    /* or NVFX_UPSCALE_BILINEAR, NVFX_UPSCALE_OFF */
```

An instance that upscales holds its two runners at half its size (a set-up call: new buffers and a new timeline), draws
each frame at half size without its colour matrix (shards and crossfades as before), and the upscaler brings it to full
size with the noise of the frame's time from the instance's seed, then applies the colour matrix. The upscaler is
compiled once per ISA like the rest of the runtime; per half-size row it runs the network on vectors of pixels (the
shipped widths compiled in, the activations in registers), skips blocks of the frame that are empty with their
neighbours (their bytes are zero anyway), and writes each pair of full-size rows from the bilinear 2x, the residuals and
the micro-detail; the noise slices are cached, two at a time. No allocation per frame. `nvfx_study_p check` verifies
that what an instance with `nvfx_instance_set_upscale` draws is, to the bit, the half-size instance's frames upscaled
by the runtime's upscaler, which is how the study's arms are drawn.

## 4. The plan and the rule

### 4.1 Measures

Per effect, size S and setting, R = 4 plays of N and of the S / 2 instance with the same seeds (B and L from the same
half-size frames), looping effects 300 frames (10 s), one-shots their run (explosion 89, mushroom 149):

**(a) Against N on the same seed** (the macro is shared, §2):
- active PSNR (study D's: pixels visible in either frame), pooled over a play's frames, mean over the plays;
- SSIM (Gaussian 11, sigma 1.5, four channels, `metrics::ssim`) on every 10th frame, mean over frames and plays;
- **the high band against N**: the radially averaged log luminance power spectrum (`metrics::stats`' spectrum, pooled
  over the plays as study D2 pools), mean |log10 P_arm - log10 P_N| over the radial bins above the half-size frame's
  Nyquist frequency (bins S / 4 + 1 to S / 2), where upscaling loses detail.

**(b) Against the simulator's real frames at S**: the teacher, the simulation on its 128-cell grid (the resolution
every model was trained and measured at), drawn by its own renderer at S px (its dynamics do not depend on the size it
draws at, §2):
- tracking at 1 s from the true state (study D's protocol, as study M3's: 2 runs per setting from an empty grid, looping
  effects warmed for 100 frames, one-shots 1; the true state with its 128-cell fine fields as the only start point, one
  continuous rollout with the run's seed), active PSNR at frame 30 against the teacher drawn at S;
- the endless statistics of studies D2, K and M3, the R plays pooled against R real runs pooled (looping effects
  warmed for 150 frames): detail spectrum distance, |ln motion ratio|, coverage and emission distances, mean-frame PSNR;
- the high band against the real runs: the same band's mean |log10 P_arm - log10 P_real|.

A 512-cell simulation would have structure at 512 px, but it takes 0.18 s a frame here (143 ms its step, 37 ms its
drawing; 9 ms and 3 ms at 128 cells), about 80 s per 10 s run: about 160 CPU-hours for this study's real runs. So the
teacher's frames at S carry no structure finer than its 128 cells but what its shading makes: the upper bands of the
spectrum at 512 px are where N's detail layer draws its own detail, which no simulation here has checked.

### 4.2 Splits and seeds (study P owns 3,600,000 to 3,999,999)

| split | settings | seeds |
|---|---|---|
| train (the upscaler's tiles) | salt-1 training runs 0 to 15 | plays 3,600,000 |
| dev (calibration, development) | salt-1 training runs 16 to 23 | plays 3,650,000 |
| **validation** (the null probe, the choice, the number of test settings) | study G's 10 validation settings (`validation_settings(10, 30)`, as study M3) | real 3,700,000, plays 3,710,000, the null 3,720,000, tracking 3,730,000 |
| **test**, once | study B's held-out settings extended as study M3's (`test_settings(n)`), n from validation | real 3,800,000, plays 3,810,000, tracking 3,830,000 |
| cost and videos | | 3,900,000 on |

Run k of setting s has seed base + 100 s + k. Settings of the test are at least 0.085 from every validation setting.

### 4.3 The null probe, the margins and the choice (validation)

- **The null**: N played again with fresh seeds (3,720,000), 4 plays pooled per setting, against N; per effect, size
  and (b) measure, **the margin is the larger of study K's margin and the null's RMS change at a setting** over the 10
  validation settings (study M3's form): K's are +0.02 for the spectrum distances (full and high band) and |ln motion
  ratio|, 10% of N's own distance for coverage and emission (at least 0.0001), -0.10 dB for the mean frame; tracking has
  no null (a tracked run takes its own seed, so N plays the same frames) and keeps -0.20 dB (studies M3, N2). The
  margins are computed from N and the null alone (`nvfx_study_p margins`), written to
  [p_margins.csv](../results/experiments/upscale/p_margins.csv) and committed before the test.
- **The choice**, on validation at 512 px: the per-effect upscalers replace the shared one only if they are better than
  it on (a)'s PSNR (per-effect minus shared, interval above zero) for at least 4 of the 6 effects and, on those, not
  worse (interval entirely on the worse side) on (a)'s SSIM or the high band. Otherwise the shared one: **shared is
  preferred when it ties.**
- **The number of test settings**: for the chosen upscaler at 512 px, each effect and each measure of the rule, the
  settings at which the rule's interval would clear its bound with probability 0.9 if the test's mean equalled
  validation's, ((1.96 + 1.28) SD / distance)^2, with the SD at the upper end of its one-sided 80% interval; the
  smallest multiple of 10 covering every finite one, at least 10, at most 20 (the time a test takes here). Measures
  whose validation mean lies on the wrong side have no such number; they are listed.

### 4.4 The rule's form

At **S = 512**, L (the chosen upscaler) is adopted as an opt-in option when all three hold:

1. **It beats B against N** for at least 5 of the 6 effects: L minus B, paired over the test settings, 95% bootstrap
   with 10,000 resamples (`metrics::paired_bootstrap`, seed 2026), interval above zero for PSNR and for SSIM, and below
   zero for the high band against N. All three for an effect to count.
2. **It is within the null's margins of N against the real frames** for at least 5 of the 6 effects: L minus N on every
   (b) measure, the interval's worse end within the effect's margin.
3. **It costs at most 0.55 of N's time per frame** for at least 5 of the 6 effects: per effect, thread CPU time per
   frame through `nvfx_render`, N and L (and B, and the half-size instance alone) interleaved within one process, their
   order rotated each round, 30 rounds of 20 frames, each instance on its own timeline; the median over the rounds of
   L / N. Absolute milliseconds are provisional (a shared machine).

S = 256 is reported the same way and does not decide. Ties are reported as ties; "beats" means an interval clear of
zero.

### 4.5 The rule's numbers (committed after the null probe and validation, before the test)

**The candidates, after calibration** (§3.3, dev split): the shared net with gain 0.10 and 30 slices a second; the
per-effect nets with gains 0.40 (explosion), 0.50 (mushroom), 0.30 (wildfire), 0.10 (launch) and 0 (smoke and the
waterfall, whose residuals gained too little SSIM or PSNR over B to spend on noise), 30 slices a second where the gain
is not 0.

**The margins** ([p_margins.csv](../results/experiments/upscale/p_margins.csv)): the null (N with fresh seeds minus N,
4 plays pooled, over the 10 validation settings) gives every margin but tracking's and emission's; they are wide where
re-seeding moves a measure a lot, as study M3 found. At 512 px (256 px within a few per cent of these, in the CSV):

| effect | spectrum | \|ln motion\| | coverage | emission | mean frame (dB) | high band vs real | tracking (dB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| explosion | +0.062 | +0.050 | +0.0056 | +0.0001 | -1.39 | +0.088 | -0.20 |
| mushroom | +0.086 | +0.099 | +0.0134 | +0.0001 | -1.46 | +0.125 | -0.20 |
| wildfire | +0.031 | +0.037 | +0.0019 | +0.0002 | -0.66 | +0.039 | -0.20 |
| launch | +0.047 | +0.134 | +0.0226 | +0.0002 | -2.11 | +0.056 | -0.20 |
| smoke | +0.049 | +0.081 | +0.0047 | +0.0001 | -1.65 | +0.055 | -0.20 |
| waterfall | +0.023 | +0.049 | +0.0079 | +0.0001 | -1.82 | +0.026 | -0.20 |

**The choice: the shared upscaler** ([p_choice_val.txt](../results/experiments/upscale/p_choice_val.txt)). At 512 px the
per-effect nets are better than the shared one on (a)'s PSNR for all six effects (+0.02 to +0.28 dB, every interval
above zero), but worse on SSIM for the mushroom, launch and smoke (by 0.0001 to 0.0028, intervals entirely below zero)
and on the high band for the waterfall (+0.004 [+0.000, +0.007]); so only the explosion and the wildfire count, 2 of the
4 the choice asks for, and by the rule the shared net is chosen. It is frozen in
[p_frozen.csv](../results/experiments/upscale/p_frozen.csv) (`cand_shared.nvfxup`, SHA-256 `4043e2c7...`, gain 0.10,
30 slices a second); the test plays it as arm `L_chosen`.

**The number of test settings: 20**, the cap ([p_power.csv](../results/experiments/upscale/p_power.csv)): the largest
finite need is 1,430 settings (a tiny but consistent difference); and for nine of the rule's (a) measures the shared
net's validation mean lies on the wrong side, where no number of settings gives power: the explosion's high band, the
wildfire's PSNR, SSIM and high band, launch's SSIM and high band, and the waterfall's PSNR, SSIM and high band.

**What validation says before the test** ([p_decide_val.txt](../results/experiments/upscale/p_decide_val.txt), the
rule's form applied to validation, not deciding): at 512 px the shared net beats B on (a) and the high band for **0 of
the 6** effects (it beats B on PSNR for four, by +0.02 to +0.17 dB, and loses on the wildfire and the waterfall; its
high band is never better than B's) and is within N's margins on (b) for all 6. So condition 1 is expected to fail. The
per-effect nets read 1 of 6 and 6 of 6 at 512 px, 3 of 6 and 5 of 6 at 256 px; the shared net reads 2 of 6 and 5 of 6 at
256 px.

**The test** then runs once as §4.2 and §4.4 say: study B's held-out settings extended to 20 (`test_settings(20)`), 4
plays per setting, 2 tracking runs, seeds 3,800,000 (real), 3,810,000 (plays) and 3,830,000 (tracking), arms N, B and
`L_chosen`, in three processes of two effects each (`--tag rule_<effects>`, each refusing a second run).

**The mushroom stays the frozen one.** Study O replaced the library's mushroom (main 37b08cd: the shared link now points
to `o_fix/models/mushroom.nvfx`, SHA-256 `4a2f08a0...`) while this test was running (it had reached the explosion's
512 px settings; the mushroom's part had not begun). The test plays the mushroom this study froze before any number
(§3.1, [p_models.csv](../results/experiments/upscale/p_models.csv): `fx_blaze/models/d/mushroom.nvfx`, `505bdfe8...`),
because its calibration, the null's margins and the choice were all measured on that model; study O's mushroom is not
measured here. The upscaler itself reads frames, not a model file, and the opt-in path works with either.

**The cost condition** is judged on a run of §4.4's protocol with the chosen net, made after this commit
(`nvfx_study_p cost --net cand_shared`, label `rule`). One run of the same protocol with the per-effect nets was made on
20 October before this commit, on a quiet machine, to see where the cost stands; its numbers are reported in §6 beside
the rule's run and do not decide.

## 5. Validation: the null, the choice (10 settings, 4 plays, 2 tracking runs)

**The null** ([p_margins.txt](../results/experiments/upscale/p_margins.txt)): re-seeding N moves its distance from the
teacher's frames by 0.02 to 0.09 in the detail spectrum, 3% to 13% in motion and 0.7 to 2.1 dB in the mean frame at a
setting (RMS over the settings), so most margins are the null's, 1.0 to 21 times study K's; emission's is study K's
(0.0001) but for the wildfire and launch (0.0002, the null's), and tracking's is -0.20 dB. The margins are in §4.5.

**N against the teacher's frames.** At 512 px N's spectrum is 0.46 (wildfire) to 1.09 (smoke) from the teacher's, and
its high band 0.54 to 1.24 (in log10 power: N draws about 3 to 17 times the teacher's high-band power), because the
teacher at 512 px is drawn from 128 cells and N's detail layer draws at 512.

**The candidates against B and N at 512 px** (validation, the rule's form applied as a reading; full tables in
[p_decide_val.txt](../results/experiments/upscale/p_decide_val.txt)):

| effect | shared: PSNR vs N, L - B | shared: SSIM, L - B | shared: high band vs N, L - B | per effect: PSNR, L - B | per effect: high band, L - B |
|---|---|---|---|---|---|
| explosion | **+0.173 [+0.145, +0.199]** | **+0.0001 [+0.0000, +0.0001]** | +0.000 [-0.022, +0.017] | **+0.451 [+0.363, +0.540]** | -0.039 [-0.092, +0.002] |
| mushroom | **+0.122 [+0.105, +0.138]** | **+0.0001 [+0.0001, +0.0002]** | -0.029 [-0.064, +0.008] | **+0.183 [+0.166, +0.199]** | **-0.153 [-0.187, -0.121]** |
| wildfire | -0.064 [-0.130, -0.008] | -0.0003 [-0.0005, -0.0001] | +0.096 [+0.064, +0.133] | **+0.147 [+0.110, +0.181]** | **-0.041 [-0.053, -0.029]** |
| launch | **+0.015 [+0.003, +0.026]** | -0.0003 [-0.0004, -0.0001] | +0.073 [+0.052, +0.095] | **+0.033 [+0.024, +0.043]** | +0.053 [+0.038, +0.069] |
| smoke | **+0.087 [+0.082, +0.092]** | **+0.0009 [+0.0008, +0.0010]** | -0.005 [-0.031, +0.023] | **+0.357 [+0.335, +0.377]** | -0.033 [-0.076, +0.014] |
| waterfall | -0.011 [-0.017, -0.004] | -0.0004 [-0.0005, -0.0003] | +0.053 [+0.047, +0.059] | **+0.032 [+0.024, +0.040]** | +0.057 [+0.053, +0.060] |

Bold: better than B (interval clear of zero). Against N on (b) both candidates are within every margin for all six
effects at 512 px, and closer to the teacher than N on most measures (the shared net: tracking +0.08 to +0.54 dB, the
spectrum 0.14 to 0.45 nearer), because they carry less of N's fine detail, as B does.

**The choice** (§4.3, [p_choice_val.txt](../results/experiments/upscale/p_choice_val.txt)): the shared net, as §4.5
says: the per-effect nets are better on PSNR for all six effects but worse on SSIM or the high band for four of them.

**B itself** (beside the rule, not deciding; [p_decide_val.csv](../results/experiments/upscale/p_decide_val.csv), rows
`B`): drawn at half size and upscaled by the bilinear 2x, every effect is within N's margins on every (b) measure at
512 px, and nearer the teacher than N on tracking (+0.10 to +0.55 dB), the spectrum (0.16 to 0.46 nearer) and the high
band (0.11 to 0.51 nearer); at 256 px for 4 of the 6 (the wildfire's and the waterfall's motion intervals reach past
their margins).

## 6. The test (once)

Study B's 10 held-out settings and 10 more drawn the same way (`test_settings(20)`), seeds 3,800,000 (real), 3,810,000
(plays) and 3,830,000 (tracking), 4 plays and 2 tracking runs per setting, the shared net frozen (`4043e2c7...`), run
once in three processes of two effects (20 October 2026; markers under `p_upscale`), the mushroom the frozen one
(§4.5) ([p_eval_test.csv](../results/experiments/upscale/p_eval_test.csv),
[p_decide_test.txt](../results/experiments/upscale/p_decide_test.txt),
[p_decide_test.csv](../results/experiments/upscale/p_decide_test.csv)).

**Condition 1, L against B on (a) and the high band, at 512 px: 0 of 6** (L minus B, paired over the 20 settings, 95%
intervals; bold where L is better, interval clear of zero):

| effect | active PSNR vs N (dB) | SSIM vs N | high band vs N (lower is better) | means: B, L PSNR (dB) |
|---|---|---|---|---|
| explosion | **+0.087 [+0.028, +0.140]** | -0.0001 [-0.0002, +0.0000] (tie) | +0.009 [-0.013, +0.028] (tie) | 40.49, 40.57 |
| mushroom | **+0.107 [+0.093, +0.120]** | **+0.0001 [+0.0001, +0.0001]** | -0.001 [-0.040, +0.039] (tie) | 36.55, 36.66 |
| wildfire | -0.045 [-0.090, -0.005] (worse) | -0.0003 [-0.0005, -0.0001] (worse) | +0.085 [+0.060, +0.113] (worse) | 42.29, 42.24 |
| launch | +0.007 [-0.000, +0.016] (tie) | -0.0004 [-0.0005, -0.0002] (worse) | +0.075 [+0.059, +0.091] (worse) | 28.89, 28.90 |
| smoke | **+0.085 [+0.082, +0.088]** | **+0.0010 [+0.0009, +0.0011]** | +0.002 [-0.019, +0.023] (tie) | 25.28, 25.37 |
| waterfall | -0.016 [-0.020, -0.011] (worse) | -0.0004 [-0.0005, -0.0003] (worse) | +0.053 [+0.048, +0.059] (worse) | 30.52, 30.51 |

The learned 2x is a little closer to N than the bilinear 2x on three effects (+0.09 to +0.11 dB) and further on two, and
**its high band is never closer to N's than B's**: on the wildfire, launch and the waterfall it has less of N's fine
energy than B has (its residual, trained for squared error, smooths where the half-size detail is not N's; at gain 0.10
the micro-detail does not make it up), on the other three it ties.

**Condition 2, L against N on the real frames, at 512 px: 6 of 6** within every margin. L is nearer the teacher's frames
than N on tracking at 1 s (+0.08 to +0.53 dB, every interval above zero), the detail spectrum (0.11 to 0.46 nearer) and
the high band (0.08 to 0.58 nearer), and moves more like the teacher (|ln motion ratio| 0.09 to 0.31 smaller for the
wildfire, launch, smoke and the waterfall); the mean frame is within 0.04 dB of N's. As §5 says, that is because L, like
B, carries less of the fine detail N's detail layer draws at 512 px, which the teacher (128 cells drawn at 512) does not
have.

**Condition 3, cost: 4 of 6** (§7).

**So by its rule the learned 2x is not adopted**: condition 1 holds for none of the six effects, condition 3 for four
(five needed); only condition 2 holds.

**At 256 px** (reported the same way, not deciding): condition 1 holds for 1 of 6 (the mushroom: +0.112 dB, +0.0005
SSIM, high band -0.071 [-0.118, -0.023]); PSNR is better for five (+0.02 to +0.19 dB) and a tie for the waterfall,
SSIM better for three, the high band worse for the wildfire, launch and the waterfall. Condition 2 holds for 4 of 6:
the wildfire's and the waterfall's |ln motion ratio| intervals reach past their margins (-0.005 [-0.045, +0.038]
against 0.031, -0.031 [-0.133, +0.070] against 0.047). Condition 3: L costs 0.46 to 0.70 of N (§7), at most 0.55 for 2
of 6.

**B against N** (beside the rule): on the test as on validation, B is within N's margins on every (b) measure for all
six effects at 512 px (for 4 of 6 at 256 px), nearer the teacher than N on tracking (+0.10 to +0.54 dB) and the
spectrum.

## 7. Cost (provisional: a busy machine)

Thread CPU time per frame through `nvfx_render`, the four instances in turn within one process (their order rotated
each round), 30 rounds of 20 frames, each on its own timeline at controls (0.6, 0.5, 0.6), the runtime's default (int8
on AVX-512 with VNNI); the median over the rounds of each ratio to N, and in brackets the 95% bootstrap interval of
the mean ratio ([p_cost.csv](../results/experiments/upscale/p_cost.csv),
[p_cost_rule.txt](../results/experiments/upscale/p_cost_rule.txt); label `rule`, the shared net, run after §4.5 was
committed). H is the half-size instance alone: what B costs when the engine's GPU does the bilinear 2x.

| effect | N, 512 px | H / N | B / N | **L / N** | N, 256 px | H / N | B / N | L / N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| explosion | 4.16 ms | 0.343 | 0.426 | 0.571 [0.557, 0.614] | 1.24 ms | 0.374 | 0.441 | 0.554 [0.527, 0.633] |
| mushroom | 6.00 | 0.310 | 0.361 | **0.453** [0.459, 0.543] | 1.76 | 0.337 | 0.391 | 0.455 [0.480, 0.555] |
| wildfire | 4.32 | 0.291 | 0.369 | 0.593 [0.558, 0.618] | 1.29 | 0.432 | 0.478 | 0.700 [0.663, 0.742] |
| launch | 4.33 | 0.302 | 0.367 | **0.536** [0.520, 0.560] | 1.32 | 0.420 | 0.484 | 0.640 [0.606, 0.673] |
| smoke | 4.70 | 0.325 | 0.395 | **0.550** [0.526, 0.599] | 1.56 | 0.396 | 0.432 | 0.537 [0.518, 0.619] |
| waterfall | 4.45 | 0.298 | 0.369 | **0.530** [0.486, 0.646] | 1.24 | 0.421 | 0.482 | 0.610 [0.590, 0.630] |

**Condition 3 is not met**: at 512 px L costs at most 0.55 of N for 4 of the 6 effects (the mushroom 0.453, launch
0.536, smoke 0.5498, the waterfall 0.530), not 5; the explosion (0.571) and the wildfire (0.593) cost more. The effects
that fill the most of their frame skip the fewest empty blocks: the network runs on every half-size pixel that has
something near it, and the learned part costs 0.55 (the mushroom) to 0.97 ms (the wildfire) of a busy core at 512 px
beyond the bilinear 2x. The run with the per-effect nets made before §4.5 was
committed, on a quiet machine ([p_cost_per_effect.txt](../results/experiments/upscale/p_cost_per_effect.txt)), read the
same: 0.477 to 0.595 at 512 px, 4 of 6 at or under 0.55 (the explosion 0.548, the mushroom, smoke and the waterfall).

**What drawing at half size saves by itself**: the half-size instance costs 0.29 to 0.34 of N at 512 px and 0.34 to
0.43 at 256 px (the coarse step, about 0.3 ms, does not shrink); the CPU's bilinear 2x adds 0.04 to 0.08 of N, nothing
on a GPU. At 256 px L costs 0.46 to 0.70 of N.

## 8. What it means

- **For an engine: draw large effects at half size and let the GPU upscale.** The half-size instance costs about a
  third of native at 512 px (0.29 to 0.34; 0.34 to 0.43 at 256 px, where the fixed coarse step weighs more), and
  against the simulator it plays as well as native or better on every measure studied. It is softer than native in the
  finest filaments; whether a game needs those at the size it draws the effect is a judgement by eye (the videos:
  videos/README.md §9b). Creating the instance at half size and drawing its texture at full size needs no new call;
  `NVFX_UPSCALE_BILINEAR` does the same on the CPU for engines that want the full-size bytes (0.04 to 0.08 of native
  more).
- **The learned 2x does not earn its cost.** Native's finest detail is not in the half-size frame, not even blurred: the
  half-size detail layer draws its own (§2), so a 2x trained for squared error can only sharpen a little (+0.07 dB over
  bilinear for the shared net on training tiles, up to +0.55 dB per effect), and seeded noise adds high-band energy only
  at the price of PSNR and SSIM. At the cost the plan could pay (128 multiply-adds per pixel, 0.55 to 0.97 ms of a busy
  core at 512 px) it did not beat bilinear on all three of (a)'s measures for any effect. The per-effect nets did better
  on PSNR and the high band (validation, §5) but lost SSIM for four effects, so the rule kept the shared one.
- **The simulator's frames at 512 px are not a reference for fine detail.** Drawn from 128 cells, they have about a
  seventeenth to a third of native's high-band power; native's detail at 512 px is the detail layer's own, checked
  against nothing. A 512-cell teacher (0.18 s a frame here) would be needed to say whether native's filaments are
  right; it is a task for the PC.
- **What would make a learned upscaler worth it**: a target it can reach (a half-size instance whose fine fields are
  native's averaged, e.g. by running the detail layer at full size and only the renderer at half size, or a renderer
  that draws filaments from the coarse flow), and a cheaper kernel for effects that fill their frame (the wildfire and
  launch skip the fewest empty blocks).

## 9. Reproduce

```sh
P="nvfx_study_p"                                  # data root $NEURALVFX_DATA; this study's files under p_upscale
$P pin && $P check && $P info                      # the six models frozen, the shared macro checked, the coverage
$P tiles --sizes 256,512                           # training tiles (training settings and seeds)
$P train --effects all --out cand_shared --steps 4000 --amp-from 2000 --threads 3
$P train --effects explosion --out cand_explosion --steps 4000 --amp-from 2000 --threads 3   # and each effect
$P calib --net cand_shared --settings 2 && $P calib --net cand_explosion --effects explosion --settings 3   # and each effect
$P eval --split val --null --cands shared=cand_shared,effect=cand_{effect}   # the null probe and validation
$P margins && $P choose                            # the margins, the choice, the number of test settings
$P eval --split test --settings 20 --cands chosen=cand_shared --effects explosion,mushroom --tag rule_explosion_mushroom   # once
#   (and wildfire,launch and smoke,waterfall; each writes its own rows and refuses a second run)
$P decide --split test                             # the rule; B against N beside it
$P cost --net cand_shared --rounds 30 --frames 20 --label rule
$P video --effect explosion --net cand_shared --size 512 --controls 0.85,0.5,0.6 --seed 3900301 --crop 30,90,170 --out upscale_explosion.mp4
```

The six models are the files frozen in [p_models.csv](../results/experiments/upscale/p_models.csv) (the data root's
`p_upscale/models`), the candidates `p_upscale/nets/cand_*.nvfxup`; the training tiles and the real runs' cache were
deleted after use (they are made again by `tiles` and by `eval`). The waterfall plays as its file says, one detail pass
a frame; study O's `passes 2` is the library's choice in its effect lines, not in the file.
