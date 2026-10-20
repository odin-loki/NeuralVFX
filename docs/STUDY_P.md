# Study P: large effects drawn at half size and upscaled

Status: **paused** (15 October 2026; §0). Audience: dev, research, the owner. The design and the plan (§1 to §4) are
committed before any validation or test number; the rule's numbers (§4.5) follow the null probe and validation and are
committed before the test, which runs once. **The test has not run; its seeds (3,800,000 on) are unspent.**

## 0. Status: paused

Paused when the owner's weekly usage ran low. Where it stands:

- **Done**: the design (§1 to §3), the runtime path (`nvfx_effect_attach_upscaler`, `nvfx_instance_set_upscale`,
  NVFX_VERSION 7, `tests/test_upscale.cpp`: 11 tests, frozen AVX2 float frames), the study tool, the six effects frozen
  (launch for cloud, §3.1), training tiles, the candidates trained (one shared, one per effect, 8, 8, 1).
- **Found** (§2): the render size enters only the detail layer and the renderer; the fine fields never feed back into
  the coarse state, so N and the half-size instance share the macro motion to the bit (checked on all six). But the
  half-size detail layer draws its own fine detail (its picture is 25 to 44 dB from N's averaged), so against N on the
  same seed a learned 2x gains little over bilinear (+0.07 dB on held-out training tiles for the shared net; +0.06 to
  +0.55 dB per effect), and micro-detail noise trades PSNR and SSIM for high-band energy. On one dev setting at 512 px
  the teacher's frames (128 cells drawn at 512) have about a tenth of N's high-band power, so B is closer to them than
  N. Cost probe (busy machine, 8 rounds): L at 0.45 to 0.56 of N at 512 px, 0.51 to 0.64 at 256 px.
- **The rule**: its form (§4.4) is committed (commit 0108862); its numbers (§4.5: margins, the choice, the number of
  test settings) are **not**, because the null probe has not finished.
- **Still running when paused** (left alone): `jobs/train_calib.sh` (calibrating the candidates' micro-detail gain and
  rate on the dev split, in place in `nets/`) and `jobs/probe.sh` (the null probe on validation: N, N with fresh seeds,
  B and the real runs; rows to `results/experiments/upscale/p_eval_val.csv` in the worktree, not committed while being
  written). Their logs end with `logs/train_calib.done` and `logs/probe.done`.
- **Data** under `/root/nvfx-data/p_upscale` (about 0.4 GB): `models/` (the six frozen effects), `tiles/` (training
  tiles, 393 MB, scratch: delete when the study ends), `nets/` (`cand_shared.nvfxup`, `cand_<effect>.nvfxup`, and
  development nets `dev_*`, `arch_*`, `hf_*`, `s512_*`, `*_881`), `cache/` (the real runs pooled per setting), `logs/`,
  `jobs/`, `bin/nvfx_study_p_v1` (the binary the probe runs), `scratch/`.

**To resume** (from the worktree, `NEURALVFX_DATA=/root/nvfx-data`, `P="nice -n 10 build/nvfx_study_p"`):

```sh
cat /root/nvfx-data/p_upscale/logs/calib_cand_*.log /root/nvfx-data/p_upscale/logs/probe.log   # finished? (*.done)
# if a job died, rerun it: both resume (training and calibration rewrite their nets; eval skips finished settings)
/root/nvfx-data/p_upscale/jobs/train_calib.sh      # or only its calib lines
/root/nvfx-data/p_upscale/jobs/probe.sh
$P margins                                         # the margins from N and the null alone: p_margins.csv
$P eval --split val --cands shared=cand_shared,effect=cand_{effect} --threads 2   # validation of the candidates
$P choose                                          # the choice and the number of test settings: p_choice.txt, p_power.csv
# write §4.5 (margins, choice, settings), commit and push, then once:
$P eval --split test --settings N --cands chosen=cand_shared   # or cand_{effect}; refuses a second run
$P decide --split test && $P cost --net cand_shared --rounds 30 --frames 20
$P video --effect explosion --net cand_shared --out videos/upscale_explosion.mp4 --size 512
```


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

Six large ones from the 21 shipped models, as the library ships them now, copied to the data root's `p_upscale/models`
and frozen by SHA-256 in [p_models.csv](../results/experiments/upscale/p_models.csv): **explosion** (study M3's),
**mushroom**, **wildfire**, **launch**, **smoke** and **waterfall**.

**One change from the suggested set: launch instead of cloud.** How much of its frame each shipped effect fills, at 8
training settings, 128 px ([p_coverage.txt](../results/experiments/upscale/p_coverage.txt), `nvfx_study_p info`):
wildfire draws 71% of its pixels, launch 48%, waterfall 42%, explosion 25%, smoke 22% and mushroom 20%, but cloud only
12% (mean alpha 0.04), 16th of 21; launch is second (mean alpha 0.31, the highest). The explosion, the mushroom and smoke
are kept as suggested: they grow to fill their frames and are the effects a game draws largest.

### 3.2 The upscaler (`include/neuralfx/upscale.hpp`, `src/core/upscale.cpp`)

On the half-size frame (premultiplied RGBA8, n x n), per half-size pixel:

- a 3 x 3 convolution, 4 -> c1, ReLU; a k2 x k2 convolution, c1 -> c2, ReLU; a 1 x 1 output layer with 20 values: the
  residuals of the pixel's 2 x 2 full-size pixels (16) and the micro-detail's amplitude per channel (4, through a ReLU);
- out = bilinear 2x + residual + gain x amplitude x eta, then the instance's colour matrix, clamped, to bytes.

**eta** is seeded noise at the full size: white noise (one hash per pixel from the instance's seed and a time slice) minus
its 3 x 3 mean, scaled to unit variance, so its energy sits at the finest scales; slices change `rate` times a second and
are blended by a smoothstep, with the variance kept at 1. It depends on the seed, the pixel and the frame's time only.
**Pixels where the bilinear 2x is empty in every channel stay exactly transparent**, as a rollout effect's empty pixels
are. The file (`.nvfxup`, magic NVFXUPS1) holds c1, c2, k2, the gain, the rate, a tag and the weights: 2.2 KB for the
shipped size.

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
double the network's part. **The study uses 8, 8, 1.** One net serves both sizes: a net trained on 512 px tiles alone did
no better on them than one trained on both (`tile-eval`).

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
- **The micro-detail's gain and rate** (`nvfx_study_p calib`, the dev split: training runs 16 on, seeds 3,650,000): the
  gain is the largest of 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75 and 1 whose PSNR and SSIM against N each keep at least half of
  the residual's own gain over B (gain 0), means over the effects and sizes; if the residual gains nothing on either
  measure, there is no micro-detail. The rate is the one of 4, 8, 15 and 30 slices a second whose motion is closest to
  N's. A high-band loss that made the residual itself sharpen (restoring N's high-pass energy from 21% to 87% of N's) was
  tried and dropped: it cost 0.6 dB of PSNR against N on held-out tiles, which the rule's first condition does not allow.

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

**(b) Against the simulator's real frames at S**: the teacher, the simulation on its 128-cell grid (the resolution every
model was trained and measured at), drawn by its own renderer at S px (its dynamics do not depend on the size it draws at,
§2):
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
  validation's, ((1.96 + 1.28) SD / distance)^2, with the SD at the upper end of its one-sided 80% interval; the smallest
  multiple of 10 covering every finite one, at least 10, at most 20 (the time a test takes here). Measures whose
  validation mean lies on the wrong side have no such number; they are listed.

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

S = 256 is reported the same way and does not decide. Ties are reported as ties; "beats" means an interval clear of zero.

### 4.5 The rule's numbers (committed after the null probe and validation, before the test)

**The candidates, after calibration** (§3.3, dev split): the shared net with gain 0.10 and 30 slices a second; the
per-effect nets with gains 0.40 (explosion), 0.50 (mushroom), 0.30 (wildfire), 0.10 (launch) and 0 (smoke and the
waterfall, whose residuals gained too little SSIM or PSNR over B to spend on noise), 30 slices a second where the gain is
not 0.

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

**The cost condition** is judged on a run of §4.4's protocol with the chosen net, made after this commit
(`nvfx_study_p cost --net cand_shared`, label `rule`). One run of the same protocol with the per-effect nets was made on
20 October before this commit, on a quiet machine, to see where the cost stands; its numbers are reported in §6 beside
the rule's run and do not decide.

## 5. Reproduce

```sh
P="nvfx_study_p"                                  # data root $NEURALVFX_DATA; this study's files under p_upscale
$P pin && $P check && $P info                      # the six models frozen, the shared macro checked, the coverage
$P tiles --sizes 256,512                           # training tiles (training settings and seeds)
$P train --effects all --out cand_shared --steps 4000 --amp-from 2000 --threads 3
$P train --effects explosion --out cand_explosion --steps 4000 --amp-from 2000 --threads 3   # and each effect
$P calib --net cand_shared --settings 2 && $P calib --net cand_explosion --effects explosion --settings 3   # and each effect
$P eval --split val --null --cands shared=cand_shared,effect=cand_{effect}   # the null probe and validation
$P margins && $P choose                            # the margins, the choice, the number of test settings
```
