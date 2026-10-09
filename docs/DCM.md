# Study G: diffusion-context mixing for generating effects

Status: **plan** (9 October 2026). Each section turns into a result when its stage finishes. Audience: owner,
research, dev.

## 1. The algorithm and where it comes from

Diffusion-context mixing (DCM) is the owner's algorithm, built in CameraDetector (Phase 12: `cabinlab/src/diffusion`,
`docs/DIFFUSION_MIXER_PLAN.md`) to detect seatbelt violations: "a mixer for the fine features and diffusion for the
macro features. A mixer search, so to speak."

```
fine experts (many cheap, weak predictions) ───────────────────────► mixer INPUTS (log-odds)
macro view ─► small denoiser at high noise ─► PCA ─► k-means ──────► mixer CONTEXTS (+ hand-made contexts)
inputs + contexts ─► PAQ8-style MixerNet: a first-layer mixer without context + up to 3 context-selected ones
                     ─► final mixer ─► adaptive probability map (APM) ─► prediction
mixer search: which inputs, contexts, rates, passes, APM; nested (the held-out part never chooses its own structure)
```

What CameraDetector found: the mixer worked, by a small margin. The diffusion contexts did not. They clustered frames
by camera and light (nuisances a classifier must ignore), and as contexts they lowered the held-out score from 0.792 to
0.770. Its decision was "stop the diffusion part; keep the mixer".

This study adapts DCM from classifying to **generating** effects. It is the owner's request: "use a content mixer,
specifically the one and algorithm I made in CameraDetector ... adapt it to generating SFX ... trial some stuff with
that and then meet both in the middle". The mixer, its frozen inference copy and its search are ported as they are.
The k-means and the denoiser are rewritten without LibTorch, which this project does not use.

## 2. From classifying to generating

DCM's split matches study D's ([REPORT.md](REPORT.md) §6): a 32 x 32 **macro** state stepped by a learned network, and
**fine** heat and soot made by a hand-made detail layer.

| CameraDetector | NeuralVFX |
|---|---|
| fine experts: strap and edge statistics, detector scores (log-odds) | fine experts: cheap per-pixel predictions of the next fine heat or soot (the advected field, the detail layer's own lock, the upsampled coarse value, new material shaped by each noise octave, gradients, the previous frame) |
| macro: denoiser features of the wide crop, clustered, as contexts | macro: a small denoiser on the 32 x 32 coarse state; as contexts, as a prior against long-run drift, and as a source of start points |
| PAQ8 MixerNet with an APM: a probability | the same structure in a value domain (ValueNet: normalised-LMS updates, a Laplace loss, an adaptive value map, a scale net): a distribution per pixel, sampled with the effect's own coherent noise |
| target: one label per crop, 3,058 crops | target: the simulator's true next fine field, dense and unlimited |
| nested leave-one-camera-out search, AUC | nested leave-one-control-bin-out search, held-out bits per pixel; re-ranked by generation statistics |
| frozen mixer, SHA-256 version | the same, recorded in the model file and in every CSV |

A predicted distribution is also a code length. So the same mixer that generates detail can code it, which links
generation to compression (§3, G3).

## 3. Designs and decision rules

| id | design | stage | kept only if |
|---|---|---|---|
| G1 | **DCM-fine:** the mixer replaces the detail layer's lock and new-material rule; it starts exactly as the hand-made rule (weight 1 on it) | S2 | on held-out settings its detail spectrum distance beats the hand-made layer's (interval excluding zero) on at least 2 of 3 effects, no effect is worse, motion and mean-frame PSNR are no worse, and it costs at most +1 ms per 128 x 128 frame. Better bits per pixel alone do not count |
| G1c | cold start: the mixer grows fine detail from a coarse state in a few frames (no warm-up, no stored fine fields) | S2 | its first second's statistics tie or beat the current start |
| G2a | denoiser features as mixer contexts | S5 | they beat the hand-made contexts in the nested search beyond the search's own seed noise; first checked by their mutual information with those contexts |
| G2b | denoiser as a prior against long-run drift (a one-step denoise every few frames) | S5 | one long rollout drifts less than without it on validation settings, within 0.5 ms per frame |
| G2c | denoiser start points (sampled, or a stored start moved to new controls) | S5 | they beat the nearest stored start rolled ahead, within 100 ms per shard |
| G3a | **codec for authored runs:** the stepper predicts, a PAQ coder stores the coarse corrections every few frames, detail is synthesised | S4 | it is a rate-distortion curve: claimed only where it crosses the flipbook and video-codec curves, at a stated quality |
| G3b | frame model plus a coded residual (study A clips) | S4, if time | as G3a |
| G3c | start points quantised, dithered with the seed's own noise and coded; fewer of them | S3 | test statistics tie or beat the current files |
| G4a | one renderer for every model: a mixer over the learned renderers, the field shader and the simulator's renderer | S8 | the explosion's first frames improve, and modules match in look |
| G5a | shard critic: a frozen binary mixer picks the most real-looking of a few rolled-ahead seeds | S8 | endless statistics improve on validation, then on test |
| G5b | a mixer over the stepper's update and a cheap solver's update | S8 | smoke tracks better after a few seconds |
| G6 | one DCM-fine for every effect (the effect is a context) | S8 | as G1 |

When a design does not pass, it is reported as a null result with its numbers, as CameraDetector reported its own.

What could repeat CameraDetector's outcome, and how we find out fast:
- **The risk:** the hand-made contexts (heat level, height, flow, controls) may already describe the regimes, so
  clusters of denoiser features only add selection noise, as camera and light did.
- **What is different:** contexts here pick a regime of a physical predictor, not a nuisance. Data is unlimited
  (about 300,000 pixel rows per effect). The objective (bits per pixel over dense targets) is far less noisy than
  per-camera AUC. A denoiser on 32 x 32 costs a fraction of a millisecond, not 6 ms.
- **Fast checks before any denoiser is built:**
  - a pilot of the default mixer against the hand-made layer on fire;
  - contexts from plain coarse statistics (no diffusion) as the floor;
  - the mutual information of any clusters with the hand-made contexts.

## 4. Protocol

- **Splits:**
  - training: study D's runs (salt 1);
  - validation, for every choice, search, calibration and re-ranking: salt-3 runs and 10 validation settings;
  - test, used once per stage: study B's 10 held-out settings with new seeds and the salt-2 tracking runs, exactly
    as study D.
- **Nested search sites:** 5 bins of the training runs' controls.
- **Metrics:**
  - endless runs: detail spectrum distance, motion ratio, coverage and emission distances, mean-frame PSNR;
  - tracking: active PSNR at 1, 8, 30 and 60 frames;
  - one step: bits per active pixel and squared error;
  - frame models: study A's metrics.
- **Bytes**, each reported on its own: disk (file bytes), resident memory, working memory per instance, decode or
  load time, ms per frame.
- **Intervals:** 95% paired bootstrap (10,000 resamples) over settings or runs; an interval covering zero is a tie.
  The search's own seed noise is measured with three seeds on fire.
- **Determinism:** every released mixer and denoiser carries the SHA-256 of its serialisation. 100 repeated steps give
  one hash.
- **Timing:** only on a quiet machine (load below 1.5), one pinned core, 200 frames, median and p90.
- **Data:** under `/root/nvfx-data/{g,f}`, never in git.

## 5. Stages

| stage | what | who | ends with |
|---|---|---|---|
| S0 | port the mixer, its frozen copy and the search; k-means and PCA without LibTorch; the value-domain mixer; tests; this plan | a porting agent and the main agent | `neuralfx_dcm` on main |
| S1 | merge the fireball optimisation branches and the context-mixing coder; re-profile; freeze v1 | main agent | the fireball after round 1, study F in the report |
| S2 | G1 and G1c | an agent, 2 threads | G1 decided |
| S3 | compression past 10x: flipbooks and video codecs (x264, AV1, VP9 with alpha) as baselines, longer training, 4 to 6-bit features, rollout start points (G3c); RAM and disk separate | an agent, 2 threads | ratios with intervals |
| S4 | G3a (and G3b) | an agent | rate-distortion curves |
| S5 | G2a, G2b, G2c with a hand-written denoiser | an agent | G2 decided per use |
| S6 | meet in the middle: v2 rollout effects (every part that passed, in the runtime with parity and zero-allocation tests); the fireball re-rendered with v2 and compared with v1 | main agent | v2, and the fireball v1 against v2 |
| S7 | a second optimisation round; the final 30 fps video | agents | final profile and video |
| S8 | extras on idle cores: G5a, G4a, G6, G5b | agents | one-line decisions |

The core is S1, S2, S3 and S6. If time runs short, the cuts are S8, then S5's extension beyond fire, then G3b, then
S7's speed targets (the final video stays).

## G2. Diffusion for the macro features (stage S5)

Status: **done for fire** (9 October 2026). The prior against drift (G2b) is kept: it passed validation and its one
test. Diffusion start points (G2c) are stopped. The diffusion contexts (G2a) are not redundant by the first check;
their decision waits for stage S2's nested search, which is not on main yet. Smoke and explosion are not started: that
is a later decision. Rules in G2.3 were fixed before any result; results are in G2.4 to G2.9.

### G2.1 The denoiser

`include/neuralfx/dcm/ddpm.hpp`, `src/dcm/ddpm.cpp`, `src/dcm/ddpm_kernels.hpp` (library `neuralfx_ddpm`). The owner's
denoiser (CameraDetector, `cabinlab/src/diffusion/denoiser.cpp`) is a LibTorch UNet; this one is written out by hand,
forward and backward, because this project uses no LibTorch.

- **What it models:** the 32 x 32 coarse state of a rollout effect (velocity, heat, soot), each channel divided by the
  stepper's channel scale (`rollout::Model::scale`), plus two position channels as inputs.
- **The network:** an epsilon-prediction UNet with levels 32 x 32 (32 channels), 16 x 16 (64) and 8 x 8 (64): a 3 x 3
  stem, two residual blocks per level (one on the way down and one on the way up at 32 and 16, two at 8), 2 x 2
  average pooling and nearest upsampling with 1 x 1 convolutions, additive skips, and a 3 x 3 output layer. A block is
  `x + conv3(SiLU(FiLM(conv3(SiLU(x)))))`. FiLM comes from one two-layer MLP over the sinusoidal embedding of the noise
  level and the controls (and the age, for one-shot effects). No attention and no GroupNorm: the second convolution of
  every block and the output layer start at zero, gradients are clipped and the learning rate warms up instead.
- **Size and cost:** 391,748 parameters and 89.5 million multiply-adds per pass. The brief asked for about 0.25 M and
  60 M; with the brief's widths (32/64/64) and two blocks per level, the 64-wide blocks alone hold 295,000 weights, so
  the brief's widths were kept and the numbers are reported as they are.
- **Kept from the owner's version:** the cosine schedule with its beta clip (T = 1000), epsilon prediction, identity
  start of every block, Adam with a linear warm-up then cosine decay to 10%, gradient clipping, an EMA copy (0.99 for
  500 steps, then 0.999), and features read at a fixed noise level with one fixed noise image per level and seed.
- **Sampling:** deterministic DDIM (eta = 0) on a 25-step grid; SDEdit noises a state to t0 and runs the same grid down.
  A fresh sample starts at t = 961, the first point of DDIM's grid, not at T: there the clipped schedule leaves
  alpha_bar at about 2e-9 and the first estimate of x0 is pure error. Found by a sanity check of the samples' channel
  statistics after training (samples started at T were several times too hot and too spread), before any G2c run.
- **Tests** (`tests/test_dcm_ddpm.cpp`): the fast AVX2 kernels equal the plain convolution patterns copied from
  `rollout_train.cpp`; the hand-written gradient matches finite differences on an 8 x 8 toy for every part of the network;
  training lowers the held-out loss on a toy set; ten repeats of a short training give one SHA-256 of the weights, for
  one or two threads; `features_at()` is bit-identical for a seed; Tweedie's denoise and the predicted noise recompose
  x_t; DDIM is deterministic and stays in range; serialisation round-trips.

### G2.2 Data and training

- **States:** study D's training recipe for fire (`record_runs`, salt 1, 160 runs of 240 frames, as `d-train`), every
  second coarse state after the effect's 1 s warm-up (frames 30 to 238): 16,800 states. Validation states: 16 runs of
  salt 3, the same frames (1,680 states). Under `NEURALVFX_DATA/g/diff/`, never in git.
- **Training:** `nvfx_dcm ddpm-train`: batch 32, Adam (learning rate 5e-4, 500 warm-up steps), clip 1, EMA 0.999, two
  threads at `nice 10`; the number of steps is set from a timed probe to fit about 3 to 4 CPU-hours. The EMA loss at
  t = 50, 200, 500 and 800 on 256 validation states (fixed noise) is logged every 500 steps
  (`results/experiments/g_diff_train.csv`).

### G2.3 Rules, fixed before any result

- **Validation:** salt-3 runs and 10 validation settings: `std::mt19937_64 rng(2027)`, each control uniform in
  [0.05, 0.95], off study B's training grid like B's own held-out settings (at least 0.05 from 0, 0.5 and 1 for intensity
  and turbulence, from 0, 0.25, 0.5, 0.75 and 1 for wind) and at least 0.05 (Euclidean distance in control space) from
  every one of B's 10 held-out settings (`results/experiments/g_diff_settings.csv`). Every choice below is made on
  validation.
- **Test, once, only for a use that passes validation:** B's 10 held-out settings with new seeds (and the salt-2 runs),
  as `d-eval` (`nvfx_experiment g-diff-test`, which refuses to run a use that did not pass).
- **Intervals:** 95% paired bootstrap, 10,000 resamples; an interval covering zero is a tie.
- **Score of generated frames:** the detail score of study D's calibration: detail spectrum distance + |ln motion
  ratio| + |ln emission ratio| + |ln coverage ratio| against a real run at the same setting (lower is better).
  Spectrum, motion, coverage and mean-frame PSNR are reported beside it.
- **Timing** only on a quiet machine (load below 1.5), one pinned core; otherwise the number is reported as an upper
  bound and marked unmeasured.

| use | alternative without diffusion | how it is chosen and scored | kept only if |
|---|---|---|---|
| **G2a** contexts: denoiser features at t = 400 and 600 from the 16 x 16 and 8 x 8 levels (blocks e1, m1, d1), averaged per 8 x 8 region, standardised, PCA-16, k-means (K = 4, 8, 16), fitted on training states | hand-made contexts: heat level (empty, then terciles), height band (4), flow speed (terciles), control bins (each control below or above 0.5); and plain coarse-statistics contexts (the same PCA and k-means on the region's raw coarse values) as the floor | check 1 on validation states: normalised mutual information, and the share of the diffusion contexts' entropy that the joint hand-made context explains, I(D; H) / H(D) | check 1: **redundant, stop**, if at every K the joint hand-made context explains at least 80% of the diffusion contexts' entropy. Otherwise the nested search of stage S2 decides: they must beat the hand-made contexts beyond the search's own seed noise (§3) |
| **G2b** prior against drift: every N frames, Tweedie's one-step denoise at a small t, x0_hat = x - sqrt((1 - abar) / abar) eps_hat(sqrt(abar) x, t), blended with weight beta into the coarse state (no noise added) | the same continuous rollout without the prior; the runtime's 6 s shards as a reference | N in {4, 8, 16}, t in {20, 50, 100}, beta in {0.25, 0.5, 1}, chosen by the mean detail score of the six 10 s windows of one continuous 60 s rollout at validation settings 1 and 2 (tuning seeds). The decision uses fresh seeds at the same two settings, paired over the 12 (setting, window) pairs | the interval of (prior - none) lies below zero, and the prior costs at most 0.5 ms per frame (one pass every N frames) |
| **G2c** start points: a fresh DDIM sample at the requested controls, or SDEdit of the nearest stored start from t0 in {300, 400, 500} | the nearest stored start rolled ahead 0.5 s at the requested controls (the stored start as it is, as a reference) | every start plays as fire does (a 1 s warm-up that grows the fine fields) and its first 2 s are scored against a real run, at the 10 validation settings. The variant is chosen on tuning seeds (2 per setting); the decision uses 2 fresh seeds per setting, paired over the 10 settings. Diversity: mean pairwise distance between 8 generated starts against 8 real states at a setting | the interval of (variant - rolled start) lies below zero, and generating costs at most 100 ms per shard (DDIM passes x one pass) |

Two amendments, made before the trained denoiser was run on any of these comparisons (only a 500-step checkpoint, to
test the pipeline):
- **Two candidates per use.** One pass of this network took 6 to 10 ms on one core of the (busy) machine, and its
  89.5 million multiply-adds need about 4.5 ms even at the kernels' best speed (20 GMAC/s), so N = 4 and N = 8 cannot
  meet 0.5 ms per frame, and a fresh 25-step sample cannot meet 100 ms per shard. Choosing only the best score
  could therefore stop a use for its cost while an affordable setting works. So two candidates go from tuning to
  the decision: the best overall, and the best that can meet the bound (N = 16 for G2b; the best SDEdit for G2c). A
  use is kept when a candidate passes both the interval and the cost bound. These are two looks at the validation
  seeds, and are reported as such; a test, if one is earned, uses the cheaper passing candidate.
- **The rolled start** steps the whole effect (coarse state, memory and fine fields) 15 frames after its start-up, as
  the runtime rolls a shard ahead. SDEdit's t0 is rounded to the 25-step grid (multiples of 40): 300, 400 and 500 start
  at 320, 400 and 520 (8, 10 and 13 passes).

What would repeat CameraDetector's outcome here: the denoiser's clusters follow what the hand-made contexts already
say (how hot a region is, how high, which controls), as CameraDetector's followed camera and light; the prior pulls
the state towards an average fire and takes the flicker with it; generated starts are softer and less varied than
real states, the usual failure of a small diffusion model trained briefly.

### G2.4 Training

10,000 steps of batch 32 (320,000 examples, 19 passes over the 16,800 states) in 3,830 s on two threads at `nice 10`:
about 2.1 CPU-hours. The probe ran at 0.59 s per step on a machine loaded by other agents; the load fell during the run
(0.38 s per step on average), so the run used less than the 3 to 4 CPU-hours it was sized for. Loss of the EMA weights on
256 validation states (salt 3), fixed noise (`results/experiments/g_diff_train.csv`):

| step | training loss | t = 50 | t = 200 | t = 500 | t = 800 |
|---:|---:|---:|---:|---:|---:|
| 500 | 0.416 | 0.513 | 0.199 | 0.099 | 0.062 |
| 1,000 | 0.110 | 0.395 | 0.174 | 0.083 | 0.047 |
| 2,000 | 0.073 | 0.291 | 0.119 | 0.055 | 0.026 |
| 4,000 | 0.060 | 0.186 | 0.087 | 0.040 | 0.017 |
| 6,000 | 0.054 | 0.168 | 0.080 | 0.037 | 0.015 |
| 8,000 | 0.051 | 0.161 | 0.077 | 0.035 | 0.014 |
| 10,000 | 0.050 | 0.158 | 0.076 | 0.035 | 0.014 |

The curve is flat over the last 2,000 steps (the learning rate has decayed to 10%). CameraDetector's denoiser ended
at 0.436 / 0.400 / 0.388 / 0.394 on its canvases; the data differ, so this is no ranking, but coarse fire states leave
much less noise unexplained at middle and high noise levels. The released denoiser is
`NEURALVFX_DATA/g/diff/fire.ddpm`, version `1152045db43ea534ff5b80099d538c1c0be555d3cb6c5e6aef701a933d8bf56a`.

What its samples look like (`nvfx_dcm ddpm-sample`, network units, at controls 0.5 / 0.5 / 0.5): fresh 25-step DDIM
samples have heat mean 0.06 and spread 0.25 where real states near those controls have 0.16 and 0.70. They are colder
and smoother than real states, and hardly change with the controls (heat mean 0.08 at 0.9 / 0.2 / 0.8, real 0.34). A
1000-step ancestral sampler (a scratch check, not used) lands nearer in mean (0.20) but over-spreads (1.0), and costs
1000 passes. SDEdit from t0 = 400 returns almost its input (channel means within 0.01 of the stored state's).

### G2.5 G2a: contexts, check 1 (mutual information)

`nvfx_dcm contexts`: fitted on 1,500 training states (96,000 regions), scored on all 1,680 validation states (107,520
regions). PCA-16 keeps 79% of the denoiser features' variance and 97% of the plain statistics'. Share of each context's
entropy explained by the hand-made contexts, I(D; H) / H(D), and NMI (`results/experiments/g_diff_nmi.csv`):

| contexts | K | entropy (bits) | heat | height | flow | controls | joint hand-made | NMI with plain, same K | ARI between two k-means seeds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| diffusion | 4 | 1.47 | 0.45 | 0.06 | 0.40 | 0.01 | **0.65** | 0.39 | 0.998 |
| diffusion | 8 | 2.38 | 0.28 | 0.10 | 0.27 | 0.04 | **0.55** | 0.42 | 0.80 |
| diffusion | 16 | 3.61 | 0.20 | 0.11 | 0.19 | 0.05 | **0.47** | 0.37 | 0.86 |
| plain statistics | 4 | 1.06 | 0.45 | 0.02 | 0.40 | 0.06 | 0.64 | | |
| plain statistics | 8 | 1.68 | 0.34 | 0.05 | 0.35 | 0.05 | 0.58 | | |
| plain statistics | 16 | 2.49 | 0.26 | 0.05 | 0.33 | 0.05 | 0.53 | | |

- **Not redundant by the rule:** the joint hand-made context explains 47% to 65% of the diffusion contexts' entropy,
  below the 80% line, at every K. They follow how hot a region is and how fast it moves (NMI 0.29 to 0.47 with heat,
  0.27 to 0.39 with flow), and add height at larger K.
- **CameraDetector's failure does not repeat in this check.** There, the clusters followed camera and light (NMI 0.10
  to 0.29), the very variables that defined the held-out sites. Here the sites of the nested search are bins of the
  controls, and the diffusion contexts carry almost nothing about the controls (NMI 0.007 to 0.063, as low as the plain
  statistics' 0.04).
- **But they are not obviously more than plain statistics.** Their agreement with plain coarse-statistics clusters is
  NMI 0.37 to 0.42, the hand-made contexts explain about as much of either (0.47 to 0.65 against 0.53 to 0.64), and
  the diffusion clusters are only more balanced (more entropy at the same K). Whether that extra resolution helps the
  fine-detail mixer is exactly what the nested search measures.
- **The full test is pending.** It needs stage S2's nested search on effect pixel rows, which is not on main. The API
  is ready for it: `dcm::ddpm::context_planes()` gives, from one coarse state and its condition, an 8 x 8 plane of
  cluster ids per K, and `nvfx_dcm contexts` writes every validation region's diffusion, plain and hand-made contexts
  to `NEURALVFX_DATA/g/diff/fire_contexts.csv`. The rule stays §3's: kept only if they beat the hand-made contexts
  beyond the search's own seed noise. Cost: two passes per frame (t = 400 and 600), about 12 ms on one core; the
  contexts could be refreshed every few frames.

### G2.6 G2b: the prior against drift

`nvfx_experiment g-diff`. One continuous 60 s rollout (the reference implementation, from the start point nearest the
controls, fire's 1 s warm-up first) at validation settings 1 and 2 (0.67 / 0.33 / 0.24 and 0.11 / 0.42 / 0.89), each
10 s window scored against a real 10 s run (`results/experiments/g_diff_prior.csv`).

Tuning (tuning seeds), mean detail score over 12 windows, best first:

| method | score | | method | score |
|---|---:|---|---|---:|
| N = 8, t = 100, beta = 1 | **0.605** | | N = 16, t = 100, beta = 1 | **0.621** |
| N = 4, t = 100, beta = 0.5 | 0.606 | | runtime shards (6 s) | 0.635 |
| N = 8, t = 100, beta = 0.5 | 0.614 | | no prior | 1.699 |
| N = 4, t = 100, beta = 0.25 | 0.614 | | N = 4, t = 20, beta = 1 (worst) | 4.328 |

Low noise levels mostly hurt: every prior at t = 20 and six of the nine at t = 50 drift **more** than no prior (1.9
to 4.3 against 1.70); only the strongest t = 50 prior (every 4 frames, beta 1) comes near the best, at 0.64. Applied
deterministically every few frames, a small bias in the predicted noise at low noise levels is a drift of its own. The
best settings sit at the edge of the grid (the largest t and beta); larger t was not tried.

Decision (fresh seeds, paired over the 12 (setting, window) pairs; cost = one pass / N on one core; the run measured
the pass on the busy machine at 5.8 ms, and a quiet measurement later gave 5.8 to 6.5 ms):

| candidate | minus no prior | minus shards | ms per frame | decision |
|---|---|---|---:|---|
| best: N = 8, t = 100, beta = 1 | −1.655 [−3.015, −0.506] | −0.069 [−0.276, +0.116] (tie) | 0.72 to 0.81 | stop: cost above 0.5 ms |
| best with N = 16: t = 100, beta = 1 | **−1.564 [−2.916, −0.396]** | +0.022 [−0.224, +0.287] (tie) | **0.36 to 0.40** | **keep** |

**Test, once** (study B's held-out settings 1 and 2, new seeds), N = 16, t = 100, beta = 1:

| | detail score | spectrum distance | motion ratio | mean-frame PSNR |
|---|---|---|---|---|
| prior minus no prior | **−5.28 [−9.01, −2.05]** | −0.66 [−1.47, −0.06] | +0.41 [+0.23, +0.59] | +6.4 [+4.5, +8.2] dB |
| prior minus shards | −0.17 [−0.41, +0.05] (tie) | −0.001 (tie) | +0.015 (tie) | +0.9 [−0.4, +2.4] (tie) |
| no prior minus shards | +5.10 [+1.86, +8.90] | | | |

Without the prior, the test's first setting froze from 20 s to 50 s (motion ratio 0.006 to 0.03, spectrum distance up
to 3.9: REPORT §6.6's failure), and the second slowed to a motion ratio of 0.06 to 0.24 in its last 30 s. With the
prior, both kept a motion ratio of 0.64 to 0.95 for the whole minute (mean-frame PSNR 32 to 41 dB). Cost: one pass
every 16 frames; one pass takes 5.8 to 6.5 ms (median, one pinned core, quiet machine, four cores tried), so **0.36 to
0.40 ms per frame**. Its weights add 1.5 MB as float32 (0.8 MB as fp16) to an 82 KB effect.

What it means: the prior does what shards do (the test ties them on every statistic) without restarts, crossfades or
stored states, so a single rollout can play for a minute. It does not beat shards. Shards are free per frame and need
no 1.5 MB network, so the prior is worth having where a crossfade every 6 s is unwanted (one continuous run, an effect
that must not jump), not as a replacement.

### G2.7 G2c: start points

First 2 s after each start (after fire's 1 s warm-up), at the 10 validation settings, 2 seeds each
(`results/experiments/g_diff_starts.csv`):

| method | tuning: detail score | spectrum distance | motion ratio | mean-frame PSNR |
|---|---:|---:|---:|---:|
| nearest stored start as it is | 1.493 | 0.721 | 0.84 | 31.97 |
| nearest stored start rolled ahead 0.5 s (the alternative) | 1.545 | 0.724 | 0.82 | 31.96 |
| SDEdit of it from t0 = 300 (320) | 1.525 | 0.730 | 0.82 | 31.95 |
| SDEdit from t0 = 400 | 1.528 | 0.731 | 0.82 | 31.96 |
| SDEdit from t0 = 500 (520) | 1.530 | 0.732 | 0.82 | 31.96 |
| fresh DDIM sample (25 passes) | 1.878 | 0.747 | 0.74 | 31.56 |

Decision (fresh seeds, paired over the 10 settings): the best generated start, SDEdit from t0 = 300 (also the best
SDEdit), minus the rolled start: **−0.079 [−0.216, +0.038], a tie**; 8 passes, 46 ms per shard. **Stop.**

Diversity, mean pairwise RMS distance between 8 starts at a setting (network units, validation settings 1 to 3): real
states (8 seeds, frame 150) 0.69 to 1.00; fresh samples 0.45 to 0.51; the stored start rolled 0.5 s with 8 seeds 0.36
to 0.52; SDEdit of the stored start 0.16 to 0.20 (`results/experiments/g_diff_diversity.csv`).

- Fresh samples are worse than every stored start (colder, smoother, too little motion) and half as varied as real
  states: the small diffusion model's usual failure, and the conditioning on the controls is weak.
- SDEdit at t0 = 300 to 500 hardly moves a state (its 8 seeds differ by 0.16 to 0.20, a fifth of real states' spread):
  at those levels the signal still dominates the noise, and the network restores it. It cannot move a stored start to
  new controls, so it ties the stored start it came from.
- Fire forgets its start within about a second (REPORT §6.1), and its 1 s warm-up rolls every start forward anyway, so
  the start matters little here. Smoke and explosions keep their start's look longer; this result does not transfer
  to them without a test.

### G2.8 Costs

| item | cost |
|---|---|
| states (176 runs simulated once) | 433 s on two threads |
| training | 3,830 s on two threads (about 2.1 CPU-hours) |
| G2a contexts (fit and score) | about 2 minutes on two threads |
| G2b and G2c on validation (`g-diff`) | 21 minutes on two threads |
| G2b test (`g-diff-test`) | 1.4 minutes on two threads |
| all of stage S5 on fire, including probes and smoke tests | about 3.4 CPU-hours (budget 5) |
| one denoiser pass (`nvfx_dcm ddpm-time`) | median 5.8 to 6.5 ms, p90 6.6 to 8.1 ms on one pinned core of a quiet machine (load 0.9 to 1.0), cores 0 to 3; 14 to 16 GMAC/s. The decisions' own measurements on the busy machine (5.8 and 6.4 ms) agree: the kernels are compute-bound and the pinned core was free |
| denoiser size | 391,748 weights: 1.5 MB float32 |

### G2.9 Decision

The rule for each use was fixed in G2.3 before any result.

| use | test | result | met? |
|---|---|---|---|
| G2a contexts | check 1: redundant if the hand-made contexts explain at least 80% of their entropy at every K | 47% to 65%; they follow heat and flow, not the controls (unlike CameraDetector's camera and light) | **not redundant** |
| G2a contexts | they beat the hand-made contexts in the nested search beyond its seed noise | not run: stage S2's search is not on main | **pending** |
| G2b prior | the drift (detail score per 10 s window of a 60 s rollout) falls, interval below zero, on validation | −1.56 [−2.92, −0.40] (N = 16, t = 100, beta = 1) | **yes** |
| G2b prior | within 0.5 ms per frame | 0.36 to 0.40 ms (median pass, quiet machine) | **yes** |
| G2b prior | the same on test, once | −5.28 [−9.01, −2.05] | **yes** |
| G2b prior | (reference) better than the runtime's shards | tie on every statistic, validation and test | no |
| G2c start points | better than the nearest stored start rolled ahead, interval below zero | SDEdit t0 = 300: −0.079 [−0.216, +0.038], a tie; fresh samples worse in tuning (1.88 against 1.55) | **no** |
| G2c start points | within 100 ms per shard | SDEdit 8 passes, 46 to 52 ms, yes; a fresh sample 25 passes, 144 to 161 ms, no | partly |

**Decision for fire: keep the denoiser as a prior against drift (G2b); stop diffusion start points (G2c); G2a waits
for the nested search.** This is not CameraDetector's outcome: there no diffusion use survived, here one does, and the
contexts do not encode the held-out sites. It is also a narrower win than it looks: the prior ties the shards that the
runtime already has, so it buys continuity (no restarts, no crossfades), not better pictures, for 0.4 ms per frame and
1.5 MB. Extending it to smoke and explosions is a later decision and has not been started.
