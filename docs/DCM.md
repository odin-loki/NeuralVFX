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

Status: **rules fixed before any result** (9 October 2026); results follow in G2.4 to G2.7 as each part finishes.

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
- **Two candidates per use.** One pass of this network takes at least 5 ms on one core of this machine, so N = 4 and
  N = 8 cannot meet 0.5 ms per frame, and a fresh 25-step sample cannot meet 100 ms per shard. Choosing only the best
  score could therefore stop a use for its cost while an affordable setting works. So two candidates go from tuning to
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
