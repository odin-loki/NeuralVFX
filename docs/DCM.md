# Study G: diffusion-context mixing for generating effects

Status: **plan** (9 October 2026), with G1 and G1c decided (stage S2, §6). Each section turns into a result when its
stage finishes. Audience: owner, research, dev.

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

## 6. G1 and G1c: DCM-fine (stage S2)

**Result in one line:** the mixer about halves the detail spectrum distance on all three effects at held-out settings,
and a cold start from the coarse state alone ties or beats v1's start (**G1c passes**), but the released smoke generator
moves too little (motion ratio 0.72 against v1's 0.85) and its average picture is 0.05 dB further away, so by the rule
fixed in advance **G1 does not pass** to stage S6. Fire passes every criterion on its own.

Code: `include/neuralfx/dcm/fine.hpp` and `src/dcm/fine.cpp` (the reference), `tools/fine_study.cpp` (the pipeline
behind `nvfx_dcm record | experts | search-fine | train-fine | eval-fine | bench-experts | fine-summary` and
`nvfx_experiment g-fine`), `tests/test_dcm_fine.cpp`. Tables: `results/experiments/g_fine_*.csv`, collected with their
intervals in `results/experiments/g_fine_summary.md`.

### 6.1 What was built

Each frame, the v1 stepper and the MacCormack advection of the detail layer run unchanged. Then, for every fine pixel
and channel (heat, soot), a ValueNet mixes up to 15 experts, all planar and made from buffers the reference already has:

| group | experts |
|---|---|
| adv | A (the MacCormack value), A_sl (the semi-Lagrangian value), A - A_sl |
| lock | L (v1's lock output), r_up A, a_up (new material) |
| noise | a_up phi_1, a_up phi_2, a_up phi_3 (new material shaped by each flicker octave) |
| coarse | C_up (the bilinear coarse value), (C - block mean of A)_up |
| shape | laplacian of A, \|grad A\| A / s_q |
| prev | the previous frame's value |
| bias | 1 |

- **Domain:** inputs and target are divided by C_up + 0.02 s_q (linear) or mapped by sign(v) log1p(\|v\| / s_q) (log);
  s_q is the renderer's input scale of the channel.
- **Contexts:** heat level (C_up in 6 bins, 0 = empty), A / C_up (4 bins), flow (speed in 3 bins x vorticity sign),
  height (4 bands), controls (intensity x turbulence, 3 x 3), age (one-shot effects, 4 bins), channel, and k-means
  clusters (K = 4 and K = 8) of regional coarse statistics (mean heat, soot, speed and \|vorticity\| over 8 x 8 coarse
  cells), the floor without diffusion. Bin edges are quantiles of the training rows. A tenth context, `extra`, is a
  per-cell hook (a callback) for stage S5's denoiser clusters.
- **Network:** a first-layer mixer without context plus up to 3 context-selected ones, the final mixer, the AVM, and
  the scale net log b = U[ctx] (1, z) with z = (\|A - A_sl\|, \|grad A\|, C_up, a_up), normalised.
- **It starts as v1.** With weight 1 on L, 0 elsewhere and the AVM off, the step is v1's detail step
  (`DcmFine.DcmFineWithTheV1WeightsIsV1`: within 1e-6 for one frame and 5e-6 after five whole frames, in both domains
  and with context-selected mixers; the lock output L of `compute_frame` is bit-identical to `rollout::detail_step`). Training
  starts there too: the search gained `SearchProblem::value_rule` (a pure addition) so that a configuration that
  includes L starts as the v1 rule instead of averaging its inputs.
- **Invisible pixels keep v1's value:** where every value a pixel is built from is below 2e-4 s_q (the renderer's gate
  is then at most 2%), the mixer is not run. It runs on 34% of the pixel-channels of a fire frame, 40% for smoke, 27%
  for explosions.
- **Generation:** v = clamp(mu + tau b xi, 0, 2 max(L, A, A_sl, prev, C_up)) in value units, with xi a two-octave
  value-noise fbm of its own seed stream mapped to Laplace quantiles. The upper bound is a local maximum principle with
  slack (it only binds on a runaway sample). Then, optionally, a relock: (1) exact, each block scaled to its coarse
  value; or (2) v1's lock applied to the mixer's output (growth limited by `grow`, the rest arriving as new material
  shaped by the flicker).
- **Codec use:** bits per pixel = -log2 of the Laplace mass of the bin [v - s_q / 512, v + s_q / 512] in value units.
  The search's objective codes every row in one bin width (the median of the rows' normalised widths); every bits
  figure below uses each pixel's own bin.
- **Determinism:** a released mixer is its serialisation with a SHA-256 version (in `g_fine_released.csv`); 100 frames
  of a mixer with grain and each relock give one hash (`DcmFine.HundredStepsGiveOneHash`).

### 6.2 Protocol as run

- **Rows** (`record`): 48 training runs (salt 1) and 16 validation runs (salt 3) per effect at 128 px, simulated on the
  fly. Every frame the v1 stepper takes the true coarse state (its memory channels carried from its own previous
  step); the experts come from the true fine fields of the frame before, and the target is the true fine field. Fire
  and smoke: every second frame from 60 to 240; explosions: frames 1 to 89. Per frame and channel 32 active and 4 empty
  pixels: 311,040 training rows per effect, about 280,000 once the invisible ones are dropped. The training runs also
  give 192 own-rollout windows of 8 frames (true coarse states, true fine targets). Data under
  `/root/nvfx-data/g/fine`, outside git; recording takes about 3 minutes per effect on 2 threads.
- **Validation** for every choice: the salt-3 rows (bits) and 10 validation settings (`std::mt19937_64 rng(2027)`,
  off study B's training grid as B's are, and at least 0.05 from every B test setting); real runs with seeds 800000 + i
  (10 s after 5 s of warm-up; explosions 3 s), single 6 s shards from the nearest start point with seeds 820000 + i
  (explosions 3 s), scored by `calibrate_detail`'s score (spectrum distance + \|log motion ratio\| + log ratios of
  light and cover).
- **Test**, once at the end: study B's 10 held-out settings, real runs with seeds 900000 + i (and 910000 + i, the
  floor), single 6 s shards (explosions 3 s) with seeds 920000 + i, and the 8 salt-2 tracking runs of `d-eval`. v1 and
  DCM-fine both run through the reference with the same stepper and seeds; only the detail layer differs.
- **Search:** `run_search` with `Objective::laplace_bits`, nested and global protocols over 5 sites (k-means, K = 5, of
  the training runs' controls), 200 configurations and 2 refinement rounds, on 100,000 of the training rows (every
  third). Chosen configurations are then trained on every row.
- **Cost** (`bench-experts`): each part of the pass in plain allocation-free float code (scalar, baseline ISA: an upper
  bound for the runtime's SIMD code), timed in thread CPU time (the least of 15 repetitions) on frames of a v1 rollout,
  in ms per 128 x 128 frame. Every configuration pays a base (the mixer's per-pixel overhead, the grain and a relock:
  0.61 ms for explosions to 0.84 ms for smoke); each expert group its experts and its inputs in the mixer without
  context (0.01 to 0.16 ms); each context its own computation and one more mixer over every input (0.08 to 0.21 ms;
  conservative, since a context used only by the AVM or the scale net needs no mixer). Budget: +1 ms.

### 6.3 Pilot: the mixer predicts the next fine field better than v1's lock

The default mixer (every expert, mixers by heat level, A / C ratio and flow, AVM by heat level) against the v1 lock used
as a predictor with a Laplace scale fitted per channel and heat-level bin. Held-out bits per active pixel on the 16
validation runs, paired over runs:

| effect | v1 lock | default mixer, linear domain | default mixer, log domain | linear - v1 | log - linear |
|---|---:|---:|---:|---|---|
| fire | 3.417 | 2.678 | 3.235 | -0.739 [-0.784, -0.692] | +0.556 [+0.528, +0.584] |
| smoke | 4.149 | 3.403 | 3.828 | -0.746 [-0.825, -0.664] | +0.425 [+0.381, +0.466] |
| explosion | 3.808 | 3.082 | 3.632 | -0.726 [-0.802, -0.649] | +0.550 [+0.496, +0.608] |

The fire pilot was the gate (an interval excluding zero): passed, so the study went on; smoke and explosions were run
afterwards for the record. The linear domain wins by about half a bit everywhere and was used from then on. Squared
error ties v1's on fire and explosions and is lower on smoke: the mixer trades a little squared error for a much better
idea of where the value is (bits). Alone, the advected value A codes fire in 3.01 bits, v1's lock in 3.42 (it has the
lower squared error, but its flicker-shaped new material spreads the errors).

### 6.4 Search: under the budget the mixer is small

| effect | seed | nested held-out bits | released configuration (global #0) | its validation bits | cost (model) |
|---|---:|---:|---|---:|---:|
| fire | 0 | 2.626 | A, A_sl, A - A_sl, bias; no context mixer; AVM by channel; scale by height | 2.645 | 0.99 ms |
| fire | 1 | 2.638 | A, A_sl, A - A_sl, prev; scale by height | 2.642 | 0.92 ms |
| fire | 2 | 2.627 | as seed 0 (other rates) | 2.648 | 0.99 ms |
| smoke | 0 | 3.393 | A, A_sl, A - A_sl, L, r_up A, a_up, prev, bias; AVM | 3.395 | 0.95 ms |
| explosion | 0 | 2.725 | A, A_sl, A - A_sl, a_up phi_1..3; mixer by channel; scale by A / C ratio | 2.909 | 0.95 ms |

- **The budget decides the shape.** Every configuration pays 0.6 to 0.8 ms of base cost (the scalar per-pixel
  overhead of the mixer, the grain and a relock), so only a few inputs and at most one or two contexts fit in +1 ms.
  Nothing is lost in bits by it: the small fire mixer codes the validation runs in 2.645 bits against the default
  mixer's 2.678 (15 inputs, 3 context mixers).
- **The search prefers advection to the lock.** One step from the truth, a blend of the MacCormack and semi-Lagrangian
  values predicts best; v1's lock and its flicker-shaped new material add spread. The released fire mixer is
  0.917 A + 0.056 A_sl (and a bias of almost 0): it carries 3% less of the advected detail than v1 and leaves the rest to
  the lock.
- **Seed noise of the search** (fire, seeds 0, 1, 2; nested bits paired over the 48 training runs): 1 - 0
  +0.012 [+0.009, +0.016]; 2 - 0 +0.000 (tie); 2 - 1 -0.012 [-0.014, -0.009]. So about 0.012 bits.
- **Context families** (fire, within the budget, nested bits paired over runs): hand-made - none
  -0.051 [-0.069, -0.033]; hand-made + regional clusters - hand-made -0.016 [-0.019, -0.012]; hand-made + clusters -
  none -0.067 [-0.086, -0.049]. Contexts help well beyond the seed noise, mostly through the cheap AVM and scale-net
  contexts; the regional clusters add 0.016 bits, about the size of the seed noise (0.012).
- **Without the budget** (fire, 60 configurations and one refinement round per family, so a smaller search):
  hand-made - none -0.057 [-0.070, -0.045], but hand-made + clusters - hand-made **+0.029 [+0.019, +0.037]**. Given
  room, the clusters make the held-out score worse, as the diffusion clusters did in CameraDetector: more choices, more
  selection noise. Their normalised mutual information with the hand-made contexts is 0.15 to 0.18 with heat level and
  flow and at most 0.12 with the rest: they partly repeat what heat level and flow already say, the risk the plan named
  (§3). For stage S5 (G2a), this is the floor a denoiser's clusters have to beat.

### 6.5 Generation on validation settings

Each of the global top 10 was trained on every training row, then given the own-rollout pass: its own output fed back
for up to 8 frames from each window's true fine fields (true coarse states), and one more pass of online learning over
those rows mixed with as many one-step rows, at the rates training ended with. (A first version restarted the learning
rates and let the Laplace gradient, sign / b with the small one-step scales, throw the weights far: validation bits went
from 2.7 to between 4 and 16. Continuing from the annealed rates fixed it.) The pass costs 0.03 to 0.08 bits one step
from the truth and changes generation little (the first candidate at the chosen setting: fire 0.772 to 0.766, smoke
0.583 to 0.594, explosion 0.676 to 0.670; scores below). The released mixers are the pass-2 versions.

Tau (0, 0.5, 1) and relock (off, exact, v1's lock) were chosen on the first candidate, then the top 10 re-ranked at that
setting. `calibrate_detail`'s score, mean over the 10 validation settings (lower is better), paired over settings:

| effect | v1 | relock off, tau 0 | exact relock, tau 0 | v1's lock, tau 0 | v1's lock, tau 0.5 | released (re-ranked) | released - v1 |
|---|---:|---:|---:|---:|---:|---:|---|
| fire | 0.815 | 15.24 | 1.276 | 0.766 | 1.101 | 0.720 (candidate 7, v1's lock) | -0.094 [-0.128, -0.059] |
| smoke | 0.581 | 1.030 | 0.597 | 0.594 | 0.774 | 0.567 (candidate 2, v1's lock) | -0.014 [-0.092, +0.049] (tie) |
| explosion | 0.710 | 0.670 | 0.715 | 0.752 | 0.727 | 0.666 (candidate 6, no relock) | -0.044 [-0.135, +0.043] (tie) |

- **The grain never helps.** Every effect chose tau = 0: drawing from the one-step Laplace adds noise of the one-step
  error's size every frame, which piles up as flicker. The released generators are deterministic.
- **The coarse coupling has to come from somewhere.** Without a relock, the lock-free fire mixer drifts away from the
  coarse state (score 15). The exact relock over-brightens peaks (it scales existing structure without bound); v1's own
  lock after the mixer works best for the looping effects. A 3-second explosion has no time to drift.
- **What the released generators are:** fire, 0.917 A + 0.056 A_sl (AVM and scale by height), then v1's lock; smoke,
  0.91 A + 0.02 A_sl + 0.07 prev (AVM), then v1's lock; explosions, a blend of A and A_sl with new material shaped by
  the three flicker octaves, mixers by A / C ratio and channel, no lock. Versions (SHA-256) in
  `g_fine_released.csv`: fire 69faa016..., smoke 4d04479d..., explosion 05ef4ae3....
- A first calibration with only the exact relock gave fire 1.01 against v1's 0.81; the v1-lock relock was then added
  as a third option on validation. Both runs are in the logs; only the second is in `g_fine_val.csv`.

### 6.6 The test (run once)

Study B's 10 held-out settings with new seeds: a single 6 s shard from the nearest start point (explosions: their 3 s)
against a real 10 s run (explosions: 3 s). v1 and DCM-fine through the reference, same stepper and seeds. Means over
settings:

| effect | method | spectrum distance | motion ratio | coverage L1 | mean-frame PSNR |
|---|---|---:|---:|---:|---:|
| fire | real, other seed (floor) | 0.080 | 1.06 | 0.0035 | 36.39 |
| fire | v1 | 0.365 | 0.96 | 0.0044 | 33.59 |
| fire | **DCM-fine** | **0.191** | 0.90 | 0.0044 | **33.72** |
| smoke | real, other seed (floor) | 0.076 | 1.04 | 0.0208 | 29.79 |
| smoke | v1 | 0.254 | 0.85 | 0.0239 | 28.31 |
| smoke | **DCM-fine** | **0.125** | 0.72 | 0.0239 | 28.27 |
| explosion | real, other seed (floor) | 0.116 | 1.07 | 0.0195 | 26.76 |
| explosion | v1 | 0.199 | 0.92 | 0.0305 | 26.16 |
| explosion | **DCM-fine** | **0.102** | 1.05 | 0.0306 | 25.86 |

DCM-fine - v1, paired over the 10 settings:

| effect | spectrum distance | \|log motion ratio\| | coverage L1 | mean-frame PSNR | calibration score |
|---|---|---|---|---|---|
| fire | **-0.174 [-0.186, -0.161]** | +0.009 (tie) | 0.000 (tie) | **+0.13 [+0.07, +0.20]** | **-0.169 [-0.221, -0.119]** |
| smoke | **-0.129 [-0.176, -0.068]** | +0.160 [+0.135, +0.180] (worse) | 0.000 (tie) | -0.05 [-0.07, -0.02] (worse) | +0.032 (tie) |
| explosion | **-0.097 [-0.151, -0.047]** | -0.033 (tie) | 0.000 (tie) | -0.30 (tie) | **-0.121 [-0.234, -0.025]** |

Tracking the 8 held-out runs from their true state (active PSNR, DCM-fine - v1, paired over runs):

| effect | 1 frame | 8 frames | 30 frames | 60 frames |
|---|---|---|---|---|
| fire | +0.10 [+0.07, +0.14] | +0.35 [+0.27, +0.45] | +0.27 [+0.14, +0.42] | +0.47 [+0.28, +0.67] |
| smoke | +0.31 [+0.20, +0.42] | +0.57 [+0.44, +0.69] | +0.53 [+0.43, +0.64] | +0.28 [+0.20, +0.36] |
| explosion | -0.45 [-0.71, -0.20] | -1.77 [-2.89, -0.37] | -1.95 [-2.40, -1.50] | -1.38 [-1.82, -1.00] |

- **Detail spectrum: about halved on all three effects**, every interval below zero. DCM-fine closes 61% (fire), 72%
  (smoke) and more than all (explosions: 0.102 against the floor's 0.116) of v1's gap to two real runs.
- **Fire improves on every measure** and follows a real run 0.1 to 0.5 dB better.
- **Smoke loses motion:** its motion ratio drops from 0.85 to 0.72 (real: 1.04), and its average picture is 0.05 dB
  further away. The released smoke mixer is 0.91 A + 0.02 A_sl + 0.07 prev: it keeps 7% of the previous frame's
  value at the pixel, unadvected, a temporal smoothing that slows change (one step from the truth it lowers the code
  length; over many frames it lowers motion).
- **Explosions** tie on motion and the average picture, but follow a real run 0.5 to 2 dB worse. The released
  explosion mixer (about two thirds MacCormack, one third semi-Lagrangian, plus new material shaped by each flicker
  octave with weights near 0.15) has no lock, so nothing ties its fine fields to the coarse state beyond what the
  experts carry; the burst's shape drifts from the true run's.

### 6.7 G1c: the first second

The first 30 frames against the real run (looping effects: its 10 s statistics; explosions: its own first second).
`usual` is v1's start (fire: a 30-frame warm-up; smoke and explosions: stored 64-pixel fine fields); `cold` starts from
the coarse state alone, with no warm-up and no stored fine fields. Calibration score, mean over the 10 test settings:

| effect | v1 usual | v1 cold | DCM-fine usual | DCM-fine cold | DCM-fine cold - v1 usual (score) | (spectrum) |
|---|---:|---:|---:|---:|---|---|
| fire | 1.693 | 2.185 | 1.552 | 2.018 | +0.33 [-0.16, +0.91] (tie) | -0.18 [-0.34, -0.06] |
| smoke | 1.671 | 1.671 | 1.536 | 1.542 | **-0.13 [-0.20, -0.07]** | **-0.19 [-0.21, -0.17]** |
| explosion | 0.510 | 0.520 | 0.484 | 0.496 | -0.01 (tie) | -0.02 (tie) |

**G1c passes:** a cold DCM-fine start ties v1's current start on fire and explosions and beats it on smoke, so fire
could skip its 30-frame (50 ms) warm-up and smoke and explosions their stored fine fields (64 to 128 KB of each
effect's 146 to 274 KB). The cold start was not trained for: the same released mixers grow the detail within the second.

### 6.8 Cost

Measured by `bench-experts` while the machine was busy (load 2 to 9 from other jobs throughout the session, never below
1.5): thread CPU time, the least of 15 repetitions, so other jobs only add time. These are **provisional upper
bounds**; a quiet-machine measurement was not possible in this session.

| effect | cost model (as searched) | itemised for the released generator |
|---|---:|---:|
| fire | 0.91 ms | about 0.43 ms: tau = 0 needs no grain, and v1's lock moves after the mixer instead of being added |
| smoke | 0.92 ms | about 0.52 ms (as fire) |
| explosion | 0.95 ms | about 0.62 ms: no grain, and no lock at all (v1's lock is saved) |

The released mixers are 2 to 4 KB of text each (spec and weights; a few hundred values).

### 6.9 Decision

By the rule fixed in advance (§3): the spectrum distance beats v1 with an interval excluding zero on 3 of 3 effects (2
needed) and the cost is within +1 ms, but **smoke is worse on two measures**: its motion ratio is further from 1
(\|log ratio\| +0.160 [+0.135, +0.180]) and its mean-frame PSNR is lower (-0.05 dB [-0.07, -0.02]). **G1 does not pass
to stage S6 as released.** Bits per pixel alone would have passed it everywhere (-0.73 to -0.75 bits in the pilot); they
do not count.

**G1c passes** (§6.7): ties on fire and explosions, better on smoke.

What this suggests for S6, without touching the test again: fire's generator passes every criterion on its own and
smoke's fails only on motion, so a per-effect choice (DCM-fine for fire, v1 for smoke; the explosion's tracking loss
deserves a lock) or a smoke mixer selected with motion weighted higher on validation are the obvious next candidates.
Each needs a new validation selection and a new test with new seeds; this stage does not claim them.
