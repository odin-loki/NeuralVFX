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
| H1 | **computing on compressed data:** feature volumes, weight tables and stored fields kept LZ78- or grammar-compressed (RePair), with products and blends computed on the compressed form (a phrase's partial sum is built from its parent's), so the cost follows the compressed size | S9 | it is faster than the dense SIMD code it replaces at the same result (bit-exact or within the parity tests), on a quiet machine |
| H2 | **run-aware fields:** the mostly empty fine fields kept as runs, so the detail step, the shader and compositing touch only material | S9 | as H1; the fireball's frame time falls |
| H3 | **LZ inside the coder:** a match model and LZ tokens in the context-mixing coder, for faster decoding at a small cost in size, and decoding of single slices without the whole file | S9 | load or decode time falls by more than the size grows |

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
| S9 | study H, the owner's addition: compute on LZ- and grammar-compressed data (H1-H3), measured against the dense code it would replace | an agent, after S1 | H1-H3 decided |

Study H is honest about where it can win. Small dense products in AVX2 registers are hard to beat, and published
speed-ups from grammar-compressed products are mostly over sparse formats on large, repetitive matrices. The likely
wins here are the large blended feature volumes of the frame models, the mostly empty fine fields (H2) and decoding
(H3).

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

## 7. G2: diffusion for the macro features (stage S5)

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
