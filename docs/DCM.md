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
