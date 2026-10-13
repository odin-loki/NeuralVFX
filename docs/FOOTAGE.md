# From footage to rollout effects: start points estimated from frames (study J)

Status: **study done; the tool is validated on simulated renders only** (11 October 2026). Audience: artists with
footage, dev, research. Code: `include/neuralfx/footage.hpp`, `src/footage/`, the tool `nvfx_estimate` and the study
`nvfx_study_j`. Results: `results/experiments/j_*.csv` and the generated tables in
[results/experiments/j_summary.md](../results/experiments/j_summary.md).

## 1. The question

A rollout effect ([REPORT.md](REPORT.md) §6) plays from stored start points: states of the simulation (a 32 x 32 grid of
velocity, heat and soot, plus fine heat and soot) from which its learned stepper moves on. Until now every start point
came from the simulation. Footage has no states, only pictures. So the question is:

> Given a few consecutive frames of an effect, can its state be estimated well enough that the effect continues what the
> frames show, better than the stored start point the runtime would use today?

No owner footage has been supplied, so the path is measured where the truth is known. The stand-in footage is the
simulator's own frames, whose states are known, and the frames the effect's learned renderer draws from those states.
Camera and codec damage is added as a separate axis: blur, noise, H.264, and footage without an alpha channel.

## 2. The path

Everything runs on the CPU, in C++, with networks trained here from random initialisation.

1. **Fine heat and soot from each frame: an inverse renderer.** A per-pixel MLP (two hidden layers of 32) sees a
   pyramid of the frame: the 3 x 3 neighbourhood of the pixel at 1, 2, 4, 8 and 16 pixels per cell (about 24 pixels of
   context), and the pixel's position. It outputs heat and soot. It is trained on the simulator's (frame, fields) pairs
   from the effect's own training runs (64 runs, 6 frames each), half of the frames with random blur (up to 1.2 px) and
   noise (up to 3/255). There are two per effect: one for footage with alpha, one that sees colour only. 6,000 to 7,500
   weights; about 3 minutes to train on one core.
2. **Refinement through the effect's own renderer** (optional, chosen on validation). Gradient descent (Adam, 100
   steps) on the start frame's fine fields, so that the effect's learned renderer draws the footage frame, with a
   small pull back to the inverse network's fields. The renderer's gradient is written by hand and checked against
   finite differences (`tests/test_footage.cpp`).
3. **Coarse heat and soot** are block averages of the fine fields, as the simulation's states are averaged.
4. **Velocity.** Pictures show where material is, not how fast the air moves, and the velocity decides where the
   material goes next. Three ways were built:
   - **Block matching** between the estimated fields of consecutive frames: for each coarse cell, the displacement
     (within 6 pixels, sub-pixel by a parabola) that best maps a 13 x 13 window of one frame onto the next. Cells
     without material are filled from their neighbours.
   - **A motion network**: a per-cell MLP (two hidden layers of 48) on the coarse grid, from the coarse heat and soot of
     the frame and the two before it (5 x 5 cells), the block-matched flow (3 x 3 cells) and the position. It is
     trained on the same training runs, through the inverse network, against the true coarse velocity. It learns what
     block matching cannot see: the air moving where there is no material, a plume's speed from its heat.
   - **Assimilation through the effect's stepper**: the stepper runs over the context frames; after each step its heat
     and soot are replaced by the estimates and its velocity is relaxed towards the measured one. The velocity it ends
     with is one that its own dynamics agree with.
5. **The start point**: the coarse state, the fine fields (kept at 64 x 64, 8 bits, as smoke's and the explosion's
   stored start points), the controls that match the footage and its time. `nvfx_estimate` writes a copy of the
   effect with these start points.

## 3. Protocol

**Effects.** The v2 rollout effects ([DCM.md](DCM.md) §11.2), read only: fire with 6-bit start states, smoke, the
explosion with the coupled stepper (SHA-256 in `results/experiments/v2_frozen.csv`). Estimated start points go through
the same storage as the effects' own (6-bit coarse states for fire, 16-bit for the others, fine fields at 8 bits).

**Splits.** Nothing is scored on what was trained on.

| what | runs |
|---|---|
| inverse and motion networks: training | the effects' own training runs (salt 1): 64 runs, 6 frames (or frame triples) each |
| validation (every choice) | 12 runs per effect with salt 3, seen by no model; endless statistics at 6 settings of salt-3 runs |
| test (once) | study D's tracking split (salt 2): 16 runs per effect (D's 8 and the next 8); endless statistics at study B's 10 held-out settings with study D's seeds |

**Where the footage is cut.** Fire and smoke: the start frame is frame 100 of a run (as in study D). Explosions: frame 8
(0.27 s after the detonation; the context needs frames before it). The estimator sees the 8 frames up to and
including the start frame and nothing after it.

**Footage conditions** (a separate axis; the primary one is the first):

| condition | what the estimator sees |
|---|---|
| `clean_rgba` | the simulator's frames, premultiplied RGBA, 8 bits |
| `clean_rgb` | colour only, alpha made as `nvfx_ingest --alpha luma` makes it (footage shot on black) |
| `blur_noise_rgb` | colour only, Gaussian blur of 1 px and noise of 2/255 |
| `h264_rgb` | colour only through H.264 (libx264, 4:2:0, CRF 23) and back through the ingest path |
| `learned_rgba` | the effect's own learned renderer drawing the true states (a second renderer) |

**Methods.** Each start point is played through the shipping runtime as one continuous rollout, with the run's controls
and noise seed (the only way a rollout can follow a given run), and drawn at the same storage:
- `true`: the true state (coarse and fine): the upper bound;
- `estimated`: the estimator;
- `stored_nearest`: today: the effect's stored start point nearest the run's controls, as the runtime picks it, timed
  so its noise lines up with the run (fire's 1 s warm-up included); for explosions it plays from its own age, and the
  ages are aligned.

The estimator itself is not given the run's seed (footage has none); it is given the controls (an artist sets them).

**Measures.**
1. State error at the start frame: coarse heat, soot and velocity RMSE (in the state's units: velocity in coarse cells
   per frame), and fine heat and soot PSNR (peak at the renderer's input scale, all pixels).
2. Tracking: active PSNR against the real run's clean frames at 0 (the start frame redrawn), 1, 8, 30 and 60 frames.
3. Endless statistics, as study D: an effect whose start points are estimated from 10 s of footage at the setting (3
   start points at 5, 6 and 7 s; explosions: one, at frame 8), played in shards with a new seed for 10 s (explosions:
   ages 9 to 89 frames, the window every clip covers), against a real run at the setting: detail spectrum distance,
   |log motion ratio|, coverage and emission distances, mean-frame PSNR; the effect with its stored start points is
   played with the same seed.

Differences are paired over runs (tracking) or settings (endless), with 95% bootstrap intervals (10,000 resamples). An
interval covering zero is a tie.


## 4. Validation: what was chosen

Every choice was made on the validation runs (salt 3, 12 per effect), mostly on `clean_rgba`, by the mean of tracking
at 8 and 30 frames, and checked on the other conditions. All rows of `results/experiments/j_val.csv`; the tables in
[j_summary.md](../results/experiments/j_summary.md).

Active PSNR (dB) at 1 / 8 / 30 / 60 frames after the start, `clean_rgba`, mean over 12 runs, and the velocity error
at the start (RMSE in coarse cells per frame; the true velocity's RMS is about 0.17):

| method | fire | smoke | explosion |
|---|---|---|---|
| true start (upper bound) | 25.45 / 19.96 / 17.84 / 16.53 | 25.84 / 20.93 / 17.24 / 15.36 | 24.04 / 24.86 / 19.37 / 17.94 |
| estimated fields, true velocity | 25.40 / 19.93 / 17.80 / 16.52 | 26.31 / 20.97 / 17.23 / 15.38 | 25.51 / 23.19 / 18.34 / 17.40 |
| motion network + assimilation (chosen: fire, smoke) | 25.09 / 18.57 / 17.40 / 15.81; 0.056 | 26.07 / 18.60 / 15.78 / 15.09; 0.056 | 25.08 / 19.15 / 15.50 / 16.11; 0.076 |
| motion network alone (3 frames) | 24.95 / 18.53 / 17.34 / 15.65; 0.067 | 26.01 / 18.52 / 15.85 / 15.17; 0.069 | 24.90 / 19.33 / 15.41 / 16.03; 0.112 |
| block matching + assimilation (chosen: explosion) | 24.55 / 16.73 / 15.97 / 15.65; 0.086 | 26.16 / 18.45 / 14.92 / 13.91; 0.092 | 25.18 / 21.41 / 17.06 / 17.06; 0.059 |
| block matching alone (2 frames) | 24.26 / 16.53 / 15.65 / 14.03; 0.103 | 26.06 / 18.38 / 14.85 / 14.28; 0.105 | 24.92 / 18.70 / 15.36 / 16.16; 0.113 |
| estimated fields, zero velocity | 20.86 / 13.41 / 14.66 / 15.10 | 23.13 / 13.68 / 12.36 / 13.10 | 19.66 / 13.19 / 10.32 / 12.32 |
| nearest stored start (today) | 15.47 / 15.53 / 16.20 / 15.02 | 12.13 / 11.93 / 13.04 / 14.06 | 14.67 / 14.76 / 13.68 / 15.19 |

What validation showed:
- **The fields are not the problem; the velocity is.** With the true velocity, the estimated fields track as well as
  the true state on fire and smoke (within 0.1 dB at 8 and 30 frames). With zero velocity they lose 6 to 12 dB at 8
  frames. The inverse networks reach 46 to 50 dB of fine-field PSNR on held-out frames with alpha, 30 to 44 dB
  without (`j_inverse.csv`).
- **The motion network measures velocity 27 to 40% better than block matching** on held-out runs
  (`j_motion.csv`). With assimilation, it gains on fire 1.8 dB at 8 frames and 1.4 at 30, on smoke 0.15 and 0.9.
  Assimilation through the stepper adds little on top of the motion network alone (0.0 to 0.05 dB, relaxing halfway
  to the measurement at each step).
- **The explosion is the exception.** Its velocity 0.27 s after the detonation is mostly the expansion, which its
  stepper produces from the age and the heat: assimilating 8 frames from block matching gets the velocity to 0.059
  (0.037 without relaxing), better than the motion network's 0.076, and tracks 2.3 dB better at 8 frames. So the motion
  network is used for looping effects only (`--motion auto`).
- **Refinement through the effect's renderer** redraws the first frames better (smoke +0.4 dB and explosion +1.4 dB at
  1 frame) and ties at 8 and 30 frames (−0.06 dB on average). Kept: the first frames are what a viewer compares with
  the footage.
- **Context**: 8 frames (4 frames: 0.25 dB less on fire, 0.07 more on smoke, 0.7 less on the explosion). **Fine fields kept at 64 x 64**: at 128 x
  128 fire's start frame is redrawn almost exactly (50.5 dB at frame 0 instead of 27.3) and frame 1 gains 1.5 dB, but
  8 and 30 frames are unchanged, for 4 times the storage; `--fine 128` is an option.
- **What the estimator is told does not matter much**: controls of 0.5 each instead of the run's changed tracking by
  +0.03 dB on average (8 and 30 frames, block matching + assimilation), and the run's own noise seed by +0.01.
- **The choice holds under every condition.** Without alpha, after blur and noise, after H.264, and on the learned
  renderer's frames, the same configurations rank first (`j_summary.md`).
- **Against today's stored start**, the choice was ahead on validation at 8 and 30 frames for every effect (fire
  +3.0 and +1.2 dB, smoke +6.7 and +2.7, explosion +6.7 and +3.4; intervals above zero), and 0.4 to 3.5 dB behind the
  true start.

**Endless statistics on validation** (6 settings, estimated minus stored start points, `j_val_endless.csv`):
- Smoke and the explosion are as good or better: coverage is closer (smoke −0.006 [−0.010, −0.001], explosion
  −0.029 [−0.053, −0.009]), the average picture is closer (mean-frame PSNR smoke +0.70 [+0.16, +1.29] dB, explosion
  +2.26 [+0.06, +4.51]), and the rest ties.
- Fire's detail is closer (spectrum distance −0.055 [−0.097, −0.018]) but it moves less: |log motion ratio| +0.077
  [+0.027, +0.133] (motion ratio 0.93 against 0.95 for the stored start points and 0.99 for a second real run).
- Two diagnostics on fire did not change that: fine fields kept at 128 x 128, and no fine fields at all, grown by the
  warm-up as fire's stored start points are (`j_val_endless_fire_fine128.csv`, `j_val_endless_fire_grown.csv`). So it
  comes from the estimated coarse state. A likely cause, not tested, is that the motion network's velocity is an
  average estimate, smoother than a real flow.

## 5. The rule (written before the test)

The test is run once (`nvfx_study_j test` and `test-endless`). For each effect, with `clean_rgba` footage:

1. **The path works (tracking):** `estimated` minus `stored_nearest`, active PSNR at 8 frames and at 30 frames, both
   95% intervals above zero (paired over the 16 test runs). If this holds, `nvfx_estimate` is the documented way to
   continue footage of that effect.
2. **Estimated start points can replace the stored ones (endless):** at study B's 10 held-out settings, for detail
   spectrum distance, |log motion ratio|, coverage distance, emission distance and mean-frame PSNR, no interval lies
   entirely on the worse side. If this fails, the tool still works, and the statistic that is worse is written down
   as a limit of effects made from footage.

Ties count as ties. The other footage conditions are reported with the same comparisons, outside the rule. Validation
suggests rule 1 holds for every effect and rule 2 may fail on fire's motion.

## 8. How to use it

You need: a rollout effect trained on the built-in simulation (fire, smoke or explosion, `.nvfx`), footage of the same
kind of effect, and a licence that allows training a shipped model on it ([DATA.md](DATA.md) §3). Start points made
from footage are data derived from it, so the footage goes through the same licence check and register as `nvfx_ingest`.

```sh
# 1. Footage into a 128 x 128 clip, after the licence check (alpha: keep for footage with real alpha, luma for footage
#    shot on black).
build/nvfx_ingest --input campfire.mov --name campfire --licence own --source "studio shoot, 2026-10-12" \
                  --alpha luma --size 128 --fps 30 --crop 720:720:600:200

# 2. The inverse networks for the effect, once per effect and kind of footage (about 6 minutes on one core; cached
#    under $NEURALVFX_DATA/j/inverse/ by `nvfx_study_j inverse` and `nvfx_study_j motion`).
build/nvfx_estimate --effect fire.nvfx --clip $NEURALVFX_DATA/clips/ingest/campfire.nfxclip \
                    --train-inverse $NEURALVFX_DATA/j/inverse/fire_rgb.nvfxinv --out fire_campfire.nvfx

# 3. Start points from frames 40, 80 and 120 of the footage, at the controls that look like it, with a preview sheet
#    (per start point: the footage frame, then the effect from it at 0, 8, 30 and 60 frames).
build/nvfx_estimate --effect fire.nvfx --clip $NEURALVFX_DATA/clips/ingest/campfire.nfxclip \
                    --at 40,80,120 --controls 0.6,0.5,0.4 --out fire_campfire.nvfx --preview campfire_starts.png
```

- `--at`: the footage frames to start from (0-based; default the last). Each uses up to `--context` frames (default 8)
  ending at it, so a start point needs a few frames before it; the motion network needs three.
- `--controls`: the effect's controls that best match the footage (intensity, wind, turbulence for the built-in
  effects). They are stored with the start points, and the runtime picks start points by the nearest controls. On
  simulated footage, telling the estimator the wrong controls (0.5 each) cost little (§4).
- `--age` (explosions): seconds since the detonation at the footage's first frame (default one frame).
- `--keep-starts`: keep the effect's stored start points as well (default: only the estimated ones, so every shard
  starts from the footage).
- `--alpha yes|no|auto`: whether the footage has a real alpha channel. `auto` says no when alpha is max(r, g, b)
  everywhere (`--alpha luma` on ingest) or zero.
- `--fine 64`: the size of the fine fields kept with each start point (the effect's own size when it keeps them).
- `--input VIDEO` or `--frames-dir DIR` instead of `--clip` run the ingest path in the tool, with `--licence`,
  `--source` (and `--author`, `--licence-note`, `--crop`, `--start`, `--duration`).

The result is an ordinary rollout effect: the runtime, the viewer and scene scripts play it like any other. Check it by
eye before shipping: the estimator was validated on simulated renders only.
