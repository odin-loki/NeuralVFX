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

<!-- sections 3 to 7 (protocol, validation, test, decision, limits) follow the study -->

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
