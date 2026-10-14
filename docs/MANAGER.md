# The effect manager: many rollout effects within a CPU budget

Status: **in progress**. Audience: dev. The API is `include/neuralfx/nvfx_manager.h`; the code is `src/manager/`.

## 5. Quality: what the levels of detail cost

### 5.1 Design (written before any run)

The manager draws a simulation's frames between its updates by interpolation, and draws distant effects at smaller
sizes. Both cost quality; this section measures how much, against **full-rate, full-resolution playback of the same
runs**, with the project's protocol (docs/REPORT.md §2 and §6).

- **Effects.** Fire, smoke and explosion (v2, `$NEURALVFX_DATA/v2/models`), steam and magic (study D's recipe,
  `$NEURALVFX_DATA/fx/models/d`).
- **Settings and seeds.** Validation: study G's ten validation settings (docs/DCM.md §4). Test: study B's ten held-out
  settings. Fresh seeds, never used before in the project: validation 7,300,000 + 100 s + e (the studies' salt-3 split),
  test 7,700,000 + 100 s + e, for setting s and effect e.
- **Runs.** 300 frames (10 s) for the looping effects, the explosion's 90; each played through the manager with its level
  locked (`nvfx_manager_set_lod_lock`), one tick per frame, keeping the frame shown at every tick (`nvfx_lod`).
- **Rate.** At 32, 64 and 128 px, updates every 2, 3 and 4 frames (15, 10 and 7.5 Hz), the frames between drawn by
  **hold** (the last update), **crossfade** (the two updates blended) or **motion** (both warped along the
  displacement of the effect's coarse flow between them, then blended); against the full-rate run at the same size. The
  keyframes are the full-rate run's frames to the bit (checked on every run), so only the frames between differ.
- **Size.** 32 and 64 px at 30 Hz against 128 px averaged down to the size shown, and drawn at 128 px (bilinear) against
  128 px.
- **Proxies.** 64 px at 7.5 Hz with the controls snapped to the centres of three cells per control and another seed, and,
  as the floor, another seed at the same controls at 64 px and 30 Hz; their statistics against 128 px averaged to 64.
- **Metrics.** Active PSNR per frame (the active-pixel PSNR of each frame against the reference frame, docs/REPORT.md §2;
  the mean over the frames drawn between updates for rate, over every frame otherwise), and the endless statistics of the
  whole run against the reference run: the detail spectrum L1 distance and the motion ratio (with the mean-frame PSNR).
- **Intervals.** Paired bootstrap over settings, 10,000 resamples, 95% percentile intervals; an interval covering zero is
  a tie.

### 5.2 The rule for the default interpolation (written before validation, committed before the test)

For each (effect, setting) pair, each method's metric is averaged over six cells: 64 and 128 px at 15, 10 and 7.5 Hz.
Paired bootstrap over the 50 pairs (ten settings of five effects).

1. **Motion** becomes the default if, on validation, (a) its active PSNR per frame minus crossfade's has a pooled interval
   entirely above zero, (b) no effect's interval for (a) lies entirely below zero, and (c) it is not worse on either
   endless statistic: the pooled interval of its |ln motion ratio| minus crossfade's has an upper bound below +0.02, and
   that of its spectrum L1 minus crossfade's an upper bound below +0.01.
2. Otherwise **crossfade** is the default (the cheaper of the two that move between updates), unless its active PSNR per
   frame minus hold's has a pooled interval entirely below zero, in which case **hold**.
3. The test runs once, after the choice is committed: the chosen default against each of the other two, per effect and
   pooled, with intervals; ties are reported as ties. The test does not change the choice.

The rate thresholds and sizes (§6) are chosen on validation from the same tables and the cost model, and reported on the
test.
