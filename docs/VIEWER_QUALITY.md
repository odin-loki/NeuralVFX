# Viewer quality notes (product feedback)

Status: **recorded** (25 October 2026). Audience: training, library, and hres owners.

These notes come from hands-on use of `nvfx_viewer` (Library and Volume modes) and from comparing catalogue entries to real footage-style references. They are **not** pass/fail gates; they guide the next training and catalogue iterations.

## 1. Dynamism and repetition

Many 2D library effects were trained from **video-derived** clips (see [FOOTAGE.md](FOOTAGE.md) and per-study reports). Rollout models are **not** looped flipbooks: they do not repeat a fixed cycle. Even so:

- Motion can feel **narrow**: the plume or burst stays in a familiar shape band; variation reads as noise on top of the same silhouette rather than genuinely different flow.
- Long playback (~10–30 s) exposes **tell-tale sameness** — not a obvious loop, but a family resemblance frame to frame.
- **Room for improvement:** wider control coverage in training, more diverse teacher runs (wind and intensity sweeps), stronger spread / activity losses where studies support them (study R2 on 3D smoke), and more start points that match in-game control ranges (study M3-style fixes on 2D).

## 2. Smoke realism

Several **smoke** catalogue entries (columns, vents, sky patches, coupled scenes) are the main concern:

- **Texture:** marbled or overly uniform; missing the large-scale billows and fine filaments of the fluid teacher at the same controls.
- **Lighting:** field-shader looks help in scenes, but the underlying learned density still reads “CG smoke” rather than buoyant, turbulent gas.
- **Coupling:** rain thinning, wind shear, and heat from nearby fire are only as convincing as the smoke module itself.

**Room for improvement:** hres 3D smoke (`smoke_hres`, study R2) as the teacher reference path; retrain weak 2D smoke modules with longer unroll and profile losses; evaluate against [LIBRARY.md](LIBRARY.md) gates and viewer headless smoke tests (`ViewerHeadless.LibrarySmokeColumnRenders`).

## 3. What the automated viewer tests guarantee

Windowless tests ([`viewer/headless.cpp`](../viewer/headless.cpp), `ctest -R ViewerHeadless`, Xvfb `neuralfx.Viewer*`) check:

- Catalogue entries **parse and play** without crashing.
- Stage frames are **non-flat** (luma spread in the picture).
- Library stage RGBA size matches **width × height** (regression for non-square GL uploads).
- Shipped **3D fire** (`fire_hres.nvfx3`) steps and renders.

They do **not** score realism, dynamism, or match to reference video. Those remain human review and study metrics (margins, edge checks, endless statistics).

## 4. Suggested review workflow

1. `nvfx_viewer --library` — browse category; note entries called out in §2.
2. `tools/viewer_test.sh` — full automated viewer + headless suite before releases.
3. Compare to teacher sim or hres validation runs under `results/3d/` when retraining.
4. Update this file when an entry is superseded (date + replacement model id).
