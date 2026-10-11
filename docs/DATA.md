# Data: clips, licences and what stays out of git

Status: **current**. Audience: dev, research.

## 1. Where data lives

Everything that is not code or a small text result lives under the data root: `$NEURALVFX_DATA`, or `~/nvfx-data`
when it is unset. Nothing under it is ever committed; `.gitignore` also blocks clips (`*.nfxclip`), weights
(`*.nvfx`, `*.pt`, `*.onnx`, ...), images outside `docs/` and videos.

| path under the data root | what |
|---|---|
| `clips/ingest/` | clips made from owner footage by `nvfx_ingest` |
| `licences.tsv` | the licence register: one row per ingested clip |
| `experiments/clips/{a,b,c}/` | simulated clips of the evaluation (`nvfx_experiment data`) |
| `experiments/models/` | trained models of the evaluation |
| `experiments/media/` | comparison sheets and side-by-side videos |

## 2. Simulated clips

`nvfx_sim` and `nvfx_experiment data` make clips with the project's own fluid solver (`src/sim`). They are
deterministic: the same parameters give the same bytes, so the evaluation can be regenerated anywhere and nothing
needs to be stored. There is no licence question: the project owns them.

## 3. Owner footage

```sh
nvfx_ingest --input fire_take3.mov --name fire_take3 --licence own --source "studio shoot, 2026-10-12" \
            --alpha luma --size 128 --fps 30 --loop-blend 16
```

- The licence is checked **before** the footage is read. Accepted: `own`, `CC0-1.0` (and public domain),
  `CC-BY-*` (with `--author`), `CC-BY-SA-*` (accepted but flagged: share-alike terms may reach a trained model; check
  before shipping), and `commercial` with `--licence-note` naming the agreement that allows training a model.
  Refused: NonCommercial (`-NC`) and NoDerivatives (`-ND`) licences, `unknown`, all rights reserved, anything else.
- Every accepted clip gets a row in `licences.tsv`: date, name, licence, author, source, note, flags, input, output,
  frames and size. Keep this file with the data; it is the record of what a shipped model was trained on.
- Alpha for footage without it: `additive` (alpha 0: pure emission on black), `luma` (alpha = brightest channel:
  black-keyed footage), `keep` (footage with straight alpha, premultiplied on ingest).
- Crops: centre square by default, or `--crop w:h:x:y` (ffmpeg syntax). Frames are area-resampled to the clip size.
- Any format ffmpeg reads works (ffmpeg must be installed); a folder of PAM or PPM frames works without it.

## 4. Splits and leakage

The evaluation never scores a model on what it trained on (see [REPORT.md](REPORT.md) §2):

- Task A compresses one clip per model, so the clip is both training and test by design (it is what a flipbook does
  too); its held-out test is the frame-interpolation variant, where only even frames are available and odd frames
  are scored.
- Task B holds out whole control settings: off-grid in every coordinate, never seen in training.
- Task C holds out whole seeds; generated variations come from new seeds.

For owner footage: hold out whole clips (never frames of a training clip), and record the split next to the register.

## 5. Third-party code fetched at build time

None of it is in git and none of it is in the runtime library (`nvfx`, `libnvfx.so`): the runtime has no third-party
dependency. The configure step fetches each item at a pinned commit (git, or a file checked by its SHA-256), in a child
CMake process, so a failed download only leaves that item out.

| component | from | pinned at | licence | used for | built into |
|---|---|---|---|---|---|
| bc7e (plain C++ port of `bc7e.ispc`), `basisu_bc7e_scalar.cpp/.h` | github.com/BinomialLLC/basis_universal, `encoder/` | commit `99f52d63aa6799cbdaecfe977111dc5ec3b31d47` (1 September 2026), files checked by SHA-256 | Apache-2.0 | BC7 encoding of the flipbook baselines | `neuralfx_bc7`, linked by `neuralfx_core` (study tools and tests) |
| bc7enc_rdo: `bc7decomp.cpp`, `bc7decomp_ref.cpp` | github.com/richgel999/bc7enc_rdo | commit `b9438627eef73a1157e84201b6fa6eb2ffd6d9f0` (30 July 2026) | MIT or public domain (Unlicense), the author's choice of two; its `bc7e.ispc` (not used) is Apache-2.0 | BC7 decoding (the reference decoder) | `neuralfx_bc7` |
| astc-encoder (the core library, `Source/astcenc_*.cpp`) | github.com/ARM-software/astc-encoder | tag 5.7.0, commit `baff485b0ff36d2f95d28961605106502c653966` (31 July 2026) | Apache-2.0 | ASTC encoding and decoding of the flipbook baselines | `neuralfx_astc`, linked by `neuralfx_core` |
| Dear ImGui | github.com/ocornut/imgui | tag v1.91.8 | MIT | the optional viewer (`NEURALFX_BUILD_VIEWER`) | `nvfx_viewer` only |

- `NEURALFX_FETCH_ENCODERS` (default ON) fetches the encoders (`cmake/encoders.cmake`). Without them, or when the
  download fails, the BC7 and ASTC flipbooks are not built: `flipbook::available()` says so, the study tools leave
  those rows out and the tests that need them skip. The original BC3-layout and raw flipbooks never need them.
- Third-party sources are compiled with their own language level (C++17) and without this project's warning flags.
  astc-encoder is built as its invariant build (no floating-point contraction), for SSE4.1 on x86-64, so it writes the
  same blocks on every x86-64 machine.
- System packages, not fetched: GoogleTest (BSD-3-Clause; the tests), zlib (zlib licence; PNG figures and the zlib
  references of study F), GLFW and OpenGL (the viewer), ffmpeg (a separate program, run through a pipe for video
  ingest and the video-codec baselines).
