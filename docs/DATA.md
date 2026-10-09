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
