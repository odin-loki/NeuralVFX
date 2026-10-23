# Trained models

The shipped models, kept in git so they survive the loss of the machine that trained them (about 10 MB):

- `2d/*.nvfx`: the 33 learned 2D effects the library, the packs and the scenes play.
- `2d/study_d/`, `2d/v2/`: earlier fire and explosion versions that the fireball's frozen tests still use.
- `3d/fire.nvfx3`: study N2's C4, the 3D fire shown in the demos; `3d/fire_study_n.nvfx3` (study N) and
  `3d/smoke_demo.nvfx3` (N2's smoke, a demo, never tested, on study N's smoke preset; kept as study hres's smoke
  baseline); `3d/fire_hres.nvfx3` (study hres fire, 32³ coarse, **C2** chosen on validation, SHA in
  `results/3d/hres/fire/frozen.csv`); `3d/smoke_hres.nvfx3` (study hres smoke, same recipe as smoke2 on the finer grid;
  **labelled demo**: it fails the edge check in `results/3d/hres/smoke/edgecheck.md`, no held-out test).
- `3d/smoke.nvfx3`: study R's 3D smoke (docs/STUDY_R.md), C4 trained on the solver's smoke checked against plume
  theory (`smoke_r`), frozen by its SHA-256 before its test (`results/3d/r/frozen.csv`) and **shipped as a labelled
  demo**: by its rule it failed (an edge bound missed by 7% on validation; it tracks real runs better than the coarse
  simulation but looks worse played on).
- `prior/fire.ddpm`: study G's denoiser prior for the fire.
- `upscale/shared.nvfxup`: study P's learned 2x (experimental; it did not pass its rule).

`layout.txt` maps each path under the data root (`NEURALVFX_DATA`) that the tests and tools read to its file here;
`SHA256SUMS` checks them. `models/restore.sh [DATA_ROOT] [--link]` rebuilds those paths. Training clips and runs stay
under the data root, not in git. When a study replaces a model, add the new file here and update both lists.
