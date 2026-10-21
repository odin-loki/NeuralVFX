# Study hres results (finer coarse 3D rollouts)

Protocol: [docs/STUDY_3D_HRES.md](../../docs/STUDY_3D_HRES.md). Drivers: `tools/study_3d/hres_fire.sh`, `hres_smoke.sh`.

| Effect | N2 baseline (16³ coarse) | hres target (32³ coarse) | Data root |
|---|---|---|---|
| Fire | `n2_3d/models/chosen.nvfx3` (C4) | `hres_3d/fire/models/hres_fire.nvfx3` | `$NEURALVFX_DATA/hres_3d/fire` |
| Smoke | `n2_3d/models/smoke2.nvfx3` | `hres_3d/smoke/models/smoke_hres.nvfx3` | `$NEURALVFX_DATA/hres_3d/smoke` |

**Fire weights (in git):** `fire/models/` — `C1.nvfx3` … `C4.nvfx3` (stage snapshots) and `hres_fire.nvfx3` (final
`finish` from `ckpt/hres.ckpt`). Copy to `$NEURALVFX_DATA/hres_3d/fire/models/` for `nvfx_3d` drivers, or pass
`--model results/3d/hres/fire/models/hres_fire.nvfx3` from the repo root.

Training checkpoints (`ckpt/*.ckpt`) and simulation runs stay under `$NEURALVFX_DATA/hres_3d` only. Committed when
validation runs: `fire/` and `smoke/` CSVs, `frozen.csv`, edge tables.
