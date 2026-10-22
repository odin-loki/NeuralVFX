# Study hres results (finer coarse 3D rollouts)

Protocol: [docs/STUDY_3D_HRES.md](../../docs/STUDY_3D_HRES.md). Drivers: `tools/study_3d/hres_fire.sh`, `hres_smoke.sh`,
`hres_machine_run.sh`. Optional GPU training: `build-cuda/nvfx_3d` with `TRAIN_DEVICE=cuda` (see §5 in the study doc);
sanity check: `bash tools/study_3d/cuda_train_check.sh`. Status: `bash tools/study_3d/hres_status.sh`.
While smoke train runs: `bash tools/study_3d/start_smoke_watcher.sh` (finish + validate + publish at 1750).
Fire validate (hours): `bash tools/study_3d/start_fire_validate.sh`; publish watcher: `start_fire_publish_watcher.sh`.
During smoke CUDA train, `pause_fire_validate.sh` avoids contention; the smoke watcher restarts fire validate at iteration 1750.

| Effect | N2 baseline (16³ coarse) | hres target (32³ coarse) | Data root |
|---|---|---|---|
| Fire | `n2_3d/models/chosen.nvfx3` (C4) | `hres_3d/fire/models/hres_fire.nvfx3` (also [models/3d/fire_hres.nvfx3](../../../models/3d/fire_hres.nvfx3) in git) | `$NEURALVFX_DATA/hres_3d/fire` |
| Smoke | `n2_3d/models/smoke2.nvfx3` | `hres_3d/smoke/models/smoke_hres.nvfx3` | `$NEURALVFX_DATA/hres_3d/smoke` |

**Fire weights (in git):** `fire/models/` — `C1.nvfx3` … `C4.nvfx3` (stage snapshots) and `hres_fire.nvfx3` (final
`finish` from `ckpt/hres.ckpt`). Copy to `$NEURALVFX_DATA/hres_3d/fire/models/` for `nvfx_3d` drivers, or pass
`--model results/3d/hres/fire/models/hres_fire.nvfx3` from the repo root.

Training checkpoints (`ckpt/*.ckpt`) and simulation runs stay under `$NEURALVFX_DATA/hres_3d` only. Committed when
validation runs: `fire/` and `smoke/` CSVs, `frozen.csv`, edge tables.
