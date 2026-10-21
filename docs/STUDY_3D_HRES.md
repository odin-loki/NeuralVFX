# Study hres: finer coarse 3D fire (high-resolution state, same detail grid)

Status: **protocol committed** (17 October 2026). Audience: dev, research, the owner. Validation and the test have not
run.

**In short.** Stage 1 (study N) used a **16 x 24 x 16** coarse state with **fine = 4** to drive **64 x 96 x 64** detail.
Study hres doubles the coarse resolution to **32 x 48 x 32** with **fine = 2**, the same detail volume, a **32-wide**
stepper and **8** memory channels. It reuses study N2's recipe (ambient edges, long rollouts, changing controls, edge
check, choice on tracking and endless play). Data live under `$NEURALVFX_DATA/hres_3d`.

This is the in-container step towards the PC plan's larger models ([TODO.md](../TODO.md) stage 2: 48³ to 64³ coarse,
256³ detail). It does not need new teacher simulations at another grid size: training runs are still recorded at
**64 x 96 x 64** and block-averaged to the finer coarse grid.

## 1. The question

Does a **finer coarse state** at the same detail resolution reduce blocky flames and improve tracking and endless play,
within the same CPU budget order as study N2, without the runaway study N had?

## 2. Model and data (committed before validation)

| Item | Study N / N2 | Study hres |
|---|---|---|
| Coarse grid | 16 x 24 x 16 | 32 x 48 x 32 |
| Fine factor | 4 | 2 |
| Detail grid | 64 x 96 x 64 | 64 x 96 x 64 |
| Stepper width | 16 | 32 |
| Memory channels | 4 | 8 |
| Edges | N: clamp; N2: ambient | ambient |
| Data root | `n_3d` / `n2_3d` | `hres_3d` |
| Tool flag | `--study n` / `n2` | `--study hres` or `n3` |

**Training runs:** 40 runs at 8 s (240 frames), seeds **9,600,000 + i**; plus **24 runs at 18 s** (540 frames) with
study N2's mix, seeds **9,620,000 + i** (written under `runs/long/`). Optionally add study N2's 64 runs with
`--also $NEURALVFX_DATA/n2_3d/runs/train` after re-blocking them is **not** valid (different coarse size); record fresh
runs only.

**Held-out settings** (intensity, wind, turbulence), none a training run's:

- validation: (0.62, 0.12, 0.48), (0.38, 0.88, 0.62), (0.72, 0.28, 0.38), (0.48, 0.52, 0.78), (0.28, 0.68, 0.22);
- test: (0.68, 0.07, 0.42), (0.52, 0.93, 0.52), (0.42, 0.22, 0.68), (0.78, 0.48, 0.32), (0.33, 0.72, 0.42),
  (0.63, 0.38, 0.88).

**Seeds** (unused blocks inside study N's range):

| Use | Validation | Test |
|---|---|---|
| Tracking | 9,650,000 + 100 s + k | 9,860,000 + 100 s + k |
| Real endless | 9,651,000 + 100 s + k | 9,861,000 + 100 s + k |
| Model plays | 9,652,000 + 100 s + k | 9,862,000 + 100 s + k |
| Edge check plays | 9,652,900 + j | 9,862,900 + j |
| Edge check sim | 9,651,900 + j | 9,861,900 + j |

(s = setting index 0..4 or 0..5, k = run 0..2, j = wind index 0..4.)

**Initial weights:** seed **9,600,777**.

## 3. Training schedule

Same stages as study N2 ([STUDY_3D.md](STUDY_3D.md) §14.3): 900 + 350 + 250 + **250** iterations, batch 8, unroll 12,
burn-in up to 24 frames in stages 2–3, long-rollout stage with burn 48–240 frames on the 540-frame runs only.
Trainer seed **9,600,000**. A snapshot at the end of every stage.

Expect roughly **eight times** the stepper work per frame versus study N2 (eight times as many coarse cells, about twice
the channel width). Budget several times study N2's train time on the same machine.

## 4. Measures and rule

As study N2 (§14.5): tracking from the true state (field and picture PSNR at 8 frames, 1 s and 2 s); endless statistics
(spectrum, motion, coverage, emission, mean picture); **edge check** over 30 s at five winds with bounds calibrated from
**several** simulated runs per wind (not one); choice among candidates on validation using tracking **and** endless play
and the edge check. The test runs **once** on the frozen model.

Comparisons: study N2's chosen model (C4) and the coarse simulation with the same detail, both re-evaluated at hres
coarse size where needed for fair field PSNR.

## 5. Commands

```sh
export NEURALVFX_DATA=~/nvfx-data
export B=build/nvfx_3d
export R=$NEURALVFX_DATA/hres_3d
mkdir -p $R

# Record (fire): short runs then long runs (two jobs or one with --from)
$B --study hres record --root $R --runs 40 --frames 240 --threads 3
$B --study hres record --root $R --runs-dir long --from 0 --runs 24 --frames 540 --threads 3

# Train (resumable)
$B --study hres train --root $R --name hres --runs-dir train --also $R/runs/long \
  --iterations 900 --finetune 350 --activity 250 --long 250 --batch 8 --unroll 12 --burn 24 \
  --seed 9600000 --threads 3

$B --study hres finish --root $R --ckpt hres --out hres_fire --runs-dir train --also $R/runs/long

# Validation then test (after freezing), same subcommands as study N2 with --study hres
```

Results tables belong in `results/3d/hres/`; update this doc and [REPORT.md](REPORT.md) when validation completes.

## 6. Smoke (study N2's smoke2 lineage)

Study N2 retrained smoke on study N's 40 smoke runs with ambient edges and a long stage (`smoke2.nvfx3`; [STUDY_3D.md](STUDY_3D.md) §14.9). It fixed the 30 s runaway but was not validated. Study hres repeats the **same training recipe** on a **finer coarse grid** for the smoke preset.

| Item | Fire hres | Smoke hres |
|---|---|---|
| Data root | `$NEURALVFX_DATA/hres_3d/fire` | `$NEURALVFX_DATA/hres_3d/smoke` |
| `--effect` | `fire` (default) | `smoke` |
| Training seeds | 9,600,000 + i | 9,610,000 + i |
| Trainer seed | 9,600,000 | 9,610,000 |
| Initial weights | 9,600,777 | 9,610,100 |
| Val / test seed blocks | §2 table | +50,000 on fire's (e.g. val track 9,655,000 + …) |
| N2 baseline to beat on pictures | `n2_3d/models/chosen.nvfx3` | `n2_3d/models/smoke2.nvfx3` |
| Success | full rule as §4 | edge check over 30 s + demo orbit video; no held-out test unless promoted |

Driver: `tools/study_3d/hres_smoke.sh`. Field PSNR against N2's 16³ model is not meaningful; compare pictures and endless statistics at hres.

## 7. Reproduce (fire, full)

```sh
tools/study_3d/hres_fire.sh build/nvfx_3d
# then validation commands printed at the end; mirror study N2 §14.13 with --study hres --root $NEURALVFX_DATA/hres_3d/fire
```

Smoke: `tools/study_3d/hres_smoke.sh build/nvfx_3d`.
