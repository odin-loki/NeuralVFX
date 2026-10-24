# Study hres: finer coarse 3D fire (high-resolution state, same detail grid)

Status: **the fire tested and passing its rule; the smoke a labelled demo** (1750 iterations each; trained and validated
23 October 2026, the fire's test edge check judged with the calibrated bounds 24 October, §7). Shipped:
`models/3d/fire_hres.nvfx3` (**C2** chosen on validation; `results/3d/hres/fire/frozen.csv`; its one test in
`compare_test_*.md` and `edgecheck_test.md`, summarised in [REPORT.md](REPORT.md) §15's hres addendum) and
`models/3d/smoke_hres.nvfx3` (demo; fails edge check, §8). Restore via `models/restore.sh` under `$NEURALVFX_DATA/hres_3d/`. Stage snapshots
`C1`–`C4` remain in `results/3d/hres/fire/models/`. Demo videos: [videos/README.md](../videos/README.md) §9. Audience:
dev, research, the owner.

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

# Train (resumable). CPU default. Optional GPU trainer (Y-slab sharded conv3, lower host RAM):
#   cmake -S . -B build-cuda -G Ninja -DCMAKE_CXX_COMPILER=g++-14 -DNEURALFX_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
#   ninja -C build-cuda nvfx_3d
#   export B=build-cuda/nvfx_3d TRAIN_DEVICE=cuda TRAIN_SHARDS=4
$B --study hres train --root $R --name hres --runs-dir train --also $R/runs/long \
  --iterations 900 --finetune 350 --activity 250 --long 250 --batch 8 --unroll 12 --burn 24 \
  --seed 9600000 --threads 3
# With GPU: append --device cuda --shards 4 (or set TRAIN_DEVICE / TRAIN_SHARDS in hres_fire.sh / hres_machine_run.sh).

$B --study hres finish --root $R --ckpt hres --out hres_fire --runs-dir train --also $R/runs/long

# Validation then test (after freezing), same subcommands as study N2 with --study hres
```

Results tables are in `results/3d/hres/` (the fire's validation in `fire/choose_val.md` and `edgecheck_val.md`, its
test in `fire/compare_test_*.md` and `edgecheck_test.md`, §7); [REPORT.md](REPORT.md) §15's hres addendum summarises
them.

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
A third lineage, the smoke on study R2's teacher (`--effect smoke_r2`, its own seeds), is §10.

## 7. Reproduce (fire, full)

```sh
tools/study_3d/hres_fire.sh build/nvfx_3d
# then validation commands printed at the end; mirror study N2 §14.13 with --study hres --root $NEURALVFX_DATA/hres_3d/fire
```

Smoke: `tools/study_3d/hres_smoke.sh build/nvfx_3d`. After training completes, `finish` + demo validation:
`bash tools/study_3d/hres_smoke_after_train.sh` (or `bash tools/study_3d/start_smoke_watcher.sh` while CPU train runs).
Smoke on study R2's teacher, with its losses (the result that asks for it, §8; the plan and its driver, §10):
`bash tools/study_3d/hres_smoke_r2.sh` prints the plan and what is there, `FORCE=1` runs every stage not done
(`STAGE=` a subset); `bash tools/study_3d/hres_smoke_r2_train.sh` is the same driver held to `record,train,finish`.
`hres_status.sh` reports the lineage's runs, training, candidates, bounds and verdicts after the fire's and the smoke's.

Quick CUDA sanity check (after `build-cuda`): `bash tools/study_3d/cuda_train_check.sh`.

Status: `bash tools/study_3d/hres_status.sh` (or `monitor_hres.sh` every 60 s). Fire validation (long): `bash tools/study_3d/start_fire_validate.sh`
(uses `nvfx_bin_link`; safe with a space in the repo path). After validate, publish into `models/3d/`: `bash tools/study_3d/start_fire_publish_watcher.sh`.
While smoke CUDA train runs: `bash tools/study_3d/pause_fire_validate.sh` (removes partial edge CSVs; resume skips finished steps).
The smoke watcher auto-starts fire validate after smoke reaches iteration 1750 if validate is not done yet.

**Calibrated edge bounds** (several simulation runs per wind, study R-style; optional after a single-run edgecheck):
`bash tools/study_3d/hres_fire_edge_bounds.sh`, then `bash tools/study_3d/hres_fire_recheck_edges.sh` to rebuild `choose_val.md` without
rerunning val track/play. Or `USE_EDGE_BOUNDS=1` on a fresh `hres_fire_validate.sh` run once `edge_teacher.csv` and `edge_bounds.csv` exist.

**The fire's test edge check, judged with the calibrated bounds** (24 October 2026; `results/3d/hres/fire/edgecheck_test.md`,
rebuilt from the same `edge_test*.csv` by `bash tools/study_3d/hres_fire_recheck_test_edges.sh`, no model or simulation
rerun). The 22 October test step ran before `edge_bounds.csv` existed and judged the frozen C2 against one simulated run
per wind (study N2's rule); that table, kept as `edgecheck_test_single.md`, marked two winds as runaway: wind 0.50 at
x = 0 (largest 3.43 over 5 to 30 s where the one run held 0.01) and wind 0.95 at z = 0 (2.36 against 0.17). Against the
bounds calibrated on four teacher runs per wind (§4's rule, the one the validation was judged by), **the learned fire
passes at every wind and side**, as does N2: the two marked sides sit at 3.43 under the bound 4.95 (0.69 of it) and
2.36 under 4.73 (0.50), no side holds more than its bound and none grows (late mean over 20 to 30 s above twice the
early mean plus 2). The largest held heat anywhere is at x = 0 for wind 0.05, 52.11 under 115.76, where the teacher
itself holds 47 to 56. The fire's rule is therefore met on validation and on its one test; [REPORT.md](REPORT.md) §15's
hres addendum and `videos/README.md` §9 say so.

**Resume smoke on GPU** (stops CPU train, resumes `ckpt/smoke.ckpt`): `bash tools/study_3d/hres_smoke_resume_cuda.sh`
(`TRAIN_SHARDS`, `BUILD_DIR=build-cuda`). Or fresh: `TRAIN_DEVICE=cuda bash tools/study_3d/hres_smoke.sh`. `nvfx_bin_link.sh` avoids
WSL path issues when the repo lives under `Neural VFX`.

**Long scripts and edits.** Bash reads a script incrementally, so editing `hres_fire_validate.sh` or a watcher while it
runs corrupts the run (`validate.log` and `after_train.log` each hold one "syntax error near unexpected token" from
22 and 23 October). `hres_fire_validate.sh` now keeps its whole body in `main()` (parsed in full before anything runs),
and `run_fire_validate.sh` runs a snapshot copy under the data root with `NVFX_REPO` exported so the copy still writes
under the repo's `results/3d/hres/fire/` (the 22 October snapshot run, before `NVFX_REPO`, wrote its test tables under
`$NEURALVFX_DATA/hres_3d/results/3d/hres/fire/` instead; they are in the repo since 24 October. `hres_status.sh`
reports them present or missing, and `bash tools/study_3d/hres_sync_fire_test_results.sh` copies any that are missing
from the repo back out of the data root; `FORCE=1` overwrites).
The watchers (`hres_smoke_after_train.sh`, `hres_fire_after_validate.sh`, `hres_train_until_done.sh`) are still bare
scripts: stop them with `restart_hres_watchers.sh` before editing them. `restart_hres_watchers.sh` is idempotent: it
never starts a train, and once `after_train.done` exists and fire validate is done it is a no-op (`FORCE=1` to redo
the smoke finish and validation).

## 8. Smoke result: the edge failure, and study R2's losses

`smoke_hres` (`finish` from `ckpt/smoke.ckpt` after 1750 iterations on the CUDA trainer, 23 October 2026) ran the
demo validation of §6: a 30 s edge check at five winds against one simulated run per wind (seeds 9,656,900 + j for the
simulation, 9,657,900 + j for the models) beside study N2's 16³ `smoke2`. Tables: `results/3d/hres/smoke/edge.csv` and
`edgecheck.md`. **Both fail.** Side heat held over 20 to 30 s (the simulation's in brackets):

| wind | smoke2 (16³) | smoke_hres (32³) |
|---:|---|---|
| 0.05 | x = max 0.84, z = max 2.48, top 1.73 grow and hold (0.00, 0.00, 0.80): runs away | x = 0 20.36 (24.44), the rest ≤ 0.80: no runaway |
| 0.25 | x = max 1.73, z = max 2.68 (0.00, 0.38): runs away | x = 0 10.02 (19.13), top 1.10 (2.02): no runaway |
| 0.50 | x = max 1.14, z = max 2.25 (0.00, 0.00): runs away | **x = 0 1.15 (0.01) grows and holds: runs away** |
| 0.75 | x = max 2.35 (4.52), z = max 4.35 (0.19): runs away | **x = 0 2.69 (0.00) grows and holds: runs away** |
| 0.95 | no runaway | **x = 0 4.87 (0.00); top 1.90 (0.63): runs away** |

What the table says:
- **The failures differ in kind.** `smoke2` leaks heat out of the downwind and lateral sides at low and middle winds (a
  haze that reaches every open side; the simulation holds nothing there). `smoke_hres` holds the sides at low wind but
  from a wind of 0.50 up builds heat at **x = 0**, where the simulation holds none: the upwind side at 0.75 and 0.95
  (the wind control is centred at 0.5; above it the wind blows towards +x, so the plume leans to x = max, where the
  simulation holds 4.52 and 8.04), the still side at 0.50. The build-up grows from the early to the late window
  (0.31 → 1.15, 0.73 → 2.69, 2.95 → 4.87). This is study N's fault in a weaker form (STUDY_3D.md §14: heat kept at an
  inflow side), and study R's C4 failed its check by the same slow build-up at one side (0.46 over 5 to 15 s, 0.83
  over 20 to 30 s; [STUDY_R2.md](STUDY_R2.md) §2.4).
- **The bounds are single-run.** Study R2's rule (§2.10) calibrates the edge bounds on four teacher runs per wind, as the
  hres fire's `edge_bounds.csv` does; the smoke's were not. The fire's calibration shows what that changes: at wind
  0.50 the single-run `x = 0` reading is 0.01 but four runs give a bound of 4.95. The smoke's 0.50 and 0.75 verdicts
  (1.15, 2.69) could fall inside a calibrated bound; the 0.95 verdict (4.87, growing, and the top) would not on the
  fire's scale. Calibrate before concluding: `nvfx_3d edge --sim --runs 4 --effect smoke` then `edgecal`, as
  `hres_fire_edge_bounds.sh` does for the fire.
- **The link to study R2's spread loss.** Study R2 measured the learned smoke's coarse state over-spreading (study R's
  C4: 2.5 times the truth's row spread within 1 s, 5 times after 15 s, with a haze 1.7 times the truth's mass;
  STUDY_R2.md §2.4) and added two losses: the **edge loss** (weight 10; a hinge on heat and soot at the four open
  sides beyond the truth's) and the **spread loss** (weight 2; each row's mass-weighted spread about its own centroid
  against the truth's). `smoke_hres` was trained with neither (study N2's recipe as `hres_smoke.sh` runs it:
  `--iterations 900 --finetune 350 --activity 250 --long 250`, no `--edge`, no `--spread`), on study N's smoke
  preset, whose teacher loses two thirds of its buoyancy flux (STUDY_R.md). An upwind build-up that grows with time
  is what the edge loss penalises directly, and a plume that widens into a haze until it reaches the sides is what the
  spread loss is for. The hypothesis, not yet measured on this model: the hres smoke over-spreads as study R's did, and
  the excess reaches the upwind wall where the wind holds it. The measurement is `nvfx_3d widthprobe` on its training
  runs (`--study hres --effect smoke`; the coarse state's row spread over the truth's, as STUDY_R2.md §2.4 tabulates).

**Next for the smoke at hres** (training not started; the rule committed first): record `smoke_r2` at
64 x 96 x 64 (study R2's teacher: momentum fixed, dM/dz over B 1.088) block-averaged to 32 x 48 x 32, and train with
`--edge 10 --spread 2` and stage 4's long burn (48 to 360 frames), then judge on bounds calibrated on four teacher runs
per wind. Study R2's own result (16³, under way) says whether the losses fix the failure before the finer state is tried.

**The driver** (24 October 2026): `tools/study_3d/hres_smoke_r2.sh`, §10 (the plan without `FORCE=1`; with it the
stages not done: record, train, finish, bounds, edge). `hres_smoke_r2_train.sh` is the same script's first three stages
under the name the first commands used (`STAGE=record,train,finish`). The seeds are the tool's own for
`--study hres --effect smoke_r2` (§10's table; `hres_effect()` in `tools/nvfx_3d.cpp`), the initial weights included.

## 9. Beside the study: the fire and smoke coupled, a demo and a simulation

Neither is part of this study's rule, and neither is measured, but both stack a fire under a smoke as a finer smoke
would be used ([STUDY_R.md](STUDY_R.md) §6.1):
- **The demo through the volume API** (`nvfx_3d video --kind coupled`; `include/neuralfx/coupled3d.hpp`): two learned
  volumes, N2's fire below and study R's smoke above, drawn as one 64 x 192 x 64 volume. The only thing that crosses
  the seam is a number: the fire's soot flux out of the top of its box (`top_soot_flux`, smoothed and set against its
  running mean by `FluxDrive`) sets the smoke's intensity. The boxes exchange no heat, soot or momentum, and the seam
  shows as a flat base under the upper smoke. `tests/test_coupled3d.cpp` checks the rule's arithmetic.
- **The simulation** (`nvfx_3d video --kind coupled-sim`; `sim3d::Stack` in `include/neuralfx/sim3d.hpp`, 24 October
  2026): two solver boxes, the fire's below and a burnerless box above (`smoke_r` by default), the lower's top row and
  the upper's bottom row each the other's border row at the start of every solver step, so heat, soot and velocity
  cross the seam both ways; the pressure solve stays each box's own, and the two together keep their heat and soot to
  within a few per cent (`Sim3d.CoupledStack*`). No flat base: what rises out of the fire is what the smoke is.
The learned stacked fire and smoke that would replace the demo, trained on `Stack`'s runs at this study's finer state
and tested as study R2 is, is not started (TODO.md, "Next"); a smoke at hres that passes its edge check (§8, §10) comes
first.

## 10. Smoke on study R2's teacher, with its losses (planned; committed before any run)

Status: **scripted, not run** (24 October 2026). The plan §8 ends with, as one resumable driver:
`tools/study_3d/hres_smoke_r2.sh`. Without `FORCE=1` it prints the plan and what is there and starts nothing;
`FORCE=1` runs every stage not done (`STAGE=record,train,finish,bounds,edge`, or some of them). Everything below is fixed
before the first run; the results, when there are any, go after this section.

**The question.** Does the hres smoke's edge failure (§8: heat building at x = 0 from a wind of 0.50 up, the upwind
side) go away when the same finer state is trained on study R2's teacher (`smoke_r2`, its momentum budget closed:
[STUDY_R2.md](STUDY_R2.md) §1.4) with study R2's two losses (the edge loss, weight 10, and the spread loss, weight 2;
§2.4 there), judged on edge bounds calibrated on four teacher runs per wind rather than one?

**Teacher and data.** `smoke_r2` at 64 x 96 x 64, block-averaged to 32 x 48 x 32 (fine 2), the same model as §2 (32-wide
stepper, 8 memory channels, ambient edges, warm smoke's optics; the tracking peaks are study R's, heat 0.5 and soot
1.2). Runs as §2: **40 runs at 8 s** (240 frames) and **24 runs at 18 s** (540 frames) of study N2's kinds, under
`$NEURALVFX_DATA/hres_3d/smoke_r2/runs/{train,long}`. `smoke_r2` costs about 0.5 CPU-seconds a frame on one thread
on the PC (two 240-frame runs took 119 and 127 s each, 24 October): about 3 CPU-hours for the runs, `J` at a time.

**Training.** Study N2's four stages as §3 (900 + 350 + 250 + 250, batch 8, unroll 12, burn-in up to 24 frames in
stages 2 and 3), **stage 4's burn-ins 48 to 360 frames** (study R2's, from the 540-frame runs), **`--spread 2 --edge 10`
in stages 2 to 4**, a checkpoint every 10 iterations, **CPU only** (the CUDA trainer has neither loss and the tool
refuses `--device cuda` with them; `TRAIN_DEVICE=cuda` is an error in the script). The hres smoke's 1750 iterations
took about a day on the CUDA trainer (§8); on a CPU budget several days at `T=4` threads, resumable from
`ckpt/smoke_r2.ckpt`. Candidates **C2, C3, C4** (the stepper after stages 2, 3 and 4) and the final `smoke_r2_hres`.

**Seeds** (its own block, unused inside study N's range; the tool's defaults for `--study hres --effect smoke_r2`):

| Use | Value |
|---|---|
| Training runs, 240 frames | 9,630,000 + i |
| Training runs, 540 frames | 9,631,000 + i |
| Initial weights | 9,630,100 |
| Trainer | 9,630,000 |
| Edge bounds' teacher runs | 9,633,000 + 100 j + k (j the wind, k the run) |
| Validation tracking / real / plays | 9,660,000 / 9,661,000 / 9,662,000 + 100 s + k |
| Validation edge-check plays (sim, single) | 9,662,900 + j (9,661,900 + j) |
| Test tracking / real / plays | 9,870,000 / 9,871,000 / 9,872,000 + 100 s + k |
| Test edge-check plays | 9,872,900 + j |

Held-out settings: §2's (the hres fire's and smoke's; none is a training run's).

**The rule.** First the edge check, the gate: the edge bounds from **four teacher runs per wind** (`nvfx_3d edge --sim
--runs 4`, then `edgecal --a 2 --b 4 --c 2`, the hres fire's B and C at this coarse size, in
`results/3d/hres/smoke_r2/edge_bounds.csv`; written before any candidate is judged), then each candidate's 30 s play
at the five winds against them (`edgecheck --bounds`; `edge.csv`, `edgecheck.md`). **A candidate passes if no side
runs away at any wind.** If one does, the smoke at hres is promoted to §4's full rule (tracking and endless play on
validation, the choice, the freeze, one test), with the seeds above and `--study hres --effect smoke_r2`; if none does,
the result is reported here and the finer state waits on study R2's own verdict (STUDY_R2.md §3 onwards). Comparison
for pictures and endless statistics: `smoke_hres` (§8) and study R2's chosen 16³ model when it exists.

**Commands** (repo root; `NEURALVFX_DATA` the data root):

```sh
bash tools/study_3d/hres_smoke_r2.sh                         # the plan and what is there; starts nothing
FORCE=1 T=4 J=4 nohup bash tools/study_3d/hres_smoke_r2.sh > ~/nvfx-data/hres_3d/smoke_r2_run.log 2>&1 &   # all stages
FORCE=1 STAGE=bounds bash tools/study_3d/hres_smoke_r2.sh    # the teacher's edge runs and the bounds alone
FORCE=1 bash tools/study_3d/hres_smoke_r2_train.sh           # record, train and finish only (STAGE=record,train,finish)
bash tools/study_3d/hres_status.sh                           # reports its runs, training, candidates, bounds and verdicts
bash -n tools/study_3d/hres_smoke_r2.sh tools/study_3d/hres_smoke_r2_train.sh   # after any edit (both parse clean, 24 October)
```

**The driver's stages, and what each leaves** (`hres_smoke_r2.sh`; `STAGE=all` is the five in order, each skipped when
its output is already there, so a stopped run is resumed by the same command; the data root is
`$NEURALVFX_DATA/hres_3d/smoke_r2`, or `HRES_R2_ROOT`, and the tables go under `results/3d/hres/smoke_r2/`):

| Stage | What runs | Leaves (data root unless a `results/` path) |
|---|---|---|
| `record` | `nvfx_3d record` once per run, `J` processes at a time on one thread each; the tool skips a run file that exists | `runs/train/*.n3run` (40), `runs/long/*.n3run` (24) |
| `train` | one `nvfx_3d train … --spread 2 --edge 10 --device cpu --threads T --checkpoint-every 10`, resumed from `ckpt/smoke_r2.ckpt`; done when the log holds `done at iteration 1750` | `ckpt/smoke_r2.ckpt`, `ckpt/smoke_r2_stage{2,3,4}.ckpt`, `logs/train_smoke_r2.log` |
| `finish` | `nvfx_3d finish` from each stage checkpoint, then from the final one; a model already written is kept | `models/C2.nvfx3`, `C3`, `C4`, `smoke_r2_hres.nvfx3` |
| `bounds` | `nvfx_3d edge --sim --runs 4` (the teacher at the five winds, 30 s; runs already in the file are skipped), then `edgecal --a 2 --b 4 --c 2`; skipped while `edge_bounds.csv` exists (delete it and `edge_teacher.csv` to calibrate again) | `results/…/edge_teacher.csv`, `edge_calibration.md`, `edge_bounds.csv` |
| `edge` | each candidate's 30 s play at the five winds (`edge_C2.csv`, written afresh each time), the parts joined to one table, `edgecheck --bounds` | `results/…/edge_C{2,3,4}.csv`, `edge.csv`, `edgecheck.md` |

The driver refuses `TRAIN_DEVICE` other than `cpu`, refuses `train` before all 64 runs exist, `edge` before the bounds and
the candidates, and never starts a second tool on the data root (`logs/train.lock` under `flock`, and a `pgrep` for a
`record`, `train` or `finish` on the same `--root`; a second call reports the first and leaves). Its knobs, all
environment variables: `T` threads for the trainer and the edge runs (4; `TTRAIN` is the same), `J` recording processes
(4; `TREC`), `NICE_LEVEL` (10), `SPREAD` and `EDGE` (the rule's 2 and 10), `EDGE_RUNS` (4), `EDGE_B` and `EDGE_C` (the
hres fire's 4 and 2), `NVFX_3D` or `BUILD` (the tool; `build/nvfx_3d` through `nvfx_bin_link.sh` by default), `NVFX_REPO`
when a copy of the script runs from outside the repo. The whole body is `main()` with one call at the end, so an edit
while a run is under way cannot corrupt it (§7's lesson). `hres_status.sh` reports, under `smoke_r2:`, the run counts, the
trainer's last line or `DONE`, whether the driver or the trainer is running, each model present, the bounds and the
candidates' verdict lines from `edgecheck.md`; before anything is recorded it names this driver.

The tool's side: with `--study hres`, `--effect smoke_r2` selects this lineage's seed blocks in `record`, `train`
(trainer and initial weights), `truth`, `real`, `play` and `edge` (`hres_effect` in `tools/nvfx_3d.cpp`); `smoke_r` is
refused there. Study N's smoke (`--effect smoke`, §6) and the fire keep their seeds.
