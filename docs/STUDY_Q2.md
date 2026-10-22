# Study Q2: study Q's leftovers, fixed and measured

Status: **planned** (21 October 2026). Audience: dev, research, artists. The plan of the model changes and the form of
their rule (§4) are written and committed before any of the study's numbers; the rule's numbers (the margins and the
number of settings) come from a null probe on training settings and are committed before validation; each chosen file
is frozen by its SHA-256 before its one test. [REPORT.md](REPORT.md) §23 will have the summary.

Study Q2 takes up what study Q ([STUDY_Q.md](STUDY_Q.md) §9, [REPORT.md](REPORT.md) §21) left: the geyser's tracking
with its first frame kept, the haboob's and the smoke ring's first frames without the losses their refits brought, the
avalanche's and heavy gas's start points, and the mushroom's cap. The method is study Q's, unchanged (its §4.2 to §4.4;
study M3's lesson, REPORT §14.9: margins from a null probe, committed before validation): every candidate is the shipped
file with one change, validated by a rule whose numbers come from a probe on training settings, frozen and tested once
on held-out settings with fresh seeds, and it replaces a shipped model only if it passes.

Seeds: study Q2's range, 33,500,000 to 33,999,999 (§4.5; library, scene, figure and trainer seeds below 2^25 =
33,554,432). Code: `tools/nvfx_ordered.cpp --study q2` (study Q's `val`, `test` and rule on Q2's seeds and tables, and
three new ways of making a candidate: `train --anchor`, `refit --finetune`, `growfine`) and `RendererOptions::keep_scale`
in the trainer (`Rollout.RendererFineTuneKeepsItsScalesAndStartsFromItsWeights`). Every file is under the data root's
`q2_fix` (outside git); tables under `results/experiments/q2_fix`.

## 0. Where the study stands (handoff)

Planned (21 October 2026, 18:00 UTC). The plan and the rule's form (§4) are committed (c6e56e9); no number of the
study exists yet. Heavy work waits for the quiet machine: another agent re-times everything (the data root's
`quiet7/QUIET_DONE` marks its end; the coordinator dropped the three-hour fallback at 19:05 UTC, since round 7's quiet
timing repeats any block that sees foreign load: no CPU job of any size before the marker). Until then only reading,
planning, docs and the code (syntax-checked, not built). Every job below is already started
(`q2_fix/jobs/go.sh`, 18:00 UTC) and waits for the quiet machine itself.

**How to pick it up.** Every job is a script in the data root's `q2_fix/jobs/` that skips what is done (a marker
`jobs/<step>.done`, or the file it makes); its log is in `q2_fix/logs/`. `go.sh` starts them all: `build.sh` (the two
tools, copied to `q2_fix/bin` with the commit), `lane_train.sh A|B` (the fine-tunes), `lane_null.sh` (the geyser's and
heavy gas's nulls), `lane_small.sh` (the renderer fine-tunes, the start-point candidates, the other nulls, the
avalanche's diagnosis) and one `pipeline.sh` per effect (margins committed, candidates' probe, `rule.sh`, `val.sh`). Each
binary run writes its tables to `q2_fix/res/<effect>/` and `merge.sh` gathers them into
`results/experiments/q2_fix`; `commit.sh` commits only that folder and pushes. The avalanche's candidates' probe also
waits for `jobs/go_probe_avalanche`, made by hand after its diagnosis (§3). `freeze.sh` and `test.sh` are run by hand
after validation. **Disk** (the coordinator's limit, 21 October): study Q2's new data (`q2_fix` and the worktree's
build) stays under 1.5 GB and `/` keeps 1 GB free; every data-making step first runs `check_disk` (common.sh), which
waits while either fails and logs `PAUSED` to `q2_fix/logs/disk.log`; each effect's probe, validation and test caches
(`q2_fix/vt`) are deleted when its step is done (the tables keep the numbers). The order (§4.6): build, the candidates and the
null probes; each effect's margins committed and pushed (`jobs/margins_<effect>.committed`) before its candidates are
played on the probe; the rule's numbers (`q2_rule.csv`) committed and pushed before validation; the chosen file frozen
(`q2_frozen.csv`) and pushed before its test, which runs once (`q2_fix/test_<effect>.done`).

| leftover | effect (shipped file) | candidates (§4.1) | where it is |
|---|---|---|---|
| 1 | geyser (`fx_elements/models/d/geyser.nvfx`) | `u30a`, `u30al` | planned |
| 2 | haboob (`fx_elements/models/d/haboob.nvfx`) | `rft`, `rfs` | planned |
| 3 | smoke ring (`fx_air/models/d/smokering.nvfx`) | `rft`, `rfs` | planned |
| 4 | avalanche (`q_fix/models/avalanche.nvfx`) | `s48` (`msg` dropped by its diagnosis before its probe, §3) | planned |
| 4 | heavy gas (`fx_air/models/d/heavygas.nvfx`) | `ms`, `msg`, `s24` | planned |
| 5 | mushroom (`q_fix/models/mushroom.nvfx`) | `u60a`, `u90a` | planned |

## 1. Bottom line

(Written when the study ends.)

## 2. The leftovers, and what each needs

From study Q's validation and §9 (its numbers: candidate minus current model, 95% intervals; margins in brackets):

| leftover | what study Q found | what is needed |
|---|---|---|
| 1. the geyser's tracking | `u30` (the stepper fine-tuned on 30-frame windows): tracking at 1 s +0.41 dB [+0.27, +0.58], the mean frame +0.20 [+0.08, +0.32], but the first frame -0.19 [-0.28, -0.11] (-0.20) | the 1 s gain with the first step kept |
| 2. the haboob's first frame | `rf` (its renderer trained again from fresh weights with 100 early samples): the first frame +2.55 [+2.37, +2.70], the spectrum distance +0.013 [+0.006, +0.022] (+0.02) | the first frame with the detail kept |
| 3. the smoke ring's first frame | `rf`: the first frame +3.52 [+3.36, +3.69], its motion error +0.023 [+0.015, +0.030] (+0.02) | the first frame with the motion kept |
| 4. the avalanche's start points | the mean state (`ms`, `ms_u30`): the mean frame +0.84 to +0.93 dB, the spectrum +0.022 to +0.024 (+0.024) | the start points moved to the controls without the detail moving away |
| 4. heavy gas's start points | not tried; its tied mean picture is its start points' (STUDY_Q.md §3.3) | start points nearer the controls asked for |
| 5. the mushroom's cap | the cap turns ragged after 3 to 4 s; its stepper's (study O, REPORT §16.7; STUDY_Q.md §3.7) | a stepper that sees more of the cap's life |

**Why not a renderer refit for the geyser.** Study Q's `u30` changed only the stepper, and lost 0.19 dB at the first
frame of tracking: the first frame from a true state is one step of the stepper and the detail layer, drawn by the
renderer. The geyser's renderer on the true fields is 45 dB (STUDY_Q.md §3.6) and its first frame from a true state
35.6 dB, so the first frame's error is the step's; a renderer refit would change every candidate's first frame alike
and leave the fine-tune's loss there. What keeps the step is a constraint on the fine-tune: round 2's anchor
([COMPOSE.md](COMPOSE.md) §10: each window that starts at a true state also pays the first step's squared distance from
the current model's first step), which there kept smoke's first frame tied, with a lower learning rate.

**Why fine-tune the renderers rather than train them again.** Study Q's `rf` trained the haboob's and the ring's
renderers from fresh weights; the new renderers drew the first frames 2.5 and 3.5 dB closer but changed the rest of the
run's look (the haboob's dust with more fine structure, the ring's smoke with less change from frame to frame). A
renderer trained on from its own weights and input scales, at a lower rate and for fewer iterations, on the same samples
(the recipe's and the early ones), should move where its samples ask (the fresh states) and little elsewhere.

**The avalanche's start points.** As shipped the avalanche plays one of 16 stored first states taken at other
controls; the mean state moves a start's coarse state to the controls asked for (study M3), and brought the average
picture closer, but drew the detail further from real. Study Q's likely cause, not checked: the moved coarse state no
longer matches the fine fields stored with it. Two ways round it: drop the stored fine fields so the runtime grows them
from the moved coarse state (`msg`), or store more start points so the nearest are nearer (`s48`). (`ms` alone on the
current file is study Q's `ms_u30` bit for bit, which validation has already rejected; it is not sent again. It is
played in the diagnosis beside, §3.)

**Heavy gas's start points.** From the true state the learned gas is close to real; as shipped, from one of its 8
stored start points at other controls, it is a different layer (STUDY_Q.md §3.3). The geyser's 24 start points did
nothing for the geyser, a chaotic effect; heavy gas is a slow layer, whose start points matter longer. Candidates: the
mean state for a looping effect (`ms`: study M's own use, fitted to the time-mean coarse state of its training runs),
the same with the fine fields grown at instance start (`msg`), and 24 start points (`s24`).

**The mushroom's cap.** Study O's `u30` (30-frame windows, a second of the cap's 5) followed a run better and held the
cap together a little longer; study O and study Q found the cap's raggedness after 3 to 4 s to be the stepper's (the
detail settings and the cover do not explain it). The candidates train on windows of 60 and 90 frames (2 and 3 s), with
the anchor, since the avalanche's 60-frame fine-tune lost its first frame (-0.35 dB on validation).

## 3. Diagnosis beside (training runs, not deciding)

Before the candidates' probe, and not deciding: `nvfx_fix parts` (study Q's taking apart; training runs, salt 1, seeds
of the diagnosis block, §4.5) on the avalanche's current file, its mean state on the current file (study Q's
`ms_u30`), `msg` and `s48`, as shipped: the detail spectrum, the cover and its height at 1, 2 and 4 s; whether the
grown fine fields keep the detail that the mean state lost. And after the mushroom's fine-tunes: the cover and its
height from the true state at 1, 2 and 4 s, and strips at 3, 4 and 5 s (the cap). The diagnosis may add or drop a
candidate before the probe's second step; any change is written here, with its reason, before validation.

**The avalanche's diagnosis** (22 October, 01:00 UTC; `nvfx_fix parts`, salt-1 training runs 0 to 5, the whole 148
frames, the runtime as shipped with seeds 33,541,000 + 100 r; the data root's `q2_fix/diag/avalanche_*`). The methods
from a true state are the same frames for all four files (start points play no part there); as shipped:

| file | spectrum | motion | cover dist. | mean frame (dB) | cover at 1 / 2 / 4 s | its height at 1 / 2 / 4 s |
|---|---:|---:|---:|---:|---|---|
| real | | | | | 0.021 / 0.027 / 0.020 | 0.052 / 0.075 / 0.098 |
| current (study Q's `u30`) | 0.151 | 1.14 | 0.0119 | 29.56 | 0.015 / 0.016 / 0.010 | 0.047 / 0.060 / 0.073 |
| `ms` (its mean state; study Q's `ms_u30` bit for bit, SHA-256 `bcc04822...`) | 0.144 | 1.11 | 0.0045 | 30.21 | 0.021 / 0.025 / 0.018 | 0.050 / 0.061 / 0.070 |
| `msg` (`ms`, fine fields grown) | 0.148 | 1.12 | 0.0045 | 30.20 | 0.021 / 0.025 / 0.018 | 0.050 / 0.061 / 0.070 |
| `s48` (48 start points) | 0.144 | 1.29 | 0.0063 | 30.02 | 0.018 / 0.021 / 0.021 | 0.052 / 0.063 / 0.083 |

- **The case for `msg` does not hold.** `msg` was to keep the detail that the mean state lost by dropping the stored fine
  fields that no longer match the moved coarse state. On these runs the mean state's start points bring the cover and
  the average picture much closer, and the grown fine fields change nothing of that, but they bring the detail no closer
  than the stored ones: the spectrum distance is 0.148 against `ms`'s 0.144 (on six runs neither shows the loss study Q
  measured over 60 settings, +0.022). The mismatch between the moved coarse state and its fine fields is therefore not
  what draws the mean state's detail away, and `msg` would be study Q's rejected `ms_u30` with its fine fields
  smoothed. **`msg` is dropped before its probe.** It is not played on the probe or validated.
- **The case for `s48` holds.** Start points nearer the controls give a cloud closer to real in size and height (at 4 s
  0.021 and 0.083 against the real 0.020 and 0.098, the current 0.010 and 0.073), the average picture 0.46 dB closer and
  the detail as close as `ms`'s, with its own fine fields; its motion is further from real (1.29 against 1.14), which
  the rule's margin judges. **The avalanche goes on with `s48` alone.**

## 4. The model changes: the plan and the rule (committed before any of their numbers)

### 4.1 Candidates

Each is a file built from the shipped model by one change, with the tools named; nothing else in the file changes.
Fine-tunes run on 2 threads with the trainer's seeds 33,540,001 and up (the training runs are the recipe's, salt 1);
renderer fine-tunes on 1 thread (the result depends on the thread count).

| effect (shipped file, SHA-256) | candidate | what it is | why (the leftover) |
|---|---|---|---|
| geyser (`fx_elements/models/d/geyser.nvfx`, `43095b69...`) | `u30a` | study Q's `u30` (600 iterations of the activity stage on 30-frame windows from its weights, lr 7e-4) with the anchor at 100: a window from a true state also pays 100 times the first step's squared distance from the current model's first step on the same input (`train --activity-stage 600 --mean-stage 0 --unroll 30 --lr-ft 7e-4 --anchor 100`) | `u30`'s tracking at 1 s, without its loss at the first frame |
| | `u30al` | the same at a lower rate (`--lr-ft 2.5e-4 --anchor 100`) | the lower rate kept smoke's first frame in round 2 |
| haboob (`fx_elements/models/d/haboob.nvfx`, `b7aa8a4f...`) | `rft` | the current renderer trained on from its own weights and input scales (`refit --finetune`), 1000 iterations at lr 1e-3 (the cosine schedule from there), on study Q's `rf` samples: the recipe's 200 and 100 more from the first 100 training runs' first 15 frames (run i: frame i mod 15) | the first frame (the fresh lock) without the fresh renderer's finer dust |
| | `rfs` | the same at lr 3e-4 | a smaller move from the current renderer |
| smoke ring (`fx_air/models/d/smokering.nvfx`, `4a2188b2...`) | `rft` | as the haboob's | the first frame without the fresh renderer's calmer smoke |
| | `rfs` | as the haboob's | |
| avalanche (`q_fix/models/avalanche.nvfx`, `d6161c1d...`, study Q's `u30`) | `msg` | study Q's mean state of the avalanche (the data root's `q_fix/meanstate/avalanche_b2.txt`: fitted to the first coarse states of its 240 training runs, quadratic in the controls per block of 2 x 2 cells; `setmean`), with the start points' stored fine fields dropped (`growfine --warmup 0`): at instance start the fine fields are the moved coarse state upsampled, and the detail layer grows them from there | the mean state's average picture without the stored fine fields that no longer match the moved coarse state |
| | `s48` | 48 start points instead of 16, chosen as the recipe chooses them (`starts --count 48`: farthest-point order in control space over its 240 training runs, at the first frame, 64-pixel fine fields) | a start nearer the controls asked for, with its own fine fields |
| heavy gas (`fx_air/models/d/heavygas.nvfx`, `8d436f90...`) | `ms` | the mean state fitted to its 160 training runs' time-mean coarse state from frame 89 (`fitmean --block 2`, `setmean`): a start point (frame 150 of a run at other controls) moved by the fit's difference between the controls asked for and its own | its start points are at other controls; its endless mean picture tied the flipbook's |
| | `msg` | `ms` with the start points' fine fields dropped (`growfine --warmup 30`: grown over 30 frames at instance start, as a fire's are) | the moved coarse state with fine fields that match it |
| | `s24` | 24 start points instead of 8 (`starts --count 24`, as study Q's geyser `s24`) | a start nearer the controls |
| mushroom (`q_fix/models/mushroom.nvfx`, `c1cd25a2...`, study Q's `rf` on study O's file) | `u60a` | the stepper fine-tuned from its weights by 600 iterations of the activity stage on 60-frame windows, lr 7e-4, with the anchor at 100 (`train --activity-stage 600 --mean-stage 0 --unroll 60 --lr-ft 7e-4 --anchor 100`); its mean state and renderer unchanged | a stepper that sees two of the cap's five seconds |
| | `u90a` | the same on 90-frame windows (`--unroll 90`) | three seconds: the cap's raggedness begins at 3 to 4 s |

The trainers' seeds: geyser `u30a` 33,540,001, `u30al` 33,540,002; mushroom `u60a` 33,540,003, `u90a` 33,540,004.

### 4.2 The probe (development, training settings only)

As study Q's (STUDY_Q.md §4.2): the controls of salt-1 training runs 0 to 39 (40 settings); per setting 8 real runs (the
reference) and 8 more (the floor); 8 plays per model through the runtime as shipped (float; one-shots their whole length
from the first frame, looping effects 10 s after a 5 s warm-up of the real runs); tracking 2 runs per setting from the
true state (each from an empty grid; a looping effect after 100 frames). In two steps: the current model and **the
null** (the current model with fresh seeds) first, which fix the margins (written to
`results/experiments/q2_fix/q2_margins.csv` and committed before any candidate is played on the probe); then the
candidates, whose paired spread fixes **the number of settings N** for each effect's validation and test: the smallest
multiple of 10 at which, for every candidate of the effect and every measure that is not a target it improves on the
probe, a candidate leaving that measure truly unchanged passes its margin with probability at least 0.9, by resampling
the probe's settings and in the normal form with the SD at the upper end of its one-sided 80% interval; at most 60 (if
the cap binds, the power at 60 is stated before validation). Every candidate of §4.1 goes to validation (no selection on
the probe). The null is played afresh for every effect, the unchanged ones too, on study Q2's seeds.

### 4.3 The rule (study Q's, unchanged: its form fixed now; its numbers from the probe, committed before validation)

**Measures** (each paired over the settings, candidate minus current model): tracking from the true state at the first
frame, at 1 s and at 2 s (active PSNR; each setting's mean over its 2 runs); and the endless statistics, 8 runs pooled
per setting: the detail spectrum distance, |ln motion ratio|, the coverage and emission distances and the mean-frame
PSNR. 95% paired bootstrap with 10,000 resamples (`metrics::paired_bootstrap`, seed 2026).

**Targets**, declared with each candidate: the renderer fine-tunes (`rft`, `rfs`) target the first frame and the mean
frame; every other candidate targets tracking at 1 s and the mean frame.

A candidate **replaces** the current model when
1. it is **better on at least one of its targets** (that interval above zero), and
2. **every measure is within its margin**, its targets included: each tracking horizon and the mean-frame PSNR at least
   their margins (dB), the spectrum, motion, coverage and emission distances at most theirs.

Each **margin** is the larger of study K's at the measure's scale (tracking -0.20 dB at each horizon, the mean frame
-0.10 dB, spectrum and motion +0.02, coverage and emission 10% of the current model's mean distance on the probe, at
least 0.0001) and the null's RMS change at a setting (over the probe's 40 settings); tracking has no null (a tracked run
takes its own seed, so the same model plays the same frames) and keeps -0.20 dB.

**The choice**, on validation, among an effect's candidates that meet the rule: the largest mean-frame gain among those
better on the mean frame; if none is, the largest gain at tracking at 1 s; if none, at the first frame. An effect with
none stops there (a null, reported with its numbers). The strict reading (no interval entirely on the worse side of zero)
and study K's margins are reported beside each decision and do not decide. An interval that holds zero is reported as a
tie.

### 4.4 Validation and the test

**Validation**: study G's 10 validation settings and more drawn the same way (`validation_settings(N, N)`), N the
effect's from the probe. **The test, once per effect** with a chosen candidate: study B's 10 held-out settings and more
drawn the same way (`test_settings(N)`), with study Q2's test seeds (§4.5). (The geyser's, the haboob's, the ring's and
heavy gas's held-out settings have never been tested; the avalanche's and the mushroom's were study Q's test settings,
with other seeds; a candidate is chosen on validation alone.) Before the test the file is copied to the data root's
`q2_fix/models/<effect>.nvfx` and frozen by SHA-256 in `results/experiments/q2_fix/q2_frozen.csv`; `nvfx_ordered
--study q2 test` refuses any other file and runs once (a marker under `q2_fix`). The same tool binary for the probe,
validation and the test (the data root's `q2_fix/bin/nvfx_ordered_q2`, its SHA-256 written in §4.6 before the probe), and
the rule's numbers from a committed table, `results/experiments/q2_fix/q2_rule.csv` (an effect's N and its eight
margins; `valq2` and `testq2` refuse an effect without a row).

An effect that meets the rule on the test is **replaced**: the shared links in the data root's `library/models` and in
every scene folder holding it are repointed at the new file (the old targets logged in `q2_fix/logs/links_before.txt`);
the file goes into git at `models/2d/<name>.nvfx` with `models/SHA256SUMS` regenerated, and `models/restore.sh` against
the data root reports 0 different; every frozen hash that changes is updated in the same change (at float, pinned to the
AVX2 build with `--isa avx2`); the videos that show it are rendered again (at most 5 MB each, their rows in
videos/README.md). Reported beside, not deciding: the distance to the floor before and after, the cost, and how it
looks (strips before and after, `docs/figures/q2_*.png`).

### 4.5 Seeds

Study Q2's range, 33,500,000 to 33,999,999; per run seed base + 100 s + k (setting s, run k; k < 8, the floor's + 50):

| use | real runs (the floor: + 50) | plays | the null's plays | tracking |
|---|---:|---:|---:|---:|
| the probe (`probeq2`) | 33,500,000 | 33,510,000 | 33,520,000 | 33,530,000 |
| the trainers' seeds; diagnosis (`parts --seed-base 33541000`); figures (33,545,000 and up); videos and library or scene checks (33,550,000 to 33,554,431) | 33,540,000 to 33,554,431 (below 2^25) | | | |
| a library setting's check (`checkq2`, not a decision) | 33,560,000 | 33,570,000 | - | 33,580,000 |
| validation (`valq2`) | 33,600,000 | 33,650,000 | - | 33,680,000 |
| the test (`testq2`) | 33,700,000 | 33,750,000 | - | 33,780,000 |
| a replaced model's round test again (`d-eval --seed-base 33800000`) | 33,800,000 to 33,950,999 | | | |

### 4.6 The probe and the rule's numbers (each effect's committed before its validation)

**The binary.** `nvfx_ordered` built (Release, g++ 14) after the quiet machine's marker (21:19 UTC) from adb9404, whose
tool and trainer code are c6e56e9's (the commits between change only this document): the data root's
`q2_fix/bin/nvfx_ordered_q2`, SHA-256 `420131ac5e3455943401bda68199660191558cacbc8deb568a48bbe551dd265a`, for the
candidates, the probe, validation and the test; `nvfx_fix_q2` (`be2d773a...`) for the diagnosis and the figures.

The probe's tables: [q2_margins.csv](../results/experiments/q2_fix/q2_margins.csv) (each effect's null, committed before
its candidates were played on the probe), [q2_power.csv](../results/experiments/q2_fix/q2_power.csv) and
`q2_probe_*.csv`; the numbers each effect's validation and test use: [q2_rule.csv](../results/experiments/q2_fix/q2_rule.csv)
(written by `jobs/rule.sh` from the two tables by §4.2's rule, which gives study Q's N for all six of its effects from
study Q's tables).

**The haboob** (the null 22:50 UTC, 7a02408; the rule 23:13, 693834e). The null (the current model with fresh seeds, 40
training settings) moves its spectrum distance at a setting by 0.021 (RMS), its motion by 6.4%, its coverage distance by
0.0024 and its mean frame by 0.51 dB: margins tracking -0.20 dB at each horizon, spectrum **+0.0213**, |ln motion ratio|
**+0.0644**, coverage **+0.0024**, emission **+0.0001** (study K's), mean frame **-0.51 dB**. On the probe (candidate minus
current, 95% intervals):
- `rft` (the renderer trained on at lr 1e-3; its PSNR on its samples' pixels 34.8 dB, study Q's fresh `rf` 30.9): the first
  frame **+3.03 dB [+2.96, +3.10]** (28.08 → 31.12 dB; a target), at 1 s +0.04 [+0.01, +0.06], the spectrum distance
  +0.003 [+0.002, +0.004] (0.101 → 0.104; `rf`'s was +0.010 on study Q's probe), its motion error +0.005 [+0.001, +0.010],
  at 2 s -0.02 [-0.03, -0.00] and the mean frame -0.04 [-0.05, -0.03] (each worse, each a tenth of its margin or less),
  coverage and emission closer: it meets the rule's form.
- `rfs` (lr 3e-4; 34.3 dB on its samples): the first frame **+2.30 [+2.25, +2.36]**, at 1 s +0.07 [+0.06, +0.09], the
  spectrum +0.006 [+0.005, +0.007], its motion error +0.004 [+0.000, +0.007], the mean frame a tie: it meets the rule's
  form.

Every measure that is not a target it improves passes its margin, if truly unchanged, with probability 1.00 at 10
settings by resampling and needs at most 2 in the normal form at the SD's upper 80% bound: **N = 10**.

**The geyser** (the null 23:05 UTC, 7d3d795; the rule 23:35, 781184b). An erupting geyser is chaotic, and the null shows
it again: it moves the spectrum distance at a setting by 0.174 (RMS), the motion by 26%, the coverage distance by 0.0076
and the mean frame by 3.05 dB. Margins: tracking -0.20 dB at each horizon, spectrum **+0.174**, |ln motion ratio|
**+0.255**, coverage **+0.0076**, emission **+0.0001** (study K's), mean frame **-3.05 dB**. On the probe:
- `u30a` (lr 7e-4 with the anchor): tracking at 1 s **+0.28 dB [+0.13, +0.44]** (a target; 20.74 → 21.02 dB), at 2 s +0.22
  [-0.02, +0.44], **the first frame a tie, +0.04 [-0.01, +0.09]** (study Q's `u30` on its probe: -0.09 [-0.13, -0.05]),
  every endless measure a tie (the mean frame +0.00 [-0.09, +0.09]): it meets the rule's form.
- `u30al` (lr 2.5e-4 with the anchor): the first frame +0.06 [+0.03, +0.09] (better, not a target), at 1 s a tie (+0.02
  [-0.13, +0.18]), the mean frame -0.07 [-0.17, +0.03] (a tie): better on no target.

The number of settings: tracking at 2 s, which `u30a` improves but which is not its target, spreads widely (an SD of 0.76
dB between settings): if truly unchanged it would pass its margin with probability 0.55 at 60 settings by resampling and
needs 187 in the normal form (`u30al`'s: 0.71, 134; its tracking at 1 s 0.92, 77). The cap binds: **N = 60**, at which
the rule's form passes with probability 0.55 (`u30a`) and 0.66 (`u30al`) if every held measure is truly unchanged and the
improved targets are as measured (q2_power.csv), stated here before any of the geyser's validation numbers.

**The smoke ring** (the null 00:36 UTC, f878700; the rule 00:48, its row in q2_rule.csv). The null moves its spectrum
distance at a setting by 0.044 (RMS), its motion by 2.1%, its coverage distance by 0.00029 and its mean frame by 2.10
dB: margins tracking -0.20 dB, spectrum **+0.0437**, |ln motion ratio| **+0.0208**, coverage **+0.00029**, emission
**+0.0001** (study K's), mean frame **-2.10 dB**. On the probe:
- `rft` (41.1 dB on its samples' pixels; study Q's fresh `rf` 39.8): the first frame **+2.98 dB [+2.82, +3.15]** (25.08 →
  28.06 dB) and the mean frame **+0.05 [+0.01, +0.08]** (both targets better), at 1 s +0.13 [+0.05, +0.20], the spectrum
  distance -0.025 [-0.029, -0.021] (0.716 → 0.691), coverage and emission closer, at 2 s a tie, and **its motion error a
  tie, +0.003 [-0.004, +0.009]** (study Q's `rf` on its probe: +0.011 [+0.001, +0.021]): it meets the rule's form.
- `rfs` (39.9 dB): the first frame **+2.19 [+2.02, +2.36]**, the spectrum -0.026, coverage and emission closer, the rest
  ties (the motion +0.001 [-0.004, +0.006]): it meets the rule's form.

Every measure that is not a target it improves passes its margin, if truly unchanged, with probability at least 0.98 at
20 settings by resampling and needs at most 19 in the normal form (at 10, `rft`'s tracking at 1 s would pass with
probability 0.83): **N = 20**, at
which the rule's form passes with probability 0.98 (`rft`) and 1.00 (`rfs`).

**The avalanche** (the null 01:47 UTC; the rule 01:51; `s48` alone, §3). The null (the current file, study Q's `u30`,
with fresh seeds) moves its spectrum distance at a setting by 0.035 (RMS), its motion by 21%, its coverage distance by
0.012 and its mean frame by 2.20 dB: margins tracking -0.20 dB, spectrum **+0.0346**, |ln motion ratio| **+0.209**,
coverage **+0.0118**, emission **+0.0001** (study K's), mean frame **-2.20 dB**. `s48` (805,951 bytes: 48 start points
with their 64-pixel fine fields, against 16) on the probe: the same frames from a true state (tracking the same at every
horizon); in endless play the mean frame **+2.23 dB [+1.18, +3.23]** (28.52 → 30.76 dB; a target), the spectrum
distance -0.024 [-0.039, -0.011] (0.129 → 0.105) and the motion error -0.093 [-0.180, -0.009] (the motion ratio 1.27 →
1.15) better, coverage a tie: it meets the rule's form. Every held measure passes its margin, if truly unchanged, with
probability at least 0.99 at 30 settings by resampling and needs at most 24 in the normal form (the motion; at 20 it
would need more): **N = 30**, at which the rule's form passes with probability 0.95.

**Heavy gas** (the null 00:50 UTC; the rule 02:20). The null moves its spectrum distance at a setting by 0.020 (RMS),
its motion by 3.9%, its coverage distance by 0.0021 and its mean frame by 1.03 dB: margins tracking -0.20 dB, spectrum
**+0.0200**, |ln motion ratio| **+0.0389**, coverage **+0.0021**, emission **+0.0001** (study K's), mean frame **-1.03
dB**. Tracking from a true state is the same frames for all three (start points play no part there). In endless play on
the probe:
- `ms`: the mean frame **+1.66 dB [+1.26, +2.07]** (25.07 → 26.73 dB; a target) and coverage closer, but its motion
  error **+0.074 [+0.040, +0.108]** (the motion ratio 1.13 → 1.29), past its margin: on the probe it does not meet the
  rule's form.
- `msg`: the mean frame **+1.87 [+1.49, +2.26]** (→ 26.94 dB), the spectrum distance -0.022 [-0.032, -0.012] (0.094 →
  0.072), its motion error -0.024 [-0.042, -0.004] and coverage -0.0039 better, emission +0.00003 (worse, a third of its
  margin): it meets the rule's form. The mean state's start moved to the controls, with fine fields grown from it, is
  closer on every endless measure; with the stored fine fields (`ms`) the layer moves too much.
- `s24`: the mean frame **+0.73 [+0.20, +1.24]**, the spectrum +0.013 [-0.004, +0.032], its end past its margin: on the
  probe it does not meet the rule's form.

The number of settings: `ms`'s motion (an SD of 0.11 between settings) and `s24`'s spectrum, motion and coverage, if
truly unchanged, would pass their margins with probability 0.78, 0.73, 0.86 and 0.86 at 60 settings by resampling and
need 105, 115, 89 and 87 in the normal form. The cap binds: **N = 60**, at which the rule's form passes with probability
0.78 (`ms`), 0.99 (`msg`) and 0.65 (`s24`) if every held measure is truly unchanged and the improved targets are as
measured (q2_power.csv), stated here before any of heavy gas's validation numbers.

**The mushroom** (the null 01:47 UTC; the rule 02:20). The null moves its spectrum distance at a setting by 0.013 (RMS),
its motion by 3.9%, its coverage distance by 0.0060 and its mean frame by 1.23 dB: margins tracking -0.20 dB, spectrum
**+0.02** (study K's), |ln motion ratio| **+0.0394**, coverage **+0.0060**, emission **+0.0001**, mean frame **-1.23
dB**. On the probe:
- `u60a`: tracking at 1 s **+0.70 dB [+0.60, +0.79]** (14.78 → 15.48 dB) and the mean frame **+1.79 [+1.28, +2.30]**
  (28.62 → 30.41 dB; both targets better), at 2 s +0.81 [+0.68, +0.94], the first frame a tie (-0.01 [-0.02, +0.01];
  the anchor held it), the spectrum -0.006 and coverage -0.0043 better, its motion error +0.017 [+0.011, +0.023] (the
  motion ratio 0.87 → 0.86; worse, within its margin): it meets the rule's form.
- `u90a`: tracking at 1 s **+0.85 [+0.73, +0.96]**, the mean frame **+1.82 [+1.25, +2.39]**, at 2 s +0.94, the first
  frame +0.05 [+0.04, +0.06] (better), the spectrum -0.007 and coverage -0.0046 better, its motion error +0.022 [+0.015,
  +0.028] (worse, within its margin): it meets the rule's form.

The number of settings: tracking at 2 s, which both improve but which is not their target, if truly unchanged would pass
its margin with probability 0.95 and 0.93 at 60 settings by resampling and needs 59 and 70 in the normal form. The cap
binds: **N = 60**, at which the rule's form passes with probability 0.95 (`u60a`) and 0.92 (`u90a`), stated here before
any of the mushroom's validation numbers.

### 4.7 Validation (the choice)

Candidate minus current model on study G's validation settings extended to the effect's N, study Q2's seeds, 8 runs
pooled per setting, tracking 2 runs per setting ([q2_val_compareq.csv](../results/experiments/q2_fix/q2_val_compareq.csv),
[q2_val_choice.csv](../results/experiments/q2_fix/q2_val_choice.csv)):

**The haboob (10 settings): both meet the rule; `rft` is chosen.**

| candidate | tracking: first frame (target) | at 1 s | at 2 s | spectrum | \|ln motion\| | coverage | mean frame (target) | reading |
|---|---|---|---|---|---|---|---|---|
| `rft` | **+3.02 [+2.82, +3.25]** | +0.01 [-0.03, +0.06] | -0.03 [-0.06, -0.01] | +0.004 [+0.002, +0.007] (margin +0.021) | +0.004 [-0.005, +0.013] | -0.0007 [-0.0011, -0.0003] | -0.04 [-0.06, -0.01] (margin -0.51) | **meets it; chosen** |
| `rfs` | **+2.30 [+2.17, +2.45]** | +0.06 [+0.02, +0.08] | -0.02 [-0.04, +0.00] | +0.007 [+0.005, +0.010] | +0.003 [-0.004, +0.009] | -0.0005 [-0.0008, -0.0003] | -0.00 [-0.03, +0.02] | meets it |

Both draw the first frame much closer (28.65 → 31.67 and 30.95 dB) and keep the dust's detail within its margin: the
spectrum distance moves by +0.004 and +0.007 (0.095 → 0.099 and 0.102), where study Q's fresh refit had moved it by
+0.013 [+0.006, +0.022] on study Q's validation. Neither is better on the mean frame (`rft` -0.04, a twenty-fifth of its
margin; `rfs` a tie), and their targets do not include tracking at 1 s, so the rule takes the larger first-frame gain:
`rft`. By study M's strict reading `rft` has three intervals wholly on the worse side (tracking at 2 s, the spectrum, the
mean frame; each a tenth of its margin or less), `rfs` one (the spectrum): reported, not deciding.

**Frozen** before its test ([q2_frozen.csv](../results/experiments/q2_fix/q2_frozen.csv), 5d3aa2f): the data root's
`q2_fix/models/haboob.nvfx` (read-only; 301,443 bytes, the size of the old file, SHA-256
`4f933a2fe7a226cc9aee138722bdf2b358dbf6330952392860b46135efdc0f23`), validated and tested by `nvfx_ordered_q2`.

**The smoke ring (20 settings): both meet the rule; `rft` is chosen.**

| candidate | tracking: first frame (target) | at 1 s | at 2 s | spectrum | \|ln motion\| | coverage | mean frame (target) | reading |
|---|---|---|---|---|---|---|---|---|
| `rft` | **+2.94 [+2.72, +3.18]** | +0.10 [-0.01, +0.21] | -0.00 [-0.06, +0.06] | **-0.022 [-0.025, -0.018]** | +0.013 [+0.007, +0.018] (margin +0.021) | -0.0001 | +0.05 [-0.01, +0.12] | **meets it; chosen** |
| `rfs` | **+2.14 [+1.91, +2.39]** | -0.02 [-0.11, +0.08] | +0.01 [-0.04, +0.06] | **-0.025 [-0.026, -0.023]** | +0.011 [+0.006, +0.015] | -0.0001 | +0.01 [-0.04, +0.07] | meets it |

Both draw the ring's first frame much closer (25.38 → 28.32 and 27.52 dB) and its detail closer through the run (the
spectrum distance 0.704 → 0.682 and 0.679). Their motion error grows by +0.013 and +0.011 (the motion ratio 0.962 →
0.945 and 0.947 of the real runs'), each interval's worse end within its margin of +0.021, where study Q's fresh refit
had grown it by +0.023 [+0.015, +0.030] on study Q's validation and stopped: the renderer trained on from its own
weights still draws the smoke with a little less change from frame to frame, by half as much. Neither is better on the
mean frame (ties), so the rule takes the larger first-frame gain: `rft`. By study M's strict reading one interval each
lies wholly on the worse side (the motion): reported, not deciding.

**Frozen** before its test (cab84db): the data root's `q2_fix/models/smokering.nvfx` (read-only; 301,443 bytes, SHA-256
`a36e5e385f249dff9df8aae6d8c7c61b0a0a3b573c6f49531df76caf5d18598d`), validated and tested by `nvfx_ordered_q2`.

**The avalanche (30 settings): `s48` meets the rule and is chosen.** Tracking from a true state is the same frames (start
points play no part there); in endless play its average picture is **+2.77 dB [+1.86, +3.70]** closer to a real run's
(29.67 → 32.45 dB; a target), its detail **-0.019 [-0.039, -0.002]** (0.133 → 0.113), its motion error **-0.143
[-0.234, -0.051]** and its coverage distance **-0.0041 [-0.0066, -0.0016]** (0.0093 → 0.0053) closer, emission a tie
(+0.00001 [-0.00000, +0.00001]). Nothing is worse by any reading. Start points nearer the controls asked for, each with
its own fine fields, give the cloud the size and place a run at those controls has, which the mean state did by moving
a start (study Q: +0.84 dB) and the 48 starts do better, without the mean state's cost in detail.

**Frozen** before its test (48cae3c): the data root's `q2_fix/models/avalanche.nvfx` (read-only; 805,951 bytes, SHA-256
`e457d083dae15fef6b720110655500e7f37d7681307c697a5e05a351e26eaa0b`), validated and tested by `nvfx_ordered_q2`.

### 4.8 The test (once)

Study B's held-out settings extended to the effect's N (`test_settings(N)`), the test's seeds (real 33,700,000 + 100 s +
k, the floor + 50, plays 33,750,000, tracking 33,780,000), the file frozen before it ran, the binary of validation, once
(the marker `q2_fix/test_<effect>.done`; [q2_test_compareq.csv](../results/experiments/q2_fix/q2_test_compareq.csv),
[q2_test_decision.csv](../results/experiments/q2_fix/q2_test_decision.csv), `q2_test_*.csv`). Candidate minus current
model, 95% paired bootstrap intervals with 10,000 resamples:

**The haboob, `rft` (10 settings): it meets the rule, and is replaced.**

| measure | current → `rft` | difference [95% interval] | margin | reading |
|---|---|---|---|---|
| tracking, the first frame (dB) | 28.43 → 31.47 | **+3.04 [+2.86, +3.21]** | -0.20 | better (a target) |
| tracking at 1 s (dB) | 18.80 → 18.79 | -0.01 [-0.04, +0.03] | -0.20 | tie |
| tracking at 2 s (dB) | 16.48 → 16.46 | -0.01 [-0.05, +0.02] | -0.20 | tie |
| detail spectrum distance | 0.112 → 0.116 | +0.004 [+0.002, +0.006] | +0.021 | worse, within its margin (a sixth of it) |
| \|ln motion ratio\| (motion ratio) | 0.189 (0.916) → 0.192 (0.913) | +0.003 [-0.004, +0.010] | +0.064 | tie |
| coverage distance | 0.0089 → 0.0084 | **-0.0005 [-0.0010, -0.0001]** | +0.0024 | better |
| emission distance | 0.00006 → 0.00006 | -0.000000 [-0.000000, +0.000000] | +0.0001 | tie |
| mean-frame PSNR (dB) | 29.67 → 29.64 | -0.03 [-0.08, +0.01] | -0.51 | tie (a target) |

The renderer trained on from its own weights draws the haboob's first frame 3.0 dB closer to the simulator's on the true
fields (study Q's fresh refit: +2.55 on its validation), and leaves the rest of the run as it was: tracking at 1 and 2 s,
motion, light and the average picture tie, the cover is a little closer, and the dust's detail moves by +0.004 (a sixth
of its margin; study Q's refit moved it by +0.013 and stopped). By study M's strict reading one interval lies wholly on
the worse side (the spectrum). Distance to the floor (a second set of 8 real runs per setting): the mean frame 6.58 →
6.61 dB below it (the floor 36.25 dB), the spectrum distance 0.112 → 0.116 (the floor's 0.029), the coverage distance
0.0089 → 0.0084 (0.0041).

**The smoke ring, `rft` (20 settings): it meets the rule, and is replaced.**

| measure | current → `rft` | difference [95% interval] | margin | reading |
|---|---|---|---|---|
| tracking, the first frame (dB) | 25.20 → 28.14 | **+2.94 [+2.72, +3.16]** | -0.20 | better (a target) |
| tracking at 1 s (dB) | 29.79 → 29.91 | **+0.11 [+0.01, +0.23]** | -0.20 | better |
| tracking at 2 s (dB) | 27.29 → 27.25 | -0.05 [-0.12, +0.04] | -0.20 | tie |
| detail spectrum distance | 0.705 → 0.681 | **-0.024 [-0.030, -0.018]** | +0.044 | better |
| \|ln motion ratio\| (motion ratio) | 0.034 (0.990) → 0.042 (0.971) | +0.009 [+0.002, +0.015] | +0.021 | worse, within its margin |
| coverage distance | 0.00142 → 0.00130 | **-0.00011 [-0.00018, -0.00005]** | +0.00029 | better |
| emission distance | 0.00012 → 0.00009 | **-0.00004 [-0.00005, -0.00003]** | +0.0001 | better |
| mean-frame PSNR (dB) | 41.45 → 41.50 | +0.05 [-0.02, +0.12] | -2.10 | tie (a target) |

The renderer trained on from its own weights draws the ring's first frame 2.9 dB closer, follows a run a little better
at 1 s, and draws its detail, cover and light closer to real through the run; its smoke changes a little less from frame
to frame (the motion ratio 0.990 → 0.971 of the real runs'; the interval's worse end, +0.015, three quarters of its
margin; study Q's fresh refit had +0.023 [+0.015, +0.030] on its validation). By study M's strict reading one interval
lies wholly on the worse side (the motion). Distance to the floor: the mean frame 3.27 → 3.22 dB below it (the floor
44.72 dB), the spectrum distance 0.705 → **0.681** (the floor's 0.017), the coverage distance 0.0014 → 0.0013 (0.0003).

**The avalanche, `s48` (30 settings): it meets the rule, and is replaced.**

| measure | current → `s48` | difference [95% interval] | margin | reading |
|---|---|---|---|---|
| tracking, the first frame, at 1 s and at 2 s (dB) | 24.43, 13.41, 11.88 (the same) | 0 [0, 0] | -0.20 | the same frames (start points play no part) |
| detail spectrum distance | 0.111 → 0.112 | +0.002 [-0.010, +0.013] | +0.035 | tie |
| \|ln motion ratio\| (motion ratio) | 0.307 (1.06) → 0.331 (1.13) | +0.024 [-0.070, +0.117] | +0.209 | tie (its worse end past study K's +0.02) |
| coverage distance | 0.0138 → 0.0121 | -0.0017 [-0.0064, +0.0026] | +0.0118 | tie |
| emission distance | 0.00010 → 0.00010 | +0.000000 [-0.000010, +0.000010] | +0.0001 | tie |
| mean-frame PSNR (dB) | 29.25 → 30.66 | **+1.41 [+0.36, +2.50]** | -2.20 | better (a target) |

With 48 start points the avalanche plays, from a start nearer the controls asked for, a cloud whose average picture is
1.4 dB closer to a real run's on held-out settings (2.8 dB on validation), with every other measure a tie. Its motion
and coverage, better on validation, tie here, with intervals as wide as the null's spread (by study K's margins the
motion's would not pass; reported, not deciding). Distance to the floor: the mean frame 11.85 → **10.44 dB** below it
(the floor 41.10 dB), the spectrum distance 0.111 → 0.112 (the floor's 0.027), the coverage distance 0.0138 → 0.0121
(0.0013): the avalanche is still far from real, but a start point is now much less of that distance. The file is 2.9
times the size (805,951 bytes against 280,927: 32 more start points with their fine fields); a frame costs the same.

## 5. What changed

(Written when the study ends.)

## 6. Reproduce

The data root's `q2_fix/jobs/*.sh` run every step (resumable, a marker `jobs/*.done` for each); in short, with
`B=q2_fix/bin/nvfx_ordered_q2`, `D` the data root and `R` this folder's `results/experiments/q2_fix`, each `nice -n 10`:

```
# the candidates (§4.1)
$B --study q2 --root $D --threads 2 train --effect geyser --model $D/fx_elements/models/d/geyser.nvfx --out cand/geyser/u30a.nvfx --activity-stage 600 --mean-stage 0 --unroll 30 --lr-ft 7e-4 --anchor 100 --seed 33540001
$B --study q2 --threads 1 refit --effect haboob --model $D/fx_elements/models/d/haboob.nvfx --early 15 --finetune --iterations 1000 --lr 1e-3 --out cand/haboob/rft.nvfx
$B setmean --model $D/q_fix/models/avalanche.nvfx --meanstate $D/q_fix/meanstate/avalanche_b2.txt --out cand/avalanche/ms.nvfx
$B growfine --model cand/avalanche/ms.nvfx --warmup 0 --out cand/avalanche/msg.nvfx
$B --study q2 --root $D --threads 2 starts --effect avalanche --model $D/q_fix/models/avalanche.nvfx --count 48 --out cand/avalanche/s48.nvfx
$B --study q2 --root $D --threads 2 fitmean --effect heavygas --block 2
# the probe: the null first (its margins committed), then the candidates (q2_power.csv)
$B --study q2 --threads 1 --results $R val --effect E --split probeq2 --current FILE --null --prefix q2_probe
$B --study q2 --threads 1 --results $R val --effect E --split probeq2 --current FILE --cand NAME=F,... --null --prefix q2_probe
# validation and the test, by the committed rule (q2_rule.csv); the chosen file frozen first
$B --study q2 --threads 2 --results $R --rule $R/q2_rule.csv val --effect E --split valq2 --current FILE --cand NAME=F,... --prefix q2_val
$B --study q2 --threads 2 --results $R --rule $R/q2_rule.csv test --effect E --current FILE --cand NAME=$D/q2_fix/models/E.nvfx --frozen $R/q2_frozen.csv
```

## 7. Checks

(Written when the study ends.)

## 8. What is left

(Written when the study ends.)
