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

Planned (21 October 2026, 18:30 UTC). The plan and the rule's form (§4) are committed; no number of the study exists
yet. Heavy work waits for the quiet machine: another agent re-times everything (the data root's `quiet7/QUIET_DONE`
marks its end; at the latest 20:40 UTC, three hours after this study began). Until then only reading, planning and the
code (syntax-checked, not built) at `nice -n 19`.

**How to pick it up.** Every job is a script in the data root's `q2_fix/jobs/` that skips what is done (a marker
`jobs/<step>.done`, or the file it makes); its log is in `q2_fix/logs/`. The order (§4.6): build, the candidates and the
null probes; each effect's margins committed and pushed (`jobs/margins_<effect>.committed`) before its candidates are
played on the probe; the rule's numbers (`q2_rule.csv`) committed and pushed before validation; the chosen file frozen
(`q2_frozen.csv`) and pushed before its test, which runs once (`q2_fix/test_<effect>.done`).

| leftover | effect (shipped file) | candidates (§4.1) | where it is |
|---|---|---|---|
| 1 | geyser (`fx_elements/models/d/geyser.nvfx`) | `u30a`, `u30al` | planned |
| 2 | haboob (`fx_elements/models/d/haboob.nvfx`) | `rft`, `rfs` | planned |
| 3 | smoke ring (`fx_air/models/d/smokering.nvfx`) | `rft`, `rfs` | planned |
| 4 | avalanche (`q_fix/models/avalanche.nvfx`) | `msg`, `s48` | planned |
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

(Filled in as the probe runs.)

### 4.7 Validation (the choice)

(Filled in after validation.)

### 4.8 The test (once)

(Filled in after the tests.)

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
