# Study Q: the weak spots of rounds 9 and 10, fixed and measured

Status: **under way** (21 October 2026). Audience: dev, research, artists. The plan of the model changes and the form of
their rule (§4) were written and committed before any of the study's numbers; the rule's numbers (the margins and the
number of settings) come from a probe on training settings and are committed before validation; each chosen file is
frozen by its SHA-256 before its one test. [REPORT.md](REPORT.md) §21 has the summary.

Study Q does for the weak spots of rounds 9 and 10 ([EFFECTS_EARTH.md](EFFECTS_EARTH.md),
[EFFECTS_AIR.md](EFFECTS_AIR.md), [EFFECTS_FIRE2.md](EFFECTS_FIRE2.md), [EFFECTS_ELEMENTS.md](EFFECTS_ELEMENTS.md)) what
study O did for earlier ones ([REPORT.md](REPORT.md) §16): each weak spot is taken apart first, to find which part makes
it (the learned stepper, the detail layer, the learned renderer, the way it is played); a fix in how an effect is played
(its detail settings, passes, tiles or look in the library and the scenes) is judged by eye from frames before and after,
with a check on training settings beside it; a fix that changes a model goes through a rule written before validation,
with margins sized by a null probe (study M3's lesson, §14.9), validated, frozen and tested once on held-out settings with
fresh seeds, and replaces a shipped model only if it passes.

Seeds: study Q's range, 33,000,000 to 33,499,999 (§4.5). Code: `tools/nvfx_fix.cpp` (`parts`, the taking apart) and
`tools/nvfx_ordered.cpp` (`--study q`: `refit`, `starts`, `train`, `fitmean`, `setmean`, `val --split probeq | valq |
checkq`, `test`). Every file is under the data root's `q_fix` (outside git); tables under `results/experiments/q_fix`.

## 1. Bottom line

(Written when the study ends.)

## 2. The weak spots

From the rounds' documents, worst first:

| effect | weak spot | the round's own reading |
|---|---|---|
| avalanche | follows a held-out run 2.2 dB worse than the coarse simulation with the same detail at 1 s; its learned cloud grows and billows less than the simulator's | weak (EFFECTS_EARTH.md §4) |
| geyser | ties the coarse simulation at 1 s and loses at 2 s; within a shard it follows its start point more than its intensity | weak (EFFECTS_ELEMENTS.md §4, §4.7) |
| haboob | its first frame is lost to its renderer (-11.7 dB); its mean picture ties the flipbook's | falls short (EFFECTS_ELEMENTS.md §4) |
| crown fire, window fire | vertical streaks in the fine detail (spectrum distance 0.43 and 0.44, six times the flipbook's) | work (EFFECTS_FIRE2.md §4) |
| heavy gas | its mean picture ties the flipbook's; its rolling fronts come out as horizontal streaks | falls short (EFFECTS_AIR.md §4) |
| smoke ring | its fine detail is the least real of any effect (spectrum distance 0.72); its first frame is 4.8 dB behind | works (EFFECTS_AIR.md §4) |
| dust explosion | its renderer fringes the unlit cloud and draws the pall faint | works (EFFECTS_FIRE2.md §4.5) |
| candle | a ragged, torn flame up close; blue-green fringes on the smoke | works (EFFECTS_AIR.md §4.6) |
| mushroom | the cap turns ragged after 3 to 4 s (study O's model) | weak, then replaced by study O (REPORT §16.3) |

## 3. Diagnosis

(Each weak spot taken apart with `nvfx_fix parts`: on training runs (salt 1, diagnosis only) the real frames against the
learned effect with one part at a time swapped: as shipped; from the run's true state (the learned stepper and detail
layer and renderer); the same states drawn by the simulator's renderer; the learned coarse state alone; the run's true
coarse states moving the detail layer (a perfect stepper); the learned renderer on the run's true fields; and, per frame
from a run's first, the learned renderer on the true fields.)

## 4. The model changes: the plan and the rule (committed before any of their numbers)

### 4.1 Candidates

Each is a file built from the shipped model by one change, with the tools named; nothing else in the file changes.

| effect (shipped file, SHA-256) | candidate | what it is | why (the weak spot) |
|---|---|---|---|
| avalanche (`fx_earth/models/d/avalanche.nvfx`, `77bf9d9d...`) | `ms` | study M3's mean state: fitted to the first coarse states of its 240 training runs (salt 1), quadratic in the controls per block of 2 x 2 cells (`fitmean --block 2`, `setmean`), so a start point stored at other controls is moved to the requested ones | its 16 start points are at other intensities; its endless average picture lost to the flipbook by 2.9 dB |
| | `u30` | the stepper fine-tuned from its weights by 600 iterations of the activity stage on windows of 30 frames (`train --activity-stage 600 --mean-stage 0 --unroll 30 --lr-ft 7e-4`), as study O's mushroom `u30` | it falls behind from 1 s: the cloud's growth is a slow, cumulative process the 16-frame windows of training see little of |
| | `u60` | the same on windows of 60 frames (`--unroll 60`) | the growth lasts two to three seconds |
| | `ms_u30` | `u30` with the mean state | both |
| geyser (`fx_elements/models/d/geyser.nvfx`, `43095b69...`) | `u30` | as the avalanche's `u30` | it ties at 1 s and loses at 2 s |
| | `s24` | 24 start points instead of 8, taken from its training runs as the recipe takes them (`starts --count 24`: farthest-point order in control space, so its first 8 are the current ones; frame 150, 64-pixel fine fields) | a shard plays one of the three start points nearest the controls, and with 8 of them those are far off: its column follows the start point more than its intensity |
| | `s24_u30` | `u30` with the 24 start points | both |
| haboob (`fx_elements/models/d/haboob.nvfx`, `b7aa8a4f...`) | `rf` | the renderer trained again as the recipe trains it (fresh weights, the recipe's options), on the recipe's 200 samples and 100 more, one from each of the first 100 training runs at one of its first 15 frames (run i: frame i mod 15; `refit --early 15`) | the recipe's samples are frames 10 to length - 10 of each run: a one-shot's first frames (the fresh lock) are never among them |
| smoke ring (`fx_air/models/d/smokering.nvfx`, `4a2188b2...`) | `rf` | as the haboob's | its first frame is 4.8 dB behind, lost to the renderer |
| dust explosion (`fx_fire2/models/d/dustexplosion.nvfx`, `65057a10...`) | `rf` | as the haboob's | its renderer fringes the unlit cloud (its first frames) and draws the pall faint |

The diagnosis (§3) may add or drop a candidate before the probe's second step; any change is written here, with its
reason, before validation. The fine-tunes run on 2 threads with seeds 33,470,001 and up (the trainer's own; the training
runs are the recipe's, salt 1); the renderer refits on 1 thread (the result depends on the thread count).

### 4.2 The probe (development, training settings only)

As study M3's and study O's (REPORT §14.9.1, §16.3): the controls of salt-1 training runs 0 to 39 (40 settings); per
setting 8 real runs (the reference) and 8 more (the floor); 8 plays per model through the runtime as shipped (float, the
frozen tests' precision; one-shots their whole length from the first frame, looping effects 10 s after a 5 s warm-up of
the real runs); tracking 2 runs per setting from the true state (each from an empty grid; a looping effect after 100
frames), through the runtime. In two steps: the current model and **the null** (the current model with fresh seeds)
first, which fix the margins (written to `results/experiments/q_fix/q_margins.csv` and committed before any candidate is
played on the probe); then the candidates, whose paired spread fixes **the number of settings N** for each effect's
validation and test: the smallest multiple of 10 at which, for every candidate of the effect and every measure that is not
a target it improves on the probe, a candidate leaving that measure truly unchanged passes its margin with probability at
least 0.9, by resampling the probe's settings and in the normal form with the SD at the upper end of its one-sided 80%
interval; at most 60 (if the cap binds, the power at 60 is stated before validation).

### 4.3 The rule (its form fixed now; its numbers from the probe, committed before validation)

**Measures** (each paired over the settings, candidate minus current model): tracking from the true state at the first
frame, at 1 s and at 2 s (active PSNR; each setting's mean over its 2 runs); and the endless statistics of studies M, M2
and M3, 8 runs pooled per setting: the detail spectrum distance, |ln motion ratio|, the coverage and emission distances
and the mean-frame PSNR. 95% paired bootstrap with 10,000 resamples (`metrics::paired_bootstrap`, seed 2026).

**Targets**, declared with each candidate (§4.1): the renderer refits (`rf`) target the first frame and the mean
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
and study K's margins are reported beside each decision and do not decide.

### 4.4 Validation and the test

**Validation**: study G's 10 validation settings and more drawn the same way (`validation_settings(N, N)`), N the
effect's from the probe. **The test, once per effect** with a chosen candidate: study B's 10 held-out settings and more
drawn the same way (`test_settings(N)`). Before it the file is copied to the data root's `q_fix/models/<effect>.nvfx` and
frozen by SHA-256 in `results/experiments/q_fix/q_frozen.csv`; `nvfx_ordered --study q test` refuses any other file and
runs once (a marker under `q_fix`). The same tool binary for the probe, validation and the test.

An effect that meets the rule on the test is **replaced**: the shared links in the data root's `library/models` and in
every scene folder holding it are repointed at the new file (the old targets logged in `q_fix/logs/links_before.txt`),
after the full ctest has passed with the new links; every frozen hash that changes is updated in the same change (at
float, pinned to the AVX2 build with `--isa avx2`), and the videos that show it are rendered again. Reported beside, not
deciding: the distance to the floor before and after, the cost, and for a replaced model its round's own test (`d-eval`,
tracking against the coarse simulation with detail, endless against the nearest flipbook) run again with fresh seeds.

### 4.5 Seeds

Study Q's range, 33,000,000 to 33,499,999 (library and script seeds stay below 2^25 = 33,554,432); per run seed
base + 100 s + k (setting s, run k):

| use | real runs (the floor: + 50) | plays | the null's plays | tracking |
|---|---:|---:|---:|---:|
| the probe (`probeq`) | 33,000,000 | 33,010,000 | 33,020,000 | 33,030,000 |
| a library setting's check (`checkq`, not a decision) | 33,040,000 | 33,050,000 | - | 33,060,000 |
| validation (`valq`) | 33,100,000 | 33,150,000 | - | 33,180,000 |
| the test (`testq`) | 33,200,000 | 33,250,000 | - | 33,280,000 |
| a replaced model's round test again (`d-eval --seed-base 33300000`) | 33,300,000 to 33,450,999 | | | |
| diagnosis, figures, the trainers' seeds | 33,460,000 to 33,499,999 | | | |

## 5. Fixes with no model change

(Written as the work goes.)

## 6. Reproduce

(Written when the study ends.)
