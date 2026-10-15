# Quiet-machine re-timing (10 October 2026)

Every timing that was measured on a busy machine during the later studies, re-run one job at a time with no other
jobs (load average 1.0 to 2.9, mostly the runs themselves). `sections.log` has the load before each section.

| file | what | used in |
|---|---|---|
| `fireball_round2_table.txt` | the fireball, round 1 (commit d41adff) against round 2, 8 configurations, least of 3 interleaved runs' medians (`tools/opt2/table.sh`) | `docs/COMPOSE.md` §7.3 |
| `h2_fireball_pairs.csv` | the fireball with empty-span skipping on and off, 6 interleaved pairs at 1 and 4 threads (`tools/study_h/h2_time.sh`) | `docs/DCM.md` §9.Q |
| `h3_decode.csv` | the coder's formats: sizes and decode times on one pinned core (`tools/study_h/h3.sh`) | `docs/DCM.md` §9.Q |
| `prior_time.txt` | the runtime's prior against drift: frame times with and without it, and against shards (`nvfx_prior time`) | `docs/DCM.md` G2.13 |

The report's cost tables (`docs/REPORT.md` §6.7 and §7) were checked in the same session: the rollout effects took
0.8 to 1.2 ms per 128 x 128 frame (smoke varied from 0.86 to 1.17 ms between three runs), the frame models and the
simulation within a few percent of the tables, so the tables stand.

## Round 3 (11 October 2026)

`int8_timing.csv` (`nvfx_experiment int8-timing --runs 5 --core 3`: frame models in float and int8 per ISA and size),
`sections_round3.log`. The fireball with the shared step scratch tied with each module's own scratch (`docs/COMPOSE.md`
§7.4), and the sparse F3 models cost the same per frame as their dense twins. This session's machine ran about 1.35
times slower than the first quiet session's (the end-of-round-2 build of the fireball took 16.8 to 18.2 ms per frame at
1280 x 720 beside the current one's 17.0 to 17.4 ms), so absolute times are compared only within a session.

## Round 4 (14 October 2026)

`sections_round4.log`. This session's machine was slower again: the fireball took 19 to 20 ms per frame at 1280 x 720
on 4 threads. `round4_fireball_pairs.csv` has it from the build before steam, magic and study D2 were merged (6bffc93)
and from the build with them (e2ac518), alternately, nine pairs (the first three old first, the rest new first): the
merged build took +0.23 ms per frame [-0.04, +0.48] more (paired, 95% bootstrap; about 1%), a tie, so the slowdown
against round 3 is the machine's.
The same holds for study D's rollout effects (two alternating rounds of `d-timing` each).

| file | what | used in |
|---|---|---|
| `round4_fireball_pairs.csv` | the fireball before and after the round-4 merges, alternately | this note |
| `round4_scene_int8.csv` | `nvfx_scene` with 8 + 16 copies of grid_m, int8 against float, three interleaved runs | `docs/REPORT.md` §7 |
| `../compression/f4_timing_quiet.md` | F4's networks against F3's, float and int8 (`nvfx_f2 timing`, two runs) | `results/compression/README.md` (F4), `docs/REPORT.md` §3 |
| `../experiments/fx/d_timing_quiet.csv` | steam and magic with study D's effects in the same rounds (`d-timing`, four rounds) | `docs/EFFECTS.md` §5 |

The portal scene (`examples/scenes/portal.nvfxs`) took 22 ms per frame (median of its stages' sum) on 2 threads and
13 ms on 4, two runs each, without an allocation in the frame loop (`docs/EFFECTS.md` §6).

## Round 5 (14 October 2026)

`/root/nvfx-data/quiet5` (outside git) has the printouts. This session's machine was slower again: the fireball took
26.2 to 26.7 ms per frame at 1280 x 720 on 4 threads (19 to 20 ms in round 4). The rollout effects, all eleven in the
same four rounds: `../experiments/round5_effect_timing_quiet.csv` (used in `docs/EFFECTS_ENERGY.md` and
`docs/EFFECTS_ATMOSPHERE.md` §5). The effect manager's crowd benchmark (`tools/manager/time_crowd.sh build OUT 3 20`):
`../manager/crowd_timing_runs_quiet.csv` and `crowd_timing_summary_quiet.csv`, and three alternating pairs with and
without `--calibrate`: `../manager/crowd_calibrate_quiet.csv` (`docs/MANAGER.md` §7.1). The moor and fire-whirl scenes:
25 to 29 ms per frame on 2 threads and 15 to 17 ms on 4.

## Round 6 (15 October 2026)

No agent ran: load average 1.0 throughout (`round6/run.log`), the runs alone on the machine. This session's machine was
fast again: the fireball took 12.8 ms per frame at 1280 x 720 on 4 threads at float and 12.2 ms at int8 (medians of
five alternating runs, `round6/fireball_frames.txt`; 45.7 and 44.0 ms on 1 thread), against 26 ms in round 5.

Two parts, both five interleaved rounds on one pinned core:

1. **Study K's set** (`tools/study_k/time.sh build OUT 5 20`): the eleven effects as study K measured them
   (`round6/effects.csv`, `effects_summary.md`), where a frame's time goes (`profile_summary.md`), the fireball, and the
   effect manager's crowd (`crowd_summary.csv`, `crowd_runs.csv`).
2. **Every model as the library ships it now** (`nvfx_study_k timing --effects every --models DATA/library/models`):
   all 21, at 64, 128 and 256 px, float and both int8 candidates, AVX2 and AVX-512 (`round6/lib21.csv`,
   `lib21_summary.md`). This is the first quiet timing of round 7's six, the four water bodies, study M's fire and
   study M3's explosion and dust.

Per 128 x 128 frame (median over the rounds), at the default on this CPU (int8 on AVX-512 with VNNI) and at float on
AVX2 (the default on a CPU without VNNI):

| effect | default | float AVX2 | effect | default | float AVX2 | effect | default | float AVX2 |
|---|---:|---:|---|---:|---:|---|---:|---:|
| fire (M) | 0.67 | 0.92 | dust (M3) | 0.62 | 0.85 | puffs | 0.54 | 0.74 |
| smoke | 0.56 | 0.79 | cloud | 0.59 | 0.84 | ink | 0.48 | 0.74 |
| explosion (M3) | 0.55 | 0.75 | wildfire | 0.53 | 0.73 | waterfall | 0.52 | 0.71 |
| steam | 0.53 | 0.74 | coals | 0.80 | 1.07 | fountain | 0.55 | 0.77 |
| magic | 0.66 | 0.80 | mushroom | 0.60 | 0.84 | splash | 0.70 | 0.90 |
| jet | 0.56 | 0.75 | launch | 0.51 | 0.73 | rain shaft | 0.51 | 0.73 |
| orb | 0.58 | 0.79 | vortex | 0.54 | 0.76 | fog | 0.56 | 0.64 |

(ms; the orb plays at float in the library and scenes, REPORT.md §13.9: 0.63 ms on AVX-512, 0.79 on AVX2.) An instance
takes 1.6 MB at 64 px, 2.4 MB at 128 and 5.8 MB at 256 (float; int8 about 85 KB more).

Ratios over the 21 models (mean of each model's ratio, 95% bootstrap):

| | 64 px | 128 px | 256 px |
|---|---|---|---|
| int8 (second convolution) / float, both on AVX-512 | 0.845 [0.827, 0.862] | 0.906 [0.896, 0.915] | 0.959 [0.942, 0.980] |
| int8 / float, both on AVX2 | 0.958 [0.950, 0.967] | 0.979 [0.972, 0.985] | 0.978 [0.969, 0.988] |
| float on AVX-512 / float on AVX2 | 0.788 [0.778, 0.800] | 0.806 [0.785, 0.838] | 0.800 [0.787, 0.815] |

These agree with study K's provisional figures (REPORT.md §13.6, §13.8): int8 pays with VNNI (about a tenth off a 128 px
frame), barely without it, and most of the gain over the float default is the AVX-512 build itself. Float on the
AVX-512 build is 0.80 of float on AVX2 at every size, on a quiet machine too. Float rollouts still default to AVX2; moving
them would change the float frames (the float convolution and the renderer round differently between builds) and so the
frozen float keyframes, which would then depend on the CPU running the tests. That is left as a decision. (Decided the
same day: float rollouts take the AVX-512 build where the CPU has it, and the frozen float tests name the AVX2 build.
The float frames turned out the same to the bit on both builds; `docs/REPORT.md` §13.10.)

The effect manager's crowd (165 effects, 4 ms budget, 1 thread): 3.95 ms CPU per frame at float and 2.70 ms at int8
(medians of five runs), against 44.5 and 31.4 ms for every effect its own unmanaged 64 px instance.
