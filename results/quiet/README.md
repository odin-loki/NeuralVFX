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

## Round 7 (21 October 2026)

Every cost measured since round 9 (the effect rounds 9 to 11, studies O, P, Q, R and hres, the detail layer, the weather
director, the FUSE demo) was taken on the shared machine at a load of 6 to 21 and marked provisional. Round 7 re-times
them in one session, 18:21 to 21:18 UTC, with nothing else running, and the documents now quote its figures (REPORT.md
§13.11; [compare_provisional.csv](round7/compare_provisional.csv) lists every figure replaced, the old value beside the
new).

**How it was run.** `tools/quiet/round7.sh BUILD OUT` (then `tools/quiet/round7_extra.sh BUILD OUT`) runs 59 blocks of
the existing tools through the harness `tools/quiet/quiet.sh`, and `tools/quiet/summary7.sh BUILD OUT round7` makes the
files here. Before every block a gate waits until the 1-minute load average is at most 0.5 and less than 0.3 of a core
has been busy over the last 5 s (so after a block the session waits a minute or two for its own load to decay); during
the block a sampler records every 2 s the machine's busy CPU and this driver's own (its process tree's, from `/proc`),
so the CPU anything else used ("foreign") is known. A block is run again when it started above 0.5, when the foreign
load averaged more than 0.5 of a core over it, or when it exceeded a whole core over any 10 s. Three blocks were run
again (two foreign bursts and one at 1.9 foreign cores: other agents' compiles and tests, which ran between blocks while
the gate waited); a first attempt of the session, stopped after its first block met 1.1 foreign cores, is not used.
[blocks.csv](round7/blocks.csv) has every try: the load at its start (0.39 to 0.50 for the kept ones), the block's own
CPU, the foreign CPU (0.02 to 0.16 of a core on average over a kept block, at most 0.58 over 10 s) and the steal (0.75%
of the CPU in all). Medians over rounds come with a distribution-free 95% interval from order statistics
(`tools/quiet/stats.awk`: with n rounds, the k-th smallest and largest for the largest k whose coverage is at least 95%;
18 rounds give the 5th and 14th, 9 or 10 the 2nd and the 2nd largest, 6 the range); ratios are paired within a round,
their median over the rounds taken per model, and averaged over the models with a 95% bootstrap interval.

**The host** differs from earlier sessions' (the container moves): an Intel Xeon of family 6, model 207, at 2.1 GHz, 4
vCPUs under KVM, AVX-512 with VNNI (and AMX). On the 21 models round 6 timed, its AVX2 build ran as fast as round 6's
(0.94 to 1.01 of round 6's medians) and its **AVX-512 build 7 to 11% slower** (1.07 to 1.11), so the AVX-512 build's
lead over AVX2 is smaller here (below). Its speed also drifted during the session, by about ±7% between rounds of the
same work (slowest from about 18:30 to 19:15, fastest after 21:00, for both builds alike), and a single configuration
often landed in a fast or a slow state about 1.35 times apart (5 to 29% of the configurations of a round were more than
15% above their median), with no foreign load and under 1.5% steal: the host's other tenants, which this guest cannot
see. So intervals are wider than round 6's, the 2D models were given 18 rounds, and ratios within rounds are steadier
than absolute milliseconds. The anchor, the fireball at 1280 x 720 on 4 threads (study D's models, run as round 6 ran
it): 15.0 ms float, 14.8 int8 (medians of ten rounds, [13.3, 16.1] and [13.4, 15.7]); its first two rounds (18:21) and
the session's last four runs (20:53 and 21:17: 12.6 to 12.7 ms) matched round 6's 12.8 and 12.2 ms, the middle of the
session ran 14 to 16.5.

| file | what | used in |
|---|---|---|
| `blocks.csv` | every block's try: start and end, the load before and after, its own, the busy and the foreign CPU, steal, the verdict | this note |
| `lib33.csv`, `lib33_medians.csv`, `lib33_summary.md` | the 33 shipped 2D models (DATA/library/models) at 64, 128 and 256 px, float and int8 (the stepper's second convolution) on AVX2 and AVX-512, one pinned core, 18 interleaved rounds (`nvfx_study_k timing`): every round, the medians with their intervals, the ratios | REPORT.md §13.11 |
| `lib512.csv`, `lib512_medians.csv`, `lib512_summary.md` | 12 of them at 512 px, 9 rounds | REPORT.md §13.11 |
| `d_timing_rounds.csv`, `d_timing_summary.csv` | the round documents' tables: `nvfx_experiment d-timing` on the twelve models of rounds 9 to 11 as shipped and study D's fire and explosion, 9 rounds (restarts and the coarse simulation included) | EFFECTS_EARTH, _AIR, _FIRE2, _ELEMENTS §5, REPORT.md §6.16 to §6.19 |
| `detail_*` | the detail layer, old (c846c80) against new (`nvfx_detail_bench`, as REPORT.md §19): 2D over the 33 models at 128, 256 and 512 px on both ISAs (12 rounds of 16 frames), the volume and the 3D layer alone (10 rounds of 6 steps), every frame compared | REPORT.md §19 |
| `p_cost.csv`, `p_cost_summary.csv` | study P's cost: N, the half-size instance, bilinear and learned 2x at 256 and 512 px, 30 rounds of 20 frames (`nvfx_study_p cost`) | STUDY_P.md §7, REPORT.md §17.4 |
| `volume_timing.csv`, `volume_summary.csv` | the 3D steps: N2's C4 fire, study hres's fire and study R's smoke in turn, 6 rounds, 1 and 4 threads (`nvfx_3d timing`) | STUDY_3D.md §14.12, STUDY_R.md §4, STUDY_3D_HRES.md §8 |
| `scenes_runs.csv`, `scenes_summary.csv`, `liftoff_runs.csv` | the volcano, crypt, forest edge, coast and weather day scenes on 2 threads, the default (int8) and float in turn, 6 rounds; the lift-off as REPORT.md §16.6 measured it, 3 rounds (`nvfx_scene_script --profile`) | the round documents §6, WEATHER.md, REPORT.md §16.6, §18 |
| `fuse_runs.csv` | the FUSE demo, `--check`, three runs (3 workers, then the game thread alone) | FUSE.md §6, §8 |
| `fireball_runs.txt`, `fireball_summary.csv` | the fireball at 1280 x 720 and 1920 x 1080, 4 threads: float, float on AVX2 and int8 in turn, 10 rounds, and 4 more at 720p at the session's end | REPORT.md §13.10, §13.11 |
| `o_cost_passes.csv`, `o_cost_models.csv` | REPORT.md §16.6: the detail layer's second pass, and the round files of the mushroom, splash, avalanche and dust explosion against their replacements (`nvfx_ordered cost`, 9 repetitions) | REPORT.md §16.6, the round documents §5 |
| `compare_provisional.csv` | every provisional figure the documents quoted, and round 7's | this note |

**1. The 33 shipped 2D models**, per 128 x 128 frame (thread CPU time through `nvfx_render` on one pinned core, the
median over 200 frames, one-shots their 88; the median of 18 rounds and its 95% interval, ms): at the default on this
CPU (int8 on the AVX-512 build), at float (the AVX-512 build: float's default since REPORT.md §13.10), and at float on
AVX2 (what a CPU without AVX-512 plays); round 6's default and float on AVX2 beside, for the 21 it timed (the mushroom's
and the splash's files have been replaced since, by studies Q and O, with the same network):

| model | default (int8, AVX-512) | float, AVX-512 | float, AVX2 | round 6: default / float AVX2 |
|---|---:|---:|---:|---:|
| avalanche | 0.62 [0.58, 0.65] | 0.64 [0.58, 0.73] | 0.74 [0.72, 0.79] |  |
| candle | 0.58 [0.54, 0.67] | 0.66 [0.62, 0.80] | 0.72 [0.72, 0.75] |  |
| cloud | 0.65 [0.61, 0.70] | 0.70 [0.66, 0.82] | 0.77 [0.76, 0.79] | 0.59 / 0.84 |
| coals | 0.84 [0.83, 0.88] | 0.92 [0.90, 1.03] | 1.04 [1.03, 1.09] | 0.80 / 1.07 |
| crownfire | 0.59 [0.57, 0.62] | 0.67 [0.63, 0.82] | 0.74 [0.70, 0.78] |  |
| dust | 0.65 [0.59, 0.67] | 0.70 [0.65, 0.78] | 0.79 [0.77, 0.81] | 0.62 / 0.85 |
| dustexplosion | 0.69 [0.66, 0.72] | 0.75 [0.74, 0.79] | 0.87 [0.85, 0.97] |  |
| eruption | 0.55 [0.53, 0.58] | 0.60 [0.59, 0.77] | 0.68 [0.66, 0.72] |  |
| explosion | 0.57 [0.55, 0.64] | 0.64 [0.61, 0.65] | 0.75 [0.71, 0.83] | 0.55 / 0.75 |
| fire | 0.75 [0.74, 0.82] | 0.80 [0.77, 0.85] | 0.93 [0.89, 0.97] | 0.67 / 0.92 |
| fog | 0.46 [0.45, 0.55] | 0.54 [0.51, 0.62] | 0.60 [0.57, 0.65] | 0.56 / 0.64 |
| fountain | 0.60 [0.57, 0.77] | 0.64 [0.62, 0.83] | 0.74 [0.72, 0.83] | 0.55 / 0.77 |
| geyser | 1.12 [1.04, 1.21] | 1.15 [1.13, 1.19] | 1.24 [1.21, 1.31] |  |
| haboob | 0.85 [0.82, 1.00] | 0.92 [0.88, 0.99] | 1.00 [0.96, 1.02] |  |
| heavygas | 0.46 [0.45, 0.50] | 0.52 [0.50, 0.56] | 0.61 [0.60, 0.63] |  |
| ink | 0.53 [0.52, 0.54] | 0.59 [0.58, 0.60] | 0.69 [0.62, 0.72] | 0.48 / 0.74 |
| jet | 0.62 [0.55, 0.69] | 0.67 [0.62, 0.76] | 0.71 [0.69, 0.73] | 0.56 / 0.75 |
| launch | 0.57 [0.53, 0.65] | 0.62 [0.61, 0.74] | 0.72 [0.71, 0.78] | 0.51 / 0.73 |
| lava | 0.54 [0.51, 0.71] | 0.60 [0.57, 0.63] | 0.71 [0.67, 0.87] |  |
| magic | 0.63 [0.61, 0.70] | 0.70 [0.68, 0.73] | 0.79 [0.77, 0.81] | 0.66 / 0.80 |
| mushroom | 0.67 [0.66, 0.69] | 0.77 [0.72, 0.80] | 0.82 [0.79, 1.02] | 0.60 / 0.84 |
| orb | 0.63 [0.58, 0.70] | 0.67 [0.65, 0.81] | 0.81 [0.77, 0.95] | 0.58 / 0.79 |
| puffs | 0.59 [0.56, 0.65] | 0.64 [0.62, 0.74] | 0.73 [0.71, 0.79] | 0.54 / 0.74 |
| rainshaft | 0.58 [0.55, 0.67] | 0.64 [0.63, 0.83] | 0.72 [0.70, 0.86] | 0.51 / 0.73 |
| seaspray | 0.48 [0.46, 0.52] | 0.53 [0.51, 0.58] | 0.61 [0.57, 0.64] |  |
| smoke | 0.63 [0.59, 0.68] | 0.67 [0.65, 0.71] | 0.84 [0.74, 0.92] | 0.56 / 0.79 |
| smokering | 0.84 [0.83, 0.86] | 0.89 [0.83, 0.92] | 0.95 [0.90, 1.00] |  |
| splash | 0.73 [0.65, 0.74] | 0.77 [0.75, 0.81] | 0.87 [0.85, 0.90] | 0.70 / 0.90 |
| steam | 0.56 [0.54, 0.70] | 0.64 [0.61, 0.77] | 0.73 [0.71, 0.74] | 0.53 / 0.74 |
| vortex | 0.63 [0.58, 0.73] | 0.67 [0.62, 0.80] | 0.74 [0.72, 0.84] | 0.54 / 0.76 |
| waterfall | 0.56 [0.53, 0.70] | 0.65 [0.61, 0.78] | 0.74 [0.71, 0.90] | 0.52 / 0.71 |
| wildfire | 0.58 [0.55, 0.77] | 0.69 [0.64, 0.84] | 0.74 [0.71, 0.83] | 0.53 / 0.73 |
| windowfire | 0.56 [0.55, 0.69] | 0.66 [0.59, 0.68] | 0.76 [0.73, 0.82] |  |

The mean over the 33: 0.63 ms at the default (0.46 for heavy gas to 1.12 for the geyser), 0.69 at float on AVX-512 and
0.79 on AVX2; 0.37 ms at 64 px and 1.51 at 256. The orb plays at float in the library and the scenes (REPORT.md §13.9).
An instance holds 1.6 MB at 64 px, 2.4 MB at 128, 5.8 MB at 256 and 19 MB at 512 (float; int8 85 KB more).

Ratios over the models (the mean of each model's median ratio over the rounds, 95% bootstrap; 33 models at 64 to 256 px,
12 at 512 px):

| | 64 px | 128 px | 256 px | 512 px | round 6 (21 models; 64, 128, 256 px) |
|---|---|---|---|---|---|
| int8 / float, both on AVX-512 | 0.853 [0.846, 0.860] | 0.906 [0.897, 0.914] | 0.955 [0.946, 0.965] | 1.003 [0.985, 1.023] | 0.845, 0.906, 0.959 |
| int8 / float, both on AVX2 | 0.960 [0.956, 0.965] | 0.973 [0.965, 0.979] | 0.978 [0.969, 0.985] | 0.996 [0.982, 1.010] | 0.958, 0.979, 0.978 |
| float on AVX-512 / float on AVX2 | 0.871 [0.861, 0.881] | 0.886 [0.875, 0.896] | 0.914 [0.902, 0.926] | 0.948 [0.929, 0.966] | 0.788, 0.806, 0.800 |
| int8 on AVX-512 / float on AVX2 | 0.745 [0.732, 0.759] | 0.802 [0.791, 0.815] | 0.872 [0.859, 0.886] | 0.956 [0.933, 0.979] | 0.663, 0.724, 0.767 |

int8's own gain on the AVX-512 build is round 6's to within 0.01 at every size (a tenth off a 128 px frame; nothing at
512 px, where the float renderer and detail layer are nearly all of the frame); the AVX-512 build's lead over AVX2 is
smaller on this host (0.87 to 0.95 instead of 0.79 to 0.81), as its AVX-512 work ran 7 to 11% slower than round 6's.

At 256 and 512 px, the 12 of the subset (ms, the median of 18 and of 9 rounds):

| model | 256 px: default | float, AVX-512 | float, AVX2 | 512 px: default | float, AVX-512 | float, AVX2 |
|---|---:|---:|---:|---:|---:|---:|
| dustexplosion | 1.84 [1.68, 1.96] | 1.92 [1.87, 2.05] | 2.13 [2.07, 2.29] | 5.94 [5.89, 6.57] | 5.98 [5.75, 6.11] | 6.11 [5.97, 6.66] |
| explosion | 1.47 [1.39, 1.67] | 1.55 [1.47, 1.63] | 1.64 [1.61, 1.72] | 4.14 [3.97, 5.46] | 4.39 [4.06, 4.46] | 4.51 [4.16, 6.30] |
| fire | 1.86 [1.81, 1.90] | 1.98 [1.92, 2.03] | 2.13 [2.09, 2.29] | 5.78 [5.25, 6.02] | 5.38 [5.19, 5.50] | 5.82 [5.62, 6.49] |
| fog | 0.93 [0.78, 0.95] | 1.01 [0.99, 1.05] | 1.12 [1.10, 1.18] | 2.60 [2.49, 3.19] | 2.60 [2.51, 2.78] | 2.72 [2.58, 3.12] |
| geyser | 2.83 [2.53, 2.87] | 2.77 [2.51, 2.80] | 2.81 [2.64, 2.92] | 6.01 [5.81, 6.45] | 6.08 [5.65, 6.61] | 6.24 [5.57, 6.85] |
| haboob | 1.23 [1.22, 1.26] | 1.31 [1.26, 1.44] | 1.42 [1.39, 1.48] | 2.50 [2.29, 3.34] | 2.68 [2.25, 3.40] | 2.81 [2.72, 2.92] |
| launch | 1.41 [1.33, 1.66] | 1.43 [1.38, 1.48] | 1.61 [1.57, 1.68] | 4.75 [4.62, 5.93] | 4.80 [4.62, 4.95] | 5.36 [5.18, 5.75] |
| mushroom | 1.85 [1.77, 1.89] | 1.94 [1.88, 2.07] | 2.06 [2.01, 2.16] | 6.65 [6.51, 7.02] | 6.65 [6.38, 6.97] | 7.02 [6.87, 7.32] |
| orb | 2.07 [1.84, 2.18] | 2.16 [1.99, 2.27] | 2.31 [2.24, 2.33] | 7.33 [6.91, 7.59] | 7.30 [7.10, 7.61] | 7.33 [7.02, 7.60] |
| smoke | 1.51 [1.47, 1.57] | 1.53 [1.51, 1.62] | 1.68 [1.64, 1.75] | 4.90 [4.89, 4.99] | 4.91 [4.80, 5.00] | 5.13 [4.97, 5.23] |
| waterfall | 1.48 [1.36, 1.73] | 1.47 [1.38, 1.83] | 1.64 [1.62, 1.69] | 4.71 [4.52, 5.70] | 4.66 [4.49, 5.76] | 5.23 [5.16, 6.09] |
| wildfire | 1.49 [1.45, 1.56] | 1.53 [1.50, 1.59] | 1.76 [1.66, 2.12] | 5.19 [4.95, 6.38] | 5.71 [5.28, 6.13] | 5.79 [5.32, 6.92] |

**2. The detail layer before and after** (REPORT.md §19; old c846c80 against this build in one process, every frame
compared: 38,016 frames of the 33 models and 480 volume and detail steps, **none differs**). 2D, new / old, the
geometric mean over the 33 models [95% bootstrap]: the detail stage 0.889 [0.880, 0.898] and 0.864 [0.852, 0.877] at 128
px (AVX2, AVX-512), 0.893 and 0.889 at 256, 0.899 and 0.905 at 512; a whole frame 0.964 to 0.973. The busy run's ratios
(27 models) were within 0.015 of these. 3D (N2's model, 1 thread): the detail layer alone 0.223 [0.204, 0.227] (AVX2)
and 0.204 [0.199, 0.207] (AVX-512), 10.0 to 10.4 ms against 47 to 50; a volume step 0.396 and 0.375, 24.6 to 25.6 ms
against 65 or 66 ms. The busy run said 0.155 to 0.176 and 0.309 to 0.319: it had slowed the old scalar layer by a third
and the new one by a few percent, so the gain is a factor of 4.5 to 4.9 for the layer and 2.5 to 2.7 for a step, not 5.7
to 6.5 and 3.1 to 3.2. On 4 threads (no free core on the busy machine): a step 9.7 to 11.3 ms against 20.6 to 25.2 (0.45
to 0.46).

**3. Study P's half size against native** (STUDY_P.md §7; the runtime's default, int8; 30 rounds): native at 512 px
took 4.43 to 6.43 ms (the busy run: 4.16 to 6.00), the half-size instance 0.29 to 0.38 of it (0.29 to 0.34), half size
and the CPU's bilinear 2x 0.35 to 0.46 (0.36 to 0.43), half size and the learned 2x 0.505 to 0.619 (0.453 to 0.593): **3
of the 6 effects meet the rule's bound of 0.55** (the mushroom, launch, the waterfall) **instead of 4** (smoke moves
from 0.550 to 0.619). At 256 px the half-size instance costs 0.36 to 0.42 of native, the learned 2x 0.52 to 0.68.

**4. The 3D steps** (through the volume API, the median of 6 rounds [95% interval], ms):

| | N2's C4 fire | study hres's fire | study R's smoke |
|---|---:|---:|---:|
| a step, 1 thread | 21.0 [18.8, 22.2] (was 70, the scalar detail layer) | 239 [233, 244] | 21.4 [20.0, 24.8] (was 26.6) |
| a step, 4 threads | 9.0 [8.5, 9.3] (was 23) | 64 [63, 68] | 9.2 [8.7, 9.9] (9.0) |
| a 256 x 256 picture, 1 thread | 55.9 [51.0, 57.4] (61) | 48.7 [44.4, 52.6] | 85.7 [79.8, 94.6] (93) |
| a 256 x 256 picture, 4 threads | 14.1 [13.5, 14.7] (16) | 12.7 [11.5, 13.3] | 22.7 [21.7, 23.9] (22) |
| a 512 x 512 picture, 1 / 4 threads | 200 / 51.6 | 176 / 46.4 | 317 / 83.4 |
| a start (24-frame warm-up), 1 / 4 threads | 546 / 224 (was 1,866 / 602) | 5,901 / 1,565 | 554 / 226 (683 / 222) |

(In brackets the first measurements, STUDY_3D.md §14.12 and STUDY_R.md §4, "was" where they differ by more than 15%.)
hres's coarse step alone takes 192 ms on one thread (N2's 10.5), so its step is 11 times N2's.

**5. The FUSE demo and the scenes** (FUSE.md §8; the round documents; the median of the runs): the demo's 600 frames in
41.8 s on 3 workers and the game thread (busy: 168 s) and 55.3 s on the game thread alone (138 s); the volume's job per
volume frame step 26.9, picture 38.7 and fields 50.7 ms (272, 105, 111); the scene's job 32.7 ms a picture (122), the
stage's 14.8 (50); `create` 1.1 s (3.1). On 2 threads at the default (int8; float in brackets), the median frame: the
volcano 13.7 ms (13.4; the busy machine's mean 20), the crypt 10.2 (10.2; busy 28 and 19), the forest edge 19.0 (19.3;
not measured before), the coast 17.3 (17.7; busy 51), the weather day 16.4 (17.3; busy 76), the lift-off 19.8 (busy 46
to 50); every frame loop without an allocation.

**6. The fireball** (REPORT.md §13.10; 4 threads, the median of 10 rounds [95% interval]): at 1280 x 720 15.0 ms [13.3,
16.1] float, 15.7 [13.9, 16.8] float on AVX2 and 14.8 [13.4, 15.7] int8; at 1920 x 1080 31.0 [28.3, 32.1], 32.8 [30.5,
33.6] and 31.2 [30.3, 32.1]. int8 / float ties at both sizes (0.997 [0.949, 1.044] and 1.010 [0.964, 1.048]: a 4-thread
frame is the compositor's), float on AVX-512 / AVX2 0.948 [0.920, 1.027] and 0.942 [0.918, 1.015]. Round 6's 12.8 ms
(float, then on AVX2) and 12.2 ms (int8) at 720p were matched by this session's first and last runs only.

**REPORT.md §16.6** (`round7_extra.sh`): launch's second detail pass costs it +0.03 to +0.24 ms a frame (4 to 16%;
busy: 8 to 18%), the waterfall's nothing measurable but float at 256 px; the round's files of the mushroom, splash,
avalanche and dust explosion cost what their replacements do (paired differences of -0.11 to +0.14 ms, inside
repetitions that ranged by a factor of up to 2 on this host).

**What changed against the provisional figures** ([compare_provisional.csv](round7/compare_provisional.csv)). **Down**:
every multi-threaded figure, often by half or more, since on the busy machine extra threads found no free core: the
scenes on 2 threads by 30 to 78%, the FUSE demo's jobs on 3 workers by 54 to 90% (on the game thread alone 16 to 86%),
the 3D steps on 4 threads; and the 3D steps on one thread (N2's fire by 70%, mostly because the detail layer moved to
vectors after its figure was taken; the smoke by 20%). **Up**: most single-core figures of the 2D models, by up to 40%
against what the documents quoted, because the busy tables gave the least of their rounds, one pinned thread lost little
to other jobs on 4 cores, and this host's AVX-512 build is slower than those sessions'; against the busy medians, where
the documents gave them (rounds 10 and 11, 60 figures), 55 of the quiet medians are within 15% (the smoke ring's and
study D's fire's defaults 16 to 18% higher, sea spray's two and one of study D's fire 15 to 18% lower). Native at 512 px
in study P rose 6 to 18%. **Ratios**: the 2D detail layer's gain and int8's held; study P's learned 2x and the 3D detail
layer's gain moved against their claims (3 effects under 0.55 instead of 4; a factor of 4.5 to 4.9 instead of 5.7 to
6.5); the AVX-512 build's lead over AVX2 is smaller on this host than on round 6's (0.87 to 0.95 against 0.79 to 0.81).
