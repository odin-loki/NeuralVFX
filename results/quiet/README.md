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
