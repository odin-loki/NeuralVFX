# Commit plan: the working tree of 24 October 2026

**Status:** a plan. Nothing has been committed. HEAD is `2249562` ("Ship hres 3D fire C2 and smoke_hres ...") on
`main`. The index holds the first draft's 77 files (a `git add -A` less `build/` and `Testing/`, both gitignored);
everything written since sits on top of it unstaged (`MM` and ` M` rows of `git status --short`) or untracked (`??`:
`COMMIT_PLAN.md`, `docs/STUDY_R2_RESTORE.md`, `docs/WAVE1_CHECKLIST.md`, `tools/manager/time_half_128.sh`,
`tools/preflight_wave1.sh`, `tools/ship_wave1.sh`, `tools/study_3d/hres_smoke_r2.sh`,
`tools/study_3d/hres_smoke_r2_train.sh`). The three commits below take the **working tree**, every row of
`git status -uall` (90: 82 tracked files changed against HEAD, about 7,150 lines added and 475 removed, and the 8
untracked), so the index must be unstaged first and rebuilt commit by commit from the lists. The tree was still being
written to while this plan was (files appeared between two readings of `git status`): run `git status -uall` once more
before "Before anything" and add any new row to the commit it belongs to. The owner makes the commits; this file only
lists them.

The working tree holds TODO's "Done since 24 October 2026": the half-size default per effect (`NVFX_MANAGER_VERSION`
3), the volume job hook (`NVFX_VERSION` 10), the coupled 3D fire and smoke (`sim3d::Stack`, `coupled3d.hpp`), the
weather director's moist air, wet ground, storm front, sun and cloud deck, the steam fog, contrail, helicopter
downwash, whiteout and drifting snow library entries, studies Q2's and R2's job scripts, study hres's fire test tables,
its smoke-on-R2's-teacher driver and polish, and the documents for each. The order puts code before content and
content before study bookkeeping, so each commit builds and tests on its own.

## Verified (24 October 2026, WSL, g++-14, Release, tests on)

On the **staged set** (the 77 files of the index, exported with `git checkout-index` and built on their own):

| Check | Result |
|---|---|
| `git diff --cached --stat` | 77 files, +5,699 / -426; no `build/`, `Testing/`, data, weights or videos; no secrets (key/token/password/private-key patterns and home paths grepped over every changed file: none) |
| `cmake --build` of the exported index | clean; the only two warnings are pre-existing (`tests/test_wrapper.cpp:153,167`, `-Wconversion` on a deliberate out-of-range enum) |
| `ctest -L neuralfx -j 1` | 587 of 587 passed (66 skipped with code 77: the data-labelled tests, no models under `NEURALVFX_DATA` in the export), 274 s |
| `nvfx_alloc_test` | `ok: no allocation per frame` (the volume on an engine's job system included: 0 allocations in 84 frames, 7,813 job runs) |
| `nvfx_gallery --library library --check` | `237 effects, 66 played, 171 parsed and checked only, 0 failed` |
| Index modes | the new scripts (`tools/study_3d/r2_*.sh`, `tools/study_q2/*.sh`) are `100755`; `hres_status.sh`, `run_fire_validate.sh` and the other hres scripts stay `100644` as they are in HEAD |

On the **working tree** (the staged set plus everything since, which is what the commits below ship): the full ctest,
**570 passed, 0 failed, 32 skipped** (602 in all; `ctest -L neuralfx` from `build/`, 24 October 2026, WSL, on a quiet
tree after concurrent edits settled; parallel `-j4` ~72 s wall, serial `-j1` ~287 s; the skips are the data-labelled
tests exiting 77 where this PC lacks models under `NEURALVFX_DATA` or `build/library.nvfxp`). An earlier run of the tree, before
`trails.nvfxl`'s duplicate `helicopter_downwash` was removed, had failed 16 tests on one cause (`look 'brownout' is
already declared in dust.nvfxl`); that version is gone. A concurrent relink of `neuralfx_weather_tests` during
`-j1` can report 26 `BAD_COMMAND` (`text file is busy`) flakes; rerun `-R Weather` or use a quiet tree. The catalogue
check on this tree: 240 entries, `drifting_snow` played with 0 allocations at both ends of its ranges (TODO.md). Run the
pre-push checklist again on the three commits before the push; the split above must match that run.

## Before anything

```sh
git reset -q                     # unstage everything (the working tree is untouched)
git status --short               # expect " M" and "??" rows only, none staged
```

**Since the draft.** The tree moved on after the draft was written; every item here is in a commit's list below.
- Untracked, now placed: this `COMMIT_PLAN.md` and `docs/WAVE1_CHECKLIST.md` (commit 3); `tools/study_3d/hres_smoke_r2.sh`
  (smoke at hres on study R2's teacher: the plan of STUDY_3D_HRES.md §10, a dry run without `FORCE=1`) and
  `tools/study_3d/hres_smoke_r2_train.sh` (a forwarder to it with `STAGE=record,train,finish`), both commit 3's;
  `tools/ship_wave1.sh` (the three `git add`s of this plan as a script, `DRY_RUN=1` by default: it prints and runs
  nothing) and `tools/preflight_wave1.sh` (the fast pre-push check: `bash -n` over the scripts, the weather and library
  tests, `nvfx_alloc_test`, the catalogue check; touches no git state), commit 3's; `docs/STUDY_R2_RESTORE.md` (carrying
  the container's `r2_smoke3d` tree to another machine, STUDY_R2.md §0's companion as STUDY_Q2_RESTORE.md is Q2's),
  commit 3's; `tools/manager/time_half_128.sh` (the quick quiet re-timing of half size's saving, MANAGER.md §7.2 and
  REPORT §22.8, which both cite it; written, not yet run), commit 3's as well; `.gitattributes` (`*.sh` and `*.awk`
  `text eol=lf`; checklist item 6), commit 3's: every `??` row goes in commit 3, the documents that cite the script
  are commit 1's and the three land on one push.
- `src/compose/compose.hpp`, unstaged: the frame's `wet`, `puddle`, `deck`, `foot` (COMPOSE.md §4.9.1), the fields
  behind script.cpp's `frame` keys and script_run.cpp's evaluation; `examples/scenes/weather_day.nvfxs` wires the
  four to `sky.*`; `src/compose/compose.cpp`, unstaged, draws them in the background pass (the wet ground and its
  puddles, COMPOSE.md §4.11; the deck's underside from `deck` and `foot`, §5.4, `wx::deck_contrast` for the foot, so
  it includes `weather.hpp`), and `tests/test_script.cpp` gains the four `Frame.*` tests that check the drawing. All
  commit 2's, with the weather director.
- Library entries written after the draft, unstaged: `helicopter_downwash` (`library/dust.nvfxl`, #37, with the looks
  `brownout` and `brownout_haze`), `whiteout` and `drifting_snow` (`library/storms.nvfxl`, #54 and #75), the
  `pyroclastic_flow` comment and `about` in `library/earth.nvfxl` (#110, no change to the effect); `drifting_snow`'s
  keyframe row in `results/library/frozen_round3.csv` and its name in `CMakeLists.txt`'s `LibraryProceduralKeyframesFrozen`
  `--only` list; three tests in `tests/test_library.cpp`; the LIBRARY.md rows and counts (240 entries, 173 learned).
  All commit 2's (`frozen_round3.csv` is needed: the ctest verifies the keyframe against it).
- The weather director's sun, cloud deck, moving storm and trees, unstaged: `wx::sun_elevation_deg`, `sun_across`,
  `sun_share`, `beam_transmittance`, `shaft_visibility` (WEATHER.md §3.14), `deck_cover` and `deck_foot_deg` (§3.15),
  `taylor_length_m` (§3.13), `trees` (§3.6, a hook), the `hour` key of `weather` in `src/compose/script.cpp` and
  `script_run.cpp`; the new members `sky.trees`, `sky.carried`, `sky.front`, `sky.hour`, `sky.elevation`, `sky.sunx`,
  `sky.sun`, `sky.shaft`, `sky.deck`, `sky.foot`; five tests in `tests/test_weather_director.cpp` (39 in all). Commit 2's.
- Study hres's smoke_r2 plan, unstaged: STUDY_3D_HRES.md §10, `tools/nvfx_3d.cpp`'s `--effect smoke_r2` with
  `--study hres` (its own seed blocks at 9,630,000), `hres_status.sh`'s smoke_r2 block. The tool change is commit 1's
  (the file goes there whole), the doc and the scripts commit 3's.
- `tools/study_q2/bootstrap.sh` rewritten as a checklist with a dry run (`Q2_DRY=1`; STUDY_Q2_RESTORE.md §3): commit 3's.
- Documents grown since: ENGINES.md §12.1 (the coupled demo through the API), `videos/README.md` §9 and §9c.1 to §9c.3
  (the HRES rows and the `--kind coupled` options), FUSE.md §9 on the adapter's four volume calls, REPORT's hres
  addendum, TODO's new blocks. Each stays with the commit that already lists its file.

## Commit 1: runtime and API

**Message:**

```
Manager v3 (half size per effect), volume jobs (NVFX_VERSION 10), coupled 3D fire and smoke (sim3d::Stack), CRLF-safe verify readers

- GNU Affero General Public License v3 (`LICENSE`; README Licence section), as the other Desktop repos.
- nvfx_manager_desc.half_px < 0: the half-size default per effect (NVFX_MANAGER_VERSION 3; MANAGER.md §2.10, §6;
  REPORT §22.11). The FUSE adapter passes it through; time_crowd.sh pins §7's baseline at --half-px 0. (MANAGER.md
  §7.2 and REPORT §22.8 cite tools/manager/time_half_128.sh, the quick quiet re-timing; the script is commit 3's.)
- nvfx_volume_create_with_jobs (NVFX_VERSION 10; ENGINES.md §12, INTEGRATION.md §3, FUSE.md §9): every parallel loop
  of a step, read or picture as one fork-join on the engine. VolumeApi.SameFieldsAndPicturesOnAnEnginesJobSystem,
  nvfx_alloc_test.
- sim3d::Stack and its Seam (STUDY_R.md §6.1): two boxes seamed top to bottom, heat, soot and velocity crossing both
  ways; nvfx_3d video --kind coupled-sim (--upper smoke_r2 takes study R2's teacher). The volume-API coupled demo rule
  in include/neuralfx/coupled3d.hpp, tested in tests/test_coupled3d.cpp (part of neuralfx_3d_tests); ctest gains
  Volume3dToolUsage and Coupled3dVideoSmoke (the latter only where ffmpeg is found at configure time).
- nvfx_3d --effect smoke_r2 with --study hres (STUDY_3D_HRES.md §10): study R2's teacher on the finer coarse grid, its
  own seed blocks (training 9,630,000 + i, long 9,631,000 + i, the bounds' teacher runs 9,633,000 + 100 j + k), the
  plume source's peaks shared with study R's smoke; smoke_r refused for hres.
- nvfx_gallery and nvfx_scene_script --verify readers strip \r: a CRLF checkout of frozen.csv had failed every hash.
- docs: ENGINES and REPORT also carry the weather director's sky.* keys and the study hres addendum (commits 2 and 3)
  rather than split a Markdown table by hand.
```

Files (31):

```
CMakeLists.txt
LICENSE
README.md
docs/ENGINES.md
docs/FUSE.md
docs/INTEGRATION.md
docs/MANAGER.md
docs/REPORT.md
docs/STUDY_R.md
engines/fuse/nvfx_fuse.hpp
include/neuralfx/coupled3d.hpp
include/neuralfx/nvfx.h
include/neuralfx/nvfx_manager.h
include/neuralfx/nvfx_volume.h
include/neuralfx/parallel.hpp
include/neuralfx/sim3d.hpp
src/manager/manager.cpp
src/manager/manager.hpp
src/manager/nvfx_manager.cpp
src/runtime/nvfx_volume.cpp
src/sim3d/sim3d.cpp
tests/alloc_test.cpp
tests/test_3d.cpp
tests/test_coupled3d.cpp
tests/test_manager.cpp
tools/manager/time_crowd.sh
tools/manager/time_crowd_summary.awk
tools/nvfx_3d.cpp
tools/nvfx_gallery.cpp
tools/nvfx_scene_script.cpp
videos/README.md
```

Overlaps, deliberately kept whole:
- `docs/ENGINES.md` also gains the weather director's new `sky.*` keys (commit 2's work) in its scene-inputs table and
  §12.1 (the coupled demo through the API); `docs/REPORT.md` also gains the study hres addendum (commit 3's work, the
  paragraph after §15). Splitting a Markdown hunk by hand buys nothing and risks a half-written table; the message says so.
- `videos/README.md` adds §9's HRES rows (now "validated and tested"), the coupled video and §9c.1 to §9c.3 (the
  `--kind coupled` and `--kind coupled-sim` options, the rule, the `Stack`), so it is this commit's.
- `CMakeLists.txt` also extends `LibraryKeyframesFrozen`'s `--only` list with `steam_fog` and `coloured_smoke_grenade`
  and `LibraryProceduralKeyframesFrozen`'s with `drifting_snow` (commit 2's entries). The first test is data-labelled
  and skips without the models; the second would fail on commit 1 alone (no `drifting_snow` yet) and passes once
  commit 2 lands, which is fine because the three land together on one push.
- `tools/nvfx_3d.cpp` also carries `--effect smoke_r2` (the tool's side of commit 3's §10 plan); the file goes here whole.

```sh
git add CMakeLists.txt LICENSE README.md docs/ENGINES.md docs/FUSE.md docs/INTEGRATION.md docs/MANAGER.md docs/REPORT.md docs/STUDY_R.md \
  engines/fuse/nvfx_fuse.hpp include/neuralfx/coupled3d.hpp include/neuralfx/nvfx.h include/neuralfx/nvfx_manager.h \
  include/neuralfx/nvfx_volume.h include/neuralfx/parallel.hpp include/neuralfx/sim3d.hpp \
  src/manager/manager.cpp src/manager/manager.hpp src/manager/nvfx_manager.cpp src/runtime/nvfx_volume.cpp src/sim3d/sim3d.cpp \
  tests/alloc_test.cpp tests/test_3d.cpp tests/test_coupled3d.cpp tests/test_manager.cpp \
  tools/manager/time_crowd.sh tools/manager/time_crowd_summary.awk tools/nvfx_3d.cpp tools/nvfx_gallery.cpp \
  tools/nvfx_scene_script.cpp videos/README.md
```

## Commit 2: the weather director and the library catalogue

**Message:**

```
Weather director: humidity, dew point, cloud base, wet ground, the storm moving with the wind, the sun and its shafts, the cloud deck (WEATHER.md §3.11 to §3.15); library: steam fog (#61), contrail (#36), helicopter downwash (#37), whiteout (#54), drifting snow (#75)

- wx::saturation_vapour_hpa, dew_point, cloud_base_m (Magnus with Alduchov and Eskridge's coefficients; Espy's rule);
  wx::evaporation_mm_h, wind_2m, kDepressionStorage (Penman's drying power, 1.5 mm storage). Read from scripts as
  sky.humidity, sky.dewpoint, sky.base, sky.water, sky.wet, sky.puddle, sky.evaporation (§3.11, §3.12).
- The storm across the scene (§3.13, Taylor's hypothesis): wx::taylor_length_m; sky.carried, sky.front.
- The sun and its shafts (§3.14): wx::sun_elevation_deg, sun_across (45 N on the equinox, Iqbal), sun_share (Kasten
  and Czeplak's diffuse fraction), beam_transmittance (Beer-Lambert along the slant path through a 300 m layer),
  shaft_visibility; sky.hour, sky.elevation, sky.sunx, sky.sun, sky.shaft; the `hour` key of `weather`
  (script.cpp's kWeather, script_run.cpp's set_hour each frame).
- The cloud deck (§3.15): wx::deck_cover (the cover as a layer, a smoothstep from 4 to 6 oktas: a ceiling in
  aviation's sense) and deck_foot_deg (where Koschmieder's haze hides the underside, arcsin(min(1, base / visibility)));
  sky.deck, sky.foot.
- The frame's wet, puddle, deck and foot (compose.hpp; the `frame` keys of script.cpp, evaluated each frame in
  script_run.cpp; COMPOSE.md §4.9.1), each 0 by default so every scene before is the same to the bit, drawn by
  compose.cpp's background pass: wet darkens the ground rows and the scene's light on them by lerp(1, 0.6, wet)
  before the haze (§4.11, Lekner and Dorf), puddle fills the ground texture's hollows deepest first and mirrors the
  sky and the hills by Fresnel's reflectance at the depression angle (Schlick), deck lays a layer's underside over
  the dome whose cover per row is deck times wx::deck_contrast along the slant path (§5.4: the CIE overcast grey with
  value-noise cells, lit by the flash, hiding the stars), foot where that contrast is 2 %. tests/test_script.cpp:
  Frame.WetDarkensTheGroundAndNothingElse, PuddlesMirrorTheSkyAndTheHillsByFresnel, TheDeckCoversTheDomeByItsContrast,
  TheWetGroundAndTheDeckAreTheSameOnAnyNumberOfThreads. examples/scenes/weather_day.nvfxs wires the four to sky.wet,
  sky.puddle, sky.deck, sky.foot (moves `ScriptedWeatherDayFrozen`: re-freeze
  `results/experiments/weather_frozen.csv` once on a machine with the library models; skipped exit 77 on this PC until
  then; `deck_drift` not drawn).
- The trees' sway by the Beaufort force, wx::trees (§3.6, a hook); sky.trees.
- script_run.cpp's "a weather has no ..." message and COMPOSE.md's table list all of it. tests/test_weather_director.cpp
  (39): WeatherRelations.TheSunAndItsShafts, TheCloudDeck, TheDeckFootHidesInDust;
  WeatherDay.TheFrameCarriesTheWetGroundAndTheCloudDeck among them;
  WeatherDirector.TheStormMovesAcrossTheSceneWithTheWind, TheSunStandsByTheHourAndShowsThroughBrokenCloud among them.
- library: steam_fog (steam.nvfxl, catalogue #61) and contrail (trails.nvfxl, #36); coloured_smoke_grenade
  (smoke.nvfxl) adjusted; helicopter_downwash (dust.nvfxl, #37: the dust model's bursts taking turns under the smoke
  model, the looks brownout and brownout_haze); whiteout (storms.nvfxl, #54: the blizzard's snow, a saltation layer
  with a threshold, the fog model in two storeys) and drifting_snow (storms.nvfxl, #75, procedural: two bands of the
  compositor's snow under a gust field); earth.nvfxl's pyroclastic_flow comment and about (#110: built from the
  avalanche model, not learned). The catalogue at 240 entries, 173 learned, --check clean. steam_fog's and the
  grenade's keyframes frozen in results/library/frozen.csv (judged with the models), drifting_snow's in
  frozen_round3.csv (procedural, no models needed); contrail, the downwash and the whiteout not yet judged by eye
  (written on a machine without the models) and so not frozen and in no --only list (LIBRARY.md §4, §10.4).
  Library.SteamFogIsFogAndSteamOverAPool, ContrailIsAColdTrailSetBackBehindTheJet,
  HelicopterDownwashIsDustTakingTurnsUnderAHazeInTheWind, BlizzardAndDriftingSnowAreBlownSnow,
  WhiteoutIsABlizzardWhoseBlownSnowHasAThreshold.
```

Files (21):

```
docs/COMPOSE.md
docs/LIBRARY.md
docs/WEATHER.md
examples/scenes/weather_day.nvfxs
library/dust.nvfxl
library/earth.nvfxl
library/smoke.nvfxl
library/steam.nvfxl
library/storms.nvfxl
library/trails.nvfxl
results/library/frozen.csv
results/library/frozen_round3.csv
src/compose/compose.cpp
src/compose/compose.hpp
src/compose/script.cpp
src/compose/script_run.cpp
src/compose/weather.cpp
src/compose/weather.hpp
tests/test_library.cpp
tests/test_script.cpp
tests/test_weather_director.cpp
```

```sh
git add docs/COMPOSE.md docs/LIBRARY.md docs/WEATHER.md examples/scenes/weather_day.nvfxs \
  library/dust.nvfxl library/earth.nvfxl library/smoke.nvfxl library/steam.nvfxl library/storms.nvfxl library/trails.nvfxl \
  results/library/frozen.csv results/library/frozen_round3.csv \
  src/compose/compose.cpp src/compose/compose.hpp src/compose/script.cpp src/compose/script_run.cpp src/compose/weather.cpp src/compose/weather.hpp \
  tests/test_library.cpp tests/test_script.cpp tests/test_weather_director.cpp
```

`results/library/frozen_round3.csv` gains one row (`drifting_snow`, 1,034,353 bytes, its SHA-256) and nothing else; a
`git diff --cached --stat` that shows the whole file rewritten means a CRLF conversion (checklist item 6). Before the
commit, `nvfx_gallery --library library --check` must read `0 failed` and no file but `dust.nvfxl` may declare the
look `brownout` (the duplicate in `trails.nvfxl` that once broke 16 tests is gone; `rg "look brownout" library/`).

## Commit 3: the studies' bookkeeping (hres, Q2, R2) and TODO

**Message:**

```
Studies hres, Q2 and R2: fire test tables, smoke_r2-at-hres drivers, CUDA spread/edge hybrid, PLAN_PC_GPU; Q2 and R2 jobs; wave 1 checklist and ship scripts

- hres (STUDY_3D_HRES.md §7, §8): the fire's test tables under results/3d/hres/fire/ (compare_test_*, edge_test*.csv,
  edgecheck_test.md judged with the calibrated bounds, the single-run table as edgecheck_test_single.md);
  hres_fire_recheck_test_edges.sh and hres_sync_fire_test_results.sh new; hres_fire_validate.sh with its body in
  main(), run_fire_validate.sh on a snapshot copy, hres_status.sh reporting the test tables, restart_hres_watchers.sh
  idempotent.
- hres, smoke on study R2's teacher (STUDY_3D_HRES.md §10): hres_smoke_r2.sh (record, train with --spread 2 --edge 10,
  TRAIN_DEVICE=cuda, finish, bounds, edge; hres_smoke_r2_resume_cuda.sh and hres_smoke_r2_train.sh); hres_status.sh
  reports smoke_r2's runs, training, models, bounds and edge check.
- docs/PLAN_PC_GPU.md and docs/WAVE1_CHECKLIST.md: stage 2 on the PC (one reference per effect, FUSE on Desktop, no
  footage); results/3d/pc.md (P0 snapshot).
- Q2's jobs as scripts (tools/study_q2/; STUDY_Q2.md §0, STUDY_Q2_RESTORE.md new); bootstrap.sh as a checklist with
  a dry run (Q2_DRY=1).
- R2's jobs as scripts (tools/study_3d/r2_*.sh; STUDY_R2.md §0, §7; STUDY_R2_RESTORE.md new). CUDA trainer: MSE on
  GPU, spread and edge on CPU in the same step (`rollout3d_train_cuda_host.cpp`, `rollout3d_train.hpp`).
- .gitignore: CTest's Testing/ directories. .gitattributes (new): *.sh and *.awk text eol=lf, so the scripts check
  out LF whatever core.autocrlf says; the 16 scripts still CRLF in the index outside these commits are renormalized in
  a separate endings-only commit (checklist item 6).
- TODO.md: "Done since 24 October 2026" (the whiteout and drifting snow among them, the fire's test written up),
  "Under way", "Next", the pointer to docs/WAVE1_CHECKLIST.md under "Order of work". docs/WAVE1_CHECKLIST.md (new: the
  per-effect ledger of wave 1, what the container has done and what the PC owes), COMMIT_PLAN.md (this plan),
  tools/ship_wave1.sh (its git adds as a script; DRY_RUN=1 by default) and tools/preflight_wave1.sh (the fast
  pre-push check; no git state touched).
```

Files (58; tracked and untracked rows of `git status`, `.gitattributes` and `results/manager/half_128/` with its `run2/` among them):

```
.gitattributes
.gitignore
COMMIT_PLAN.md
TODO.md
docs/PLAN_PC_GPU.md
docs/STUDY_3D_HRES.md
docs/STUDY_Q2.md
docs/STUDY_Q2_RESTORE.md
docs/STUDY_R2.md
docs/STUDY_R2_RESTORE.md
docs/WAVE1_CHECKLIST.md
include/neuralfx/rollout3d_train.hpp
results/3d/pc.md
results/3d/hres/fire/compare_test_learned_vs_N2.csv
results/3d/hres/fire/compare_test_learned_vs_N2.md
results/3d/hres/fire/compare_test_learned_vs_coarse.csv
results/3d/hres/fire/compare_test_learned_vs_coarse.md
results/3d/hres/fire/edge_test.csv
results/3d/hres/fire/edge_test_N2.csv
results/3d/hres/fire/edge_test_learned.csv
results/3d/hres/fire/edge_test_sim.csv
results/3d/hres/fire/edgecheck_test.md
results/3d/hres/fire/edgecheck_test_single.md
results/manager/half_128/ab_summary.csv
results/manager/half_128/ab_summary.txt
results/manager/half_128/load.txt
results/manager/half_128/run2/ab_summary.csv
results/manager/half_128/run2/ab_summary.txt
results/manager/half_128/run2/load.txt
results/manager/half_128/run2/runs.csv
results/manager/half_128/runs.csv
src/train/rollout3d_train.cpp
src/train/rollout3d_train_cuda_host.cpp
tools/manager/time_half_128.sh
tools/preflight_wave1.sh
tools/ship_wave1.sh
tools/study_3d/hres_fire_recheck_test_edges.sh
tools/study_3d/hres_fire_validate.sh
tools/study_3d/hres_smoke_r2.sh
tools/study_3d/hres_smoke_r2_resume_cuda.sh
tools/study_3d/hres_smoke_r2_train.sh
tools/study_3d/hres_status.sh
tools/study_3d/hres_sync_fire_test_results.sh
tools/study_3d/r2_bootstrap.sh
tools/study_3d/r2_env.sh
tools/study_3d/r2_resume.sh
tools/study_3d/r2_status.sh
tools/study_3d/r2_test.sh
tools/study_3d/r2_train.sh
tools/study_3d/r2_validate.sh
tools/study_3d/restart_hres_watchers.sh
tools/study_3d/run_fire_validate.sh
tools/study_q2/after_val.sh
tools/study_q2/bootstrap.sh
tools/study_q2/common.sh
tools/study_q2/restore_from.sh
tools/study_q2/resume.sh
tools/study_q2/val.sh
```

```sh
git add .gitattributes .gitignore COMMIT_PLAN.md TODO.md docs/STUDY_3D_HRES.md docs/STUDY_Q2.md docs/STUDY_Q2_RESTORE.md docs/STUDY_R2.md \
  docs/STUDY_R2_RESTORE.md docs/WAVE1_CHECKLIST.md \
  results/3d/hres/fire/ results/manager/half_128/ src/train/rollout3d_train.cpp tools/manager/time_half_128.sh \
  tools/preflight_wave1.sh tools/ship_wave1.sh tools/study_3d/hres_fire_recheck_test_edges.sh tools/study_3d/hres_fire_validate.sh \
  tools/study_3d/hres_smoke_r2.sh tools/study_3d/hres_smoke_r2_train.sh tools/study_3d/hres_status.sh \
  tools/study_3d/hres_sync_fire_test_results.sh tools/study_3d/r2_bootstrap.sh tools/study_3d/r2_env.sh tools/study_3d/r2_resume.sh \
  tools/study_3d/r2_status.sh tools/study_3d/r2_test.sh tools/study_3d/r2_train.sh tools/study_3d/r2_validate.sh \
  tools/study_3d/restart_hres_watchers.sh tools/study_3d/run_fire_validate.sh tools/study_q2/
git status --short   # expect nothing left
```

31 + 21 + 58 = 110: after the three `git add`s, `git diff --cached --stat` over all three must read the same 110 files as
`git status -uall` did before "Before anything", and `git status --short` is empty (on WSL; a Windows git that has
re-read the 16 CRLF scripts under `.gitattributes` shows them ` M` until the endings-only commit of checklist item 6).

The scripts under `tools/study_q2/` and `tools/study_3d/r2_*.sh` must keep their executable bit (`git ls-files -s`
shows `100755`); this checkout has `core.filemode false`, so check the mode in the index, not on disk, and fix any
`100644` with `git update-index --chmod=+x <file>` before the commit. The other hres scripts are `100644` in HEAD and
stay so (they are run with `bash` or `sh`). The five new untracked scripts take the mode `git add` gives them on this
PC (`100644`), so after commit 3's `git add`:

```sh
git update-index --chmod=+x tools/manager/time_half_128.sh tools/preflight_wave1.sh tools/ship_wave1.sh \
  tools/study_3d/hres_smoke_r2.sh tools/study_3d/hres_smoke_r2_train.sh
```

(`time_half_128.sh`'s siblings `time_crowd.sh` and `time_halfsize.sh` are `100755`.) `bash -n` on each `.sh` first (the
header of `hres_smoke_r2.sh` asks for it after any edit; `preflight_wave1.sh` does it).

## Pre-push checklist (on the three commits, from `build/`)

**Git identity.** The WSL checkout that runs the ship has no committer identity (`git var GIT_COMMITTER_IDENT` fails:
"empty ident name"; `git config --get user.name` and `user.email` are both unset there, 24 October 2026). A live
`DRY_RUN=0 FORCE=1 bash tools/ship_wave1.sh` lists that as a blocker in step 0 and stops before its first git command;
`git commit` by hand would fail the same way. Before the live ship the **owner** sets both, in WSL, in this checkout
(omit `--global` to keep it to the repo):
```sh
git config user.name "Your Name"
git config user.email "you@example.com"
git var GIT_COMMITTER_IDENT        # must print the identity, not "empty ident name"
```
The agent does not run `git config` (identity is the owner's, like the commits). The dry run (`DRY_RUN=1`, the
default) commits nothing, so a missing identity is a note there, not a blocker: a dry run that ends "nothing changed"
with no BLOCKERS block is a pass whether or not the identity is set yet.

The same commands every merge into main has used (TODO.md "Done since 15 October 2026"). Item 2 is the optional fast
version; it replaces none of the others.

1. **Configure and build everything**, tests on (the build dir is configured with `g++-14`):
   ```sh
   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DNEURALFX_BUILD_TESTS=ON
   cmake --build build -j
   ```
2. **Optional, the fast check first** (about a minute on a built tree, not the suite), from anywhere, on WSL:
   ```sh
   bash tools/preflight_wave1.sh                  # build/ is the default build dir
   NO_BUILD=1 BUILD=build-cuda bash tools/preflight_wave1.sh   # another build dir, already built
   ```
   Six steps: `bash -n` over `tools/study_3d/*.sh`, `tools/study_q2/*.sh`, `tools/manager/*.sh`, `tools/ship_wave1.sh`
   and itself (a script with CRLF line endings is named as such: bash reads `do\r` as a command, item 6's point);
   `cmake --build` of the four programs it runs; then `neuralfx_weather_tests`, `neuralfx_tests
   --gtest_filter='Library.*'`, `nvfx_alloc_test` (item 4) and `nvfx_gallery --library library --check --threads 2`
   (item 5). Every step runs even after a failure (a program that did not build is skipped and counted); it ends
   `ok: all six steps passed` with status 0, or a list of the failures with status 1. A failure here is one item 3
   would find later, found sooner. It touches no git state (no add, commit, stash, reset or push), so it can also be
   run on the working tree before "Before anything". It does not replace items 3 to 5: the manager, 3D, pack,
   wrapper and frozen-keyframe tests are not in it. On this PC's working tree (24 October 2026) it passes in 34 s
   with the programs built. `tools/manager/time_crowd.sh` was CRLF in HEAD and the index (`git ls-files --eol`
   reads `i/crlf`), so `bash -n` had never passed on it; the working copy is now LF (`perl -pi -e 's/\r\n/\n/g'`,
   24 October 2026) and `bash -n` passes with no `tr -d '\r'` strip. Because the index is CRLF, commit 1's
   `git diff --cached --stat` on this one file shows every line rewritten (39 insertions, 68 deletions); that is the
   intended LF conversion, not item 6's failure. `git diff --cached --ignore-cr-at-eol -- tools/manager/time_crowd.sh`
   shows the real change (the header shortened, the awk summary moved to `time_crowd_summary.awk`).
   `time_half_128.sh` and `time_halfsize.sh` were already LF.
3. **The full test suite, serially** (the library and scene tests share keyframe directories under `build/`, and the
   timing tests want a quiet core):
   ```sh
   cd build && ctest -j1 --output-on-failure
   ```
   Expect 602 tests (`ctest -N -L neuralfx`), every one passing or skipping with code 77 (the data-labelled tests skip
   without the models under `NEURALVFX_DATA` or `models/`); `neuralfx.Coupled3dVideoSmoke` exists only where ffmpeg was
   on the path at configure time. The working tree's last full run on this PC read **570 passed, 0 failed, 32 skipped**
   ("Verified" above); the commits must give the same. Against the last merge's (study Q2's: 555 of 555) the new tests
   are `Sim3d.CoupledStack*`,
   `VolumeApi.CoupledDemoFeedsTheSmokeFromTheFiresSoot`, `VolumeApi.SameFieldsAndPicturesOnAnEnginesJobSystem`,
   `Manager.HalfSizePerEffectDefaults`, the five library tests, the weather director's eleven (39 in all),
   `Volume3dToolUsage`, `Coupled3dVideoSmoke`.
4. **No allocation per frame**, the volume's job hook included: `./nvfx_alloc_test` ends `ok: no allocation per frame`.
5. **The library catalogue parses, plays and its frozen keyframes match**: `./nvfx_gallery --library ../library --check`
   ends `0 failed` (240 entries). `LibraryKeyframesFrozen` in ctest covers the hashes where the models exist
   (`steam_fog`'s and `coloured_smoke_grenade`'s rows are the two added to `results/library/frozen.csv`, both in its
   `--only`); `drifting_snow`'s is in `frozen_round3.csv` and checked by `LibraryProceduralKeyframesFrozen` without
   models. **Not frozen, on purpose:** `contrail`, `helicopter_downwash` and `whiteout` have no row in any frozen CSV
   and are in no `--only` list (LIBRARY.md §4 and §10.4 say so beside the entries). They play learned models, were
   written on a machine without them and have not been judged by eye; a hash frozen before that would pin an unseen
   frame. Do not add rows for them before the push, and do not read their absence as an omission: `frozen.csv` must
   gain exactly the two rows above against HEAD (`git diff --cached --stat` reads `+2`), `frozen_round3.csv` exactly
   one. Freezing the three is a follow-up where the models are, with the command in LIBRARY.md §4.
6. **Line endings.** `core.autocrlf` is `true` on this PC. `git diff --cached --stat` on each commit must not show whole
   files rewritten; if it does, the file was converted to CRLF (`git diff --cached --ignore-cr-at-eol` shows the real
   change). The one exception is `tools/manager/time_crowd.sh` in commit 1, CRLF in the index and LF in the tree (item
   2): its whole-file diff is the conversion the other way and is wanted. `results/library/frozen.csv`,
   `frozen_round3.csv` and the `.sh` scripts in particular must stay LF (`git diff`
   on this PC already warns "LF will be replaced by CRLF" for a dozen files: that is the working copy, not the index,
   and is harmless as long as the staged diff is the real change). **`.gitattributes`** (root, new, untracked) pins
   `*.sh text eol=lf` and `*.awk text eol=lf`, so the scripts check out LF on this PC whatever `core.autocrlf` says
   and `bash -n` (item 2) no longer depends on OneDrive leaving the endings alone. It takes effect in the working
   tree only after a renormalize:
   ```sh
   git add --renormalize -- '*.sh' '*.awk'   # then: git ls-files --eol -- '*.sh' '*.awk' reads i/lf throughout
   ```
   Scope that before running it: `git ls-files --eol -- '*.sh'` on HEAD shows 18 scripts still `i/crlf` beyond
   `time_crowd.sh` (`engines/godot/test.sh`, `tools/detail_bench/*.sh`, `tools/opt2/*.sh`, `tools/study_h/*.sh`,
   `tools/study_k/time.sh`), none of them in the three commits; an unscoped renormalize stages all 18 as whole-file
   rewrites. Either name only the scripts in the commit at hand on the `--renormalize` line, or make the 18 a separate
   endings-only commit after the three (the diff reads clean under `--ignore-cr-at-eol`). `.gitattributes` is
   **commit 3's** (its Files list and message name it; `ship_wave1.sh`'s `FILES3` too), beside the scripts it governs;
   commit 1's `time_crowd.sh` is LF on its own (the working copy was converted by hand, item 2), so it needs no
   attribute to check out LF, and the three land on one push. The endings-only commit for the 18 comes after.
7. **The Godot test** (`engines/godot/test.sh`), where a Godot build exists; skip and say so otherwise.
8. `git log --oneline -3` reads as the three messages above; `git status --short` is clean.

Do **not** run `git commit` or `git push` from this plan: it lists the commits; the owner makes them.
`tools/ship_wave1.sh` prints the same `git add`s with `DRY_RUN=1` (its default) and commits only with `DRY_RUN=0 FORCE=1`,
which is the owner's call too.
