# Wave 1 checklist: the tier 1 effects, what this container has done and what waits for the PC

**Status:** a checklist, written 24 October 2026. It is the per-effect ledger for [TODO.md](../TODO.md) "Order of
work", step 1 (wave 1, tier 1) as [PLAN_PC_GPU.md](PLAN_PC_GPU.md) would make it: the PC's GPU simulating, training and
playing each effect at 64 x 96 x 64 with 256 x 384 x 256 detail, 5 to 9 hours an effect, after each family's teacher has
passed its physical checks (PLAN_PC_GPU.md §3.4). **Owner direction (24 October 2026):** run **this checklist to
completion on the PC** via PLAN_PC_GPU §8 (all gas/liquid/grain rows with a GPU teacher, then P7 weather rows)—not
stop at hres. **Under way on the PC:** hres **smoke_r2** (STUDY_3D_HRES.md §10), learning from hres **fire** (shipped,
passes edge check) and **smoke_hres** (shipped, fails edge check) before P3. **Not started:** P0 (`nvfx_pc`, GPU sim,
train, play; PLAN_PC_GPU §5)—blocked on §2 PC details for P0, not on abandoning the plan. The catalogue's rows, numbers
and checks are TODO's; this file only says, row by row, what exists here and what the PC still owes.

**Reading the columns.**
- **Container-done**: what exists in this repository today for that row: the 2D model it maps to (`models/2d/`), its
  library entries (`library/*.nvfxl`, [LIBRARY.md](LIBRARY.md)), the 2D weather director's part of it
  ([WEATHER.md](WEATHER.md)), and any 3D study. A 2D effect counts as done only as **the far level of detail**
  (PLAN_PC_GPU.md §9); it is not the 3D effect.
- **PC-pending**: the phase of PLAN_PC_GPU.md §8 that makes the 3D effect, its teacher (§3.2, §3.3) and the check it
  must pass first (§3.4).
- **Scripts and docs**: the study automation that exists for the row (all under `tools/`), and where its numbers are.
  Where none exists the cell says so: the PC's `nvfx_pc effect <name>` (PLAN_PC_GPU.md §5) does not exist yet either.

**Phases, for the table:** P0 the PC set up and measured; P1 the GPU trainer, GPU `sim3d` and GPU player, with N2's,
hres's and R's models retrained from their own data; P2 the gas teacher checked at three resolutions; P3 the gas family
model and the first gate (fire and smoke); P4 the other gas effects, one a night; P5 the GPU FLIP and the liquids; P6
the GPU MPM and the grains; P7 TODO's weather system and the rest of the catalogue (the clouds, fog, rain and snow rows
are P7: their teachers are the research codes, outside the hours target).

## 1. Fire, smoke and blasts (the gas family; the first gate)

| # | effect | container-done | PC-pending | scripts and docs |
|---|---|---|---|---|
| 0 | pool fire | **The most advanced row.** 2D `fire` (rows 0 and 1). 3D: study N's fire (16 x 24 x 16; tracks a real run better than the coarse simulation at 1 s, +1.1 dB in temperature; `models/3d/fire_study_n.nvfx3`), N2's C4 (no runaway over 30 s; `models/3d/fire.nvfx3`, shown by the owner's choice after missing one edge bound by 3%), **hres's C2** (32 x 48 x 32; tested once and frozen: +1.69 dB [+1.07, +2.32] in pictures at 1 s over the coarse simulation with detail, the mean picture +2.20 dB [+1.28, +3.14] over N2's C4, motion 0.99 of the real runs'; its test edge check passes at every wind with the bounds calibrated on four teacher runs; `models/3d/fire_hres.nvfx3`). The teacher is our `sim3d` fire, **not a validated fire code**: it puffs at 0.5 to 0.65 Hz where Heskestad's and the 1.5/√D rule put a fire of its size near 2.7 Hz (STUDY_3D.md §13). The coupled fire and smoke exist as demos (`sim3d::Stack`, `--kind coupled-sim`; the volume-API coupling, `coupled3d.hpp`). | P1: retrain N2's and hres's fire on the GPU from their own data; `nvfx_gpu_sim` from `sim3d`. P2: the fire teacher at 128, 256 and 512 across: flame height (Heskestad), McCaffrey's profiles, puffing near 1.5/√D; a finer grid is the first thing to try for the puffing; one reference: **FDS** (`reference.md` per §3.3.1). **P3: the first gate** (§6.4: beat the coarse simulation, its own CPU version, N2's C4 and hres; no runaway over 60 s; the chain in 9 hours; the step within the GPU budget in FUSE). | `tools/study_3d/hres_run_all.sh`, `hres_fire.sh`, `hres_fire_validate.sh` (via `run_fire_validate.sh`), `hres_fire_edge_bounds.sh`, `hres_fire_recheck_edges.sh`, `hres_fire_recheck_test_edges.sh`, `hres_fire_after_validate.sh`, `hres_publish_models.sh`, `hres_sync_fire_test_results.sh`, `hres_status.sh`, `restart_hres_watchers.sh`; the N2 study ran from `nvfx_3d` by hand (STUDY_3D.md §14). Numbers: `results/3d/n2/`, `results/3d/hres/fire/` (`frozen.csv`, `choose_val.md`, `edge_calibration.md`, `edgecheck_test.md`, `compare_test_*`); [STUDY_3D.md](STUDY_3D.md), [STUDY_3D_HRES.md](STUDY_3D_HRES.md), REPORT §15 and its hres addendum; `videos/README.md` §9. |
| 1 | campfire and bonfire | 2D `fire`; library `campfire`, `bonfire`, `brazier`, `burning_ground`, `carried_torch` and the rest of `fire.nvfxl`; `campfire`'s keyframe frozen (`LibraryKeyframesFrozen`). No crib or log-pile teacher: `sim3d` has a burner, no solid pyrolysis. | P3 or P4 (gas: fire and combustion): a fine-tune of the family model with a crib-shaped source; one reference: **FDS** solid-pyrolysis crib case (§3.3.1), plus #0's family checks. | None of its own; it rides #0's. [EFFECTS.md](EFFECTS.md) for the 2D fire; LIBRARY.md §1 "Fire". |
| 5 | grass fire | 2D `wildfire` (a spreading front; EFFECTS_BLAZE.md); library `grass_fire`, `burning_field`, `forest_fire_front`, `distant_forest_fire`, `torching_trees`; the forest-edge town scene (round 10). No 3D front. | P4: a spreading front needs the family model with a fuel field and a level set or a burning-rate map; one reference: **CSIRO grassland** rate-of-spread data (§3.3.1). | None of its own. [EFFECTS_BLAZE.md](EFFECTS_BLAZE.md) (the 2D wildfire), [EFFECTS_FIRE2.md](EFFECTS_FIRE2.md) (the crown and window fires, tier 2). |
| 16 | smoke column | 2D `smoke`; library `smoke_column`, `chimney_smoke*`, `signal_smoke`, `black_oil_smoke`, `smoke_screen`. 3D: **study R's teacher `smoke_r` is a plume by MTT's measures** (buoyancy flux conserved, alpha 0.14 to 0.15, centreline exponents within 0.1 of MTT's) but a third too slow for its flux; **its learned smoke fails its test** (tracks at 1 s +1.59 dB over the coarse simulation, but spreads too wide played on, the mean picture -2.52 dB; one edge bound missed by 7%) and ships as a labelled demo (`models/3d/smoke.nvfx3`). **Study R2 is under way**: the teacher's momentum deficit fixed (`smoke_r2`, dM/dz over B 1.088), the model retraining with a spread loss and an edge loss under a rule committed before validation (its margins from the teacher's own spread); stage 4 and the rule's teacher runs were left running on the container. hres's smoke (study N's preset as teacher) fails the edge check at three winds and ships as a demo (`models/3d/smoke_hres.nvfx3`). | P1: retrain R's smoke on the GPU. P2: `smoke_r2` at three resolutions; one reference: **MTT** (`reference.md` per §3.3.1). **P3: the first gate with the fire** (beat the coarse simulation, its CPU version and R's C4; no edge runaway over 60 s). Until R2 reports, the smoke's edge and spread failure is the open fault the PC inherits (STUDY_3D_HRES.md §8). | `tools/study_3d/r2_env.sh`, `r2_bootstrap.sh`, `r2_resume.sh` (the one command; STUDY_R2.md §0), `r2_train.sh`, `r2_validate.sh`, `r2_test.sh`, `r2_status.sh`; the container's originals in `tools/study_r2/`; `hres_smoke.sh`, `hres_smoke_validate.sh`, `hres_smoke_after_train.sh`, `hres_smoke_resume_cuda.sh`; new and untracked, `hres_smoke_r2_train.sh` (smoke at hres on R2's teacher, a dry run by default). Numbers: `results/3d/r/`, `results/3d/r2/` (`margins.md`, `edge_calibration.md`), `results/3d/hres/smoke/`; [STUDY_R.md](STUDY_R.md), [STUDY_R2.md](STUDY_R2.md), STUDY_3D_HRES.md §8, REPORT §20 and §24. |
| 23 | high-explosive ground burst | 2D `explosion` (v2; study M3's renderer); library `large_blast`, `medium_explosion`, `small_explosion`, `frag_grenade`, `mushroom_blast`, `airburst`, `nuke`, `ground_slam`, `debris_shower`; the fireball demo (`nvfx_fireball`). Study O's mushroom replaced by a held-out test; Q2's dust-explosion and mushroom renderers retrained. No shock, no crater: the 2D model starts after the blast. | P4 (gas: blasts): the family model after the shock, from `nvfx_gpu_sim`; one reference: **mass-scaling and scaled-distance correlations** (§3.3.1); the crater and dust skirt wait for P6's grains. | None of its own. [EFFECTS.md](EFFECTS.md), [EFFECTS_FLOW.md](EFFECTS_FLOW.md) (the mushroom), [STUDY_Q.md](STUDY_Q.md) and [STUDY_Q2.md](STUDY_Q2.md) §0 (the mushroom's cap still ragged; its one test pending through `tools/study_q2/`). |

## 2. Weather (the research codes; P7, outside the hours target)

| # | effect | container-done | PC-pending | scripts and docs |
|---|---|---|---|---|
| 41 | fair-weather cumulus | 2D `cloud` (study M3's family); library `sky_cloud`, `storm_cloud`. The 2D weather director's clouds are **puffs, not a deck**: the cloud fraction per regime, and since 24 October the cloud base by Espy's rule from the dew point (`sky.base`: fair-weather cumulus about 1200 m; WEATHER.md §3.11), which nothing draws yet. There is no "fair weather" regime: the director's seven are clear, overcast, rain, storm, snow, fog and dust storm. | P7: PALM or MicroHH on BOMEX; cloud fraction, liquid water, base and top against the intercomparison; the regional model of TODO's "Weather systems". The deck drawn in 2D first (TODO "The weather director, next"). | None. [WEATHER.md](WEATHER.md) §2.1 (the regimes' targets), §3.11; [EFFECTS_ATMOSPHERE.md](EFFECTS_ATMOSPHERE.md) (the 2D cloud). |
| 44 | stratus and overcast | The director's overcast regime (cloud 0.95, a 560 m base, visibility 40 km, daylight 0.37 by Kasten and Czeplak); no deck drawn. | P7: as #41, on DYCOMS-II RF01. | None. WEATHER.md §2.1, §4.6. |
| 51 | rain and rainshafts | 2D `rainshaft` (study O fixed its banding); library `rain`, `rain_shaft`, `rain_shaft_drawn`, `downpour`, `drizzle`, `rain_on_puddles`; the director's rain and storm regimes with Marshall and Palmer's drop sizes, Atlas, Srivastava and Sekhon's fall speeds (checked against Gunn and Kinzer), the rain's extinction, seeded rain cells, and since 24 October the wet ground (depression storage 1.5 mm, Penman's drying; `sky.water`, `sky.wet`, `sky.puddle`, `sky.evaporation`; WEATHER.md §3.12), not drawn yet. | P7: PALM or MicroHH, CM1 for storms; the outflow front against gravity-current theory; the shafts below a cloud base in 3D. The `frame wet` and `frame puddle` keys in 2D first. | None. WEATHER.md §3.2 to §3.4, §3.12; [EFFECTS_WATER.md](EFFECTS_WATER.md) (the 2D rain shaft). |
| 53 | snowfall | The director's snow regime (flake sizes exponential on 2 to 10 mm at 1 mm/h, Locatelli and Hobbs's fall speeds; WEATHER.md §3); library `snow`, `snow_flurry`, `blizzard`, `drifting_snow`, `sleet`, `snow_slide`. All 2D, flakes as particles. | P7: PALM or MicroHH with our flake dynamics; fall speeds against Locatelli and Hobbs in 3D. | None. WEATHER.md §3. |
| 58 | radiation fog | 2D `fog` (study M3's family); library `ground_fog`, `morning_mist`, `swamp_mist`, `stage_fog`, `frost_mist`; the director's fog regime (Koschmieder's visibility, the dew point at the ground: `sky.dewpoint`; fog keeps the ground wet, as Penman's zero deficit says). | P7: PALM on a measured fog night (Cabauw or Cardington): liquid water and depth. | None. WEATHER.md §3.1, §3.11; EFFECTS_ATMOSPHERE.md. |
| 62 | haze and smog | The director's `visibility` and `extinction` (Koschmieder) veil the day scene's hills (`WeatherDay.TheHillsFadeIntoTheHazeAsKoschmiederSays`). No aerosol model, no sky model. | P7 with #130: aerosol optical depth against measurements, through the sky model. | None. WEATHER.md §3.1, §4.6. |
| 73 | gusts and turbulent wind | The director's gusts (Kaimal's integral time scale at 10 m, IEC 61400-1; 3-second means; the spread and autocorrelation checked against the process), Beaufort lean and leaves (`sky.beaufort`, `sky.lean`, `sky.leaves`); the 2D effects take the wind as a control; library `wind_gust`, `falling_leaves`, `cherry_petals`. | P7: PALM's boundary layer; spectra against Kaimal and von Kármán; the wind field around buildings and trees in 3D. | None. WEATHER.md §3 (the gusts row), [COMPOSE.md](COMPOSE.md) §14 (the gusts' costs). |
| regimes | clear, fair weather, overcast, steady rain, snowfall, fog | Five of the six exist in the 2D director (clear, overcast, rain, snow, fog), with ordered transitions, seeded variability per request, moist air and wet ground in the state; fair weather is not a regime of its own. One place, in 2D. | P7: the regional model (TODO "Weather systems") from PALM, MicroHH and CM1 runs on BOMEX, DYCOMS-II and the idealised cases; the local effects fed from it; the weather API (`nvfx_weather.h`, not started). | `tests/test_weather_director.cpp` (31 tests); the day scene and its video (`videos/README.md`); WEATHER.md. |

## 3. Water (the free-surface family; P5, after the GPU FLIP)

| # | effect | container-done | PC-pending | scripts and docs |
|---|---|---|---|---|
| 78 | open ocean | Library `whitecaps`, `shore_waves`, `storm_surf`, `boat_wake`, `waterspout` built from the water system's pieces (EFFECTS_WATER.md: sheets, jets, drips, bubbles, pools, ripples). **No spectral wave solver**: the catalogue's "exact spectral waves with seeded phases" (JONSWAP, Pierson-Moskowitz) does not exist here. | P5 and later: the ocean family is its own (TODO "The models": exact spectral waves, learned breaking and whitecaps); Basilisk for the breaking; whitecap coverage against Monahan. Not in PLAN_PC_GPU.md's hours: it has no GPU teacher planned. | None. [EFFECTS_WATER.md](EFFECTS_WATER.md). |
| 82 | waterfall | 2D `waterfall` (study O fixed its banding); library `waterfall`, `waterfall_drawn`, `tall_waterfall`, `wide_falls`, `cascade`; `waterfall_drawn`'s keyframe frozen. | P5: `nvfx_gpu_flip` (new; about a week to build), one reference per row: **Basilisk** falling sheet (§3.3.1); break-up length in §3.4. | None of its own. EFFECTS_WATER.md, [STUDY_P.md](STUDY_P.md) (half size on the waterfall). |
| 83 | fountain | 2D `fountain`; library `fountain`, `garden_fountain`, `broken_hydrant`, `water_geyser`; `fountain`'s and `garden_fountain`'s keyframes frozen. | P5: as #82; trajectory with drag and Rayleigh-Plateau break-up. | None of its own. EFFECTS_WATER.md. |
| 85 | splash | 2D `splash` (study O replaced it by a held-out test, +0.40 dB [0.21, 0.60]); library `splash`, `boulder_splash`, `dive_splash`, `cannonball_splash`; `boulder_splash`'s keyframe frozen. | P5: as #82; the splashing threshold (Mundo, Sommerfeld and Tropea) and Duez's sphere-entry cavities. | None of its own. EFFECTS_WATER.md, REPORT §16 (study O). |
| 86 | rain on water and puddles | Library `rain_on_puddles` (the water system's ripples: rings that grow and fade, flattened by the view); the director's puddles as a number (`sky.puddle`), not drawn. | P5: Basilisk's crowns and rings; ring speeds against the capillary-gravity dispersion relation. | None. EFFECTS_WATER.md (ripples), WEATHER.md §3.12. |
| 91 | bubbles and bubble plumes | Library `underwater_bubbles`, `fish_tank`, `blood_in_water`, `dye_plume`, `silt_cloud` (the water system's `bubbles`: rise, wobble, grow, pop at the surface; a rim and a glint; 0.013 ms a frame for 12 a second). Seeded particles, no learned model, by principle 1. | P5: Basilisk; terminal velocities and shapes against Clift, Grace and Weber for the statistics the particles are seeded from. | None. EFFECTS_WATER.md §bubbles. |

## 4. Dust and dense gases (gas: dense and laden, and the grains; P4 and P6)

| # | effect | container-done | PC-pending | scripts and docs |
|---|---|---|---|---|
| 100 | dust burst | 2D `dust` (study M3: dust with a mean state, EFFECTS_ATMOSPHERE.md §9); library `dust_impact`, `landing_dust`, `sand_burst`, `rock_slide_dust`, `bullet_impact_dirt`, `dust_motes`; `dust_impact`'s keyframe frozen. | P4 (gas: dense and laden): `nvfx_gpu_sim` with density; settling against measured terminal velocities, the front against particle-laden gravity currents. The thrown grains are P6's MPM (the grains in dust). | None of its own. EFFECTS_ATMOSPHERE.md, REPORT §14 (study M3). |
| 101 | vehicle dust wake | Library `vehicle_dust_wake` (the dust model behind a moving source), keyframe frozen; `helicopter_downwash` and `hover_downwash` beside it. Consistency / seeded dust stats (no separate reference). | P4: as #100 with a moving source; one reference: **LES + particle** case or consistency (§3.3.1; no footage). | None. LIBRARY.md §1 "Dust". |
| 118 | dry-ice and liquid-nitrogen fog | 2D `heavygas` (round 9 air: **falls short**, its mean picture ties the nearest flipbook's; study Q fixed its bright lines by eye, its mean picture still weak); library `dry_ice_fog`, `poison_gas`, `poison_cloud`, `toxic_spill`, `swamp_gas`, `coloured_smoke_grenade`'s dense cloud. | P4 (dense flows): `nvfx_gpu_sim` with density; the front speed against the gravity current's Froude number (the 2D haboob's runs at Fr 1.15 against Huppert and Simpson's 1.19). | None of its own. [EFFECTS_AIR.md](EFFECTS_AIR.md), STUDY_Q.md (heavy gas), STUDY_Q2.md §0 (heavy gas's validation pending: `tools/study_q2/after_val.sh heavygas` after `restore_from.sh`). |

## 5. Optics (computed, not learned)

| # | effect | container-done | PC-pending | scripts and docs |
|---|---|---|---|---|
| 130 | sky and sun | The director's `daylight` (Kasten and Czeplak's dimming under cloud, WEATHER.md §3.7) and the 2D day scene's sky. **No atmospheric scattering model** (Bruneton). | The sky model, in FUSE or a precomputed table; radiance against the CIE standard skies. Not in PLAN_PC_GPU.md's phases: it is rendering, not simulation. | None. WEATHER.md §4.6. |
| 131 | crepuscular and god rays | `render3d`'s single scattering through the 3D volumes (STUDY_3D.md: Planck's law, CIE 1931, soot absorption); the designed `holy_beam` and `resurrection_light` are not this. | Light shafts through the clouds and smoke in FUSE's volumetric renderer, from the volumes the API exposes (`nvfx_volume.h` fields). | None. STUDY_3D.md (the renderer), [FUSE.md](FUSE.md). |
| 135 | heat haze and mirages | Library `heat_haze`, `desert_shimmer`, `exhaust_shimmer` (2D, a refraction look on the temperature field). | Refraction from the 3D temperature field (Gladstone-Dale) in the engine; mirage geometry against the temperature profile. | None. LIBRARY.md. |
| 137 | cloud shadows | Nothing. | From the cloud volumes once #41 and #44 exist in 3D. | None. |
| 138 | fire and lightning light on the scene | The volume renderer's emission; the FUSE host relights the 3D fire's picture over its background with its own point light (ENGINES.md §12). The 2D director's lightning `flash` and `light` keys. | Emission as a light source in FUSE, flickering with the effect (TODO "Runtime and engines: for FUSE"). | None. ENGINES.md §12, FUSE.md. |

## 6. The wave's infrastructure

| item | container-done | PC-pending | docs |
|---|---|---|---|
| the volume API | **Done and growing**: `nvfx_volume.h` (NVFX_VERSION 6), load, create, seed, controls, step, fields, a CPU picture; no allocation per frame, SIMD, int8, threads and shards; **the job hook** `nvfx_volume_create_with_jobs` (NVFX_VERSION 10, 24 October 2026; the same frames to the bit on an engine's job system); the coupled demo through two volumes. | The GPU step of a `.nvfx3` in FUSE (PLAN_PC_GPU.md §9: compute shaders on sparse bricks, fields kept on the GPU, integer arithmetic for determinism); a start spread over frames. | ENGINES.md §12, INTEGRATION.md §3, FUSE.md §9; `tests/test_3d.cpp` (`VolumeApi.*`), `tests/alloc_test.cpp`. |
| sparse volumes | **Not done.** The detail layer is dense; REPORT §19 found no empty 4 x 4 x 4 brick to skip at study N's size, so the gain came from vectors instead. | Sparse bricks at 256 x 384 x 256, where 30 to 40% of the volume is filled (PLAN_PC_GPU.md §9); the brick atlas for the engine. | REPORT §19, STUDY_3D.md §13. |
| the manager for volumes | **Not done.** The 2D effect manager (budget, sharing, proxies, interpolation; half size per effect since 24 October, NVFX_MANAGER_VERSION 3) takes sprites only. | The manager choosing among the three levels of detail (GPU volume, CPU volume, 2D sprite) within a budget (PLAN_PC_GPU.md §9). | [MANAGER.md](MANAGER.md), REPORT §22. |
| the 3D scene language | **Not done.** The scene language is 2D (COMPOSE.md); the 3D coupling exists as two demos (`sim3d::Stack` in the solver, `coupled3d.hpp` through the volume API). | Positions, sizes, wind and cameras as 3D vectors, modules as volumes, the field bus as a 3D grid; a 3D scene C API (TODO "3D scenes and the 3D library"). | COMPOSE.md, STUDY_R.md §6.1. |
| the pipeline (`nvfx_pc`) | **Not done.** Its pieces exist as study scripts with the same shape (markers, resume, a rule committed before validation, a null probe, one test): `tools/study_3d/hres_*.sh` and `r2_*.sh`, `tools/study_q2/*.sh` (STUDY_Q2_RESTORE.md and STUDY_R2_RESTORE.md: `NEURALVFX_DATA`-relative, run from any machine; what to copy from the container and what to run after). The ship of 24 October has its own two: `tools/preflight_wave1.sh` (the fast pre-push check: `bash -n` over the study and ship scripts, the weather and library tests, `nvfx_alloc_test`, the catalogue check; minutes, no git state touched) and `tools/ship_wave1.sh` (COMMIT_PLAN.md's three `git add`s and its checklist as a script, `DRY_RUN=1` by default). | `nvfx_pc effect <name>` and `nvfx_pc family <name>` (PLAN_PC_GPU.md §5), logging each step's time to `results/3d/<effect>/time.csv`. | PLAN_PC_GPU.md §5, [STUDY_Q2_RESTORE.md](STUDY_Q2_RESTORE.md), [STUDY_R2_RESTORE.md](STUDY_R2_RESTORE.md), STUDY_R2.md §0. |
| the GPU build | **Not done.** The CUDA 3D rollout trainer exists (`rollout3d_train.cpp`, `--device cuda`; hres's smoke trained on it) for the mean-squared, profile and activity losses only; it refuses study R2's spread and edge losses rather than drop them. No GPU solver, player or renderer. | P1: `nvfx_gpu_train` on LibTorch with all the losses, `nvfx_gpu_sim`, `nvfx_gpu_play`, the differentiable renderer, and their parity tests; `-DNEURALFX_GPU=ON`, off by default. | PLAN_PC_GPU.md §5, STUDY_3D_HRES.md (the CUDA trainer's use), `tools/study_3d/cuda_train_check.sh`. |

## 7. In short

- **Container-done for wave 1:** every tier 1 row but cloud shadows (#137) has a 2D model or library entry as its far
  level of detail; the fire (#0) has a 3D model that passed its one test (hres C2) and the smoke (#16) a validated
  teacher with its learned model under study (R2); the volume API is complete through the job hook; the 2D weather
  director holds five of the six first regimes with the rain's, snow's, fog's and gusts' relations from the literature.
- **PC-pending for wave 1:** every 3D effect but the fire; every teacher's physical check at 256 (P2); the GPU trainer,
  solvers and player (P1); the GPU FLIP (P5) before any liquid; sparse volumes, the manager for volumes, the 3D scene
  language and `nvfx_pc`; the weather rows and the regional model (P7); the sky model and the optics in FUSE.
- **The first gate** is P3's fire and smoke, where studies N2 and R failed: a pass there is what the rest of the wave
  waits for.

## 8. PC next steps

What to do on the PC first, with what exists today (PLAN_PC_GPU.md §8, P0 and the start of P1).

- **The data root.** Everything outside git lives under `NEURALVFX_DATA` (default `$HOME/nvfx-data`; [DATA.md](DATA.md)
  §1). Set it once per shell, then `models/restore.sh` puts the shipped models where the tests and tools read them:
  ```sh
  export NEURALVFX_DATA=/mnt/d/nvfx-data      # ~5.4 TiB free on D: (PLAN_PC_GPU.md §2.1); mkdir once; hres may stay on $HOME/nvfx-data until you migrate
  bash models/restore.sh                      # models/layout.txt -> $NEURALVFX_DATA/...; checks SHA256SUMS first
  ```
  CMake reads the same variable at configure time (`NEURALFX_MODELS_D` defaults to
  `$NEURALVFX_DATA/experiments/models/d`), so configure after setting it. The study scripts take it too:
  `tools/study_3d/hres_*.sh` put their files under `$NEURALVFX_DATA/hres_3d/<effect>`, `r2_*.sh` under
  `$NEURALVFX_DATA/r2_smoke3d` (or `R2_ROOT`; what to copy from the container and what to run after,
  [STUDY_R2_RESTORE.md](STUDY_R2_RESTORE.md)), `tools/study_q2/*.sh` as STUDY_Q2_RESTORE.md says.
- **The CUDA build** (`CMakeLists.txt`; today's `NEURALFX_CUDA` is the only GPU option, PLAN_PC_GPU.md §5's
  `NEURALFX_GPU` does not exist yet):
  ```sh
  cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DNEURALFX_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DNEURALFX_BUILD_TESTS=ON
  ninja -C build-cuda neuralfx_3d_tests nvfx_3d
  build-cuda/neuralfx_3d_tests --gtest_filter='Rollout3d.Cuda*'   # or: bash tools/study_3d/cuda_train_check.sh build-cuda
  ```
  `NEURALFX_CUDA=ON` enables the CUDA language (standard 20), builds `neuralfx_train_cuda` from
  `src/train/rollout3d_train_cuda.cu` and defines `NEURALFX_HAVE_CUDA`; off, the stub is linked and `--device cuda`
  refuses. `CMAKE_CUDA_ARCHITECTURES` defaults to 86 (Ampere): set it to the PC's card (89 for Ada, 90 for Hopper,
  120 for Blackwell) once `results/3d/pc.md` records the GPU (P0). `cuda_train_check.sh` also passes
  `-DCMAKE_CXX_COMPILER=g++-14`; use the PC's compiler.
- **The first effect to train: the fire (#0), from hres's own data.** It is the most advanced row, the first gate's
  half that has already passed a test, and the one the CUDA trainer can take as it is: the trainer has the
  mean-squared, profile and activity losses and refuses R2's spread and edge losses (`rollout3d_train.cpp`), so the
  smoke (#16) waits for P1's `nvfx_gpu_train`. Copy the container's `hres_3d/fire/` (runs, not `bin/`) under
  `$NEURALVFX_DATA`, then:
  ```sh
  TRAIN_DEVICE=cuda TRAIN_SHARDS=4 sh tools/study_3d/hres_fire.sh build-cuda/nvfx_3d   # records are skipped where the runs exist
  ```
  Its gate is P1's: the GPU-trained fire ties or beats hres's C2 on its own held-out test (`results/3d/hres/fire/`,
  `tools/study_3d/hres_fire_validate.sh`), with the step's time logged. N2's fire follows the same way from `n2_3d/`.

**The recommendation, in one paragraph.** After `models/restore.sh` has put the shipped models under `NEURALVFX_DATA`,
the first GPU effect to train is **the fire (#0) at hres's size (32 x 48 x 32), from hres's own runs**: it is the only
tier 1 row whose 3D model has passed a test, its losses are the three the CUDA trainer already has, and its gate is
written (tie or beat hres's C2 on the held-out test); the smoke (#16) needs R2's spread and edge losses, which only
P1's `nvfx_gpu_train` will have, so it comes second. Configure with the flags `CMakeLists.txt` defines today:
`-DNEURALFX_CUDA=ON` (the one GPU option; it calls `enable_language(CUDA)`, sets `CMAKE_CUDA_STANDARD 20`, builds
`neuralfx_train_cuda` from `src/train/rollout3d_train_cuda.cu` and `rollout3d_train_cuda_host.cpp` with
`CUDA_SEPARABLE_COMPILATION OFF`, and defines `NEURALFX_HAVE_CUDA`; off, `rollout3d_train_cuda_stub.cpp` is linked
instead), `-DCMAKE_CUDA_ARCHITECTURES=<the card's>` (86 if unset), `-DNEURALFX_BUILD_TESTS=ON` (defaults to
`BUILD_TESTING`; needed for `Rollout3d.Cuda*`) and `-DCMAKE_BUILD_TYPE=Release`, with `NEURALVFX_DATA` exported before
configuring so `NEURALFX_MODELS_D` resolves under it. Then `ninja -C build-cuda nvfx_3d neuralfx_3d_tests`, the CUDA
tests once, and `TRAIN_DEVICE=cuda TRAIN_SHARDS=4 sh tools/study_3d/hres_fire.sh build-cuda/nvfx_3d`.
