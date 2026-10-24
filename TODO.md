# To do: every effect in 3D — a small demo here first, then big on a PC

**Status:** stage 1 is done (study N, `docs/STUDY_3D.md`, and its follow-up N2, §14). **Stage 2 is the owner's active
goal:** [docs/PLAN_PC_GPU.md](docs/PLAN_PC_GPU.md) for **all effects** with a GPU teacher in that plan (§8 P0–P7,
`nvfx_pc effect <name>`, [docs/WAVE1_CHECKLIST.md](docs/WAVE1_CHECKLIST.md) row by row)—not optional, not "much later."
**Now on the PC:** study hres's **smoke_r2** run (`tools/study_3d/hres_smoke_r2.sh`) to learn from hres **fire** (pass)
and **smoke** (fail) before the GPU stack; hres fire/smoke N are already shipped. **PLAN_PC_GPU code (P0+) is not
started** until P0's §2 answers are on the record and P1's `nvfx_gpu_*` exists.
Since 15 October 2026, here: round 9, studies O and P, float on AVX-512, the faster detail layer, the 2D weather
director, the FUSE sample, round 10's fire and elements and study Q (the weak spots of rounds 9 and 10) are done and
merged, and so is study R (a validated 3D smoke): its teacher is a plume by MTT's measures, its learned smoke failed its
test and ships as a labelled demo ("Here, until the PC"). Round 7 of the quiet re-timing measured all of their costs on
a quiet machine (REPORT §13.11). The effect manager can draw large effects at half size (REPORT §22): on by default
per effect for fire, smoke, steam and explosions from 256 px on screen, off for magic (it saved too little) and effects
not measured (§22.11, NVFX_MANAGER_VERSION 3). Study hres, the first finer coarse 3D state (32 x 48 x 32), is
trained, validated and shipped (23 October 2026): its fire (C2) as `models/3d/fire_hres.nvfx3`, which meets its rule on
validation and on its one test (the test edge check judged with the calibrated bounds 24 October; REPORT §15's hres
addendum), its smoke as a labelled demo that fails the edge check (`models/3d/smoke_hres.nvfx3`). Since 24 October ("Done since 24 October 2026" below):
the half-size default per effect, the coupled fire and smoke as one simulation in the 3D solver (`sim3d::Stack`, a
demo), the weather director's humidity, dew point and cloud base, its wet ground, its sun and shafts, and its cloud deck
and trees as hooks (`sky.deck`, `sky.foot`, `sky.trees`), the frame's `wet`, `puddle`, `deck` and `foot` keys wired to
them in the day scene and drawn (the ground darkened and puddled, the deck over the dome), the steam fog, contrail, helicopter downwash, whiteout
and drifting snow library entries (the pyroclastic flow labelled as built, not learned), studies Q2's and R2's job
scripts and the smoke-at-hres driver moved into git so that they run from any machine, the wave 1 checklist
(`docs/WAVE1_CHECKLIST.md`), the commit plan for all of it (`COMMIT_PLAN.md`) and the scripts that carry it out
(`tools/ship_wave1.sh`, `tools/preflight_wave1.sh`, `.gitattributes`).

- **Stage 1, here and now (study N): done.** A small 3D demo in this container ([docs/STUDY_3D.md](docs/STUDY_3D.md),
  REPORT §15). What it built:
  - our own 3D solver (64 x 96 x 64; fire and a smoke preset; not a validated fire code);
  - a 3D rollout model (a 16 x 24 x 16 learned state, 14,218 weights, driving seeded 64 x 96 x 64 detail);
  - a physically based volume renderer (Planck's law and CIE 1931 to RGB, soot absorption and single scattering);
  - the volume C API (`include/neuralfx/nvfx_volume.h`, `NVFX_VERSION` 6);
  - demo videos (`videos/3d_*.mp4`).

  What it found, on held-out settings with fresh seeds, the rule written before validation:
  - **The learned 3D fire tracks a real run better than the cheap 3D simulation with the same seeded detail.** At
    1 s it is +1.1 dB in temperature and +2.2 dB in pictures from four cameras.
  - **Played on, it is better on detail and the mean picture but worse on motion and coverage**, so it does not beat
    the cheap simulation outright.
  - **Most of the "worse" is one fault:** with wind blowing in, heat builds up at the box's upwind side and a second
    fire grows there. The coarse advection's samples are clamped to the grid, so inflow keeps its own heat. The
    `nvfx_3d edge` check catches it.
  - **Cost (provisional, a busy machine):** a step takes about 60 to 70 ms on one core, 33 ms on four. The detail layer
    is 59 ms of that and the network 15. A 256² picture takes about 49 ms on the CPU. (Study N2's model on a quiet
    machine, with the detail layer on vectors since: 21 ms a step on one core, 9 on four; a 256² picture 56 ms;
    STUDY_3D.md §14.12.)

  What study N2 found (STUDY_3D.md §14):
  - **The fault is fixed.** With ambient air at the open edges and a training stage of long rollouts, the same small
    model has no runaway over 30 s at any wind; study N's model fills the box at every wind given 30 s without shards.
  - **It beats study N's model on a new held-out test**: +1.95 dB in pictures at 1 s, better on four of five endless
    statistics, and it beats holding the first frame. Against the coarse simulation it is too calm (motion 0.68 of the
    real runs').
  - **By its rule it did not pass**: an edge bound set from one simulated run per wind was 3% too tight on one side.
    The owner chose to show C4 as the demo anyway (15 October 2026); `videos/3d_*` say so. Calibrate such bounds on the
    teacher's run-to-run spread.
  - Study N's smoke also runs away given 30 s; with the fix it does not. Threads scale: a step is 23 ms on four.

  What stage 2 takes from it:
  - choose models on endless play as well as tracking;
  - bring ambient air in at open edges (zero heat, soot and memory) and check with `nvfx_3d edge` (done in N2);
  - train on long rollouts at the controls' extremes and with controls that change (done in N2: the long stage is
    what made the model hold over 30 s);
  - make the detail layer cheap (SIMD, sparse bricks) and draw pictures on the GPU;
  - use a finer state (32³) or a smoother lock against blocky flames;
  - start from a validated teacher (FDS);
  - snapshot every training stage.
- **Stage 2, on this PC (PLAN_PC_GPU, all effects):** accurate 3D at **64 x 96 x 64** learned state and **256 x 384 x 256**
  detail, each effect **5 to 9 hours** unattended (`nvfx_pc`), GPU sim + train + play, teachers checked (§3.4), gates §6.4.
  The plan is [docs/PLAN_PC_GPU.md](docs/PLAN_PC_GPU.md); the per-effect ledger is [docs/WAVE1_CHECKLIST.md](docs/WAVE1_CHECKLIST.md).
  **Prerequisite on the PC today:** finish hres **smoke_r2** and document what hres fire (pass) and smoke_hres (fail) imply
  for spread/edge, calibrated bounds and teacher choice (PLAN_PC_GPU §1.1); then P0–P3 (GPU stack + first gas gate), then
  P4–P7 for the catalogue. [docs/GENERATOR.md](docs/GENERATOR.md) remains a parallel bet, not a substitute unless pilots
  beat §6.4. Then 3D scenes, a 3D library and the full weather system (P7).

Wave 1 here finishes the 2D director, library and study scripts; stage 2 on the PC is the main line after smoke_r2.

## Here, until the PC

Work that runs in this container meanwhile, in 2D and in small 3D, measured as every study is.

**Done since 15 October 2026** (each merged into main after the full ctest, `nvfx_alloc_test`, `nvfx_gallery --check`
and the Godot test; details in its doc and REPORT section):
- **Round 9, earth** (docs/EFFECTS_EARTH.md, REPORT §6.16): lava, a volcanic eruption and a powder-snow avalanche, each
  tested once. Lava works (tracking at 1 s +1.09 dB [+0.70, +1.43] over the coarse simulation with detail); the
  eruption works narrowly (+0.50 dB [+0.06, +0.96]); the avalanche is weak (-2.23 dB [-4.42, -0.07]). Ten library
  entries, the volcano scene, six videos.
- **Round 9, air** (docs/EFFECTS_AIR.md, REPORT §6.17): a candle flame, smoke rings and heavy gas, each tested once. The
  candle (+2.90 dB [+2.35, +3.41]) and the smoke ring (+7.51 dB [+6.62, +8.44]) work; heavy gas falls short (its mean
  picture ties the nearest flipbook's). Twelve library entries, the crypt scene, six videos.
- **Study O, the most visible flaws** (REPORT §16): the waterfall's and rain shaft's banding, launch's square block and
  the thruster's curling exhaust fixed by eye; the mushroom (+1.18 dB [0.91, 1.45] in the mean picture) and the splash
  (+0.40 dB [0.21, 0.60]) replaced by a held-out test.
- **Study P, cheaper big effects** (docs/STUDY_P.md, REPORT §17): the learned 2x upscale did not pass its rule (it beat
  bilinear on none of six effects); drawing at half size with a bilinear 2x costs 0.36 to 0.43 of native and stays
  within native's margins against the simulator for all six (busy machine; 0.36 to 0.42 on a quiet one, round 7).
  Shipped as an opt-in, the learned mode marked experimental (NVFX_VERSION 7).
- **Float on AVX-512** (REPORT §13.10): float rollouts take the AVX-512 build where the CPU has it, 0.80 of AVX2's time;
  frames bit-identical to AVX2's with this compiler.
- **The detail layer's cost** (REPORT §19, `results/detail/`): the same frames to the bit on every ISA; in 3D the
  detail layer takes 0.16 to 0.18 of its time and a step 0.31 to 0.32; in 2D the detail stage 0.87 to 0.90 and a frame
  0.96 to 0.98 (busy machine; on a quiet one, round 7: in 3D 0.20 to 0.22 and 0.38 to 0.40, in 2D the same). A
  semi-Lagrangian option (0.75) changes every frame and was not shipped.
- **The 2D weather director** (docs/WEATHER.md, REPORT §18): seven regimes, ordered transitions, seeded gusts, rain cells
  and lightning, rain and snow by Marshall-Palmer, Atlas and Koschmieder, a 40 s day scene and its video; engines drive
  it through the scene inputs. Weak: no sun, the wet ground and the cloud base as numbers in the state that nothing
  draws, clouds as puffs rather than a deck, one place in 2D.
- **For FUSE** (docs/FUSE.md, `engines/fuse`): the interface FUSE implements, an adapter (pack, effect manager on FUSE's
  jobs, a scripted scene, library entries, the 3D fire), a headless demo with its test and a video.
- **CI:** the g++ job's one failure since the packs came in (the pack verify test without the models) fixed.
- **Study Q, the weak spots of rounds 9 and 10** (finished on its branch `study-q`, 21 October 2026, with the checks;
  docs/STUDY_Q.md, REPORT §21; seeds 33.0 to 33.5 million):
  each weak spot taken apart first (which part makes it), then fixed by eye or by a model change under a rule committed
  before validation, its margins from a null probe, tested once on held-out settings. **Replaced** after their tests:
  the avalanche (its stepper fine-tuned on 30-frame windows: tracking at 1 s +0.34 dB [0.18, 0.50], the mean picture
  +0.16 dB [0.04, 0.28]), the dust explosion and the mushroom (their renderers trained again with the first frames among
  their samples: the first frame +2.85 dB [2.54, 3.16] and +2.04 dB [1.97, 2.12]). **Fixed by eye**: the crown and
  window fires' streaks, heavy gas's bright lines and the smoke ring's detail (detail settings in the library and the
  scenes). **Stopped at validation**, files kept: the haboob's and the smoke ring's refits and the geyser's fine-tune
  (each better on a target, past a margin elsewhere). Still weak: the avalanche (its start points), the geyser, the
  haboob's and the ring's first frames, heavy gas's mean picture, the mushroom's cap.

- **Study R, a validated 3D smoke** (finished on its branch `study-smoke3d`, 21 October 2026, with the checks;
  docs/STUDY_R.md, REPORT §20; seeds 29.0 to 29.9 million): **the teacher first.** Study N's smoke preset lost two
  thirds of its buoyancy flux up the box; `smoke_r` (no cooling, fading or damping; heat and soot in flux form)
  conserves it, spreads at a top-hat alpha of 0.14 to 0.15 (the measured band for pure plumes 0.10 to 0.16), and its
  centreline laws are within about 0.1 of Morton, Taylor and Turner's exponents; it is a third too slow for its flux
  (0.78 of the momentum its buoyancy gives), and its wind is a drag, so the bent plume is not Briggs's. **The learned
  smoke** (N2's recipe; edge bounds calibrated on four teacher runs per wind, committed before validation) **fails its
  test**: it tracks real runs better than the coarse simulation with detail (+1.59 dB [+1.02, +2.20] in pictures at
  1 s), but missed one edge bound by 7% on validation and played on spreads too wide (coverage, light and the mean
  picture worse: -2.52 dB [-3.79, -1.29]). Shipped as a labelled demo (`models/3d/smoke.nvfx3`), with the plume
  video and the fire and smoke coupled through the volume API.

- **Half size where it pays: done** (branch `manager-halfsize`, 22 October 2026, with the checks; REPORT §22,
  MANAGER.md §2.10 and §7.2; seeds 35.0 to 35.1 million). The effect manager draws a simulation that is at least
  `half_px` pixels (or `half_share` of the view's height) on screen at the level below its level, and the bilinear 2x
  brings it back: the engine's sampler by default (the picture is the half-size drawing: no CPU, a quarter of the
  upload), the manager's own with `NVFX_UPSCALE_BILINEAR` (to the bit what `nvfx_instance_set_upscale` draws); never the
  learned 2x. Hysteresis, never for locks or proxies, per model (`nvfx_manager_model_desc.half_px`: 0 never, > 0 its
  own), no allocation per frame, the same pictures on any number of threads; NVFX_VERSION 8; exposed in the C++ wrapper
  and the FUSE adapter (the library's stages and scenes draw at the sizes their scripts name, so they have nothing to
  switch). **By its rule, committed first, the all-or-nothing default stayed off:** on 20 held-out settings at 256 px it stays within the
  null's margins of full size against the simulator on every measure for all seven effects measured (the crowd's fire,
  smoke, steam and explosion, the library's fire, explosion and magic), closer than full size on tracking at 1 s
  (+0.22 to +0.89 dB) and the detail spectrum, and costs 0.31 to 0.42 of full size for six of them, but **0.54 for
  magic** (the rule allowed 0.5; the quiet session's separate runs give 0.53 too). Against full size on the same seed it
  loses the finest filaments (24 to 37 dB active PSNR). With no budget a close-up scene's tick went from 17.1 to 15.2
  ms (0.886 [0.869, 0.901]); with a budget the planner runs more simulations of their own (13.2 to 17.4 at 8 ms) and
  fewer proxies; the crowd hardly changes (provisional timings: other agents' load ran throughout). Also fixed: a race
  in the manager's worker start that could hang a tick on several threads (main's code). Its default per effect is
  24 October's (below).

- **Study hres, a finer coarse 3D state: done** (23 October 2026; [docs/STUDY_3D_HRES.md](docs/STUDY_3D_HRES.md),
  `results/3d/hres/`; seeds 9.60 to 9.87 million inside study N's block; the smoke trained on the CUDA trainer, the
  fire on the CPU). A **32 x 48 x 32** coarse state with fine = 2 (study N's 64 x 96 x 64 detail), a 32-wide stepper
  and 8 memory channels, study N2's recipe. **Fire:** C2 chosen on validation (edge check passed at every wind with
  bounds calibrated on four teacher runs per wind, `fire/edge_calibration.md`; mean rank 1.00 over the endless
  statistics; `fire/choose_val.md`); tested once, frozen (`fire/frozen.csv`): over the coarse simulation with detail,
  tracking at 1 s +1.69 dB [+1.07, +2.32] in pictures and +1.29 dB [+0.98, +1.60] in temperature; against study N2's C4
  on endless play, the mean picture +2.20 dB [+1.28, +3.14], the spectrum and motion better, coverage and emission
  ties; motion 0.99 of the real runs' (N2's 0.66). **Its test edge check passes** with the bounds calibrated on four
  teacher runs per wind (`fire/edgecheck_test.md`, judged 24 October by `hres_fire_recheck_test_edges.sh`): the two
  winds the single-run bounds had marked as runaway (0.50 at x = 0, largest 3.43; 0.95 at z = 0, largest 2.36;
  `fire/edgecheck_test_single.md`) sit under the calibrated 4.95 and 4.73, and no side grows. Shipped as `models/3d/fire_hres.nvfx3`; C1 to C4 and the final `finish` in `fire/models/`. **Smoke**
  (study N2's smoke2 lineage, study N's smoke preset as the teacher, no held-out test by its rule): **fails the edge
  check** at winds 0.50, 0.75 and 0.95 (heat building at the x = 0 side over 20 to 30 s where the simulation holds
  none: 1.15, 2.69 and 4.87; the top too at 0.95), as the 16³ smoke2 baseline fails at four winds (`smoke/edgecheck.md`);
  shipped as a labelled demo, `models/3d/smoke_hres.nvfx3`. Its failure is the one study R2's edge and spread losses
  are aimed at (STUDY_3D_HRES.md §8). The fire's test tables (`edgecheck_test.md`, `compare_test_*`, `edge_test*.csv`;
  the 22 October snapshot run wrote them under the data root) are in `results/3d/hres/fire/` since 24 October
  (`hres_status.sh` reports them present and whether the edge check is the calibrated one;
  `hres_sync_fire_test_results.sh` copies them back if a run strays them again, without overwriting the repo's).
- **Round 10, fire** (`effects-fire2`, docs/EFFECTS_FIRE2.md, REPORT §6.18; seeds 26.0 to 26.9 million): a forest crown
  fire (#6), fire venting from a window (#8) and a dust explosion (#31). **Finished and merged into main** (21 October
  2026): each tested once, all three work by the rule (tracking at 1 s over the coarse
  simulation with detail: crown fire +2.60 dB [+2.16, +3.04], window fire +3.31 dB [+2.91, +3.73], dust explosion
  +8.44 dB [+7.88, +9.04]; endless, the mean picture closer than the nearest flipbook's, motion no worse). Ten library
  entries, the forest-edge town scene, six videos. Still weak: the two fires' detail streaks (spectrum six times the
  flipbook's); the dust explosion's renderer fringes its unlit cloud and draws its pall faint, and its start follows the
  ignition control only roughly.
- **Round 10, elements** (`effects-elements`, docs/EFFECTS_ELEMENTS.md, REPORT §6.19; seeds 27.0 to 27.9 million): a
  geyser (#88), a haboob dust wall (#70, #71) and sea spray off breaking waves (#77). **Finished and merged into main**
  (21 October 2026): each tested once. Sea spray works (tracking at 1 s over the coarse simulation with detail +1.92 dB
  [+1.01, +2.83]; endless mean picture +5.88 dB [+4.77, +7.11] closer than the nearest flipbook's, motion tied); the
  haboob falls short (+1.06 dB [+0.25, +1.87] at 1 s, its mean picture tied with the flipbook's; the simulator's front
  runs at Fr = 1.15 on its head, Huppert and Simpson's 1.19); the geyser is weak (+1.10 dB [-0.33, +2.57] at 1 s, a
  tie). Ten library entries, the coast scene under the weather director, six videos. Still weak: the geyser never
  erupts at low intensity and within a shard follows its start point more than its intensity; the haboob's first
  frames (its renderer) and a weak current that runs 25% fast; sea spray's detail (tufts, not thin plumes). To try: a
  field-shader look that draws heat as white vapour (steam and fine spray are heat, which a look draws as light).
- **Quiet re-timing, round 7** (`quiet-round7`, results/quiet/README.md "Round 7", REPORT §13.11): every cost since
  round 9 re-timed in one session with nothing else running (59 blocks, each started at a load of at most 0.5, three run
  again for other agents' load), and the documents' provisional figures replaced
  (`results/quiet/round7/compare_provisional.csv`). The 33 shipped 2D models: 0.63 ms per 128 px frame on average at the
  int8 default, int8 0.906 of float on the AVX-512 build (round 6's 0.906); multi-threaded figures far lower than on the
  busy machine (the scenes 30 to 78%, the FUSE demo's jobs 54 to 90%), single-core ones up to 40% higher than the least
  busy rounds the documents quoted (within 15% of the busy medians for 55 of 60); the 3D detail layer 4.5 to 4.9 times
  faster, not 5.7 to 6.5; study P's learned 2x under 0.55 of native for 3 of 6 effects, not 4. This host's AVX-512 build
  ran 7 to 11% slower than round 6's and its speed drifted by about ±7%: absolute milliseconds belong to their session.

**Done since 24 October 2026** (on the working tree, to be committed as `COMMIT_PLAN.md`'s three commits; each item's
tests are named in its doc):
- **Half size, the default per effect** (REPORT §22.11, MANAGER.md §2.10 and §6; NVFX_MANAGER_VERSION 3): the rule's two
  conditions read per effect from the runs already made, no new run. `nvfx_manager_desc.half_px` below 0 (the new
  default, −1) draws fire, smoke, steam and explosions at half size from 256 px on screen by the model's name, magic and
  every effect not measured never; 0 is off for every effect, above 0 one threshold for every effect; a model's own
  `half_px` still comes first. The FUSE adapter passes the default through; `time_crowd.sh` keeps §7's baseline at full
  size (`--half-px 0`); ENGINES.md, FUSE.md and the README say so. Tests: `Manager.HalfSizePerEffectDefaults`, `ApiErrors`.
  Run 24 October 2026: `tools/manager/time_half_128.sh` (MANAGER.md §7.2), the quick quiet re-timing of the saving;
  results in `results/manager/half_128/` (`ab_summary.txt`, `runs.csv`, `load.txt`; a second run the same day in
  `half_128/run2/` confirms the close-scene ratio and the crowd tie):
  the two no-budget rows (the close scene and the crowd on four threads), off and on back to back with their order
  swapped each round, each round waiting for a load of at most `MAX_LOAD`, `nvfx_crowd --ab` for the medians and
  intervals; about 3.5 minutes at its defaults; `MAX_SIZE=128` times 128 px pictures drawn at 64, the level not measured.
- **3D fire and smoke as one coupled simulation** (STUDY_R.md §6.1; `include/neuralfx/sim3d.hpp`, `src/sim3d/sim3d.cpp`):
  `sim3d::Stack`, two boxes of the same width one above the other; at the start of every solver step the lower's top
  row and the upper's bottom row become each other's border row (a `Seam`), so heat, soot and velocity cross the seam
  both ways; the pressure solve stays each box's own, the upper box has no burner (`Params::source`), a box with no
  seams is the single box to the bit, and the two together keep their heat and soot to within a few per cent (the test
  allows 8%). `nvfx_3d video --kind coupled-sim` draws it (the fire below, `--upper smoke_r` above): a demo, not
  measured. The volume-API demo's rule (`--kind coupled`: the fire's top soot flux drives the smoke's intensity, no
  exchange) moved into `include/neuralfx/coupled3d.hpp`, shared by the tool and `tests/test_coupled3d.cpp`. Tests:
  `Sim3d.CoupledStack*` (the seam exchanges heat, the fire feeds the smoke above, the same fields on any number of
  threads), `VolumeApi.CoupledDemoFeedsTheSmokeFromTheFiresSoot`. Not done: a learned model trained on the stacked teacher.
- **The weather director's moist air** (WEATHER.md §3.11; `wx::saturation_vapour_hpa`, `dew_point`, `cloud_base_m`): the
  state carries the relative humidity (a target per regime, blended with the cloud), and from it the dew point by the
  Magnus formula (Alduchov and Eskridge's coefficients, checked against Sonntag's table) and the cloud deck's base by
  Espy's rule (125 m a degree, within about 15% of Lawrence's fit): fair-weather cumulus about 1200 m, an overcast
  deck 560 m, rain and snow about 200 m, fog at the ground, a dust storm's 3.8 km. Read as `sky.humidity`,
  `sky.dewpoint` and `sky.base` (COMPOSE.md, ENGINES.md); the deck itself is the frame-keys item below; the library
  entry is unchanged. Tests: `tests/test_weather_director.cpp` (`WeatherRelations.DewPointAndCloudBase`).
- **The weather director's wet ground** (WEATHER.md §3.12; `wx::evaporation_mm_h`, `wind_2m`, `kDepressionStorage`): the
  state carries the water standing on the ground, filled by the rain up to an impervious surface's depression storage
  (1.5 mm; ASCE 1.3 to 2.5) and dried by Penman's drying power of the air (his 1948 aerodynamic term, 0.2625 (1 +
  0.537 U_2) times the vapour pressure deficit, mm/day; zero in saturated air, so fog keeps the ground wet). From it the
  ground's darkening over the first 0.1 mm, its puddles (the share of the storage) and the drying now: `sky.water`,
  `sky.wet`, `sky.puddle`, `sky.evaporation`. Moderate rain darkens the ground in a minute and puddles it in 15; a clear
  day dries the puddles in 5 hours, a dust storm in 27 minutes. Drawn through `frame wet` and `frame puddle`, which
  the day scene wires (the frame-keys item below; COMPOSE.md §4.11); the library entry is unchanged. Tests:
  `test_weather_director.cpp`
  (`WeatherRelations.PenmansDryingPowerOfTheAir`, `WeatherDirector.TheGroundWetsInTheRainAndDriesByPenman`).
- **The weather director's sun and its shafts** (WEATHER.md §3.14; `wx::sun_elevation_deg`, `sun_across`, `sun_share`,
  `beam_transmittance`, `shaft_visibility`; the `weather` statement's `hour`): the state carries the solar hour (a
  script expression re-evaluated each frame, or an input the game sets; noon without it), and from it the sun's
  elevation and its place across the picture at 45 N on the equinox (up at 6 on the left, 45 degrees at noon, down at
  18 on the right), the direct sunlight on the ground by Kasten and Czeplak's diffuse fraction (0.7 (1 - cloud^2) of
  the daylight) through Beer-Lambert's slant path in the weather (clear air passes 0.96 at noon, fog and a haboob's
  load hide it), and the shafts' visibility where a broken deck's beams light hazy air (4 c (1 - c) times the beam
  times what the air scatters: under 0.05 in clear air, 0.27 in a 2 km haze at half cover, peaking in the clear-to-fog
  and clear-to-dust transitions). Read as `sky.hour`, `sky.elevation`, `sky.sunx`, `sky.sun`, `sky.shaft` (COMPOSE.md,
  ENGINES.md). Nothing draws it yet (what a scene would wire is in §3.14: a lamp at the sun, the rainbow opposite it, a
  `frame shaft` key); the daylight does not follow the hour, so the day scene and the library entry are unchanged.
  Tests: `test_weather_director.cpp` (`WeatherRelations.TheSunAndItsShafts`,
  `WeatherDirector.TheSunStandsByTheHourAndShowsThroughBrokenCloud`; the sun in the jump, target and engine tests).
- **The weather director's cloud deck and trees, as hooks** (WEATHER.md §3.15 and §3.6; `wx::deck_cover`,
  `deck_foot_deg`): `sky.base` stays Espy's condensation level; `sky.deck` is how much of the sky that base organises
  into a layer rather than puffs (0 below four oktas, 1 by six), and `sky.foot` the elevation above the horizon where
  Koschmieder's haze hides the layer's underside (`arcsin(base / visibility)`: 0 in fog, just above the horizon under a
  1.2 km base in clear air, near 90° in a dust storm). `sky.trees` is the trees' sway by the Beaufort force, a
  smoothstep from force 4 (small branches move) to 7 (whole trees in motion), carried with the gusts, the direction
  `sky.u`'s. The deck is drawn through the frame's `deck` and `foot` keys (below); no layer bends by `sky.trees` yet:
  a scene gates its cloud masses with `sky.deck`, places their underside with `sky.foot`, and shears a sprite or an
  engine's own trees by `sky.trees` (`nvfx_scene_get_input`). Tests: `WeatherRelations.TheCloudDeck`,
  `TheDeckFootHidesInDust`, `TheBeaufortScale`.
- **The frame's `wet`, `puddle`, `deck` and `foot`, wired** (COMPOSE.md §4.9.1; `src/compose/compose.hpp`, `script.cpp`,
  `script_run.cpp`): four keys of `frame`, expressions re-evaluated every frame as `daylight` is, each 0 by default so
  every scene before stays as it was, to the bit; `examples/scenes/weather_day.nvfxs` wires them to `sky.wet`,
  `sky.puddle`, `sky.deck` and `sky.foot`. `src/compose/compose.cpp` draws all four in the background pass. The wet
  ground (COMPOSE.md §4.11): `wet` darkens the ground rows, their own colour and the scene's light on them, by
  lerp(1, 0.6, wet) before the haze (Lekner and Dorf's wetted rough surface; by night too, so a night rain darkens
  the earth under the fire's light); `puddle` fills the ground texture's hollows deepest first, each mirroring the
  sky and the hills with Fresnel's reflectance at the depression angle its row's distance gives (Schlick: 0.98 at
  the horizon, 0.72 at the bottom of the picture), the darkened ground showing through the rest. The deck (§5.4):
  over the day sky's dome, and over the night, a layer's underside whose cover on each row is `deck` times
  Koschmieder's contrast along the slant path (`wx::deck_contrast`: 0 at the horizon, about 2% at the foot), the CIE
  overcast grey at that elevation with two octaves of value noise for its cells, lit whole by the flash, hiding the
  stars. Tests: `tests/test_script.cpp` (`Frame.WetDarkensTheGroundAndNothingElse`,
  `PuddlesMirrorTheSkyAndTheHillsByFresnel`, `TheDeckCoversTheDomeByItsContrast`,
  `TheWetGroundAndTheDeckAreTheSameOnAnyNumberOfThreads`); `tests/test_weather_director.cpp` (39 in all:
  `WeatherDay.TheFrameCarriesTheWetGroundAndTheCloudDeck`, `WeatherDay.TheHillsFadeIntoTheHazeAsKoschmiederSays`,
  `WeatherRelations.TheDeckUndersideFadesIntoTheHazeByKoschmieder`,
  `WeatherDirector.TheDeckFormsBeforeTheRainAndItsFootFollowsTheHaze`). Not drawn: the rain's rings on the puddles
  and the modules' reflections (§4.11's open questions). The day scene's frozen keyframes move with the wet ground
  and the deck (`ScriptedWeatherDayFrozen` skips on this PC, which has no library models): check and refreeze them
  once where the models are.
- **Steam fog in the library** (`library/steam.nvfxl`, `steam_fog`, the catalogue's #61; LIBRARY.md §1, Steam and water):
  fog smoking off warm water, built from what the library has (four releases of the fog model and five of the steam
  model over a seeded pool, a ceiling that keeps the wisps in the mixing layer, a gusty breeze); 235 entries, 169 of
  them learned. Its keyframe is frozen (`results/library/frozen.csv`) and in `LibraryKeyframesFrozen`; test
  `Library.SteamFogIsFogAndSteamOverAPool`; `nvfx_gallery --verify` reads a CRLF `frozen.csv`. Fair by eye: the fog's
  four releases show as humps and the flattened top is a straight lid in a still.
- **A contrail in the library** (`library/trails.nvfxl`, `contrail`, the catalogue's #36; LIBRARY.md §1, Trails): a jet's
  condensation trail built from what the library has: a trail of the explosion model started empty late in its life
  and left cold (no heat, so it neither rises nor glows), drawn white by the sky, its source set back along its own
  velocity (the exhaust mixes with the cold air before its mixing line crosses saturation: the Schmidt-Appleman
  criterion), the wake's sink a small downward deposit velocity, the air's humidity how much it leaves (thin material
  the model sheds within a second: a dry sky's short trail), the wind aloft a wind field on its tiles; 237 entries, 171
  of them learned. Test `Library.ContrailIsAColdTrailSetBackBehindTheJet`; the catalogue check passes. Not yet judged
  by eye or frozen: written on a machine without the library's models.
- **A helicopter's downwash in the library** (`library/dust.nvfxl`, `helicopter_downwash`, the catalogue's #37, a
  rotor's wash lifting dust; LIBRARY.md §1, Dust): a brownout built from what the library has: the rotor's induced
  velocity by momentum theory (the thrust is the craft's weight: a heavier craft or a smaller disc blows harder), its
  wall jet lifting a curtain of dust where it passes the grains' threshold, the curtain the dust model's bursts (a
  one-shot of 3 s, so two tiles take turns, each started again by a rule and cross-faded by its age), the fine dust
  drawn back up through the disc a tile of the smoke model sunk under a ceiling at the rotor's height, the ground wind
  or the craft's drift a wind field on all three; the looks `brownout` and `brownout_haze`; 238 entries, 172 of them
  learned. Test `Library.HelicopterDownwashIsDustTakingTurnsUnderAHazeInTheWind`; the catalogue check passes. Not yet
  judged by eye or frozen: written on a machine without the library's models (`hover_downwash`, the launch model's
  flow, stays beside it).
- **The pyroclastic flow labelled** (`library/earth.nvfxl`, `pyroclastic_flow`, the catalogue's #110; LIBRARY.md §1,
  Earth): its comment and `about` now say what it is, the avalanche model recoloured dark ash with a lamp at its base,
  built from what the library has as `steam_fog` and `contrail` are, not a learned flow (no co-ignimbrite plume, no heat
  of its own); the effect is unchanged, and a learned one stays under "More 2D effects" below.
- **A whiteout in the library** (`library/storms.nvfxl`, `whiteout`, the catalogue's #54, blizzard and blowing snow:
  "falling and blown snow, whiteout"; LIBRARY.md §1, Weather and sky): the blizzard at its worst, built from what the
  library has: the blizzard's falling snow, a saltation layer of snow streaming nearly flat along the ground whose
  amount is the blowing-snow flux (the excess of the wind's square over its threshold's, Pomeroy and Gray: nothing
  lifts under a wind of about 135, and the range reaches there), and the fog model, white, in two storeys for the haze
  that fills the air until the ground and the sky merge (three releases on the ground, three at the suspension's
  height, thin until the wind is strong); 239 entries, 173 of them learned. Test
  `Library.WhiteoutIsABlizzardWhoseBlownSnowHasAThreshold`; the catalogue check passes. Not yet judged by eye or
  frozen: written on a machine without the library's models. Heavy: six fog modules.
- **Drifting snow in the library** (`library/storms.nvfxl`, `drifting_snow`, the catalogue's #75, drifting snow and
  spindrift, and the blown half of #54; LIBRARY.md §1, Weather and sky): the smallest slice, procedural only: two
  bands of the compositor's snow fed from upwind, a dense low saltation layer of small grains driven fast and nearly
  held up against their fall, and a sparse haze of the finest above it whose amount grows with the cube of the wind
  (Bagnold's law, for snow as for sand), a gust field bunching both into streamers; nothing falls from the sky (that
  is `blizzard`); 240 entries, 173 of them learned. Test `Library.BlizzardAndDriftingSnowAreBlownSnow`; the catalogue
  check plays it with 0 allocations at both ends of its ranges; its keyframe is frozen in `frozen_round3.csv` and in
  `LibraryProceduralKeyframesFrozen` (no models needed). Fair by eye: a low band of grains bunched by the gusts; in a
  still at night the grains look like stars, as `snow`'s do. Left: spindrift off a ridge (a plume from a crest, the
  anchor the crest), the saltation layer following a sloping ground, and a learned one (PALM with saltation).
- **Study Q2's jobs in git** (`tools/study_q2/`: `common.sh`, `bootstrap.sh`, `resume.sh`, `val.sh`, `after_val.sh`,
  `restore_from.sh`; docs/STUDY_Q2_RESTORE.md; STUDY_Q2.md §0): the container's `q2_fix/jobs` steps written against
  `NEURALVFX_DATA` (and `Q2_REPO`, `Q2_BIN`, `Q2_THREADS`, `Q2_NO_PUSH`), so another machine runs them without edits.
  Checked from this PC: the committed tables hold no validation row for the geyser, heavy gas or the mushroom, and
  their candidates exist only under the container's `q2_fix/cand` (`restore_from.sh` copies them; a dry run unless
  `APPLY=1`).
- **Study R2's jobs in git** (`tools/study_3d/r2_{env,bootstrap,resume,train,validate,test,status}.sh`; STUDY_R2.md §0
  and §7): the data root `$NEURALVFX_DATA/r2_smoke3d` (the default `$HOME/nvfx-data`), the tool pinned by its git hash
  on first use, each long stage detached from a snapshot copy, markers for the hand steps; without the container's data
  the same commands record the 48 training runs and train on the CPU. The trainer refuses `--device cuda` with
  `--spread` or `--edge` above 0 rather than drop study R2's losses without a word (`rollout3d_train.cpp`).
- **Study hres, polish** (STUDY_3D_HRES.md §7 and §8): the fire's test tables copied from the data root into
  `results/3d/hres/fire/` (`hres_sync_fire_test_results.sh` does it again if a run strays them); `hres_fire_validate.sh`
  keeps its body in `main()` and `run_fire_validate.sh` runs a snapshot copy with `NVFX_REPO` set (an edit mid-run had
  corrupted two logs and sent the tables under the data root); `hres_status.sh` reports missing test tables, the
  smoke's validate state and the published models; `restart_hres_watchers.sh` is idempotent (never starts a train;
  `FORCE=1` redoes the smoke's finish); §8 reads the smoke's edge failure beside study R2's edge and spread losses, with
  the smoke-at-hres plan. **The fire's test edge check judged with the calibrated bounds** (§7;
  `hres_fire_recheck_test_edges.sh` rebuilds `fire/edgecheck_test.md` from the same `edge_test*.csv`, the single-run
  table kept as `edgecheck_test_single.md`): the learned fire and N2 pass at every wind and side; the two sides the
  single-run bounds had marked sit at 0.69 and 0.50 of their bounds. Written up: REPORT §15's hres addendum and
  `videos/README.md` §9 (the HRES rows no longer say "validation not run").
- **Smoke at hres on study R2's teacher, scripted** (STUDY_3D_HRES.md §10, committed before any run;
  `tools/study_3d/hres_smoke_r2.sh`; `nvfx_3d --study hres --effect smoke_r2`, its own seed blocks at 9,630,000): one
  resumable driver for the plan of §8: record `smoke_r2` at 64 x 96 x 64 block-averaged to 32 x 48 x 32 (40 short
  runs and 24 long), train on the CPU with study N2's four stages, stage 4's long burn and study R2's two losses
  (`--spread 2 --edge 10`; the CUDA trainer has neither, and the tool refuses it), finish C2 to C4, calibrate the
  edge bounds on four teacher runs per wind, judge the candidates' 30 s edge check against them. Without `FORCE=1` it
  prints the plan and what is there and starts nothing; `STAGE=record,train,...` picks stages; one trainer per data
  root (a lock and a `pgrep`); its body in `main()`, as `hres_fire_validate.sh`'s lesson asks.
  `hres_smoke_r2_train.sh` is its first three stages under the name the first commands used; `hres_status.sh` reports
  its runs, training, models, bounds and edge check. Not run: study R2's own result says first whether the losses fix
  the failure ("Next" below).
- **The wave 1 checklist and the commit plan** ([docs/WAVE1_CHECKLIST.md](docs/WAVE1_CHECKLIST.md), `COMMIT_PLAN.md`):
  the checklist is the per-effect ledger of "Order of work" step 1, row by row (what this container has as the far
  level of detail or a 3D study, which of PLAN_PC_GPU.md's phases makes the 3D effect and the check it must pass first,
  the scripts and the numbers), the wave's infrastructure, the first gate (P3's fire and smoke) and the PC's first
  steps (the data root, the CUDA build, the hres fire retrained from its own data). The commit plan partitions this
  working tree into three commits (runtime and API; the weather director and the library; the studies' bookkeeping and
  this file),   with the pre-push checklist and its result on this PC (570 passed, 0 failed, 32 skipped without the models, of 602;
  `COMMIT_PLAN.md`'s Verified, 24 October 2026); the owner makes the commits. `tools/ship_wave1.sh` carries the plan out (its three
  `git add`s, the modes, the line-endings check per commit and the checklist; `DRY_RUN=1` by default prints and
  changes nothing, and checks that its file lists and messages read as the plan's and that every changed path of
  `git status -uall` is in one of them), `tools/preflight_wave1.sh` is the fast pre-push check (`bash -n` over the
  scripts, the weather and library tests, `nvfx_alloc_test`, the catalogue check; no git state touched), and
  `.gitattributes` pins `*.sh` and `*.awk` to LF so that bash reads the scripts on a Windows checkout.

**Under way:**
- **Study R2, the 3D smoke again** (`study-smoke3d-r2`, docs/STUDY_R2.md, REPORT §24; seeds 34.0 to 34.9 million): the
  teacher's momentum deficit found and fixed (`smoke_r2`: dM/dz over B 1.088, from `smoke_r`'s 0.80); the learned smoke
  retrained with a spread loss and an edge loss under a rule with margins from the teacher's own spread, to be tested
  once. Resumable: training in stage 4 and the rule's teacher runs were left running on the container; from any machine
  `bash tools/study_3d/r2_resume.sh` (STUDY_R2.md §0: the hand steps). On this PC, with no `r2_smoke3d` under
  `~/nvfx-data`, the same command starts from the 48 training runs and trains on the CPU.

**Next:**
- **Study Q2's last three** (STUDY_Q2.md §0, STUDY_Q2_RESTORE.md): copy `cand/` from the container
  (`tools/study_q2/restore_from.sh`), then `bootstrap.sh`, `resume.sh` and `after_val.sh <effect>` for the geyser's,
  heavy gas's and the mushroom's validations, freezes and one test each; the mushroom's cap still ragged by eye.
- **Smoke at hres again,** once study R2 says its losses fix the failure: record `smoke_r2` at 64 x 96 x 64
  block-averaged to 32 x 48 x 32, train with `--edge 10 --spread 2` and stage 4's long burn, judge on bounds calibrated
  on four teacher runs per wind, its own seed block and rule committed first (STUDY_3D_HRES.md §8 and §10). The driver
  is written: `FORCE=1 bash tools/study_3d/hres_smoke_r2.sh` runs every stage not done, hours to days on a CPU.
- **3D fire and smoke coupled, learned:** a model trained on the stacked teacher (`sim3d::Stack`) and tested as R2 is;
  until then both couplings are demos (`--kind coupled` through the volume API, `--kind coupled-sim` in the solver).
- **Half size, next** (REPORT §22.9): re-time its cost and the manager's A/B on a quiet machine (the jobs under the data
  root's `halfsize/jobs`; the quick form `tools/manager/time_half_128.sh`, MANAGER.md §7.2, waits for a load at most
  `MAX_LOAD` before each round and needs the 2D models under `$NEURALVFX_DATA`, which the 64-core machine lacks; the
  rule's verdict stands on its own run); measure 128 px pictures drawn at 64 (`MAX_SIZE=128` there) and more of the
  library's models at 256 px (a model measured and passing both conditions joins `half_default_on`); the 256 px pool
  (one slot per effect by default) limits what it switches in a crowd.
- **More 2D effects**, from the catalogue rows the 2D solver can carry: a learned coloured smoke grenade (#18; the
  library's `coloured_smoke_grenade` is the heavy gas model for the dense dye cloud under the smoke model's vent plume,
  LIBRARY.md §1 Smoke, not yet judged by eye: written where the models are not; the combat recolours still ride the
  smoke model and rise), a learned contrail (#36; the library's `contrail` is a trail of the explosion model left cold,
  its source set back behind the jet for the Schmidt-Appleman mixing, LIBRARY.md §1 Trails, not yet judged by eye:
  written where the models are not; a learned one would carry the ice's growth and sublimation, which today are only
  how much the trail leaves), helicopter downwash (#37; the library's `helicopter_downwash` is the dust model's bursts
  taking turns under a sunk tile of the smoke model, LIBRARY.md §1 Dust, not yet judged by eye: written where the
  models are not; a learned one would carry the wall jet and the recirculation through the disc), learned blown snow (#54, #75; the library's `blizzard`,
  `whiteout` and `drifting_snow` are the compositor's snow and the fog model: the saltation layer's hops, splash and
  threshold are only how many grains there are and how fast; spindrift off a ridge is not there at all), a learned steam fog
  (#61; the library's `steam_fog` is the fog and steam models over a seeded pool), a learned pyroclastic flow (#110; the
  library's entry today is built from existing models).
- **The weather director, next:** what the cloud deck and the wet ground leave out now that `frame deck`, `foot`,
  `wet` and `puddle` draw them (COMPOSE.md §4.11, §5.4; "Done since 24 October"): the deck's cells drifting with the
  air the wind has carried past (`deck_drift`, `sky.carried`), the puffs and the pannus below it, the rain's rings
  breaking the puddles' mirror and the modules reflected in them; the day scene's keyframes refrozen where the
  models are; the sun drawn (where it stands, how much comes through and whether its shafts show are in the state,
  `sky.sunx`, `sky.elevation`, `sky.sun`, `sky.shaft`, WEATHER.md §3.14; left: a lamp at the sun, the rainbow
  opposite it, a `frame shaft` key, the daylight following the hour), a storm that moves across the scene, trees
  bending by the Beaufort force (how far they sway is in the state, `sky.trees`, WEATHER.md §3.6; left: a layer that
  bends by it).
- **A job hook in the volume API**, so an engine's job system can run a volume's step (FUSE.md §9): **shipped** (24
  October 2026: `nvfx_volume_create_with_jobs(fx, run, context, &vol)`, NVFX_VERSION 10, ENGINES.md §12, INTEGRATION.md
  §3; every parallel loop of a step, a read or a picture is one fork-join on the engine, the same frames to the bit, no
  allocation; `VolumeApi.SameFieldsAndPicturesOnAnEnginesJobSystem`, `nvfx_alloc_test`). Left: the FUSE adapter still
  runs the volume on threads of its own inside one job (its `JobSystem::run` is game thread only, so the hook would
  nest: FUSE.md §4, §9); a start spread over frames (a step is one call).

## Contents

1. [The goal](#the-goal)
2. [Principles](#principles)
3. [The catalogue: every effect](#the-catalogue-every-effect)
4. [Weather systems](#weather-systems)
5. [The simulators](#the-simulators)
6. [The models](#the-models)
7. [The pipeline every effect goes through](#the-pipeline-every-effect-goes-through)
8. [Runtime and engines](#runtime-and-engines)
9. [3D scenes and the 3D library](#3d-scenes-and-the-3d-library)
10. [Measurement](#measurement)
11. [Order of work](#order-of-work)
12. [What it costs](#what-it-costs)
13. [Seeds](#seeds)
14. [What goes in git](#what-goes-in-git)

## The goal

Every visual effect a game engine needs, in 3D. Each one plays as today's fire does:
- from start points, with a learned stepper driven by seeded noise;
- pseudorandom but deterministic;
- for as long as it is wanted, in a vast range of settings;
- the same from any camera, so it can be walked around, flown through or seen from above.

Realism comes from teachers that are as accurate as the PC can run, and that are checked against measurements of the
real thing. Cost is chosen at the end from measured numbers, so the effects stay CPU-light where it counts.

## Principles

1. **Model what's ordered, leave random to random** (study M).
   - Learned: what has structure, such as plumes, rolls, fronts, waves, flames and collapses.
   - Seeded random: what is random at the scale it is seen, such as spray, sparks, foam, droplets, fine flicker,
     grains, leaves and rain streaks.
   - Exact physics, where it is cheap and known: wave dispersion, ballistics, settling, optics.
2. **Accuracy first, checked against the real thing.** Every teacher is an established simulator, or a solver of
   ours, that has passed checks against measurements before any training. Where nothing real exists (magic), say so.
3. **One family of models.**
   - The 3D rollout of stage 1, with channels per group.
   - A level-set variant for liquids.
   - A continuum-plus-grains variant for granular flows.
   - Seeded procedural growth for thin structures (lightning, frost, cracks).
4. **Deterministic to the bit.** The same seed gives the same frames on any number of threads, so every client of a
   multiplayer game sees the same fire.
5. **The engine's GPU draws; the library's CPU steps.** Volumes, level sets and particles go to the engine each frame.
   A CPU picture exists for tests, videos and engines without volumetrics.
6. **The house rules for measuring:**
   - rules are written before the test, and the test runs once;
   - held-out settings and seeds;
   - paired bootstrap 95% intervals with 10,000 resamples;
   - ties reported as ties, and nulls reported with their numbers.

## The catalogue: every effect

Every row has a number (#), which gives it its seeds (see [Seeds](#seeds)).
- **Tier 1** is the first wave on the PC, **Tier 2** the second, **Tier 3** later or hard.
- **"By eye against footage"** in the table below is the legacy label for rows with no single published scalar check.
  For **stage 2 / [PLAN_PC_GPU.md](docs/PLAN_PC_GPU.md)** the owner uses **sim ensembles only** (§1.3, §2.1 #7): capture
  the effect's **statistical range and spread**, not one footage sample; one `reference.md` per effect plus §6.4. Tier
  2+ rows keep the label until each gets a sim or correlation anchor.
- **"Consistency only"** means there is no real counterpart.

Today's 2D effects map onto these rows:

| 2D effect | 3D row | 2D effect | 3D row | 2D effect | 3D row |
|---|---|---|---|---|---|
| fire | 0, 1 | orb | 148 | launch | 32 |
| smoke | 16 | vortex | 10, 74 | puffs | 20 |
| explosion | 23 | fog | 58 | ink | 92 |
| steam | 61, 112, 117 | dust | 100 | waterfall | 82 |
| magic | 139 | cloud | 41 | fountain | 83 |
| jet | 4 | wildfire | 5 | splash | 85 |
| coals | 11 | mushroom | 28 | rainshaft | 51 |
| lava | 94 | eruption | 109 | avalanche | 105 |
| candle | 2 | smoke ring | 157 | heavy gas | 118, 19 |

The water system's sheets, jets, drips, bubbles and pools are rows 82 to 91.

### Fire and combustion

| # | effect | the case and what matters | simulator | checked against | tier |
|---|---|---|---|---|---|
| 0 | pool fire (fuel spill, oil fire) | a burning pool, 0.3 to 3 m; heat release rate, soot yield, wind | FDS | burning rate against Babrauskas's correlation; flame height against Heskestad's L = 0.235 Q^(2/5) - 1.02 D; puffing against f = 1.5 / sqrt(D); centreline temperature and velocity against McCaffrey | 1 |
| 1 | campfire and bonfire | a wood crib or a log pile; size, moisture, wind | FDS (solid pyrolysis) | crib burning rate against Block's experiments; flame height and puffing as #0 | 1 |
| 2 | torch and candle | small laminar to flickering flames; wick, draughts | PeleLM with a detailed mechanism (a DNS) | flame height and flicker frequency against measured candle flames | 2 |
| 3 | gas jet fire (burst pipe, leak) | a high-momentum gas jet burning; flow, angle, wind | FDS or PeleLM | flame length against Delichatsios's correlations; lift-off height against Kalghatgi's | 2 |
| 4 | flamethrower | a burning liquid stream (Lagrangian droplets); flow, aim, wind | FDS | jet flame length as #3; a gas jet flame against the Sandia piloted flame D data (TNF workshop) first | 2 |
| 5 | grass fire | a front spreading through grass; wind, fuel load, moisture, slope | FDS (vegetation, level set) | rate of spread and front depth against the CSIRO grassland experiments, as Mell and colleagues' simulations were | 1 |
| 6 | forest crown fire | fire climbing into and running through tree crowns | FDS (vegetation) | spread rate and flame heights against the International Crown Fire Modelling Experiment | 2 |
| 7 | room fire, flashover and backdraft | a fire in a compartment: hot layer, flashover, an opened door feeding a backdraft | FDS | hot-layer temperature against the McCaffrey-Quintiere-Harkleroad correlation; flows against Steckler's compartment experiments; flashover near a 600 °C upper layer | 2 |
| 8 | fire out of a window | flames and smoke leaving an opening up a facade | FDS | external flame height and temperature against Law's method (as in the Eurocode) | 2 |
| 9 | vehicle fire | a burning car: tyres, fuel, glass breaking | FDS | heat release curves against measured car fires | 2 |
| 10 | fire whirl | a fire in rotating flow; circulation, heat release rate | FDS | flame height against circulation from laboratory fire whirls | 2 |
| 11 | coals and embers | a bed of glowing charring fuel, low flames, hot air | FDS (char oxidation) | surface temperatures and flicker against infrared measurements of ember beds | 2 |
| 12 | BLEVE fireball | a ruptured tank's fireball; mass of fuel | PeleLM | diameter and duration against the CCPS correlations (D = 5.8 M^(1/3), D in m, M the fuel's mass in kg) | 2 |
| 13 | molotov (burning liquid splash) | a breaking bottle spreading burning fuel | Basilisk, then FDS | by eye against footage | 3 |
| 14 | fireworks | shells, stars, trails, crackles and smoke | our particles, with the LES for the smoke | star trajectories with drag; colours from metal salts' emission (strontium red, barium green, copper blue, sodium yellow) | 2 |
| 15 | ember showers and spotting | embers lofted from a fire, landing ahead of it | FDS (particles) | lofting and transport against Albini's spotting model | 2 |

### Smoke and hot gas

| # | effect | the case and what matters | simulator | checked against | tier |
|---|---|---|---|---|---|
| 16 | smoke column | a fire's plume far above its flames; wind bends it, the sky colours it | FDS | rise and spread against plume theory (Morton, Taylor and Turner); light blocked by soot's mass extinction coefficient, 8.7 m²/g for flaming soot (Mulholland and Croarkin) | 1 |
| 17 | black and white smoke | sooty tyre smoke; white smoke from wet fuel (water droplets) | FDS | smoke yields and extinction against measurements | 2 |
| 18 | coloured smoke grenade | a dense dye aerosol, nearly unbuoyant, drifting | FDS (no combustion) | by eye against footage | 2 |
| 19 | smoke filling a room | a ceiling jet, a descending layer, smoke leaking under doors | FDS | ceiling-jet temperature and velocity against Alpert's correlations | 2 |
| 20 | chimney and exhaust puffs (a steam train) | pulsed releases rising as thermals | FDS | puff rise and growth against Scorer's thermals | 2 |
| 21 | cigarette and incense wisps | a laminar plume that turns turbulent | a DNS (PeleLM or MicroHH) | transition height against measured laminar plumes | 3 |
| 22 | smouldering | smoke without flame: thin, pale, slow | FDS | smoke yield against measurements | 3 |
| 157 | smoke and vortex rings | a ring puffed from an opening, rising and growing as it rolls, then breaking up | PeleLM or MicroHH (a DNS) | the formation number (about 4) against Gharib, Rambod and Shariff's experiments; ring speed against vortex-ring theory | 2 |

### Explosions, blasts and impacts

| # | effect | the case and what matters | simulator | checked against | tier |
|---|---|---|---|---|---|
| 23 | high-explosive ground burst | blast, fireball, dust skirt, crater | PeleC, then PeleLM; ejecta by DEM | overpressure and impulse against distance by the Kingery-Bulmash charts; crater size against Holsapple's crater scaling | 1 |
| 24 | air burst | a burst above the ground; the reflected shock forms a Mach stem | PeleC | triple-point height against the reflected-blast charts (UFC 3-340-02) | 2 |
| 25 | vapour-cloud explosion | a fuel-air cloud deflagrating among obstacles | PeleC | blast strength against the TNO multi-energy and Baker-Strehlow-Tang methods | 2 |
| 26 | grenade with fragments | a small burst throwing fragments and dirt | PeleC, with fragments as particles | fragment speed by Gurney's equations; fragment sizes by Mott's distribution | 2 |
| 27 | impact with ejecta (a shell or meteor) | crater, ejecta curtain, dust | PeleC with DEM or MPM ejecta | crater and ejecta against Holsapple's scaling | 2 |
| 28 | mushroom cloud | a large burst's fireball rising on a stem into a rolling cap, through stratified air | PeleC, then PeleLM | rise against buoyant-thermal theory in stratified air (Morton, Taylor and Turner) | 2 |
| 29 | condensation (Wilson) cloud | humid air condensing in the blast's negative phase | PeleC with moist thermodynamics | by eye against footage of blasts in humid air | 2 |
| 30 | underwater explosion | a gas bubble pulsing, a water dome and plume | Basilisk (compressible) | bubble period against the Rayleigh-Willis formula | 3 |
| 31 | dust explosion | a dust cloud deflagrating (a mill, a mine) | PeleC with particles | rate of pressure rise against dust explosion tests (K_St) | 3 |

### Engines, rockets and guns

| # | effect | the case and what matters | simulator | checked against | tier |
|---|---|---|---|---|---|
| 32 | rocket launch | under-expanded exhaust on the pad, a ground cloud billowing both ways, the deluge's steam | PeleC | Mach disk location against x/D = 0.67 sqrt(p0 / pa) (Crist, Sherman and Glass) | 2 |
| 33 | rocket or missile in flight | an afterburning plume and its smoke trail | PeleC, then the LES | by eye against footage | 2 |
| 34 | rocket landing | the plume eroding the ground and spraying dust | PeleC with DEM | erosion and spray against plume-surface interaction tests | 3 |
| 35 | jet afterburner | shock diamonds in a hot exhaust | PeleC | shock-cell spacing against Pack's formula | 2 |
| 36 | contrails | condensation trails forming, spreading and fading | the atmospheric LES with ice | formation against the Schmidt-Appleman criterion | 2 |
| 37 | helicopter downwash | a rotor's wash lifting dust (brownout) or water spray | the LES with particles | induced velocity against momentum theory | 2 |
| 38 | muzzle flash and blast | the flash, the blast and the smoke at a gun's muzzle | PeleC | flash regions against Klingenberg's measurements | 2 |
| 39 | tyre burnout smoke | dense white smoke from a spinning tyre | FDS | by eye against footage | 3 |
| 40 | vehicle exhaust | a cold start's steam and soot, diesel smoke | FDS | by eye against footage | 3 |

### Weather: clouds

| # | effect | the case and what matters | simulator | checked against | tier |
|---|---|---|---|---|---|
| 41 | fair-weather cumulus | shallow cumulus over a warm sea or land | PALM or MicroHH | cloud fraction, liquid water and cloud base and top against the BOMEX intercomparison (Siebesma and colleagues) | 1 |
| 42 | towering cumulus | clouds deepening through the afternoon | PALM or MicroHH | the shallow-to-deep transition intercomparison (the LBA case) | 2 |
| 43 | stratocumulus deck | a broken deck under an inversion | PALM or MicroHH | the DYCOMS-II intercomparisons (RF01, and RF02 for drizzle) | 2 |
| 44 | stratus and overcast | a grey sky with a moving texture | PALM or MicroHH | as #43 | 1 |
| 45 | altocumulus and mid-level layers | cloudlets in rows at mid levels | PALM or MicroHH | by eye against footage | 3 |
| 46 | cirrus and contrail cirrus | thin ice clouds and fall streaks | PALM or MicroHH (ice) | the GCSS cirrus cases | 2 |
| 47 | cumulonimbus | anvil, overshooting top, mammatus | CM1 | the Weisman-Klemp idealised storm | 2 |
| 48 | lenticular and cap clouds | standing clouds over mountains | CM1 or PALM with terrain | wavelength against linear mountain-wave theory (the Scorer parameter) | 2 |
| 49 | pyrocumulus | a fire's plume capped by its own cloud | FDS coupled with the moist LES | plume rise with moisture; by eye against footage | 3 |

### Weather: precipitation

| # | effect | the case and what matters | simulator | checked against | tier |
|---|---|---|---|---|---|
| 50 | drizzle | fine drops from low cloud | PALM or MicroHH | as #43 (RF02) | 2 |
| 51 | rain and rainshafts | light to torrential rain, shafts below a cloud base, the cold outflow | PALM or MicroHH, CM1 for storms | drop sizes against Marshall-Palmer; fall speeds against Gunn and Kinzer; drop shapes against Beard and Chuang; the outflow front against gravity-current theory (Benjamin) | 1 |
| 52 | virga | rain evaporating before it reaches the ground | PALM or MicroHH | evaporation against the microphysics' own checks | 2 |
| 53 | snowfall | light to heavy snow, flakes fluttering | PALM or MicroHH, our flake dynamics | flake fall speeds against Locatelli and Hobbs | 1 |
| 54 | blizzard and blowing snow | falling and blown snow, whiteout | PALM with saltation | blowing-snow flux against field measurements | 2 |
| 55 | sleet and freezing rain | rain freezing in a cold layer or on contact (glaze) | PALM or MicroHH | the melting-layer profiles | 3 |
| 56 | hail | hailstones growing in updrafts and falling, bouncing | CM1 with hail microphysics | hail sizes and fall speeds against measurements | 2 |
| 57 | graupel and diamond dust | soft pellets; glittering ice crystals in clear cold air | PALM or MicroHH | halos (#133) | 3 |

### Weather: fog and haze

| # | effect | the case and what matters | simulator | checked against | tier |
|---|---|---|---|---|---|
| 58 | radiation fog | fog forming in a calm cooling night, lifting in the morning | PALM | liquid water and depth against a measured fog case (tower measurements, such as those of Cabauw or Cardington) | 1 |
| 59 | advection and sea fog | fog rolling in over cold ground or sea | PALM | by eye against footage; visibility against liquid water | 2 |
| 60 | valley fog and cold pools | fog filling valleys | PALM with terrain | by eye against footage | 2 |
| 61 | steam fog | fog smoking off warm water into cold air | PALM or MicroHH | saturation by mixing | 2 |
| 62 | haze and smog | aerosol dimming and colouring the distance | the sky model (#130) | aerosol optical depth against measurements | 1 |
| 63 | low cloud over hills | cloud sweeping over ridges | PALM with terrain | by eye against footage | 3 |

### Weather: storms

| # | effect | the case and what matters | simulator | checked against | tier |
|---|---|---|---|---|---|
| 64 | thunderstorm (single and multicell) | towers, anvils, rain and outflow, lightning | CM1 | the Weisman-Klemp storm with weak shear | 2 |
| 65 | squall line and shelf cloud | a line of storms with a shelf cloud on its gust front | CM1 | Rotunno, Klemp and Weisman's theory | 2 |
| 66 | supercell | a rotating storm: wall cloud, rear-flank downdraft, hail | CM1 | the Weisman-Klemp storm with strong shear | 3 |
| 67 | tornado | a funnel, its debris cloud, multiple vortices | CM1 at very high resolution | Orf and colleagues' simulated tornado; wind profiles against Doppler radar observations; laboratory vortex chambers (swirl ratio) | 3 |
| 68 | waterspout | a vortex over water with a spray ring | CM1 or PALM | by eye against footage | 3 |
| 69 | downburst and microburst | a cold downdraft hitting the ground and spreading | CM1 | outflow against Fujita's analyses and the JAWS measurements | 2 |
| 70 | haboob (dust storm) | a wall of dust lifted by a storm's outflow | CM1 with dust | a gravity-current front; dust lifting by saltation (Bagnold) | 2 |
| 71 | sandstorm | sand and dust streaming low in strong wind | PALM with saltation | saltation flux against Bagnold's law | 2 |
| 72 | hurricane | near: rain bands, wind, spray; far: the eyewall and bands | CM1 | the Rotunno-Emanuel idealised hurricane | 3 |

### Weather: wind and what it carries

| # | effect | the case and what matters | simulator | checked against | tier |
|---|---|---|---|---|---|
| 73 | gusts and turbulent wind | the wind field near the ground, around buildings and trees | PALM | spectra against the Kaimal and von Kármán spectra | 1 |
| 74 | dust devils | vortices forming in a hot afternoon, lifting dust | PALM with particles | dust devils in boundary-layer LES (Raasch and Franke); core radius and pressure drop against field measurements | 2 |
| 75 | drifting snow and spindrift | snow streaming over ground and ridges | PALM with saltation | as #54 | 2 |
| 76 | sand streamers | sand skimming over dunes | PALM with saltation | Bagnold's law | 3 |
| 77 | sea spray | spray torn off whitecaps | Basilisk, then the LES | sea-spray generation against measurements | 2 |

Lightning is #122 and aurora #128 (electrical); rainbows, halos and the sky are optics (#130 to #138).

### Water

| # | effect | the case and what matters | simulator | checked against | tier |
|---|---|---|---|---|---|
| 78 | open ocean | wind waves and swell, whitecaps, foam | exact spectral waves with seeded phases; Basilisk for breaking | spectra against JONSWAP and Pierson-Moskowitz; whitecap coverage against Monahan's law | 1 |
| 79 | breaking waves and surf | plunging and spilling breakers on a beach, swash | Basilisk | breaker type against the Iribarren number; breaking against Deike, Melville and Popinet's simulations and experiments | 2 |
| 80 | ship wakes and bow spray | a hull's Kelvin wake, a speedboat's rooster tail | Basilisk | the Kelvin wake angle (19.47°) | 2 |
| 81 | rivers, rapids and hydraulic jumps | flow over rocks and weirs, standing waves | Basilisk | jump heights against Bélanger's equation | 2 |
| 82 | waterfall | a sheet or column breaking up, plunging, spray and mist | Basilisk, the LES for the mist | break-up length against falling-jet measurements | 1 |
| 83 | fountain | a jet thrown up, breaking into drops, falling into a pool | Basilisk | trajectory with drag; break-up length (Rayleigh-Plateau) | 1 |
| 84 | hoses and firefighting jets | a jet's reach and break-up; water hitting fire | Basilisk, coupled with FDS | reach against hose-stream measurements | 2 |
| 85 | splash | an object or a character entering water: cavity, crown, Worthington jet, droplets | Basilisk | the splashing threshold K = Oh Re^1.25 (Mundo, Sommerfeld and Tropea); cavities against Duez and colleagues' sphere-entry experiments | 1 |
| 86 | rain on water and puddles | crowns, rings, capillary ripples | Basilisk | ring speeds against the capillary-gravity dispersion relation | 1 |
| 87 | drips, drops and rivulets on glass | dripping taps; drops sliding and merging on a window | Basilisk with contact angles | contact-angle hysteresis; dripping-tap regimes | 2 |
| 88 | geyser | periodic eruptions of water and steam | Basilisk with phase change, or designed | intervals and heights against geyser records | 3 |
| 89 | flood and dam-break wave | a wall of water through streets | Basilisk | Ritter's solution and dam-break experiments | 3 |
| 90 | sloshing and pouring | water in a tank, a glass filling | Basilisk | sloshing frequencies against tank experiments | 3 |
| 91 | bubbles and bubble plumes | bubbles rising and wobbling; a bubble curtain | Basilisk | terminal velocities and shapes against Clift, Grace and Weber | 1 |
| 92 | ink, silt and blood in water | a slightly dense plume curling and spreading | Basilisk (single phase) | spreading against plume theory; Rayleigh-Taylor growth against Dimonte and colleagues | 2 |
| 93 | propeller wash and cavitation | a propeller's wake and vapour bubbles | needs a cavitation model | by eye against footage | 3 |

### Other liquids

| # | effect | the case and what matters | simulator | checked against | tier |
|---|---|---|---|---|---|
| 94 | lava flow | glowing flow, cooling crust, folds and breakouts | Basilisk with temperature-dependent viscosity, or our solver | Griffiths and Fink's wax analogue experiments; length against effusion rate (Walker) | 2 |
| 95 | mud and debris flow | thick flow with boulders | Basilisk (Herschel-Bulkley) or MPM | the USGS debris-flow flume experiments | 2 |
| 96 | blood spatter and pools | drops in flight, stains, pooling | Basilisk | stain shapes against impact angle (sin θ = width / length) | 2 |
| 97 | viscous liquids (honey, slime, paint) | coiling, buckling and slow spreading | Basilisk | coiling frequencies against viscous rope-coiling experiments | 3 |
| 98 | oil slick on water | a spreading film, rainbow sheen | our spreading model | spreading against Fay's regimes; sheen by thin-film interference | 3 |
| 99 | molten metal pour | a glowing stream, splashes and sparks | Basilisk | by eye against footage | 3 |

### Granular and particle-laden flows

| # | effect | the case and what matters | simulator | checked against | tier |
|---|---|---|---|---|---|
| 100 | dust burst | dust thrown up by an impact or a fall, spreading and settling | the LES with particles | settling against measured terminal velocities; the spreading front against particle-laden gravity currents | 1 |
| 101 | vehicle dust wake | dust behind a vehicle on a dirt road | the LES with particles | by eye against footage | 1 |
| 102 | footstep and landing dust | small puffs from feet and landings | the LES with particles | by eye against footage | 3 |
| 103 | sand: falling, piles and slides | sand pouring, piling and sliding | DEM (LIGGGHTS) or MPM | angle of repose; flow against the μ(I) rheology | 2 |
| 104 | building collapse with dust | rubble falling, a dust cloud rolling out | DEM with the LES | runout against granular column collapse (Lube and colleagues, Lajeunesse and colleagues) | 2 |
| 105 | snow avalanche | a dense flow under a powder cloud | MPM with the LES | the Vallée de la Sionne measurements | 2 |
| 106 | landslide and rockfall | slides and falling rock with dust | DEM or MPM | runout against volume (Heim's ratio, Scheidegger) | 2 |
| 107 | leaves and petals in wind | fluttering, tumbling, swirling in gusts | our coupled light solids | falling regimes against Field and colleagues' falling-disk experiments | 2 |
| 108 | ash fall | ash settling from a plume | the LES with particles | settling velocities | 2 |

### Volcanic

| # | effect | the case and what matters | simulator | checked against | tier |
|---|---|---|---|---|---|
| 109 | eruption column | a Plinian column rising and spreading as an umbrella cloud | a multiphase LES (choose the code when this starts) | height against eruption rate (Mastin and colleagues: H = 2.00 V^0.241, H in km, V the dense-rock eruption rate in m³/s) | 2 |
| 110 | pyroclastic flow | a hot, ground-hugging cloud of ash and gas | a multiphase LES | the large-scale PELE experiments (Lube and colleagues) | 2 |
| 111 | lava fountain | molten rock jetting and falling as spatter | ballistics, with Basilisk for the spatter | fountain heights against observations | 2 |
| 112 | fumaroles and steam vents | steam jetting and drifting | PALM or MicroHH | as steam (#61) | 2 |
| 113 | volcanic lightning | lightning in an ash plume | the lightning model (#122) in the plume | by eye against footage | 3 |
| 114 | lahar | a volcanic mudflow | as #95 | as #95 | 3 |

### Ice, snow and cold

| # | effect | the case and what matters | simulator | checked against | tier |
|---|---|---|---|---|---|
| 115 | frost on surfaces and windows | frost growing in ferns and feathers | our growth models (Reiter's automaton, Gravner and Griffeath's) | crystal habits against the Nakaya diagram and Libbrecht's experiments | 2 |
| 116 | snow crystal growth close-up | a single crystal growing | as #115 | as #115 | 3 |
| 117 | freezing breath and cold steam | breath and kettles condensing in cold air | MicroHH | saturation by mixing | 2 |
| 118 | dry-ice and liquid-nitrogen fog | cold, heavy fog flowing over edges | PALM or MicroHH | front speed against gravity-current theory (Benjamin) | 1 |
| 119 | icicles | icicles growing and dripping | our growth model | shapes against Short, Baygents and Goldstein's theory | 3 |
| 120 | ice cracking and shattering | cracks running, ice breaking | our fracture model | by eye against footage | 3 |
| 121 | snow accumulation and melt | snow building up and melting on surfaces | our accumulation model | by eye against footage | 3 |

### Electrical and plasma

| # | effect | the case and what matters | simulator | checked against | tier |
|---|---|---|---|---|---|
| 122 | lightning | stepped leaders, branching, the return stroke, flicker, afterglow; cloud-to-ground and in-cloud | our dielectric breakdown model (Niemeyer, Pietronero and Wiesmann) | leader steps and branching against high-speed camera records (Rakov and Uman's compilation) | 2 |
| 123 | electric arcs and shorts | arcs climbing (a Jacob's ladder), shorting sparks | designed on the breakdown model | by eye against footage | 3 |
| 124 | welding arc and sparks | the arc's glare, spatter, sparks | our particles | spark bursts against carbon content (spark-testing charts) | 2 |
| 125 | grinder sparks | a fan of sparks, some bursting | our particles | as #124 | 2 |
| 126 | plasma globe | filaments reaching for a hand | designed | by eye against footage | 3 |
| 127 | St Elmo's fire | a corona glow on points | designed | by eye against footage | 3 |
| 128 | aurora | curtains folding and rippling | designed dynamics with physical light | emission lines (557.7 nm green, 630.0 nm red, 427.8 nm blue); by eye against all-sky footage | 2 |
| 129 | ball lightning | a glowing ball drifting | designed | consistency only | 3 |

### Optics and light (computed from the volumes and the camera, not learned)

| # | effect | the case and what matters | method | checked against | tier |
|---|---|---|---|---|---|
| 130 | sky and sun | day, sunset, twilight, night | precomputed atmospheric scattering (Bruneton's method) | sky radiance against the CIE standard skies | 1 |
| 131 | crepuscular and god rays | light shafts through cloud, smoke and trees | single scattering in the volumes | by eye against footage | 1 |
| 132 | rainbow and fogbow | primary, secondary and supernumerary bows | Mie and Airy theory from the drop sizes | angles (42° and 51°) and the bows' widths for the drop sizes | 2 |
| 133 | halos and sun dogs | the 22° halo and sun dogs from ice crystals | ray tracing through hexagonal crystals | angles and colours | 3 |
| 134 | coronas and glories | coloured rings around the sun or a shadow | Mie theory | ring radii against droplet size | 3 |
| 135 | heat haze and mirages | shimmer over hot ground and exhausts; inferior and superior mirages | refraction from the temperature field (Gladstone-Dale) | mirage geometry against the temperature profile | 1 |
| 136 | caustics and underwater light shafts | light focused by a moving water surface | from the water surface (#78 to #91) | by eye against footage | 2 |
| 137 | cloud shadows | shadows running over the land | from the cloud volumes | consistency only | 1 |
| 138 | fire and lightning light on the scene | light from emission, flickering with it | from the volumes' emission | consistency only | 1 |

### Designed effects (no real counterpart)

These are real 3D fluids with designed sources and forces, as the 2D presets are. They are checked for consistency
only, and they say so.

| # | effect | # | effect | # | effect |
|---|---|---|---|---|---|
| 139 | portal | 143 | auras and buffs | 147 | healing motes |
| 140 | spell projectile | 144 | teleport | 148 | plasma orb and energy core |
| 141 | shield bubble | 145 | summoning circle | 149 | force-push shockwave |
| 142 | energy beam | 146 | curse smoke | 150 | black hole with lensing |

All are tier 2, except the black hole (tier 3).

### Coupled solids (they move with the fluids; a separate plan)

These need mesh-based learned simulators, a different model family, which gets a plan of its own when the fluids are
done. All are tier 3.

| # | effect | # | effect |
|---|---|---|---|
| 151 | cloth and flags in wind | 154 | objects floating and sinking |
| 152 | grass and trees in wind | 155 | fracture of glass, wood and stone |
| 153 | debris lifted by wind and tornadoes | 156 | objects burning, charring and collapsing |

## Weather systems

A weather system is the state of the sky and the air over a whole world, changing over minutes to days. It drives the
local effects near the camera.

### Two scales

- **Regional**, 10 to 50 km across, minutes to days:
  - the cloud field, wind, rain and snow fields, fronts, storms, and the day-night cycle;
  - simulated by CM1, PALM, MicroHH, or WRF in LES mode (for real terrain);
  - learned as a coarse 3D rollout over a large domain (for example 64 x 64 x 32 cells over 20 km), conditioned on the
    large-scale forcing (sounding, surface fluxes, wind shear, time of day);
  - seeded and deterministic, stepped a few times a second and interpolated.
- **Local**, tens of metres around the camera, every frame:
  - rain streaks and splashes, snow, gusts, leaves, fog wisps, lightning bolts, drifting snow and spray;
  - fed by the regional state at the camera: rain rate and drop sizes, wind, temperature, humidity, visibility, cloud
    base and lightning probability.

### Regimes

Each regime is a set of soundings and forcings whose simulated runs are its training data.

| regime | what you see | the case it is simulated from | checked against |
|---|---|---|---|
| clear and calm | blue sky, haze, heat shimmer by day, stars and radiation fog by night | a dry boundary layer through a day | the sky model; fog as #58 |
| fair weather | cumulus over land or sea | BOMEX; a diurnal land case | BOMEX |
| trade cumulus with showers | cumulus raining in short showers | RICO | the RICO intercomparison (vanZanten and colleagues) |
| overcast | stratocumulus or stratus, drizzle | DYCOMS-II | RF01 and RF02 |
| steady rain | a warm front's nimbostratus | an idealised warm front | rain rates and drop sizes |
| showers | scattered convective showers | a moist, weakly sheared sounding | rain statistics |
| thunderstorms | towers, anvils, lightning, outflow | the Weisman-Klemp sounding, weak shear | the Weisman-Klemp storm |
| squall line | a line of storms, a shelf cloud, a gust front | the RKW case | Rotunno, Klemp and Weisman |
| supercell and tornado | a rotating storm, hail, a funnel | the Weisman-Klemp sounding, strong shear | the Weisman-Klemp storm; Orf and colleagues |
| hail storm | hail falling, bouncing, piling | the supercell with hail microphysics | hail sizes |
| downburst | a sudden cold blast of wind and rain | a microburst case | JAWS |
| dust storm | a haboob wall, then dusty dimness | an outflow over dry ground | gravity-current speeds |
| sandstorm | streaming sand, low visibility | strong wind over sand | Bagnold |
| snowfall | light to heavy snow | a cold moist profile | flake fall speeds |
| blizzard | heavy and blowing snow, whiteout | cold, windy, snowing | blowing-snow flux |
| sleet and freezing rain | ice pellets, glaze on everything | a warm nose over a cold layer | the melting-layer profile |
| fog | radiation, advection or valley fog | as #58 to #60 | as #58 |
| heat wave | haze, mirages, convective plumes, dust devils | a hot dry boundary layer | dust devils as #74 |
| hurricane | near: extreme wind, rain bands, spray; far: the eyewall | the Rotunno-Emanuel case | the idealised hurricane |
| mountain weather | lenticular clouds, cap clouds, föhn, orographic rain | flow over terrain | mountain-wave theory |
| polar | diamond dust, halos, aurora, blowing snow | a cold, clear, stable layer | halos; aurora lines |
| monsoon downpour | torrential rain, flooding streets | a very moist sounding | rain rates |

### Transitions, controls and couplings

- **Transitions.** An engine asks for a regime and a time ("a storm in ten minutes", "clearing by dawn"). The regional
  model's controls move smoothly from one forcing to the next, deterministically from the world's seed. Time of day
  and season are controls too. The 2D weather director ([docs/WEATHER.md](docs/WEATHER.md)) does this today for one
  place, with targets chosen rather than simulated; the regional model would supply them.
- **Couplings with the local effects** (several exist in 2D today):
  - rain wets surfaces, thins smoke and puts out fires;
  - wind bends flames and smoke, carries embers ahead of a wildfire, builds waves and drives drifting snow;
  - snow accumulates and melts;
  - lightning ignites fires;
  - fog and haze set the visibility;
  - clouds cast moving shadows;
  - the air's temperature decides whether breath and steam show.
- **Light.** The sky model with the clouds' multiple scattering, sun shafts, lightning lighting clouds from inside, and
  the fire's glow on low cloud and smoke.
- **Measurement.**
  - Regional statistics against the simulations: cloud fraction, cloud base and top, rain rate, wind spectra.
  - The simulations against their intercomparison cases.
  - Local statistics against measurements: drop sizes and fall speeds, flake fall speeds, gust spectra.

## The simulators

All of them run outside the repository. The repository holds only our case files (small text inputs) and readers for
their output. Check each licence for using its outputs as training data, and record it in the licence register
(`docs/DATA.md`).

| simulator | for | notes |
|---|---|---|
| FDS (NIST) | fire, smoke, wildfire, compartments | Fortran, public domain; LES, low-Mach, combustion, soot, radiation, vegetation, particles; MPI |
| PeleC and PeleLM | blasts, rockets, jet flames, small flames (DNS) | C++, open source; compressible and low-Mach reacting flow with adaptive mesh refinement |
| CM1 | thunderstorms, supercells, tornadoes, squall lines, hurricanes, downbursts | Fortran, freely available; the standard for idealised storms |
| PALM | boundary layers, clouds, fog, dust devils, saltation, terrain, cities | Fortran, GPL; LES with cloud microphysics and Lagrangian particles |
| MicroHH | clouds, plumes, DNS of small flows | C++, GPL; LES and DNS with moist thermodynamics |
| WRF (LES mode) | regional weather over real terrain | Fortran, public domain |
| Basilisk | splashes, waves, jets, drops, bubbles, liquids | C, GPL; adaptive octree, volume of fluid, surface tension |
| LIGGGHTS | granular flows (DEM) | C++, GPL |
| an MPM code (chosen at the time) | snow, mud and sand at large deformation | |
| our own (C++, in the repository) | stage 1's solver, particles, lightning, frost and icicle growth, ocean spectra, optics | tested here |

## The models

| family | for | the state (coarse, learned) | seeded random (fine) |
|---|---|---|---|
| gas volume | fire, smoke, explosions after their shocks, clouds, fog, steam, dust clouds, designed | velocity, temperature, the group's scalars (soot, vapour, cloud water, rain, ice, dust), memory | fine temperature and scalars, flicker, grains |
| free surface | waterfalls, fountains, splashes, rivers, surf, lava, mud | velocity, liquid fraction or level set, memory | spray, foam, bubbles, droplets as particles |
| ocean | open water | exact spectral waves with seeded phases; learned breaking and whitecaps | foam and spray |
| granular | sand, rubble, avalanches, landslides | density and velocity of the continuum | the visible grains and rocks as particles |
| regional weather | weather systems | the atmosphere's coarse state, conditioned on forcing | local weather effects |
| thin structures | lightning, frost, cracks, icicles | none: procedural growth models, seeded | the whole effect |

Every learned family:
- steps with 3 x 3 x 3 convolutions with FiLM on the controls;
- advects (semi-Lagrangian or MacCormack) and projects pressure in 3D;
- is driven by seeded 3D noise;
- grows from start points;
- plays forever with shards;
- trains with hand-written gradients checked against finite differences, the statistics loss (study D2) and long
  windows (study L).

## The pipeline every effect goes through

0. **Measure the PC** once: cores, RAM, disk, CPU features. Record them in `results/3d/pc.md`.
1. **Pilot.**
   - One case at three resolutions, each twice as fine as the last.
   - The row's checks must converge, then pass within the measurements' scatter.
   - Acceptance bands go in `results/3d/<group>/validation.md` before the runs.
   - Measure the memory per cell and the time per cell-step. The production grid is the largest that fits in 80% of
     RAM at an acceptable time per run.
2. **Production runs.**
   - Settings that matter in a game, with several seeds per setting.
   - Looping effects: a warm-up, then 30 to 60 s at 30 fps. One-shots: from start to end.
   - The held-out split goes in `results/3d/<effect>/split.md` before the first production run. Splits are by run.
3. **Ingest.**
   - One reader per simulator, into one 3D clip format: FDS slices or Plot3D, AMReX plotfiles, NetCDF (PALM, MicroHH,
     CM1, WRF), and a Basilisk output routine of ours.
   - Ingest each chunk as it is written, then delete it.
   - A 256³ grid of six float32 quantities is 402 MB a frame, so 362 GB for 30 s at 30 fps. Keep the coarse grids, the
     fine fields and 16-bit storage. Keep full fields for a few runs only.
4. **Train.**
   - Start where stage 1 ends. Climb (48³ and 64³ grids, steppers 48 to 64 wide, 8 memory channels, 256³ detail),
     keeping each step only if it wins on validation.
   - Keep the stepper at most 64 channels wide, or extend the int8 path, which takes at most 64.
   - Checkpoints, so it can stop and resume.
5. **Measure** ([Measurement](#measurement)).
6. **Ship**, if it wins:
   - the effect pack;
   - frozen keyframes, pinned to float;
   - demo videos;
   - library entries.

   Today's 2D effect stays as the light option.

## Runtime and engines

- **The volume API** (`include/neuralfx/nvfx_volume.h`, started in stage 1):
  - load, create, seed, controls, step;
  - fields as 3D textures, level sets and particle lists;
  - a CPU picture for a camera;
  - no allocation per frame, per-ISA SIMD, int8, threads and shards.
- **Sparse volumes.** Only bricks with material in them are stored and stepped. They are uploaded as a brick atlas, as
  sparse volume formats do.
- **Level of detail.** Grid size (16³, 32³ or 64³ per instance), step rate (every frame, every other frame), and the 2D
  sprite for the far distance. The effect manager (budget, sharing, proxies, interpolation) extends to volumes.
- **For FUSE:**
  - gases as 3D textures or bricks for its volumetric renderer, lit with the scene;
  - emission as a light source, flickering with the effect;
  - liquids as level sets to mesh or ray-march;
  - particles as instanced sprites or meshes;
  - weather as a state the game reads (rain rate, wind) and sets (regime, time).
- **The weather API** (`nvfx_weather.h`):
  - set the regime, time and season;
  - read the local state at a point;
  - get the regional cloud volume and wind field;
  - deterministic from the world's seed.
- **The ABI test, `INTEGRATION.md`'s sizes and `NVFX_VERSION`** are updated with every public struct.

## 3D scenes and the 3D library

- **The scene language in 3D:**
  - positions, sizes and wind as 3D vectors, and a camera;
  - modules as volumes in the world;
  - the field bus as a 3D grid, with couplings (pushes, transfers, hand-overs, shocks) and force fields in 3D;
  - inputs and rules as today.
- **The procedural parts in 3D:** particles (sparks, embers, debris, flakes, rain, snow, ash, leaves, hail, spatter,
  shards, streaks), emitters (beams, shields, glints, meteors) and the water system's pieces.
- **The library rebuilt in 3D:** every 2D entry (149 today, more with rounds 7 and 8) made again from the 3D effects,
  plus new entries for every catalogue row, each checked from several cameras.
- **A 3D scene C API** for engines, as `nvfx_scene.h` is for 2D.

## Measurement

For every effect:
- **Against its simulation, in 3D:**
  - tracking from the true state (8 frames, 1 s, 2 s), as field errors and as pictures from several cameras;
  - endless statistics (the 3D detail spectrum, motion, coverage, emission or extinction, the mean volume).
- **Against reality:** the model's own runs are measured with the row's physical checks, exactly as the simulation's
  were. If real footage is supplied, also compare it with study J's statistics. If there is none, say so.
- **Against cheaper ways:**
  - the simulator run at the model's coarse grid with the same seeded detail (study D0's baseline);
  - today's 2D sprite from the front.
- **Cost:**
  - the step and the picture separately, at each grid size;
  - 1 and 4 threads; AVX2 and AVX-512; float and int8;
  - memory per instance.
- **Demo videos** in `videos/`: each effect walked around, swept through its settings, and beside its simulation.

What the later studies here taught the rules:
- **Size margins and the number of settings from a null probe first** (study M3, REPORT §14.9): re-run the current
  model with fresh seeds, measure how much each measure moves, and choose margins and settings with known power.
  Margins tighter than the measures' own noise reject unchanged models (study M2).
- **Calibrate any bound on the teacher's run-to-run spread**, from several runs, never one (study N2).
- **Choose on endless play as well as tracking** (studies N and N2).

## Order of work

0. **Stage 1, here (done):** study N's fire and smoke, and N2's fix. They established the 3D code, the model, the
   renderer, the volume API and the videos.
1. **Wave 1 (PC, tier 1)** proves every model family once:
   - the checklist, row by row (what the container has, the PC phase, what waits on R2, Q2 or hres, the scripts): [docs/WAVE1_CHECKLIST.md](docs/WAVE1_CHECKLIST.md);
   - pool fire (#0), campfire (#1), grass fire (#5), smoke column (#16) and ground-burst explosion (#23);
   - fair-weather cumulus (#41), overcast (#44), rain (#51), snow (#53), radiation fog (#58) and haze (#62);
   - wind (#73), ocean (#78), waterfall (#82), fountain (#83), splash (#85), rain on water (#86) and bubbles (#91);
   - dust burst (#100), vehicle dust wake (#101) and cold fog (#118);
   - optics #130, #131, #135, #137 and #138;
   - the first weather regimes: clear, fair weather, overcast, steady rain, snowfall and fog;
   - the volume API, sparse volumes and the manager for volumes;
   - the 3D scene language.
2. **Wave 2 (tier 2):**
   - the rest of fire, smoke, blasts and engines;
   - storms (thunderstorm, squall line, downburst, dust storm and sandstorm);
   - lightning, blizzard, rivers, surf, lava, mud and granular flows;
   - volcanic effects, frost, aurora and rainbows;
   - the designed effects;
   - the library rebuilt in 3D;
   - the full weather system with transitions.
3. **Wave 3 (tier 3):** supercells and tornadoes, hurricanes, the hardest liquids, halos and coronas, the remaining
   designed effects, and coupled solids (their own plan).

## What it costs

### To play

These are estimates scaled from today's 2D numbers, to be measured.

- **The 2D stepper** works on 32 x 32 cells with 3 x 3 convolutions. Its network stage takes about 0.17 ms a frame on
  one core at float, and 0.10 ms at int8 with VNNI (REPORT.md §13.8).
- **A 32³ stepper** has about 100 times that work:
  - about 16 ms a frame on one core at float, and 10 at int8;
  - about 3 to 4 ms on four cores.
- **A 64³ stepper** is 8 times that again.
- **Sparse bricks** cut this to the cells with material in them. Lower step rates and sharing cut it further.
- **A CPU picture** is about 6 million samples at 256 x 256 px and 96 steps a ray. With the engine's GPU drawing the
  volumes, the CPU pays only the step.
- **Stage 1 measures the first real numbers.**

### To make

Days to weeks of the PC per effect for the simulations, and as much again for training. The pilots measure the real
numbers.

Storage is the binding limit: ingest as runs are written. The disk needs room for one run's raw output at a time,
plus every effect's kept grids.

## Seeds

- **Stage 1 (study N):** 9,000,000 to 9,999,999.
- **Stage 2:** catalogue row # owns 10,000,000 + 100,000 x #, up to 25,799,999 for row 157. The weather regimes take
  30,000,000 + 100,000 x their row in the regimes table.

Ranges already used elsewhere:
- 5.3 to 5.9 million: studies G, K and L;
- 3.0 to 3.5 million: study O; 3.6 to 3.9 million: study P;
- 6.0 to 6.8 million: round 9's effects (earth 6.0 to 6.4, air 6.4 to 6.8); 6.9 million: an earlier smoke test;
- 26 million: round 10's fire effects (the test 26,000,000 to 26,150,999; the library's and the scene's module seeds,
  which a script takes only up to 2^24, are 2,630,001 and up);
- 7 million: water;
- 27 million: round 10's elements (27,000,000 to 27,999,999; the library's and the scene's module seeds 2,700,001 to
  2,710,031, as a script takes seeds up to 2^24);
- 8 million: studies M, M2 and M3.

## What goes in git

These go in:
- code;
- the simulators' case files under `sim3d/<simulator>/<effect>/`, so every run can be repeated;
- docs, small result tables and figures;
- the demo videos.

These never go in: the simulators themselves, runs, frames, volumes, renders or model weights.
