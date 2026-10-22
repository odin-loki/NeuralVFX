# To do: every effect in 3D — a small demo here first, then big on a PC

**Status:** stage 1 is done (study N, `docs/STUDY_3D.md`, and its follow-up N2, §14); stage 2 is planned, not started.
Since 15 October 2026, here: round 9, studies O and P, float on AVX-512, the faster detail layer, the 2D weather
director, the FUSE sample, round 10's fire and elements and study Q (the weak spots of rounds 9 and 10) are done and
merged, and so is study R (a validated 3D smoke): its teacher is a plume by MTT's measures, its learned smoke failed its
test and ships as a labelled demo ("Here, until the PC"). Round 7 of the quiet re-timing measured all of their costs on
a quiet machine (REPORT §13.11). The effect manager can draw large effects at half size (REPORT §22): off by default
by its rule (magic saved too little), available to set.

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
- **Stage 2, on a PC, much later:** everything below. Accurate 3D simulations of every effect, however long they take,
  checked against measured physics. How the PC's GPU simulates, trains and plays the high-resolution effects (a
  64 x 96 x 64 state, 256 x 384 x 256 detail), each made in 5 to 9 hours, and which simulators teach them (our GPU
  solvers and Houdini, checked against measurements), is planned in [docs/PLAN_PC_GPU.md](docs/PLAN_PC_GPU.md).
  Large 3D models trained on them, as fine as the PC allows. Then 3D scenes, a 3D library and a full weather system.

First we demo, then we go big.

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
  it through the scene inputs. Weak: no sun or wet ground, clouds as puffs rather than a deck, one place in 2D.
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
  switch). **By its rule, committed first, it is off by default:** on 20 held-out settings at 256 px it stays within the
  null's margins of full size against the simulator on every measure for all seven effects measured (the crowd's fire,
  smoke, steam and explosion, the library's fire, explosion and magic), closer than full size on tracking at 1 s
  (+0.22 to +0.89 dB) and the detail spectrum, and costs 0.31 to 0.42 of full size for six of them, but **0.54 for
  magic** (the rule allowed 0.5; the quiet session's separate runs give 0.53 too). Against full size on the same seed it
  loses the finest filaments (24 to 37 dB active PSNR). With no budget a close-up scene's tick went from 17.1 to 15.2
  ms (0.886 [0.869, 0.901]); with a budget the planner runs more simulations of their own (13.2 to 17.4 at 8 ms) and
  fewer proxies; the crowd hardly changes (provisional timings: other agents' load ran throughout). Also fixed: a race
  in the manager's worker start that could hang a tick on several threads (main's code).

**Under way:**
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

**Next:**
- **Study Q's leftovers** (STUDY_Q.md §9): the avalanche's start points (the mean state with fine fields regrown from
  the moved coarse state), the geyser's tracking with its first frame kept, the haboob's and the ring's first frames
  by fine-tuning their renderers rather than training them again, heavy gas's start points, the mushroom's cap (a
  stepper that sees its whole life).
- **Half size, next** (REPORT §22.9): re-time its cost and the manager's A/B on a quiet machine (the jobs under the data
  root's `halfsize/jobs`; the rule's verdict stands on its own run); a default per effect (on for the effects that passed
  both conditions, off for magic) needs its own rule; measure 128 px pictures drawn at 64 and more of the library's
  models at 256 px; the 256 px pool (one slot per effect by default) limits what it switches in a crowd.
- **More 2D effects**, from the catalogue rows the 2D solver can carry: coloured smoke grenades (#18), contrails (#36),
  helicopter downwash (#37), blizzard and drifting snow (#54, #75), steam fog (#61), a learned pyroclastic flow (#110;
  the library's entry today is built from existing models).
- **The weather director, next:** a cloud deck rather than puffs, the sun and its shafts, wet ground and puddles, a storm
  that moves across the scene, trees bending by the Beaufort force.
- **Small 3D:**
  - **finer coarse 3D (study hres):** fire (N2 C4 lineage) and smoke (N2 smoke2 lineage); [docs/STUDY_3D_HRES.md](docs/STUDY_3D_HRES.md), `tools/study_3d/hres_fire.sh` and `hres_smoke.sh`; `nvfx_3d --study hres` (32 x 48 x 32, fine 2);
  - a second 3D effect, smoke: **the teacher done** (study R: `smoke_r`, measured against plume theory); the learned
    smoke failed its test by spreading too wide (STUDY_R.md §6.3). Next: a loss on the plume's spread, more teacher
    time, hres's finer state trained on `smoke_r` (its own seed block and rule), and the solver's momentum deficit
    (0.78; a stable less dissipative velocity step);
  - 3D fire and smoke coupled: a demo through the volume API (study R, `nvfx_3d video --kind coupled`); a coupled
    simulation (the boxes exchanging heat and momentum) is not done;
  - a job hook in the volume API, so an engine's job system can run a volume's step (FUSE.md §9).

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
- **"By eye against footage"** means no measured quantity exists that a picture could be checked against: those are
  compared with real footage by study J's statistics, if footage is supplied.
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
