# More effects: fog, dust and cloud

Status: **in progress** (14 October 2026): the design of the test is fixed; fog and dust are frozen; cloud is training. Audience: dev, research, artists. Three atmosphere effects for a game
engine's library, after steam and magic ([EFFECTS.md](EFFECTS.md)): **fog** (a cold, heavy gas that sinks and rolls
along the ground), **dust** (an impact's burst that rolls out low along the ground and settles; it plays once) and
**cloud** (a persistent, slowly boiling mass with no source to be seen). All three come from the project's own 2D fluid
simulator with new sources, forces and drawing. They were trained as rollout effects with study D's recipe
([REPORT.md](REPORT.md) §6) and tested as study D, steam and magic were: on held-out settings and seeds, by tracking from
the true state and by endless statistics against real runs and the nearest training flipbook. A scene script shows them
together with study D's fire (§6).

## 1. Bottom line

(To come, after the test.)

## 2. The effects

All three are presets of the existing solver (`src/sim/sim.cpp`, in a block of their own: `add_atmos_sources`,
`add_atmos_forces`, `add_atmos_divergence`, `render_atmos`): the same grid, advection, vorticity confinement, curl noise
and pressure solve, with their own sources, forces and drawing. None of them has heat. Their material is the solver's
soot, its weight is what moves them up or down, and they give no light, so a scene neither lights nor hazes the air
around them. Each has the study's three controls (intensity, wind, turbulence), normalised to [0, 1]. Lengths are in
cells of a 128-cell frame and times in seconds.

| | fog | dust | cloud |
|---|---|---|---|
| what it is | cold, heavy gas released a little above the ground: it sinks, runs out along the ground both ways with rolling, curling fronts, and slowly thins | dust thrown up by an impact: a burst that rises a little, falls back and runs out along the ground as a low, wide, rolling bank, then settles and thins | a slowly boiling mass of cloud or thick smoke, wider than tall and flatter below, with no source to be seen; it never fades away |
| plays | loops | **once** (3 s), as the explosion | loops |
| source | a release 26 to 46 cells wide (intensity), 26 cells up: gas density 0.5 to 1.0, poured down at 4 to 12 cells per second, broken up by the flicker noise at a rate of 1.0 per second | one burst at the start: a mound 15 to 23 cells wide and 6 to 9 cells high on the ground, density 1.0 to 1.6, its edge and lumps from the flicker noise (rate 1.4, as the explosion's) | none to be seen: inside an extent 67 to 102 cells wide and 33 to 51 cells high (intensity), whose edge the flicker noise moves, the density is drawn towards a target (0.6 to 1.2 times a pattern of puffs from the flicker noise, rate 0.6) at 0.4 per second; outside it, what the flow carries out thins at 1.5 per second |
| forces | weight 20 (it sinks); wind up to 20 cells per second with drag 1.5 per second; velocity damping 0.4 per second; confinement 0.2 to 1.4 and curl noise 3 to 17 (turbulence) | weight 150; the burst throws the dust out: a divergence of 22 to 34 per second (intensity) where the dust is thick, fading with a time constant of 0.1 s (the explosion's expansion, tied to the dust instead of the heat); wind up to 50; damping 0.6; confinement 1.5 to 5.5 and curl noise 20 to 100 | weight -20 (its thick parts rise a little); wind up to 40 cells per second with drag 1.0; damping 1.3 (slow, smooth motion); confinement 0.2 to 1.0 and curl noise 3 to 19 |
| ground | a band of still air 3 cells high at the bottom of the frame, over a closed bottom border (no flow through the ground) | the same | none (open, as every other effect) |
| decay | 0.12 per second | 0.25 per second | 0.10 per second (outside the extent 1.5) |
| drawing | pale, soft mist: cover 1 - exp(-1.0 density), grey-white (0.86, 0.88, 0.90) and barely shaded (light towards the light exp(-0.08 depth)) | brown-grey, deeply self-shadowed: cover 1 - exp(-2.0 density), (0.62, 0.54, 0.43) in light to (0.19, 0.17, 0.14) in shade (exp(-0.45 depth)) | white (0.97) with grey-blue shade (0.42, 0.45, 0.52) (exp(-0.45 depth)), cover 1 - exp(-2.6 density) |
| controls | intensity (release rate and volume), wind (drift left or right), turbulence | intensity (impact energy and size), wind, turbulence | intensity (density and coverage), wind (drift), turbulence (boiling) |

They keep the study's control names (`sim::control_names`). The earlier effects' clips are unchanged to the byte: the
hash test (`Sim.StudyEffectsAreUnchangedByTheNewOnes`) now also holds steam's and magic's clips, made before this round,
and full-size clips of all five at two settings were compared byte for byte before and after. Tests of the new presets:
`Sim.AtmosphereEffectsHaveNoHeatAndGiveNoLight`, `Sim.FogSinksAndSpreadsAlongTheGround`,
`Sim.DustRisesSpreadsAndSettles`, `Sim.CloudStaysAliveAndDriftsWithTheWind`.

**How they were designed.** Each was looked at with `nvfx_sim --sheet` (and a scratch probe that printed the amount,
centroid and extent of the material) at several control settings until it read as what it is at 128 px. What changed
on the way:
- **The ground.** The solver's borders are open, so a heavy gas first poured out of the bottom of the frame. The ground
  is a band of still air (a force, before the pressure solve, as a learned stepper's network can supply) over a closed
  bottom border in the pressure solve. With the band alone fog spreads about as well, but part of dust's burst escapes
  downwards through the open border (in the probe its cloud rose to 0.26 of the frame in half a second instead of
  0.31). A learned stepper's pressure solve is open on all sides ([REPORT.md](REPORT.md) §6.2), so its network has to
  learn what the closed border does; §4 says how well.
- **Dust** took the most tries. A burst that expands (as the explosion's) rose as a mushroom of smoke with a hollow
  middle; a vortex pair blowing onto a bed of dust rolled thin curls along the ground, the kind of thin, regular
  structure that magic's lesson says the 32-cell state cannot carry. What reads as dust is a heavy mound thrown out by
  a short burst where the dust itself is (so it stays filled), which falls back and runs out along the ground as two
  rolling fronts.
- **Cloud** first grew tentacles (strong confinement and curl noise pulled arms out of the mass). It is now slow and
  well damped, its puffs come from its source (drawn towards a pattern the flicker noise breaks up, so a learned
  stepper sees the noise that shapes it), and the wind blows material through it: puffs form upwind, drift across and
  thin away downwind, while the mass stays where it is.

**How chaotic they are** (`nvfx_experiment d-chaos --effects fog,dust,cloud`, as [REPORT.md](REPORT.md) §6.1: active
PSNR against the undisturbed run, mean over 4 runs; [d_chaos.csv](../results/experiments/fx_atmos/d_chaos.csv)):

| effect | case | 1 frame | 0.5 s | 1 s | 2 s | 3 s | 8 s |
|---|---|---:|---:|---:|---:|---:|---:|
| fog | start nudged by 1e-3, same seed | 91.4 | 76.3 | 75.9 | 74.7 | 73.2 | 74.7 |
| fog | start nudged by 1e-1, same seed | 62.8 | 55.7 | 53.9 | 51.8 | 51.3 | 55.2 |
| fog | same start, other seed | 23.3 | 22.5 | 20.6 | 18.8 | 19.2 | 19.9 |
| fog | other start, same seed | 16.4 | 18.1 | 18.7 | 19.8 | 20.9 | 29.2 |
| dust | start nudged by 1e-3, same seed | 80.4 | 73.7 | 68.6 | 64.4 | 59.2 | - |
| dust | start nudged by 1e-1, same seed | 54.2 | 49.7 | 43.6 | 40.9 | 37.8 | - |
| dust | same start, other seed | 54.9 | 21.8 | 15.9 | 14.7 | 15.0 | - |
| dust | other start, same seed | 19.7 | 22.2 | 21.2 | 21.5 | 23.6 | - |
| cloud | start nudged by 1e-3, same seed | 88.6 | 76.6 | 78.1 | 79.6 | 80.2 | 90.4 |
| cloud | start nudged by 1e-1, same seed | 65.7 | 57.9 | 57.5 | 59.0 | 59.5 | 67.1 |
| cloud | same start, other seed | 43.8 | 23.6 | 20.6 | 17.1 | 17.0 | 17.2 |
| cloud | other start, same seed | 17.2 | 19.5 | 21.2 | 24.7 | 27.9 | 41.6 |

All three are calmer than fire and steam: an error in the start stays invisible for seconds (fog and cloud are still
above 70 dB after 8 s from a 1e-3 nudge, and a 10% nudge heals). The seed still takes over, within a second for dust
and one to two seconds for cloud; for fog more slowly and less (20 dB after 1 s against steam's 14), since its noise moves
only the release and a weak stir. Cloud is forced as magic is: two runs with the same seed from different start points
meet (42 dB after 8 s), so what a cloud looks like after a few seconds is decided by its seed.

## 3. Training

Study D's recipe, unchanged (`rollout::recipe_for`), as steam and magic were trained: runs from an empty grid with
random controls and seeds of their own (training salt 1), the stepper's three stages, the renderer on 200 frames of 100
runs, the detail constants chosen by matching frame statistics of endless runs against real runs at the controls of
four training runs with new seeds (validation; the test's settings and seeds are never used), and the start points.
Two of the recipe's branches apply as they do to study D's own effects, and the code now says so instead of naming the
explosion and smoke:
- **Dust plays once, so it takes the explosion's branch** (`!sim::effect_loops(e)`): 240 runs of 90 frames (3 s), 16
  start points at its first frame with 64-pixel fine fields (the burst's shape is the look), burn-in of up to 32
  frames. The looping recipe's 160 runs of 240 frames would have trained frames that the one-shot test and runtime
  never play (both stop at 90).
- **Fog and cloud keep 64-pixel fine fields with their 8 start points, as smoke does** (the recipe's reason for smoke,
  "it lingers: growing its detail would take seconds", holds for them more). Steam and magic, gone within a second or
  two, grow theirs at instance start as fire does.

Each was trained in study D's three saved steps (`d-train --stages12 1`, `d-tune` for stage 3, `d-finish`), one thread
each, `nice 10`, two at a time, on a machine shared with three other agents (load 6 to 13 on 4 cores while fog and
dust trained). The container restarted once, during cloud's stages 1 and 2 (at iteration 1250 of 4000); nothing of that
step was saved, so it ran again from the start.

| effect | runs x frames | simulation | stages 1-2 (loss, s) | stage 3 (loss, s) | renderer (training samples) | contrast, swirl, grow | detail score | start points | stored |
|---|---|---:|---|---|---:|---|---:|---:|---:|
| fog | 160 x 240 | 21.3 min, recorded in 353 s | 0.0169, 7028 | 0.0156, 1395 | 43.1 dB | 1, 0, 1 | 0.52 | 8 (64 px fine) | 146 KB |
| dust | 240 x 90 | 12.0 min, recorded in 141 s | 0.0256, 7107 | 0.0382, 1256 | 38.1 dB | 1, 0, 2 | 0.71 | 16 (64 px fine) | 274 KB |
| cloud | 160 x 240 | (training) | | | | | | | |

The last step (renderer, detail constants, start points) took 17 minutes for fog, 10 for dust and (to come) for
cloud. About 2.6 CPU-hours per effect in all, less than the 3.5 of steam and magic (the same iterations, on a faster or
quieter share of the machine). The loss columns are not comparable between effects (each is normalised by its own
state's scale), and dust's stage 3 adds the activity term, so its loss rises. The detail calibration chose no sub-grid
swirl for all three, as it did for magic; fog keeps existing structure as it is (grow 1).

Each model was frozen before its test: [fx_atmos_frozen.csv](../results/experiments/fx_atmos_frozen.csv) has their
SHA-256 (the files are under the data root's `fx_atmos/models/d`); fog's and dust's were recorded while cloud trained,
cloud's when it was done.

## 4. The test

Its design was committed before it ran, and each model was frozen before its own test
([fx_atmos_frozen.csv](../results/experiments/fx_atmos_frozen.csv) has their SHA-256). **The test ran in two parts on
this one design:** fog and dust were frozen and tested while cloud was still training (cloud's stages 1 and 2 had to run
again after the container restarted, and two cores were idle); cloud was frozen and tested in the same way afterwards.
Each test ran once, with the same tool, settings and seeds; the second run's rows were appended to the first's CSVs. It
is study D's protocol ([REPORT.md](REPORT.md) §6.4 to §6.6), as steam and magic were tested ([EFFECTS.md](EFFECTS.md) §4), with
`nvfx_experiment d-eval` unchanged (`--effects fog,dust`, then `--effects cloud`). Nothing the test uses was seen in training or in the choice
of the detail constants: training used runs of salt 1 (seeds 600000 upwards), and the detail constants were chosen on
validation runs at the controls of the first four training runs with seeds of their own (the training seed x 7 + 12345
for the real runs, x 31 + 777 for the model's).

- **Tracking (§4.1):** 8 held-out runs per effect (salt 2: seeds 700000 to 700007, with controls drawn from that salt),
  each started from its true state (coarse and fine; fog and cloud after 100 frames, dust after its first frame) with
  its own noise seed. Methods: the neural rollout through the runtime; the simulation on the same 32-cell grid with the
  same detail layer (`coarse_sim_detail`); the plain coarse simulation; holding the first frame. Active PSNR at 1 and 8
  frames and 1, 2 and 8 s (dust: to its last frame, 3 s), mean over runs, and neural minus `coarse_sim_detail` paired
  over runs.
- **Endless (§4.2):** the ten held-out settings of study B (its generator, seed 2026), 10 s each (dust: its 3 s). The
  reference is a real run at that setting with seed 900000 + setting (fog and cloud warmed up 5 s). Methods: a second
  real run with seed 910000 + setting (the floor); the neural rollout with seed 920000 + setting, played as it ships
  (shards from the start points); the nearest of the 45 training flipbooks of study B's grid, made for these effects
  as study B made its library (64 frames at 128 px, seed 1; looped, dust held on its last frame); the coarse simulation
  with and without the detail layer, seed 930000 + setting. Statistics against the reference: detail spectrum
  distance, motion ratio, coverage distance and mean-frame PSNR.
- **One minute (§4.3):** a 60 s neural run in shards (seed 940000 + setting) at the first two held-out settings, cut
  into 10 s windows, each scored against the 10 s reference (fog and cloud; dust plays once).
- **Intervals:** paired over runs or settings, 95% bootstrap intervals from 10,000 resamples
  (`metrics::paired_bootstrap`); an interval across zero is a tie.
- **No pass mark,** as for steam and magic. Each effect is reported against the coarse simulation with the same detail
  layer (does the learned stepper earn its place?), the nearest flipbook (does it beat a 45 MB library?) and the floor
  (how far from real is it?).

(Results to come.)

## 5. Memory and cost

(To come.)

## 6. The scene

(To come.)

## 7. What it means

(To come.)

## 8. Reproduce

```sh
build/nvfx_sim --effect fog --intensity 0.8 --wind 0.6 --sheet fog.png       # the simulator's presets alone
build/nvfx_sim --effect dust --intensity 0.6 --frames 90 --sheet dust.png
build/nvfx_sim --effect cloud --turbulence 0.8 --sheet cloud.png
R="--root $NEURALVFX_DATA/fx_atmos --results results/experiments/fx_atmos"
build/nvfx_experiment d-chaos  --effects fog,dust,cloud $R --threads 1
build/nvfx_experiment d-train  --effects fog --stages12 1 $R --threads 1
build/nvfx_experiment d-tune   --effects fog $R --threads 1
build/nvfx_experiment d-finish --effects fog $R --threads 1
#   (the same for dust and cloud; or d-train without --stages12 for all three steps at once)
build/nvfx_experiment d-eval   --effects fog,dust,cloud $R --threads 2 --figures DIR
build/nvfx_experiment d-report $R                                            # D_SUMMARY.md
build/nvfx_experiment d-timing --effects fire,smoke,fog,dust,cloud --root DIR2 --results results/experiments/fx_atmos
build/nvfx_scene_script --script examples/scenes/moor.nvfxs --models DIR3 --out moor.mp4 --keyframes DIR --sheet sheet.png
```

`DIR2/models/d` holds study D's `fire.nvfx` and `smoke.nvfx` and the three models; `DIR3` holds `fog.nvfx`,
`dust.nvfx`, `cloud.nvfx` and study D's v2 `fire.nvfx` (`ctest -R ScriptedMoorFrozen` reads it from
`NEURALFX_MODELS_ATMOS`, by default `$NEURALVFX_DATA/fx_atmos/models/scene`).
