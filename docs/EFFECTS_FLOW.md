# Flow effects: launch, puffs and ink

Status: **in progress** (14 October 2026). Audience: dev, research, artists. Three effects for a game engine's library,
after the atmosphere and energy rounds ([EFFECTS_ATMOSPHERE.md](EFFECTS_ATMOSPHERE.md),
[EFFECTS_ENERGY.md](EFFECTS_ENERGY.md)): **launch** (a rocket's or an engine's exhaust blown down onto the ground,
billowing out both ways), **puffs** (smoke leaving a chimney or an exhaust in distinct pulses) and **ink** (a plume with
almost no buoyancy that curls and spreads slowly, as ink, blood or silt does in water). All three come from the
project's own 2D fluid simulator with new sources, forces and drawing, and are trained and tested exactly as the last
six were: study D's recipe and protocol ([REPORT.md](REPORT.md) §6), on held-out settings and seeds.

## 1. Bottom line

(Written when the test has run.)

## 2. The effects

All three are presets of the existing solver (`src/sim/sim.cpp`, in a block of their own: `add_flow_sources`,
`add_flow_forces`, `flow_decay`, `render_flow`): the same grid, advection, vorticity confinement, curl noise and
pressure solve, with their own sources, forces, decay and drawing. Each has three controls normalised to [0, 1].
Lengths are in cells of a 128-cell frame and times in seconds.

The lessons of the last rounds shaped them. Magic's fast, regular spiral smeared on the 32-cell state; buoyant plumes,
jets, heavy gas and dust learned well. So these keep their flow at fire's speeds or below and their structures broad,
with their detail from turbulence rather than thin regular arms. And every force is a function of what a learned
stepper sees: the state where it acts, the position, the controls and the run's two noise fields (the curl noise's
stream function and the source flicker). Nothing follows a clock of its own; puffs' pulses are timed by the flicker
noise at the chimney's mouth (below).

| | launch | puffs | ink |
|---|---|---|---|
| what it is | an engine above the ground blows a hot, fast jet straight down; it hits the ground, runs out along it both ways and rolls up into grey-white billows that rise as they cool; brightest at the impact | warm smoke leaves a chimney's mouth at the bottom of the frame in distinct puffs, each kicked up and rolling into a ring; they rise, drift to the right on a light breeze and merge into a trail | dye released slowly a quarter of the way up the middle of the frame, in a direction that wanders; it curls and spreads into soft clouds and long tendrils and drifts with the current; no heat and no light |
| source | a nozzle in the middle, 56% up, 9 to 14 cells across (intensity), blowing straight down at 100 to 160 cells per second (intensity): heat 1.6 to 2.2 and a little soot, broken up by the flicker noise (rate 3.0 per second, as the jet's) | a mouth 36% from the left, 6 cells up, 13 to 20 cells across (intensity), open while the gate is: warm smoke (heat 0.5 to 0.8, soot 1.2 to 2.0) kicked up at 60 to 100 cells per second. **The gate** opens when the size of the flicker noise at the middle of the mouth, \|f\|, rises above a level that the rate control lowers (0.40 to 0.14; flicker rate 4 per second) | a round source of radius 6 to 10 cells (intensity), 62 cells from the left and 34 up: dye of density 1.0 to 1.8, pushed out at 12 to 24 cells per second in a direction 90 degrees plus 63 times the curl noise's stream function there (it wanders slowly), broken up by the flicker noise (rate 0.8 per second) |
| forces | buoyancy 20 (fire: 150), so the jet reaches the ground; its smoke is light (weight -8: it rises); wind up to 30 cells per second with drag 1.0 per second on the material; velocity damping 0.3 per second; confinement 0.4 to 2.0 and curl noise 8 to 38 (turbulence) | buoyancy 40, soot weight 0.5; a fixed breeze of 14 cells per second to the right, with drag 1.0 per second on the smoke; damping 0.4 per second; confinement 0.2 to 1.4 and curl noise 5 to 30 | no heat; the dye is a little lighter than the water (weight -3); **the current**: all the water is dragged (0.7 per second) towards a level flow of up to 22 cells per second to the left or the right; it is stirred everywhere by the curl noise (2 to 8, as well as 3 to 11 where the dye is); damping 0.6 per second; confinement 0.05 to 0.4 |
| ground | the atmosphere effects' ground: a band of still air 3 cells high over a closed bottom border | none (open, as fire's) | none |
| decay | heat 1.6 per second; the hot gas turns to smoke and steam as it cools (soot grows by 2.2 times the heat per second); smoke 0.15 per second | heat 0.6 per second, smoke 0.22 per second | dye 0.05 per second |
| drawing | emitted light by heat (the jet's ramp: dark red, orange, yellow, white, bluish white at its hottest) over grey-white clouds, lit from above and to the left and shaded by themselves | light grey smoke, lit from above and to the left and shaded by itself | dark blue-black dye, barely shaded, thin where it is thin |
| controls | intensity (thrust), wind (drift left or right), turbulence | intensity (puff size and strength), **rate** (pulses per second), turbulence | intensity (amount), **current** (drift left or right), turbulence |

The second controls of puffs and ink are named `rate` and `current` (`sim::control_names`, so the model files, the
runtime and scene scripts call them that; `nvfx_sim` takes `--rate` and `--current`); launch keeps the study's names.
All three loop. Every earlier effect's clips are unchanged to the byte: `Sim.StudyEffectsAreUnchangedByTheNewOnes` now
also hashes clips of jet, orb, vortex, fog, dust and cloud made before these three existed, and full-size clips (128
px, 64 frames) of all eleven earlier effects at two settings were compared byte for byte before and after. Tests of the
new presets: `Sim.FlowEffectsHaveTheirOwnControls`, `Sim.LaunchBlastsTheGroundAndBillowsOutBothWays`,
`Sim.PuffsComeInPulsesThatTheRateControlQuickens`, `Sim.InkIsSlowDarkDyeThatTheCurrentCarries`, and in the recipe
`Rollout.FlowEffectsKeepFineFieldsAsSmokeDoes`.

**Puffs' pulses.** The learned stepper has no clock: what it sees is the state, the position, the controls and the two
noise fields. So the pulses are timed by the flicker noise at the middle of the mouth, which the stepper sees as one of
its inputs (the mouth's middle is a coarse cell's centre, so it sees exactly that value), and the rate control sets the
level the noise must rise above. Pulses per second of the gate (16 seeds of 60 s each):

| rate | 0 | 0.25 | 0.5 | 0.75 | 1 |
|---|---:|---:|---:|---:|---:|
| pulses per second | 0.49 | 0.91 | 1.48 | 2.03 | 2.57 |
| gate open | 4% | 10% | 19% | 31% | 47% |
| gap between pulses, coefficient of variation | 0.90 | 0.83 | 0.76 | 0.63 | 0.56 |

The rate grows nearly in proportion to the control, but the pulses are **irregular**: the gaps between them vary by
56% to 90% of their mean, as a signal fire's or a labouring engine's do, not like a steam locomotive's steady beat. A
regular beat would need a clock that the stepper can see; the noise gives none (it is smooth value noise in time, with
random values at its lattice points), and an oscillator kept in the state (a heat that builds up and is let go) would
ask the 32-cell state and a smoothed learned step to keep a sharp threshold, the kind of structure that fails on it.

**How they were designed.** Each was looked at with `nvfx_sim --sheet` at several settings (low, middle and high
controls, and their extremes) and a scratch probe that printed the amount, centroid and extent of the material, its
speeds, and puffs' pulses. What changed on the way:
- **Launch** first drew thin orange swirls everywhere: the hot gas kept its heat as it spread and was stirred hard. It
  now cools faster (1.6 per second) and turns into smoke as it cools (2.2 times its heat), with less confinement and curl
  noise, so the light stays at the jet and the impact and the clouds are grey-white billows. With fire-like buoyancy
  the gas rose before it spread; at 20 the billows roll out along the ground first. The ground is the atmosphere
  effects' (a band of still air over a closed bottom border), which fog and dust learned.
- **Puffs** first came out of a narrow mouth as thin stalks. A wider mouth, a sharper gate and a kick of 60 to 100 cells
  per second roll each puff into a ring (a mushroom cap at 128 px). A one-sided gate (f above a level) gave 0.5 to 1.5
  pulses a second at best and merged them at high rates; the two-sided gate (\|f\| above a level) doubles the crossings,
  5 times from the lowest rate to the highest. A breeze on all the air bent the puffs into hooks; on the smoke only (as
  the other effects' wind) it leans the trail gently.
- **Ink** first released too little dye to read. A bigger, denser source fixed that; at high turbulence the curl noise
  and confinement tangled it into kinked, boxy tendrils (the value noise's lattice), so both were reduced: its curls stay
  broad and slow, as magic's lesson asks.

How fast they move, in the steady state at two settings (all controls 0.5 and all 1; cells of a 128-cell frame per
second where there is material, median and 99th percentile, measured by the same probe for fire and smoke; a coarse cell
is 4 of them, a frame 1/30 s):

| effect | median | 99th percentile | in coarse cells a frame (99th) |
|---|---:|---:|---:|
| fire | 50 to 71 | 99 to 139 | 0.82 to 1.16 |
| smoke | 37 to 52 | 106 to 156 | 0.88 to 1.30 |
| launch | 20 to 30 | 114 to 136 | 0.95 to 1.13 |
| puffs | 16 to 29 | 73 to 107 | 0.61 to 0.89 |
| ink | 7 to 15 | 18 to 27 | 0.15 to 0.23 |

Launch's jet is as fast as fire's flames; puffs are slower and ink is the slowest effect so far.

![Launch, puffs and ink, the simulator's presets: two frames a second apart at three settings (low, middle, high controls)](figures/flow_presets.png)

**How chaotic they are** (`nvfx_experiment d-chaos --effects launch,puffs,ink`, as [REPORT.md](REPORT.md) §6.1: active
PSNR against the undisturbed run, mean over 4 runs; [d_chaos.csv](../results/experiments/fx_flow/d_chaos.csv)):

| effect | case | 1 frame | 0.5 s | 1 s | 2 s | 8 s |
|---|---|---:|---:|---:|---:|---:|
| launch | start nudged by 1e-3, same seed | 71.1 | 64.6 | 62.9 | 60.1 | 50.1 |
| launch | start nudged by 1e-1, same seed | 46.1 | 34.9 | 32.1 | 28.5 | 23.3 |
| launch | same start, other seed | 37.8 | 17.6 | 13.0 | 10.4 | 10.3 |
| launch | other start, same seed | 10.4 | 10.2 | 10.2 | 10.4 | 13.5 |
| puffs | start nudged by 1e-3, same seed | 72.7 | 65.8 | 65.0 | 64.0 | 62.4 |
| puffs | start nudged by 1e-1, same seed | 51.4 | 39.7 | 37.7 | 36.4 | 34.1 |
| puffs | same start, other seed | 49.7 | 17.1 | 13.8 | 12.3 | 13.8 |
| puffs | other start, same seed | 9.5 | 10.1 | 10.9 | 13.2 | 21.7 |
| ink | start nudged by 1e-3, same seed | 83.8 | 73.3 | 74.6 | 74.1 | 79.1 |
| ink | start nudged by 1e-1, same seed | 60.1 | 50.4 | 49.7 | 50.3 | 57.9 |
| ink | same start, other seed | 31.0 | 21.4 | 17.8 | 15.1 | 15.6 |
| ink | other start, same seed | 14.0 | 14.0 | 14.3 | 14.8 | 22.0 |

- **Launch behaves as fire does, more so**: a small error in the start stays invisible for seconds (63 dB after 1 s, 50
  after 8 s), and the seed takes over within a second; two runs with other seeds are 10 dB apart after 2 s, the lowest
  of any effect so far (fire: 12.4), since its turbulent clouds fill most of the frame.
- **Puffs' seed decides when they come**: another seed is down to 14 dB within a second, as the pulses then come at
  other times; and runs from other starts with the same seed are pulled together (10 dB at first, 22 dB after 8 s), as
  the orb's and the cloud's are by their forcing.
- **Ink is the calmest effect so far**: an error of 1e-3 in the start is still invisible after 8 s (79 dB), and one of
  10% heals (50 dB after 1 s, 58 after 8). Its seed takes over more slowly than any other's (18 dB after 1 s, 15 after
  2), since its dye moves slowly and its noise only aims the release and stirs the water gently.

## 3. Training

(Written as each model is trained.)

## 4. The test

Written before any test was run. Each model is frozen (its SHA-256 in
[fx_flow_frozen.csv](../results/experiments/fx_flow_frozen.csv)) before its own test, and each test runs once, with
`nvfx_experiment d-eval` and `d-report` unchanged. The models train one at a time (the machine is shared), so the test
runs in parts on this one design: each effect is frozen and tested when its training ends (`d-eval --effects launch`,
then `puffs`, then `ink`), its rows added to the same CSVs. Every run of the test is independent and seeded by its
effect, run and setting, so the split changes no number. It is study D's protocol ([REPORT.md](REPORT.md) §6.4 to
§6.6), exactly as the last six effects were tested ([EFFECTS_ATMOSPHERE.md](EFFECTS_ATMOSPHERE.md) §4,
[EFFECTS_ENERGY.md](EFFECTS_ENERGY.md) §4):

- **Tracking from the true state:** 8 held-out runs per effect (salt 2: seeds 700000 to 700007, with controls drawn
  from that salt; no training or validation run used them), each started after 100 frames from its true state (coarse
  and fine fields) with its own noise seed; 240 frames. The neural rollout through the runtime against the simulation
  on the same 32-cell grid with the same detail layer (`coarse_sim_detail`), the plain coarse simulation, holding the
  first frame, and the neural dynamics drawn by the simulation's renderer. Active PSNR at 1 and 8 frames and at 1, 2
  and 8 s; the neural run minus `coarse_sim_detail`, paired over the 8 runs.
- **Endless runs at settings never seen:** the ten held-out settings of study B (`b_test_settings`, its generator with
  seed 2026: none on study B's training grid), 10 s each after a 5 s warm-up of the real runs, with seeds never used for
  these effects (900000 + k and 910000 + k for the two real runs, 920000 + k for the neural run, 930000 + k for the
  coarse simulation; training used 600000 to 600159, the held-out tracking runs 700000 and up, and the validation of the
  detail constants seeds derived from the training runs'). Frame statistics against a real run: detail spectrum
  distance, motion ratio (as the error \|ln ratio\|), coverage distance and mean-frame PSNR. Compared with a second real
  run with another seed (the floor), the nearest of the 45 training flipbooks of study B's grid (made for these effects
  as study B made its library: 64 frames at 128 px, seed 1; looped), and the coarse simulation with the detail layer and
  without.
- **One minute:** a 60 s neural run in 6 s shards at the first two held-out settings, cut into 10 s windows, each
  against the real 10 s run.
- **Precision:** the runtime plays the effects as they ship, at its default precision. Since study K
  ([REPORT.md](REPORT.md) §13) that is int8 (the stepper's second convolution in 8 bits, on the AVX-512 build where the
  CPU has VNNI, as this machine's has). `d-eval` sets no precision, so the test measures int8.
- **Statistics:** every difference paired (over runs or settings), with a 95% bootstrap interval of 10,000 resamples
  (`metrics::paired_bootstrap`); an interval across zero is a tie. Every number is reported, ties as ties.

What would count as working, decided now (the energy round's rule): an effect **works as steam did** if it tracks its
runs better than `coarse_sim_detail` at 1 s (interval above zero) and, endless, its average picture is closer than the
nearest flipbook's (mean-frame PSNR interval above zero) and its motion error is no worse than the flipbook's (interval
not above zero). An effect that ties or loses to the coarse simulation at tracking at 1 s, or whose average picture or
motion loses to the flipbook, is reported as **weak**, as magic was, with what it does well and what it does badly.

## 5. Memory and cost

(Measured after the test.)

## 6. The scene

(Written when the effects are trained.)

## 7. What it means

(Written last.)

## 8. Reproduce

```sh
build/nvfx_sim --effect launch --intensity 0.6 --sheet launch.png      # the simulator's presets alone
build/nvfx_sim --effect puffs --rate 0.5 --bg black --sheet puffs.png
build/nvfx_sim --effect ink --current 0.8 --sheet ink.png
R="--root $NEURALVFX_DATA/fx_flow --results results/experiments/fx_flow --threads 1"
build/nvfx_experiment d-chaos  --effects launch,puffs,ink $R
build/nvfx_experiment d-train  --effects launch --stages12 1 $R
build/nvfx_experiment d-tune   --effects launch $R
build/nvfx_experiment d-finish --effects launch $R
#   (the same for puffs and ink)
build/nvfx_experiment d-eval   --effects launch $R --figures DIR     # the test, an effect at a time
build/nvfx_experiment d-report $R                                    # D_SUMMARY.md
```
