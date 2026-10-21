# Elements: a geyser, a haboob and sea spray

Status: **the presets, the training plan and the test's design and rule (§4) are written and committed before any model
is trained** (21 October 2026); the results will follow them here. Audience: dev, research, artists.

Three effects of the elements, after the earth and air rounds ([EFFECTS_EARTH.md](EFFECTS_EARTH.md),
[EFFECTS_AIR.md](EFFECTS_AIR.md)), from the catalogue in [TODO.md](../TODO.md): a **geyser** (#88: a periodic eruption
of a column of water and steam; it erupts, holds, collapses and steams, and is quiet between eruptions), a **haboob**
(#70 and #71: a storm's outflow running along the ground as a gravity current, a tall rolling wall of dust with dimmer
dust behind it; at its lowest a sandstorm streaming low; it plays once, for 5 s) and **sea spray** (#77: spray and
spume torn off breaking wave crests by the wind, drifting downwind and falling back into the sea, a fine spray drifting
on). They come from the project's own 2D fluid simulator with new sources, forces and drawing, and are trained and
tested as the last twenty-seven effects were: study D's recipe and protocol ([REPORT.md](REPORT.md) §6), on held-out
settings and seeds of this round's own (27,000,000 to 27,999,999).

## 1. Bottom line

To be written from the tests (§4), once each has run.

## 2. The effects

All three are presets of the existing solver (`src/sim/sim.cpp`, in a block of their own: `add_elements_sources`,
`add_elements_forces`, `add_elements_divergence`, `elements_decay`, `render_elements`): the same grid, advection,
vorticity confinement, curl noise and pressure solve, with their own sources, forces, decay and drawing. Each has three
controls in [0, 1]:

| | geyser | haboob | seaspray |
|---|---|---|---|
| controls | intensity (the column's height: 10 to 18 m), **wind** (0 from the right, 0.5 still air, 1 from the left: it carries the steam), turbulence | intensity (the outflow's density: a stronger storm runs faster), **height** (the outflow's depth: a low sandstorm at 0, a tall wall at 1), turbulence | intensity (the sea's size: higher crests, more breaking), **gale** (the wind's strength, a near gale to a storm, blowing to the right; a game mirrors the module for the left), turbulence |
| plays | loops | **once**, 150 frames (5 s) | loops |

The haboob's second control is named `height` and sea spray's `gale` (`sim::control_names`, so the model files, the
runtime and scene scripts call them that; `nvfx_sim` takes `--height` and `--gale`); the geyser keeps `wind`. Every
force is a function of what a learned stepper sees: the state where it acts, the position, the controls and the run's
two noise fields (the curl noise's stream function ψ and the source flicker f). Nothing follows a clock of its own. The
haboob, which plays once, is made at its first step (as the explosion's burst is) and does not otherwise follow the time
since its start.

**The geyser's rhythm.** A geyser is a relaxation oscillator: its conduit is heated from below until water deep in it
boils, the boiling empties the conduit in an eruption, and it refills and heats again. A learned stepper has no clock of
its own, so the conduit's state is a noise field it sees: the source flicker at the coarse cell beside the vent (cells
62.5, 6.5 of a 128-cell frame, a coarse cell's centre, where the stepper's flicker input is exactly that value), whose
time axis is slow (0.3 per second), so it rises and falls over seconds. Its *charge* is (f + 0.14) / 0.32: below 0 the
vent only simmers; from 0 to 1 it steams harder and surges; from 1 it erupts, with a strength smoothstep(1, 1.35,
charge), holds while the charge stays high and collapses as it falls; the steam it threw up drifts off. Over 40 seeds and
two minutes each the conduit is past its eruption level 23% of the time, an eruption lasts a median 1.9 s (the 10th and
90th percentiles 0.5 and 5.3 s, some of them surges within one eruption) and the gaps between them a median 5.1 s (0.7 to
15 s): irregular, as real geysers are (Old Faithful's intervals run from about an hour to two). A geyser that should erupt
on a game's cue would be a one-shot started again, as the smoke rings are ([EFFECTS_AIR.md](EFFECTS_AIR.md) §2); this
one keeps its own irregular time.

Lengths are in cells of a 128-cell frame and times in seconds. The geyser's frame is about 25 m across, so gravity (9.8
m/s²) is 50 cells/s²; sea spray's about 12 m.

| | geyser | haboob | seaspray |
|---|---|---|---|
| what it is | a vent in a low sinter mound in the middle of the bottom of the frame (its top 8 cells up): between eruptions a wisp of steam; as the conduit's charge rises it steams and surges; then a white column of water shoots up, holds, and collapses back around the vent while its steam billows up and drifts off on the wind | a storm's cold outflow let go at the left side of the frame: it runs right along the ground as a gravity current, its head rolling up into billows, the dust it lifts from the ground making it a dark wall, and dimmer dust behind it settling out; at 5 s the front has crossed the frame at most settings | wave crests break along the sea's surface (12 cells up): white caps rise where they break and the wind tears spray off their tops; the drops fly downwind and fall back into the sea, a fine spray drifts on and evaporates |
| source | the vent, 8 to 12 cells wide (intensity), straddling the ground: steam (heat (0.45 to 0.7) × the steaming strength, at least 0.1) always, and in an eruption water ((1.1 to 1.4) × its strength) thrown up at V = sqrt(2 g H) for a column H = (50 to 90) × (0.15 + 0.85 × the eruption's strength) cells high, broken into surges by the flicker noise; the vent discharges this volume (a divergence source under it of V over its depth of 4 cells, as the explosion's hot gas expands) | at its first step a lock of cold, dusty air 26 cells wide and 30 to 64 cells deep (height), its density 0.6 to 1.2 (intensity) and lumpy with the flicker noise, half of that in dust; after it the storm goes on feeding it at the left edge (the last 10 cells): drawn (4 per second) towards that density over that depth, moving forward at 0.8 sqrt(g' h) (a constant-flux current) | where the flicker along the surface (rate 0.9 per second) is above a level set by the share of the surface where crests break, a crest rises 4 to 10 cells (intensity) where it breaks hardest, with a white cap 3 cells deep; at its top, drops (soot, 0.9) and a fine spray (heat, 0.6 to 1.1, gale) are thrown up at (25 to 55) × the crest's strength and forward at the wind's speed there; how many crests break follows Monahan and O'Muircheartaigh's (1980) whitecap cover, 3.84×10⁻⁶ U^3.41 for a wind U of 12 to 25 m/s (gale), times 1.2 to 2.4 (intensity): from 2% of the surface to 54% |
| forces | **the water flies as drops**, not as a jet of air: an extra advection moves it at its drop velocity less the air's, which is a function of the position, the controls and the eruption's strength: in the column's core (a cone of about 7 degrees, below the height the eruption throws water to, rounded at its crown) it rises at sqrt(V² - 2 g z) and fans out, more as it slows; outside and above it it falls at 40 cells per second (8 m/s), moving out from the column (8 to 22, turbulence) and high up with the wind; it drags the air in the column with it (2 per second towards 0.3 of its velocity); buoyancy 90 on the steam; the wind (drag 1.5 per second towards 40 (2 wind - 1), weaker near the ground) carries the steam; bare air loses its momentum (2 per second); the ground is still but for the vent; confinement 0.2 to 1.2, curl noise 4 to 26, damping 0.6 | weight 20 on the cold air (g' = 20 × its density, cells/s²), none on the dust: it is carried by the cold air; the rough ground holds back the air within 2 cells of it (3 per second), so the front's nose is raised; the ground under it; no damping; confinement 0.3 to 1.3, curl noise 8 to 40 | the wind over the sea, everywhere: all the air drawn (1.2 per second) towards a logarithmic profile, (30 to 90 at the top of the frame, gale) × ln(1 + z / 1.5) / ln(1 + 116 / 1.5) at a height z over the surface; buoyancy 6 on the fine spray; the surface stops what falls onto it; confinement 0.1 to 0.6, curl noise 8 to 38 |
| decay | water that comes down onto the ground away from the vent runs off (12 per second) and gives half of itself as steam; hot water flashes to steam as it flies (0.25 per second); steam 0.55 per second | **saltation**: where the air within 6 cells of the ground runs faster than half of sqrt(g' h), it lifts dust at a rate that grows as the cube of its speed (Bagnold 1941), up to twice the outflow's density; the dust settles out at 0.25 per second | the drops fall through the air at their own speed, 20 cells per second (2 m/s; an upwind step), and what falls below the surface is gone; the fine spray evaporates at 0.45 per second; clear air comes in on the upwind side |
| drawing | white, frothy water, grey-blue in its own shade (little: it is thick and aerated); soft pale steam | ochre dust, deeply shaded by itself, its opacity from the lifted dust and (less) the cold air | white drops and caps, grey-blue in their shade; the fine spray a pale haze |

All three stand on the ground or the sea: the pressure solve's closed bottom border, as fog's, dust's and the earth
effects'. Where material leaves the frame, the outermost cells take their inner neighbours' values (as the earth effects').
The geyser and sea spray warm up for 150 frames before a clip is recorded, as fog and the water do. None of them draws
its ground or sea: the scene's or the game's does.

**The haboob's front speed against gravity-current theory.** A gravity current's front runs at U = Fr sqrt(g' h), with
g' its reduced gravity and h its depth. Benjamin (1968) found that for an ideal fluid, when the current is much shallower
than the fluid around it, Fr = √2, and for an energy-conserving current filling half the depth, Fr = 1/√2 (on the
current's depth); Huppert and Simpson (1980) measured Fr = 1.19 for currents in deep surroundings in the laboratory,
the difference from √2 being the unsteady, three-dimensional motion at the front (as reported by Hogg and Woods 2001,
J. Fluid Mech. 449, 201-224). The simulator's frame is open above (deep surroundings). Measured with a scratch probe on
36 runs (nine settings of intensity and height from 0.2 to 0.8, turbulence 0.5, four seeds each, 128 cells), the front's
speed from 1.5 to 3 s (or until it is 115 cells along) and the current's head over that time (its height, the highest
cell with cold air denser than 0.25 within 25 cells behind the front, and its mean density there):

| intensity, height | U (cells/s) | sqrt(g' h₀) of the feed | U / sqrt(g' h₀) | head height h_f (cells) | head density | Fr = U / sqrt(g' h_f) |
|---|---:|---:|---:|---:|---:|---:|
| 0.2, 0.2 | 20.0 | 23.0 | 0.87 | 30.3 | 0.58 | 1.08 |
| 0.2, 0.5 | 22.5 | 26.0 | 0.86 | 38.9 | 0.57 | 1.08 |
| 0.2, 0.8 | 24.8 | 28.7 | 0.87 | 49.3 | 0.57 | 1.07 |
| 0.5, 0.2 | 22.8 | 25.7 | 0.89 | 29.3 | 0.71 | 1.14 |
| 0.5, 0.5 | 25.8 | 29.1 | 0.89 | 36.4 | 0.70 | 1.17 |
| 0.5, 0.8 | 27.7 | 32.1 | 0.86 | 44.4 | 0.69 | 1.14 |
| 0.8, 0.2 | 25.3 | 28.2 | 0.90 | 28.6 | 0.83 | 1.19 |
| 0.8, 0.5 | 28.2 | 31.9 | 0.88 | 34.2 | 0.82 | 1.21 |
| 0.8, 0.8 | 31.0 | 35.1 | 0.88 | 40.5 | 0.82 | 1.23 |

**The simulator's current runs as gravity-current theory says.** Its front's speed is a constant 0.86 to 0.90 of
sqrt(g' h) of what feeds it at every setting (so it scales as the theory's square root of the reduced gravity and the
depth), and on its head, Fr = 1.15 ± 0.23 over the 36 runs (range 0.74 to 1.51): Huppert and Simpson's laboratory value,
1.19, and under Benjamin's ideal √2. `Sim.HaboobRunsAlongTheGroundAsAGravityCurrentAtItsSpeedAndLiftsDust` checks the
first measure (at 64 cells, four seeds: 0.85). For scale: a haboob's cold pool is a few kelvin colder than the air it runs
into (g' = g Δθ/θ, about 0.1 to 0.3 m/s²) and 1 to 2 km deep, so sqrt(g' h) is 10 to 25 m/s, and at Fr about 1 its front
runs at that speed, as haboobs' do. The tile's frame is shown faster than real time; the scene sets the pace.

Every earlier effect's clips are unchanged to the byte: `Sim.StudyEffectsAreUnchangedByTheNewOnes` hashes clips of all
27 earlier effects, unchanged, and full-size clips (128 px; 64 frames, the one-shots their own length) of all 27 at two
settings were compared byte for byte before and after (54 SHA-256, identical; the data root's
`fx_elements/jobs/state/baseline_clips.sha256` and `after_clips.sha256`). Tests of the new presets:
`Sim.ElementsEffectsHaveTheirOwnControlsAndOnlyTheHaboobPlaysOnce`, `Sim.GeyserEruptsHoldsCollapsesIntoSteamAndIsQuietBetween`,
`Sim.HaboobRunsAlongTheGroundAsAGravityCurrentAtItsSpeedAndLiftsDust`, `Sim.SeaSprayIsTornOffTheCrestsDriftsDownwindAndFallsBack`,
and in the recipe `Rollout.ElementsEffectsTakeTheirRecipes`.

**How they were designed.** Each was looked at with sheets of several settings and a scratch probe that printed the
amount, extent and speed of the material over time (and the haboob's front and head). What changed on the way:
- **Geyser.** A jet of water blown up from a vent in the ground hardly rose: the closed ground let no air into the vent,
  so the pressure solve took the jet's speed away; the vent is now a volume source (water and steam come out of the
  conduit). Then the column rose as a single-phase jet does, pushing a mushroom cap ahead of it that spread into a T,
  and the water lay on the ground as a sheet. Water moved as drops (its own velocity through the air, by position, the
  eruption's strength and the controls) gives a column that rises at the drops' speed, rounds off at its crown and falls
  back around itself; water that lands runs off fast and steams.
- **Haboob.** It was first designed as a loop in the frame of the front, as in a moving-floor flume (Britter and Simpson
  1978): the air ahead coming on at the front's speed and the ground moving under it. The head was not stable in that
  frame: within 15% either way of the right speed it was pushed back into its feed or ran out of the frame, and nothing
  in a function of the state, position and noise holds it. In the ground's frame, let go from a lock and fed from
  behind, it runs as a gravity current should (above), and plays once. Its first current was low and its dust uniform; a
  deeper outflow and the dust lifted by the gust behind the front (Bagnold's cube law) give a taller head that is darker
  than the dust behind it.
- **Sea spray.** The first crests were blocks of white water with spray that stayed low; spray thrown off the crests'
  tops at the wind's speed there and lifted harder, drops falling through the air at their own speed (not pushing the
  air down, which stirs it), and thin white caps instead of whole crests read as spray torn off the sea.

![Geyser, haboob and sea spray, the simulator's presets: two frames at each of three settings (low, middle and high controls); the geyser at its fullest frame and 1.5 s later (at the lowest setting this seed's conduit stays quiet for the 8 s), the haboob at 1.5 and 3.5 s, sea spray 1 and 2.5 s into its loop](figures/elements_presets.png)

How fast they move, at two settings (all controls 0.5 and all 1; cells of a 128-cell frame per second where there is
material, median and 99th percentile; the looping effects over frames 150 to 240, the haboob over its 150; a coarse cell
is 4 cells, a frame 1/30 s; the same probe for fire):

| effect | median | 99th percentile | in coarse cells a frame (99th) |
|---|---:|---:|---:|
| fire | 55 to 75 | 97 to 126 | 0.80 to 1.05 |
| geyser (the air) | 12 to 13 | 71 to 77 | 0.59 to 0.64 |
| geyser (its drops, at most) | - | 71 to 95 | 0.59 to 0.79 |
| haboob | 23 to 28 | 61 to 78 | 0.51 to 0.65 |
| seaspray | 28 to 52 | 49 to 85 | 0.41 to 0.71 |

All three move slower than fire's fastest flow, and none has a fast regular rotation (study L's lesson: magic's arms at
1.3 to 1.9 coarse cells a frame smeared on the 32-cell state).

## 3. Training

Study D's recipe (`rollout::recipe_for`), in its three saved steps (`d-train --stages12 1`, `d-tune`, `d-finish`), one
effect at a time per thread, one thread each, `nice 10`, each from a frozen copy of the tool built from the committed
presets (`DATA/fx_elements/bin`, its SHA-256 and commit beside it), by a resumable queue with a marker for every step
(`DATA/fx_elements/jobs`, logs under `DATA/fx_elements/logs`). Its branches decide the rest:
- **the geyser and sea spray** linger (the geyser's steam drifts for seconds after an eruption, sea spray's fine spray
  across the frame), so they keep 64-pixel fine fields with their 8 start points, as smoke does (one new line in the
  recipe); otherwise fire's settings: 160 runs of 240 frames with random controls and the recipe's seeds;
- **the haboob** plays once, so it takes the one-shot branch (`!sim::effect_loops(e)`: 240 runs of its 150 frames, 16
  start points at its first frame with 64-pixel fine fields, burn-in of up to 32 frames).

**Decided before any training: the haboob gets study M3's mean state** ([REPORT.md](REPORT.md) §14.9), as the smoke
ring did. Its start points are its first states, and its first state is the lock it is let go from, whose depth and
density are its controls: a start point stored at another setting would play the wrong outflow (the explosion's and
dust's fault, §14.1). The mean state moves each start point to the requested controls by a quadratic fit of the
training runs' first coarse states (`nvfx_ordered --study m3 fitmean --block 2` on its own salt-1 training runs, then
`setmean`); it is added after `d-finish` and before the model is frozen, and the file that is frozen and tested is the
one with the mean state. Nothing else from the later studies is used, for the reasons round 9 gave
([EFFECTS_AIR.md](EFFECTS_AIR.md) §3): study L's MacCormack stepper is for flows faster than these, study M's
ensemble-mean fine-tune changed nothing measurable on steam and the jet, and study O's two detail passes are for the
falling water's bands, which these do not have.

## 4. The test

**Written before any model was trained, and committed with the presets.** Each model is frozen by its SHA-256
(`results/experiments/fx_elements_frozen.csv`) before its own test, and each test runs **once**, with `nvfx_experiment
d-eval` and `d-report` unchanged: study D's protocol ([REPORT.md](REPORT.md) §6.4 to §6.6), exactly as round 9 tested
its effects ([EFFECTS_EARTH.md](EFFECTS_EARTH.md) §4, [EFFECTS_AIR.md](EFFECTS_AIR.md) §4). The models may train
side by side on the shared machine, so the test runs in parts on this one design (`d-eval --effects geyser`, then the
others as each is frozen), each into a folder of its own under the data root (`DATA/fx_elements/test/<effect>`), the
rows then merged into one set of CSVs under `results/experiments/fx_elements`. Every run of the test is independent and
seeded by its effect, run and setting, so the split changes no number.

- **Seeds: this round's own** (`--seed-base 27000000`; this round chooses no seed below 27,000,000 or above 27,999,999):
  the held-out tracking runs 27,000,000 + run; the real runs at the held-out settings 27,100,000 + setting and
  27,110,000 + setting; the neural run 27,120,000 + setting; the coarse simulation 27,130,000 + setting; the one-minute
  runs 27,140,000 + setting; `d-chaos` 27,150,000 + run (and 27,150,999 + run). What the protocol fixes is kept, as every
  round kept it: the training runs' seeds (the recipe's, 600,000 + run), the detail calibration's (derived from the
  training runs') and study B's flipbooks' (seed 1, a baseline's construction).
- **Precision:** the runtime plays the effects as they ship, at its default precision, int8 since study K
  ([REPORT.md](REPORT.md) §13), on the AVX-512 build with VNNI, which this machine has; `d-eval` sets none, so the
  `neural` rows are int8, and `neural_dynamics_true_renderer` is the float reference implementation.
- **Tracking from the true state:** 8 held-out runs per effect (salt 2's settings, which no training or validation run
  used), each started from its true state (coarse and fine fields; the geyser and sea spray after 100 frames, the haboob
  after its first frame) with its own noise seed: 240 frames (the haboob: to its last frame, 149). The neural rollout
  through the runtime against the simulation on the same 32-cell grid with the same detail layer
  (`coarse_sim_detail`), the plain coarse simulation, holding the first frame, and the neural dynamics drawn by the
  simulation's renderer. Active PSNR at 1 and 8 frames and at 1, 2 and 8 s (the haboob: 1 and 8 frames, 1, 2 and 4 s);
  the neural run minus `coarse_sim_detail`, paired over the 8 runs.
- **Endless runs at settings never seen:** the ten held-out settings of study B (`b_test_settings`: none on study B's
  training grid), 10 s each after a 5 s warm-up of the real runs (the haboob: its 5 s from its first frame). Frame
  statistics against a real run: detail spectrum distance, motion ratio (as the error |ln ratio|), coverage distance and
  mean-frame PSNR. Compared with a second real run with another seed (the floor), the nearest of the 45 training
  flipbooks of study B's grid (made for these effects as study B made its library: 128 px, seed 1; 64 frames, looped;
  **the haboob's are its whole 150 frames**, as the avalanche's were, since 64 frames held on the last would leave more
  than half of it still), and the coarse simulation with the detail layer and without. A 64-frame flipbook of the geyser
  holds 2.1 s, less than one of its cycles: that is the baseline as study B made it, and it is reported as it falls.
- **One minute:** a 60 s neural run in shards at the first two held-out settings, cut into 10 s windows, each against
  the real 10 s run (the geyser and sea spray; the haboob plays once).
- **Statistics:** every difference paired (over runs or settings), with a 95% bootstrap interval of 10,000 resamples
  (`metrics::paired_bootstrap`); an interval across zero is a tie, reported as a tie. Every number is reported.

**The rule, decided now, as rounds 7 to 9 decided theirs.** An effect
- **works** if it tracks its runs better than `coarse_sim_detail` at 1 s (interval above zero) and, played endlessly,
  its average picture is closer than the nearest flipbook's (mean-frame PSNR interval above zero) and its motion error is
  no worse than the flipbook's (interval not above zero);
- is **weak** if it ties or loses to `coarse_sim_detail` at tracking at 1 s, or if, played endlessly, its average picture
  or its motion error loses to the flipbook's (interval on the worse side);
- otherwise **falls short**: it neither works nor is weak (as launch, ink and heavy gas did: tracking better, the average
  picture tied with the flipbook's).

Detail (spectrum distance) and coverage are reported but do not decide.

**Answered beside the rule, not by it** (scratch measurements on the test's own runs, decided now):
- *Does the learned haboob run at the simulator's speed?* At the test's first two held-out settings, the front's position
  over the 5 s (the rightmost column holding more than 3 cells' worth of cold air, as in §2, from the real run's and the
  neural run's coarse soot) and the speed from 1.5 to 3 s, real against learned.
- *Does the learned geyser erupt when its conduit says, and stay quiet between?* Over the one-minute runs at the first
  two held-out settings, the share of frames with a water column (alpha above a half over a fifth of the frame's height
  above the vent) while the conduit's charge is above 1.35, and while it is below 0, in the real run and the learned one.

## 8. Reproduce

```sh
build/nvfx_sim --effect geyser --intensity 0.7 --wind 0.6 --frames 240 --bg black --sheet geyser.png   # the presets alone
build/nvfx_sim --effect haboob --intensity 0.6 --height 0.8 --frames 150 --sheet haboob.png
build/nvfx_sim --effect seaspray --intensity 0.6 --gale 0.8 --frames 120 --bg black --sheet spray.png
D=$NEURALVFX_DATA/fx_elements
nohup setsid $D/jobs/pipeline.sh geyser &   # (and haboob, seaspray) study D's three steps, the haboob's mean state,
                                            # d-chaos, freeze, the test once (--seed-base 27000000); resumable
#   by hand, per effect E:
build/nvfx_experiment d-train --stages12 1 --effects E --root $D --results $D/rows/E --threads 1
build/nvfx_experiment d-tune   --effects E --root $D --results $D/rows/E --threads 1
build/nvfx_experiment d-finish --effects E --root $D --results $D/rows/E --threads 1
build/nvfx_ordered fitmean --effect haboob --study m3 --root $D --block 2                # the haboob's mean state
build/nvfx_ordered setmean --model haboob.finish.nvfx --meanstate $D/m3/meanstate/haboob_b2.txt --out haboob.nvfx
build/nvfx_experiment d-chaos --effects E --root $D --results $D/chaos/E --threads 1 --seed-base 27000000
build/nvfx_experiment d-eval  --effects E --root $D --results $D/test/E --figures FIGS --threads 1 --seed-base 27000000
$D/jobs/merge_rows.sh                                                                     # the rows into results/
build/nvfx_experiment d-report --root $D --results results/experiments/fx_elements      # D_SUMMARY.md
$D/jobs/timing.sh 4                                                                       # d-timing, four rounds
build/nvfx_scene_script --script examples/scenes/coast.nvfxs --models $D/models/scene --out coast.mp4 --keyframes DIR
build/nvfx_gallery --library library --models $NEURALVFX_DATA/library/models --only erupting_geyser,haboob,sea_spray --sheet sheet.png
```

The data root's `fx_elements/jobs` holds the queue (`train.sh`, `pipeline.sh`, which refuses to test a model whose
SHA-256 is not the frozen one and refuses to test twice), the frozen tool's SHA-256 and commit (`binaries.sha256`,
`binaries.commit`), and the clips' hashes before and after the presets (`state/baseline_clips.sha256`,
`state/after_clips.sha256`); the scratch probes this document quotes are in `fx_elements/dev` (outside git).
