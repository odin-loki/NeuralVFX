# Fire effects: wildfire, coals and a mushroom cloud

Status: **in progress** (14 October 2026). Audience: dev, research, artists. Three effects a game would want next to
fire, smoke and the explosion: a **wildfire** (a wide front of flames along the ground, as in a burning field, a wall
of fire or a grass fire: the whole width burns, where study D's fire makes one flame per tile), **coals** (a bed of
glowing embers on the ground, as in a dying campfire, a forge or cracked lava ground) and a **mushroom** cloud (a large
explosion's fireball that rises on a stem and cools into a rolling cap of smoke; it plays once, for 5 s). They come
from the project's own 2D fluid simulator with new sources, forces and drawing, and are trained and tested exactly as
the last six were ([EFFECTS_ENERGY.md](EFFECTS_ENERGY.md), [EFFECTS_ATMOSPHERE.md](EFFECTS_ATMOSPHERE.md)): study D's
recipe and protocol ([REPORT.md](REPORT.md) §6), on held-out settings and seeds.

## 1. Bottom line

To come, after the test.

## 2. The effects

All three are presets of the existing solver (`src/sim/sim.cpp`, in a block of their own: `add_blaze_sources`,
`add_blaze_forces`, `add_blaze_divergence`, `blaze_decay`, `render_blaze`): the same grid, advection, vorticity
confinement, curl noise and pressure solve, with their own sources, forces, decay and drawing. Each has the study's
three controls (intensity, wind, turbulence; `sim::control_names`), normalised to [0, 1]. Lengths are in cells of a
128-cell frame and times in seconds.

The lessons of the earlier rounds shaped them. Magic's fast, regular spiral smeared on the 32-cell state, so these keep
their flow at fire's speeds or below and their structures broad: the wildfire's tongues are 10 to 20 cells across, the
coal bed is 13 to 23 cells high (four to six coarse cells), and the mushroom's cap rolls over in two broad rolls of 20
to 30 cells, not a thin ring. And every force is a function of what a learned stepper sees: the state where it acts,
the position, the controls and the run's two noise fields (the curl noise's stream function ψ and the source flicker);
the mushroom, which plays once, also follows the time since its start, but only through the two decays a learned
one-shot sees as inputs (exp(-t / 0.15 s) and exp(-t / 1 s)).

| | wildfire | coals | mushroom |
|---|---|---|---|
| what it is | a front of flames along the bottom of the frame, 84% of its width: the whole width burns, in tongues that merge and part; smoke rolls off the top; the wind leans it | a low, wide bed of glowing coals: dark char with hot patches that crawl and pulse, short flames licking up here and there, a thin smoke | a fireball on the ground that rises on a stem of dust and cools into a cap that rolls over at its edges, darkening from orange to grey-brown smoke as it climbs, then stops and spreads |
| plays | loops | loops | **once**, 150 frames (5 s): the cap needs about 2.5 s to rise and roll over and 2.5 s more to darken and spread (the explosion and dust play 3 s) |
| source | a band 6% up, 8 to 13 cells high (intensity), its ends tapering over the outer fifth: heat (1.0 to 1.6) × (0.35 + 0.65 c(flicker)) × (0.6 + 0.4 ψ), where c is fire's flicker curve; so all of it burns, in tongues where the flicker is high and stronger and weaker stretches that move along it with the curl noise. Smoke 0.3 to 0.6 per second, an upward push of 16 to 34 cells per second; flicker rate 2.2 per second (fire: 2.6) | a bed 90% of the frame wide (its ends taper), its top 10 to 18% up (intensity) and lumpy (it moves with the flicker: ±20%). In it the heat is drawn at 5 per second towards (0.9 to 1.6) × (0.08 + 0.92 c(flicker)) × (1 + 0.4 ψ): hot patches where the flicker is high, dull ones elsewhere, brightening and dimming with the curl noise (the pulse); flicker rate 0.8 per second, so the pattern crawls. Char (the solver's soot) is drawn up to 0.9 at the same rate. Where the flicker is highest (more of the bed at higher intensity), flames start at the bed's top: heat 0.8 to 1.3, pushed up at 14 to 34 cells per second, a little smoke | one burst at the start: a ball of radius 10 to 15 cells (intensity) on the ground, heat 2.2 to 3.0, a little smoke, its edge and lumps from the flicker noise (rate 1.4, as the explosion's). Then the base, a patch 15 to 23 cells wide: heat (1.4 to 1.9) × (0.12 + 0.88 exp(-t / 1 s)), dust (3 to 4 per second) and an upward draught (25 to 45 cells per second) × (0.3 + 0.7 exp(-t / 1 s)): a bright base that dims, and a stem of dust drawn up into the cap |
| forces | buoyancy 115, soot weight 6; wind up to 70 cells per second with drag 2.0; velocity damping 0.2; confinement 0.6 to 3.6 and curl noise 10 to 65 (turbulence) | the bed's air is still (its velocity zeroed below 60% of its top, fading out by 90%), so only the flames rise; buoyancy 90, soot weight 2; wind up to 45 with drag 1.8; damping 0.6; confinement 0.3 to 1.8, curl noise 5 to 35 | the hot gas expands for its first moments (a divergence of 6 to 12 per second, fading with a time constant of 0.15 s, as the explosion's); buoyancy 45, soot weight 2; a closed ground (the pressure solve's bottom border, as fog's and dust's); **stable air** above 55 to 65% of the height (intensity): hot gas there is pushed down (3 × buoyancy × heat, fading in over 15% of the height), so the cap stops rising and spreads; **sheared wind**: material is dragged (1.2 per second) towards a wind of (2 wind - 1) × 45 cells per second times the height fraction, so the cap leans further than the stem; damping 0.5; confinement 0.3 to 1.3, curl noise 5 to 30 |
| decay | heat 3.2 - 1.0 intensity per second (short flames at low intensity), smoke 0.3; the burning gas turns to smoke at 1.5 × its heat per second | heat 4.5 per second (the flames are short), smoke 3.0 (it is thin) | heat 0.45 per second (the cap glows for two or three seconds), smoke 0.12; the burning gas turns to smoke at 0.45 × its heat per second |
| drawing | emitted light in fire's colours over grey smoke shaded towards the light | emitted light from deep red through red and orange to yellow over dark char (0.08 grey where dense, paler where thin) | the explosion's colours (fire's at 0.75 of the heat) over brown-grey smoke |
| controls | intensity (height and fierceness), wind (lean), turbulence | intensity (how hot and lively: brighter patches, more flames), wind (the flames and smoke lean), turbulence | intensity (yield: the size of the burst, the cap's height), wind (lean and shear), turbulence |

They keep the study's control names. `sim::one_shot_frames` gives a one-shot's length (90 for the explosion and dust,
150 for the mushroom), which the recipe and study D's steps now read where they had 90 (§3). Every earlier effect's
clips are unchanged to the byte: `Sim.StudyEffectsAreUnchangedByTheNewOnes` now also hashes clips of jet, orb, vortex,
fog, dust and cloud made before this round, and full-size clips of all eleven at two settings were compared byte for
byte before and after. Tests of the new presets: `Sim.BlazeEffectsAreNamedAndOnlyTheMushroomPlaysOnce`,
`Sim.WildfireBurnsAcrossTheWholeWidthWithSmokeAboveAndLeansWithTheWind`, `Sim.CoalsGlowLowAndWideAndTheirGlowCrawls`,
`Sim.MushroomRisesIntoACapThatSpreadsAndDarkens`, and `Rollout.OneShotEffectsTakeTheExplosionsRecipe` for the
mushroom's recipe.

**How they were designed.** Each was looked at with `nvfx_sim --sheet` and a scratch probe (sheets of several settings
side by side, and the amount, centroid, extent and speed of the material over time) at several control settings until
it read as what it is at 128 px. What changed on the way:
- **Wildfire.** The first front, shaped by the flicker alone, burnt in a few separate tongues with gaps between them at
  low intensity, and its flames reached the top of the frame as thin red streaks; its smoke was invisible. A base level
  of burning along the whole front (35% of the source wherever the flicker is low), heat that cools faster (more so at
  low intensity), a thicker source and smoke made from what burns gave a continuous burning front, tongues of flame in
  the lower third to half and a rolling bank of smoke above. Its fastest flow at the strongest settings was 1.4 coarse
  cells a frame; less buoyancy and a weaker push brought it to 1.2 (fire: 1.0, magic: 1.9).
- **Coals.** The first bed, 8 cells high, was a thin glowing line, and its flames rose as small smoke rings on stalks
  (mushroom-shaped puffs). The bed is now twice as high; its hot patches are sparser (most of it dull red under dark
  char); the flames' heat fades at 4.5 per second (fire's: 1.25) and their smoke at 3, so they are short tongues and a
  haze rather than puffs.
- **Mushroom.** At first the fireball rose to the top of the frame within a second and its cap broke into thin swirls,
  the kind of thin, regular structure that magic's lesson warns against. Less buoyancy, a smaller and slower burst,
  weaker confinement and the stable layer above it slowed it to about 2.5 s for the climb, kept its rolls broad and the
  cap inside the frame. The cap then split into two lobes with a gap between them (a rising 2D thermal becomes a vortex
  pair); a stem of dust that keeps rising from the base fills the gap and reads as the stem.

![Wildfire, coals and mushroom, the simulator's presets: two frames at each of three settings (low, middle and high controls), wildfire and coals a second apart, the mushroom at 1.5 and 4.5 s](figures/blaze_presets.png)

How fast they move, at two settings (all controls 0.5 and all 1; cells of a 128-cell frame per second where there is
material, median and 99th percentile; a coarse cell is 4 of them, a frame 1/30 s; looping effects in the steady state,
frames 150 to 240; the one-shots over their whole length):

| effect | median | 99th percentile | in coarse cells a frame (99th) |
|---|---:|---:|---:|
| fire | 52 to 71 | 96 to 124 | 0.80 to 1.04 |
| explosion | 32 to 63 | 70 to 153 | 0.58 to 1.27 |
| wildfire | 33 to 82 | 85 to 144 | 0.71 to 1.20 |
| coals | 0 to 2 | 19 to 41 | 0.16 to 0.34 |
| mushroom | 17 to 26 | 52 to 68 | 0.44 to 0.56 |

The wildfire moves as fire does (its strongest settings a little faster), the coals' bed stands still and only its
small flames move, and the mushroom is slower than the explosion: a large cloud seen from further away.

**How chaotic they are** (`nvfx_experiment d-chaos --effects wildfire,coals,mushroom`, as [REPORT.md](REPORT.md) §6.1:
active PSNR against the undisturbed run, mean over 4 runs; [d_chaos.csv](../results/experiments/fx_blaze/d_chaos.csv)):

| effect | case | 1 frame | 0.5 s | 1 s | 2 s | 4 s | 8 s |
|---|---|---:|---:|---:|---:|---:|---:|
| wildfire | start nudged by 1e-3, same seed | 76.7 | 69.0 | 66.6 | 62.3 | 57.7 | 53.2 |
| wildfire | start nudged by 1e-1, same seed | 53.4 | 44.2 | 41.0 | 37.2 | 35.7 | 33.7 |
| wildfire | same start, other seed | 26.5 | 20.9 | 19.9 | 19.1 | 17.3 | 18.5 |
| wildfire | other start, same seed | 16.0 | 17.3 | 18.7 | 21.0 | 24.9 | 28.4 |
| coals | start nudged by 1e-3, same seed | 90.8 | 88.1 | 89.0 | 95.8 | 99.0 | 96.0 |
| coals | start nudged by 1e-1, same seed | 66.3 | 62.4 | 65.2 | 69.0 | 75.1 | 84.3 |
| coals | same start, other seed | 25.4 | 13.4 | 14.8 | 13.5 | 14.6 | 13.6 |
| coals | other start, same seed | 17.1 | 29.8 | 36.3 | 41.6 | 52.5 | 64.5 |
| mushroom | start nudged by 1e-3, same seed | 73.6 | 68.7 | 66.6 | 64.0 | 62.1 | - |
| mushroom | start nudged by 1e-1, same seed | 51.5 | 40.9 | 36.8 | 31.8 | 30.0 | - |
| mushroom | same start, other seed | 31.0 | 18.4 | 14.4 | 12.2 | 13.0 | - |
| mushroom | other start, same seed | 16.3 | 17.9 | 18.3 | 17.0 | 16.9 | - |

- **The wildfire behaves as fire does**, with a seed that takes over a little more slowly (20 dB from another seed
  after a second, fire's 13): its front always burns in the same place, only its tongues and smoke move.
- **The coals are forced, as cloud and the orb are**: an error in the start heals rather than grows (a 10% nudge is at
  84 dB after 8 s), another seed decides the look within half a second, and two runs with the same seed from different
  start points meet (65 dB after 8 s). What a bed of coals looks like is decided by its seed and nothing else.
- **The mushroom behaves as the explosion does**: the seed takes over within a second, and a run from another start
  with the same seed never meets it (17 dB): the burst decides the cloud's shape.

## 3. Training

Study D's recipe, unchanged (`rollout::recipe_for`), as the last six were trained: runs from an empty grid with random
controls and seeds of their own (training salt 1), the stepper's three stages, the renderer on 200 frames of 100 runs,
the detail constants chosen by matching frame statistics of endless runs against real runs at the controls of four
training runs with new seeds (validation; the test's settings and seeds are never used), and the start points. Its
branches decide the rest:
- **Wildfire and coals take fire's settings**: 160 runs of 240 frames, 8 start points, fine fields grown at instance
  start (a 1 s warm-up).
- **The mushroom plays once, so it takes the explosion's branch** (`!sim::effect_loops(e)`): 240 runs, 16 start points
  at its first frame with 64-pixel fine fields, burn-in of up to 32 frames, at its own length: 150 frames instead of
  90. That length is the only thing new in the recipe (`sim::one_shot_frames`), and two places that assumed 90 frames
  now read it: the detail constants are calibrated over the one-shot's whole length (`finish_model`; the explosion's
  and dust's 89 frames, as before) and study D's steps (`d-chaos`, `d-eval`) track and play it to its end.

Each is trained in study D's three saved steps (`d-train --stages12 1`, `d-tune` for stage 3, `d-finish`), one
thread, `nice 10`, one effect at a time, from a copy of the tool (`DATA/fx_blaze/jobs/train.sh`, resumable after a
restart; logs under `DATA/fx_blaze/logs`).

To come: the training table.

## 4. The test

Written before any test is run, as the last two rounds' were. Each model is frozen (its SHA-256 in
`results/experiments/fx_blaze_frozen.csv`) before the test, and the test runs once, for all three effects
(`nvfx_experiment d-eval --effects wildfire,coals,mushroom`, then `d-report`), on study D's protocol exactly as the
last six were tested ([EFFECTS_ENERGY.md](EFFECTS_ENERGY.md) §4). Every run of the test is independent and seeded by
its effect, run and setting.

- **Precision.** The runtime plays the models as it ships them: rollout effects at int8 by default since study K
  ([REPORT.md](REPORT.md) §13: the stepper's second convolution in 8 bits, on the AVX-512 build with VNNI, which this
  machine has). The `neural` rows are int8; `neural_dynamics_true_renderer` is the float reference implementation
  (`rollout::step`), as in every earlier round.
- **Tracking from the true state:** 8 held-out runs per effect (salt 2: seeds 700000 to 700007 with controls drawn
  from that salt, which no training or validation run used), each started from its true state (coarse and fine
  fields; wildfire and coals after 100 frames, the mushroom after its first frame) with its own noise seed: 240 frames
  (the mushroom: to its last frame, 149). The neural rollout through the runtime against the simulation on the same
  32-cell grid with the same detail layer (`coarse_sim_detail`), the plain coarse simulation, holding the first frame,
  and the neural dynamics drawn by the simulation's renderer. Active PSNR at 1 and 8 frames and at 1, 2 and 8 s (the
  mushroom: 1 and 8 frames, 1, 2 and 4 s); the neural run minus `coarse_sim_detail`, paired over the 8 runs.
- **Endless runs at settings never seen:** the ten held-out settings of study B (`b_test_settings`: none on study B's
  training grid), 10 s each after the start (the mushroom: its 5 s), with seeds never used for these effects (900000 + k
  and 910000 + k for the two real runs, wildfire and coals warmed up 5 s; 920000 + k for the neural run, played as it
  ships, in shards from the start points; 930000 + k for the coarse simulation). Frame statistics against a real run:
  detail spectrum distance, motion ratio (as the error |ln ratio|), coverage distance and mean-frame PSNR. Compared with
  a second real run with another seed (the floor), the nearest of the 45 training flipbooks of study B's grid (made for
  these effects as study B made its library: 128 px, seed 1, 64 frames looped; **the mushroom's are its whole 150
  frames**, since 64 frames held on the last would leave more than half of its 5 s still: a flipbook library 2.3 times
  study B's), and the coarse simulation with the detail layer and without.
- **One minute:** a 60 s neural run in shards at the first two held-out settings, cut into 10 s windows, each against
  the real 10 s run (wildfire and coals; the mushroom plays once).
- **Statistics:** every difference paired (over runs or settings), with a 95% bootstrap interval of 10,000 resamples
  (`metrics::paired_bootstrap`); an interval across zero is a tie. Every number is reported, ties as ties.

What counts as working, decided now, as for the energy effects: an effect **works** if it tracks its runs better than
`coarse_sim_detail` at 1 s (interval above zero) and, played endlessly, its average picture is closer than the nearest
flipbook's and its motion error is no worse than the flipbook's (interval not above zero). An effect that ties or loses
to the coarse simulation at tracking at 1 s, or whose detail and motion both lose to the flipbook, is reported as
**weak**, as magic, the orb and the vortex were.

To come: the results.

## 5. Memory and cost

To come: `d-timing` beside study D's fire in the same session (provisional, a shared machine).

## 6. The library and a scene

To come.

## 7. What it means

To come.

## 8. Reproduce

```sh
build/nvfx_sim --effect wildfire --intensity 0.6 --wind 0.3 --sheet wildfire.png   # the simulator's presets alone
build/nvfx_sim --effect coals --intensity 0.8 --sheet coals.png
build/nvfx_sim --effect mushroom --intensity 0.8 --frames 150 --sheet mushroom.png
R="--root $NEURALVFX_DATA/fx_blaze --results results/experiments/fx_blaze"
build/nvfx_experiment d-chaos  --effects wildfire,coals,mushroom $R --threads 1
build/nvfx_experiment d-train  --effects wildfire --stages12 1 $R --threads 1
build/nvfx_experiment d-tune   --effects wildfire $R --threads 1
build/nvfx_experiment d-finish --effects wildfire $R --threads 1
#   (the same for coals and mushroom)
build/nvfx_experiment d-eval   --effects wildfire,coals,mushroom $R --threads 2 --figures DIR   # the test, once
build/nvfx_experiment d-report $R                                                             # D_SUMMARY.md
```
