# Energy and motion effects: jet, orb and vortex

Status: **in progress**: the test's design (§4) is written before any test is run; the jet and the orb are trained and
frozen, the vortex is training. Audience:
dev, research, artists. Three effects a game would want next to fire, smoke, the explosion, steam and magic: a
**jet** (a flamethrower or a rocket thruster: a narrow jet of burning gas from a nozzle), an **orb** (a ball of cyan
plasma that boils inside a soft boundary and breathes) and a **vortex** (a dust devil seen from the side). They come
from the project's own 2D fluid simulator with new sources, forces and drawing, and were trained and tested exactly as
steam and magic were ([EFFECTS.md](EFFECTS.md)): study D's recipe and protocol ([REPORT.md](REPORT.md) §6), on held-out
settings and seeds. A scene script plays all three together (§6).

## 1. Bottom line

(Written after the test.)

## 2. The effects

All three are presets of the existing solver (`src/sim/sim.cpp`, in a block of their own: `add_energy_sources`,
`add_energy_forces`, `add_energy_divergence`, `energy_decay`, `render_energy`): the same grid, advection, vorticity
confinement, curl noise and pressure solve, with their own sources, forces, decay and drawing. Each has three
normalised controls. Lengths are in cells of a 128-cell frame and times in seconds.

The lesson of magic shaped them: its fast, regular spiral (a median speed of 120 to 170 cells per second, up to 1.9
coarse cells a frame) smeared on the 32-cell state. These three keep their flow at fire's and steam's speeds or below,
their structures broad, and their detail from turbulence and the dust or smoke it carries rather than from thin
regular arms. And every force is a function of what the learned stepper sees: the state where it acts, the position,
the controls and the run's two noise fields (the curl noise's stream function and the source flicker). Nothing
follows a clock of its own: the orb breathes and the vortex sways with the curl noise where they are, so a learned
model can see what drives them.

| | jet | orb | vortex |
|---|---|---|---|
| what it is | a narrow jet of burning gas from a nozzle at the left edge, nearly level; it widens and breaks up along its length, burns from bluish white through yellow and orange to dark smoke, and its end curls up | a ball of plasma in the middle of the frame that boils inside a soft boundary, throws off short wisps that fade at once, and breathes | a column of warm air rising from a patch of hot ground, which lifts dust from the floor into a funnel that sways, leans and spreads towards the top |
| source | a nozzle 5% in from the left edge and 40% up, 8 to 12 cells across (intensity), aimed 20 degrees below to 30 degrees above level (aim); heat 1.5 to 2.2 and a little soot, the gas set moving at 85 to 150 cells per second along the aim; broken up by the flicker noise at 3.0 per second (fire: 2.6) | a disc of radius 20 to 33 cells (intensity) filled with energy (the solver's heat) of 1.1 to 1.6, brightest in the middle, and a faint haze (its soot), broken up by the flicker noise at 2.2 per second; the energy swells with the pulse (below) | warm air (heat 0.5 to 0.8) over a patch of floor 9 to 17 cells wide (intensity); dust (the solver's soot) lifted from the floor within three and a half times that where the air rushes along it, at 14 to 24 per second, broken up by the flicker noise at 1.5 per second |
| forces | buoyancy 38 (fire: 150), soot weight 3; no wind; velocity damping 0.45 per second; confinement 0.8 to 4.0 and curl noise 12 to 67 (turbulence) | no buoyancy and no wind. The hot plasma is drawn to the middle (140 times its heat, in cells per second squared), which keeps it a ball and makes it boil; the air beyond its edge is held still (damped at up to 4 per second). The **pulse**: a source of divergence in its core of (1 + 5 pulse) tanh((1 + 4 pulse) ψ) per second, where ψ is the curl noise's stream function there (in [-1, 1], changing at 0.8 per second), so the ball swells and shrinks with it, more often sharply at a high pulse. Confinement 1.5 to 5, curl noise 20 to 80, damping 1 per second | buoyancy 115, dust weight 2; damping 0.35 per second. **Sway**: everywhere, the air is dragged (1.5 per second) towards a level wind of (8 + 42 sway) ψ cells per second, which leans and bends the column and makes it wander. The column draws air in at its foot and spreads as it rises (a divergence of -6 per second in the warm air near the floor, +3 (0.6 + 0.8 intensity) times the height fraction above). Little turbulence at its foot, more towards its top: confinement 0.3 to 0.8 everywhere and up to 0.5 to 3.5 more, curl noise 4 to 12 and up to 8 to 58 more, growing from 10% to 85% of the height |
| decay | heat 1.1 per second, smoke 0.4; the burning gas turns to smoke (soot grows by 0.7 times the heat per second) | energy 1.0 per second, haze 1.2; beyond the ball's edge both fade at up to 3.5 per second more | warm air 0.4 per second, dust 0.75; dust spreads sideways as it rises, at 25 (height fraction)² cells² per second: the funnel widens towards its top |
| drawing | emitted light by heat: dark red, orange, yellow, white, and bluish white at its hottest, over dark smoke shaded towards the light | emitted light: deep blue where faint, through cyan to white, over a faint blue haze | reflected light only: brown-grey dust, lit from above and to the left and shaded by itself |
| controls | intensity (flow and length), **aim**, turbulence | intensity (size and brightness), **pulse** (how strongly and how sharply it breathes), turbulence | intensity (strength and width), **sway** (how far and fast it wanders), turbulence |

The second controls are named `aim`, `pulse` and `sway` (`sim::control_names`, so the model files, the runtime and
scene scripts call them that; `nvfx_sim` takes `--aim`, `--pulse` and `--sway`). Every earlier effect's clips are
unchanged to the byte: `Sim.StudyEffectsAreUnchangedByTheNewOnes` now hashes a clip of fire, smoke and the explosion
(made before steam and magic existed) and of steam and magic (made before these three existed). Tests check what each
one is: the jet comes from the left in fire colours with dark smoke and its aim raises it
(`Sim.JetIsAFlameFromTheLeftThatItsAimRaises`); the orb stays in the middle, is cyan and its lit area swings more
with a strong pulse (`Sim.OrbIsAContainedBallOfCyanLightThatBreathesWithItsPulse`); the vortex's dust reaches the top
of the frame, is brown-grey reflected light, and wanders more with its sway
(`Sim.VortexIsAColumnOfDustThatSwaysWithItsControl`).

How fast they move, in the steady state at two settings (all controls 0.5 and all 1; cells of a 128-cell frame per
second where there is material, median and 99th percentile; a coarse cell is 4 of them, a frame 1/30 s):

| effect | median | 99th percentile | in coarse cells a frame (99th) |
|---|---:|---:|---:|
| fire | 47 to 65 | 118 to 121 | 0.98 to 1.01 |
| steam | 27 to 57 | 90 to 132 | 0.75 to 1.10 |
| magic | 118 to 168 | 191 to 223 | 1.59 to 1.86 |
| jet | 25 to 51 | 115 to 148 | 0.95 to 1.24 |
| orb | 17 to 25 | 43 to 65 | 0.36 to 0.54 |
| vortex | 20 to 36 | 100 to 136 | 0.83 to 1.14 |

**How chaotic they are** (`nvfx_experiment d-chaos --effects jet,orb,vortex`, as [REPORT.md](REPORT.md) §6.1: active
PSNR against the undisturbed run, mean over 4 runs):

| effect | case | 1 frame | 0.5 s | 1 s | 2 s | 8 s |
|---|---|---:|---:|---:|---:|---:|
| jet | start nudged by 1e-3, same seed | 71.8 | 64.6 | 62.1 | 56.9 | 44.5 |
| jet | start nudged by 1e-1, same seed | 48.9 | 34.8 | 29.5 | 26.0 | 23.7 |
| jet | same start, other seed | 36.8 | 20.9 | 16.1 | 14.8 | 15.6 |
| jet | other start, same seed | 14.0 | 14.2 | 14.6 | 16.3 | 20.2 |
| orb | start nudged by 1e-3, same seed | 82.6 | 73.2 | 72.4 | 69.4 | 58.5 |
| orb | start nudged by 1e-1, same seed | 59.3 | 48.3 | 46.7 | 44.5 | 44.3 |
| orb | same start, other seed | 19.4 | 14.6 | 12.2 | 12.1 | 13.8 |
| orb | other start, same seed | 13.0 | 17.5 | 19.1 | 23.3 | 40.1 |
| vortex | start nudged by 1e-3, same seed | 73.7 | 69.1 | 67.1 | 64.8 | 59.9 |
| vortex | start nudged by 1e-1, same seed | 51.3 | 41.6 | 37.8 | 36.5 | 42.8 |
| vortex | same start, other seed | 43.8 | 24.6 | 20.6 | 19.0 | 17.3 |
| vortex | other start, same seed | 14.8 | 16.6 | 17.8 | 19.6 | 25.4 |

- **The jet behaves as fire does**: a small error in the start stays invisible for seconds, the seed takes over within a
  second, and the noise slowly pulls runs with the same seed together.
- **The orb is forced more**, as magic was, though less: after a second its seed decides what it looks like (12 dB from
  another seed), and two runs with the same seed from different start points meet over a few seconds (40 dB at 8 s;
  magic: 47 dB).
- **The vortex's seed takes over more slowly**: its column stands where its hot ground is, so another seed is still at
  21 dB after a second and 17 dB after eight (fire: 13).

## 3. Training

(Filled in after training.)

## 4. The test

Written before any test was run. Each model is frozen (its SHA-256 in `results/experiments/fx_energy_frozen.csv`)
before its own test, and each test runs once. The test runs in two parts on this one design: the jet and the orb were
frozen and tested while the vortex was still training (`d-eval --effects jet,orb`), and the vortex is frozen and
tested when its training ends (`d-eval --effects vortex`, its rows added to the same CSVs). Every run of the test is
independent and seeded by its effect, run and setting, so the split changes no number. It is study D's protocol,
exactly as steam and magic were tested ([EFFECTS.md](EFFECTS.md) §4; `nvfx_experiment d-eval`, then `d-report`):

- **Tracking from the true state:** 8 held-out runs per effect (salt 2: seeds and settings no training or validation
  run used), each started at 100 frames from its true state (coarse and fine fields) with its own noise seed; 240
  frames. The neural rollout through the runtime against the simulation on the same 32-cell grid with the same detail
  layer (`coarse_sim_detail`), the plain coarse simulation, and holding the first frame. Active PSNR at 1 and 8 frames
  and at 1, 2 and 8 s; the neural run minus `coarse_sim_detail`, paired over the 8 runs.
- **Endless runs at settings never seen:** the ten held-out settings of study B (`b_test_settings`: none on study B's
  training grid), 10 s each after the start, with seeds never used for these effects (900000 + k and 910000 + k for the
  two real runs, 920000 + k for the neural run, 930000 + k for the coarse simulation; training used 600000 to 600159,
  the held-out tracking runs 700000 and up, validation seeds derived from the training runs'). Frame statistics against
  a real run: detail spectrum distance, motion ratio (as the error |ln ratio|), coverage distance and mean-frame PSNR.
  Compared with a second real run with another seed (the floor), the nearest of the 45 training flipbooks of study B's
  grid (made for these effects as study B made its library: 64 frames at 128 px, seed 1; looped), the coarse
  simulation with the detail layer and without.
- **One minute:** a 60 s neural run in 6 s shards at the first two held-out settings, cut into 10 s windows, each
  against the real 10 s run.
- **Statistics:** every difference paired (over runs or settings), with a 95% bootstrap interval of 10,000 resamples
  (`metrics::paired_bootstrap`); an interval across zero is a tie. Every number is reported, ties as ties.

What would count as working, decided now: an effect **works as steam did** if it tracks its runs better than
`coarse_sim_detail` at 1 s (interval above zero) and, endless, its average picture is closer than the nearest
flipbook's and its motion error is no worse than the flipbook's. An effect that ties the coarse simulation at tracking,
or whose detail and motion lose to the flipbook, is reported as **weak**, as magic was.

## 8. Reproduce

```sh
build/nvfx_sim --effect jet --aim 0.8 --sheet jet.png             # the simulator's presets alone (--pulse, --sway)
build/nvfx_experiment d-chaos  --effects jet,orb,vortex --root $NEURALVFX_DATA/fx_energy --results results/experiments/fx_energy
build/nvfx_experiment d-train  --effects jet --stages12 1 --root $NEURALVFX_DATA/fx_energy --results results/experiments/fx_energy --threads 1
build/nvfx_experiment d-tune   --effects jet --root $NEURALVFX_DATA/fx_energy --results results/experiments/fx_energy --threads 1
build/nvfx_experiment d-finish --effects jet --root $NEURALVFX_DATA/fx_energy --results results/experiments/fx_energy --threads 1
#   (the same for orb and vortex)
build/nvfx_experiment d-eval   --effects jet,orb,vortex --root $NEURALVFX_DATA/fx_energy --results results/experiments/fx_energy --threads 2 --figures DIR
build/nvfx_experiment d-report --root $NEURALVFX_DATA/fx_energy --results results/experiments/fx_energy    # D_SUMMARY.md
build/nvfx_experiment d-timing --effects fire,jet,orb,vortex --root ROOT --results results/experiments/fx_energy
build/nvfx_scene_script --script examples/scenes/firewhirl.nvfxs --models DIR --out firewhirl.mp4 --keyframes DIR --sheet sheet.png
```

`ROOT/models/d` for the timing holds study D's `fire.nvfx` with the three new effects. `DIR` for the scene holds
`jet.nvfx`, `orb.nvfx` and `vortex.nvfx`.
