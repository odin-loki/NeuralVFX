# Fire effects, round 10: a crown fire, a window fire and a dust explosion

Status: **in progress** (21 October 2026). The design of the test and the rule below (§4) were written and committed
before any model was trained, and so was the one method of a later study that this round uses (§3): nothing below §4's
design has been run yet. Audience: dev, research, artists.

Three more fire effects from the catalogue ([TODO.md](../TODO.md), rows 6, 8 and 31), after the air round
([EFFECTS_AIR.md](EFFECTS_AIR.md)): a **crown fire** (fire running through a forest canopy: the crowns torch, flames
stand over them and lean before the wind, embers thrown ahead light spot fires in the crowns downwind, and a tall column
of smoke rises over it and bends over in the wind; distinct from round 7's wildfire, a front of flames on the ground),
a **window fire** (a room fire venting from a window: flames pour out of the upper part of the opening and roll up the
wall over it, the room glows through the lower part, and black smoke runs up to the eave and spreads out under it as a
ceiling jet) and a **dust explosion** (a cloud of dust, in a mill or a mine, lit at one end or in its middle: a fast
deflagration runs through it, the burning gas expands into a brown-orange fireball that billows up, and it leaves a
brown pall of dust and ash; it plays once, for 4 s). They come from the project's own 2D fluid simulator with new
sources, forces and drawing, and are trained and tested as the earlier rounds' were: study D's recipe and protocol
([REPORT.md](REPORT.md) §6), on held-out settings with seeds of this round's own (26,000,000 to 26,999,999).

## 2. The effects

All three are presets of the existing solver (`src/sim/sim.cpp`, in a block of their own: `add_fire2_sources`,
`add_fire2_forces`, `add_fire2_divergence`, `fire2_decay`, `render_fire2`). Each has three controls in [0, 1]:

| | crownfire | windowfire | dustexplosion |
|---|---|---|---|
| controls | intensity (how fierce: the zone's width, the flames' heat and lift, the smoke), wind (0.5 still air; less to the left, more to the right), turbulence | intensity (the room fire's size: the outflow's heat and speed, a lower neutral plane, more smoke), wind (along the facade), turbulence | intensity (the cloud's size and how much dust it holds), **ignition** (where it is lit: the cloud's left end at 0, its middle at 0.5, its right end at 1), turbulence |
| plays | loops | loops | **once**, 120 frames (4 s) |

Every force is a function of what a learned stepper sees: the state where it acts, the position, the controls and the
run's two noise fields (the curl noise's stream function and the source flicker). Nothing follows a clock of its own.
The dust explosion, which plays once, is made at its first step (its cloud and the spark that lights it, as the
explosion's burst is) and then burns by its own state alone: where its dust is hot it burns, giving heat and expanding,
and the heat spreads into the dust beside it, so the flame runs through the cloud as a front. It follows the time since
its start only through the slower of the two decays a learned one-shot sees as inputs (exp(-t / 1 s)), which sets how
long the cloud can burn (fully for its first 0.9 s, not at all after 2.1 s), as the mushroom's base fades with it.

**The dust explosion's length, 120 frames (4 s).** The flame runs through the cloud in its first 0.5 s (lit in its
middle) to 1 s (lit at an end); the fireball swells and billows up over the next second, burning on where its dust is
richest; by 2.5 s it has burnt out, and in the last 1.5 s its pall of dust and ash rises a little more, spreads and thins.
Three seconds (the explosion's length) would cut off the pall as it forms; five (the mushroom's) would add a second of
a fading haze, and the mushroom showed that a long one-shot, whose look changes over its length, is the hard case for
the learned stepper.

Lengths are in cells of a 128-cell frame and times in seconds; d = 2 wind - 1 and ψ is the curl noise's stream function
where the force acts.

| | crownfire | windowfire | dustexplosion |
|---|---|---|---|
| what it is | a forest canopy across the lower part of the frame (crowns from 22 cells up to their tops, 40 to 46 cells up, a tree every 16 cells; the trunks and the ground are the scene's to draw); a burning zone 60 to 80 cells wide (intensity), 8 d cells upwind of the middle, whose crowns torch in flames that stand 20 to 40 cells over the canopy and lean downwind; spot fires in the crowns beyond it; a tall column of grey-brown smoke to the top of the frame that bends over downwind | the facade seen front-on: a window 36 cells wide and 32 tall, its sill 12 cells up, its head 44; the room's hot gas pours out through the opening above the neutral plane (24 to 30 cells up, lower for a bigger fire) and its flames roll up the wall over the head; below it the room glows through the inflowing air; black smoke runs up to the eave (the top of the frame) and spreads out both ways under it as a ceiling jet | an oval cloud of dust on the ground in the middle of the frame, 60 to 88 cells wide and 28 to 40 tall (intensity), lit low in the cloud at its left end, its middle or its right end (ignition): the flame runs through it, the burning gas expands into a brown-orange fireball that billows up and burns out by 2.5 s, and a brown pall of dust and ash is left that rises a little, spreads and thins |
| source | in the crowns, through their depth: heat (1.5 to 1.9) k and an upward flow of (24 to 40) k cells per second, with k the crown's shape (1 in its middle, 0.6 between crowns) times the zone (its downwind half burning up to a quarter harder in a wind: the head) times the torching (0.5 + 0.5 smoothstep(-0.2, 0.6) of the flicker noise, rate 2.2) times (0.75 + 0.25 ψ); smoke at (1.2 to 1.8) k a second. Spot fires: 0.7 smoothstep(0.3, 0.6) of the flicker in a band beyond the zone's edges, out to 4 cells in still air, to 36 cells downwind in the strongest wind and none upwind | above the neutral plane the heat is drawn (10 per second) towards (1.3 to 1.8)(0.5 + 0.5 f) (f the flicker noise's smoothstep, rate 2.6), rising from nothing over the plane's first 40% of the way to the head, the flow upwards towards (18 to 32)(0.4 + 0.6 of that way) cells per second, and smoke at (0.3 to 0.5) a second; below the plane (and wherever the outflow is weaker) the room's glow, heat drawn (4 per second) towards (0.75 to 1)(0.35 + 0.65 f), in still air; the wall's air under the sill is still | at its first step: dust (0.7 to 1.2)(0.7 + 0.3 of its lumps) inside an edge the flicker noise makes ragged (rate 1.4), at rest, and a spark of heat 1.4, 4.5 cells across, at 0.75 of the half-width from the middle towards the lit end |
| forces | buoyancy 110, smoke weight 3; **the wind** drags material (2 per second) towards d (16 + 50 smoothstep(40, 110, y)) cells per second: slowed in the canopy, 66 d at the top of the frame, so the column bends over; a closed ground; damping 0.2; confinement 0.5 to 3.0, curl noise 8 to 53 (turbulence) | buoyancy 110, smoke weight 2; **the eave**: upward flow stops within 5 cells of the top of the frame, over a closed top border, so the smoke spreads out under it; a wind along the facade drags material (1.2 per second) towards 22 d (0.4 + 0.6 y / 128); damping 0.2; confinement 0.4 to 2.0, curl noise 8 to 48 | buoyancy 35, dust weight 5; **the burning gas expands**: a divergence of 2.5 times its burn rate in the pressure solve, which pushes the cloud out ahead of the flame and swells the fireball; a closed ground; damping 0.6; confinement 0.2 to 0.8, curl noise 4 to 24 |
| decay | above a heat of 0.4 the flames burn out at up to 1.7 per second and leave 0.6 of the heat they lose as smoke; the warm column cools at 0.3 per second, the smoke at 0.08 | once out of the window, the gas above a heat of 0.4 burns out at up to 1.6 per second and leaves 0.8 of it as black smoke; the warm smoke cools at 0.5 per second, the smoke at 0.2 | **the burn**: the heat spreads at 40 cells squared a second (0.5 to 1.5 times that with the flicker noise, so the front wrinkles), only where there is dust; where the dust is hotter than its ignition (0.35 to 0.7) it burns at up to 80 times its amount a second, up to a cap of 1.25 to 1.6 (richer where ψ is higher: its air used up) and while the cloud can burn (exp(-t / 1 s) from 0.4 down to 0.12); a unit burnt gives 4 of heat and leaves 0.9 of itself as ash; the gas cools at 1.3 per second, the dust at 0.08 |
| drawing | emitted light from a ramp that starts at 0.4 (dull red, orange, yellow, a pale yellow core), so the warm column is not seen, over grey-brown smoke lit from above and to the left | the same ramp, which also draws the room's glow, over black smoke shaded towards the light | emitted light from a brown-orange ramp that starts at 0.3, over tan-brown dust lit from above and to the left, darker where it is hot (charred) |

All three keep the study's control names but for the dust explosion's second, `ignition` (`sim::control_names`, so the
model files, the runtime and scene scripts call it that; `nvfx_sim` takes `--ignition`). The window fire's room is not
drawn either: its glow is what the flames inside show through the opening, and the facade, the window's frame and the
eave are the scene's. Every earlier effect's clips are unchanged to the byte: `Sim.StudyEffectsAreUnchangedByTheNewOnes`
hashes small clips of all 27 earlier effects, and full-size clips (128 px; the loops 64 frames, the one-shots their
length) of all 27 at two settings were compared byte for byte before and after (the data root's
`fx_fire2/jobs/state/baseline_clips.sha256` and `now_clips.sha256`). Tests of the new presets:
`Sim.Fire2EffectsHaveTheirOwnControlsAndOnlyTheDustExplosionPlaysOnce`,
`Sim.CrownFireBurnsInTheCanopyWithSpotFiresAheadAndATallColumn`,
`Sim.WindowFirePoursOutOfItsWindowUpTheWallAndSpreadsUnderTheEave`,
`Sim.DustExplosionBurnsThroughItsCloudFromWhereItIsLitIntoAFireballAndAPall`, and in the recipe
`Rollout.Fire2EffectsTakeTheirRecipes`.

**How they were designed.** Each was looked at with contact sheets at several settings (from an empty grid and in the
steady state) and a scratch probe that printed the amount, centroid and extent of the heat and the smoke over time, a
column's profile and the speed where there is material. What changed on the way:
- **Crown fire.** The first fire, a zone of crowns 16 to 38 cells up burning out fast, gave small flames in the middle
  and a faint column; burning out more slowly and a stronger lift gave one tall flame in the middle, which read as a
  bonfire. What reads as a crown fire is the crowns themselves: the canopy raised off the ground, a crown every 16
  cells (four coarse cells, so the coarse state can carry it), each crown torching in its own flame, a zone wider than
  half the frame, and a warm column that cools slowly (so it keeps rising) under flames that burn out above a heat of
  0.4 and draw no light below it (at first the column glowed dull red). A wind that only dragged the flames moved the
  smoke too little and an upwind shift of the zone undid it; a stronger wind that grows with height bends the column
  over.
- **Window fire.** The first outflow poured soot with its flames and they read as murky; its smoke kept thickening for
  seconds (the layer under the eave never settled) and a bright line marked the neutral plane, where the outflow's
  heat began with no smoke over it. Less smoke in the outflow and more from the flames as they burn out, smoke that thins
  faster, an outflow whose heat rises from nothing at the plane and a room glow wherever the outflow is weaker, and no
  burning inside the opening (gas burning there in slow air left a dark band of soot) give a window that glows, flames
  that pour from its top and black smoke that settles into a layer under the eave.
- **Dust explosion.** The first burn ran through the cloud at 35 cells a second and stopped before it reached the far
  end; it also overshot its heat in one step (the burn faster than the step) and left a uniform pale fireball. Four
  passes of the spread a step, a burn capped at what the step can give, a cap that varies with the curl noise (so the
  fireball's heat is uneven), dust that chars dark as it burns, less buoyancy (a strong rise drew the fireball out into
  fingers) and more dust left as ash give a flame that crosses the cloud in about a second from an end and half that
  from the middle, a brown-orange fireball and a pall. A square front (the five-point spread) became round with a
  nine-point one, and wrinkled with the flicker noise.

![Crown fire, window fire and dust explosion, the simulator's presets: two frames at each of three settings (all controls 0.2, 0.5 and 0.9); the crown and window fires 1 s apart in the steady state (5 and 6 s), the dust explosion at 0.5 and 1.5 s](figures/fire2_presets.png)

How fast they move, at two settings (all controls 0.5 and all 1; cells of a 128-cell frame per second where there is
material, median and 99th percentile; the looping effects over frames 150 to 240, the dust explosion over its 120 frames;
a coarse cell is 4 cells, a frame 1/30 s; the same probe for fire and smoke):

| effect | median | 99th percentile | in coarse cells a frame (99th) |
|---|---:|---:|---:|
| fire | 52 to 71 | 96 to 124 | 0.80 to 1.03 |
| smoke | 37 to 41 | 92 to 108 | 0.76 to 0.90 |
| crownfire | 31 to 42 | 99 to 114 | 0.82 to 0.95 |
| windowfire | 21 to 24 | 67 to 85 | 0.56 to 0.71 |
| dustexplosion | 14 | 34 to 38 | 0.28 to 0.32 |

The crown fire moves at fire's pace (its flames and column), the window fire more slowly (its outflow, then the layer
under the eave). The dust explosion's gas is slow; what moves fast in it is its flame, which crosses the cloud by burning
at about 50 to 60 cells a second (lit at an end: 0.4 to 0.5 coarse cells a frame) rather than by flowing.

## 3. Training

Study D's recipe (`rollout::recipe_for`), in its three saved steps (`d-train --stages12 1`, `d-tune`, `d-finish`), one
thread per effect, `nice 10`, each from a frozen copy of the tool, resumable (the data root's `fx_fire2/jobs`). Its
branches decide the rest: the crown fire and the window fire take fire's settings (160 runs of 240 frames, 8 start
points), but their smoke lingers (the crown fire's column, the window fire's layer under the eave), as smoke's and the
eruption's do, so they keep 64-pixel fine fields with their start points (one new line in the recipe); the dust
explosion plays once, so it takes the one-shot branch (240 runs of its 120 frames, 16 start points at its first frame
with 64-pixel fine fields). The machine is shared with four other agents and has restarted twice today, so the three
effects train at the same time, each as a job of its own that leaves its state after every step (a restart resumes at
the step that was running), each writing its rows into a folder of its own; the rows are merged afterwards. Nothing in
a run depends on another effect's.

**A method of a later study, decided before any training or validation: the dust explosion gets study M3's mean
state** ([REPORT.md](REPORT.md) §14.9), as round 9's smoke ring did ([EFFECTS_AIR.md](EFFECTS_AIR.md) §3). A one-shot
played from one of 16 stored first states at other controls stays the size of its stored start (the explosion's and
dust's fault, §14.1); the mean state moves each start point to the requested controls by a quadratic fit of the training
runs' first coarse states (`nvfx_ordered --study m3 fitmean --block 2`, on this effect's own salt-1 training runs, under
the data root's `fx_fire2`; `setmean` writes it into the finished model), and study M3 found it better on held-out
settings for both one-shots it was tried on and worse on nothing. It is added after `d-finish` and before the model is
frozen; the file that is frozen and tested is the one with the mean state. Nothing else from studies D2, L, M, M2 or M3
is used: the ensemble-mean fine-tune helped fire and magic but not steam or the jet (§14.8), the swirl at 16 pixels
changes only the detail spectrum and helps or hurts with the swirl's strength, and two detail passes are for flows faster
than a coarse cell a frame, which these are not (§2).

## 4. The test

**Written before any model was trained, and committed with it.** Each model is frozen by its SHA-256
(`results/experiments/fx_fire2_frozen.csv`) before its own test, and each test runs **once**, with `nvfx_experiment
d-eval` and `d-report` unchanged: study D's protocol ([REPORT.md](REPORT.md) §6.4 to §6.6), exactly as round 9 tested
its effects ([EFFECTS_EARTH.md](EFFECTS_EARTH.md) §4, [EFFECTS_AIR.md](EFFECTS_AIR.md) §4). The models train apart, so
the test runs in parts on this one design (`d-eval --effects crownfire`, then the others as each is frozen), each into a
folder of its own, the rows then merged into one set of CSVs under `results/experiments/fx_fire2`. Every run of the test
is independent and seeded by its effect, run and setting, so the split changes no number.

- **Seeds: this round's own, 26,000,000 and up** (`--seed-base 26000000`; this round owns 26,000,000 to 26,999,999 and
  uses no other seeds in its test): the held-out tracking runs 26,000,000 + run; the real runs at the held-out settings
  26,100,000 + setting and 26,110,000 + setting; the neural run 26,120,000 + setting; the coarse simulation 26,130,000 +
  setting; the one-minute runs 26,140,000 + setting. Training and validation use the recipe's own seeds (600,000 + run,
  and those derived from the training runs for the detail constants), and `d-chaos` its own, as every round's did.
- **Tracking from the true state:** 8 held-out runs per effect (salt 2's settings, which no training or validation run
  used), each started from its true state (coarse and fine fields; the crown fire and the window fire after 100 frames,
  the dust explosion after its first frame) with its own noise seed: 240 frames (the dust explosion: to its last frame,
  119). The neural rollout through the runtime against the simulation on the same 32-cell grid with the same detail
  layer (`coarse_sim_detail`), the plain coarse simulation, holding the first frame, and the neural dynamics drawn by the
  simulation's renderer. Active PSNR at 1 and 8 frames and at 1, 2 and 8 s (the dust explosion: 1 and 8 frames, 1 and 2
  s; `d-eval`'s next horizon, 4 s, is past its last frame); the neural run minus `coarse_sim_detail`, paired over the 8
  runs.
- **Endless runs at settings never seen:** the ten held-out settings of study B (`b_test_settings`: none on study B's
  training grid), 10 s each after a 5 s warm-up of the real runs (the dust explosion: its 4 s from its start). Frame
  statistics against a real run: detail spectrum distance, motion ratio (as the error |ln ratio|), coverage distance and
  mean-frame PSNR. Compared with a second real run with another seed (the floor), the nearest of the 45 training
  flipbooks of study B's grid (made for these effects as study B made its library: 64 frames at 128 px, seed 1, looped;
  the dust explosion's are 64 frames held on the last, as the explosion's and the smoke ring's, since its 120 frames are
  less than twice 64), and the coarse simulation with the detail layer and without.
- **One minute:** a 60 s neural run in shards at the first two held-out settings, cut into 10 s windows, each against
  the real 10 s run (the crown fire and the window fire; the dust explosion plays once).
- **Precision:** the runtime plays the effects as they ship, at its default precision, int8 since study K
  ([REPORT.md](REPORT.md) §13); `d-eval` sets none, so the `neural` rows are int8.
- **Statistics:** every difference paired (over runs or settings), with a 95% bootstrap interval of 10,000 resamples
  (`metrics::paired_bootstrap`); an interval across zero is a tie, reported as a tie. Every number is reported.

**The rule, decided now, as rounds 7 to 9 decided theirs.** An effect
- **works** if it tracks its runs better than `coarse_sim_detail` at 1 s (interval above zero) and, played endlessly,
  its average picture is closer than the nearest flipbook's (mean-frame PSNR interval above zero) and its motion error is
  no worse than the flipbook's (interval not above zero);
- is **weak** if it ties or loses to `coarse_sim_detail` at tracking at 1 s, or if, played endlessly, its average picture
  or its motion error loses to the flipbook's (interval on the worse side);
- otherwise **falls short**: it neither works nor is weak (as launch and ink did in round 7 and heavy gas in round 9:
  tracking better, the average picture tied with the flipbook's).

For the dust explosion the question its design raises is answered beside the rule, not by it: whether the learned flame
**runs through the cloud from where it was lit**, as the real one does, rather than the whole cloud flashing at once or the
fireball starting where a stored start point was lit. It is judged on the test's first two held-out settings, from the
real run and the neural run as `d-eval` makes them (seeds 26,100,000 + setting and 26,120,000 + setting), by eye from
their frames side by side and by the centre across the frame of the light the picture gives (pixels whose colour is above
their cover) over the first 1.5 s, and the time at which that light first reaches within a tenth of the frame of the
cloud's far end.
