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

(The rest of this section, the figures and the speeds, are written when the presets are final.)

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
