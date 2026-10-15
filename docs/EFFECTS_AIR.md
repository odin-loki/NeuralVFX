# Air effects: a candle, smoke rings and heavy gas

Status: **in progress** (15 October 2026). The design of the test and the rule below (§4) were written and committed
before any model was trained, and so was the one method of a later study that this round uses (§3): nothing below §4's
design has been run yet. Audience: dev, research, artists.

Three effects of the air, after the water round ([EFFECTS_WATER.md](EFFECTS_WATER.md)): a **candle** (a small, calm,
teardrop flame on a wick, mostly steady, flickering and leaning in a draught, with a thin wisp of smoke; it must read
at small sizes: a candle, a torch head, a lantern), a **smoke ring** (one vortex ring puffed from an opening, rising and
spreading slowly as it rolls, then breaking up; it plays once, for 4 s) and **heavy gas** (a dense, cold vapour released
near the ground that spills and flows along it as a gravity current with a rolling front, pools in the low places and
thins: dry-ice fog, a toxic spill, swamp gas, stage fog). They come from the project's own 2D fluid simulator with new
sources, forces and drawing, and are trained and tested as the earlier rounds' were: study D's recipe and protocol
([REPORT.md](REPORT.md) §6), on held-out settings with seeds of this round's own.

## 2. The effects

All three are presets of the existing solver (`src/sim/sim.cpp`, in a block of their own: `add_air_sources`,
`add_air_forces`, `air_decay`, `render_air`). Each has three controls in [0, 1]:

| | candle | smokering | heavygas |
|---|---|---|---|
| controls | intensity (size), **draught** (0.5 still air; less blows from the right and leans it left, more from the left), turbulence (flicker) | intensity (strength), wind, turbulence | intensity (release rate), **flow** (0 to the left, 0.5 both ways, 1 to the right), turbulence |
| plays | loops | **once**, 120 frames (4 s) | loops |

Every force is a function of what a learned stepper sees: the state where it acts, the position, the controls and the
run's two noise fields (the curl noise's stream function and the source flicker). Nothing follows a clock of its own.
The smoke ring, which plays once, is made at its first step (as the explosion's burst is) and otherwise follows the
time since its start only through the two decays a learned one-shot sees as inputs (exp(-t / 0.15 s), exp(-t / 1 s)),
as the mushroom does. Rings that repeat (a smoker's rings, a censer's) are the one-shot spawned again, so their timing is
the game's, not a noise gate's (round 7's puffs were weak where a gate timed by noise set their rhythm).

**The smoke ring's length, 120 frames (4 s).** It forms in its first half second (the puff rolls up into the two
cores), rises and spreads as it rolls for about two seconds, slowing as its rotation decays, and in the last second and
a half the curl noise, which stirs it harder as it ages, breaks it up while its smoke thins. Three seconds (the
explosion's length) would cut it off while it still rolls as a ring; five (the mushroom's) would add a second of a
faint, broken haze, and the mushroom showed that a long one-shot, whose look changes over its length, is the hard case
for the learned stepper.

(The rest of this section, the figures and the speeds, are written when the presets are final.)

## 3. Training

Study D's recipe (`rollout::recipe_for`), in its three saved steps (`d-train --stages12 1`, `d-tune`, `d-finish`), one
effect at a time, one thread, `nice 15`, each from a frozen copy of the tool, resumable (the data root's
`fx_air/jobs`). Its branches decide the rest: the candle takes fire's settings (160 runs of 240 frames, 8 start points,
fine fields grown at instance start); heavy gas lingers as fog does, so it keeps 64-pixel fine fields with its 8 start
points (one new line in the recipe); the smoke ring plays once, so it takes the one-shot branch (240 runs of its 120
frames, 16 start points at its first frame with 64-pixel fine fields).

**A method of a later study, decided before any training or validation: the smoke ring gets study M3's mean state**
([REPORT.md](REPORT.md) §14.9). A one-shot played from one of 16 stored first states at another intensity stays the
size of its stored start (the explosion's and dust's fault, §14.1); the mean state moves each start point to the
requested controls by a quadratic fit of the training runs' first coarse states (`nvfx_ordered --study m3 fitmean
--block 2`, on this effect's own salt-1 training runs, under the data root's `fx_air`; `setmean` writes it into the
finished model), and study M3 found it better on held-out settings for both one-shots it was tried on and worse on
nothing. It is added after `d-finish` and before the model is frozen; the file that is frozen and tested is the one with
the mean state. Nothing else from studies D2, L, M, M2 or M3 is used: the ensemble-mean fine-tune helped fire and magic
but not steam or the jet (§14.8), the swirl at 16 pixels changes only the detail spectrum and helps or hurts with the
swirl's strength, and two detail passes are for flows faster than a coarse cell a frame, which these are not.

## 4. The test

**Written before any model was trained, and committed with it.** Each model is frozen by its SHA-256
(`results/experiments/fx_air_frozen.csv`) before its own test, and each test runs **once**, with `nvfx_experiment d-eval`
and `d-report` unchanged: study D's protocol ([REPORT.md](REPORT.md) §6.4 to §6.6), exactly as rounds 7 and 8 tested
their effects ([EFFECTS_BLAZE.md](EFFECTS_BLAZE.md) §4, [EFFECTS_FLOW.md](EFFECTS_FLOW.md) §4). The models train one at a
time on the shared machine, so the test runs in parts on this one design (`d-eval --effects candle`, then the others as
each is frozen), each into a folder of its own, the rows then merged into one set of CSVs under
`results/experiments/fx_air`. Every run of the test is independent and seeded by its effect, run and setting, so the
split changes no number.

- **Seeds: this round's own, 6,400,000 and up** (`--seed-base 6400000`; this round owns 6,400,000 to 6,799,999 and uses
  no other seeds in its test): the held-out tracking runs 6,400,000 + run; the real runs at the held-out settings
  6,500,000 + setting and 6,510,000 + setting; the neural run 6,520,000 + setting; the coarse simulation 6,530,000 +
  setting; the one-minute runs 6,540,000 + setting. Training and validation use the recipe's own seeds (600,000 + run,
  and those derived from the training runs for the detail constants), as every round's did.
- **Tracking from the true state:** 8 held-out runs per effect (salt 2's settings, which no training or validation run
  used), each started from its true state (coarse and fine fields; the candle and heavy gas after 100 frames, the smoke
  ring after its first frame) with its own noise seed: 240 frames (the smoke ring: to its last frame, 119). The neural
  rollout through the runtime against the simulation on the same 32-cell grid with the same detail layer
  (`coarse_sim_detail`), the plain coarse simulation, holding the first frame, and the neural dynamics drawn by the
  simulation's renderer. Active PSNR at 1 and 8 frames and at 1, 2 and 8 s (the smoke ring: 1 and 8 frames, 1 and 2 s;
  `d-eval`'s next horizon, 4 s, is past its last frame); the neural run minus `coarse_sim_detail`, paired over the 8
  runs.
- **Endless runs at settings never seen:** the ten held-out settings of study B (`b_test_settings`: none on study B's
  training grid), 10 s each after a 5 s warm-up of the real runs (the smoke ring: its 4 s from its start). Frame
  statistics against a real run: detail spectrum distance, motion ratio (as the error |ln ratio|), coverage distance and
  mean-frame PSNR. Compared with a second real run with another seed (the floor), the nearest of the 45 training
  flipbooks of study B's grid (made for these effects as study B made its library: 64 frames at 128 px, seed 1, looped;
  the smoke ring's are 64 frames held on the last, as the explosion's, since its 120 frames are less than twice 64), and
  the coarse simulation with the detail layer and without.
- **One minute:** a 60 s neural run in shards at the first two held-out settings, cut into 10 s windows, each against
  the real 10 s run (the candle and heavy gas; the smoke ring plays once).
- **Precision:** the runtime plays the effects as they ship, at its default precision, int8 since study K
  ([REPORT.md](REPORT.md) §13); `d-eval` sets none, so the `neural` rows are int8.
- **Statistics:** every difference paired (over runs or settings), with a 95% bootstrap interval of 10,000 resamples
  (`metrics::paired_bootstrap`); an interval across zero is a tie, reported as a tie. Every number is reported.

**The rule, decided now, as rounds 7 and 8 decided theirs.** An effect
- **works** if it tracks its runs better than `coarse_sim_detail` at 1 s (interval above zero) and, played endlessly,
  its average picture is closer than the nearest flipbook's (mean-frame PSNR interval above zero) and its motion error is
  no worse than the flipbook's (interval not above zero);
- is **weak** if it ties or loses to `coarse_sim_detail` at tracking at 1 s, or if, played endlessly, its average picture
  or its motion error loses to the flipbook's (interval on the worse side);
- otherwise **falls short**: it neither works nor is weak (as launch and ink did in round 7: tracking better, the average
  picture tied with the flipbook's).

For the smoke ring the question the study L lesson raises is answered beside the rule, not by it: whether the learned
ring holds its shape (two cores carrying the smoke, rising together) through its first two seconds, judged on the
test's first held-out setting by eye, from frames of the real run and the neural run side by side, and by the share of
its smoke in the left and the right half of the frame (a ring that holds its shape keeps it near a half each).
