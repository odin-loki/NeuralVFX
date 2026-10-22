# The accelerated version: every effect as a small seeded generator

**Status:** a design, written 22 October 2026 from the owner's idea:
- training only has to extract each effect's probability distribution;
- compression loss can be used as noise;
- most features (the owner's estimate: 60%) don't matter;
- somewhere there is a small, fast, RNG-like generator for all these effects.

Nothing in it has started. It changes how [PLAN_PC_GPU.md](PLAN_PC_GPU.md) trains, and its first bets can be tested in
2D in this container (§8).

## 0. In short

- **The object we are making is a generator, not a predictor.** An effect is:
  - its controls (a few numbers);
  - a seed (64 bits);
  - a small code that says which effect it is;
  - one shared network per family that turns these into frames, the way an RNG turns a seed into numbers.

  The project's own measurements say this is what an effect is (§1).
- **Five rules make it fast.** Each is a kind of information the training should not pay for (§2):
  1. Don't learn what's known: physics stays as code.
  2. Don't learn what's random: seeded noise with the right statistics.
  3. Don't learn below the noise floor: an error smaller than a second real run's difference is invisible, so it is
     free to spend on compression, pruning, coarser teachers and early stopping.
  4. Don't learn anything twice: one network per family, one small-scale generator for all effects, symmetries and
     scaling laws.
  5. Don't compute what isn't there: empty space, insignificant channels, still time, settings already known.
- **The pattern behind the five rules:** spend bits and FLOPs only on the information the effect really carries,
  beyond what physics, universality and its own noise give for free. The ideal generator is the shortest program that
  reproduces the effect's distribution within the noise floor. Training is the search for that program; acceleration
  is searching a smaller space.
- **What it buys, if the bets hold** (each is measured and gated, §6):
  - a new effect is a **code fit** of a few hundred numbers against a frozen family network: minutes of GPU, not hours;
  - teachers are coarser and fewer;
  - the network is a fraction of the FLOPs.

  An effect would take about **1 to 2 hours** end to end, most of it the measuring, against PLAN_PC_GPU.md's 5 to 9.
  All 33 would take about **1 to 2 weeks** of the PC, against 4 to 6, plus building the GPU solvers.
- **What can't be accelerated** (§7):
  - the noise floor itself;
  - the number of held-out runs a fair test needs;
  - the GPU's arithmetic.

  The small-scale random detail is already close to maxed out: it is seeded noise today.

## 1. What an effect is: the evidence

| finding | where | what it says |
|---|---|---|
| A start point nudged by 1e-3 stays at 62 dB after 1 s; the same start with another seed is at 13 dB, the level of two unrelated runs | REPORT §6.1 (study D0) | after about a second **the seed decides what is seen**, not the state: the effect is a seeded random process, and a stored effect needs start points and seeds, not frames |
| Taken apart, the model's fluctuations have the real runs' RMS within 3%; what is wrong is the average picture and the random part's scale and timing | REPORT §14.7 (study M) | the random part is easy and nearly right; **the ordered part is where the information is** |
| Noise added to the stepper's inputs broke the flame up | REPORT §14.7 | noise helps only with the effect's own statistics: **unshaped noise is not free** |
| 8-bit integers for the stepper: every measure a tie or far inside its margin | REPORT §13.8 (study K) | **compression error below the noise floor is invisible**: the owner's "compression loss as noise", measured |
| An authored run stored as coarse corrections over the stepper: 50 bytes to 2.5 KB | REPORT §12 (study G's codec) | a whole run's information, given the dynamics, is **a few hundred bytes** |
| A one-step denoise every 16 frames stops drift; it ties the 6 s shards | REPORT §12 (study G) | a learned prior can hold an endless generator on its distribution |
| The 3D models fail on edges and over-spreading, not on detail (N2, R) | STUDY_3D.md §14, STUDY_R.md | the 3D failures are ordered, large-scale and boundary errors |

So the generator is:

```
state(t+1) = F(state(t), controls, effect_code, noise(seed, t))     the learned, ordered part, small and coarse
picture(t) = Optics(state(t) + Detail(state(t), seed, t), camera)   seeded detail and exact optics: no learning
```

Here `F` is physics as code plus a small learned closure. Its information is the family network (shared), the effect
code (hundreds of numbers), the controls and the seed.

## 2. The five rules, and what each does to training

### 2.1 Don't learn what's known: physics as code

Advection, pressure projection, buoyancy, conservation and the optics (Planck's law, CIE 1931, absorption,
scattering) are code, as they are today. The network learns only the **closure**: what the coarse grid can't
resolve, i.e. the sub-grid forcing that turbulence models supply in large-eddy simulation.
- A closure is small, local and smooth.
- It generalises across effects and controls, because the physics around it carries the effect-specific part.
- It is stable: conservation is in the code, not learned.

**Speed:** fewer parameters, shorter training, fewer runs needed.

**Already partly so:** the 3D stepper advects and projects (TODO "The models"). New: make the network's output a
forcing, bounded and zero-mean where physics says it must be.

### 2.2 Don't learn what's random: the distribution, not the trajectory

Beyond the chaos horizon (about 1 s, §1) a run can't be predicted, only its distribution can.
- **Training:** tracking losses only up to the horizon, which is about 30 frames, so gradient windows of at most 30
  frames, not 64 to 128. Beyond it, losses on **statistics** of free-running rollouts, with no gradient through the
  burn-in:
  - the mean volume;
  - the 3D spectrum by scale;
  - the temporal correlation by scale;
  - coverage and motion;
  - the edge profile.

  This is study D2's statistics loss, applied where the evidence says it belongs.
- **Data:** a run beyond its first second contributes its statistics, not its frames, so most of a run is kept as
  **sufficient statistics**: running means, spectra and correlations per region. Frames are kept for the first second
  after each start point.
- **Speed:** windows 2 to 4 times shorter, and kept data about 10 times smaller.

### 2.3 Don't learn below the noise floor: compression error as noise, shaped

**The noise floor** is how far a second real run with another seed is from the first, measured by the null probe the
house rules already run. Any error smaller than that cannot be detected by the test, by construction of the rule's
margins. So it is a budget, to be spent:
- **Precision:**
  - int8 is free (study K);
  - try int4 and 2:4 sparsity;
  - teacher data at 16 bits or less, as far as the floor allows.
- **Shape the compression error.** Make it look like the effect's own fluctuations: the same spectrum and timing, as
  dithering does in audio. Then it adds to the random part instead of breaking structure (study M's lesson: unshaped
  noise broke the flame).
- **Teachers:** the coarse dynamics need a teacher only about twice as fine as the learned state. That is 128³ for a
  64³ state, not 256³: 8 times fewer cells and about 16 times less simulation time. The small-scale statistics come
  from a few fine runs per family, made once (§2.4).
- **Stopping:** training stops when validation reaches the floor. Past it, nothing is measurable.
- **Pruning:** a channel, a layer, a memory slot or a control is removed if removing it moves no measure past its
  margin. That is the owner's "60% of features don't matter", made testable.
- **Speed:** fewer cells to simulate, fewer bits to store, fewer channels to train and play.

### 2.4 Don't learn anything twice: shared, symmetric, scaled

- **One network per family, and a code per effect.** The family network is trained once. Each effect is a learned
  vector of about 64 to 512 numbers (FiLM on every layer). **A new effect fits only its code**, with the network
  frozen, as a decoder that is told only "which one". That is fewer than 1/1000 of the parameters, so it converges in
  far fewer windows. Codes also interpolate: between fire and smoke, or towards a designed effect.
- **One small-scale generator for all effects.** Small scales of turbulence are close to universal: Kolmogorov's
  spectrum, the same intermittency. So the seeded detail is **one generator**, conditioned on the local large-scale
  state (energy and dissipation, temperature, scalar gradients). It is trained once from the fine runs of all families
  and shared. Procedural methods of this kind exist and are the "small, fast, RNG-like generator" for the small scales:
  wavelet turbulence (Kim et al., 2008) and curl noise (Bridson et al., 2007). Ours is the seeded detail layer, which
  study D2 tuned.
- **Symmetries:**
  - rotation about the vertical by 90° and mirror images give 8 copies of every run for free, where gravity is the only
    preferred direction (wind breaks some, and is rotated with them);
  - equivariant convolutions, or simply the 8-fold augmentation, mean fewer runs.
- **Scaling laws:** fires and plumes collapse onto a few dimensionless numbers.
  - Q*, the Froude-scaled heat release: flame height ∝ Q*^(2/5), MTT's self-similar plumes.
  - Wind over plume velocity.
  - Richardson for dense flows; Weber and Froude for liquids; fireball size ∝ mass^(1/3).

  Train on the dimensionless form, and a control space of four dimensions (size, fuel, wind, temperature) becomes one
  of two. That cuts the runs needed to cover it by several times, and sizes come out right everywhere.
- **Speed:** a per-effect fit in minutes; fewer runs per effect; the detail generator and the optics trained or written
  once.

### 2.5 Don't compute what isn't there

- **Space:** sparse bricks: only cells with material, typically 30 to 40% of a box. This is in the plan for runtime, and
  should be used in training too.
- **Channels:** structured pruning (§2.3). Keeping 40% of the channels keeps 16% of the multiply-adds, because the cost
  goes with channels squared.
- **Convolutions:**
  - 3 x 3 x 3 split into three 1-D passes costs a third;
  - depthwise-separable costs about a twentieth at 64 channels.

  Each is a candidate, measured against the full one.
- **Time:** large scales change slowly, so the learned step runs at 10 to 15 Hz with interpolation, while the seeded
  detail and the optics run every frame. That halves to thirds the step's cost at runtime and in training.
- **Settings:** active sampling. Simulate 20 settings, train, then simulate more only where the model is furthest from
  its teacher, instead of a fixed grid of 100. The pipeline already logs where it misses.
- **Speed:** each is a few times; together they multiply.

## 3. The pattern behind the rules

Each rule removes one source of information the training was paying for twice:

| what's removed | given instead by |
|---|---|
| what physics already knows | code (rule 1) |
| what can't be predicted | the seed (rule 2) |
| what can't be measured | the noise floor (rule 3) |
| what other effects, scales and symmetries already taught | sharing (rule 4) |
| where nothing happens | sparsity (rule 5) |

What's left is the effect's own information: its code, its controls and its seed. It is small. The study G codec found
a whole authored run is a few hundred bytes given the dynamics.

That is the minimum-description-length view of an effect:

```
cost of an effect  =  physics (shared, code)  +  family network (shared, trained once)
                    +  small-scale generator (shared, trained once)  +  effect code (hundreds of numbers)
                    +  controls and seed (per instance)
```

**The accelerated version is the one that pays only the last two lines per effect.** Training a new effect is then a
search over hundreds of numbers, not millions, inside a space whose physics, randomness and noise floor are already
fixed.

The same pattern applies to the pattern itself: each rule is information that is shared, random, invisible or absent,
found once and reused. When none of the five applies to some part of the cost, that part is at its floor (§7).

## 4. The accelerated pipeline

Per family, once:
1. **The noise floor first.** A null probe on the family's teacher (runs with other seeds) sets the margins, the
   precision budget, the stopping point and the pruning threshold for everything after.
2. **The teacher at twice the learned state's resolution** (128³ for a 64³ state), checked against measurements as
   PLAN_PC_GPU.md §3.4 asks. A few fine runs (256³) per family feed the shared small-scale generator.
3. **Dimensionless controls and the 8 symmetries.** Runs are placed over the collapsed control space by active
   sampling.
4. **The family network:** physics plus a closure, depthwise-separable or split convolutions and sparse bricks.
   - It is trained on short windows (to the horizon) and statistics beyond.
   - Its training includes noise-shaped int8 or int4 compression error.
   - It stops at the floor, then is pruned to the floor.

Per effect:
1. About 30 runs at the teacher's resolution: active sampling over its collapsed controls, times 8 symmetries.
2. **A code fit:** the network frozen; a few hundred numbers fitted with short windows and statistics. If the code
   alone can't reach the floor, a small fine-tune of the last layers, measured as such.
3. Measured by the house rules: the null probe, validation, freeze, the one test.
4. Shipped: the code (bytes), the shared network (once per family), videos.

## 5. Estimates, if each bet holds

Rough factors against PLAN_PC_GPU.md's per-effect hours. They multiply only where they act on the same cost. Each is
an estimate until measured.

| bet | acts on | estimate | the evidence so far |
|---|---|---|---|
| teacher at 2x the state, not 4x | simulation | 8 to 16x less | the closure needs the resolved scales only; to be checked per family |
| about 30 runs, not 100 (dimensionless, symmetric, active) | simulation, data | about 3x less | none yet |
| sufficient statistics past the first second | data | about 10x less | D0's horizon; D2's losses |
| code fit, not fine-tune | per-effect training | 5 to 20x less | none yet: the first bet (§8) |
| windows to the horizon, not 64 to 128 | training | 2 to 4x less | D0's horizon; study L's 30-frame unroll worked |
| channels pruned to the floor (keeping about 40%) | training, runtime | up to 6x less | study K: int8 free; pruning not tried |
| separable or split convolutions | training, runtime | 3 to 20x less | not tried; may cost quality |
| the step at 10 to 15 Hz | training, runtime | 2 to 3x less | not tried |
| int4 with shaped error | runtime, memory | about 2x less | int8 measured free; int4 not tried |

With the plausible half of these holding:
- **per effect:** about 1 to 2 hours, most of it the simulating at 128³ (minutes), the code fit (15 to 60 min) and the
  measuring (about an hour, which statistics fixes, §7);
- **per family:** the network in about half a day to a day;
- **runtime:** a hero effect's GPU step a fraction of PLAN_PC_GPU.md's 1 to 3 ms;
- **all 33 effects:** about 1 to 2 weeks of the PC, plus building the GPU solvers.

## 6. Each bet, as a test

Every bet is tested under the house rules, like any study:
- the rule committed before validation;
- held-out settings and fresh seeds;
- the one test;
- paired bootstrap 95% intervals.

The rule for a speed bet is that quality stays within the null probe's margins of the slower way, on every measure,
and the time falls by at least a stated factor. A bet that costs quality past the floor is dropped, and the estimate
above is struck out with its numbers.

## 7. What can't be accelerated: the floors

- **The noise floor.** Once a model's error is a second real run's, more training measures nothing. The rules above
  get there sooner; they don't go past it.
- **The test's statistics.** A fair verdict needs enough held-out settings for the power the null probe shows. That is
  the measuring hour. It can be cut only by variance reduction, and the tests already pair seeds and settings (common
  random numbers). It is the one cost that stays per effect.
- **The seed's information.** An instance's randomness is 64 bits, and the generator can't be smaller than physics plus
  its code.
- **The arithmetic.** At the floor of FLOPs per frame, a faster effect needs a smaller state or a faster GPU.
- **Already near its floor:**
  - the small-scale detail, which is seeded noise today;
  - the optics, which is exact physics;
  - int8 on CPUs with VNNI;
  - float on AVX-512.

  Little is left to win there, except sharing the detail generator across effects (§2.4).

## 8. The first bets, testable here in 2D

They need no PC and no GPU, and the 2D effects are cheap. Each would be a study with its own committed rule:
1. **Codes, not models** (the biggest bet). For the fire family in 2D (fire, wildfire, crown fire, window fire,
   candle), does one shared network with a per-effect code match each effect's own model within the margins? And how
   fast does a new effect's code fit, against training its model?
2. **Pruned to the floor.** How many of the 33 shipped models' channels can go before any measure moves past its
   margin? This tests the owner's 60%.
3. **Shaped compression error.** int4 weights with error shaped to the effect's spectrum, against plain int4 and int8.
4. **Statistics past the horizon.** Training on windows of at most 30 frames plus statistics losses, against today's
   recipe, at equal quality: how much faster?
5. **Dimensionless fire.** One fire model on Q* and the wind ratio across sizes, against per-size data, with fewer runs.

These are not queued. They wait for the owner's word, as the plan does.
