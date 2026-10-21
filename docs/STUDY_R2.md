# Study R2: the 3D smoke again, on a teacher with its momentum fixed

Status: **under way** (21 October 2026). Audience: dev, research, the owner. A second try at study R's learned 3D smoke
([STUDY_R.md](STUDY_R.md)): first the teacher's momentum deficit (study R §1.4: the plume gains only 0.78 of the
momentum its buoyancy gives), found and fixed and the plume measured against Morton, Taylor and Turner again (§1); then
the learned smoke retrained on the fixed teacher with study N2's recipe and two losses aimed at study R's two failures
(the edge check and the over-spreading), under a new rule whose margins are calibrated on the teacher's own runs,
committed before validation and tested once (§2 onwards).

## 0. Where the study stands (a handoff, kept current)

As of 21 October 2026, 18:20 UTC: the code is written and unit-tested (commit 962c538: the momentum audit, the velocity
in flux form and `smoke_r2`, the trainer's spread and edge losses, the rule's margins and verdict); the teacher's
diagnosis (§1.2) waits for the quiet machine (another agent's timing round), as the brief asked.
- Data root `$NEURALVFX_DATA/r2_smoke3d` (default `/root/nvfx-data/r2_smoke3d`): job scripts in `jobs/`, logs in
  `logs/`, scratch in `scratch/`. Every job resumes from what it has written.
- Next: §1.2's runs (`jobs/diag.sh`, started by `jobs/wait_quiet.sh` when `quiet7/QUIET_DONE` appears or at 20:40 UTC).

## 1. The teacher's momentum

### 1.1 What study R found

Study R's teacher, `smoke_r` (study N's smoke without its sinks, heat and soot carried in flux form), is a plume by
MTT's conserved flux and its spreading rate, but **gains only 0.78 [0.74, 0.83] of the vertical momentum its buoyancy
gives** (dM/dz against the plane's buoyancy, heights 24 to 92, at every source strength), its flux-balance parameter
stays at 2.5 to 3.7 (a lazy plume) and its centreline velocity is about a third below MTT's for its flux. Study R put it
down to the semi-Lagrangian velocity step's numerical dissipation; MacCormack velocity, which would have helped, was
unstable at 64 x 96 x 64 (STUDY_R.md §1.2). Nothing measured which operator loses the momentum.

### 1.2 Where the momentum goes: a budget per operator

In a steady plume the vertical momentum of a slab between two heights is fixed: what the advection carries in through
its bottom and out through its top (the momentum flux M = the plane's sum of w^2, including the turbulent part), what
the buoyancy adds (the plane's sum of g' over the slab), what the pressure takes (the plane's sum of p at the top minus
at the bottom), and what the other forces add (vorticity confinement, the curl noise; no damping in `smoke_r`). So
dM/dz = B only if the pressure's part is small **and the advection conserves momentum**. `sim3d::MomentumAudit` sums each
operator's change of the slab's vertical momentum as the solver runs (it only reads: every effect's fields are the same
with it; tested), `nvfx_3d plume` records it with each plane's pressure, and `plumefit --momentum` gives each term as a
fraction of what the buoyancy adds, with intervals by resampling the runs.

Two runs of `smoke_r` (exploration seeds 34,000,000 and 34,000,001, intensity 0.5, wind 0.5, turbulence 0.5, 6 s of
spin-up and 16 s averaged, as study R's) and the same two of the candidate fix (§1.3). (Results: to come.)

### 1.3 The fix, and the criterion written before its answer

**The candidate**: the velocity carried in **flux form** (`Params::velocity` 2), each component exactly as the scalars'
flux form carries heat (second-order upwind with the monotonised central limiter, one sweep per axis in alternating
order, the face velocity the mean of its two cells', substeps where a sweep's Courant number would pass 0.95), by the
velocity after the forces, frozen for the step; the border is the velocity's own (outflow at the sides and the top, the
floor reflecting v, so no momentum crosses it). What leaves a cell enters its neighbour: the advection conserves
momentum (tested: its change of a slab's momentum is its faces' fluxes to float rounding). The limiter keeps the sweeps
total-variation diminishing, which is where MacCormack's correction (not conservative, and only clamped) went unstable.
It is a new preset, **`smoke_r2`**: `smoke_r` with this velocity step and nothing else changed. The fire, study N's
smoke and `smoke_r` are unchanged (their recorded runs compared byte for byte, §1.5).

**The criterion** (written 21 October, 18:20 UTC, before any run of the fix): `smoke_r2` becomes study R2's teacher if,
on §1.2's two exploration seeds, (a) its momentum budget closes where `smoke_r`'s does not: dM/dz over B within 0.90 to
1.10 (study R's criterion), **or**, if the audit shows the pressure takes a part, d(M + P)/dz over B within 0.90 to 1.10
with the advection's change within 5% of the momentum flux its planes carry; (b) its buoyancy flux is still conserved
(the planes' flux within 10% of its mean from 24 to 92 cells); and (c) it is stable: finite, with no speed growing
over 30 s, at four extreme settings (intensity 1, turbulence 1, no wind; intensity 0.8 with the wind at 0 and at 1;
intensity 0.2 with no turbulence). If it fails, the audit says which operator to look at next (the projection, or finer
time steps); every such try is reported. The final measurement (§1.6) uses seeds of its own, 34,010,000 + 100 c + k.

### 1.4 Stability

(To come.)

### 1.5 The other effects, unchanged

(To come: fire, study N's smoke and `smoke_r` recorded with the main branch's tool and with this branch's, compared byte
for byte.)

### 1.6 The plume against Morton, Taylor and Turner, again

(To come: study R §1.3's measurement and fits on `smoke_r2`, seeds 34,010,000 + 100 c + k.)

## 2. The learned smoke: design and rule (written and committed before validation)

(To come, before validation.)

## 7. Reproduce

(To come.)
