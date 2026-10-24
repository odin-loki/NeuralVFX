# Study R2: carrying the container's `r2_smoke3d` to another machine

Companion to [STUDY_R2.md](STUDY_R2.md) §0, the counterpart of [STUDY_Q2_RESTORE.md](STUDY_Q2_RESTORE.md) for the 3D
smoke. Study R2's scripts are in git (`tools/study_3d/r2_*.sh`) and run on any machine with a build; what is **not** in
git is the study's data tree, `/root/nvfx-data/r2_smoke3d` on the container: the 48 training runs (1.1 GB, about 8
CPU-hours to record again), the checkpoints and the candidates of a training run that took most of a day, the teacher
runs behind the rule's numbers and the plume measurement, and the markers that say which stage is done. Without it the
same commands start from nothing and redo all of that on the CPU (STUDY_R2.md §0, *Without the container's data*).
This page is the carry-over: what to copy, the rsync line, the checks, and the commands that follow.

**The PC as it is (24 October 2026).** WSL Ubuntu; its data root `~/nvfx-data` (`/home/odin/nvfx-data`) holds `hres_3d/`
(about 22 GB; STUDY_3D_HRES.md), `n2_3d/` (STUDY_3D.md), the empty `q2_fix/` of STUDY_Q2_RESTORE.md §3 and, if
`r2_bootstrap.sh` has been run here, an `r2_smoke3d/` with the empty folders it makes and a pinned tool under `bin/`
(nothing else: no runs, no checkpoints, no markers). That is the destination; rsync fills the empty folders. The scripts
touch nothing outside `$NEURALVFX_DATA/r2_smoke3d` (and the tables under the repository's `results/3d/r2`). `rsync`,
`sha256sum`, `pgrep`, `setsid` and `flock` are installed; the disk has about 740 GB free.

What the study needs, and where each comes from:

| what | where it comes from | copied? |
|---|---|---|
| the training runs, `runs/long/*.n3run`, `runs/short/*.n3run` (24 each, 1.1 GB) | the container; the same seeds give the same runs, so they *can* be recorded again (`r2_train.sh`, about 10 CPU-minutes a run) | **yes, §2** (saves 8 CPU-hours) |
| training's checkpoints `ckpt/` (`r2.ckpt`, `r2_stage1..4.ckpt`) and its log `logs/train_r2.log` | the container only (a training run on another machine is a different run: the same seed, another binary and thread count) | **yes, §2** |
| the candidates `models/C2.nvfx3`, `C3`, `C4` (and `chosen.nvfx3`, `chosen.txt` once validation has run) | the container only, written from its checkpoints | **yes, §2** |
| the stage markers, `markers/` | the container; they say what is done so the scripts do not redo it | **yes, §2** |
| the teacher runs: `results/edge_teacher_w*.csv` (the edge bounds'), `eval/cal/` (the margins'), `plume/<label>/seed_*.bin` (§1.6's) | the container; recomputable from the seeds, but hours of CPU | yes, §2 |
| validation's and the test's runs, `eval/val/`, `eval/test/`, `results/edge_val*.csv`, `edge_test*.csv` | the container, if those stages have run there | yes, §2 (whatever is there) |
| the rule's numbers and every table | git, `results/3d/r2/` (`margins.csv`, `edge_bounds.csv`, `edge_teacher.csv`, …) | no |
| the tool, `nvfx_3d` | built here (`cmake --build build --target nvfx_3d`); `r2_bootstrap.sh` pins a copy as `bin/nvfx_3d_<git hash>` | **no, never `bin/`** (§5) |
| the job logs, `logs/job_*.log`, `logs/bin.log` | the container; the record of what ran there | optional (small) |

## 1. The data root: `NEURALVFX_DATA`

Every script reads the data root from `NEURALVFX_DATA` (default `$HOME/nvfx-data`, the tool's own; the container's was
`/root/nvfx-data`). Study R2's tree is `$NEURALVFX_DATA/r2_smoke3d` (`D` below; `R2_ROOT` overrides the whole path).
Pick a folder with about 5 GB free (the container kept the tree under 2.5 GB, `R2_BUDGET_MB`; validation and the test
add their runs) and export it once per shell:

```
export NEURALVFX_DATA=$HOME/nvfx-data        # the PC: the root hres_3d and n2_3d already use; on WSL a Linux path, not /mnt/c (file locking, speed)
D=$NEURALVFX_DATA/r2_smoke3d
```

On the PC the repository is on the Windows drive (`/mnt/c/Users/odinl/OneDrive/Desktop/Neural VFX`) and the data root
on the Linux one; that is the intended split (the tables and the rule in git, the data outside). Run the scripts from
the repository: `cd '/mnt/c/Users/odinl/OneDrive/Desktop/Neural VFX'`. They are bash and use `sha256sum`, `stat -c`,
`pgrep`, `setsid`, `flock`, `df --output`, `du`: Linux (or WSL on the PC), not PowerShell. The other knobs, all
optional, are in `tools/study_3d/r2_env.sh`'s header (`T`, `J`, `NICE_LEVEL`, `BUILD`, `NVFX_3D`, `REPIN=1`,
`R2_ROOT`, `R2_BUDGET_MB`).

## 1a. The dry run: a checklist before anything is copied

In order, each with what it should say on the PC. Nothing before step 4 writes into `r2_smoke3d` but rsync's empty
destination folders (and step 2's, which are the same folders).

1. **The build.** `cmake --build build --target nvfx_3d` in WSL; `ls -la build/nvfx_3d` shows a Linux executable from
   today. (`r2_bootstrap.sh` pins a copy of it; a Windows build would not run under WSL.)
2. **`r2_bootstrap.sh`, before the copy.** `NEURALVFX_DATA=$HOME/nvfx-data bash tools/study_3d/r2_bootstrap.sh`. It
   makes the empty folders (the data root's own folder must already exist: a mistyped root stops it with exit 2 rather
   than grow a second, empty tree; `MKDATA=1` makes a new root on purpose), then says `data: NONE of the container's
   r2_smoke3d is here: no training run, checkpoint, trainer log, candidate, edge teacher run or plume run` and prints
   §2's rsync line with this machine's destination filled in (`HOST` to be replaced), pins the tool (`tool:
   …/bin/nvfx_3d -> …/bin/nvfx_3d_<hash> (SHA-256 …)`), and goes on with the stage lines: `training runs: 0 long, 0
   short of 24 each (… or copy the container's tree first …)`, `training: not started (no logs/train_r2.log)`,
   `candidate C2: not written` (C3, C4 likewise), `plume (§1.6): 0 of 18 runs`, `markers: ` (empty, or
   `rule_committed` alone: that one is made here from git, §3). Exit 0; it starts nothing. **Do not run `r2_resume.sh`
   yet**: on an empty tree it would start recording the 48 runs and training from scratch, which is exactly what the
   copy saves.
3. **The rsync, dry.** §2's line with `-n -i` added (what would be copied, nothing written). Expected: the itemized
   list with 48 `>f+++++++++ runs/{long,short}/*.n3run`, the `ckpt/*.ckpt` files, `logs/train_r2.log`, `models/*.nvfx3`,
   `markers/*`, `results/edge_teacher_w0..4.csv`, whatever `eval/` and `plume/` hold; no `bin/`, no `*.tmp`, no
   `*.lock`; a total of about 1.5 to 2.5 GB. A connection error here is ssh's: the host, the port or the key (`-e 'ssh
   -p PORT -i KEY'`), or the container is not reachable from the PC and §2's push or tarball is the way.
4. **The rsync, for real.** The same line without `-n -i`. Then the checks of §2: 48 runs, the candidates' SHA-256s
   against the container's, `du -sh`.
5. **`r2_bootstrap.sh` again.** Now: `data: the container's r2_smoke3d is here: runs/ (24 long, 24 short); ckpt/ (N);
   logs/train_r2.log; models/ (3); …` (or `data: PART of the container's r2_smoke3d is here: …` with a `not here:`
   line and the rsync line again, which is a copy cut short, or a stage the container had not reached), then
   `training runs: 24 long, 24 short of 24 each (complete)`, `training: DONE (logs/train_r2.log: done at iteration
   1750)` (or its last `it …` line if the container had not finished), the three `candidate C?: …/models/C?.nvfx3`
   lines, `candidates: done (marker)`, `edge bounds' teacher runs: 5 of 5`, `plume (§1.6): N of 18 runs`, `markers:
   candidates_done rule_committed …`. Exit 0. Still starts nothing.
6. **`r2_status.sh`**, one screen of the same; then **`r2_resume.sh`** (§4) starts the stage that is due.

## 2. The copy from the container: the tree, all but `bin/`

**When to copy.** The tree is safe to copy at any time as far as its files go: every run, unit, profile, model and
checkpoint is written whole (a temporary file renamed, so `--exclude='*.tmp'` leaves out the half-written one), and
the two kinds of file the tool appends line by line (`eval/*/track/*.csv`, `results/edge_*.csv`) are trimmed to whole
rows and whole 61-row edge runs by `r2_resume.sh`'s cleaning before anything reads them. **But do not run the same
stage on both machines.** If the container's training is still going when the tree is copied, the PC's `r2_resume.sh`
would resume it from the copied checkpoint, and the two would diverge (another binary, another thread count: the same
seed does not give the same run); likewise validation. Either copy after the container's stage has written its marker
(`markers/candidates_done`; `val_done` and `frozen`), or stop the container's jobs first (`pkill -f 'nvfx_3d.*--study
r2'`), copy, and go on from the PC alone. Copying for a look (step 3 of §1a, `r2_status.sh`) is harmless either way.

**By hand with rsync, one line** (the template, then the PC's instance; `rsync` is on both sides):

```
rsync -a --partial --info=progress2 --exclude='bin/' --exclude='*.tmp' --exclude='*.lock' [-e 'ssh -p PORT -i KEY'] USER@HOST:/root/nvfx-data/r2_smoke3d/ "$NEURALVFX_DATA/r2_smoke3d/"
```

```
rsync -a --partial --info=progress2 --exclude='bin/' --exclude='*.tmp' --exclude='*.lock' root@HOST:/root/nvfx-data/r2_smoke3d/ /home/odin/nvfx-data/r2_smoke3d/
```

Fill in: `USER@HOST`, the container's ssh login (`root@` and its address; the `[…]` part only when ssh needs a port or
a key, else leave it out); `/root/nvfx-data/r2_smoke3d/`, the container's tree (its data root is `/root/nvfx-data`);
the destination, this machine's `$NEURALVFX_DATA/r2_smoke3d/` (`/home/odin/nvfx-data/…` on the PC; `~` works,
`/mnt/c/…` does not, §1). Both trailing slashes stay (`r2_smoke3d/` into `r2_smoke3d/`: the contents, not an
`r2_smoke3d/r2_smoke3d`). Flags: `-a` keeps times and modes, `--partial` resumes a cut copy, `--info=progress2` shows
one progress line; `--exclude='bin/'` leaves the container's pinned tools (this machine pins its own, §5),
`--exclude='*.tmp'` a half-written file, `--exclude='*.lock'` the `flock` files under `logs/` (a lock copied from the
container means nothing here; the scripts make their own). Nothing is deleted on either side (no `--delete`); a
second run copies only what changed, so the line can be run again after the container has gone further. Add `-n -i`
for the dry run (§1a's step 3). The destination's folders are made by rsync, or by `r2_bootstrap.sh` run first.
(`r2_bootstrap.sh` prints this line itself, with the destination filled in, whenever the container's data is missing
or partial; its form leaves out the `*.tmp` and `*.lock` excludes, which only matter when the container's jobs are
still running, and `r2_resume.sh`'s cleaning deletes a copied `*.tmp` anyway.)

A smaller copy, when the disk or the link is short: the runs are 1.1 GB of the tree and are the one part that can be
remade from the seeds alone; everything else is small. `--exclude='runs/'` then, and `r2_train.sh` records them again
(about 8 CPU-hours at `J=8`) before it goes on; `--exclude='eval/' --exclude='plume/'` likewise leaves the teacher
runs to be redone (validation's `truth`/`real` runs and §1.6's plume runs are recomputed from their seeds; the margins
and the edge bounds themselves are in git and are not recomputed).

**By hand with scp** (no rsync on one side; `-p` keeps times, `-r` recurses; a cut copy starts over; `bin/` and the
locks come along and are removed after):

```
mkdir -p "$NEURALVFX_DATA"
scp -rp root@HOST:/root/nvfx-data/r2_smoke3d "$NEURALVFX_DATA/"
rm -rf "$D/bin"; find "$D" -name '*.tmp' -o -name '*.lock' | xargs -r rm -f
```

**From the container's side** (a push; when the PC can be reached from the container but not the other way, e.g. the
PC's WSL has `sshd` and a forwarded port, or a relay host sits between). On the container, `odin@PC` being the WSL
login at the PC's address:

```
rsync -an -i --exclude='bin/' --exclude='*.tmp' --exclude='*.lock' /root/nvfx-data/r2_smoke3d/ odin@PC:nvfx-data/r2_smoke3d/   # dry run
rsync -a --partial --info=progress2 --exclude='bin/' --exclude='*.tmp' --exclude='*.lock' /root/nvfx-data/r2_smoke3d/ odin@PC:nvfx-data/r2_smoke3d/
sha256sum /root/nvfx-data/r2_smoke3d/models/*.nvfx3 /root/nvfx-data/r2_smoke3d/ckpt/*.ckpt     # to compare with the PC's
```

(`nvfx-data/…` without a leading slash is relative to the login's home, `/home/odin/nvfx-data`. A relay: `rsync -a -e
'ssh -J relay' …`.)

If neither side can reach the other, the tree as a tarball from the container's side, moved however works (the
container's web console download, a bucket, a USB copy), then unpacked into the data root:

```
tar -C /root/nvfx-data --exclude='r2_smoke3d/bin' --exclude='*.tmp' --exclude='*.lock' -czf /tmp/r2_smoke3d.tgz r2_smoke3d   # on the container; near the tree's own size (the runs are fp16 and compress little)
tar -C "$NEURALVFX_DATA" -xzf r2_smoke3d.tgz                                                                                 # on the PC
```

**The tree after the copy** (what `r2_bootstrap.sh` expects; the layout of STUDY_R2.md §0, *Where the files are*):

```
$D/runs/long/*.n3run  (24)     $D/runs/short/*.n3run  (24)        the 48 training runs, 1.1 GB
$D/ckpt/r2.ckpt  r2_stage1..4.ckpt                                 training's checkpoints (stage snapshots as each stage ends)
$D/logs/train_r2.log                                               the trainer's log; "done at iteration 1750" is how training is known to be done
$D/models/C2.nvfx3  C3.nvfx3  C4.nvfx3  [chosen.nvfx3  chosen.txt]  the candidates (and the frozen choice, after validation)
$D/markers/{candidates_done, plume_done, rule_committed, val_done, frozen, frozen_committed, test_done}   whichever the container had made
$D/results/edge_teacher_w0..4.csv  [edge_val_C?.csv edge_val.csv edge_test.csv]   the raw edge tables
$D/eval/cal/  [eval/val/  eval/test/]                              the margins' 45 teacher runs; validation's and the test's runs if run
$D/plume/v_i50 v_i20 v_i80 v_t00 v_t100/seed_*.bin                 §1.6's runs (18 when complete)
$D/logs/job_*.log                                                  the container's job logs, for the record
```

**The checks**, on both sides (the same numbers and the same 64 hex characters per file):

```
ls "$D"/runs/long/*.n3run "$D"/runs/short/*.n3run | wc -l            # 48
sha256sum "$D"/models/*.nvfx3 "$D"/ckpt/*.ckpt                        # against the container's
grep -c '^it ' "$D/logs/train_r2.log"; tail -1 "$D/logs/train_r2.log" # the same last line as the container's
ls "$D/markers"; du -sh "$D"
```

`r2_validate.sh` freezes the chosen candidate by its SHA-256 (`results/3d/r2/frozen.csv`) and the tool refuses the
test on any other bytes, so a candidate that did not copy whole cannot be tested; but it would spend a validation.
Compare the three before `r2_resume.sh`.

## 3. `r2_bootstrap.sh`: the folders, the tool and what is there

```
cmake --build build --target nvfx_3d                                   # the tool, if not built (the default BUILD)
NEURALVFX_DATA=$HOME/nvfx-data bash tools/study_3d/r2_bootstrap.sh     # before and after the copy; starts nothing
```

Stops (exit 2, nothing made) if the data root's folder itself does not exist (a mistyped `NEURALVFX_DATA` or `R2_ROOT`
must not grow a second, empty tree elsewhere; `MKDATA=1` makes a new root on purpose). Otherwise makes
`$D/{logs,markers,models,results,runs,ckpt,bin,plume}`, says which of the container's stages are here at all
(`data: NONE …` with §2's rsync line and what starting from nothing costs; `data: PART …` with what is not here and
the same line, so a cut copy is not mistaken for a fresh start; `data: the container's r2_smoke3d is here: …`), pins
the tool (`bin/nvfx_3d -> nvfx_3d_<git hash>`, a copy of `build/nvfx_3d`; `REPIN=1` refreshes it; §5), reports the
disk, checks that the rule's five files are in
`results/3d/r2` and makes `markers/rule_committed` when git holds `margins.csv` and `edge_bounds.csv` unchanged (the
one marker made on this machine rather than copied: it says the rule was committed before validation, which is true
of the repository wherever it is checked out), then says what is there of each stage: the runs (`N long, N short of 24
each`), training (`DONE` by the log's last line, or the last `it` line and the checkpoints present), each candidate,
the `candidates_done` marker, the five edge-teacher files (whole at 244 rows each), the plume runs, the markers, and
the frozen choice if there is one (and whether `models/chosen.nvfx3`, `chosen.txt` and `frozen.csv` agree). It makes
no number and starts no job; exit 0 data or no data (an empty tree is a legitimate start); safe to run again at any
time. There is no dry mode: what it writes is the empty folders, the tool's copy and the rule's marker, none of which
the copy minds.

Two things it says that are not errors: `data: NONE of the container's r2_smoke3d is here` with `training: not started
(no logs/train_r2.log)` and `candidate C?: not written`, before the copy (that is the PC's empty tree, §1a step 2);
and `tool: … (SHA-256 …)` with a hash different from the container's (every build but the container's; §5). A `tool:
MISSING` is step 1 of §1a; a `rule: … is MISSING` is a repository file gone (`git checkout`), not the data's.

Checked 24 October 2026, midday (WSL): `bash -n` passes on all `tools/study_3d/r2_*.sh` and `hres_smoke_r2*.sh`; `r2_bootstrap.sh` with a nonexistent `NEURALVFX_DATA` printed the "does not exist; nothing made" message and **exit 2** (no folder made), and against a throwaway root under `/tmp` (`NVFX_3D` pointing at `build/nvfx_3d`, so nothing pinned) printed `data: NONE …`, made only the eight empty folders and `markers/rule_committed`, **exit 0**; `~/nvfx-data/r2_smoke3d` untouched.

## 4. `r2_resume.sh`, `r2_status.sh`, and the hand steps

```
NEURALVFX_DATA=... bash tools/study_3d/r2_resume.sh      # starts the stage that is due, detached; again after any restart
NEURALVFX_DATA=... bash tools/study_3d/r2_status.sh      # one screen, at any time
```

`r2_resume.sh` first cleans what a kill could have cut (the appended tables trimmed to whole rows and whole edge runs,
`*.tmp` deleted), then runs the stages in order, each behind the one before, starting the long ones detached (`setsid
nohup`, logs `$D/logs/job_<name>.log`) and leaving a running job alone: training and the candidates
(`markers/candidates_done`: skipped if the copy brought it), the plume runs and their fits (`plume_done`), the rule's
marker, validation with the choice and the freeze (`val_done`, `frozen`; `r2_validate.sh`), the test (`test_done`;
`r2_test.sh`). With the container's tree copied after its training finished, the first thing it starts here is
validation (and the rest of the plume runs beside it). The hand steps, as STUDY_R2.md §0 says: after validation,
commit and push `results/3d/r2/{choice_val.md,frozen.csv,edge_val.csv,edgecheck_val.md,compare_val_*}`, `touch
$D/markers/frozen_committed`, run `r2_resume.sh` again for the test; after the test, commit `verdict_test.md` and the
test's tables, and ship only on a pass (§2.9 of the study).

A copied tree whose stage was cut mid-way (the container restarted, or its jobs were stopped for the copy) resumes the
same way here: training from `ckpt/r2.ckpt` (a checkpoint every 10 iterations; the losses reproduce exactly on the
same binary, STUDY_R2.md §0, and only approximately on another), validation and the test from the runs already
written (the tool skips a run file that exists). Resuming the container's training on this machine's binary is the one
case where the copy and a fresh run differ: say so in the write-up (the binary's hash is in `logs/bin.log` and
`logs/job_train.log`'s first line).

## 5. The binary and its hash

The container pinned two builds of the same solver and trainer, `cb31ccb` for training and the teacher's runs and
`a5637ed` for validation and the test (STUDY_R2.md §0). A build on another machine will have a different hash
(`r2_bootstrap.sh` prints it; `bin/nvfx_3d_<git hash>` names it by the commit, `-dirty` if the solver's sources are
modified), and that is allowed: every job logs the binary it ran with (`logs/bin.log`, each `job_*.log`'s first line),
the frozen row records the chosen file's bytes, and the validation and test numbers depend on the seeds and the files,
not on the thread count or the order of the runs. What changes is the provenance, which is then said in STUDY_R2.md §0.
To run the container's very bytes instead, copy its `bin/nvfx_3d_a5637ed` too (not by §2's line, which excludes
`bin/`; Linux x86-64 only), put it under `$D/bin/` and point the link at it (`ln -sfn nvfx_3d_a5637ed $D/bin/nvfx_3d`),
or set `NVFX_3D=/path/to/nvfx_3d_a5637ed` for every script; `r2_bootstrap.sh` then reports that hash.
