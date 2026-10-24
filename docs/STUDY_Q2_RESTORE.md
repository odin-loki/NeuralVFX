# Study Q2: running the last three validations on another machine

Companion to [STUDY_Q2.md](STUDY_Q2.md) §0. The geyser's, heavy gas's and the mushroom's validations, freezes and
tests are still to run (no validation row in the committed tables as of 24 October 2026). The scripts that run them are
in git (`tools/study_q2/`), but the **candidates** (§4.1's `u30a`, `u30al`, `ms`, `msg`, `s24`, `u60a`, `u90a`) exist
only on the container, under `/root/nvfx-data/q2_fix/cand/<effect>/`; the owner's PC has none. This page is the
restore: what to copy from the container, where to put it, and the commands that follow.

**The PC as it is (24 October 2026).** WSL Ubuntu; its data root `~/nvfx-data` (`/home/odin/nvfx-data`) holds the 3D
studies' folders only: `hres_3d/` (about 22 GB: `fire/`, `smoke/`, `results/`; STUDY_3D_HRES.md) and `n2_3d/`
(`models/`; STUDY_3D.md), plus their logs. Nothing of study Q2's: no candidates, no caches, and none of the
container's `fx_*`/`q_fix` folders (the shipped files then come from git's `models/2d`, which is fine, below); the only
`q2_fix/` is the empty one a `bootstrap.sh` run made that morning (its folders and the tool's copy in `bin/`, 4 MB; §3
says what it reports). `q2_fix` goes next to `hres_3d` and `n2_3d` under the same root; the scripts touch nothing
outside `$NEURALVFX_DATA/q2_fix`. `rsync`, `sha256sum`, `pgrep` and `setsid` are installed; the disk has about 740 GB
free.

What is needed for the validations, and where each comes from:

| what | where it comes from | copied? |
|---|---|---|
| the candidates, `q2_fix/cand/<effect>/<name>.nvfx` | the container only (the fine-tunes and `starts` need its training runs) | **yes, §2** |
| the statistics caches, `q2_fix/vt/` (`real/`, `plays/<sha16>/`, `realp/`, `playp/`) | the container; keyed by the seeds and each file's SHA-256, so a copy is reused as is | optional, §2 (they only save recomputing; a run without them gives the same numbers, §6 of the study) |
| the shipped files (the current models) | git's `models/2d/geyser.nvfx`, `heavygas.nvfx`, `mushroom.nvfx` are the shipped files to the byte (§4.1's hashes); `common.sh` falls back to them when the data root has no `fx_*`/`q_fix` copy | no |
| the rule and the tables | git, `results/experiments/q2_fix/` (`q2_rule.csv`, `q2_val_choice.csv`, `q2_frozen.csv`, …) | no |
| the tool, `nvfx_ordered` | built here (`cmake --build build --target nvfx_ordered`); `bootstrap.sh` copies it to `q2_fix/bin/nvfx_ordered_q2` and compares its SHA-256 with §4.6's | no (§5) |

## 0. The whole restore in four lines (fill in `CONTAINER`)

`CONTAINER` is the one placeholder: the container's ssh address as `ssh` takes it (`root@HOST`, or a `Host` alias
from `~/.ssh/config`; a port or a key goes to rsync's `-e 'ssh -p PORT -i KEY'`, the script's `RSYNC_OPTS`). The
source is the container's
`/root/nvfx-data/q2_fix`, the destination this machine's `$NEURALVFX_DATA/q2_fix`; the copy takes `cand/` and `vt/`
only (the rest is in git or built here, §2). In WSL, from the repository:

```
export NEURALVFX_DATA=$HOME/nvfx-data                                   # the PC's data root (hres_3d and n2_3d sit there already)
mkdir -p "$NEURALVFX_DATA/q2_fix"                                       # rsync makes the last folder only, not the root
rsync -a --partial --info=progress2 --exclude='*.tmp' --include='/cand/***' --include='/vt/***' --exclude='*' \
      CONTAINER:/root/nvfx-data/q2_fix/  "$NEURALVFX_DATA/q2_fix/"      # add -n -i first to see the list; drop --include='/vt/***' for cand/ only
bash tools/study_q2/bootstrap.sh                                        # must end: ready: yes; everything reads ok (§3a); then resume.sh
```

The same copy through the script, which is the dry run until `APPLY=1` (§2): `RSYNC_SRC=CONTAINER:/root/nvfx-data/q2_fix
bash tools/study_q2/restore_from.sh`, then with `APPLY=1` (and `Q2_RESTORE_VT=1` for `vt/`). Either way the trailing
slashes matter (`q2_fix/` to `q2_fix/` copies the contents, not a `q2_fix/q2_fix`), nothing of this machine's is
deleted (no `--delete`), and a half-written `*.tmp` is left behind. Then `sha256sum "$NEURALVFX_DATA"/q2_fix/cand/*/*.nvfx`
here and `sha256sum /root/nvfx-data/q2_fix/cand/*/*.nvfx` on the container must agree on all seven (§2's list).

Checked on 24 October 2026 (WSL, rsync 3.2.7, against a throwaway `q2_fix` in `/tmp` with every folder of the
container's layout and a `*.tmp` in `cand/` and in `vt/`): the filter copied `cand/**` and `vt/**`, skipped both
`*.tmp`, and took nothing from `jobs/`, `logs/`, `bin/`, `models/`, `res/` or `scratch/`; with `-n -i` the list is
`cd+++++++++ ./`, `cd+++++++++ cand/`, one `cd+++++++++` per effect, one `>f+++++++++` per candidate, then `vt/`'s.
Without the `mkdir -p` rsync fails with `mkdir "…/q2_fix" failed: No such file or directory` when the data root itself
is missing. Everything downstream of the copy is checked in §3a.

## 1. The data root: `NEURALVFX_DATA`

Every script reads the data root from `NEURALVFX_DATA` (default `/root/nvfx-data`, the container's). Study Q2's files
are under `$NEURALVFX_DATA/q2_fix` (`Q` below). Pick a folder on this machine with about 2 GB free (the study's limit
for `q2_fix` is 1.5 GB) and export it once per shell:

```
export NEURALVFX_DATA=$HOME/nvfx-data        # the PC: the root hres_3d and n2_3d already use; on WSL a Linux path, not /mnt/c (file locking, speed)
Q=$NEURALVFX_DATA/q2_fix
```

On the PC the repository is on the Windows drive (`/mnt/c/Users/odinl/OneDrive/Desktop/Neural VFX`) and the data
root on the Linux one; that is the intended split (the tables and the models in git, the data outside). Run the
scripts from the repository: `cd '/mnt/c/Users/odinl/OneDrive/Desktop/Neural VFX'`.

The scripts are bash and use `sha256sum`, `stat -c`, `pgrep`, `setsid`, `df`, `du`, `date -d`: Linux (or WSL on the
PC), not PowerShell. Other knobs, all optional (`tools/study_q2/common.sh`): `Q2_REPO` (this worktree; the default is
the repository the scripts are in), `Q2_BIN` (the tool; default `$Q/bin/nvfx_ordered_q2`), `Q2_BUILD_BIN` (where
`bootstrap.sh` copies the tool from; default `$Q2_REPO/build/nvfx_ordered`), `Q2_DRY=1` (`bootstrap.sh` checks and
writes nothing, §1a), `Q2_THREADS` (default 2), `Q2_NO_PUSH=1` (freeze without committing).

## 1a. The dry run: a checklist before anything is copied

Both scripts have a mode that writes nothing: `Q2_DRY=1 bootstrap.sh` (check only) and `restore_from.sh` without
`APPLY=1` (rsync's `--dry-run --itemize-changes`). In order, each with what it should say on the PC:

1. **The build.** `cmake --build build --target nvfx_ordered` in WSL; `ls -la build/nvfx_ordered` shows a Linux
   executable from today. (`bootstrap.sh` copies it; a Windows build would not run under WSL.)
2. **`bootstrap.sh`, dry.** `NEURALVFX_DATA=$HOME/nvfx-data Q2_DRY=1 bash tools/study_q2/bootstrap.sh`. Expected:
   `data root: also holds hres_3d n2_3d (other studies'; untouched …)`; `disk: … MB free` (hundreds of thousands);
   `q2_fix: … is absent; would be made` (or `exists`); `tool: … would copy …/build/nvfx_ordered` (or the two
   `NOT §4.6's binary` lines, §5); `rule: … q2_rule.csv ok (7 lines)`; three effect lines each with `current ok
   (git's …/models/2d/<effect>.nvfx; the data root has no fx_*/q_fix copy, which is fine)` and `candidates MISSING:
   …`; `vt: … is empty or absent (fine …)`; and the checklist `ready: no; 1 to do` naming the candidates with the
   exact `restore_from.sh` line to run. Exit 1. Anything else in the checklist (a missing data root, a missing build,
   a missing rule, `/mnt/c` as the data root, a tool not installed) is fixed first.
3. **`restore_from.sh`, dry.** `RSYNC_SRC=root@HOST:/root/nvfx-data/q2_fix NEURALVFX_DATA=$HOME/nvfx-data bash
   tools/study_q2/restore_from.sh`. Expected: `restore_from: … -> …/q2_fix; cand; DRY RUN (set APPLY=1 to copy)`, then
   rsync's itemized list: possibly a `.d..t...... ./` line (the folder's time), three `cd+++++++++ <effect>/` lines
   and exactly seven `>f+++++++++ <effect>/<name>.nvfx` lines (§2's list; no `*.tmp`), a total size of a few MB (each candidate is the shipped file with one change: about
   150 to 300 KB, like `models/2d`'s), and `(DRY RUN)` on the last line. A connection error here is ssh's: the host,
   the port or the key (`RSYNC_OPTS="ssh -p PORT -i KEY"`), or the container is not reachable from the PC and §2's
   push or tarball is the way.
4. **`restore_from.sh`, for real.** The same line with `APPLY=1` (add `Q2_RESTORE_VT=1` for `vt/`). It ends with the
   seven candidates' SHA-256s and sizes; compare with `sha256sum /root/nvfx-data/q2_fix/cand/*/*.nvfx` on the
   container. All seven must match before going on.
5. **`bootstrap.sh`, dry again**, then **for real.** Dry: the effect lines now read `candidates all present: u30a(…
   bytes) …` and the checklist `ready: yes`. Then without `Q2_DRY`: the folders, the tool's copy, `ready: yes;
   everything reads ok`, `next: … resume.sh`. Exit 0.
6. **`resume.sh`** (§4) starts the three validations. Nothing before this step runs the tool or writes a number.

The dry run leaves no trace: `ls -A $NEURALVFX_DATA` still shows only `hres_3d`, `n2_3d` and the logs (checked on a
copy of that layout on 24 October: `Q2_DRY=1 bootstrap.sh` made nothing; `restore_from.sh` without `APPLY=1` made the
empty `q2_fix/cand/` only, which is rsync's destination and harmless).

## 2. The copy from the container: `cand/` (and `vt/`)

Only `cand/` is needed; `vt/` is worth taking if the container is reachable, since the three validations had already
cached part of their settings before the last restart (§0 of the study). Neither `jobs/`, `logs/`, `res/`, `scratch/`
nor `bin/` is wanted: the jobs are in git, the tables are in git, and the binary is built here (§5).

**With the script** (`tools/study_q2/restore_from.sh`; rsync, a dry run until `APPLY=1`; nothing of this machine's is
deleted, `*.tmp` half-written files are skipped):

```
RSYNC_SRC=root@HOST:/root/nvfx-data/q2_fix tools/study_q2/restore_from.sh                     # dry run: what would be copied
RSYNC_SRC=root@HOST:/root/nvfx-data/q2_fix APPLY=1 tools/study_q2/restore_from.sh             # cand/ only
RSYNC_SRC=root@HOST:/root/nvfx-data/q2_fix APPLY=1 Q2_RESTORE_VT=1 tools/study_q2/restore_from.sh   # cand/ and vt/
RSYNC_OPTS="ssh -p 2222 -i ~/.ssh/container" ...                                              # a port or a key
```

`RSYNC_SRC` can also be a local folder (a mounted backup, a USB copy): rsync then copies without ssh. When done with
`APPLY=1` the script lists every candidate's SHA-256 and size (compare with the container's `sha256sum
/root/nvfx-data/q2_fix/cand/*/*.nvfx`), then says to run `bootstrap.sh`.

**By hand with rsync, one line** (the same copy the script makes; this is also the line `bootstrap.sh` prints in its
checklist when candidates are missing, with the PC's paths filled in). The template, then the PC's instance:

```
rsync -a --partial --info=progress2 --exclude='*.tmp' [-e 'ssh -p PORT -i KEY'] USER@HOST:/root/nvfx-data/q2_fix/cand/ "$NEURALVFX_DATA/q2_fix/cand/"
```

```
rsync -a --partial --info=progress2 --exclude='*.tmp' root@HOST:/root/nvfx-data/q2_fix/cand/ /home/odin/nvfx-data/q2_fix/cand/
```

Fill in: `USER@HOST`, the container's ssh login (`root@` and its address; the one line in `[…]` only when ssh needs a
port or a key, else leave it out); `/root/nvfx-data/q2_fix/cand/`, the container's candidates (its data root is the
default `/root/nvfx-data`); the destination, this machine's `$NEURALVFX_DATA/q2_fix/cand/` (`/home/odin/nvfx-data/…` on
the PC; `~` works, `/mnt/c/…` does not, §1). Both trailing slashes stay (`cand/` into `cand/`: the contents, not a
`cand/cand`). Flags: `-a` keeps times and modes, `--partial` resumes a cut copy, `--exclude='*.tmp'` leaves out a
half-written file, `--info=progress2` shows one progress line; nothing here is deleted (no `--delete`). Add `-n` for
the dry run (what would be copied, nothing written; with `-i` the itemized list of §1a's step 3). The destination's
folders are made by rsync, or by `bootstrap.sh` run first. `vt/` is the same line with `vt/` for `cand/` on both
sides, optional:

```
rsync -a --partial --info=progress2 --exclude='*.tmp' root@HOST:/root/nvfx-data/q2_fix/vt/ /home/odin/nvfx-data/q2_fix/vt/     # optional
```

**By hand with scp** (no rsync on either side; `-p` keeps times, `-r` recurses; a cut copy starts over):

```
mkdir -p "$Q"
scp -rp root@HOST:/root/nvfx-data/q2_fix/cand "$Q/"
scp -rp root@HOST:/root/nvfx-data/q2_fix/vt   "$Q/"      # optional; then: find "$Q/vt" -name '*.tmp' -delete
```

**From the container's side** (a push; when the PC can be reached from the container but not the other way, e.g.
the PC's WSL has `sshd` and a forwarded port, or a relay host sits between). Run on the container, `odin@PC` being the
WSL login at the PC's address and the destination the PC's data root:

```
rsync -avn --exclude='*.tmp' /root/nvfx-data/q2_fix/cand/ odin@PC:nvfx-data/q2_fix/cand/        # dry run (-n): the seven files
rsync -a --partial --info=progress2 --exclude='*.tmp' /root/nvfx-data/q2_fix/cand/ odin@PC:nvfx-data/q2_fix/cand/
rsync -a --partial --info=progress2 --exclude='*.tmp' /root/nvfx-data/q2_fix/vt/   odin@PC:nvfx-data/q2_fix/vt/     # optional
sha256sum /root/nvfx-data/q2_fix/cand/*/*.nvfx                                                     # to compare with the PC's
```

(`nvfx-data/…` without a leading slash is relative to the login's home, `/home/odin/nvfx-data`; the folders on the PC
are made by rsync, or by `bootstrap.sh` run first.) The trailing slashes matter on both sides: `cand/` to `cand/`
copies the contents, not a `cand/cand`. A relay: `rsync -a -e 'ssh -J relay' …`.

If neither side can reach the other, the same two folders as a tarball from the container's side
(`tar -C /root/nvfx-data/q2_fix -czf /tmp/q2_cand.tgz cand` and, optionally, `vt`), moved however works (the
container's web console download, a bucket, a USB copy), then `tar -C "$Q" -xzf q2_cand.tgz` on the PC.

The candidates after the copy (what `bootstrap.sh` expects; §4.1 of the study; the haboob's, the ring's and the
avalanche's are replaced and not needed):

```
$Q/cand/geyser/u30a.nvfx    $Q/cand/geyser/u30al.nvfx
$Q/cand/heavygas/ms.nvfx    $Q/cand/heavygas/msg.nvfx    $Q/cand/heavygas/s24.nvfx
$Q/cand/mushroom/u60a.nvfx  $Q/cand/mushroom/u90a.nvfx
```

Check them against the container (`sha256sum "$Q"/cand/*/*.nvfx` on both sides; the same 64 hex characters per file).
`after_val.sh` checks the chosen one again against validation's row before freezing it, so a wrong copy cannot be
frozen or tested, but it would spend a validation.

## 3. `bootstrap.sh`: the folders, the tool and the state

```
cmake --build build --target nvfx_ordered                 # the tool, if not built (the default Q2_BUILD_BIN)
NEURALVFX_DATA=$HOME/nvfx-data Q2_DRY=1 tools/study_q2/bootstrap.sh    # dry run: check only, write nothing
NEURALVFX_DATA=$HOME/nvfx-data tools/study_q2/bootstrap.sh             # make and check
```

Makes `$Q/{jobs,logs,bin,models,vt,res,scratch,cand/<effect>}`, copies the tool to `$Q/bin/nvfx_ordered_q2` if it is
not there and says whether its SHA-256 is §4.6's (§5 below), checks the rule's table, and for each of the three effects
prints its shipped file (ok, by §4.1's hash) its candidates (all present with their sizes, or which are missing) and
where it stands (at validation; validated, next `after_val.sh`; tested). Its sections, in order: `the data root`
(what else the root holds, e.g. `hres_3d n2_3d`; the free disk; a warning for a `/mnt/c` root or a missing Linux
tool), `the folders`, `the tool`, `the tables`, `the effects` (and the `vt/` cache's count) and `checklist`: `ready:
yes` with the next command, or `ready: no; N to do, in order:` with each open item said in full. When candidates are
missing the item lists them (`candidates: 7 missing under …/q2_fix/cand/`, then the seven `cand/<effect>/<name>.nvfx`
paths), the two `restore_from.sh` lines (dry, then `APPLY=1`) and §2's one-line rsync with this machine's destination
filled in, and points to §2 for the no-ssh ways; `HOST` in those lines is the container's address, to be replaced.
Exit 0 when ready, 1 when not; 2 when `q2_fix` itself cannot be made (the message then says which: the data root
does not exist, is not writable by this login, or something else, with `mkdir`'s own error above it). With `Q2_DRY=1`
nothing is written: each line says what *would* be made or copied. Makes no number; safe to run again at any time.

Two things it says that are not errors: `current ok (git's …/models/2d/<effect>.nvfx; the data root has no fx_*/q_fix
copy, which is fine)`, the PC's case (the container keeps a copy of each shipped file under `fx_*`/`q_fix`; git's is
the same bytes, checked by §4.1's hash); and `tool: … is NOT §4.6's binary`, every build but the container's (§5).
A third, `tool: the build at …/build/nvfx_ordered differs from $Q/bin/nvfx_ordered_q2`, means the tree was rebuilt
after the copy: `bootstrap.sh` never replaces a copied tool (one binary per study, §4.6), so either `rm
"$Q/bin/nvfx_ordered_q2"` and run it again to take the new build, or leave it; `val.sh` and `after_val.sh` log the
hash they run with either way. (The PC on 24 October: the copy in `bin/` is from a 10:52 build, the tree was rebuilt at
11:43; since no validation has run yet, taking the new build costs nothing.)

What the PC printed on 24 October, dry, before any copy: `data root: also holds hres_3d n2_3d`, `757300 MB free`,
`q2_fix … exists (4 MB)` (the earlier run's empty folders and the tool), the two tool lines above, `rule … ok (7
lines)`, the three effects `current ok (git's …)`, `candidates MISSING: u30a u30al` / `ms msg s24` / `u60a u90a`, all
`at validation (not done)`, `vt … is empty`, and `ready: no; 1 to do`: the candidates (§2). Exit 1. Run again later
that morning, not dry: the same, with the data root now also holding `r2_smoke3d` (STUDY_R2.md's; untouched), the
folders already there (`exists (4 MB)`), and the one open item in its longer form above, ending in the PC's rsync
line: `rsync -a --partial --info=progress2 --exclude='*.tmp' root@HOST:/root/nvfx-data/q2_fix/cand/
/home/odin/nvfx-data/q2_fix/cand/`. Exit 1; `ls -A ~/nvfx-data/q2_fix/cand/*` still empty.

## 3a. `bootstrap.sh`: the success criteria, verified

`bootstrap.sh` has succeeded when **all** of these hold; each is a line it prints (the wording is the script's,
`tools/study_q2/bootstrap.sh`) or its exit code. Anything else is an open item: it is printed as `  - …` where it is
found and again, numbered, under `## checklist`.

| criterion | the line, when it holds | when it does not |
|---|---|---|
| exit code | `0` | `1` (`ready: no`); `2` when `$Q` cannot be made (mkdir's error and the reason: no data root, not writable, disk) |
| the checklist | `ready: yes; everything reads ok` | `ready: no; N to do, in order:` then `1. …`, and `then: … bootstrap.sh again until it reads ready: yes` |
| the next step | after a real run: `next: NEURALVFX_DATA=… tools/study_q2/resume.sh (starts the validations that are not done; after_val.sh is by hand, §0)`; after a dry run with the folders or the tool still to make: `next: … bootstrap.sh (without Q2_DRY) to make the folders and copy the tool, then resume.sh` | — |
| the data root | `disk: N MB free at …` with N at least 2048 | `disk: … has only N MB free`; `data root: … does not exist`; `data root: … is a Windows drive mounted in WSL` (`/mnt/[a-z]/…`) |
| the Linux tools | nothing printed | `tool: <sha256sum\|pgrep\|setsid\|df\|du> is not installed` (`rsync: not installed` is only a remark) |
| the folders | `q2_fix: … exists (N MB)` (or, dry, `is absent; would be made (…)`; real, `is absent; making it`) | exit 2, above |
| the tool | `tool: … is §4.6's binary (…)` **or** `tool: … is NOT §4.6's binary (SHA-256 …): allowed; …` — both pass (§5); dry: `tool: … is missing; would copy …/build/nvfx_ordered (SHA-256 …)`; real: `tool: copied … to …/bin/nvfx_ordered_q2` | `tool: … is missing and so is …/build/nvfx_ordered (build it: …)` |
| the rule | `rule: …/q2_rule.csv ok (7 lines)` | `rule: … is missing (…; git checkout results/experiments/q2_fix)` |
| each of `geyser`, `heavygas`, `mushroom` | `<effect>: current ok (…); candidates all present: <name>(N bytes) …; <state>`; `current ok (git's …/models/2d/<effect>.nvfx; the data root has no fx_*/q_fix copy, which is fine)` passes; `<state>` is `at validation (not done)` before any run | `current MISSING: …` (no shipped file, or not §4.1's hash); `candidates MISSING: <names>` followed by the `candidates: N missing under …/cand/` item with the exact `restore_from.sh` and `rsync` lines (§0, §2) |
| `vt/` | either `vt: … is empty or absent (fine: the validations recompute; …)` or `vt: … has N cached statistics files (reused as they are; …)` | — (never an open item) |

Two lines that look like warnings and are not: `tool: … is NOT §4.6's binary` (every build but the container's, §5) and
`tool: the build at …/build/nvfx_ordered differs from …/bin/nvfx_ordered_q2` (the tree was rebuilt after the copy, §3).
Neither stops `ready: yes`.

**Verified on 24 October 2026** (the PC, WSL; the repository on `/mnt/c`, `build/nvfx_ordered` from 11:53; every
run below against a throwaway data root under `/tmp` holding empty `hres_3d/` and `n2_3d/`, so that `~/nvfx-data`
was not touched; the whole sequence takes a few seconds):

1. Dry, no `q2_fix`, no candidates: `data root: also holds hres_3d n2_3d`, `q2_fix: … is absent; would be made`,
   `tool: … is missing; would copy …`, `rule … ok (7 lines)`, three `current ok (git's …)` + `candidates MISSING`
   lines, `vt … empty or absent`, `ready: no; 1 to do` (the candidates), exit 1. `ls -A` of the root afterwards:
   `hres_3d n2_3d`, nothing made.
2. Seven placeholder files put under `q2_fix/cand/<effect>/` (200 000 bytes each; the real ones are the container's),
   then dry: the effect lines read `candidates all present: u30a(200000 bytes) u30al(200000 bytes)` etc.,
   `ready: yes; everything reads ok`, `next: … bootstrap.sh (without Q2_DRY) to make the folders and copy the tool,
   then resume.sh`, exit 0.
3. For real: `tool: copied …/build/nvfx_ordered to …/bin/nvfx_ordered_q2`, `tool: … is NOT §4.6's binary (… allowed
   …)`, `ready: yes; everything reads ok`, `next: … resume.sh (…)`, exit 0; `q2_fix/` then holds
   `bin cand jobs logs models res scratch vt`.
4. For real, again: the same `ready: yes` and `next: … resume.sh`, exit 0 (it is safe to run again).
5. `restore_from.sh` without `APPLY=1` from a local folder holding the same `cand/` plus a `*.tmp`: the itemized
   list of §1a step 3 (`.d..t...... ./`, three `cd+++++++++`, seven `>f+++++++++`, no `*.tmp`), `(DRY RUN)`, exit 0;
   the destination afterwards holds only the empty `q2_fix/cand/`.

The same dry run against the real `~/nvfx-data` that day printed §3's "what the PC printed" (the root also holding
`r2_smoke3d`): `ready: no; 1 to do`, the seven candidates, exit 1. So on the PC exactly one thing stands between the
checklist and `ready: yes`: §0's copy.

Re-checked 24 October 2026, midday (WSL): `bash -n` passes on all six `tools/study_q2/*.sh`; `Q2_DRY=1 bootstrap.sh` against the real `~/nvfx-data` printed §3's lines and `ready: no; 1 to do` (the seven candidates), **exit 1**, the data root's listing unchanged.

## 4. `resume.sh`, then `after_val.sh` per effect

```
NEURALVFX_DATA=... tools/study_q2/resume.sh
```

The cleaning of §0 (caches written in the ten minutes before this machine's boot, `*.tmp`, a stale git lock, a test
cut off before its result), then one `val.sh <effect>` per effect not done, each detached (`setsid nohup`, 2 threads,
`Q2_THREADS`), logs in `$Q/logs/val_<effect>.log`. A validation is 60 settings; its tracking runs are not cached and
start over, its statistics are (`$Q/vt`, so a copied `vt/` is used). Each writes `q2_val_*.csv` and its `chosen` row to
`results/experiments/q2_fix/`, then `$Q/jobs/val_<effect>.done`. Run `resume.sh` again after any interruption; it
leaves a running job alone. Then, **by hand**, when `$Q/jobs/val_<effect>.done` exists:

```
git add results/experiments/q2_fix && git commit -m "Study Q2: the <effect>'s validation" && git push   # the tables first (§4.6)
NEURALVFX_DATA=... tools/study_q2/after_val.sh geyser      # likewise heavygas, mushroom, each once
```

`after_val.sh` reads the `chosen` row (`none` ends the effect as a null, nothing frozen), checks the chosen file's
SHA-256 against validation's row, copies it read-only to `$Q/models/<effect>.nvfx`, appends its row to `q2_frozen.csv`
(bytes, SHA-256, the binary's SHA-256), commits and pushes that one file (or stops with `Q2_NO_PUSH=1` for a push by
hand, then run it again), and runs the test once; the marker `$Q/test_<effect>.done` says `done <sha> met|not met`.
Commit and push `results/experiments/q2_fix` (`q2_test_*.csv`) after it. If `met`, the script prints the ship steps
(the links, `models/2d`, `SHA256SUMS`, the frozen rows, videos, the write-up: §0 of the study).

## 5. The binary and its SHA-256

§4.6 of the study: the probe, validation and the test ran one binary, `q2_fix/bin/nvfx_ordered_q2`, SHA-256
`420131ac5e3455943401bda68199660191558cacbc8deb568a48bbe551dd265a` (`BIN_SHA_Q2` in `common.sh`). A build on another
machine will almost surely have a different hash (compiler, flags, libraries), and that is allowed: `bootstrap.sh`
says so (`tool: ... is NOT §4.6's binary`), `val.sh` and `after_val.sh` log the hash they ran with, and the frozen row
records whichever binary tested the file. The numbers depend only on the seeds and the files (§6 of the study: every run
and play is its own job; thread count and order change nothing), so the result is the same; what changes is the
provenance, which is then said in STUDY_Q2.md §0 and in `q2_frozen.csv`'s last field. To run the very same bytes
instead, copy the container's `q2_fix/bin/nvfx_ordered_q2` too (not done by `restore_from.sh`; Linux x86-64 only) and
check it: `sha256sum "$Q/bin/nvfx_ordered_q2"` must print the hash above. A copied `vt/` is keyed by the seeds and
the files' hashes, not the binary's (`vt/real/<effect>_<seed>`, `vt/plays/<sha16>/...`), so a build of the same
source reads it; if the tool's statistics code has changed since §4.6's commit, delete `$Q/vt` and let the validations
recompute.
