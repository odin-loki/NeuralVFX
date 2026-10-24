#!/usr/bin/env bash
# Ship wave 1: the three commits of COMMIT_PLAN.md (24 October 2026) and its pre-push checklist, as one script.
#
# What it does, in order (COMMIT_PLAN.md is the authority; this file only carries it out, and its lists must match
# the plan's "Files" blocks):
#   0. checks, read-only, in both modes: repo root, branch main, every file the plan lists exists, no file listed
#      twice, nothing changed in the tree (`git status -uall`) that the plan does not place, every .sh the plan
#      stages is LF and passes `bash -n`, no file but library/dust.nvfxl declares the look `brownout` (the plan's
#      note under commit 2), git can commit (a committer identity, no merge or rebase under way). A dry run reports
#      every blocker and still prints the plan; a live run stops before its first git command. A missing identity is
#      a blocker for a live run only: a dry run commits nothing, so it says so as a note and passes (the plan's
#      pre-push checklist, "Git identity", says what the owner sets before DRY_RUN=0).
#   1. `git reset -q` ("Before anything"): the index unstaged, the working tree untouched.
#   2. commit 1 (runtime and API), commit 2 (the weather director and the library), commit 3 (the studies'
#      bookkeeping, the scripts and TODO; this script goes in it beside the plan): for each, `git add` of the plan's
#      file list; `git update-index --chmod=+x` on the scripts the plan says must be 100755 (tools/study_q2/*.sh,
#      tools/study_3d/r2_*.sh and the new bash scripts; this checkout has core.filemode false, so the mode is fixed in
#      the index, not on disk; the other hres scripts stay 100644 as in HEAD); the staged diff checked for a
#      whole-file rewrite by line endings (checklist item 6: a staged copy that became CRLF is a fault and stops the
#      run; one that became LF, time_crowd.sh's, is reported and wanted); then `git commit` with the plan's message,
#      subject and body.
#   3. `git log --oneline -3` and `git status --short` (checklist item 8).
#   4. the pre-push checklist, from the build directory: configure and build (item 1), `bash tools/preflight_wave1.sh`
#      (item 2, optional: PREFLIGHT=1), `ctest -j1 --output-on-failure` (item 3), `./nvfx_alloc_test` (item 4),
#      `./nvfx_gallery --library ../library --check` (item 5). Item 7 (the Godot test) is not run: it is reported as
#      skipped with its command.
#
# Modes:
#   DRY_RUN=1 (the default): prints every git command and every checklist command it would run, runs the read-only
#       checks of step 0, changes nothing (no reset, no add, no commit, no build).
#   DRY_RUN=0 FORCE=1: does it. Both are needed: DRY_RUN=0 alone is refused before anything is touched. No
#       `git commit` happens on any other setting. This script never runs `git push`.
#
# Usage (repo root, or anywhere: it cds to the repo):
#   bash tools/ship_wave1.sh                        # dry run: the plan as commands
#   DRY_RUN=0 FORCE=1 bash tools/ship_wave1.sh      # reset, three commits, then the checklist
#   DRY_RUN=0 FORCE=1 CHECKS=0 bash tools/ship_wave1.sh   # the commits only
#   DRY_RUN=0 FORCE=1 COMMITS=0 bash tools/ship_wave1.sh  # the checklist only, on the state as it stands
# Env: BUILD (the build directory, default build), BUILD_TYPE (Release), GENERATOR (Ninja), CHECKS (1; 0 skips step
#   4), COMMITS (1; 0 skips steps 1 to 3), PREFLIGHT (0; 1 runs tools/preflight_wave1.sh as checklist item 2),
#   ALLOW_UNLISTED (0; 1 lets a changed path the plan does not place stay in the tree, uncommitted), ALLOW_CRLF (0; 1
#   commits past a staged diff that looks like a CRLF rewrite), NVFX_REPO (the checkout, default: this file's parent's
#   parent).
#
# Failures: a blocker in step 0 stops the script before any git command. A failure in step 2 leaves the commits made
# so far in place (the index holds the partly staged next commit; `git reset -q` undoes the staging, `git reset
# --soft HEAD~N` the commits). A failure in step 4 leaves all three commits in place and says so: fix, amend, re-run
# with COMMITS=0. Nothing is pushed by this script in any case.
#
# After any edit: bash -n tools/ship_wave1.sh
set -euo pipefail

# ---- the plan's three commits: messages (subject, then body) and file lists -----------------------------------------

SUBJ1='Manager v3 (half size per effect), volume jobs (NVFX_VERSION 10), coupled 3D fire and smoke (sim3d::Stack), CRLF-safe verify readers'
BODY1=$(cat <<'EOF'
- GNU Affero General Public License v3 (`LICENSE`; README Licence section), as the other Desktop repos.
- nvfx_manager_desc.half_px < 0: the half-size default per effect (NVFX_MANAGER_VERSION 3; MANAGER.md §2.10, §6;
  REPORT §22.11). The FUSE adapter passes it through; time_crowd.sh pins §7's baseline at --half-px 0. (MANAGER.md
  §7.2 and REPORT §22.8 cite tools/manager/time_half_128.sh, the quick quiet re-timing; the script is commit 3's.)
- nvfx_volume_create_with_jobs (NVFX_VERSION 10; ENGINES.md §12, INTEGRATION.md §3, FUSE.md §9): every parallel loop
  of a step, read or picture as one fork-join on the engine. VolumeApi.SameFieldsAndPicturesOnAnEnginesJobSystem,
  nvfx_alloc_test.
- sim3d::Stack and its Seam (STUDY_R.md §6.1): two boxes seamed top to bottom, heat, soot and velocity crossing both
  ways; nvfx_3d video --kind coupled-sim (--upper smoke_r2 takes study R2's teacher). The volume-API coupled demo rule
  in include/neuralfx/coupled3d.hpp, tested in tests/test_coupled3d.cpp (part of neuralfx_3d_tests); ctest gains
  Volume3dToolUsage and Coupled3dVideoSmoke (the latter only where ffmpeg is found at configure time).
- nvfx_3d --effect smoke_r2 with --study hres (STUDY_3D_HRES.md §10): study R2's teacher on the finer coarse grid, its
  own seed blocks (training 9,630,000 + i, long 9,631,000 + i, the bounds' teacher runs 9,633,000 + 100 j + k), the
  plume source's peaks shared with study R's smoke; smoke_r refused for hres.
- nvfx_gallery and nvfx_scene_script --verify readers strip \r: a CRLF checkout of frozen.csv had failed every hash.
- docs: ENGINES and REPORT also carry the weather director's sky.* keys and the study hres addendum (commits 2 and 3)
  rather than split a Markdown table by hand.
EOF
)

SUBJ2='Weather director: humidity, dew point, cloud base, wet ground, the storm moving with the wind, the sun and its shafts, the cloud deck (WEATHER.md §3.11 to §3.15); library: steam fog (#61), contrail (#36), helicopter downwash (#37), whiteout (#54), drifting snow (#75)'
BODY2=$(cat <<'EOF'
- wx::saturation_vapour_hpa, dew_point, cloud_base_m (Magnus with Alduchov and Eskridge's coefficients; Espy's rule);
  wx::evaporation_mm_h, wind_2m, kDepressionStorage (Penman's drying power, 1.5 mm storage). Read from scripts as
  sky.humidity, sky.dewpoint, sky.base, sky.water, sky.wet, sky.puddle, sky.evaporation (§3.11, §3.12).
- The storm across the scene (§3.13, Taylor's hypothesis): wx::taylor_length_m; sky.carried, sky.front.
- The sun and its shafts (§3.14): wx::sun_elevation_deg, sun_across (45 N on the equinox, Iqbal), sun_share (Kasten
  and Czeplak's diffuse fraction), beam_transmittance (Beer-Lambert along the slant path through a 300 m layer),
  shaft_visibility; sky.hour, sky.elevation, sky.sunx, sky.sun, sky.shaft; the `hour` key of `weather`
  (script.cpp's kWeather, script_run.cpp's set_hour each frame).
- The cloud deck (§3.15): wx::deck_cover (the cover as a layer, a smoothstep from 4 to 6 oktas: a ceiling in
  aviation's sense) and deck_foot_deg (where Koschmieder's haze hides the underside, arcsin(min(1, base / visibility)));
  sky.deck, sky.foot.
- The frame's wet, puddle, deck and foot (compose.hpp; the `frame` keys of script.cpp, evaluated each frame in
  script_run.cpp; COMPOSE.md §4.9.1), each 0 by default so every scene before is the same to the bit, drawn by
  compose.cpp's background pass: wet darkens the ground rows and the scene's light on them by lerp(1, 0.6, wet)
  before the haze (§4.11, Lekner and Dorf), puddle fills the ground texture's hollows deepest first and mirrors the
  sky and the hills by Fresnel's reflectance at the depression angle (Schlick), deck lays a layer's underside over
  the dome whose cover per row is deck times wx::deck_contrast along the slant path (§5.4: the CIE overcast grey with
  value-noise cells, lit by the flash, hiding the stars), foot where that contrast is 2 %. tests/test_script.cpp:
  Frame.WetDarkensTheGroundAndNothingElse, PuddlesMirrorTheSkyAndTheHillsByFresnel, TheDeckCoversTheDomeByItsContrast,
  TheWetGroundAndTheDeckAreTheSameOnAnyNumberOfThreads. examples/scenes/weather_day.nvfxs wires the four to sky.wet,
  sky.puddle, sky.deck, sky.foot (moves `ScriptedWeatherDayFrozen`: re-freeze
  `results/experiments/weather_frozen.csv` once on a machine with the library models; skipped exit 77 on this PC until
  then; `deck_drift` not drawn).
- The trees' sway by the Beaufort force, wx::trees (§3.6, a hook); sky.trees.
- script_run.cpp's "a weather has no ..." message and COMPOSE.md's table list all of it. tests/test_weather_director.cpp
  (39): WeatherRelations.TheSunAndItsShafts, TheCloudDeck, TheDeckFootHidesInDust;
  WeatherDay.TheFrameCarriesTheWetGroundAndTheCloudDeck among them;
  WeatherDirector.TheStormMovesAcrossTheSceneWithTheWind, TheSunStandsByTheHourAndShowsThroughBrokenCloud among them.
- library: steam_fog (steam.nvfxl, catalogue #61) and contrail (trails.nvfxl, #36); coloured_smoke_grenade
  (smoke.nvfxl) adjusted; helicopter_downwash (dust.nvfxl, #37: the dust model's bursts taking turns under the smoke
  model, the looks brownout and brownout_haze); whiteout (storms.nvfxl, #54: the blizzard's snow, a saltation layer
  with a threshold, the fog model in two storeys) and drifting_snow (storms.nvfxl, #75, procedural: two bands of the
  compositor's snow under a gust field); earth.nvfxl's pyroclastic_flow comment and about (#110: built from the
  avalanche model, not learned). The catalogue at 240 entries, 173 learned, --check clean. steam_fog's and the
  grenade's keyframes frozen in results/library/frozen.csv (judged with the models), drifting_snow's in
  frozen_round3.csv (procedural, no models needed); contrail, the downwash and the whiteout not yet judged by eye
  (written on a machine without the models) and so not frozen and in no --only list (LIBRARY.md §4, §10.4).
  Library.SteamFogIsFogAndSteamOverAPool, ContrailIsAColdTrailSetBackBehindTheJet,
  HelicopterDownwashIsDustTakingTurnsUnderAHazeInTheWind, BlizzardAndDriftingSnowAreBlownSnow,
  WhiteoutIsABlizzardWhoseBlownSnowHasAThreshold.
EOF
)

SUBJ3="Studies hres, Q2 and R2: fire test tables, smoke_r2-at-hres drivers, CUDA spread/edge hybrid, PLAN_PC_GPU; Q2 and R2 jobs; wave 1 checklist and ship scripts"
BODY3=$(cat <<'EOF'
- hres (STUDY_3D_HRES.md §7, §8): the fire's test tables under results/3d/hres/fire/ (compare_test_*, edge_test*.csv,
  edgecheck_test.md judged with the calibrated bounds, the single-run table as edgecheck_test_single.md);
  hres_fire_recheck_test_edges.sh and hres_sync_fire_test_results.sh new; hres_fire_validate.sh with its body in
  main(), run_fire_validate.sh on a snapshot copy, hres_status.sh reporting the test tables, restart_hres_watchers.sh
  idempotent.
- hres, smoke on study R2's teacher (STUDY_3D_HRES.md §10): hres_smoke_r2.sh (record, train with --spread 2 --edge 10,
  TRAIN_DEVICE=cuda, finish, bounds, edge; hres_smoke_r2_resume_cuda.sh and hres_smoke_r2_train.sh); hres_status.sh
  reports smoke_r2's runs, training, models, bounds and edge check.
- docs/PLAN_PC_GPU.md and docs/WAVE1_CHECKLIST.md: stage 2 on the PC (one reference per effect, FUSE on Desktop, no
  footage); results/3d/pc.md (P0 snapshot).
- Q2's jobs as scripts (tools/study_q2/; STUDY_Q2.md §0, STUDY_Q2_RESTORE.md new); bootstrap.sh as a checklist with
  a dry run (Q2_DRY=1).
- R2's jobs as scripts (tools/study_3d/r2_*.sh; STUDY_R2.md §0, §7; STUDY_R2_RESTORE.md new). CUDA trainer: MSE on
  GPU, spread and edge on CPU in the same step (`rollout3d_train_cuda_host.cpp`, `rollout3d_train.hpp`).
- .gitignore: CTest's Testing/ directories. .gitattributes (new): *.sh and *.awk text eol=lf, so the scripts check
  out LF whatever core.autocrlf says; the 16 scripts still CRLF in the index outside these commits are renormalized in
  a separate endings-only commit (checklist item 6).
- TODO.md: "Done since 24 October 2026" (the whiteout and drifting snow among them, the fire's test written up),
  "Under way", "Next", the pointer to docs/WAVE1_CHECKLIST.md under "Order of work". docs/WAVE1_CHECKLIST.md (new: the
  per-effect ledger of wave 1, what the container has done and what the PC owes), COMMIT_PLAN.md (this plan),
  tools/ship_wave1.sh (its git adds as a script; DRY_RUN=1 by default) and tools/preflight_wave1.sh (the fast
  pre-push check; no git state touched).
EOF
)

FILES1=(
  CMakeLists.txt
  LICENSE
  README.md
  docs/ENGINES.md
  docs/FUSE.md
  docs/INTEGRATION.md
  docs/MANAGER.md
  docs/REPORT.md
  docs/STUDY_R.md
  engines/fuse/nvfx_fuse.hpp
  include/neuralfx/coupled3d.hpp
  include/neuralfx/nvfx.h
  include/neuralfx/nvfx_manager.h
  include/neuralfx/nvfx_volume.h
  include/neuralfx/parallel.hpp
  include/neuralfx/sim3d.hpp
  src/manager/manager.cpp
  src/manager/manager.hpp
  src/manager/nvfx_manager.cpp
  src/runtime/nvfx_volume.cpp
  src/sim3d/sim3d.cpp
  tests/alloc_test.cpp
  tests/test_3d.cpp
  tests/test_coupled3d.cpp
  tests/test_manager.cpp
  tools/manager/time_crowd.sh
  tools/manager/time_crowd_summary.awk
  tools/nvfx_3d.cpp
  tools/nvfx_gallery.cpp
  tools/nvfx_scene_script.cpp
  videos/README.md
)

FILES2=(
  docs/COMPOSE.md
  docs/LIBRARY.md
  docs/WEATHER.md
  examples/scenes/weather_day.nvfxs
  library/dust.nvfxl
  library/earth.nvfxl
  library/smoke.nvfxl
  library/steam.nvfxl
  library/storms.nvfxl
  library/trails.nvfxl
  results/library/frozen.csv
  results/library/frozen_round3.csv
  src/compose/compose.cpp
  src/compose/compose.hpp
  src/compose/script.cpp
  src/compose/script_run.cpp
  src/compose/weather.cpp
  src/compose/weather.hpp
  tests/test_library.cpp
  tests/test_script.cpp
  tests/test_weather_director.cpp
)

FILES3=(
  .gitattributes
  .gitignore
  COMMIT_PLAN.md
  TODO.md
  docs/STUDY_3D_HRES.md
  docs/STUDY_Q2.md
  docs/STUDY_Q2_RESTORE.md
  docs/STUDY_R2.md
  docs/STUDY_R2_RESTORE.md
  docs/PLAN_PC_GPU.md
  docs/WAVE1_CHECKLIST.md
  include/neuralfx/rollout3d_train.hpp
  results/3d/pc.md
  results/3d/hres/fire/compare_test_learned_vs_N2.csv
  results/3d/hres/fire/compare_test_learned_vs_N2.md
  results/3d/hres/fire/compare_test_learned_vs_coarse.csv
  results/3d/hres/fire/compare_test_learned_vs_coarse.md
  results/3d/hres/fire/edge_test.csv
  results/3d/hres/fire/edge_test_N2.csv
  results/3d/hres/fire/edge_test_learned.csv
  results/3d/hres/fire/edge_test_sim.csv
  results/3d/hres/fire/edgecheck_test.md
  results/3d/hres/fire/edgecheck_test_single.md
  results/manager/half_128/ab_summary.csv
  results/manager/half_128/ab_summary.txt
  results/manager/half_128/load.txt
  results/manager/half_128/run2/ab_summary.csv
  results/manager/half_128/run2/ab_summary.txt
  results/manager/half_128/run2/load.txt
  results/manager/half_128/run2/runs.csv
  results/manager/half_128/runs.csv
  src/train/rollout3d_train.cpp
  src/train/rollout3d_train_cuda_host.cpp
  tools/manager/time_half_128.sh
  tools/preflight_wave1.sh
  tools/ship_wave1.sh
  tools/study_3d/hres_fire_recheck_test_edges.sh
  tools/study_3d/hres_fire_validate.sh
  tools/study_3d/hres_smoke_r2.sh
  tools/study_3d/hres_smoke_r2_resume_cuda.sh
  tools/study_3d/hres_smoke_r2_train.sh
  tools/study_3d/hres_status.sh
  tools/study_3d/hres_sync_fire_test_results.sh
  tools/study_3d/r2_bootstrap.sh
  tools/study_3d/r2_env.sh
  tools/study_3d/r2_resume.sh
  tools/study_3d/r2_status.sh
  tools/study_3d/r2_test.sh
  tools/study_3d/r2_train.sh
  tools/study_3d/r2_validate.sh
  tools/study_3d/restart_hres_watchers.sh
  tools/study_3d/run_fire_validate.sh
  tools/study_q2/after_val.sh
  tools/study_q2/bootstrap.sh
  tools/study_q2/common.sh
  tools/study_q2/restore_from.sh
  tools/study_q2/resume.sh
  tools/study_q2/val.sh
)

# The scripts that must be 100755 in the index (the plan's note after commit 3): tools/study_q2/*.sh, the r2_*.sh,
# and the new bash scripts, which `git add` on this PC would otherwise record as 100644. The other hres scripts are
# 100644 in HEAD and stay so.
EXEC=(
  tools/manager/time_half_128.sh
  tools/preflight_wave1.sh
  tools/ship_wave1.sh
  tools/study_3d/hres_smoke_r2.sh
  tools/study_3d/hres_smoke_r2_resume_cuda.sh
  tools/study_3d/hres_smoke_r2_train.sh
  tools/study_3d/r2_bootstrap.sh
  tools/study_3d/r2_env.sh
  tools/study_3d/r2_resume.sh
  tools/study_3d/r2_status.sh
  tools/study_3d/r2_test.sh
  tools/study_3d/r2_train.sh
  tools/study_3d/r2_validate.sh
  tools/study_q2/after_val.sh
  tools/study_q2/bootstrap.sh
  tools/study_q2/common.sh
  tools/study_q2/restore_from.sh
  tools/study_q2/resume.sh
  tools/study_q2/val.sh
)

# ---- settings and helpers --------------------------------------------------------------------------------------------

DRY_RUN=${DRY_RUN:-1}
FORCE=${FORCE:-0}
CHECKS=${CHECKS:-1}
COMMITS=${COMMITS:-1}
PREFLIGHT=${PREFLIGHT:-0}
ALLOW_UNLISTED=${ALLOW_UNLISTED:-0}
ALLOW_CRLF=${ALLOW_CRLF:-0}
BUILD=${BUILD:-build}
BUILD_TYPE=${BUILD_TYPE:-Release}
GENERATOR=${GENERATOR:-Ninja}

die() { echo "ship_wave1: error: $*" >&2; exit 1; }
note() { echo "ship_wave1: $*"; }

# One argument as the shell would take it: bare where it can be, single-quoted otherwise (a multi-line commit body
# is shown as its first line and a count).
q() {
  local a=$1 sq="'"
  if [[ $a == *$'\n'* ]]; then
    printf " '%s ...' # +%d more lines" "${a%%$'\n'*}" "$(( $(printf '%s\n' "$a" | wc -l) - 1 ))"
  elif [[ $a =~ ^[A-Za-z0-9_./=+:,@%-]+$ ]]; then printf ' %s' "$a"
  else printf " '%s'" "${a//$sq/$sq\\$sq$sq}"; fi
}
show() { local a; printf '+'; for a in "$@"; do q "$a"; done; printf '\n'; }

# Print a command; in a dry run that is all, otherwise run it too.
run() {
  show "$@"
  if [ "$DRY_RUN" = 0 ]; then "$@"; fi
}

# Print a command that is run from a directory (the checklist's build-directory commands).
run_in() {
  local dir=$1; shift
  printf '+ (cd %s &&' "$dir"
  local a; for a in "$@"; do q "$a"; done
  printf ')\n'
  if [ "$DRY_RUN" = 0 ]; then (cd "$dir" && "$@"); fi
}

shell_scripts() { local f; for f in "$@"; do case "$f" in *.sh) echo "$f";; esac; done; }

# The fenced block that follows the n-th line of COMMIT_PLAN.md matching a regex ("Files (" or "**Message:**").
plan_block() {
  tr -d '\r' < COMMIT_PLAN.md | PLAN_RE=$1 awk -v want="$2" '
    done { next }
    $0 ~ ENVIRON["PLAN_RE"] { n++; if (n == want) armed = 1; next }
    armed && /^```/ { if (inb) done = 1; inb = 1; next }
    inb { print }'
}

files_of() { case "$1" in 1) printf '%s\n' "${FILES1[@]}";; 2) printf '%s\n' "${FILES2[@]}";; 3) printf '%s\n' "${FILES3[@]}";; esac; }

# The members of EXEC among the given files.
exec_among() {
  local f e
  for f in "$@"; do for e in "${EXEC[@]}"; do [ "$f" = "$e" ] && { echo "$f"; break; }; done; done
  return 0
}

# Checklist item 6: a staged file whose change shrinks to little once CR-at-EOL is ignored had its line endings
# converted. If the staged blob carries CRs the file was converted to CRLF (the fault); if it carries none the
# conversion was to LF (wanted: time_crowd.sh, CRLF in HEAD, LF in the tree). Prints both; returns 1 on a fault.
crlf_suspects() {
  local bad=0 raw ign file a b c d
  while IFS=$'\t' read -r a b file; do
    [ -n "$file" ] || continue
    [ "$a" = - ] && continue   # binary
    raw=$((a + b))
    read -r c d _ < <(git diff --cached --numstat --ignore-cr-at-eol -- "$file" | head -1 || true)
    [ "${c:--}" = - ] && continue
    ign=$((${c:-0} + ${d:-0}))
    if [ "$raw" -gt 20 ] && [ "$ign" -lt $((raw / 4)) ]; then
      if git show ":$file" | grep -q $'\r'; then
        echo "  CRLF! $file: $raw lines changed, $ign once CR at end of line is ignored; the staged copy is CRLF"
        bad=1
      else
        echo "  to LF: $file: $raw lines changed, $ign once CR at end of line is ignored; the staged copy is LF (wanted)"
      fi
    fi
  done < <(git diff --cached --numstat)
  return $bad
}

# Stage one commit: add the files, fix the modes the plan asks for, check line endings, commit.
make_commit() {
  local n=$1 subj=$2 body=$3; shift 3
  local -a files=("$@") modes
  echo
  echo "=== commit $n: ${#files[@]} files ==="
  run git add -- "${files[@]}"
  mapfile -t modes < <(exec_among "${files[@]}")
  if [ "${#modes[@]}" -gt 0 ]; then
    run git update-index --chmod=+x -- "${modes[@]}"
  fi
  if [ "$DRY_RUN" = 0 ]; then
    echo "--- staged (git diff --cached --stat, the tail) ---"
    git diff --cached --stat | tail -n 12
    local staged
    staged=$(git diff --cached --name-only | wc -l | tr -d ' ')
    [ "$staged" = "${#files[@]}" ] || note "warning: commit $n stages $staged paths, the plan lists ${#files[@]} (a listed file unchanged against HEAD is a no-op for git add)"
    if [ "${#modes[@]}" -gt 0 ]; then
      local bad_mode
      bad_mode=$(git ls-files -s -- "${modes[@]}" | awk '$1 != "100755" {print $4}')
      [ -z "$bad_mode" ] || die "commit $n: not 100755 in the index after --chmod=+x: $bad_mode"
    fi
    if ! crlf_suspects; then
      if [ "$ALLOW_CRLF" = 1 ]; then
        note "commit $n: CRLF suspects above, committing anyway (ALLOW_CRLF=1)"
      else
        die "commit $n: the staged diff looks like a CRLF rewrite (checklist item 6); 'git diff --cached --ignore-cr-at-eol' shows the real change; make the file LF, 'git reset -q' and re-run; or ALLOW_CRLF=1"
      fi
    fi
  else
    echo "+ git diff --cached --stat                # checklist item 6: no whole file rewritten (CRLF); ${#modes[@]} script(s) 100755"
  fi
  run git commit -q -m "$subj" -m "$body"
}

# ---- main ----------------------------------------------------------------------------------------------------------

main() {
  local REPO f missing unlisted all_files head_branch
  REPO=${NVFX_REPO:-$(cd "$(dirname "$0")/.." && pwd)}
  [ -f "$REPO/COMMIT_PLAN.md" ] || die "COMMIT_PLAN.md not found at $REPO (running a copy? set NVFX_REPO)"
  cd "$REPO"
  git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "$REPO is not a git checkout"

  case "$DRY_RUN" in 0|1) ;; *) die "DRY_RUN must be 0 or 1 (got '$DRY_RUN')";; esac
  if [ "$DRY_RUN" = 0 ] && [ "$FORCE" != 1 ]; then
    die "DRY_RUN=0 needs FORCE=1 as well; nothing was touched. Read the dry run first: bash tools/ship_wave1.sh"
  fi

  echo "=== ship wave 1 $(date -Iseconds) ==="
  if [ "$DRY_RUN" = 1 ]; then
    echo "mode: DRY RUN (every command below is printed, not run; DRY_RUN=0 FORCE=1 runs them)"
  elif [ "$COMMITS" = 1 ]; then
    echo "mode: LIVE (DRY_RUN=0 FORCE=1): the index is reset and three commits are made$([ "$CHECKS" = 1 ] && echo ', then the checklist runs')"
  else
    echo "mode: LIVE (DRY_RUN=0 FORCE=1, COMMITS=0): the checklist only, on the state as it stands"
  fi
  head_branch=$(git rev-parse --abbrev-ref HEAD)
  echo "HEAD: $(git log --oneline -1) on $head_branch"
  echo "core.autocrlf=$(git config --get core.autocrlf || echo unset) core.filemode=$(git config --get core.filemode || echo unset)"
  [ "$head_branch" = main ] || note "warning: the plan was written on main; HEAD is on $head_branch"

  all_files=("${FILES1[@]}" "${FILES2[@]}" "${FILES3[@]}")
  echo "plan: ${#FILES1[@]} + ${#FILES2[@]} + ${#FILES3[@]} = ${#all_files[@]} files in three commits; ${#EXEC[@]} scripts to be 100755"

  # ---- step 0: checks (read-only, both modes). Blockers are collected, not fatal one by one: a dry run prints them
  # all and still shows the plan, then exits 1; a live run stops here, before any git command.
  local -a blockers=()

  missing=
  for f in "${all_files[@]}"; do [ -e "$f" ] || missing="$missing $f"; done
  [ -z "$missing" ] || blockers+=("listed by the plan but not in the tree:$missing")

  local dups
  dups=$(printf '%s\n' "${all_files[@]}" | sort | uniq -d | tr '\n' ' ')
  [ -z "$dups" ] || blockers+=("listed in more than one commit: $dups")
  for f in "${EXEC[@]}"; do
    printf '%s\n' "${all_files[@]}" | grep -Fxq "$f" || blockers+=("EXEC names $f, which no commit lists")
  done

  # nothing changed that the plan does not place (git status against HEAD, staged or not, untracked included)
  local status_paths
  status_paths=$(git status --porcelain=v1 --untracked-files=all | cut -c4- | sed 's/^.* -> //')
  unlisted=$(printf '%s\n' "$status_paths" | grep -Fxv -f <(printf '%s\n' "${all_files[@]}") || true)
  # A tracked path whose change against HEAD vanishes under --ignore-cr-at-eol is .gitattributes' renormalization of
  # a CRLF script (checklist item 6: a separate endings-only commit after the three, not one of these). A git that has
  # re-read the file (Windows') lists it as " M"; one trusting its stat cache (WSL's) does not. Reported, not a blocker.
  local endings_only= p
  for p in $unlisted; do
    git ls-files --error-unmatch -- "$p" >/dev/null 2>&1 || continue
    [ -n "$(git diff HEAD --numstat -- "$p" 2>/dev/null)" ] || continue
    [ -z "$(git diff HEAD --numstat --ignore-cr-at-eol -- "$p" 2>/dev/null)" ] || continue
    endings_only="$endings_only $p"
    unlisted=$(printf '%s\n' "$unlisted" | grep -Fxv "$p" || true)
  done
  [ -z "$endings_only" ] || { echo "line endings only against HEAD (the .gitattributes renormalize; a separate commit after the three, checklist item 6):"; printf '  %s\n' $endings_only; }
  if [ -n "$unlisted" ]; then
    echo "changed in the tree but in no commit of the plan:"
    printf '  %s\n' $unlisted
    if [ "$ALLOW_UNLISTED" = 1 ]; then
      note "ALLOW_UNLISTED=1: they stay in the working tree, uncommitted; 'git status --short' is not clean after commit 3"
    else
      blockers+=("the plan places none of the paths above: add each to COMMIT_PLAN.md and to a FILES list here, or ALLOW_UNLISTED=1 leaves them uncommitted")
    fi
  else
    echo "tree: every changed path ($(printf '%s\n' "$status_paths" | grep -c . || true) rows of git status -uall) is in one of the three lists"
  fi

  # listed files not changed against HEAD (harmless: git add stages nothing for them), for the record
  local unchanged
  unchanged=$(printf '%s\n' "${all_files[@]}" | grep -Fxv -f <(printf '%s\n' "$status_paths") || true)
  [ -z "$unchanged" ] || { echo "listed but unchanged against HEAD (git add is a no-op for these):"; printf '  %s\n' $unchanged; }

  # every script the plan stages: LF line endings (checklist item 6; bash reads "do\r" as a command and fails), then
  # bash -n (the plan's note after commit 3)
  local crlf= bad_syntax=
  for f in $(shell_scripts "${all_files[@]}"); do
    if grep -q $'\r' "$f"; then crlf="$crlf $f"
    elif ! bash -n "$f" 2>/dev/null; then bad_syntax="$bad_syntax $f"; fi
  done
  [ -z "$crlf" ] || blockers+=("CRLF line endings in the working copy (make them LF, e.g. sed -i 's/\\r\$//' FILE):$crlf")
  [ -z "$bad_syntax" ] || blockers+=("bash -n fails on:$bad_syntax")
  [ -n "$crlf$bad_syntax" ] || echo "bash -n: $(shell_scripts "${all_files[@]}" | wc -l | tr -d ' ') scripts parse, all LF"

  # the modes in the index today (fixed by --chmod=+x at staging; new files have no index entry yet)
  local m644
  m644=$(git ls-files -s -- "${EXEC[@]}" | awk '$1 != "100755" {print $4}')
  [ -z "$m644" ] || { echo "index mode not 100755 today (git update-index --chmod=+x when staged):"; printf '  %s\n' $m644; }

  # git can commit here (an identity; not inside a rebase or merge). The identity is the environment's, not the
  # tree's or the plan's: a dry run commits nothing, so a missing one is a note and the dry run still passes; a live
  # run would fail at its first `git commit`, so there it is a blocker (COMMIT_PLAN.md, pre-push checklist, "Git identity").
  if git var GIT_COMMITTER_IDENT >/dev/null 2>&1; then
    :
  elif [ "$DRY_RUN" = 0 ]; then
    blockers+=("git has no committer identity here: the owner sets user.name and user.email first (COMMIT_PLAN.md, pre-push checklist, Git identity)")
  else
    note "note: git has no committer identity here; a live run (DRY_RUN=0 FORCE=1) needs user.name and user.email set first (COMMIT_PLAN.md, pre-push checklist, Git identity)"
  fi
  [ ! -e "$(git rev-parse --git-dir)/MERGE_HEAD" ] && [ ! -d "$(git rev-parse --git-dir)/rebase-merge" ] || blockers+=("a merge or rebase is in progress")

  # the plan's note under commit 2: no file but dust.nvfxl declares the look brownout
  local brownout
  brownout=$(grep -rl --include='*.nvfxl' -E '^[[:space:]]*look[[:space:]]+brownout([[:space:]]|$)' library/ | grep -Fxv library/dust.nvfxl || true)
  [ -z "$brownout" ] || blockers+=("the look brownout is declared outside library/dust.nvfxl (16 tests once failed on it): $(echo $brownout)")

  # this file against the plan: the n-th "Files (" block and the n-th "**Message:**" block of COMMIT_PLAN.md must
  # read as FILESn and SUBJn + BODYn (the plan is the authority; a drift here is a blocker)
  local -a subjs=("$SUBJ1" "$SUBJ2" "$SUBJ3") bodies=("$BODY1" "$BODY2" "$BODY3")
  local n want got d
  for n in 1 2 3; do
    d=$(diff <(files_of "$n" | sort) <(plan_block '^Files \(' "$n" | sort) | grep '^[<>]' | tr '\n' ' ' || true)
    [ -z "$d" ] || blockers+=("FILES$n differs from the plan's commit $n list (< this file, > the plan): $d")
    want=$(printf '%s\n\n%s' "${subjs[$((n - 1))]}" "${bodies[$((n - 1))]}")
    got=$(plan_block '^\*\*Message:\*\*' "$n")
    [ "$want" = "$got" ] || blockers+=("commit $n's message (SUBJ$n and BODY$n) differs from the plan's Message block; copy the plan's text into this file")
  done
  echo "plan: the three file lists and messages read as COMMIT_PLAN.md's$( [ "${#blockers[@]}" -gt 0 ] && echo ' (see the blockers)')"

  if [ "${#blockers[@]}" -gt 0 ]; then
    echo
    echo "BLOCKERS (${#blockers[@]}): a live run stops here, before any git command"
    printf '  - %s\n' "${blockers[@]}"
    [ "$DRY_RUN" = 1 ] || die "fix the blockers above and re-run"
  fi

  # ---- steps 1 to 3: reset and the three commits
  if [ "$COMMITS" = 1 ]; then
    echo
    echo "=== before anything: unstage everything (the working tree is untouched) ==="
    [ "$DRY_RUN" = 1 ] || STARTED=1
    run git reset -q
    if [ "$DRY_RUN" = 0 ]; then
      if git status --porcelain=v1 | grep -q '^[MADRC]'; then
        git status --short
        die "something is still staged after git reset"
      fi
    else
      echo "+ git status --short                     # expect ' M' and '??' rows only, none staged"
    fi

    make_commit 1 "$SUBJ1" "$BODY1" "${FILES1[@]}"
    make_commit 2 "$SUBJ2" "$BODY2" "${FILES2[@]}"
    make_commit 3 "$SUBJ3" "$BODY3" "${FILES3[@]}"

    echo
    echo "=== after the commits (checklist item 8) ==="
    run git log --oneline -3
    run git status --short
    if [ "$DRY_RUN" = 0 ]; then
      if [ -n "$(git status --porcelain=v1 --untracked-files=all)" ]; then
        note "warning: the working tree is not clean after commit 3 (the paths listed above as unlisted, if any)"
      else
        echo "clean."
      fi
    fi
  else
    echo
    note "COMMITS=0: steps 1 to 3 skipped (no reset, no commit)"
  fi

  # ---- step 4: the pre-push checklist
  if [ "$CHECKS" = 1 ]; then
    echo
    echo "=== pre-push checklist (COMMIT_PLAN.md; build directory $BUILD) ==="
    echo "--- 1. configure and build, tests on ---"
    run cmake -S . -B "$BUILD" -G "$GENERATOR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE" -DNEURALFX_BUILD_TESTS=ON
    run cmake --build "$BUILD" -j
    if [ "$PREFLIGHT" = 1 ]; then
      echo "--- 2. the fast checks (tools/preflight_wave1.sh; must end 'preflight_wave1: ok') ---"
      BUILD="$BUILD" run bash tools/preflight_wave1.sh
    else
      echo "--- 2. the fast checks: optional, not run (PREFLIGHT=1 runs bash tools/preflight_wave1.sh here) ---"
    fi
    echo "--- 3. the full test suite, serially (602 tests; expect 570 passed, 0 failed, 32 skipped with 77 on this PC; the split goes in COMMIT_PLAN.md's Verified) ---"
    run_in "$BUILD" ctest -j1 --output-on-failure
    echo "--- 4. no allocation per frame, the volume's job hook included (ends 'ok: no allocation per frame') ---"
    run_in "$BUILD" ./nvfx_alloc_test
    echo "--- 5. the library catalogue parses, plays and its frozen keyframes match (240 entries, ends '0 failed') ---"
    run_in "$BUILD" ./nvfx_gallery --library ../library --check
    echo "--- 6. line endings: checked per commit above (git diff --cached --numstat with and without --ignore-cr-at-eol) ---"
    echo "--- 7. the Godot test: SKIPPED by this script (needs a Godot build; by hand: bash engines/godot/test.sh) ---"
    echo "--- 8. git log --oneline -3 reads as the three messages, git status --short is clean: printed above ---"
    if [ "$DRY_RUN" = 0 ]; then
      echo
      echo "checklist passed. The three commits are local; nothing was pushed. Push when satisfied: git push origin main"
    fi
  else
    echo
    note "CHECKS=0: the pre-push checklist skipped"
  fi

  if [ "$DRY_RUN" = 1 ]; then
    echo
    if [ "${#blockers[@]}" -gt 0 ]; then
      echo "dry run only: nothing changed. ${#blockers[@]} blocker(s) above would stop a live run before its first git command."
      return 1
    fi
    echo "dry run only: nothing changed. DRY_RUN=0 FORCE=1 bash tools/ship_wave1.sh runs the above."
  fi
}

# A failure once git has been touched must say where things stand.
STARTED=0
on_exit() {
  local rc=$?
  if [ "$rc" -ne 0 ] && [ "$STARTED" = 1 ]; then
    echo "ship_wave1: failed (exit $rc) after git was touched. Commits made so far stand (git log --oneline -3); nothing was pushed." >&2
    echo "ship_wave1: a partly staged commit is undone by 'git reset -q'; the commits by 'git reset --soft HEAD~N'." >&2
  fi
}
trap on_exit EXIT

main "$@"
