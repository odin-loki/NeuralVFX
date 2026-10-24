#!/usr/bin/env bash
# preflight_wave1.sh: the fast pre-push check for the 24 October 2026 ship (COMMIT_PLAN.md, "Pre-push checklist").
#
# Minutes, not the full ctest. It touches no git state: no add, commit, stash, reset or push. It runs, in order:
#   1. bash -n over the study and ship scripts (tools/study_3d, tools/study_q2, tools/manager, tools/ship_wave1.sh),
#      and names any of them with CRLF line endings (a warning, not a failure: bash -n then runs on a CR-stripped copy,
#      since bash reads "do\r" as a command, not the keyword; the checklist's line-endings item wants every .sh LF);
#   2. cmake --build of the four programs below (skipped with NO_BUILD=1);
#   3. neuralfx_weather_tests, all of it (the weather director, commit 2);
#   4. neuralfx_tests with --gtest_filter='Library.*' (the catalogue's tests, commit 2);
#   5. nvfx_alloc_test (must end "ok: no allocation per frame"; commit 1);
#   6. nvfx_gallery --library library --check (must end "0 failed"; commit 2).
# Every step runs (a program that did not build is skipped); the summary at the end lists the failures and the exit
# status is 1 if there were any. It does not replace the checklist's full `ctest -L neuralfx -j 1`: the manager, 3D,
# pack, wrapper and frozen-keyframe tests are not here.
#
#   bash tools/preflight_wave1.sh                 # from anywhere; the build dir is <repo>/build
#   BUILD=build-cuda bash tools/preflight_wave1.sh
#   NO_BUILD=1 bash tools/preflight_wave1.sh      # the programs are already built
#   THREADS=4 bash tools/preflight_wave1.sh       # nvfx_gallery --threads (default 2, as the ctest entry has it)
set -uo pipefail
shopt -s nullglob

REPO=${NVFX_REPO:-$(cd "$(dirname "$0")/.." && pwd)}
BUILD=${BUILD:-$REPO/build}
case "$BUILD" in /*) ;; *) BUILD=$REPO/$BUILD ;; esac   # a relative BUILD is taken from the repo root
NO_BUILD=${NO_BUILD:-0}
THREADS=${THREADS:-2}
PROGRAMS=(neuralfx_weather_tests neuralfx_tests nvfx_alloc_test nvfx_gallery)

t0=$(date +%s)
failures=()
step() { printf '\npreflight_wave1 [%4ds] %s\n' "$(( $(date +%s) - t0 ))" "$*"; }
failed() { printf 'preflight_wave1: FAILED: %s\n' "$*"; failures+=("$*"); }   # stdout, so the log stays in order
# run_program NAME ARGS...: the program's exit status is the verdict; skipped (and counted failed) if it did not build.
run_program() {
  local name=$1; shift
  if [[ ! -x "$BUILD/$name" ]]; then failed "$name: not built ($BUILD/$name missing)"; return; fi
  "$BUILD/$name" "$@" || failed "$name $*"
}

cd "$REPO" || { echo "preflight_wave1: cannot cd to $REPO" >&2; exit 1; }

# 1. Syntax of the scripts the ship carries (they run on the container and the PC; a typo here is found before push).
step "1/6 bash -n over the study and ship scripts"
scripts=(tools/study_3d/*.sh tools/study_q2/*.sh tools/manager/*.sh tools/ship_wave1.sh tools/preflight_wave1.sh)
bad=0
for f in "${scripts[@]}"; do
  [[ -f "$f" ]] || continue
  if grep -q $'\r' "$f"; then
    echo "  $f: CRLF (bash -n on a stripped copy; convert to LF before push, COMMIT_PLAN checklist step 5)"
    if ! tr -d '\r' < "$f" | bash -n 2>/dev/null; then
      echo "  $f: syntax error after CRLF strip:"; tr -d '\r' < "$f" | bash -n 2>&1 | sed 's/^/    /'
      bad=$((bad + 1))
    fi
  elif ! bash -n "$f" 2>/dev/null; then
    echo "  $f: syntax error:"; bash -n "$f" 2>&1 | sed 's/^/    /'
    bad=$((bad + 1))
  fi
done
if (( bad == 0 )); then echo "  ${#scripts[@]} scripts parse"; else failed "$bad of ${#scripts[@]} scripts do not parse"; fi

# 2. The four programs, built in place (the build dir is already configured; nothing is configured here).
step "2/6 build: ${PROGRAMS[*]} (NO_BUILD=$NO_BUILD)"
if [[ ! -f "$BUILD/CMakeCache.txt" ]]; then
  failed "$BUILD is not a configured build directory (set BUILD=...)"
elif [[ "$NO_BUILD" != 1 ]]; then
  cmake --build "$BUILD" -j --target "${PROGRAMS[@]}" || failed "cmake --build ${PROGRAMS[*]}"
fi

# 3. to 6. The programs themselves.
step "3/6 neuralfx_weather_tests"
run_program neuralfx_weather_tests --gtest_brief=1

step "4/6 neuralfx_tests --gtest_filter='Library.*'"
run_program neuralfx_tests --gtest_filter='Library.*' --gtest_brief=1

step "5/6 nvfx_alloc_test"
run_program nvfx_alloc_test

step "6/6 nvfx_gallery --library library --check --threads $THREADS"
if [[ -x "$BUILD/nvfx_gallery" ]]; then
  "$BUILD/nvfx_gallery" --library "$REPO/library" --check --threads "$THREADS" | tail -n 3 || failed "nvfx_gallery --check"
else
  failed "nvfx_gallery: not built ($BUILD/nvfx_gallery missing)"
fi

# The verdict.
if (( ${#failures[@]} == 0 )); then
  step "ok: all six steps passed (the full checklist: COMMIT_PLAN.md, Pre-push checklist)"
  exit 0
fi
step "${#failures[@]} failure(s):"
printf '  - %s\n' "${failures[@]}"
exit 1
