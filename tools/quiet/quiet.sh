#!/usr/bin/env bash
# The quiet-machine harness (results/quiet/README.md): sourced by a round's driver (tools/quiet/round7.sh).
#
#   . tools/quiet/quiet.sh OUT            OUT receives blocks.csv, monitor.csv and each block's log
#   quiet_start                           starts the sampler (every 2 s: the load, the CPU busy, the steal, and the
#                                         CPU of this driver's own processes, so what anything else used is known)
#   block NAME CMD...                     one block: waits until the machine is idle (the gate), runs CMD with its
#                                         output in OUT/logs/NAME.log and QUIET_BLOCK_DIR (OUT/raw/NAME, emptied before
#                                         each try) for its files, and records in blocks.csv the load before and after,
#                                         the block's own CPU and the CPU anything else used meanwhile. A block is run
#                                         again (at most QUIET_TRIES = 8 times) when it started at a load above
#                                         QUIET_LOAD (0.5), when the rest of the machine used more than QUIET_FOREIGN
#                                         (0.5) of a core over it on average, or more than QUIET_BURST (1.0) over any
#                                         10 s of it
#   quiet_stop
#
# The gate: the 1-minute load average at most QUIET_LOAD and less than QUIET_IDLE (0.3) of a core busy over the last
# 5 s (what this driver's previous block left in the load average decays in about a minute: 4 threads take two).
# "Foreign" CPU is the machine's busy time (/proc/stat: user, nice, system, irq, softirq) less this driver's process
# tree's (utime + stime + the reaped children's, /proc/PID/stat), so it is what other jobs (and the kernel's own work)
# took; steal (the hypervisor's) is recorded beside it. QUIET_NOGATE=1 skips the gate (a dry run of the commands).
set -uo pipefail
QUIET_OUT=${1:?OUT}
QUIET_LOAD=${QUIET_LOAD:-0.5}
QUIET_IDLE=${QUIET_IDLE:-0.3}
QUIET_FOREIGN=${QUIET_FOREIGN:-0.5}
QUIET_TRIES=${QUIET_TRIES:-8}
QUIET_BURST=${QUIET_BURST:-1.0}
QUIET_DRIVER=$$
mkdir -p "$QUIET_OUT/logs"
[ -s "$QUIET_OUT/blocks.csv" ] || echo "block,try,start_utc,end_utc,wall_s,load1_start,load5_start,load15_start,load1_end,own_cpu_s,busy_cpu_s,foreign_cpu_s,foreign_cores,foreign_cores_max10s,steal_s,rc,verdict" > "$QUIET_OUT/blocks.csv"

# One line: epoch seconds (ms), the machine's busy and steal ticks, this driver tree's ticks less the sampler's, the
# load averages and the running count. Ticks are USER_HZ (100 a second) in both files.
quiet_snapshot() {
  local skip=${QUIET_SAMPLER_PID:-0}
  { date +%s.%N; head -1 /proc/stat; cat /proc/loadavg; cat /proc/[0-9]*/stat 2>/dev/null; } | awk -v root="$QUIET_DRIVER" -v skip="$skip" '
    NR == 1 { t = $1; next }
    NR == 2 { busy = $2 + $3 + $4 + $7 + $8; steal = $9; next }
    NR == 3 { l1 = $1; l5 = $2; l15 = $3; split($4, rq, "/"); next }
    {
      i = 0; for (k = length($0); k > 0; k--) if (substr($0, k, 1) == ")") { i = k; break }
      if (!i) next
      pid = $1 + 0; n = split(substr($0, i + 2), f, " ")
      par[pid] = f[2] + 0; cpu[pid] = f[12] + f[13] + f[14] + f[15]
    }
    END {
      own = 0
      for (p in cpu) {
        q = p + 0; mine = 0; hops = 0
        while (q > 1 && hops < 64) { if (q == skip) { mine = 0; break } if (q == root) { mine = 1; break } q = par[q]; hops++ }
        if (mine) own += cpu[p]
      }
      printf "%.3f %d %d %d %s %s %s %d\n", t, busy, steal, own, l1, l5, l15, rq[1]
    }'
}

quiet_start() {
  [ -s "$QUIET_OUT/monitor.csv" ] || echo "utc,dt_s,busy_cores,own_cores,foreign_cores,load1,running" > "$QUIET_OUT/monitor.csv"
  (
    QUIET_SAMPLER_PID=$BASHPID
    prev=""
    while :; do
      s=$(quiet_snapshot)
      if [ -n "$prev" ]; then
        awk -v a="$prev" -v b="$s" 'BEGIN { split(a, x, " "); split(b, y, " "); dt = y[1] - x[1]; if (dt <= 0) exit
          printf "%s,%.2f,%.3f,%.3f,%.3f,%s,%s\n", strftime("%Y-%m-%dT%H:%M:%SZ", int(y[1]), 1), dt, (y[2] - x[2]) / 100 / dt,
                 (y[4] - x[4]) / 100 / dt, ((y[2] - x[2]) - (y[4] - x[4])) / 100 / dt, y[5], y[8] }' >> "$QUIET_OUT/monitor.csv"
      fi
      prev=$s
      sleep 2
    done
  ) &
  QUIET_SAMPLER_PID=$!
  export QUIET_SAMPLER_PID
}

quiet_stop() { [ -n "${QUIET_SAMPLER_PID:-}" ] && kill "$QUIET_SAMPLER_PID" 2> /dev/null; wait "$QUIET_SAMPLER_PID" 2> /dev/null; }

# busy cores over the next N seconds (the whole machine)
quiet_busy() {
  local a b
  a=$(head -1 /proc/stat)
  sleep "$1"
  b=$(head -1 /proc/stat)
  awk -v a="$a" -v b="$b" -v n="$1" 'BEGIN { split(a, x, " "); split(b, y, " "); printf "%.3f", ((y[2]+y[3]+y[4]+y[7]+y[8]) - (x[2]+x[3]+x[4]+x[7]+x[8])) / 100 / n }'
}

quiet_gate() {
  [ "${QUIET_NOGATE:-0}" = 1 ] && return 0
  local t0 l1 b waited=0
  t0=$(date +%s)
  while :; do
    l1=$(cut -d' ' -f1 /proc/loadavg)
    b=$(quiet_busy 5)
    if awk -v l="$l1" -v b="$b" -v L="$QUIET_LOAD" -v I="$QUIET_IDLE" 'BEGIN { exit !(l <= L && b < I) }'; then break; fi
    if [ $(( $(date +%s) - t0 )) -gt $(( 120 + waited )) ]; then  # still busy after two minutes: say who
      waited=$((waited + 120))
      { echo "gate $(date -u +%FT%TZ): load $l1, busy $b; the busiest processes:"; ps -eo pid,user,pcpu,etime,comm --sort=-pcpu | head -8; } >> "$QUIET_OUT/gate.log"
    fi
    sleep 5
  done
  echo "$(date -u +%FT%TZ) gate passed after $(( $(date +%s) - t0 )) s (load $l1, busy $b)" >> "$QUIET_OUT/gate.log"
}

# block NAME CMD...: see the top. Returns the command's exit status of its last try.
block() {
  local name=$1 try=1 rc s0 s1 t0 t1 verdict
  shift
  while :; do
    quiet_gate
    export QUIET_BLOCK_DIR="$QUIET_OUT/raw/$name"
    rm -rf "$QUIET_BLOCK_DIR"
    mkdir -p "$QUIET_BLOCK_DIR"
    t0=$(date -u +%FT%TZ)
    s0=$(quiet_snapshot)
    "$@" > "$QUIET_OUT/logs/$name.try$try.log" 2>&1
    rc=$?
    s1=$(quiet_snapshot)
    t1=$(date -u +%FT%TZ)
    local line
    line=$(awk -v a="$s0" -v b="$s1" -v m="$QUIET_OUT/monitor.csv" -v t0="$t0" -v t1="$t1" -v F="$QUIET_FOREIGN" -v L="$QUIET_LOAD" -v X="$QUIET_BURST" '
      BEGIN {
        split(a, x, " "); split(b, y, " "); wall = y[1] - x[1]
        busy = (y[2] - x[2]) / 100; own = (y[4] - x[4]) / 100; steal = (y[3] - x[3]) / 100; foreign = busy - own
        # the largest foreign load over 10 s (five samples) of the monitor within the block
        mx = 0; n = 0
        while ((getline line < m) > 0) { split(line, c, ","); if (c[1] < t0 || c[1] > t1) continue; v[++n] = c[5] }
        for (i = 5; i <= n; i++) { s = (v[i] + v[i-1] + v[i-2] + v[i-3] + v[i-4]) / 5; if (s > mx) mx = s }
        fc = wall > 0 ? foreign / wall : 0
        verdict = (x[5] > L) ? "repeat: load at start" : (fc > F ? "repeat: foreign load" : (mx > X ? "repeat: foreign burst" : "ok"))
        printf "%.1f,%s,%s,%s,%s,%.1f,%.1f,%.1f,%.3f,%.3f,%.1f|%s\n", wall, x[5], x[6], x[7], y[5], own, busy, foreign, fc, mx, steal, verdict
      }')
    verdict=${line#*|}
    line=${line%|*}
    local wall=${line%%,*} rest=${line#*,}
    echo "$name,$try,$t0,$t1,$wall,$rest,$rc,$verdict" >> "$QUIET_OUT/blocks.csv"
    echo "$t1 $name try $try: rc $rc, $verdict ($line)"
    if [ "$verdict" = ok ] || [ "$try" -ge "$QUIET_TRIES" ]; then
      cp "$QUIET_OUT/logs/$name.try$try.log" "$QUIET_OUT/logs/$name.log"
      return "$rc"
    fi
    try=$((try + 1))
  done
}
