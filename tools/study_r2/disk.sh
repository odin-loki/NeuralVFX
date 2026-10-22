# Study R2: refuse to start a job that would leave the shared disk short (the coordinator's limit: R2 under ~2.5 GB).
# need G: at least G GB free on /, else stop the job (exit 9) and say so.
need() {
  local free_kb=$(df --output=avail -k / | tail -1)
  local used=$(du -sk /root/nvfx-data/r2_smoke3d | cut -f1)
  echo "disk: $((free_kb / 1024)) MB free on /; r2_smoke3d holds $((used / 1024)) MB"
  local lim=$(awk -v g="$1" 'BEGIN { printf "%d", g * 1048576 }')
  if [ "$free_kb" -lt "$lim" ]; then echo "DISK SHORT: under $1 GB free; stopping"; exit 9; fi
  if [ "$used" -gt $(( 2500 * 1024 )) ]; then echo "DISK BUDGET: r2_smoke3d over 2.5 GB; stopping"; exit 9; fi
}
