#!/bin/bash
# Study R2 §1.2 to §1.4, again on smoke_r2 with still ambient inflow (cb31ccb): the momentum budget on the two exploration
# seeds and the stability at the four extremes. Resumable.
B=${B:-/root/nvfx-data/r2_smoke3d/bin/nvfx_3d_cb31ccb}
. /root/nvfx-data/r2_smoke3d/jobs/disk.sh; need 0.6
cd /home/user/wt-r2
{
  for k in 0 1; do echo "plume --label d_r2b_i50 --effect smoke_r2 --spin 6 --seconds 16 --from $k --seeds $((k + 1)) --seed-base 34000000"; done
  echo "stability --seconds 30 --intensity 1.0 --wind 0.5 --turbulence 1.0 --seed 34000900"
  echo "stability --seconds 30 --intensity 0.8 --wind 0.0 --turbulence 1.0 --seed 34000901"
  echo "stability --seconds 30 --intensity 0.8 --wind 1.0 --turbulence 0.8 --seed 34000902"
  echo "stability --seconds 45 --intensity 0.2 --wind 0.5 --turbulence 0.0 --seed 34000903"
} | xargs -P 6 -I{} sh -c "nice -n 10 $B {} --study r2 --threads 1"
echo DIAG2_DONE
