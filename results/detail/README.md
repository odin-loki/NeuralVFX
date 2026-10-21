# The detail layer made cheaper: measurements (docs/REPORT.md §19)

Old against new in one process: `tools/nvfx_detail_bench.cpp` loads two probe modules
(`tools/detail_bench/probe.cpp`), one built from this branch (`build/nvfx_detail_probe.so`) and one from main before
the change, c846c80 (`tools/detail_bench/build_old.sh c846c80 DIR`), and plays their runners in lockstep, frame by
frame, in alternating order (old-new, new-old, ...), with the same model, size, ISA, precision, seed and controls.
Every frame of both is compared bit for bit (2D: the fine heat and soot, the coarse state, the flow and the picture;
3D: the temperature and soot fields). Everything here came from one run of `tools/detail_bench/run_all.sh OLD.so
NEW.so` (resumable; a CSV that exists is skipped).

**The machine** (21 October 2026, after the container moved at 01:18 UTC): an Intel Xeon at 2.80 GHz, 4 cores,
AVX-512 with VNNI (Cascade Lake class), shared with five other agents' jobs, six of them training (load 18 to 21). So
absolute milliseconds are provisional and slow (each process got about a fifth of a core's share); the ratios are
interleaved and paired and are what to read. Times are the thread's CPU time (the 3D runs on 4 threads: wall time).

| file | what |
|---|---|
| `frames_2d_128.csv`, `_256`, `_512` | 27 models (DATA/library/models) x AVX2 and AVX-512, float: 12 rounds of 16 frames after 8 warm-up frames; per round the medians of each runner's step, frame (step and picture) and detail stage (study K's profile), and the frames that differ |
| `frames_2d_*.md` | their summaries: per size and ISA the geometric mean over models of each model's median ratio over rounds (95% bootstrap over models); per model the median ratio over rounds (95% bootstrap over rounds) |
| `volume_3d.csv`, `.md` | the 3D model (DATA/n2_3d/models/chosen.nvfx3) through the volume C API, one continuous rollout, 1 and 4 threads x AVX2 and AVX-512: 10 rounds of 6 steps |
| `detail_3d.csv`, `.md` | the 3D detail layer alone, in the new build: rollout3d::detail_step (the code every volume ran before) against the volume runtime's on two copies of one rollout, 10 rounds of 6 steps |
| `identity_2d_int8.csv` | bit identity of the 27 runners at int8 (the default on AVX-512 with VNNI), 256 px, both ISAs, 60 frames each |
| `identity_api.csv` | bit identity of the pictures through the C API (`nvfx_render`: shards, crossfades, warm-ups), 27 models at 128 px, both ISAs, the default precision, 300 frames (10 s) each |
| `option_semilagrangian_256.csv` | not shipped, an option that changes frames: a semi-Lagrangian detail layer (`back_row` returning the forward sample, no round trip; built in a scratch copy of `rt_rollout.hpp`) against this branch's exact layer ("old" in the file), 27 models at 256 px, 4 rounds of 16 frames; `min_psnr_db` is a round's lowest picture PSNR against the exact layer |

The 3D files are from the first run of the script (at f2100a4, whose 3D code is the final one); the 2D and identity files
from its second run, on the final 2D code (04d7fe2), after the 2D files of the first were set aside.

Summary (REPORT.md §19): the 3D detail layer alone 0.155 to 0.176 of its time on one core (a step 0.31 to 0.32); the 2D
detail stage 0.87 to 0.90; no frame, field or picture of any run differs.
