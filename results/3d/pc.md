# This PC (P0 measurement, 24 October 2026)

Recorded for [PLAN_PC_GPU.md](../../docs/PLAN_PC_GPU.md) §4 and §2.1.

| | |
|---|---|
| Host OS | Windows 10/11 |
| WSL | WSL2, Linux 6.6.x, x86_64 |
| CPU | 1× Intel Xeon Gold 6242 @ 2.80 GHz, **64** logical processors |
| RAM | **48 GiB** |
| GPU | **NVIDIA GeForce RTX 3090**, **24 GiB** VRAM, driver **610.88** |
| CUDA (WSL) | **12.0** (`nvcc`) |
| CMake | 3.28.3, g++-14 |
| `NEURALVFX_DATA` (recommended) | `/mnt/d/nvfx-data` (~**5.4 TiB** free on `D:`) |
| `NEURALVFX_DATA` (hres studies today) | `/home/odin/nvfx-data` on 1 TiB WSL ext4 (~730 GiB free) |
| Repo | `Neural VFX` on OneDrive (`/mnt/c/...`), build in WSL |
| FUSE | **`C:\Users\odinl\OneDrive\Desktop\FUSE`** (git); adapter in `engines/fuse/` |
| References | **One per effect** (`reference.md`, §3.3.1); no paid VFX sims |
| ctest (`-L neuralfx`, this tree) | 602 tests, 0 failed, 32 skipped (data-gated) |

**Next P0:** `bash models/restore.sh` under `/mnt/d/nvfx-data` when starting P1 data; wire adapter against Desktop FUSE during P3 budget check.
