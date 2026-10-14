# nvfx_f2 timing (results/compression/README.md, F4 cost), two runs on a quiet machine, 14 October 2026 (the tool prints "provisional: the machine is shared" on every run; here no other job ran)

## run 1
timing on core 2, load average 1.91 2.31 2.57 (provisional: the machine is shared)
| model | precision | stored KB | resident KB | scratch KB | ms per 128 x 128 frame (thread CPU, median of 200 frames, least of 7 runs) |
|---|---|---:|---:|---:|---:|
| fire_0__g32c8h32l2t16_b4_q_sp_i6000.nvfx | float | 32.5 | 36.7 | 43.4 | 0.970 |
| fire_0__g32c8h32l2t16_b4_q_sp_i6000.nvfx | int8 | 32.5 | 36.7 | 186.5 | 0.500 |
| fire_0__p16.4.16+64.2.16h32l2_b4_q_sp_wq_i6000.nvfx | float | 31.8 | 37.0 | 47.0 | 0.868 |
| fire_0__p16.4.16+64.2.16h32l2_b4_q_sp_wq_i6000.nvfx | int8 | 31.8 | 37.0 | 107.9 | 0.556 |
| fire_0__p16.8.16+64.2.16h32l2_b4_q_sp_wq_i6000.nvfx | float | 36.5 | 42.7 | 53.5 | 1.043 |
| fire_0__p16.8.16+64.2.16h32l2_b4_q_sp_wq_i6000.nvfx | int8 | 36.5 | 42.7 | 112.4 | 0.579 |

## run 2
timing on core 2, load average 1.84 2.29 2.56 (provisional: the machine is shared)
| model | precision | stored KB | resident KB | scratch KB | ms per 128 x 128 frame (thread CPU, median of 200 frames, least of 7 runs) |
|---|---|---:|---:|---:|---:|
| fire_0__g32c8h32l2t16_b4_q_sp_i6000.nvfx | float | 32.5 | 36.7 | 43.4 | 0.943 |
| fire_0__g32c8h32l2t16_b4_q_sp_i6000.nvfx | int8 | 32.5 | 36.7 | 186.5 | 0.489 |
| fire_0__p16.4.16+64.2.16h32l2_b4_q_sp_wq_i6000.nvfx | float | 31.8 | 37.0 | 47.0 | 0.848 |
| fire_0__p16.4.16+64.2.16h32l2_b4_q_sp_wq_i6000.nvfx | int8 | 31.8 | 37.0 | 107.9 | 0.556 |
| fire_0__p16.8.16+64.2.16h32l2_b4_q_sp_wq_i6000.nvfx | float | 36.5 | 42.7 | 53.5 | 1.033 |
| fire_0__p16.8.16+64.2.16h32l2_b4_q_sp_wq_i6000.nvfx | int8 | 36.5 | 42.7 | 112.4 | 0.582 |
