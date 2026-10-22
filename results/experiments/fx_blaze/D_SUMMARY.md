
## D: start points and learned dynamics

### Chaos horizon (the simulation itself)

Active PSNR against a run of the simulation from a stored start point, by frames after it (30 per second), mean over runs. Perturbations are relative to the velocity RMS.

| effect | case | 1 | 4 | 8 | 15 | 30 | 60 | 120 | 240 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wildfire | start+1e-3_same_seed | 76.67 | 72.50 | 70.72 | 69.01 | 66.55 | 62.30 | 57.70 | 53.19 |
| wildfire | start+1e-1_same_seed | 53.35 | 48.13 | 45.82 | 44.18 | 41.01 | 37.15 | 35.72 | 33.72 |
| wildfire | same_start_other_seed | 26.52 | 24.12 | 22.11 | 20.87 | 19.89 | 19.07 | 17.34 | 18.51 |
| wildfire | other_start_same_seed | 15.99 | 16.38 | 16.70 | 17.25 | 18.68 | 21.02 | 24.85 | 28.40 |
| coals | start+1e-3_same_seed | 90.75 | 83.44 | 82.13 | 88.14 | 88.95 | 95.79 | 99.00 | 96.01 |
| coals | start+1e-1_same_seed | 66.34 | 62.92 | 62.22 | 62.38 | 65.17 | 68.96 | 75.05 | 84.26 |
| coals | same_start_other_seed | 25.35 | 17.35 | 14.82 | 13.36 | 14.76 | 13.45 | 14.58 | 13.63 |
| coals | other_start_same_seed | 17.11 | 21.46 | 25.19 | 29.79 | 36.33 | 41.58 | 52.48 | 64.53 |
| mushroom | start+1e-3_same_seed | 73.56 | 71.86 | 70.93 | 68.70 | 66.59 | 64.02 | 62.07 | - |
| mushroom | start+1e-1_same_seed | 51.45 | 46.40 | 43.08 | 40.94 | 36.81 | 31.76 | 29.98 | - |
| mushroom | same_start_other_seed | 30.96 | 26.53 | 22.55 | 18.42 | 14.36 | 12.24 | 13.00 | - |
| mushroom | other_start_same_seed | 16.32 | 16.62 | 17.18 | 17.87 | 18.32 | 16.96 | 16.94 | - |

### Training

Stepper loss: stages 1-2 (normalised squared error plus profiles) and stage 3 (plus activity). Renderer PSNR on its training samples (true fields). Detail constants as calibrated.

| effect | runs x frames | minutes of simulation | stepper loss 1-2 | stepper s | stage 3 loss | stage 3 s | renderer dB | contrast | swirl | grow | start points | KB stored |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wildfire | 160 x 240 | 21.33 | 0.02745 | 7063 | 0.02772 | 1218 | 34.65 | 1 | 0.5 | 1 | 8 | 81.6 |
| coals | 160 x 240 | 21.33 | 0.05530 | 8134 | 0.05294 | 2363 | 37.82 | 0.5 | 0 | 1 | 8 | 81.6 |
| mushroom | 240 x 150 | 20.00 | 0.02823 | 7904 | 0.04843 | 2300 | 29.16 | 1 | 0.5 | 1 | 16 | 274.2 |

Renderer, detail constants and start points from `d-finish` on the trained steppers; stage 3 from `d-tune`.

### Tracking a held-out run from its true start point

Active PSNR against the true run, mean over held-out runs (other seeds and settings), by frames after the start point. `neural`: the rollout effect through the runtime, started from the true state (coarse and fine) with the run's noise seed. `coarse_sim_detail`: the simulation on the same 32-cell grid with the same detail layer, drawn by the simulation's own renderer. `neural_dynamics_true_renderer`: the neural dynamics drawn by that renderer too (separates dynamics from rendering). `renderer_on_true_fields`: the learned renderer on the true fields (frames 1-30, then their mean).

**coals**

| method | 1 | 4 | 8 | 16 | 30 | 60 | 120 | 240 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| neural | 31.27 | 25.89 | 23.49 | 21.13 | 20.87 | 19.60 | 19.98 | 19.14 |
| neural_dynamics_true_renderer | 34.71 | 26.39 | 23.60 | 21.06 | 20.66 | 19.38 | 19.56 | 18.82 |
| coarse_sim_detail | 30.65 | 21.86 | 18.78 | 16.36 | 16.22 | 14.80 | 15.55 | 14.87 |
| coarse_sim | 21.81 | 20.70 | 19.78 | 18.28 | 18.86 | 17.73 | 18.55 | 17.43 |
| frozen | 34.02 | 22.64 | 18.03 | 15.55 | 16.13 | 13.29 | 13.87 | 13.10 |
| renderer_on_true_fields | 33.55 | 33.47 | 33.26 | 32.64 | 33.08 | 32.97 | 32.97 | 32.97 |

neural - coarse_sim_detail, paired over runs: frame 1: +0.61 [-0.95, +2.30] (tie); frame 8: +4.71 [+4.11, +5.38]; frame 30: +4.65 [+4.18, +5.06]; frame 60: +4.80 [+4.15, +5.46]; frame 240: +4.27 [+3.44, +5.11];

neural_dynamics_true_renderer - coarse_sim_detail, paired over runs: frame 1: +4.06 [+2.50, +5.69]; frame 8: +4.83 [+4.20, +5.53]; frame 30: +4.44 [+3.93, +4.92]; frame 60: +4.59 [+3.87, +5.27]; frame 240: +3.96 [+3.21, +4.73];

**mushroom**

| method | 1 | 4 | 8 | 16 | 30 | 60 | 120 | 240 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| neural | 15.48 | 14.53 | 15.24 | 14.06 | 13.68 | 12.51 | 11.98 | - |
| neural_dynamics_true_renderer | 17.45 | 14.68 | 15.22 | 13.72 | 13.37 | 12.35 | 11.98 | - |
| coarse_sim_detail | 21.43 | 15.23 | 14.82 | 14.39 | 14.78 | 15.06 | 15.02 | - |
| coarse_sim | 19.26 | 17.15 | 16.78 | 16.08 | 16.29 | 16.33 | 16.02 | - |
| frozen | 11.42 | 6.21 | 5.39 | 5.16 | 5.19 | 6.31 | 8.09 | - |
| renderer_on_true_fields | 15.70 | 16.89 | 18.45 | 21.35 | 24.49 | 20.81 | 20.81 | 20.81 |

neural - coarse_sim_detail, paired over runs: frame 1: -5.96 [-7.49, -4.37]; frame 8: +0.42 [-0.26, +1.05] (tie); frame 30: -1.10 [-1.76, -0.53]; frame 60: -2.55 [-3.27, -1.76];

neural_dynamics_true_renderer - coarse_sim_detail, paired over runs: frame 1: -3.98 [-5.21, -2.53]; frame 8: +0.39 [-0.14, +0.92] (tie); frame 30: -1.41 [-2.07, -0.85]; frame 60: -2.70 [-3.41, -1.92];

**wildfire**

| method | 1 | 4 | 8 | 16 | 30 | 60 | 120 | 240 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| neural | 29.05 | 25.96 | 24.29 | 23.19 | 22.83 | 22.22 | 21.27 | 19.86 |
| neural_dynamics_true_renderer | 30.17 | 26.21 | 24.32 | 23.18 | 22.75 | 22.16 | 21.27 | 19.76 |
| coarse_sim_detail | 24.43 | 23.49 | 21.52 | 20.63 | 19.84 | 18.23 | 16.05 | 15.91 |
| coarse_sim | 24.84 | 23.52 | 22.52 | 21.84 | 21.14 | 19.24 | 16.90 | 16.58 |
| frozen | 29.96 | 20.60 | 17.60 | 16.05 | 15.50 | 15.49 | 14.95 | 15.08 |
| renderer_on_true_fields | 34.07 | 34.22 | 34.35 | 34.91 | 35.21 | 34.66 | 34.66 | 34.66 |

neural - coarse_sim_detail, paired over runs: frame 1: +4.62 [+3.81, +5.38]; frame 8: +2.77 [+2.10, +3.40]; frame 30: +2.98 [+2.56, +3.36]; frame 60: +3.99 [+3.50, +4.56]; frame 240: +3.95 [+3.47, +4.45];

neural_dynamics_true_renderer - coarse_sim_detail, paired over runs: frame 1: +5.74 [+4.96, +6.57]; frame 8: +2.79 [+2.14, +3.41]; frame 30: +2.91 [+2.49, +3.28]; frame 60: +3.93 [+3.45, +4.47]; frame 240: +3.85 [+3.41, +4.33];

### Endless runs at held-out settings

The ten held-out settings of study B, a new seed each, 10 s per run (explosions: their 3 s). Distances of each method's frame statistics to a real run of the simulation at that setting; `real_other_seed` is a second real run with another seed, the floor. Lower is better except motion ratio (1 is right) and mean-frame PSNR (higher is closer).

**coals**

| method | spectrum L1 | motion ratio | coverage L1 | emission L1 | mean-frame PSNR |
|---|---:|---:|---:|---:|---:|
| real_other_seed | 0.069 | 0.99 | 0.0043 | 0.0005 | 31.55 |
| neural | 0.137 | 0.94 | 0.0054 | 0.0007 | 29.70 |
| flipbook_nearest | 0.073 | 1.29 | 0.0044 | 0.0005 | 27.75 |
| coarse_sim_detail | 0.090 | 0.91 | 0.0123 | 0.0004 | 27.35 |
| coarse_sim | 0.456 | 0.72 | 0.0077 | 0.0005 | 29.86 |

Spectrum distance, paired over settings: neural - grid_k8 -; neural - flipbook_nearest +0.064 [+0.034, +0.095]; neural - coarse_sim_detail +0.047 [+0.022, +0.073]; neural - real_other_seed +0.068 [+0.032, +0.102].

Mean-frame PSNR (dB), paired: neural - flipbook_nearest +1.94 [+0.62, +3.15]; neural - coarse_sim_detail +2.35 [+1.39, +3.30]; neural - real_other_seed -1.85 [-2.73, -1.01]. Motion error |ln(motion ratio)|, paired: neural - flipbook_nearest -0.126 [-0.230, -0.021]; neural - coarse_sim_detail -0.041 [-0.103, +0.023] (tie); neural - real_other_seed +0.020 [-0.025, +0.062] (tie). Coverage distance, paired: neural - flipbook_nearest +0.0010 [-0.0004, +0.0024] (tie); neural - real_other_seed +0.0012 [+0.0005, +0.0019].

**mushroom**

| method | spectrum L1 | motion ratio | coverage L1 | emission L1 | mean-frame PSNR |
|---|---:|---:|---:|---:|---:|
| real_other_seed | 0.089 | 1.01 | 0.0173 | 0.0000 | 24.90 |
| neural | 0.260 | 0.78 | 0.0377 | 0.0001 | 23.03 |
| flipbook_nearest | 0.098 | 1.10 | 0.0196 | 0.0000 | 25.43 |
| coarse_sim_detail | 0.122 | 0.96 | 0.0125 | 0.0000 | 24.46 |
| coarse_sim | 0.875 | 0.68 | 0.0187 | 0.0001 | 24.40 |

Spectrum distance, paired over settings: neural - grid_k8 -; neural - flipbook_nearest +0.162 [+0.101, +0.244]; neural - coarse_sim_detail +0.138 [+0.056, +0.238]; neural - real_other_seed +0.171 [+0.094, +0.268].

Mean-frame PSNR (dB), paired: neural - flipbook_nearest -2.40 [-3.43, -1.52]; neural - coarse_sim_detail -1.43 [-2.43, -0.52]; neural - real_other_seed -1.87 [-2.96, -0.96]. Motion error |ln(motion ratio)|, paired: neural - flipbook_nearest +0.155 [-0.007, +0.317] (tie); neural - coarse_sim_detail +0.176 [+0.025, +0.324]; neural - real_other_seed +0.183 [+0.006, +0.367]. Coverage distance, paired: neural - flipbook_nearest +0.0181 [+0.0092, +0.0296]; neural - real_other_seed +0.0204 [+0.0050, +0.0355].

**wildfire**

| method | spectrum L1 | motion ratio | coverage L1 | emission L1 | mean-frame PSNR |
|---|---:|---:|---:|---:|---:|
| real_other_seed | 0.039 | 1.04 | 0.0252 | 0.0015 | 29.02 |
| neural | 0.159 | 0.95 | 0.0308 | 0.0022 | 26.01 |
| flipbook_nearest | 0.121 | 1.27 | 0.0551 | 0.0020 | 21.94 |
| coarse_sim_detail | 0.278 | 0.78 | 0.0524 | 0.0026 | 22.58 |
| coarse_sim | 0.740 | 0.40 | 0.0429 | 0.0027 | 22.49 |

Spectrum distance, paired over settings: neural - grid_k8 -; neural - flipbook_nearest +0.038 [-0.016, +0.088] (tie); neural - coarse_sim_detail -0.119 [-0.204, -0.043]; neural - real_other_seed +0.119 [+0.089, +0.152].

Mean-frame PSNR (dB), paired: neural - flipbook_nearest +4.08 [+2.62, +5.36]; neural - coarse_sim_detail +3.43 [+2.07, +4.67]; neural - real_other_seed -3.00 [-3.91, -2.19]. Motion error |ln(motion ratio)|, paired: neural - flipbook_nearest -0.088 [-0.206, +0.027] (tie); neural - coarse_sim_detail -0.100 [-0.181, -0.016]; neural - real_other_seed +0.107 [+0.047, +0.161]. Coverage distance, paired: neural - flipbook_nearest -0.0244 [-0.0356, -0.0116]; neural - real_other_seed +0.0055 [+0.0006, +0.0124].

### One minute of play

Statistics of each 10 s window of a 60 s neural run (shards of 6 s from start points, the default) against a real 10 s run at the same setting (two held-out settings): if the effect drifted, the distances would grow window by window.

| effect | setting | window | spectrum L1 | motion ratio | coverage L1 | mean-frame PSNR |
|---|---:|---:|---:|---:|---:|---:|
| wildfire | 0 | 0 | 0.1028 | 0.6849 | 0.0247 | 29.133 |
| wildfire | 0 | 1 | 0.1298 | 0.8144 | 0.0181 | 29.195 |
| wildfire | 0 | 2 | 0.1381 | 0.8954 | 0.0337 | 27.900 |
| wildfire | 0 | 3 | 0.1083 | 0.7257 | 0.0211 | 29.865 |
| wildfire | 0 | 4 | 0.1055 | 0.7165 | 0.0336 | 27.421 |
| wildfire | 0 | 5 | 0.1101 | 0.7430 | 0.0326 | 28.664 |
| wildfire | 1 | 0 | 0.2192 | 0.9782 | 0.0237 | 27.467 |
| wildfire | 1 | 1 | 0.2097 | 1.2066 | 0.0339 | 26.743 |
| wildfire | 1 | 2 | 0.2108 | 1.0312 | 0.0219 | 28.061 |
| wildfire | 1 | 3 | 0.1849 | 1.0433 | 0.0296 | 26.307 |
| wildfire | 1 | 4 | 0.2103 | 1.3248 | 0.0359 | 26.331 |
| wildfire | 1 | 5 | 0.1867 | 1.0454 | 0.0365 | 27.601 |
| coals | 0 | 0 | 0.0904 | 1.0166 | 0.0054 | 30.283 |
| coals | 0 | 1 | 0.1165 | 1.0288 | 0.0048 | 31.125 |
| coals | 0 | 2 | 0.0943 | 1.1131 | 0.0039 | 31.369 |
| coals | 0 | 3 | 0.1007 | 1.0854 | 0.0070 | 30.313 |
| coals | 0 | 4 | 0.0894 | 1.1632 | 0.0047 | 29.482 |
| coals | 0 | 5 | 0.0857 | 1.0508 | 0.0063 | 29.521 |
| coals | 1 | 0 | 0.1254 | 0.8932 | 0.0040 | 30.635 |
| coals | 1 | 1 | 0.0943 | 1.0624 | 0.0038 | 31.391 |
| coals | 1 | 2 | 0.0895 | 1.1006 | 0.0023 | 30.999 |
| coals | 1 | 3 | 0.0882 | 1.0744 | 0.0044 | 31.610 |
| coals | 1 | 4 | 0.1243 | 1.0638 | 0.0038 | 30.676 |
| coals | 1 | 5 | 0.1264 | 1.0354 | 0.0038 | 30.126 |

### Cost

Median (90th percentile) ms per frame through `nvfx_render` on one pinned core; `restart` is a seek (a restart from a start point, including any warm-up). The coarse simulation row is the 32-cell solver plus its renderer at 128 px (scalar code).

| model | ISA | size | ms | restart ms | MAC/px | KB stored / resident | KB scratch |
|---|---|---:|---:|---:|---:|---|---:|
| rollout_fire | avx2 | 64 | 0.4809 (4.8831) | 19.41 | 2510 | 81.6 / 162.8 | 1636.6 |
| rollout_fire | avx2 | 128 | 0.9136 (1.2281) | 15.07 | 1012 | 81.6 / 162.8 | 2489.4 |
| rollout_fire | avx2 | 256 | 2.4266 (6.5002) | 31.89 | 637 | 81.6 / 162.8 | 5826.9 |
| rollout_fire | baseline | 128 | 1.8363 (5.8711) | 44.06 | 1012 | 81.6 / 162.8 | 2526.5 |
| rollout_fire | avx512 | 128 | 0.7283 (1.3416) | 18.94 | 1012 | 81.6 / 162.8 | 2502.9 |
| coarse_sim_fire | scalar | 128 | 1.3661 (5.4481) | - | - | - | - |
| rollout_explosion | avx2 | 64 | 0.4558 (0.6172) | 0.11 | 2510 | 274.2 / 803.6 | 1636.6 |
| rollout_explosion | avx2 | 128 | 0.7927 (4.6521) | 0.25 | 1012 | 274.2 / 803.6 | 2489.4 |
| rollout_explosion | avx2 | 256 | 1.7839 (6.2206) | 4.89 | 637 | 274.2 / 803.6 | 5826.9 |
| rollout_explosion | baseline | 128 | 1.5086 (5.5184) | 0.37 | 1012 | 274.2 / 803.6 | 2526.6 |
| rollout_explosion | avx512 | 128 | 0.6808 (0.9030) | 0.29 | 1012 | 274.2 / 803.6 | 2503.0 |
| coarse_sim_explosion | scalar | 128 | 2.5683 (6.6041) | - | - | - | - |
| rollout_wildfire | avx2 | 64 | 0.4599 (0.9475) | 18.91 | 2510 | 81.6 / 162.8 | 1636.6 |
| rollout_wildfire | avx2 | 128 | 0.7106 (4.7106) | 21.44 | 1012 | 81.6 / 162.8 | 2489.4 |
| rollout_wildfire | avx2 | 256 | 1.7329 (5.8486) | 36.87 | 637 | 81.6 / 162.8 | 5826.9 |
| rollout_wildfire | baseline | 128 | 1.4350 (5.5448) | 38.97 | 1012 | 81.6 / 162.8 | 2526.5 |
| rollout_wildfire | avx512 | 128 | 0.4978 (1.0550) | 13.94 | 1012 | 81.6 / 162.8 | 2502.9 |
| coarse_sim_wildfire | scalar | 128 | 2.4557 (8.3787) | - | - | - | - |
| rollout_coals | avx2 | 64 | 0.6269 (1.2185) | 14.91 | 2510 | 81.6 / 162.8 | 1636.6 |
| rollout_coals | avx2 | 128 | 0.9808 (5.0093) | 17.17 | 1012 | 81.6 / 162.8 | 2489.4 |
| rollout_coals | avx2 | 256 | 2.0760 (6.2512) | 27.67 | 637 | 81.6 / 162.8 | 5826.9 |
| rollout_coals | baseline | 128 | 1.7212 (5.7847) | 42.63 | 1012 | 81.6 / 162.8 | 2526.5 |
| rollout_coals | avx512 | 128 | 0.7700 (4.7811) | 13.54 | 1012 | 81.6 / 162.8 | 2502.9 |
| coarse_sim_coals | scalar | 128 | 8.9636 (9.6921) | - | - | - | - |
| rollout_mushroom | avx2 | 64 | 0.5458 (0.7714) | 0.10 | 2510 | 274.2 / 803.6 | 1636.6 |
| rollout_mushroom | avx2 | 128 | 1.1335 (4.9095) | 0.29 | 1012 | 274.2 / 803.6 | 2489.4 |
| rollout_mushroom | avx2 | 256 | 1.9095 (6.5408) | 0.87 | 637 | 274.2 / 803.6 | 5826.9 |
| rollout_mushroom | baseline | 128 | 1.8627 (5.9073) | 0.28 | 1012 | 274.2 / 803.6 | 2526.6 |
| rollout_mushroom | avx512 | 128 | 0.7527 (0.9435) | 0.21 | 1012 | 274.2 / 803.6 | 2503.0 |
| coarse_sim_mushroom | scalar | 128 | 2.8937 (6.8445) | - | - | - | - |
