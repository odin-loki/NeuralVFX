
## D: start points and learned dynamics

### Chaos horizon (the simulation itself)

Active PSNR against a run of the simulation from a stored start point, by frames after it (30 per second), mean over runs. Perturbations are relative to the velocity RMS.

| effect | case | 1 | 4 | 8 | 15 | 30 | 60 | 120 | 240 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| launch | start+1e-3_same_seed | 71.08 | 67.60 | 65.95 | 64.58 | 62.93 | 60.06 | 55.58 | 50.11 |
| launch | start+1e-1_same_seed | 46.09 | 40.17 | 37.26 | 34.88 | 32.05 | 28.52 | 25.48 | 23.27 |
| launch | same_start_other_seed | 37.79 | 31.16 | 23.72 | 17.56 | 13.03 | 10.41 | 9.62 | 10.33 |
| launch | other_start_same_seed | 10.35 | 10.28 | 10.22 | 10.19 | 10.19 | 10.44 | 11.36 | 13.53 |
| puffs | start+1e-3_same_seed | 72.68 | 68.54 | 67.43 | 65.79 | 65.02 | 63.98 | 62.78 | 62.44 |
| puffs | start+1e-1_same_seed | 51.35 | 44.90 | 42.14 | 39.74 | 37.65 | 36.41 | 32.96 | 34.13 |
| puffs | same_start_other_seed | 49.68 | 24.93 | 21.18 | 17.13 | 13.78 | 12.30 | 11.74 | 13.82 |
| puffs | other_start_same_seed | 9.52 | 9.64 | 9.78 | 10.11 | 10.85 | 13.21 | 16.35 | 21.74 |
| ink | start+1e-3_same_seed | 83.80 | 76.18 | 75.65 | 73.33 | 74.61 | 74.09 | 74.23 | 79.08 |
| ink | start+1e-1_same_seed | 60.09 | 54.36 | 52.04 | 50.37 | 49.67 | 50.34 | 52.02 | 57.86 |
| ink | same_start_other_seed | 31.02 | 27.73 | 24.94 | 21.40 | 17.75 | 15.14 | 14.91 | 15.63 |
| ink | other_start_same_seed | 14.00 | 13.97 | 13.96 | 14.01 | 14.30 | 14.79 | 16.93 | 22.04 |

### Training

Stepper loss: stages 1-2 (normalised squared error plus profiles) and stage 3 (plus activity). Renderer PSNR on its training samples (true fields). Detail constants as calibrated.

| effect | runs x frames | minutes of simulation | stepper loss 1-2 | stepper s | stage 3 loss | stage 3 s | renderer dB | contrast | swirl | grow | start points | KB stored |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| launch | 160 x 240 | 21.33 | 0.04987 | 6773 | 0.05152 | 1203 | 31.20 | 1 | 3 | 1 | 8 | 145.7 |
| puffs | 160 x 240 | 21.33 | 0.18094 | 8472 | 0.50446 | 2096 | 31.95 | 1 | 1.5 | 1 | 8 | 145.7 |
| ink | 160 x 240 | 21.33 | 0.01820 | 8307 | 0.01694 | 2176 | 42.90 | 0.5 | 1.5 | 1 | 8 | 145.7 |

Renderer, detail constants and start points from `d-finish` on the trained steppers; stage 3 from `d-tune`.

### Tracking a held-out run from its true start point

Active PSNR against the true run, mean over held-out runs (other seeds and settings), by frames after the start point. `neural`: the rollout effect through the runtime, started from the true state (coarse and fine) with the run's noise seed. `coarse_sim_detail`: the simulation on the same 32-cell grid with the same detail layer, drawn by the simulation's own renderer. `neural_dynamics_true_renderer`: the neural dynamics drawn by that renderer too (separates dynamics from rendering). `renderer_on_true_fields`: the learned renderer on the true fields (frames 1-30, then their mean).

**ink**

| method | 1 | 4 | 8 | 16 | 30 | 60 | 120 | 240 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| neural | 35.26 | 29.95 | 25.92 | 23.13 | 21.93 | 20.14 | 18.70 | 17.38 |
| neural_dynamics_true_renderer | 39.28 | 30.78 | 26.15 | 23.22 | 22.05 | 20.19 | 18.69 | 17.35 |
| coarse_sim_detail | 29.36 | 27.83 | 24.61 | 21.81 | 19.76 | 18.34 | 17.22 | 17.82 |
| coarse_sim | 23.15 | 23.22 | 23.03 | 22.22 | 20.83 | 19.44 | 18.36 | 18.69 |
| frozen | 37.68 | 26.16 | 20.95 | 16.80 | 14.36 | 12.97 | 12.54 | 12.87 |
| renderer_on_true_fields | 37.84 | 37.81 | 37.81 | 37.85 | 37.81 | 37.82 | 37.82 | 37.82 |

neural - coarse_sim_detail, paired over runs: frame 1: +5.89 [+3.10, +8.34]; frame 8: +1.31 [+1.01, +1.61]; frame 30: +2.18 [+1.86, +2.50]; frame 60: +1.80 [+1.62, +1.99]; frame 240: -0.44 [-1.41, +0.46] (tie);

neural_dynamics_true_renderer - coarse_sim_detail, paired over runs: frame 1: +9.92 [+7.26, +12.24]; frame 8: +1.54 [+1.19, +1.90]; frame 30: +2.29 [+1.95, +2.64]; frame 60: +1.85 [+1.64, +2.06]; frame 240: -0.47 [-1.41, +0.39] (tie);

**launch**

| method | 1 | 4 | 8 | 16 | 30 | 60 | 120 | 240 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| neural | 22.76 | 18.42 | 15.79 | 14.03 | 13.23 | 12.02 | 11.27 | 10.43 |
| neural_dynamics_true_renderer | 23.86 | 18.69 | 15.87 | 14.03 | 13.20 | 11.98 | 11.25 | 10.30 |
| coarse_sim_detail | 24.95 | 17.92 | 14.85 | 12.71 | 11.70 | 10.62 | 10.02 | 10.11 |
| coarse_sim | 16.66 | 16.32 | 15.73 | 14.65 | 13.72 | 12.22 | 11.40 | 11.15 |
| frozen | 22.72 | 14.87 | 12.39 | 10.44 | 9.03 | 7.86 | 7.42 | 7.97 |
| renderer_on_true_fields | 28.58 | 28.60 | 28.68 | 28.83 | 28.87 | 28.77 | 28.77 | 28.77 |

neural - coarse_sim_detail, paired over runs: frame 1: -2.19 [-2.82, -1.56]; frame 8: +0.94 [+0.78, +1.12]; frame 30: +1.53 [+1.09, +2.04]; frame 60: +1.40 [+1.03, +1.74]; frame 240: +0.32 [-0.06, +0.75] (tie);

neural_dynamics_true_renderer - coarse_sim_detail, paired over runs: frame 1: -1.09 [-1.66, -0.59]; frame 8: +1.02 [+0.84, +1.20]; frame 30: +1.50 [+1.05, +2.03]; frame 60: +1.36 [+0.99, +1.71]; frame 240: +0.20 [-0.19, +0.62] (tie);

**puffs**

| method | 1 | 4 | 8 | 16 | 30 | 60 | 120 | 240 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| neural | 25.47 | 20.38 | 17.64 | 15.61 | 14.42 | 13.62 | 13.35 | 13.42 |
| neural_dynamics_true_renderer | 29.00 | 21.20 | 17.91 | 15.61 | 14.40 | 13.64 | 13.40 | 13.51 |
| coarse_sim_detail | 29.72 | 21.31 | 17.76 | 15.37 | 14.79 | 14.59 | 14.44 | 15.49 |
| coarse_sim | 18.81 | 17.82 | 17.11 | 16.34 | 16.20 | 15.83 | 15.50 | 16.54 |
| frozen | 22.41 | 13.69 | 11.71 | 11.03 | 10.87 | 10.50 | 10.97 | 11.91 |
| renderer_on_true_fields | 28.57 | 28.51 | 28.58 | 29.10 | 29.61 | 29.05 | 29.05 | 29.05 |

neural - coarse_sim_detail, paired over runs: frame 1: -4.26 [-7.12, -1.41]; frame 8: -0.12 [-1.38, +1.22] (tie); frame 30: -0.37 [-1.12, +0.45] (tie); frame 60: -0.97 [-1.52, -0.50]; frame 240: -2.07 [-2.76, -1.44];

neural_dynamics_true_renderer - coarse_sim_detail, paired over runs: frame 1: -0.72 [-2.82, +1.62] (tie); frame 8: +0.15 [-1.14, +1.54] (tie); frame 30: -0.39 [-1.20, +0.51] (tie); frame 60: -0.94 [-1.53, -0.44]; frame 240: -1.98 [-2.68, -1.36];

### Endless runs at held-out settings

The ten held-out settings of study B, a new seed each, 10 s per run (explosions: their 3 s). Distances of each method's frame statistics to a real run of the simulation at that setting; `real_other_seed` is a second real run with another seed, the floor. Lower is better except motion ratio (1 is right) and mean-frame PSNR (higher is closer).

**ink**

| method | spectrum L1 | motion ratio | coverage L1 | emission L1 | mean-frame PSNR |
|---|---:|---:|---:|---:|---:|
| real_other_seed | 0.088 | 0.95 | 0.0092 | 0.0000 | 28.24 |
| neural | 0.198 | 1.08 | 0.0246 | 0.0000 | 24.79 |
| flipbook_nearest | 0.226 | 1.14 | 0.0251 | 0.0000 | 24.25 |
| coarse_sim_detail | 0.164 | 1.04 | 0.0103 | 0.0000 | 27.84 |
| coarse_sim | 1.026 | 0.42 | 0.0087 | 0.0000 | 27.98 |

Spectrum distance, paired over settings: neural - grid_k8 -; neural - flipbook_nearest -0.028 [-0.094, +0.034] (tie); neural - coarse_sim_detail +0.034 [-0.011, +0.086] (tie); neural - real_other_seed +0.110 [+0.079, +0.149].

Mean-frame PSNR (dB), paired: neural - flipbook_nearest +0.54 [-1.08, +2.01] (tie); neural - coarse_sim_detail -3.05 [-4.46, -1.89]; neural - real_other_seed -3.45 [-5.17, -2.00]. Motion error |ln(motion ratio)|, paired: neural - flipbook_nearest -0.112 [-0.241, +0.026] (tie); neural - coarse_sim_detail +0.072 [+0.014, +0.136]; neural - real_other_seed +0.074 [-0.027, +0.202] (tie). Coverage distance, paired: neural - flipbook_nearest -0.0005 [-0.0108, +0.0085] (tie); neural - real_other_seed +0.0154 [+0.0113, +0.0193].

**launch**

| method | spectrum L1 | motion ratio | coverage L1 | emission L1 | mean-frame PSNR |
|---|---:|---:|---:|---:|---:|
| real_other_seed | 0.039 | 1.04 | 0.0183 | 0.0006 | 20.81 |
| neural | 0.122 | 1.19 | 0.0813 | 0.0011 | 16.36 |
| flipbook_nearest | 0.093 | 1.10 | 0.0603 | 0.0010 | 16.10 |
| coarse_sim_detail | 0.096 | 1.20 | 0.0378 | 0.0008 | 17.78 |
| coarse_sim | 0.883 | 0.27 | 0.0213 | 0.0008 | 17.98 |

Spectrum distance, paired over settings: neural - grid_k8 -; neural - flipbook_nearest +0.029 [+0.003, +0.056]; neural - coarse_sim_detail +0.026 [+0.004, +0.049]; neural - real_other_seed +0.083 [+0.058, +0.120].

Mean-frame PSNR (dB), paired: neural - flipbook_nearest +0.26 [-1.28, +1.77] (tie); neural - coarse_sim_detail -1.42 [-2.78, +0.14] (tie); neural - real_other_seed -4.44 [-5.67, -3.14]. Motion error |ln(motion ratio)|, paired: neural - flipbook_nearest +0.032 [-0.020, +0.092] (tie); neural - coarse_sim_detail -0.055 [-0.180, +0.085] (tie); neural - real_other_seed +0.139 [+0.057, +0.255]. Coverage distance, paired: neural - flipbook_nearest +0.0210 [-0.0005, +0.0435] (tie); neural - real_other_seed +0.0630 [+0.0400, +0.0854].

**puffs**

| method | spectrum L1 | motion ratio | coverage L1 | emission L1 | mean-frame PSNR |
|---|---:|---:|---:|---:|---:|
| real_other_seed | 0.083 | 0.96 | 0.0209 | 0.0000 | 26.33 |
| neural | 0.193 | 1.30 | 0.0344 | 0.0003 | 23.93 |
| flipbook_nearest | 0.138 | 1.04 | 0.0240 | 0.0000 | 22.33 |
| coarse_sim_detail | 0.128 | 0.98 | 0.0235 | 0.0000 | 24.43 |
| coarse_sim | 1.187 | 0.44 | 0.0292 | 0.0000 | 23.92 |

Spectrum distance, paired over settings: neural - grid_k8 -; neural - flipbook_nearest +0.055 [-0.037, +0.142] (tie); neural - coarse_sim_detail +0.065 [+0.011, +0.118]; neural - real_other_seed +0.110 [+0.061, +0.161].

Mean-frame PSNR (dB), paired: neural - flipbook_nearest +1.60 [+0.60, +2.46]; neural - coarse_sim_detail -0.50 [-1.55, +0.44] (tie); neural - real_other_seed -2.40 [-3.49, -1.02]. Motion error |ln(motion ratio)|, paired: neural - flipbook_nearest +0.046 [-0.100, +0.164] (tie); neural - coarse_sim_detail +0.067 [-0.034, +0.171] (tie); neural - real_other_seed +0.117 [+0.001, +0.236]. Coverage distance, paired: neural - flipbook_nearest +0.0104 [-0.0014, +0.0225] (tie); neural - real_other_seed +0.0135 [+0.0028, +0.0251].

### One minute of play

Statistics of each 10 s window of a 60 s neural run (shards of 6 s from start points, the default) against a real 10 s run at the same setting (two held-out settings): if the effect drifted, the distances would grow window by window.

| effect | setting | window | spectrum L1 | motion ratio | coverage L1 | mean-frame PSNR |
|---|---:|---:|---:|---:|---:|---:|
| launch | 0 | 0 | 0.0855 | 1.4195 | 0.0531 | 17.093 |
| launch | 0 | 1 | 0.0867 | 1.4717 | 0.0893 | 17.095 |
| launch | 0 | 2 | 0.0792 | 1.5067 | 0.0694 | 17.590 |
| launch | 0 | 3 | 0.0851 | 1.3429 | 0.0414 | 18.391 |
| launch | 0 | 4 | 0.0755 | 1.3663 | 0.0646 | 18.257 |
| launch | 0 | 5 | 0.0794 | 1.4384 | 0.0504 | 19.098 |
| launch | 1 | 0 | 0.0742 | 1.1359 | 0.1178 | 14.966 |
| launch | 1 | 1 | 0.1055 | 1.1699 | 0.1145 | 16.524 |
| launch | 1 | 2 | 0.1051 | 1.1749 | 0.0982 | 16.075 |
| launch | 1 | 3 | 0.0551 | 1.5217 | 0.0505 | 18.186 |
| launch | 1 | 4 | 0.0942 | 1.2530 | 0.1369 | 15.135 |
| launch | 1 | 5 | 0.0640 | 1.2115 | 0.1095 | 17.228 |
| ink | 0 | 0 | 0.1972 | 1.2448 | 0.0311 | 25.552 |
| ink | 0 | 1 | 0.1921 | 0.9312 | 0.0160 | 28.601 |
| ink | 0 | 2 | 0.1883 | 0.9506 | 0.0139 | 28.269 |
| ink | 0 | 3 | 0.1626 | 1.2139 | 0.0277 | 25.804 |
| ink | 0 | 4 | 0.1662 | 0.9129 | 0.0128 | 26.772 |
| ink | 0 | 5 | 0.2151 | 0.8744 | 0.0151 | 28.274 |
| ink | 1 | 0 | 0.1838 | 1.0328 | 0.0123 | 25.407 |
| ink | 1 | 1 | 0.1492 | 1.8954 | 0.0130 | 26.941 |
| ink | 1 | 2 | 0.1424 | 1.4241 | 0.0114 | 28.658 |
| ink | 1 | 3 | 0.2424 | 1.7549 | 0.0054 | 26.980 |
| ink | 1 | 4 | 0.1529 | 2.0410 | 0.0106 | 26.995 |
| ink | 1 | 5 | 0.1407 | 1.8880 | 0.0172 | 28.274 |
| puffs | 0 | 0 | 0.1202 | 1.3141 | 0.0457 | 21.183 |
| puffs | 0 | 1 | 0.2177 | 1.1032 | 0.0203 | 25.738 |
| puffs | 0 | 2 | 0.2305 | 1.1780 | 0.0240 | 25.374 |
| puffs | 0 | 3 | 0.1223 | 1.3402 | 0.0440 | 21.473 |
| puffs | 0 | 4 | 0.2215 | 1.1539 | 0.0235 | 24.612 |
| puffs | 0 | 5 | 0.2509 | 1.1013 | 0.0225 | 24.754 |
| puffs | 1 | 0 | 0.1655 | 1.3421 | 0.0142 | 25.861 |
| puffs | 1 | 1 | 0.1260 | 1.4370 | 0.0262 | 26.063 |
| puffs | 1 | 2 | 0.2021 | 1.1990 | 0.0181 | 27.793 |
| puffs | 1 | 3 | 0.1063 | 1.7258 | 0.0203 | 23.907 |
| puffs | 1 | 4 | 0.1394 | 1.5723 | 0.0380 | 23.570 |
| puffs | 1 | 5 | 0.1233 | 1.5112 | 0.0221 | 28.017 |
