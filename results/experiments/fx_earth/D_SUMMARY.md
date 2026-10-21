
## D: start points and learned dynamics

### Chaos horizon (the simulation itself)

Active PSNR against a run of the simulation from a stored start point, by frames after it (30 per second), mean over runs. Perturbations are relative to the velocity RMS.

| effect | case | 1 | 4 | 8 | 15 | 30 | 60 | 120 | 240 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lava | start+1e-3_same_seed | 84.73 | 81.07 | 81.52 | 80.83 | 81.90 | 84.31 | 95.50 | 93.84 |
| lava | start+1e-1_same_seed | 65.80 | 60.67 | 59.46 | 59.41 | 61.19 | 64.50 | 68.60 | 73.71 |
| lava | same_start_other_seed | 32.56 | 22.13 | 18.27 | 15.57 | 14.01 | 12.50 | 11.62 | 11.11 |
| lava | other_start_same_seed | 12.01 | 12.76 | 13.60 | 14.87 | 17.12 | 20.82 | 27.83 | 39.35 |
| eruption | start+1e-3_same_seed | 77.15 | 72.65 | 70.65 | 69.18 | 68.61 | 66.76 | 60.90 | 51.42 |
| eruption | start+1e-1_same_seed | 53.80 | 47.31 | 44.39 | 42.02 | 39.97 | 36.95 | 29.18 | 22.49 |
| eruption | same_start_other_seed | 30.51 | 26.76 | 23.70 | 20.45 | 16.11 | 12.70 | 11.15 | 11.20 |
| eruption | other_start_same_seed | 10.69 | 10.67 | 10.65 | 10.59 | 10.47 | 10.49 | 11.39 | 11.71 |
| avalanche | start+1e-3_same_seed | 94.15 | 88.93 | 87.27 | 89.32 | 88.47 | 87.08 | 88.16 | - |
| avalanche | start+1e-1_same_seed | 73.36 | 64.56 | 63.54 | 61.45 | 61.87 | 63.09 | 61.29 | - |
| avalanche | same_start_other_seed | 22.89 | 17.55 | 17.27 | 17.43 | 16.01 | 12.35 | 16.92 | - |
| avalanche | other_start_same_seed | 21.05 | 23.20 | 24.42 | 25.78 | 27.34 | 28.63 | 29.64 | - |

### Training

Stepper loss: stages 1-2 (normalised squared error plus profiles) and stage 3 (plus activity). Renderer PSNR on its training samples (true fields). Detail constants as calibrated.

| effect | runs x frames | minutes of simulation | stepper loss 1-2 | stepper s | stage 3 loss | stage 3 s | renderer dB | contrast | swirl | grow | start points | KB stored |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lava | 160 x 240 | 21.33 | 0.06704 | 5236 | 0.06308 | 1247 | 34.74 | 0.5 | 0.5 | 1 | 8 | 145.7 |
| eruption | 160 x 240 | 21.33 | 0.06857 | 6238 | 0.06594 | 1235 | 29.05 | 1 | 3 | 1 | 8 | 145.7 |
| avalanche | 240 x 150 | 20.00 | 0.01966 | 4685 | 0.02022 | 1112 | 35.46 | 0.5 | 1.5 | 1 | 16 | 274.2 |

Renderer, detail constants and start points from `d-finish` on the trained steppers; stage 3 from `d-tune`.

### Tracking a held-out run from its true start point

Active PSNR against the true run, mean over held-out runs (other seeds and settings), by frames after the start point. `neural`: the rollout effect through the runtime, started from the true state (coarse and fine) with the run's noise seed. `coarse_sim_detail`: the simulation on the same 32-cell grid with the same detail layer, drawn by the simulation's own renderer. `neural_dynamics_true_renderer`: the neural dynamics drawn by that renderer too (separates dynamics from rendering). `renderer_on_true_fields`: the learned renderer on the true fields (frames 1-30, then their mean).

**avalanche**

| method | 1 | 4 | 8 | 16 | 30 | 60 | 120 | 240 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| neural | 24.34 | 20.54 | 17.41 | 14.47 | 12.69 | 11.75 | 13.30 | - |
| neural_dynamics_true_renderer | 25.09 | 21.31 | 17.73 | 14.63 | 12.73 | 11.76 | 13.35 | - |
| coarse_sim_detail | 22.59 | 18.68 | 17.20 | 15.67 | 14.93 | 14.84 | 16.39 | - |
| coarse_sim | 20.75 | 18.95 | 18.23 | 17.07 | 16.01 | 15.63 | 17.45 | - |
| frozen | 16.12 | 10.10 | 8.10 | 6.62 | 5.81 | 5.60 | 7.45 | - |
| renderer_on_true_fields | 33.47 | 31.08 | 29.56 | 29.19 | 29.37 | 29.72 | 29.72 | 29.72 |

neural - coarse_sim_detail, paired over runs: frame 1: +1.75 [+0.70, +2.70]; frame 8: +0.21 [-1.96, +2.23] (tie); frame 30: -2.23 [-4.42, -0.07]; frame 60: -3.10 [-5.17, -0.96];

neural_dynamics_true_renderer - coarse_sim_detail, paired over runs: frame 1: +2.50 [+1.36, +3.53]; frame 8: +0.53 [-1.66, +2.54] (tie); frame 30: -2.20 [-4.37, -0.04]; frame 60: -3.08 [-5.16, -0.96];

**eruption**

| method | 1 | 4 | 8 | 16 | 30 | 60 | 120 | 240 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| neural | 24.20 | 20.57 | 17.53 | 15.86 | 15.28 | 12.95 | 11.66 | 10.68 |
| neural_dynamics_true_renderer | 26.70 | 21.21 | 17.73 | 15.91 | 15.34 | 13.01 | 11.71 | 10.72 |
| coarse_sim_detail | 24.12 | 20.62 | 17.27 | 15.45 | 14.78 | 13.17 | 11.55 | 10.81 |
| coarse_sim | 16.90 | 16.68 | 16.38 | 16.04 | 15.43 | 13.69 | 12.01 | 11.04 |
| frozen | 27.55 | 17.92 | 14.43 | 11.81 | 10.24 | 9.32 | 8.68 | 8.93 |
| renderer_on_true_fields | 27.84 | 27.55 | 27.38 | 27.81 | 27.92 | 27.77 | 27.77 | 27.77 |

neural - coarse_sim_detail, paired over runs: frame 1: +0.08 [-1.18, +1.25] (tie); frame 8: +0.26 [-0.08, +0.58] (tie); frame 30: +0.50 [+0.06, +0.96]; frame 60: -0.22 [-0.75, +0.35] (tie); frame 240: -0.13 [-1.20, +0.75] (tie);

neural_dynamics_true_renderer - coarse_sim_detail, paired over runs: frame 1: +2.58 [+1.85, +3.22]; frame 8: +0.46 [+0.09, +0.80]; frame 30: +0.55 [+0.09, +1.03]; frame 60: -0.16 [-0.69, +0.42] (tie); frame 240: -0.09 [-1.17, +0.78] (tie);

**lava**

| method | 1 | 4 | 8 | 16 | 30 | 60 | 120 | 240 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| neural | 30.15 | 25.97 | 21.98 | 18.20 | 16.13 | 14.53 | 13.98 | 13.88 |
| neural_dynamics_true_renderer | 36.31 | 27.41 | 22.31 | 18.17 | 15.96 | 14.45 | 13.89 | 13.81 |
| coarse_sim_detail | 24.87 | 22.32 | 19.56 | 16.72 | 15.03 | 13.89 | 13.50 | 13.39 |
| coarse_sim | 17.85 | 17.62 | 17.29 | 17.04 | 16.89 | 16.23 | 16.18 | 16.27 |
| frozen | 38.49 | 27.12 | 22.23 | 17.74 | 14.30 | 11.80 | 10.91 | 10.71 |
| renderer_on_true_fields | 31.11 | 31.06 | 31.03 | 30.94 | 31.10 | 31.01 | 31.01 | 31.01 |

neural - coarse_sim_detail, paired over runs: frame 1: +5.28 [+4.22, +6.30]; frame 8: +2.42 [+2.13, +2.77]; frame 30: +1.09 [+0.70, +1.43]; frame 60: +0.63 [+0.18, +1.01]; frame 240: +0.49 [-0.26, +1.20] (tie);

neural_dynamics_true_renderer - coarse_sim_detail, paired over runs: frame 1: +11.44 [+10.42, +12.38]; frame 8: +2.75 [+2.47, +3.11]; frame 30: +0.93 [+0.56, +1.24]; frame 60: +0.56 [+0.12, +0.91]; frame 240: +0.42 [-0.33, +1.15] (tie);

### Endless runs at held-out settings

The ten held-out settings of study B, a new seed each, 10 s per run (explosions: their 3 s). Distances of each method's frame statistics to a real run of the simulation at that setting; `real_other_seed` is a second real run with another seed, the floor. Lower is better except motion ratio (1 is right) and mean-frame PSNR (higher is closer).

**avalanche**

| method | spectrum L1 | motion ratio | coverage L1 | emission L1 | mean-frame PSNR |
|---|---:|---:|---:|---:|---:|
| real_other_seed | 0.124 | 0.99 | 0.0052 | 0.0000 | 30.29 |
| neural | 0.167 | 0.96 | 0.0228 | 0.0001 | 24.64 |
| flipbook_nearest | 0.146 | 0.97 | 0.0125 | 0.0000 | 27.55 |
| coarse_sim_detail | 0.165 | 0.93 | 0.0064 | 0.0000 | 29.73 |
| coarse_sim | 0.583 | 0.61 | 0.0060 | 0.0000 | 29.98 |

Spectrum distance, paired over settings: neural - grid_k8 -; neural - flipbook_nearest +0.021 [-0.046, +0.084] (tie); neural - coarse_sim_detail +0.002 [-0.056, +0.072] (tie); neural - real_other_seed +0.043 [-0.028, +0.114] (tie).

Mean-frame PSNR (dB), paired: neural - flipbook_nearest -2.90 [-5.71, -0.89]; neural - coarse_sim_detail -5.09 [-7.87, -2.89]; neural - real_other_seed -5.65 [-7.67, -3.68]. Motion error |ln(motion ratio)|, paired: neural - flipbook_nearest +0.155 [-0.320, +0.651] (tie); neural - coarse_sim_detail +0.318 [+0.013, +0.680]; neural - real_other_seed +0.367 [+0.013, +0.760]. Coverage distance, paired: neural - flipbook_nearest +0.0103 [-0.0045, +0.0301] (tie); neural - real_other_seed +0.0176 [+0.0022, +0.0375].

**eruption**

| method | spectrum L1 | motion ratio | coverage L1 | emission L1 | mean-frame PSNR |
|---|---:|---:|---:|---:|---:|
| real_other_seed | 0.078 | 0.99 | 0.0284 | 0.0000 | 21.66 |
| neural | 0.154 | 1.09 | 0.0436 | 0.0000 | 20.01 |
| flipbook_nearest | 0.172 | 1.04 | 0.0529 | 0.0000 | 18.70 |
| coarse_sim_detail | 0.200 | 1.83 | 0.0536 | 0.0000 | 18.90 |
| coarse_sim | 0.760 | 0.44 | 0.0946 | 0.0000 | 17.64 |

Spectrum distance, paired over settings: neural - grid_k8 -; neural - flipbook_nearest -0.018 [-0.072, +0.034] (tie); neural - coarse_sim_detail -0.046 [-0.108, +0.021] (tie); neural - real_other_seed +0.076 [+0.039, +0.114].

Mean-frame PSNR (dB), paired: neural - flipbook_nearest +1.31 [+0.04, +2.55]; neural - coarse_sim_detail +1.11 [-0.22, +2.57] (tie); neural - real_other_seed -1.65 [-3.41, -0.05]. Motion error |ln(motion ratio)|, paired: neural - flipbook_nearest +0.001 [-0.089, +0.078] (tie); neural - coarse_sim_detail -0.422 [-0.540, -0.294]; neural - real_other_seed +0.043 [-0.003, +0.088] (tie). Coverage distance, paired: neural - flipbook_nearest -0.0093 [-0.0178, -0.0014]; neural - real_other_seed +0.0152 [+0.0009, +0.0305].

**lava**

| method | spectrum L1 | motion ratio | coverage L1 | emission L1 | mean-frame PSNR |
|---|---:|---:|---:|---:|---:|
| real_other_seed | 0.048 | 1.00 | 0.0064 | 0.0003 | 24.61 |
| neural | 0.143 | 1.19 | 0.0261 | 0.0005 | 21.89 |
| flipbook_nearest | 0.104 | 1.38 | 0.0141 | 0.0003 | 20.67 |
| coarse_sim_detail | 0.234 | 1.46 | 0.0131 | 0.0014 | 22.22 |
| coarse_sim | 0.772 | 0.71 | 0.0060 | 0.0009 | 23.47 |

Spectrum distance, paired over settings: neural - grid_k8 -; neural - flipbook_nearest +0.039 [+0.014, +0.061]; neural - coarse_sim_detail -0.091 [-0.125, -0.059]; neural - real_other_seed +0.095 [+0.066, +0.125].

Mean-frame PSNR (dB), paired: neural - flipbook_nearest +1.22 [+0.11, +2.45]; neural - coarse_sim_detail -0.33 [-1.22, +0.38] (tie); neural - real_other_seed -2.72 [-4.33, -1.22]. Motion error |ln(motion ratio)|, paired: neural - flipbook_nearest -0.069 [-0.205, +0.060] (tie); neural - coarse_sim_detail -0.127 [-0.239, -0.026]; neural - real_other_seed +0.163 [+0.071, +0.250]. Coverage distance, paired: neural - flipbook_nearest +0.0120 [+0.0024, +0.0200]; neural - real_other_seed +0.0197 [+0.0157, +0.0232].

### One minute of play

Statistics of each 10 s window of a 60 s neural run (shards of 6 s from start points, the default) against a real 10 s run at the same setting (two held-out settings): if the effect drifted, the distances would grow window by window.

| effect | setting | window | spectrum L1 | motion ratio | coverage L1 | mean-frame PSNR |
|---|---:|---:|---:|---:|---:|---:|
| lava | 0 | 0 | 0.1082 | 1.3760 | 0.0281 | 22.527 |
| lava | 0 | 1 | 0.1520 | 1.4827 | 0.0258 | 22.566 |
| lava | 0 | 2 | 0.1070 | 1.3852 | 0.0117 | 25.515 |
| lava | 0 | 3 | 0.1178 | 1.3637 | 0.0292 | 22.194 |
| lava | 0 | 4 | 0.1159 | 1.3944 | 0.0181 | 23.637 |
| lava | 0 | 5 | 0.1309 | 1.3654 | 0.0260 | 23.767 |
| lava | 1 | 0 | 0.0890 | 1.0918 | 0.0223 | 22.222 |
| lava | 1 | 1 | 0.1152 | 1.6312 | 0.0196 | 23.916 |
| lava | 1 | 2 | 0.0950 | 1.3852 | 0.0266 | 24.727 |
| lava | 1 | 3 | 0.0974 | 1.0947 | 0.0233 | 23.013 |
| lava | 1 | 4 | 0.1615 | 1.7630 | 0.0260 | 21.347 |
| lava | 1 | 5 | 0.1149 | 1.4894 | 0.0230 | 24.008 |
| eruption | 0 | 0 | 0.1162 | 1.0579 | 0.0426 | 20.825 |
| eruption | 0 | 1 | 0.1226 | 1.1701 | 0.0391 | 20.538 |
| eruption | 0 | 2 | 0.1200 | 1.1207 | 0.0370 | 21.659 |
| eruption | 0 | 3 | 0.1192 | 1.1107 | 0.0461 | 20.742 |
| eruption | 0 | 4 | 0.1160 | 1.0028 | 0.0443 | 22.571 |
| eruption | 0 | 5 | 0.1150 | 1.1062 | 0.0568 | 21.857 |
| eruption | 1 | 0 | 0.1346 | 1.1199 | 0.0223 | 22.731 |
| eruption | 1 | 1 | 0.1264 | 1.2214 | 0.0400 | 18.864 |
| eruption | 1 | 2 | 0.1063 | 1.3140 | 0.0540 | 23.373 |
| eruption | 1 | 3 | 0.1992 | 1.0576 | 0.0367 | 20.898 |
| eruption | 1 | 4 | 0.1069 | 1.3923 | 0.0396 | 18.355 |
| eruption | 1 | 5 | 0.1565 | 1.2145 | 0.0495 | 19.023 |
