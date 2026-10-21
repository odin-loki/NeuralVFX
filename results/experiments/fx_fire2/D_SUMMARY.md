
## D: start points and learned dynamics

### Chaos horizon (the simulation itself)

Active PSNR against a run of the simulation from a stored start point, by frames after it (30 per second), mean over runs. Perturbations are relative to the velocity RMS.

| effect | case | 1 | 4 | 8 | 15 | 30 | 60 | 120 | 240 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| crownfire | start+1e-3_same_seed | 72.59 | 69.04 | 67.46 | 65.77 | 64.32 | 61.74 | 52.48 | 41.08 |
| crownfire | start+1e-1_same_seed | 50.22 | 44.45 | 41.90 | 39.86 | 36.86 | 34.73 | 30.84 | 24.80 |
| crownfire | same_start_other_seed | 26.52 | 22.84 | 19.61 | 18.21 | 17.62 | 15.95 | 15.40 | 15.46 |
| crownfire | other_start_same_seed | 14.47 | 14.87 | 15.36 | 16.19 | 17.10 | 17.85 | 18.83 | 19.40 |
| windowfire | start+1e-3_same_seed | 75.15 | 72.06 | 70.96 | 69.11 | 68.06 | 65.93 | 62.19 | 53.25 |
| windowfire | start+1e-1_same_seed | 52.66 | 47.43 | 45.01 | 42.97 | 39.66 | 34.97 | 34.41 | 34.54 |
| windowfire | same_start_other_seed | 29.85 | 21.35 | 19.66 | 18.62 | 16.53 | 16.35 | 16.81 | 17.00 |
| windowfire | other_start_same_seed | 14.51 | 15.27 | 15.48 | 15.74 | 16.27 | 17.67 | 20.17 | 22.05 |
| dustexplosion | start+1e-3_same_seed | 78.34 | 63.66 | 40.83 | 32.37 | 29.52 | 29.12 | 28.23 | - |
| dustexplosion | start+1e-1_same_seed | 57.62 | 40.28 | 32.93 | 28.77 | 27.23 | 24.54 | 23.65 | - |
| dustexplosion | same_start_other_seed | 31.96 | 25.05 | 20.82 | 18.65 | 15.49 | 11.09 | 12.20 | - |
| dustexplosion | other_start_same_seed | 18.39 | 17.81 | 16.72 | 14.78 | 13.43 | 12.59 | 13.95 | - |

### Training

Stepper loss: stages 1-2 (normalised squared error plus profiles) and stage 3 (plus activity). Renderer PSNR on its training samples (true fields). Detail constants as calibrated.

| effect | runs x frames | minutes of simulation | stepper loss 1-2 | stepper s | stage 3 loss | stage 3 s | renderer dB | contrast | swirl | grow | start points | KB stored |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| crownfire | 160 x 240 | 21.33 | 0.04632 | 11099 | 0.04670 | 2164 | 29.42 | 0.5 | 0 | 2 | 8 | 145.7 |
| windowfire | 160 x 240 | 21.33 | 0.03311 | 11005 | 0.03275 | 2166 | 33.39 | 0.5 | 1.5 | 1 | 8 | 145.7 |
| dustexplosion | 240 x 120 | 16.00 | 0.03074 | 10761 | 0.08177 | 2139 | 28.34 | 0.5 | 0 | 4 | 16 | 274.2 |

Renderer, detail constants and start points from `d-finish` on the trained steppers; stage 3 from `d-tune`.

### Tracking a held-out run from its true start point

Active PSNR against the true run, mean over held-out runs (other seeds and settings), by frames after the start point. `neural`: the rollout effect through the runtime, started from the true state (coarse and fine) with the run's noise seed. `coarse_sim_detail`: the simulation on the same 32-cell grid with the same detail layer, drawn by the simulation's own renderer. `neural_dynamics_true_renderer`: the neural dynamics drawn by that renderer too (separates dynamics from rendering). `renderer_on_true_fields`: the learned renderer on the true fields (frames 1-30, then their mean).

**crownfire**

| method | 1 | 4 | 8 | 16 | 30 | 60 | 120 | 240 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| neural | 27.01 | 23.29 | 21.44 | 19.79 | 18.69 | 17.55 | 16.10 | 15.90 |
| neural_dynamics_true_renderer | 31.26 | 23.66 | 21.36 | 19.60 | 18.37 | 17.26 | 15.86 | 15.60 |
| coarse_sim_detail | 27.19 | 22.14 | 19.39 | 17.38 | 16.09 | 15.80 | 15.24 | 15.37 |
| coarse_sim | 22.72 | 21.80 | 20.89 | 20.18 | 19.28 | 18.96 | 17.82 | 17.90 |
| frozen | 27.30 | 18.44 | 15.88 | 14.69 | 14.09 | 13.98 | 14.01 | 13.84 |
| renderer_on_true_fields | 28.02 | 28.20 | 28.17 | 28.42 | 28.72 | 28.41 | 28.41 | 28.41 |

neural - coarse_sim_detail, paired over runs: frame 1: -0.18 [-0.70, +0.31] (tie); frame 8: +2.06 [+1.86, +2.23]; frame 30: +2.60 [+2.16, +3.04]; frame 60: +1.75 [+1.29, +2.21]; frame 240: +0.54 [-0.01, +1.06] (tie);

neural_dynamics_true_renderer - coarse_sim_detail, paired over runs: frame 1: +4.07 [+3.12, +5.03]; frame 8: +1.97 [+1.76, +2.18]; frame 30: +2.28 [+1.84, +2.69]; frame 60: +1.45 [+0.98, +1.91]; frame 240: +0.23 [-0.29, +0.76] (tie);

**dustexplosion**

| method | 1 | 4 | 8 | 16 | 30 | 60 | 120 | 240 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| neural | 22.03 | 20.63 | 18.15 | 16.73 | 15.22 | 13.82 | - | - |
| neural_dynamics_true_renderer | 29.96 | 22.66 | 18.56 | 16.45 | 14.84 | 13.31 | - | - |
| coarse_sim_detail | 20.79 | 13.73 | 10.29 | 7.96 | 6.78 | 7.04 | - | - |
| coarse_sim | 18.50 | 13.61 | 10.28 | 8.03 | 6.89 | 7.20 | - | - |
| frozen | 27.43 | 17.79 | 12.83 | 9.41 | 7.63 | 6.79 | - | - |
| renderer_on_true_fields | 22.51 | 23.39 | 24.95 | 27.12 | 27.17 | 26.08 | 26.08 | 26.08 |

neural - coarse_sim_detail, paired over runs: frame 1: +1.25 [-0.17, +2.81] (tie); frame 8: +7.86 [+7.05, +8.68]; frame 30: +8.44 [+7.88, +9.04]; frame 60: +6.77 [+5.98, +7.67];

neural_dynamics_true_renderer - coarse_sim_detail, paired over runs: frame 1: +9.17 [+7.56, +10.62]; frame 8: +8.28 [+7.44, +9.13]; frame 30: +8.06 [+7.49, +8.66]; frame 60: +6.27 [+5.51, +7.12];

**windowfire**

| method | 1 | 4 | 8 | 16 | 30 | 60 | 120 | 240 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| neural | 28.91 | 24.54 | 22.36 | 20.72 | 19.67 | 18.60 | 18.11 | 18.36 |
| neural_dynamics_true_renderer | 31.56 | 24.70 | 22.15 | 20.39 | 19.35 | 18.29 | 17.84 | 18.11 |
| coarse_sim_detail | 32.03 | 22.93 | 19.77 | 17.70 | 16.36 | 15.74 | 16.09 | 16.35 |
| coarse_sim | 23.02 | 21.14 | 20.16 | 19.30 | 18.07 | 17.12 | 17.43 | 17.46 |
| frozen | 29.13 | 19.06 | 15.90 | 13.97 | 13.53 | 13.45 | 13.72 | 14.67 |
| renderer_on_true_fields | 31.61 | 31.58 | 31.62 | 32.00 | 31.85 | 31.88 | 31.88 | 31.88 |

neural - coarse_sim_detail, paired over runs: frame 1: -3.13 [-4.32, -1.82]; frame 8: +2.59 [+2.09, +3.10]; frame 30: +3.31 [+2.91, +3.73]; frame 60: +2.86 [+2.44, +3.31]; frame 240: +2.01 [+1.66, +2.42];

neural_dynamics_true_renderer - coarse_sim_detail, paired over runs: frame 1: -0.48 [-1.49, +0.62] (tie); frame 8: +2.39 [+1.91, +2.88]; frame 30: +2.99 [+2.56, +3.40]; frame 60: +2.55 [+2.11, +2.99]; frame 240: +1.76 [+1.39, +2.18];

### Endless runs at held-out settings

The ten held-out settings of study B, a new seed each, 10 s per run (explosions: their 3 s). Distances of each method's frame statistics to a real run of the simulation at that setting; `real_other_seed` is a second real run with another seed, the floor. Lower is better except motion ratio (1 is right) and mean-frame PSNR (higher is closer).

**crownfire**

| method | spectrum L1 | motion ratio | coverage L1 | emission L1 | mean-frame PSNR |
|---|---:|---:|---:|---:|---:|
| real_other_seed | 0.051 | 1.00 | 0.0203 | 0.0005 | 28.88 |
| neural | 0.431 | 0.93 | 0.0260 | 0.0005 | 25.89 |
| flipbook_nearest | 0.076 | 1.04 | 0.0199 | 0.0006 | 24.13 |
| coarse_sim_detail | 0.673 | 1.14 | 0.0226 | 0.0016 | 24.10 |
| coarse_sim | 0.710 | 0.49 | 0.0260 | 0.0006 | 25.69 |

Spectrum distance, paired over settings: neural - grid_k8 -; neural - flipbook_nearest +0.355 [+0.324, +0.387]; neural - coarse_sim_detail -0.242 [-0.286, -0.201]; neural - real_other_seed +0.380 [+0.342, +0.417].

Mean-frame PSNR (dB), paired: neural - flipbook_nearest +1.76 [+1.07, +2.49]; neural - coarse_sim_detail +1.78 [+1.27, +2.22]; neural - real_other_seed -3.00 [-3.68, -2.15]. Motion error |ln(motion ratio)|, paired: neural - flipbook_nearest -0.041 [-0.149, +0.056] (tie); neural - coarse_sim_detail -0.041 [-0.143, +0.058] (tie); neural - real_other_seed +0.022 [-0.052, +0.095] (tie). Coverage distance, paired: neural - flipbook_nearest +0.0061 [+0.0021, +0.0097]; neural - real_other_seed +0.0057 [+0.0031, +0.0083].

**dustexplosion**

| method | spectrum L1 | motion ratio | coverage L1 | emission L1 | mean-frame PSNR |
|---|---:|---:|---:|---:|---:|
| real_other_seed | 0.076 | 1.07 | 0.0091 | 0.0008 | 22.99 |
| neural | 0.127 | 0.89 | 0.0137 | 0.0012 | 22.70 |
| flipbook_nearest | 0.091 | 0.49 | 0.0278 | 0.0041 | 19.56 |
| coarse_sim_detail | 0.391 | 2.51 | 0.1283 | 0.0121 | 16.36 |
| coarse_sim | 0.533 | 1.96 | 0.1357 | 0.0103 | 16.29 |

Spectrum distance, paired over settings: neural - grid_k8 -; neural - flipbook_nearest +0.035 [+0.012, +0.057]; neural - coarse_sim_detail -0.264 [-0.322, -0.207]; neural - real_other_seed +0.051 [+0.026, +0.075].

Mean-frame PSNR (dB), paired: neural - flipbook_nearest +3.15 [+2.80, +3.56]; neural - coarse_sim_detail +6.35 [+5.31, +7.34]; neural - real_other_seed -0.29 [-0.89, +0.28] (tie). Motion error |ln(motion ratio)|, paired: neural - flipbook_nearest -0.589 [-0.681, -0.495]; neural - coarse_sim_detail -0.753 [-0.894, -0.600]; neural - real_other_seed +0.033 [-0.064, +0.121] (tie). Coverage distance, paired: neural - flipbook_nearest -0.0141 [-0.0259, -0.0044]; neural - real_other_seed +0.0047 [-0.0007, +0.0083] (tie).

**windowfire**

| method | spectrum L1 | motion ratio | coverage L1 | emission L1 | mean-frame PSNR |
|---|---:|---:|---:|---:|---:|
| real_other_seed | 0.030 | 1.01 | 0.0123 | 0.0015 | 30.97 |
| neural | 0.439 | 0.93 | 0.0160 | 0.0013 | 27.27 |
| flipbook_nearest | 0.074 | 1.08 | 0.0120 | 0.0016 | 25.49 |
| coarse_sim_detail | 0.614 | 1.02 | 0.0382 | 0.0017 | 23.34 |
| coarse_sim | 0.521 | 0.50 | 0.0477 | 0.0015 | 23.69 |

Spectrum distance, paired over settings: neural - grid_k8 -; neural - flipbook_nearest +0.365 [+0.335, +0.394]; neural - coarse_sim_detail -0.176 [-0.195, -0.158]; neural - real_other_seed +0.409 [+0.370, +0.445].

Mean-frame PSNR (dB), paired: neural - flipbook_nearest +1.78 [+1.16, +2.34]; neural - coarse_sim_detail +3.93 [+3.38, +4.48]; neural - real_other_seed -3.70 [-4.78, -2.77]. Motion error |ln(motion ratio)|, paired: neural - flipbook_nearest +0.001 [-0.056, +0.056] (tie); neural - coarse_sim_detail -0.030 [-0.081, +0.025] (tie); neural - real_other_seed +0.049 [+0.019, +0.080]. Coverage distance, paired: neural - flipbook_nearest +0.0040 [-0.0018, +0.0094] (tie); neural - real_other_seed +0.0037 [-0.0004, +0.0080] (tie).

### One minute of play

Statistics of each 10 s window of a 60 s neural run (shards of 6 s from start points, the default) against a real 10 s run at the same setting (two held-out settings): if the effect drifted, the distances would grow window by window.

| effect | setting | window | spectrum L1 | motion ratio | coverage L1 | mean-frame PSNR |
|---|---:|---:|---:|---:|---:|---:|
| crownfire | 0 | 0 | 0.5471 | 1.1253 | 0.0289 | 26.403 |
| crownfire | 0 | 1 | 0.5166 | 1.0634 | 0.0183 | 24.091 |
| crownfire | 0 | 2 | 0.5625 | 1.0494 | 0.0247 | 24.585 |
| crownfire | 0 | 3 | 0.5197 | 1.0945 | 0.0306 | 25.952 |
| crownfire | 0 | 4 | 0.5598 | 1.1727 | 0.0346 | 24.541 |
| crownfire | 0 | 5 | 0.5880 | 1.1051 | 0.0396 | 24.356 |
| crownfire | 1 | 0 | 0.4585 | 1.0883 | 0.0142 | 27.084 |
| crownfire | 1 | 1 | 0.4740 | 1.0239 | 0.0218 | 27.309 |
| crownfire | 1 | 2 | 0.4753 | 0.9744 | 0.0092 | 27.382 |
| crownfire | 1 | 3 | 0.4428 | 0.9960 | 0.0106 | 27.872 |
| crownfire | 1 | 4 | 0.4098 | 1.0588 | 0.0136 | 26.275 |
| crownfire | 1 | 5 | 0.4475 | 0.9872 | 0.0148 | 25.254 |
| windowfire | 0 | 0 | 0.5236 | 1.0347 | 0.0256 | 27.273 |
| windowfire | 0 | 1 | 0.5301 | 0.9479 | 0.0200 | 28.344 |
| windowfire | 0 | 2 | 0.4997 | 0.8730 | 0.0147 | 27.957 |
| windowfire | 0 | 3 | 0.4965 | 0.9918 | 0.0186 | 29.597 |
| windowfire | 0 | 4 | 0.4813 | 0.9913 | 0.0175 | 27.978 |
| windowfire | 0 | 5 | 0.5015 | 1.0034 | 0.0182 | 27.879 |
| windowfire | 1 | 0 | 0.4024 | 0.9830 | 0.0187 | 28.011 |
| windowfire | 1 | 1 | 0.4531 | 1.0296 | 0.0145 | 28.452 |
| windowfire | 1 | 2 | 0.4489 | 0.9817 | 0.0081 | 28.942 |
| windowfire | 1 | 3 | 0.4794 | 0.9964 | 0.0105 | 28.995 |
| windowfire | 1 | 4 | 0.4467 | 1.0242 | 0.0148 | 28.713 |
| windowfire | 1 | 5 | 0.4638 | 1.0811 | 0.0139 | 28.030 |
