
## D: start points and learned dynamics

### Chaos horizon (the simulation itself)

Active PSNR against a run of the simulation from a stored start point, by frames after it (30 per second), mean over runs. Perturbations are relative to the velocity RMS.

| effect | case | 1 | 4 | 8 | 15 | 30 | 60 | 120 | 240 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| geyser | start+1e-3_same_seed | 85.60 | 82.56 | 82.03 | 81.54 | 80.44 | 77.57 | 75.19 | 71.80 |
| geyser | start+1e-1_same_seed | 63.05 | 59.34 | 57.34 | 55.58 | 53.25 | 49.10 | 50.69 | 50.60 |
| geyser | same_start_other_seed | 42.69 | 37.32 | 27.30 | 20.33 | 16.33 | 14.62 | 17.39 | 9.99 |
| geyser | other_start_same_seed | 10.56 | 11.19 | 12.03 | 12.85 | 14.70 | 17.64 | 24.73 | 33.11 |
| haboob | start+1e-3_same_seed | 99.00 | 94.61 | 91.49 | 86.44 | 82.41 | 79.37 | 75.20 | - |
| haboob | start+1e-1_same_seed | 75.20 | 69.27 | 66.54 | 65.07 | 61.70 | 58.47 | 58.09 | - |
| haboob | same_start_other_seed | 54.40 | 43.46 | 36.16 | 28.94 | 20.99 | 15.12 | 15.08 | - |
| haboob | other_start_same_seed | 24.93 | 25.55 | 26.36 | 27.57 | 29.16 | 28.88 | 26.19 | - |
| seaspray | start+1e-3_same_seed | 71.27 | 69.06 | 67.67 | 67.05 | 66.76 | 66.33 | 70.19 | 80.56 |
| seaspray | start+1e-1_same_seed | 49.37 | 44.75 | 43.12 | 42.13 | 42.65 | 46.95 | 53.75 | 57.00 |
| seaspray | same_start_other_seed | 19.76 | 15.95 | 14.23 | 13.75 | 10.37 | 11.44 | 12.86 | 11.99 |
| seaspray | other_start_same_seed | 11.74 | 12.71 | 14.20 | 16.74 | 20.87 | 27.53 | 41.50 | 52.56 |

### Training

Stepper loss: stages 1-2 (normalised squared error plus profiles) and stage 3 (plus activity). Renderer PSNR on its training samples (true fields). Detail constants as calibrated.

| effect | runs x frames | minutes of simulation | stepper loss 1-2 | stepper s | stage 3 loss | stage 3 s | renderer dB | contrast | swirl | grow | start points | KB stored |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| geyser | 160 x 240 | 21.33 | 0.11420 | 11043 | 0.11076 | 2135 | 34.29 | 0.5 | 0 | 1 | 8 | 145.7 |
| haboob | 240 x 150 | 20.00 | 0.02115 | 10862 | 0.02193 | 2135 | 31.59 | 0.5 | 0 | 1.5 | 16 | 274.2 |
| seaspray | 160 x 240 | 21.33 | 0.10753 | 11016 | 0.12636 | 2237 | 42.07 | 0.5 | 0 | 1 | 8 | 145.7 |

Renderer, detail constants and start points from `d-finish` on the trained steppers; stage 3 from `d-tune`.

### Tracking a held-out run from its true start point

Active PSNR against the true run, mean over held-out runs (other seeds and settings), by frames after the start point. `neural`: the rollout effect through the runtime, started from the true state (coarse and fine) with the run's noise seed. `coarse_sim_detail`: the simulation on the same 32-cell grid with the same detail layer, drawn by the simulation's own renderer. `neural_dynamics_true_renderer`: the neural dynamics drawn by that renderer too (separates dynamics from rendering). `renderer_on_true_fields`: the learned renderer on the true fields (frames 1-30, then their mean).

**geyser**

| method | 1 | 4 | 8 | 16 | 30 | 60 | 120 | 240 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| neural | 31.70 | 26.76 | 23.67 | 20.48 | 18.84 | 16.74 | 21.27 | 16.13 |
| neural_dynamics_true_renderer | 34.75 | 27.72 | 23.89 | 20.57 | 18.96 | 16.89 | 21.26 | 16.24 |
| coarse_sim_detail | 31.37 | 24.69 | 21.68 | 18.92 | 17.74 | 18.91 | 20.88 | 18.79 |
| coarse_sim | 24.39 | 22.60 | 21.39 | 19.51 | 18.43 | 19.49 | 21.74 | 19.60 |
| frozen | 41.55 | 30.27 | 25.00 | 20.82 | 18.28 | 15.98 | 13.94 | 10.95 |
| renderer_on_true_fields | 34.72 | 34.96 | 35.33 | 36.60 | 37.09 | 36.14 | 36.14 | 36.14 |

neural - coarse_sim_detail, paired over runs: frame 1: +0.34 [-2.74, +3.13] (tie); frame 8: +1.99 [+0.40, +3.80]; frame 30: +1.10 [-0.33, +2.57] (tie); frame 60: -2.17 [-3.80, -0.46]; frame 240: -2.65 [-5.24, +0.03] (tie);

neural_dynamics_true_renderer - coarse_sim_detail, paired over runs: frame 1: +3.38 [+1.56, +5.42]; frame 8: +2.21 [+0.47, +4.19]; frame 30: +1.22 [-0.20, +2.66] (tie); frame 60: -2.01 [-3.64, -0.33]; frame 240: -2.55 [-5.24, +0.30] (tie);

**haboob**

| method | 1 | 4 | 8 | 16 | 30 | 60 | 120 | 240 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| neural | 27.45 | 26.26 | 24.37 | 20.71 | 18.48 | 16.66 | 15.14 | - |
| neural_dynamics_true_renderer | 37.95 | 29.61 | 25.46 | 20.64 | 18.37 | 16.65 | 15.04 | - |
| coarse_sim_detail | 39.18 | 32.49 | 27.12 | 21.18 | 17.42 | 14.34 | 12.86 | - |
| coarse_sim | 24.36 | 24.97 | 25.37 | 24.58 | 21.17 | 16.88 | 14.62 | - |
| frozen | 43.12 | 32.42 | 26.49 | 19.42 | 14.17 | 10.91 | 9.39 | - |
| renderer_on_true_fields | 27.87 | 28.50 | 29.08 | 29.52 | 29.04 | 29.11 | 29.11 | 29.11 |

neural - coarse_sim_detail, paired over runs: frame 1: -11.73 [-13.41, -9.62]; frame 8: -2.75 [-3.76, -1.89]; frame 30: +1.06 [+0.25, +1.87]; frame 60: +2.32 [+1.34, +3.23];

neural_dynamics_true_renderer - coarse_sim_detail, paired over runs: frame 1: -1.23 [-2.89, +0.84] (tie); frame 8: -1.66 [-2.58, -0.88]; frame 30: +0.94 [+0.00, +1.90]; frame 60: +2.30 [+1.33, +3.24];

**seaspray**

| method | 1 | 4 | 8 | 16 | 30 | 60 | 120 | 240 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| neural | 30.13 | 23.16 | 20.56 | 18.50 | 17.15 | 17.12 | 17.81 | 17.46 |
| neural_dynamics_true_renderer | 30.52 | 23.10 | 20.49 | 18.47 | 17.10 | 17.11 | 17.82 | 17.50 |
| coarse_sim_detail | 23.87 | 21.77 | 19.33 | 16.60 | 15.22 | 14.82 | 14.99 | 14.58 |
| coarse_sim | 20.40 | 19.63 | 18.80 | 17.30 | 16.05 | 15.66 | 15.84 | 15.20 |
| frozen | 28.57 | 18.46 | 14.83 | 12.76 | 10.46 | 10.15 | 11.23 | 10.40 |
| renderer_on_true_fields | 40.49 | 40.44 | 40.60 | 40.93 | 39.94 | 40.73 | 40.73 | 40.73 |

neural - coarse_sim_detail, paired over runs: frame 1: +6.26 [+4.13, +8.63]; frame 8: +1.23 [-0.21, +2.55] (tie); frame 30: +1.92 [+1.01, +2.83]; frame 60: +2.30 [+0.42, +4.17]; frame 240: +2.88 [+1.85, +3.78];

neural_dynamics_true_renderer - coarse_sim_detail, paired over runs: frame 1: +6.65 [+4.56, +8.95]; frame 8: +1.16 [-0.27, +2.49] (tie); frame 30: +1.87 [+0.94, +2.83]; frame 60: +2.29 [+0.39, +4.18]; frame 240: +2.93 [+1.92, +3.80];

### Endless runs at held-out settings

The ten held-out settings of study B, a new seed each, 10 s per run (explosions: their 3 s). Distances of each method's frame statistics to a real run of the simulation at that setting; `real_other_seed` is a second real run with another seed, the floor. Lower is better except motion ratio (1 is right) and mean-frame PSNR (higher is closer).

**geyser**

| method | spectrum L1 | motion ratio | coverage L1 | emission L1 | mean-frame PSNR |
|---|---:|---:|---:|---:|---:|
| real_other_seed | 0.921 | 6.50 | 0.0395 | 0.0000 | 26.18 |
| neural | 0.587 | 4.54 | 0.0382 | 0.0002 | 25.31 |
| flipbook_nearest | 0.608 | 16.23 | 0.0444 | 0.0000 | 20.03 |
| coarse_sim_detail | 1.032 | 2.24 | 0.0389 | 0.0000 | 23.28 |
| coarse_sim | 1.330 | 2.19 | 0.0398 | 0.0000 | 23.13 |

Spectrum distance, paired over settings: neural - grid_k8 -; neural - flipbook_nearest -0.021 [-0.271, +0.235] (tie); neural - coarse_sim_detail -0.445 [-1.055, +0.060] (tie); neural - real_other_seed -0.334 [-0.722, +0.047] (tie).

Mean-frame PSNR (dB), paired: neural - flipbook_nearest +5.27 [+2.12, +8.42]; neural - coarse_sim_detail +2.03 [-0.79, +4.66] (tie); neural - real_other_seed -0.87 [-3.90, +2.12] (tie). Motion error |ln(motion ratio)|, paired: neural - flipbook_nearest -0.310 [-0.747, +0.130] (tie); neural - coarse_sim_detail -0.938 [-2.483, +0.206] (tie); neural - real_other_seed -0.384 [-1.134, +0.330] (tie). Coverage distance, paired: neural - flipbook_nearest -0.0063 [-0.0160, +0.0031] (tie); neural - real_other_seed -0.0013 [-0.0126, +0.0079] (tie).

**haboob**

| method | spectrum L1 | motion ratio | coverage L1 | emission L1 | mean-frame PSNR |
|---|---:|---:|---:|---:|---:|
| real_other_seed | 0.059 | 1.11 | 0.0155 | 0.0000 | 25.90 |
| neural | 0.124 | 0.97 | 0.0139 | 0.0001 | 25.82 |
| flipbook_nearest | 0.096 | 0.86 | 0.0156 | 0.0000 | 26.75 |
| coarse_sim_detail | 0.162 | 0.79 | 0.0251 | 0.0000 | 24.59 |
| coarse_sim | 0.600 | 0.41 | 0.0162 | 0.0000 | 26.18 |

Spectrum distance, paired over settings: neural - grid_k8 -; neural - flipbook_nearest +0.028 [-0.005, +0.056] (tie); neural - coarse_sim_detail -0.038 [-0.078, +0.009] (tie); neural - real_other_seed +0.065 [+0.047, +0.083].

Mean-frame PSNR (dB), paired: neural - flipbook_nearest -0.93 [-2.13, +0.19] (tie); neural - coarse_sim_detail +1.23 [+0.47, +2.00]; neural - real_other_seed -0.08 [-1.34, +0.87] (tie). Motion error |ln(motion ratio)|, paired: neural - flipbook_nearest -0.039 [-0.159, +0.068] (tie); neural - coarse_sim_detail -0.087 [-0.182, +0.021] (tie); neural - real_other_seed -0.046 [-0.140, +0.048] (tie). Coverage distance, paired: neural - flipbook_nearest -0.0018 [-0.0071, +0.0030] (tie); neural - real_other_seed -0.0016 [-0.0083, +0.0033] (tie).

**seaspray**

| method | spectrum L1 | motion ratio | coverage L1 | emission L1 | mean-frame PSNR |
|---|---:|---:|---:|---:|---:|
| real_other_seed | 0.077 | 1.13 | 0.0121 | 0.0000 | 29.88 |
| neural | 0.357 | 0.85 | 0.0124 | 0.0001 | 30.48 |
| flipbook_nearest | 0.099 | 1.16 | 0.0136 | 0.0000 | 24.61 |
| coarse_sim_detail | 0.135 | 0.57 | 0.0103 | 0.0000 | 28.27 |
| coarse_sim | 1.048 | 0.51 | 0.0107 | 0.0000 | 28.45 |

Spectrum distance, paired over settings: neural - grid_k8 -; neural - flipbook_nearest +0.257 [+0.204, +0.309]; neural - coarse_sim_detail +0.222 [+0.159, +0.265]; neural - real_other_seed +0.280 [+0.238, +0.325].

Mean-frame PSNR (dB), paired: neural - flipbook_nearest +5.88 [+4.77, +7.11]; neural - coarse_sim_detail +2.22 [+1.25, +3.09]; neural - real_other_seed +0.60 [-1.30, +1.96] (tie). Motion error |ln(motion ratio)|, paired: neural - flipbook_nearest +0.040 [-0.074, +0.163] (tie); neural - coarse_sim_detail -0.298 [-0.519, -0.105]; neural - real_other_seed +0.050 [-0.068, +0.168] (tie). Coverage distance, paired: neural - flipbook_nearest -0.0011 [-0.0049, +0.0021] (tie); neural - real_other_seed +0.0003 [-0.0023, +0.0031] (tie).

### One minute of play

Statistics of each 10 s window of a 60 s neural run (shards of 6 s from start points, the default) against a real 10 s run at the same setting (two held-out settings): if the effect drifted, the distances would grow window by window.

| effect | setting | window | spectrum L1 | motion ratio | coverage L1 | mean-frame PSNR |
|---|---:|---:|---:|---:|---:|---:|
| geyser | 0 | 0 | 0.5346 | 0.5955 | 0.0347 | 26.168 |
| geyser | 0 | 1 | 0.3956 | 0.6941 | 0.0336 | 26.330 |
| geyser | 0 | 2 | 0.4320 | 0.8502 | 0.0365 | 26.860 |
| geyser | 0 | 3 | 0.5499 | 0.4066 | 0.0261 | 24.732 |
| geyser | 0 | 4 | 0.7194 | 0.6480 | 0.0457 | 26.334 |
| geyser | 0 | 5 | 0.3797 | 0.7287 | 0.0456 | 23.944 |
| geyser | 1 | 0 | 1.2817 | 0.1181 | 0.0138 | 27.291 |
| geyser | 1 | 1 | 1.3498 | 0.1258 | 0.0143 | 27.045 |
| geyser | 1 | 2 | 0.1561 | 1.3109 | 0.0110 | 32.134 |
| geyser | 1 | 3 | 0.8087 | 0.4641 | 0.0176 | 30.645 |
| geyser | 1 | 4 | 0.1419 | 1.1047 | 0.0153 | 31.949 |
| geyser | 1 | 5 | 0.7919 | 0.1912 | 0.0145 | 27.955 |
| seaspray | 0 | 0 | 0.2986 | 0.6504 | 0.0090 | 26.568 |
| seaspray | 0 | 1 | 0.3118 | 0.7913 | 0.0103 | 28.454 |
| seaspray | 0 | 2 | 0.2069 | 0.7103 | 0.0119 | 27.464 |
| seaspray | 0 | 3 | 0.1855 | 0.7202 | 0.0186 | 27.833 |
| seaspray | 0 | 4 | 0.2922 | 0.8163 | 0.0108 | 29.313 |
| seaspray | 0 | 5 | 0.3232 | 0.7818 | 0.0104 | 30.959 |
| seaspray | 1 | 0 | 0.2138 | 1.5149 | 0.0109 | 30.169 |
| seaspray | 1 | 1 | 0.2600 | 1.2300 | 0.0070 | 30.303 |
| seaspray | 1 | 2 | 0.1448 | 2.0470 | 0.0167 | 26.499 |
| seaspray | 1 | 3 | 0.1442 | 1.5880 | 0.0119 | 27.069 |
| seaspray | 1 | 4 | 0.1477 | 1.5443 | 0.0136 | 27.835 |
| seaspray | 1 | 5 | 0.2459 | 1.3592 | 0.0093 | 30.360 |
