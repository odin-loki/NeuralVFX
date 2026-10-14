
## D: start points and learned dynamics

### Chaos horizon (the simulation itself)

Active PSNR against a run of the simulation from a stored start point, by frames after it (30 per second), mean over runs. Perturbations are relative to the velocity RMS.

| effect | case | 1 | 4 | 8 | 15 | 30 | 60 | 120 | 240 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fog | start+1e-3_same_seed | 91.43 | 78.79 | 77.73 | 76.30 | 75.86 | 74.71 | 72.71 | 74.68 |
| fog | start+1e-1_same_seed | 62.82 | 58.70 | 57.18 | 55.66 | 53.91 | 51.75 | 50.91 | 55.21 |
| fog | same_start_other_seed | 23.25 | 23.43 | 23.47 | 22.51 | 20.61 | 18.84 | 19.72 | 19.88 |
| fog | other_start_same_seed | 16.41 | 16.84 | 17.42 | 18.09 | 18.71 | 19.79 | 21.92 | 29.22 |
| dust | start+1e-3_same_seed | 80.37 | 75.31 | 74.87 | 73.69 | 68.59 | 64.40 | - | - |
| dust | start+1e-1_same_seed | 54.15 | 52.29 | 51.68 | 49.69 | 43.57 | 40.85 | - | - |
| dust | same_start_other_seed | 54.90 | 40.79 | 31.39 | 21.77 | 15.85 | 14.68 | - | - |
| dust | other_start_same_seed | 19.73 | 20.54 | 21.74 | 22.22 | 21.23 | 21.45 | - | - |
| cloud | start+1e-3_same_seed | 88.62 | 81.30 | 79.21 | 76.63 | 78.08 | 79.58 | 88.07 | 90.37 |
| cloud | start+1e-1_same_seed | 65.73 | 61.09 | 59.00 | 57.91 | 57.53 | 59.01 | 60.77 | 67.10 |
| cloud | same_start_other_seed | 43.83 | 32.60 | 27.48 | 23.61 | 20.55 | 17.05 | 16.72 | 17.20 |
| cloud | other_start_same_seed | 17.23 | 17.77 | 18.43 | 19.46 | 21.20 | 24.67 | 30.62 | 41.64 |

### Training

Stepper loss: stages 1-2 (normalised squared error plus profiles) and stage 3 (plus activity). Renderer PSNR on its training samples (true fields). Detail constants as calibrated.

| effect | runs x frames | minutes of simulation | stepper loss 1-2 | stepper s | stage 3 loss | stage 3 s | renderer dB | contrast | swirl | grow | start points | KB stored |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fog | 160 x 240 | 21.33 | 0.01692 | 7028 | 0.01564 | 1395 | 43.08 | 1 | 0 | 1 | 8 | 145.7 |
| dust | 240 x 90 | 12.00 | 0.02564 | 7107 | 0.03818 | 1256 | 38.06 | 1 | 0 | 2 | 16 | 274.2 |
| cloud | 160 x 240 | 21.33 | 0.01065 | 8147 | 0.01000 | 2172 | 38.34 | 1 | 0 | 1 | 8 | 145.7 |

Renderer, detail constants and start points from `d-finish` on the trained steppers; stage 3 from `d-tune`.

### Tracking a held-out run from its true start point

Active PSNR against the true run, mean over held-out runs (other seeds and settings), by frames after the start point. `neural`: the rollout effect through the runtime, started from the true state (coarse and fine) with the run's noise seed. `coarse_sim_detail`: the simulation on the same 32-cell grid with the same detail layer, drawn by the simulation's own renderer. `neural_dynamics_true_renderer`: the neural dynamics drawn by that renderer too (separates dynamics from rendering). `renderer_on_true_fields`: the learned renderer on the true fields (frames 1-30, then their mean).

**cloud**

| method | 1 | 4 | 8 | 16 | 30 | 60 | 120 | 240 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| neural | 33.53 | 32.32 | 30.41 | 27.88 | 25.43 | 23.45 | 21.91 | 21.13 |
| neural_dynamics_true_renderer | 43.99 | 36.70 | 32.23 | 28.40 | 25.36 | 23.14 | 21.59 | 20.97 |
| coarse_sim_detail | 45.62 | 36.66 | 31.02 | 26.14 | 22.83 | 20.38 | 19.52 | 19.25 |
| coarse_sim | 28.39 | 28.01 | 27.26 | 25.85 | 24.32 | 22.69 | 22.16 | 21.92 |
| frozen | 46.15 | 34.98 | 29.60 | 24.73 | 20.95 | 17.18 | 16.55 | 16.78 |
| renderer_on_true_fields | 33.88 | 33.76 | 33.60 | 33.36 | 33.41 | 33.49 | 33.49 | 33.49 |

neural - coarse_sim_detail, paired over runs: frame 1: -12.09 [-13.30, -10.71]; frame 8: -0.62 [-1.15, -0.11]; frame 30: +2.59 [+1.81, +3.28]; frame 60: +3.08 [+2.31, +3.87]; frame 240: +1.88 [+1.39, +2.40];

neural_dynamics_true_renderer - coarse_sim_detail, paired over runs: frame 1: -1.63 [-2.71, -0.49]; frame 8: +1.21 [+0.88, +1.53]; frame 30: +2.52 [+1.88, +3.07]; frame 60: +2.76 [+2.07, +3.44]; frame 240: +1.72 [+1.24, +2.21];

**dust**

| method | 1 | 4 | 8 | 16 | 30 | 60 | 120 | 240 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| neural | 23.39 | 19.08 | 20.15 | 19.32 | 18.35 | 16.55 | - | - |
| neural_dynamics_true_renderer | 23.67 | 19.29 | 20.28 | 19.43 | 18.40 | 16.53 | - | - |
| coarse_sim_detail | 23.17 | 19.60 | 18.61 | 17.18 | 14.92 | 13.42 | - | - |
| coarse_sim | 19.15 | 19.13 | 19.04 | 17.92 | 16.11 | 15.76 | - | - |
| frozen | 11.07 | 8.08 | 7.96 | 8.06 | 8.43 | 9.87 | - | - |
| renderer_on_true_fields | 29.66 | 30.59 | 31.19 | 31.77 | 32.78 | 31.63 | 31.63 | 31.63 |

neural - coarse_sim_detail, paired over runs: frame 1: +0.22 [-0.89, +1.24] (tie); frame 8: +1.54 [+0.45, +2.69]; frame 30: +3.42 [+2.69, +4.07]; frame 60: +3.13 [+2.28, +3.97];

neural_dynamics_true_renderer - coarse_sim_detail, paired over runs: frame 1: +0.50 [-0.60, +1.51] (tie); frame 8: +1.66 [+0.48, +2.93]; frame 30: +3.48 [+2.71, +4.15]; frame 60: +3.11 [+2.27, +3.93];

**fog**

| method | 1 | 4 | 8 | 16 | 30 | 60 | 120 | 240 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| neural | 35.44 | 30.28 | 27.41 | 24.17 | 21.95 | 20.64 | 19.56 | 19.32 |
| neural_dynamics_true_renderer | 39.69 | 31.30 | 27.82 | 24.18 | 21.92 | 20.62 | 19.50 | 19.17 |
| coarse_sim_detail | 35.41 | 27.96 | 24.45 | 21.67 | 19.84 | 18.62 | 18.14 | 18.50 |
| coarse_sim | 24.51 | 24.05 | 23.61 | 22.99 | 22.26 | 21.62 | 21.41 | 21.76 |
| frozen | 39.10 | 27.67 | 22.71 | 18.55 | 16.31 | 14.90 | 14.56 | 15.08 |
| renderer_on_true_fields | 37.66 | 37.73 | 37.83 | 38.00 | 37.88 | 37.92 | 37.92 | 37.92 |

neural - coarse_sim_detail, paired over runs: frame 1: +0.03 [-2.21, +2.78] (tie); frame 8: +2.97 [+2.19, +3.83]; frame 30: +2.11 [+1.70, +2.53]; frame 60: +2.02 [+1.60, +2.46]; frame 240: +0.82 [+0.12, +1.62];

neural_dynamics_true_renderer - coarse_sim_detail, paired over runs: frame 1: +4.28 [+2.02, +7.05]; frame 8: +3.37 [+2.59, +4.21]; frame 30: +2.09 [+1.68, +2.54]; frame 60: +2.00 [+1.59, +2.41]; frame 240: +0.67 [-0.04, +1.46] (tie);

### Endless runs at held-out settings

The ten held-out settings of study B, a new seed each, 10 s per run (explosions: their 3 s). Distances of each method's frame statistics to a real run of the simulation at that setting; `real_other_seed` is a second real run with another seed, the floor. Lower is better except motion ratio (1 is right) and mean-frame PSNR (higher is closer).

**cloud**

| method | spectrum L1 | motion ratio | coverage L1 | emission L1 | mean-frame PSNR |
|---|---:|---:|---:|---:|---:|
| real_other_seed | 0.066 | 0.99 | 0.0039 | 0.0000 | 31.71 |
| neural | 0.345 | 1.19 | 0.0069 | 0.0002 | 29.56 |
| flipbook_nearest | 0.132 | 1.44 | 0.0082 | 0.0000 | 28.08 |
| coarse_sim_detail | 0.677 | 1.22 | 0.0052 | 0.0000 | 29.16 |
| coarse_sim | 0.688 | 0.72 | 0.0046 | 0.0000 | 30.13 |

Spectrum distance, paired over settings: neural - grid_k8 -; neural - flipbook_nearest +0.212 [+0.119, +0.310]; neural - coarse_sim_detail -0.333 [-0.388, -0.278]; neural - real_other_seed +0.278 [+0.201, +0.355].

Mean-frame PSNR (dB), paired: neural - flipbook_nearest +1.49 [+0.33, +2.41]; neural - coarse_sim_detail +0.40 [-0.58, +1.24] (tie); neural - real_other_seed -2.14 [-3.35, -1.12]. Motion error |ln(motion ratio)|, paired: neural - flipbook_nearest -0.179 [-0.323, -0.028]; neural - coarse_sim_detail -0.020 [-0.069, +0.039] (tie); neural - real_other_seed +0.075 [-0.010, +0.173] (tie). Coverage distance, paired: neural - flipbook_nearest -0.0013 [-0.0037, +0.0011] (tie); neural - real_other_seed +0.0030 [+0.0018, +0.0043].

**dust**

| method | spectrum L1 | motion ratio | coverage L1 | emission L1 | mean-frame PSNR |
|---|---:|---:|---:|---:|---:|
| real_other_seed | 0.108 | 0.99 | 0.0102 | 0.0000 | 28.36 |
| neural | 0.128 | 0.98 | 0.0199 | 0.0000 | 26.37 |
| flipbook_nearest | 0.146 | 0.60 | 0.0141 | 0.0000 | 28.29 |
| coarse_sim_detail | 0.196 | 0.97 | 0.0117 | 0.0000 | 26.00 |
| coarse_sim | 1.270 | 0.61 | 0.0103 | 0.0000 | 26.54 |

Spectrum distance, paired over settings: neural - grid_k8 -; neural - flipbook_nearest -0.018 [-0.076, +0.039] (tie); neural - coarse_sim_detail -0.068 [-0.108, -0.027]; neural - real_other_seed +0.020 [-0.040, +0.067] (tie).

Mean-frame PSNR (dB), paired: neural - flipbook_nearest -1.92 [-2.98, -0.77]; neural - coarse_sim_detail +0.37 [-0.87, +1.79] (tie); neural - real_other_seed -1.99 [-3.54, -0.54]. Motion error |ln(motion ratio)|, paired: neural - flipbook_nearest -0.241 [-0.553, +0.039] (tie); neural - coarse_sim_detail +0.071 [-0.116, +0.272] (tie); neural - real_other_seed +0.159 [+0.004, +0.312]. Coverage distance, paired: neural - flipbook_nearest +0.0057 [+0.0000, +0.0116]; neural - real_other_seed +0.0096 [+0.0008, +0.0209].

**fog**

| method | spectrum L1 | motion ratio | coverage L1 | emission L1 | mean-frame PSNR |
|---|---:|---:|---:|---:|---:|
| real_other_seed | 0.096 | 1.14 | 0.0035 | 0.0000 | 34.22 |
| neural | 0.281 | 1.27 | 0.0053 | 0.0000 | 29.29 |
| flipbook_nearest | 0.153 | 1.42 | 0.0054 | 0.0000 | 30.30 |
| coarse_sim_detail | 0.233 | 1.12 | 0.0056 | 0.0000 | 30.85 |
| coarse_sim | 1.309 | 0.53 | 0.0042 | 0.0000 | 32.95 |

Spectrum distance, paired over settings: neural - grid_k8 -; neural - flipbook_nearest +0.128 [-0.022, +0.245] (tie); neural - coarse_sim_detail +0.047 [-0.079, +0.167] (tie); neural - real_other_seed +0.185 [+0.106, +0.269].

Mean-frame PSNR (dB), paired: neural - flipbook_nearest -1.01 [-2.51, +0.21] (tie); neural - coarse_sim_detail -1.55 [-3.05, -0.28]; neural - real_other_seed -4.93 [-5.91, -3.95]. Motion error |ln(motion ratio)|, paired: neural - flipbook_nearest -0.133 [-0.337, +0.100] (tie); neural - coarse_sim_detail +0.072 [-0.065, +0.246] (tie); neural - real_other_seed +0.081 [-0.066, +0.251] (tie). Coverage distance, paired: neural - flipbook_nearest -0.0001 [-0.0024, +0.0021] (tie); neural - real_other_seed +0.0018 [-0.0000, +0.0037] (tie).

### One minute of play

Statistics of each 10 s window of a 60 s neural run (shards of 6 s from start points, the default) against a real 10 s run at the same setting (two held-out settings): if the effect drifted, the distances would grow window by window.

| effect | setting | window | spectrum L1 | motion ratio | coverage L1 | mean-frame PSNR |
|---|---:|---:|---:|---:|---:|---:|
| fog | 0 | 0 | 0.4378 | 1.1762 | 0.0089 | 28.310 |
| fog | 0 | 1 | 0.3130 | 0.9643 | 0.0043 | 32.695 |
| fog | 0 | 2 | 0.2018 | 1.0109 | 0.0032 | 32.152 |
| fog | 0 | 3 | 0.4127 | 1.1399 | 0.0099 | 28.019 |
| fog | 0 | 4 | 0.2304 | 0.9400 | 0.0076 | 32.085 |
| fog | 0 | 5 | 0.1897 | 0.9640 | 0.0077 | 32.727 |
| fog | 1 | 0 | 0.4306 | 1.6830 | 0.0010 | 32.752 |
| fog | 1 | 1 | 0.4783 | 2.2336 | 0.0042 | 31.478 |
| fog | 1 | 2 | 0.4863 | 1.9428 | 0.0028 | 34.010 |
| fog | 1 | 3 | 0.7028 | 1.6361 | 0.0065 | 30.552 |
| fog | 1 | 4 | 0.5295 | 2.1964 | 0.0057 | 29.463 |
| fog | 1 | 5 | 0.5075 | 1.7777 | 0.0026 | 33.311 |
| cloud | 0 | 0 | 0.2777 | 1.2297 | 0.0073 | 32.254 |
| cloud | 0 | 1 | 0.3676 | 1.1085 | 0.0039 | 32.532 |
| cloud | 0 | 2 | 0.3029 | 1.1355 | 0.0038 | 31.348 |
| cloud | 0 | 3 | 0.3644 | 1.3118 | 0.0103 | 30.010 |
| cloud | 0 | 4 | 0.3395 | 1.0417 | 0.0058 | 27.838 |
| cloud | 0 | 5 | 0.3699 | 0.9201 | 0.0073 | 29.611 |
| cloud | 1 | 0 | 0.4775 | 1.5395 | 0.0019 | 30.219 |
| cloud | 1 | 1 | 0.4716 | 1.6023 | 0.0045 | 29.874 |
| cloud | 1 | 2 | 0.5283 | 1.5150 | 0.0024 | 32.467 |
| cloud | 1 | 3 | 0.5008 | 1.5262 | 0.0061 | 29.423 |
| cloud | 1 | 4 | 0.5077 | 1.9564 | 0.0070 | 29.244 |
| cloud | 1 | 5 | 0.5646 | 1.6915 | 0.0029 | 32.320 |
