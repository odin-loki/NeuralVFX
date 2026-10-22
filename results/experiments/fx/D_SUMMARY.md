
## D: start points and learned dynamics

### Chaos horizon (the simulation itself)

Active PSNR against a run of the simulation from a stored start point, by frames after it (30 per second), mean over runs. Perturbations are relative to the velocity RMS.

| effect | case | 1 | 4 | 8 | 15 | 30 | 60 | 120 | 240 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| steam | start+1e-3_same_seed | 73.54 | 69.49 | 68.21 | 66.85 | 65.56 | 63.34 | 61.38 | 58.78 |
| steam | start+1e-1_same_seed | 52.55 | 47.50 | 44.93 | 42.57 | 39.75 | 35.03 | 34.37 | 38.71 |
| steam | same_start_other_seed | 28.74 | 22.68 | 19.49 | 16.66 | 14.10 | 13.09 | 13.93 | 12.27 |
| steam | other_start_same_seed | 11.14 | 11.52 | 12.00 | 12.76 | 14.61 | 17.05 | 19.43 | 26.76 |
| magic | start+1e-3_same_seed | 67.12 | 64.29 | 62.86 | 61.59 | 63.02 | 58.14 | 54.37 | 54.10 |
| magic | start+1e-1_same_seed | 41.24 | 39.95 | 39.49 | 38.97 | 43.91 | 48.35 | 48.95 | 50.83 |
| magic | same_start_other_seed | 20.61 | 17.42 | 16.79 | 16.24 | 15.78 | 14.47 | 15.64 | 14.63 |
| magic | other_start_same_seed | 16.43 | 18.54 | 20.55 | 23.29 | 38.01 | 43.20 | 47.14 | 47.16 |

### Training

Stepper loss: stages 1-2 (normalised squared error plus profiles) and stage 3 (plus activity). Renderer PSNR on its training samples (true fields). Detail constants as calibrated.

| effect | runs x frames | minutes of simulation | stepper loss 1-2 | stepper s | stage 3 loss | stage 3 s | renderer dB | contrast | swirl | grow | start points | KB stored |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| steam | 160 x 240 | 21.33 | 0.08041 | 9317 | 0.09740 | 1855 | 37.68 | 1 | 0.5 | 4 | 8 | 81.6 |
| magic | 160 x 240 | 21.33 | 0.03079 | 11257 | 0.02902 | 1876 | 37.57 | 1 | 0 | 1 | 8 | 81.6 |

Renderer, detail constants and start points from `d-finish` on the trained steppers; stage 3 from `d-tune`.

### Tracking a held-out run from its true start point

Active PSNR against the true run, mean over held-out runs (other seeds and settings), by frames after the start point. `neural`: the rollout effect through the runtime, started from the true state (coarse and fine) with the run's noise seed. `coarse_sim_detail`: the simulation on the same 32-cell grid with the same detail layer, drawn by the simulation's own renderer. `neural_dynamics_true_renderer`: the neural dynamics drawn by that renderer too (separates dynamics from rendering). `renderer_on_true_fields`: the learned renderer on the true fields (frames 1-30, then their mean).

**magic**

| method | 1 | 4 | 8 | 16 | 30 | 60 | 120 | 240 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| neural | 21.44 | 18.65 | 18.18 | 17.15 | 17.64 | 17.28 | 17.34 | 16.36 |
| neural_dynamics_true_renderer | 21.52 | 18.66 | 18.21 | 17.18 | 17.66 | 17.27 | 17.36 | 16.34 |
| coarse_sim_detail | 22.20 | 19.96 | 18.32 | 17.35 | 17.55 | 17.33 | 17.18 | 17.11 |
| coarse_sim | 19.92 | 19.00 | 18.42 | 17.65 | 17.87 | 17.94 | 17.70 | 17.77 |
| frozen | 32.29 | 22.64 | 18.65 | 14.87 | 14.73 | 14.72 | 14.73 | 15.43 |
| renderer_on_true_fields | 33.13 | 33.43 | 34.21 | 32.95 | 33.58 | 33.50 | 33.50 | 33.50 |

neural - coarse_sim_detail, paired over runs: frame 1: -0.76 [-1.76, +0.18] (tie); frame 8: -0.14 [-0.90, +0.62] (tie); frame 30: +0.09 [-1.04, +1.19] (tie); frame 60: -0.06 [-0.65, +0.54] (tie); frame 240: -0.75 [-3.04, +0.92] (tie);

neural_dynamics_true_renderer - coarse_sim_detail, paired over runs: frame 1: -0.68 [-1.69, +0.33] (tie); frame 8: -0.12 [-0.88, +0.64] (tie); frame 30: +0.11 [-1.02, +1.20] (tie); frame 60: -0.06 [-0.67, +0.54] (tie); frame 240: -0.77 [-3.07, +0.88] (tie);

**steam**

| method | 1 | 4 | 8 | 16 | 30 | 60 | 120 | 240 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| neural | 27.87 | 22.75 | 19.88 | 17.63 | 16.59 | 14.99 | 13.77 | 13.56 |
| neural_dynamics_true_renderer | 28.63 | 23.00 | 19.99 | 17.69 | 16.61 | 15.02 | 13.82 | 13.57 |
| coarse_sim_detail | 27.94 | 20.95 | 17.18 | 15.34 | 13.99 | 13.40 | 12.29 | 11.61 |
| coarse_sim | 20.89 | 19.48 | 18.34 | 17.19 | 15.73 | 14.61 | 13.17 | 12.51 |
| frozen | 23.12 | 14.41 | 12.41 | 11.54 | 11.62 | 11.40 | 11.01 | 11.46 |
| renderer_on_true_fields | 36.29 | 36.33 | 36.34 | 36.64 | 37.61 | 36.73 | 36.73 | 36.73 |

neural - coarse_sim_detail, paired over runs: frame 1: -0.07 [-0.87, +0.72] (tie); frame 8: +2.70 [+2.20, +3.30]; frame 30: +2.60 [+2.12, +3.02]; frame 60: +1.59 [+0.58, +2.52]; frame 240: +1.95 [+1.07, +2.92];

neural_dynamics_true_renderer - coarse_sim_detail, paired over runs: frame 1: +0.68 [-0.15, +1.51] (tie); frame 8: +2.81 [+2.29, +3.44]; frame 30: +2.62 [+2.14, +3.03]; frame 60: +1.61 [+0.58, +2.60]; frame 240: +1.96 [+1.09, +2.92];

### Endless runs at held-out settings

The ten held-out settings of study B, a new seed each, 10 s per run (explosions: their 3 s). Distances of each method's frame statistics to a real run of the simulation at that setting; `real_other_seed` is a second real run with another seed, the floor. Lower is better except motion ratio (1 is right) and mean-frame PSNR (higher is closer).

**magic**

| method | spectrum L1 | motion ratio | coverage L1 | emission L1 | mean-frame PSNR |
|---|---:|---:|---:|---:|---:|
| real_other_seed | 0.024 | 1.02 | 0.0072 | 0.0007 | 35.03 |
| neural | 0.162 | 1.17 | 0.0090 | 0.0013 | 27.97 |
| flipbook_nearest | 0.101 | 1.00 | 0.0129 | 0.0010 | 24.21 |
| coarse_sim_detail | 0.116 | 1.33 | 0.0095 | 0.0013 | 28.76 |
| coarse_sim | 1.115 | 0.76 | 0.0099 | 0.0029 | 27.29 |

Spectrum distance, paired over settings: neural - grid_k8 -; neural - flipbook_nearest +0.062 [+0.028, +0.094]; neural - coarse_sim_detail +0.046 [+0.007, +0.085]; neural - real_other_seed +0.139 [+0.101, +0.178].

Mean-frame PSNR (dB), paired: neural - flipbook_nearest +3.77 [+1.45, +5.85]; neural - coarse_sim_detail -0.79 [-2.02, +0.37] (tie); neural - real_other_seed -7.05 [-9.36, -5.11]. Motion error |ln(motion ratio)|, paired: neural - flipbook_nearest +0.084 [+0.040, +0.127]; neural - coarse_sim_detail -0.127 [-0.156, -0.099]; neural - real_other_seed +0.130 [+0.089, +0.173]. Coverage distance, paired: neural - flipbook_nearest -0.0039 [-0.0065, -0.0015]; neural - real_other_seed +0.0018 [+0.0003, +0.0035].

**steam**

| method | spectrum L1 | motion ratio | coverage L1 | emission L1 | mean-frame PSNR |
|---|---:|---:|---:|---:|---:|
| real_other_seed | 0.094 | 1.17 | 0.0160 | 0.0000 | 30.39 |
| neural | 0.203 | 0.99 | 0.0183 | 0.0002 | 29.76 |
| flipbook_nearest | 0.120 | 1.10 | 0.0166 | 0.0000 | 25.46 |
| coarse_sim_detail | 0.330 | 0.63 | 0.0163 | 0.0000 | 28.10 |
| coarse_sim | 1.344 | 0.35 | 0.0159 | 0.0000 | 28.03 |

Spectrum distance, paired over settings: neural - grid_k8 -; neural - flipbook_nearest +0.083 [+0.041, +0.129]; neural - coarse_sim_detail -0.127 [-0.166, -0.086]; neural - real_other_seed +0.109 [+0.064, +0.151].

Mean-frame PSNR (dB), paired: neural - flipbook_nearest +4.31 [+2.64, +5.73]; neural - coarse_sim_detail +1.66 [+0.17, +3.12]; neural - real_other_seed -0.63 [-1.51, +0.17] (tie). Motion error |ln(motion ratio)|, paired: neural - flipbook_nearest -0.135 [-0.219, -0.043]; neural - coarse_sim_detail -0.371 [-0.505, -0.217]; neural - real_other_seed -0.072 [-0.124, -0.031]. Coverage distance, paired: neural - flipbook_nearest +0.0017 [-0.0027, +0.0062] (tie); neural - real_other_seed +0.0023 [-0.0010, +0.0059] (tie).

### One minute of play

Statistics of each 10 s window of a 60 s neural run (shards of 6 s from start points, the default) against a real 10 s run at the same setting (two held-out settings): if the effect drifted, the distances would grow window by window.

| effect | setting | window | spectrum L1 | motion ratio | coverage L1 | mean-frame PSNR |
|---|---:|---:|---:|---:|---:|---:|
| steam | 0 | 0 | 0.3173 | 1.1143 | 0.0173 | 27.710 |
| steam | 0 | 1 | 0.2725 | 0.8163 | 0.0125 | 32.684 |
| steam | 0 | 2 | 0.2402 | 0.9299 | 0.0096 | 32.509 |
| steam | 0 | 3 | 0.2777 | 1.1057 | 0.0182 | 28.402 |
| steam | 0 | 4 | 0.2973 | 0.7388 | 0.0122 | 34.862 |
| steam | 0 | 5 | 0.2544 | 0.9105 | 0.0105 | 32.719 |
| steam | 1 | 0 | 0.2825 | 1.4209 | 0.0203 | 30.274 |
| steam | 1 | 1 | 0.2323 | 1.1208 | 0.0196 | 34.077 |
| steam | 1 | 2 | 0.2351 | 1.0179 | 0.0079 | 32.676 |
| steam | 1 | 3 | 0.2162 | 1.1607 | 0.0168 | 30.670 |
| steam | 1 | 4 | 0.1626 | 0.9518 | 0.0116 | 31.645 |
| steam | 1 | 5 | 0.2925 | 1.3523 | 0.0212 | 30.640 |
| magic | 0 | 0 | 0.3293 | 1.3773 | 0.0127 | 23.141 |
| magic | 0 | 1 | 0.2450 | 1.3014 | 0.0080 | 28.090 |
| magic | 0 | 2 | 0.2596 | 1.2959 | 0.0076 | 28.152 |
| magic | 0 | 3 | 0.3544 | 1.3876 | 0.0170 | 22.004 |
| magic | 0 | 4 | 0.2385 | 1.3067 | 0.0079 | 25.388 |
| magic | 0 | 5 | 0.2489 | 1.2702 | 0.0099 | 27.331 |
| magic | 1 | 0 | 0.1239 | 1.1482 | 0.0043 | 29.775 |
| magic | 1 | 1 | 0.1619 | 1.1896 | 0.0061 | 30.635 |
| magic | 1 | 2 | 0.1178 | 1.1115 | 0.0056 | 31.223 |
| magic | 1 | 3 | 0.1618 | 1.1754 | 0.0070 | 29.392 |
| magic | 1 | 4 | 0.1553 | 1.2376 | 0.0061 | 29.323 |
| magic | 1 | 5 | 0.1317 | 1.2277 | 0.0062 | 30.076 |
