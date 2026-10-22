
## D: start points and learned dynamics

### Chaos horizon (the simulation itself)

Active PSNR against a run of the simulation from a stored start point, by frames after it (30 per second), mean over runs. Perturbations are relative to the velocity RMS.

| effect | case | 1 | 4 | 8 | 15 | 30 | 60 | 120 | 240 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| candle | start+1e-3_same_seed | 75.60 | 73.52 | 72.42 | 70.60 | 68.75 | 65.74 | 60.48 | 62.13 |
| candle | start+1e-1_same_seed | 54.37 | 51.52 | 49.54 | 47.39 | 43.42 | 37.95 | 29.83 | 38.04 |
| candle | same_start_other_seed | 36.39 | 28.73 | 27.28 | 25.13 | 21.94 | 18.82 | 17.88 | 17.59 |
| candle | other_start_same_seed | 18.56 | 18.85 | 18.98 | 19.14 | 18.86 | 20.34 | 21.59 | 26.63 |
| smokering | start+1e-3_same_seed | 74.56 | 73.69 | 71.03 | 70.35 | 69.30 | 66.53 | 66.25 | - |
| smokering | start+1e-1_same_seed | 54.91 | 51.94 | 50.23 | 48.86 | 48.08 | 49.55 | 53.75 | - |
| smokering | same_start_other_seed | 60.92 | 48.54 | 39.09 | 30.91 | 24.73 | 21.39 | 24.55 | - |
| smokering | other_start_same_seed | 31.32 | 32.01 | 32.83 | 34.10 | 36.26 | 37.94 | 41.92 | - |
| heavygas | start+1e-3_same_seed | 77.37 | 74.41 | 73.48 | 70.89 | 70.24 | 69.16 | 68.08 | 67.04 |
| heavygas | start+1e-1_same_seed | 58.33 | 53.12 | 50.72 | 48.59 | 46.41 | 43.67 | 43.92 | 46.44 |
| heavygas | same_start_other_seed | 39.06 | 28.92 | 24.65 | 20.97 | 15.99 | 13.04 | 12.19 | 12.87 |
| heavygas | other_start_same_seed | 10.32 | 10.38 | 10.43 | 10.53 | 10.81 | 11.68 | 13.98 | 18.23 |

### Training

Stepper loss: stages 1-2 (normalised squared error plus profiles) and stage 3 (plus activity). Renderer PSNR on its training samples (true fields). Detail constants as calibrated.

| effect | runs x frames | minutes of simulation | stepper loss 1-2 | stepper s | stage 3 loss | stage 3 s | renderer dB | contrast | swirl | grow | start points | KB stored |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| candle | 160 x 240 | 21.33 | 0.03997 | 5210 | 0.03908 | 1306 | 34.98 | 0.5 | 3 | 1 | 8 | 81.6 |
| smokering | 240 x 120 | 16.00 | 0.00665 | 6151 | 0.00643 | 1086 | 42.30 | 0.5 | 0 | 1 | 16 | 274.2 |
| heavygas | 160 x 240 | 21.33 | 0.02122 | 4865 | 0.02083 | 1184 | 39.97 | 1 | 0 | 1 | 8 | 145.7 |

Renderer, detail constants and start points from `d-finish` on the trained steppers; stage 3 from `d-tune`.

### Tracking a held-out run from its true start point

Active PSNR against the true run, mean over held-out runs (other seeds and settings), by frames after the start point. `neural`: the rollout effect through the runtime, started from the true state (coarse and fine) with the run's noise seed. `coarse_sim_detail`: the simulation on the same 32-cell grid with the same detail layer, drawn by the simulation's own renderer. `neural_dynamics_true_renderer`: the neural dynamics drawn by that renderer too (separates dynamics from rendering). `renderer_on_true_fields`: the learned renderer on the true fields (frames 1-30, then their mean).

**candle**

| method | 1 | 4 | 8 | 16 | 30 | 60 | 120 | 240 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| neural | 29.14 | 22.32 | 20.17 | 18.54 | 18.16 | 16.27 | 14.40 | 14.81 |
| neural_dynamics_true_renderer | 31.32 | 22.30 | 20.15 | 18.57 | 18.18 | 16.23 | 14.39 | 14.73 |
| coarse_sim_detail | 27.69 | 19.83 | 17.32 | 15.81 | 15.26 | 14.65 | 14.13 | 14.09 |
| coarse_sim | 20.07 | 18.26 | 17.52 | 16.77 | 16.11 | 15.39 | 14.72 | 14.93 |
| frozen | 31.08 | 20.71 | 17.19 | 15.64 | 15.33 | 15.09 | 15.48 | 14.59 |
| renderer_on_true_fields | 31.75 | 31.77 | 31.91 | 32.22 | 32.37 | 32.13 | 32.13 | 32.13 |

neural - coarse_sim_detail, paired over runs: frame 1: +1.45 [-0.07, +2.77] (tie); frame 8: +2.85 [+1.99, +3.73]; frame 30: +2.90 [+2.35, +3.41]; frame 60: +1.61 [+0.76, +2.41]; frame 240: +0.72 [-0.28, +1.70] (tie);

neural_dynamics_true_renderer - coarse_sim_detail, paired over runs: frame 1: +3.63 [+2.05, +5.10]; frame 8: +2.83 [+2.04, +3.62]; frame 30: +2.92 [+2.32, +3.46]; frame 60: +1.58 [+0.72, +2.36]; frame 240: +0.65 [-0.31, +1.49] (tie);

**heavygas**

| method | 1 | 4 | 8 | 16 | 30 | 60 | 120 | 240 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| neural | 32.12 | 25.54 | 21.54 | 18.05 | 17.03 | 15.14 | 12.96 | 11.48 |
| neural_dynamics_true_renderer | 34.73 | 26.10 | 21.80 | 18.19 | 17.12 | 15.22 | 13.05 | 11.46 |
| coarse_sim_detail | 35.62 | 25.55 | 20.51 | 16.85 | 15.43 | 13.03 | 11.71 | 11.58 |
| coarse_sim | 19.07 | 19.05 | 18.89 | 18.54 | 17.56 | 15.68 | 14.09 | 14.33 |
| frozen | 34.93 | 23.36 | 18.13 | 13.81 | 10.90 | 8.75 | 8.28 | 8.29 |
| renderer_on_true_fields | 35.99 | 35.91 | 35.74 | 35.61 | 35.56 | 35.69 | 35.69 | 35.69 |

neural - coarse_sim_detail, paired over runs: frame 1: -3.50 [-5.16, -1.89]; frame 8: +1.03 [+0.57, +1.46]; frame 30: +1.61 [+0.92, +2.15]; frame 60: +2.11 [+1.57, +2.64]; frame 240: -0.10 [-0.44, +0.29] (tie);

neural_dynamics_true_renderer - coarse_sim_detail, paired over runs: frame 1: -0.90 [-2.11, +0.53] (tie); frame 8: +1.29 [+0.77, +1.78]; frame 30: +1.70 [+0.97, +2.30]; frame 60: +2.20 [+1.66, +2.72]; frame 240: -0.12 [-0.46, +0.28] (tie);

**smokering**

| method | 1 | 4 | 8 | 16 | 30 | 60 | 120 | 240 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| neural | 24.63 | 25.85 | 27.78 | 29.61 | 29.88 | 27.47 | - | - |
| neural_dynamics_true_renderer | 30.94 | 31.27 | 31.62 | 31.94 | 30.72 | 27.58 | - | - |
| coarse_sim_detail | 29.42 | 28.83 | 27.08 | 24.22 | 22.38 | 20.49 | - | - |
| coarse_sim | 27.37 | 27.53 | 26.98 | 25.30 | 23.73 | 21.47 | - | - |
| frozen | 24.25 | 15.49 | 12.17 | 10.66 | 10.67 | 11.97 | - | - |
| renderer_on_true_fields | 25.12 | 26.58 | 29.78 | 32.44 | 35.07 | 31.48 | 31.48 | 31.48 |

neural - coarse_sim_detail, paired over runs: frame 1: -4.79 [-5.79, -3.94]; frame 8: +0.69 [+0.32, +1.07]; frame 30: +7.51 [+6.62, +8.44]; frame 60: +6.98 [+5.63, +8.34];

neural_dynamics_true_renderer - coarse_sim_detail, paired over runs: frame 1: +1.52 [+0.65, +2.24]; frame 8: +4.54 [+4.21, +4.86]; frame 30: +8.34 [+7.35, +9.32]; frame 60: +7.09 [+5.60, +8.62];

### Endless runs at held-out settings

The ten held-out settings of study B, a new seed each, 10 s per run (explosions: their 3 s). Distances of each method's frame statistics to a real run of the simulation at that setting; `real_other_seed` is a second real run with another seed, the floor. Lower is better except motion ratio (1 is right) and mean-frame PSNR (higher is closer).

**candle**

| method | spectrum L1 | motion ratio | coverage L1 | emission L1 | mean-frame PSNR |
|---|---:|---:|---:|---:|---:|
| real_other_seed | 0.015 | 1.01 | 0.0019 | 0.0000 | 38.49 |
| neural | 0.132 | 1.06 | 0.0051 | 0.0002 | 26.83 |
| flipbook_nearest | 0.080 | 0.94 | 0.0051 | 0.0002 | 24.30 |
| coarse_sim_detail | 0.127 | 0.92 | 0.0064 | 0.0003 | 26.55 |
| coarse_sim | 0.708 | 0.39 | 0.0108 | 0.0003 | 25.89 |

Spectrum distance, paired over settings: neural - grid_k8 -; neural - flipbook_nearest +0.052 [+0.026, +0.079]; neural - coarse_sim_detail +0.006 [-0.009, +0.019] (tie); neural - real_other_seed +0.117 [+0.098, +0.137].

Mean-frame PSNR (dB), paired: neural - flipbook_nearest +2.53 [+0.53, +4.58]; neural - coarse_sim_detail +0.28 [-1.02, +1.76] (tie); neural - real_other_seed -11.66 [-14.91, -8.72]. Motion error |ln(motion ratio)|, paired: neural - flipbook_nearest -0.042 [-0.203, +0.105] (tie); neural - coarse_sim_detail +0.001 [-0.096, +0.093] (tie); neural - real_other_seed +0.092 [+0.018, +0.178]. Coverage distance, paired: neural - flipbook_nearest -0.0000 [-0.0023, +0.0025] (tie); neural - real_other_seed +0.0032 [+0.0018, +0.0049].

**heavygas**

| method | spectrum L1 | motion ratio | coverage L1 | emission L1 | mean-frame PSNR |
|---|---:|---:|---:|---:|---:|
| real_other_seed | 0.078 | 0.97 | 0.0111 | 0.0000 | 26.35 |
| neural | 0.129 | 1.06 | 0.0141 | 0.0001 | 22.28 |
| flipbook_nearest | 0.198 | 0.98 | 0.0132 | 0.0000 | 22.12 |
| coarse_sim_detail | 0.156 | 1.11 | 0.0156 | 0.0000 | 23.09 |
| coarse_sim | 1.218 | 0.54 | 0.0085 | 0.0000 | 24.30 |

Spectrum distance, paired over settings: neural - grid_k8 -; neural - flipbook_nearest -0.069 [-0.121, -0.015]; neural - coarse_sim_detail -0.027 [-0.065, +0.015] (tie); neural - real_other_seed +0.051 [+0.006, +0.093].

Mean-frame PSNR (dB), paired: neural - flipbook_nearest +0.16 [-1.42, +1.71] (tie); neural - coarse_sim_detail -0.80 [-2.03, +0.42] (tie); neural - real_other_seed -4.07 [-6.17, -1.68]. Motion error |ln(motion ratio)|, paired: neural - flipbook_nearest -0.120 [-0.260, +0.019] (tie); neural - coarse_sim_detail -0.002 [-0.092, +0.079] (tie); neural - real_other_seed +0.013 [-0.094, +0.119] (tie). Coverage distance, paired: neural - flipbook_nearest +0.0009 [-0.0036, +0.0055] (tie); neural - real_other_seed +0.0030 [-0.0049, +0.0084] (tie).

**smokering**

| method | spectrum L1 | motion ratio | coverage L1 | emission L1 | mean-frame PSNR |
|---|---:|---:|---:|---:|---:|
| real_other_seed | 0.049 | 1.01 | 0.0010 | 0.0000 | 33.68 |
| neural | 0.718 | 1.03 | 0.0016 | 0.0001 | 33.89 |
| flipbook_nearest | 0.091 | 0.84 | 0.0051 | 0.0000 | 30.19 |
| coarse_sim_detail | 0.987 | 0.92 | 0.0010 | 0.0000 | 32.93 |
| coarse_sim | 0.319 | 0.82 | 0.0011 | 0.0000 | 33.33 |

Spectrum distance, paired over settings: neural - grid_k8 -; neural - flipbook_nearest +0.627 [+0.594, +0.662]; neural - coarse_sim_detail -0.268 [-0.340, -0.190]; neural - real_other_seed +0.669 [+0.642, +0.698].

Mean-frame PSNR (dB), paired: neural - flipbook_nearest +3.70 [+2.63, +4.64]; neural - coarse_sim_detail +0.96 [-0.24, +2.28] (tie); neural - real_other_seed +0.21 [-1.91, +2.22] (tie). Motion error |ln(motion ratio)|, paired: neural - flipbook_nearest -0.150 [-0.237, -0.071]; neural - coarse_sim_detail -0.015 [-0.052, +0.026] (tie); neural - real_other_seed +0.013 [-0.028, +0.052] (tie). Coverage distance, paired: neural - flipbook_nearest -0.0035 [-0.0053, -0.0018]; neural - real_other_seed +0.0006 [+0.0002, +0.0011].

### One minute of play

Statistics of each 10 s window of a 60 s neural run (shards of 6 s from start points, the default) against a real 10 s run at the same setting (two held-out settings): if the effect drifted, the distances would grow window by window.

| effect | setting | window | spectrum L1 | motion ratio | coverage L1 | mean-frame PSNR |
|---|---:|---:|---:|---:|---:|---:|
| candle | 0 | 0 | 0.1869 | 0.8041 | 0.0036 | 25.481 |
| candle | 0 | 1 | 0.1450 | 0.8645 | 0.0037 | 29.257 |
| candle | 0 | 2 | 0.1760 | 0.9751 | 0.0031 | 29.967 |
| candle | 0 | 3 | 0.1747 | 0.7626 | 0.0035 | 25.371 |
| candle | 0 | 4 | 0.1705 | 0.9226 | 0.0029 | 28.905 |
| candle | 0 | 5 | 0.1552 | 0.8625 | 0.0035 | 24.804 |
| candle | 1 | 0 | 0.1168 | 1.5767 | 0.0032 | 24.727 |
| candle | 1 | 1 | 0.0831 | 1.4622 | 0.0024 | 28.708 |
| candle | 1 | 2 | 0.1166 | 1.4228 | 0.0024 | 25.839 |
| candle | 1 | 3 | 0.0830 | 1.5222 | 0.0043 | 25.114 |
| candle | 1 | 4 | 0.0934 | 1.4691 | 0.0031 | 25.129 |
| candle | 1 | 5 | 0.0814 | 1.2994 | 0.0015 | 26.597 |
| heavygas | 0 | 0 | 0.0648 | 0.9892 | 0.0128 | 21.229 |
| heavygas | 0 | 1 | 0.1326 | 0.9563 | 0.0145 | 24.744 |
| heavygas | 0 | 2 | 0.1061 | 1.0363 | 0.0088 | 24.916 |
| heavygas | 0 | 3 | 0.0568 | 1.1898 | 0.0099 | 19.246 |
| heavygas | 0 | 4 | 0.0513 | 0.9886 | 0.0077 | 25.258 |
| heavygas | 0 | 5 | 0.1298 | 0.9932 | 0.0048 | 25.357 |
| heavygas | 1 | 0 | 0.0886 | 0.9808 | 0.0122 | 20.842 |
| heavygas | 1 | 1 | 0.0834 | 1.3923 | 0.0066 | 22.652 |
| heavygas | 1 | 2 | 0.1452 | 1.3790 | 0.0151 | 22.772 |
| heavygas | 1 | 3 | 0.1348 | 0.9839 | 0.0097 | 19.499 |
| heavygas | 1 | 4 | 0.0902 | 1.3716 | 0.0065 | 24.088 |
| heavygas | 1 | 5 | 0.1635 | 1.3918 | 0.0103 | 24.509 |
