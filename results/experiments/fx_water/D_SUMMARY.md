
## D: start points and learned dynamics

### Chaos horizon (the simulation itself)

Active PSNR against a run of the simulation from a stored start point, by frames after it (30 per second), mean over runs. Perturbations are relative to the velocity RMS.

| effect | case | 1 | 4 | 8 | 15 | 30 | 60 | 120 | 240 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| waterfall | start+1e-3_same_seed | 72.65 | 70.69 | 69.64 | 68.17 | 63.44 | 60.80 | 56.61 | 59.13 |
| waterfall | start+1e-1_same_seed | 49.58 | 45.77 | 43.92 | 41.06 | 35.67 | 32.79 | 32.30 | 43.96 |
| waterfall | same_start_other_seed | 33.66 | 30.28 | 27.77 | 24.36 | 19.37 | 15.51 | 14.73 | 15.52 |
| waterfall | other_start_same_seed | 14.66 | 14.79 | 14.98 | 15.30 | 16.32 | 19.24 | 25.71 | 35.59 |
| fountain | start+1e-3_same_seed | 73.88 | 70.77 | 69.55 | 68.04 | 64.48 | 60.19 | 51.84 | 34.29 |
| fountain | start+1e-1_same_seed | 52.56 | 48.04 | 45.68 | 42.19 | 35.11 | 29.27 | 21.90 | 17.59 |
| fountain | same_start_other_seed | 46.38 | 37.05 | 28.34 | 21.69 | 17.37 | 16.14 | 13.64 | 13.74 |
| fountain | other_start_same_seed | 15.01 | 14.89 | 14.77 | 14.69 | 15.07 | 16.64 | 16.83 | 15.60 |
| splash | start+1e-3_same_seed | 87.28 | 89.76 | 85.09 | 84.98 | 84.45 | 80.49 | - | - |
| splash | start+1e-1_same_seed | 58.15 | 59.91 | 60.79 | 55.94 | 49.86 | 47.37 | - | - |
| splash | same_start_other_seed | 19.91 | 13.56 | 13.21 | 13.75 | 13.89 | 13.13 | - | - |
| splash | other_start_same_seed | 21.85 | 28.80 | 33.28 | 32.39 | 31.26 | 31.35 | - | - |
| rainshaft | start+1e-3_same_seed | 75.14 | 73.50 | 72.46 | 71.39 | 68.65 | 66.23 | 66.51 | 70.03 |
| rainshaft | start+1e-1_same_seed | 51.04 | 48.87 | 48.27 | 45.85 | 38.58 | 35.73 | 36.23 | 47.56 |
| rainshaft | same_start_other_seed | 30.29 | 26.16 | 23.71 | 21.44 | 19.14 | 17.22 | 16.51 | 17.08 |
| rainshaft | other_start_same_seed | 16.38 | 16.60 | 16.90 | 17.49 | 18.57 | 22.46 | 29.94 | 41.20 |

### Training

Stepper loss: stages 1-2 (normalised squared error plus profiles) and stage 3 (plus activity). Renderer PSNR on its training samples (true fields). Detail constants as calibrated.

| effect | runs x frames | minutes of simulation | stepper loss 1-2 | stepper s | stage 3 loss | stage 3 s | renderer dB | contrast | swirl | grow | start points | KB stored |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| waterfall | 160 x 240 | 21.33 | 0.03331 | 8241 | 0.03330 | 1337 | 37.26 | 1 | 3 | 1 | 8 | 81.6 |
| fountain | 160 x 240 | 21.33 | 0.06851 | 5337 | 0.06660 | 1386 | 37.55 | 0.5 | 3 | 1 | 8 | 81.6 |
| splash | 240 x 90 | 12.00 | 0.05438 | 4675 | 0.05723 | 1165 | 40.68 | 1 | 0 | 1.5 | 16 | 274.2 |
| rainshaft | 160 x 240 | 21.33 | 0.03249 | 4662 | 0.03124 | 1213 | 41.96 | 1 | 3 | 1 | 8 | 81.6 |

Renderer, detail constants and start points from `d-finish` on the trained steppers; stage 3 from `d-tune`.

### Tracking a held-out run from its true start point

Active PSNR against the true run, mean over held-out runs (other seeds and settings), by frames after the start point. `neural`: the rollout effect through the runtime, started from the true state (coarse and fine) with the run's noise seed. `coarse_sim_detail`: the simulation on the same 32-cell grid with the same detail layer, drawn by the simulation's own renderer. `neural_dynamics_true_renderer`: the neural dynamics drawn by that renderer too (separates dynamics from rendering). `renderer_on_true_fields`: the learned renderer on the true fields (frames 1-30, then their mean).

**fountain**

| method | 1 | 4 | 8 | 16 | 30 | 60 | 120 | 240 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| neural | 29.80 | 23.93 | 20.23 | 17.57 | 16.83 | 14.91 | 13.87 | 13.08 |
| neural_dynamics_true_renderer | 31.22 | 24.17 | 20.27 | 17.59 | 16.82 | 14.90 | 13.80 | 13.01 |
| coarse_sim_detail | 29.70 | 22.55 | 18.34 | 15.66 | 14.42 | 12.98 | 10.37 | 9.90 |
| coarse_sim | 20.59 | 19.51 | 18.24 | 16.61 | 15.40 | 13.92 | 10.88 | 10.21 |
| frozen | 27.08 | 16.77 | 13.59 | 11.54 | 10.27 | 10.12 | 10.80 | 10.46 |
| renderer_on_true_fields | 35.52 | 35.51 | 35.49 | 35.45 | 35.48 | 35.45 | 35.45 | 35.45 |

neural - coarse_sim_detail, paired over runs: frame 1: +0.11 [-1.31, +1.74] (tie); frame 8: +1.89 [+1.64, +2.18]; frame 30: +2.41 [+1.94, +2.94]; frame 60: +1.93 [+1.23, +2.66]; frame 240: +3.18 [+2.26, +4.06];

neural_dynamics_true_renderer - coarse_sim_detail, paired over runs: frame 1: +1.52 [+0.14, +3.10]; frame 8: +1.93 [+1.68, +2.24]; frame 30: +2.40 [+1.95, +2.91]; frame 60: +1.92 [+1.22, +2.66]; frame 240: +3.11 [+2.21, +3.98];

**rainshaft**

| method | 1 | 4 | 8 | 16 | 30 | 60 | 120 | 240 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| neural | 34.30 | 30.78 | 27.87 | 25.16 | 23.52 | 21.73 | 21.37 | 19.79 |
| neural_dynamics_true_renderer | 35.30 | 31.07 | 27.89 | 24.97 | 23.30 | 21.59 | 21.19 | 19.76 |
| coarse_sim_detail | 31.73 | 27.82 | 25.07 | 22.45 | 21.11 | 20.03 | 19.13 | 19.15 |
| coarse_sim | 28.89 | 27.25 | 25.97 | 24.28 | 22.88 | 21.46 | 20.68 | 20.35 |
| frozen | 35.71 | 25.82 | 22.46 | 19.77 | 17.66 | 16.17 | 16.52 | 15.55 |
| renderer_on_true_fields | 40.41 | 40.47 | 40.46 | 40.17 | 40.60 | 40.38 | 40.38 | 40.38 |

neural - coarse_sim_detail, paired over runs: frame 1: +2.57 [+1.42, +3.77]; frame 8: +2.81 [+2.25, +3.26]; frame 30: +2.41 [+1.62, +3.00]; frame 60: +1.71 [+0.79, +2.53]; frame 240: +0.64 [-0.74, +1.94] (tie);

neural_dynamics_true_renderer - coarse_sim_detail, paired over runs: frame 1: +3.57 [+2.50, +4.71]; frame 8: +2.83 [+2.27, +3.30]; frame 30: +2.19 [+1.40, +2.79]; frame 60: +1.57 [+0.68, +2.30]; frame 240: +0.61 [-0.75, +1.92] (tie);

**splash**

| method | 1 | 4 | 8 | 16 | 30 | 60 | 120 | 240 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| neural | 20.39 | 15.67 | 14.73 | 14.12 | 13.50 | 13.78 | - | - |
| neural_dynamics_true_renderer | 21.38 | 16.01 | 14.95 | 14.14 | 13.54 | 13.83 | - | - |
| coarse_sim_detail | 11.51 | 11.03 | 11.24 | 10.53 | 9.81 | 10.00 | - | - |
| coarse_sim | 12.12 | 11.40 | 11.45 | 11.14 | 11.15 | 12.24 | - | - |
| frozen | 16.64 | 7.70 | 6.10 | 5.35 | 6.06 | 7.41 | - | - |
| renderer_on_true_fields | 27.33 | 27.62 | 27.31 | 32.40 | 36.05 | 31.42 | 31.42 | 31.42 |

neural - coarse_sim_detail, paired over runs: frame 1: +8.88 [+7.56, +10.64]; frame 8: +3.49 [+2.58, +4.49]; frame 30: +3.69 [+3.38, +3.95]; frame 60: +3.78 [+2.90, +4.79];

neural_dynamics_true_renderer - coarse_sim_detail, paired over runs: frame 1: +9.88 [+8.40, +11.78]; frame 8: +3.71 [+2.84, +4.67]; frame 30: +3.73 [+3.42, +4.00]; frame 60: +3.84 [+2.96, +4.85];

**waterfall**

| method | 1 | 4 | 8 | 16 | 30 | 60 | 120 | 240 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| neural | 28.13 | 23.84 | 20.77 | 18.28 | 17.32 | 17.07 | 16.72 | 16.44 |
| neural_dynamics_true_renderer | 29.38 | 24.27 | 20.91 | 18.33 | 17.34 | 17.09 | 16.76 | 16.43 |
| coarse_sim_detail | 27.84 | 22.21 | 19.34 | 16.85 | 15.50 | 14.77 | 14.47 | 13.31 |
| coarse_sim | 23.06 | 21.73 | 20.74 | 19.14 | 17.85 | 17.07 | 16.62 | 15.12 |
| frozen | 28.27 | 19.20 | 16.24 | 14.63 | 13.93 | 13.62 | 13.94 | 13.67 |
| renderer_on_true_fields | 34.57 | 34.63 | 34.76 | 34.78 | 35.25 | 34.84 | 34.84 | 34.84 |

neural - coarse_sim_detail, paired over runs: frame 1: +0.29 [-0.47, +1.00] (tie); frame 8: +1.43 [+1.27, +1.57]; frame 30: +1.82 [+1.52, +2.10]; frame 60: +2.31 [+1.93, +2.71]; frame 240: +3.13 [+1.81, +4.79];

neural_dynamics_true_renderer - coarse_sim_detail, paired over runs: frame 1: +1.54 [+0.90, +2.10]; frame 8: +1.57 [+1.34, +1.76]; frame 30: +1.84 [+1.54, +2.12]; frame 60: +2.32 [+1.90, +2.77]; frame 240: +3.13 [+1.82, +4.74];

### Endless runs at held-out settings

The ten held-out settings of study B, a new seed each, 10 s per run (explosions: their 3 s). Distances of each method's frame statistics to a real run of the simulation at that setting; `real_other_seed` is a second real run with another seed, the floor. Lower is better except motion ratio (1 is right) and mean-frame PSNR (higher is closer).

**fountain**

| method | spectrum L1 | motion ratio | coverage L1 | emission L1 | mean-frame PSNR |
|---|---:|---:|---:|---:|---:|
| real_other_seed | 0.043 | 0.97 | 0.0049 | 0.0000 | 27.03 |
| neural | 0.223 | 0.88 | 0.0198 | 0.0004 | 22.60 |
| flipbook_nearest | 0.084 | 1.07 | 0.0162 | 0.0000 | 19.35 |
| coarse_sim_detail | 0.178 | 0.68 | 0.0169 | 0.0000 | 16.34 |
| coarse_sim | 0.895 | 0.19 | 0.0127 | 0.0000 | 16.15 |

Spectrum distance, paired over settings: neural - grid_k8 -; neural - flipbook_nearest +0.138 [+0.117, +0.158]; neural - coarse_sim_detail +0.045 [+0.029, +0.059]; neural - real_other_seed +0.180 [+0.170, +0.190].

Mean-frame PSNR (dB), paired: neural - flipbook_nearest +3.25 [+1.57, +4.97]; neural - coarse_sim_detail +6.26 [+5.10, +7.43]; neural - real_other_seed -4.44 [-6.49, -2.54]. Motion error |ln(motion ratio)|, paired: neural - flipbook_nearest +0.093 [-0.070, +0.270] (tie); neural - coarse_sim_detail -0.125 [-0.218, -0.028]; neural - real_other_seed +0.207 [+0.066, +0.355]. Coverage distance, paired: neural - flipbook_nearest +0.0036 [-0.0022, +0.0087] (tie); neural - real_other_seed +0.0149 [+0.0104, +0.0209].

**rainshaft**

| method | spectrum L1 | motion ratio | coverage L1 | emission L1 | mean-frame PSNR |
|---|---:|---:|---:|---:|---:|
| real_other_seed | 0.062 | 1.07 | 0.0256 | 0.0000 | 28.13 |
| neural | 0.293 | 0.99 | 0.0305 | 0.0000 | 26.73 |
| flipbook_nearest | 0.141 | 1.11 | 0.0401 | 0.0000 | 22.10 |
| coarse_sim_detail | 0.635 | 1.19 | 0.0243 | 0.0000 | 25.70 |
| coarse_sim | 0.200 | 0.52 | 0.0242 | 0.0000 | 25.57 |

Spectrum distance, paired over settings: neural - grid_k8 -; neural - flipbook_nearest +0.152 [+0.083, +0.222]; neural - coarse_sim_detail -0.342 [-0.389, -0.298]; neural - real_other_seed +0.232 [+0.155, +0.307].

Mean-frame PSNR (dB), paired: neural - flipbook_nearest +4.63 [+3.27, +5.94]; neural - coarse_sim_detail +1.03 [-0.53, +2.56] (tie); neural - real_other_seed -1.41 [-2.91, +0.09] (tie). Motion error |ln(motion ratio)|, paired: neural - flipbook_nearest -0.056 [-0.134, +0.032] (tie); neural - coarse_sim_detail -0.010 [-0.102, +0.089] (tie); neural - real_other_seed +0.095 [+0.047, +0.148]. Coverage distance, paired: neural - flipbook_nearest -0.0096 [-0.0229, +0.0020] (tie); neural - real_other_seed +0.0048 [-0.0036, +0.0135] (tie).

**splash**

| method | spectrum L1 | motion ratio | coverage L1 | emission L1 | mean-frame PSNR |
|---|---:|---:|---:|---:|---:|
| real_other_seed | 0.086 | 0.98 | 0.0027 | 0.0000 | 29.84 |
| neural | 0.140 | 0.92 | 0.0069 | 0.0000 | 27.30 |
| flipbook_nearest | 0.086 | 0.87 | 0.0064 | 0.0000 | 27.20 |
| coarse_sim_detail | 0.273 | 1.09 | 0.0031 | 0.0000 | 23.16 |
| coarse_sim | 0.629 | 0.74 | 0.0102 | 0.0000 | 22.92 |

Spectrum distance, paired over settings: neural - grid_k8 -; neural - flipbook_nearest +0.054 [+0.014, +0.097]; neural - coarse_sim_detail -0.133 [-0.195, -0.064]; neural - real_other_seed +0.054 [+0.016, +0.092].

Mean-frame PSNR (dB), paired: neural - flipbook_nearest +0.09 [-0.96, +1.22] (tie); neural - coarse_sim_detail +4.14 [+3.23, +5.00]; neural - real_other_seed -2.54 [-3.57, -1.62]. Motion error |ln(motion ratio)|, paired: neural - flipbook_nearest +0.018 [-0.146, +0.183] (tie); neural - coarse_sim_detail +0.121 [-0.016, +0.246] (tie); neural - real_other_seed +0.121 [+0.002, +0.229]. Coverage distance, paired: neural - flipbook_nearest +0.0005 [-0.0029, +0.0044] (tie); neural - real_other_seed +0.0042 [+0.0012, +0.0076].

**waterfall**

| method | spectrum L1 | motion ratio | coverage L1 | emission L1 | mean-frame PSNR |
|---|---:|---:|---:|---:|---:|
| real_other_seed | 0.033 | 1.04 | 0.0240 | 0.0000 | 28.80 |
| neural | 0.237 | 0.98 | 0.0340 | 0.0001 | 23.26 |
| flipbook_nearest | 0.074 | 0.87 | 0.0297 | 0.0000 | 22.32 |
| coarse_sim_detail | 0.284 | 1.09 | 0.0374 | 0.0000 | 19.86 |
| coarse_sim | 0.595 | 0.48 | 0.0271 | 0.0000 | 20.80 |

Spectrum distance, paired over settings: neural - grid_k8 -; neural - flipbook_nearest +0.162 [+0.116, +0.203]; neural - coarse_sim_detail -0.047 [-0.077, -0.018]; neural - real_other_seed +0.204 [+0.153, +0.247].

Mean-frame PSNR (dB), paired: neural - flipbook_nearest +0.94 [-0.81, +2.91] (tie); neural - coarse_sim_detail +3.40 [+1.80, +4.75]; neural - real_other_seed -5.54 [-8.20, -3.27]. Motion error |ln(motion ratio)|, paired: neural - flipbook_nearest -0.053 [-0.135, +0.032] (tie); neural - coarse_sim_detail +0.002 [-0.048, +0.064] (tie); neural - real_other_seed +0.034 [-0.018, +0.099] (tie). Coverage distance, paired: neural - flipbook_nearest +0.0043 [-0.0037, +0.0121] (tie); neural - real_other_seed +0.0100 [+0.0002, +0.0199].

### One minute of play

Statistics of each 10 s window of a 60 s neural run (shards of 6 s from start points, the default) against a real 10 s run at the same setting (two held-out settings): if the effect drifted, the distances would grow window by window.

| effect | setting | window | spectrum L1 | motion ratio | coverage L1 | mean-frame PSNR |
|---|---:|---:|---:|---:|---:|---:|
| waterfall | 0 | 0 | 0.2242 | 0.8085 | 0.0416 | 21.820 |
| waterfall | 0 | 1 | 0.1844 | 0.7013 | 0.0531 | 19.977 |
| waterfall | 0 | 2 | 0.3034 | 1.0113 | 0.0543 | 21.684 |
| waterfall | 0 | 3 | 0.1733 | 0.6771 | 0.0592 | 18.889 |
| waterfall | 0 | 4 | 0.2848 | 1.0673 | 0.0385 | 21.697 |
| waterfall | 0 | 5 | 0.2974 | 1.0430 | 0.0273 | 21.297 |
| waterfall | 1 | 0 | 0.1134 | 0.8916 | 0.0186 | 27.026 |
| waterfall | 1 | 1 | 0.1275 | 0.9668 | 0.0153 | 24.650 |
| waterfall | 1 | 2 | 0.1343 | 0.9410 | 0.0069 | 26.391 |
| waterfall | 1 | 3 | 0.1650 | 0.9772 | 0.0064 | 22.824 |
| waterfall | 1 | 4 | 0.1338 | 0.9312 | 0.0125 | 27.681 |
| waterfall | 1 | 5 | 0.1306 | 0.8829 | 0.0355 | 23.082 |
| fountain | 0 | 0 | 0.2303 | 1.0328 | 0.0151 | 24.748 |
| fountain | 0 | 1 | 0.2263 | 0.8663 | 0.0120 | 24.235 |
| fountain | 0 | 2 | 0.2140 | 1.0565 | 0.0090 | 23.648 |
| fountain | 0 | 3 | 0.2169 | 0.8668 | 0.0150 | 22.577 |
| fountain | 0 | 4 | 0.2359 | 1.2193 | 0.0145 | 23.438 |
| fountain | 0 | 5 | 0.2186 | 1.0989 | 0.0132 | 24.397 |
| fountain | 1 | 0 | 0.2110 | 1.0373 | 0.0106 | 21.940 |
| fountain | 1 | 1 | 0.2022 | 1.1591 | 0.0136 | 21.959 |
| fountain | 1 | 2 | 0.2012 | 1.1989 | 0.0124 | 21.848 |
| fountain | 1 | 3 | 0.1881 | 1.2387 | 0.0095 | 21.754 |
| fountain | 1 | 4 | 0.1977 | 1.1245 | 0.0054 | 22.905 |
| fountain | 1 | 5 | 0.2308 | 0.9255 | 0.0159 | 20.069 |
| rainshaft | 0 | 0 | 0.2082 | 0.8330 | 0.0458 | 24.646 |
| rainshaft | 0 | 1 | 0.2092 | 0.8430 | 0.0479 | 23.862 |
| rainshaft | 0 | 2 | 0.2686 | 0.9680 | 0.0352 | 25.392 |
| rainshaft | 0 | 3 | 0.2076 | 0.7963 | 0.0424 | 24.681 |
| rainshaft | 0 | 4 | 0.2927 | 1.1591 | 0.0161 | 28.710 |
| rainshaft | 0 | 5 | 0.2740 | 1.0324 | 0.0313 | 28.364 |
| rainshaft | 1 | 0 | 0.1602 | 0.8482 | 0.0484 | 24.066 |
| rainshaft | 1 | 1 | 0.1640 | 0.8184 | 0.0446 | 24.811 |
| rainshaft | 1 | 2 | 0.2052 | 0.8699 | 0.0328 | 27.450 |
| rainshaft | 1 | 3 | 0.1863 | 0.8311 | 0.0344 | 26.705 |
| rainshaft | 1 | 4 | 0.1749 | 0.7745 | 0.0433 | 25.142 |
| rainshaft | 1 | 5 | 0.1918 | 0.7833 | 0.0388 | 25.854 |
