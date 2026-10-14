
## D: start points and learned dynamics

### Chaos horizon (the simulation itself)

Active PSNR against a run of the simulation from a stored start point, by frames after it (30 per second), mean over runs. Perturbations are relative to the velocity RMS.

| effect | case | 1 | 4 | 8 | 15 | 30 | 60 | 120 | 240 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| jet | start+1e-3_same_seed | 71.76 | 68.95 | 66.68 | 64.56 | 62.08 | 56.91 | 52.59 | 44.52 |
| jet | start+1e-1_same_seed | 48.91 | 42.96 | 39.17 | 34.78 | 29.47 | 26.01 | 23.76 | 23.69 |
| jet | same_start_other_seed | 36.78 | 32.17 | 25.76 | 20.88 | 16.12 | 14.80 | 14.56 | 15.55 |
| jet | other_start_same_seed | 13.98 | 14.02 | 14.06 | 14.19 | 14.63 | 16.26 | 17.86 | 20.23 |
| orb | start+1e-3_same_seed | 82.58 | 81.63 | 75.18 | 73.23 | 72.42 | 69.39 | 68.69 | 58.53 |
| orb | start+1e-1_same_seed | 59.29 | 54.04 | 50.82 | 48.27 | 46.69 | 44.50 | 49.03 | 44.27 |
| orb | same_start_other_seed | 19.40 | 18.08 | 16.10 | 14.62 | 12.18 | 12.07 | 12.64 | 13.84 |
| orb | other_start_same_seed | 13.01 | 13.99 | 15.28 | 17.48 | 19.12 | 23.30 | 33.01 | 40.12 |
| vortex | start+1e-3_same_seed | 73.69 | 71.12 | 70.03 | 69.09 | 67.09 | 64.84 | 65.42 | 59.93 |
| vortex | start+1e-1_same_seed | 51.25 | 45.94 | 43.53 | 41.61 | 37.77 | 36.48 | 37.99 | 42.83 |
| vortex | same_start_other_seed | 43.83 | 35.68 | 30.28 | 24.58 | 20.62 | 19.01 | 18.53 | 17.33 |
| vortex | other_start_same_seed | 14.84 | 15.25 | 15.84 | 16.60 | 17.77 | 19.62 | 22.22 | 25.41 |

### Training

Stepper loss: stages 1-2 (normalised squared error plus profiles) and stage 3 (plus activity). Renderer PSNR on its training samples (true fields). Detail constants as calibrated.

| effect | runs x frames | minutes of simulation | stepper loss 1-2 | stepper s | stage 3 loss | stage 3 s | renderer dB | contrast | swirl | grow | start points | KB stored |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| jet | 160 x 240 | 21.33 | 0.08516 | 7274 | 0.09965 | 1436 | 34.87 | 1 | 3 | 1 | 8 | 81.6 |
| orb | 160 x 240 | 21.33 | 0.09065 | 7211 | 0.08587 | 1370 | 36.18 | 0.5 | 0.5 | 1 | 8 | 81.6 |

Renderer, detail constants and start points from `d-finish` on the trained steppers; stage 3 from `d-tune`.

### Tracking a held-out run from its true start point

Active PSNR against the true run, mean over held-out runs (other seeds and settings), by frames after the start point. `neural`: the rollout effect through the runtime, started from the true state (coarse and fine) with the run's noise seed. `coarse_sim_detail`: the simulation on the same 32-cell grid with the same detail layer, drawn by the simulation's own renderer. `neural_dynamics_true_renderer`: the neural dynamics drawn by that renderer too (separates dynamics from rendering). `renderer_on_true_fields`: the learned renderer on the true fields (frames 1-30, then their mean).

**jet**

| method | 1 | 4 | 8 | 16 | 30 | 60 | 120 | 240 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| neural | 25.22 | 21.20 | 18.85 | 16.83 | 16.02 | 14.58 | 14.40 | 14.20 |
| neural_dynamics_true_renderer | 25.74 | 21.28 | 18.80 | 16.71 | 15.92 | 14.46 | 14.28 | 14.12 |
| coarse_sim_detail | 23.95 | 18.90 | 16.21 | 14.33 | 13.44 | 13.14 | 13.81 | 13.35 |
| coarse_sim | 19.58 | 18.29 | 17.15 | 15.83 | 14.50 | 13.94 | 14.73 | 14.13 |
| frozen | 22.60 | 15.04 | 13.67 | 13.28 | 12.90 | 13.08 | 13.66 | 13.30 |
| renderer_on_true_fields | 32.46 | 32.37 | 32.52 | 32.68 | 32.78 | 32.62 | 32.62 | 32.62 |

neural - coarse_sim_detail, paired over runs: frame 1: +1.27 [+0.22, +2.20]; frame 8: +2.63 [+2.08, +3.11]; frame 30: +2.57 [+1.98, +3.17]; frame 60: +1.44 [+0.71, +2.08]; frame 240: +0.85 [+0.61, +1.11];

neural_dynamics_true_renderer - coarse_sim_detail, paired over runs: frame 1: +1.79 [+0.81, +2.64]; frame 8: +2.59 [+2.04, +3.07]; frame 30: +2.47 [+1.88, +3.07]; frame 60: +1.32 [+0.60, +1.94]; frame 240: +0.77 [+0.52, +1.04];

**orb**

| method | 1 | 4 | 8 | 16 | 30 | 60 | 120 | 240 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| neural | 29.48 | 24.46 | 20.87 | 17.96 | 16.24 | 15.57 | 15.32 | 15.69 |
| neural_dynamics_true_renderer | 31.20 | 24.81 | 20.88 | 17.87 | 16.11 | 15.44 | 15.26 | 15.61 |
| coarse_sim_detail | 32.90 | 26.33 | 22.96 | 19.30 | 16.61 | 15.96 | 15.84 | 16.69 |
| coarse_sim | 23.78 | 23.55 | 23.01 | 21.36 | 18.95 | 18.44 | 18.10 | 19.21 |
| frozen | 32.46 | 21.87 | 17.54 | 14.46 | 12.64 | 11.80 | 12.08 | 12.00 |
| renderer_on_true_fields | 33.90 | 34.02 | 33.98 | 34.13 | 34.00 | 34.01 | 34.01 | 34.01 |

neural - coarse_sim_detail, paired over runs: frame 1: -3.42 [-4.60, -2.43]; frame 8: -2.09 [-3.19, -1.11]; frame 30: -0.38 [-1.68, +0.89] (tie); frame 60: -0.39 [-1.69, +0.89] (tie); frame 240: -1.00 [-2.83, +0.61] (tie);

neural_dynamics_true_renderer - coarse_sim_detail, paired over runs: frame 1: -1.70 [-2.85, -0.75]; frame 8: -2.08 [-3.18, -1.11]; frame 30: -0.50 [-1.83, +0.79] (tie); frame 60: -0.52 [-1.83, +0.77] (tie); frame 240: -1.08 [-2.84, +0.48] (tie);

### Endless runs at held-out settings

The ten held-out settings of study B, a new seed each, 10 s per run (explosions: their 3 s). Distances of each method's frame statistics to a real run of the simulation at that setting; `real_other_seed` is a second real run with another seed, the floor. Lower is better except motion ratio (1 is right) and mean-frame PSNR (higher is closer).

**jet**

| method | spectrum L1 | motion ratio | coverage L1 | emission L1 | mean-frame PSNR |
|---|---:|---:|---:|---:|---:|
| real_other_seed | 0.045 | 1.04 | 0.0123 | 0.0006 | 30.41 |
| neural | 0.183 | 0.86 | 0.0196 | 0.0013 | 26.31 |
| flipbook_nearest | 0.061 | 0.90 | 0.0186 | 0.0008 | 24.42 |
| coarse_sim_detail | 0.193 | 0.76 | 0.0181 | 0.0010 | 21.57 |
| coarse_sim | 1.020 | 0.35 | 0.0165 | 0.0012 | 21.20 |

Spectrum distance, paired over settings: neural - grid_k8 -; neural - flipbook_nearest +0.122 [+0.091, +0.153]; neural - coarse_sim_detail -0.010 [-0.028, +0.008] (tie); neural - real_other_seed +0.138 [+0.090, +0.184].

Mean-frame PSNR (dB), paired: neural - flipbook_nearest +1.89 [+1.28, +2.61]; neural - coarse_sim_detail +4.75 [+3.41, +5.91]; neural - real_other_seed -4.10 [-4.93, -3.25]. Motion error |ln(motion ratio)|, paired: neural - flipbook_nearest -0.013 [-0.138, +0.113] (tie); neural - coarse_sim_detail -0.150 [-0.281, -0.028]; neural - real_other_seed +0.083 [+0.036, +0.134]. Coverage distance, paired: neural - flipbook_nearest +0.0009 [-0.0083, +0.0094] (tie); neural - real_other_seed +0.0073 [+0.0020, +0.0126].

**orb**

| method | spectrum L1 | motion ratio | coverage L1 | emission L1 | mean-frame PSNR |
|---|---:|---:|---:|---:|---:|
| real_other_seed | 0.069 | 1.04 | 0.0065 | 0.0077 | 28.80 |
| neural | 0.205 | 0.89 | 0.0063 | 0.0074 | 27.32 |
| flipbook_nearest | 0.167 | 1.10 | 0.0065 | 0.0079 | 26.04 |
| coarse_sim_detail | 0.386 | 1.09 | 0.0060 | 0.0070 | 28.72 |
| coarse_sim | 0.802 | 0.71 | 0.0060 | 0.0069 | 29.09 |

Spectrum distance, paired over settings: neural - grid_k8 -; neural - flipbook_nearest +0.037 [-0.053, +0.132] (tie); neural - coarse_sim_detail -0.181 [-0.246, -0.107]; neural - real_other_seed +0.136 [+0.060, +0.230].

Mean-frame PSNR (dB), paired: neural - flipbook_nearest +1.28 [+0.23, +2.32]; neural - coarse_sim_detail -1.40 [-3.42, +0.61] (tie); neural - real_other_seed -1.48 [-3.68, +0.39] (tie). Motion error |ln(motion ratio)|, paired: neural - flipbook_nearest -0.010 [-0.100, +0.086] (tie); neural - coarse_sim_detail +0.038 [-0.041, +0.121] (tie); neural - real_other_seed +0.059 [-0.021, +0.139] (tie). Coverage distance, paired: neural - flipbook_nearest -0.0002 [-0.0018, +0.0017] (tie); neural - real_other_seed -0.0001 [-0.0016, +0.0013] (tie).

### One minute of play

Statistics of each 10 s window of a 60 s neural run (shards of 6 s from start points, the default) against a real 10 s run at the same setting (two held-out settings): if the effect drifted, the distances would grow window by window.

| effect | setting | window | spectrum L1 | motion ratio | coverage L1 | mean-frame PSNR |
|---|---:|---:|---:|---:|---:|---:|
| jet | 0 | 0 | 0.1863 | 0.9187 | 0.0106 | 25.564 |
| jet | 0 | 1 | 0.1899 | 0.9218 | 0.0097 | 29.498 |
| jet | 0 | 2 | 0.1984 | 0.8790 | 0.0076 | 27.928 |
| jet | 0 | 3 | 0.1908 | 0.9585 | 0.0093 | 27.136 |
| jet | 0 | 4 | 0.1927 | 0.8496 | 0.0190 | 29.041 |
| jet | 0 | 5 | 0.1938 | 0.8675 | 0.0170 | 28.388 |
| jet | 1 | 0 | 0.1010 | 0.8866 | 0.0061 | 26.582 |
| jet | 1 | 1 | 0.0994 | 0.7763 | 0.0106 | 28.627 |
| jet | 1 | 2 | 0.0968 | 0.7704 | 0.0140 | 27.200 |
| jet | 1 | 3 | 0.0956 | 0.8577 | 0.0103 | 27.659 |
| jet | 1 | 4 | 0.1095 | 0.9236 | 0.0080 | 26.954 |
| jet | 1 | 5 | 0.1000 | 0.8575 | 0.0100 | 28.602 |
| orb | 0 | 0 | 0.0920 | 0.9003 | 0.0050 | 30.417 |
| orb | 0 | 1 | 0.1076 | 0.8445 | 0.0058 | 29.470 |
| orb | 0 | 2 | 0.1047 | 1.0905 | 0.0098 | 24.234 |
| orb | 0 | 3 | 0.1429 | 0.9226 | 0.0093 | 24.995 |
| orb | 0 | 4 | 0.1404 | 0.7985 | 0.0043 | 28.937 |
| orb | 0 | 5 | 0.0850 | 1.0029 | 0.0170 | 22.259 |
| orb | 1 | 0 | 0.3181 | 0.9853 | 0.0031 | 29.706 |
| orb | 1 | 1 | 0.3009 | 1.1018 | 0.0061 | 26.520 |
| orb | 1 | 2 | 0.2861 | 1.0543 | 0.0042 | 27.155 |
| orb | 1 | 3 | 0.3064 | 1.0476 | 0.0047 | 27.847 |
| orb | 1 | 4 | 0.3122 | 1.0962 | 0.0039 | 28.418 |
| orb | 1 | 5 | 0.2595 | 1.0840 | 0.0051 | 28.906 |
