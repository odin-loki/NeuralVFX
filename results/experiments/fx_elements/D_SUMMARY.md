
## D: start points and learned dynamics

### Chaos horizon (the simulation itself)

Active PSNR against a run of the simulation from a stored start point, by frames after it (30 per second), mean over runs. Perturbations are relative to the velocity RMS.

| effect | case | 1 | 4 | 8 | 15 | 30 | 60 | 120 | 240 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| haboob | start+1e-3_same_seed | 99.00 | 94.61 | 91.49 | 86.44 | 82.41 | 79.37 | 75.20 | - |
| haboob | start+1e-1_same_seed | 75.20 | 69.27 | 66.54 | 65.07 | 61.70 | 58.47 | 58.09 | - |
| haboob | same_start_other_seed | 54.40 | 43.46 | 36.16 | 28.94 | 20.99 | 15.12 | 15.08 | - |
| haboob | other_start_same_seed | 24.93 | 25.55 | 26.36 | 27.57 | 29.16 | 28.88 | 26.19 | - |

### Training

Stepper loss: stages 1-2 (normalised squared error plus profiles) and stage 3 (plus activity). Renderer PSNR on its training samples (true fields). Detail constants as calibrated.

| effect | runs x frames | minutes of simulation | stepper loss 1-2 | stepper s | stage 3 loss | stage 3 s | renderer dB | contrast | swirl | grow | start points | KB stored |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| geyser | 160 x 240 | 21.33 | 0.11420 | 11043 | - | - | 0.00 | 1 | 0.8 | 1 | 8 | 145.7 |
| haboob | 240 x 150 | 20.00 | 0.02115 | 10862 | 0.02193 | 2135 | 31.59 | 0.5 | 0 | 1.5 | 16 | 274.2 |
| seaspray | 160 x 240 | 21.33 | 0.10753 | 11016 | - | - | 0.00 | 1 | 0.8 | 1 | 8 | 145.7 |

Renderer, detail constants and start points from `d-finish` on the trained steppers; stage 3 from `d-tune`.

### Tracking a held-out run from its true start point

Active PSNR against the true run, mean over held-out runs (other seeds and settings), by frames after the start point. `neural`: the rollout effect through the runtime, started from the true state (coarse and fine) with the run's noise seed. `coarse_sim_detail`: the simulation on the same 32-cell grid with the same detail layer, drawn by the simulation's own renderer. `neural_dynamics_true_renderer`: the neural dynamics drawn by that renderer too (separates dynamics from rendering). `renderer_on_true_fields`: the learned renderer on the true fields (frames 1-30, then their mean).

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

### Endless runs at held-out settings

The ten held-out settings of study B, a new seed each, 10 s per run (explosions: their 3 s). Distances of each method's frame statistics to a real run of the simulation at that setting; `real_other_seed` is a second real run with another seed, the floor. Lower is better except motion ratio (1 is right) and mean-frame PSNR (higher is closer).

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

### One minute of play

Statistics of each 10 s window of a 60 s neural run (shards of 6 s from start points, the default) against a real 10 s run at the same setting (two held-out settings): if the effect drifted, the distances would grow window by window.

| effect | setting | window | spectrum L1 | motion ratio | coverage L1 | mean-frame PSNR |
|---|---:|---:|---:|---:|---:|---:|
