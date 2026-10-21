# hold minus coarse (val; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Tracking from the true state (active PSNR, dB; paired over runs):

| measure | horizon | hold | coarse | difference |
|---|---:|---:|---:|---|
| field: temperature | 8 frames | 18.12 | 19.35 | -1.23 [-1.64, -0.82] (worse) |
| field: temperature | 1 s | 17.23 | 17.54 | -0.31 [-0.77, +0.14] (tie) |
| field: temperature | 2 s | 17.94 | 17.61 | +0.33 [-0.30, +0.98] (tie) |
| field: soot | 8 frames | 17.61 | 18.84 | -1.23 [-1.64, -0.82] (worse) |
| field: soot | 1 s | 16.72 | 17.03 | -0.31 [-0.77, +0.15] (tie) |
| field: soot | 2 s | 17.44 | 17.11 | +0.33 [-0.30, +0.99] (tie) |
| picture (4 cameras) | 8 frames | 14.27 | 21.23 | -6.96 [-7.36, -6.57] (worse) |
| picture (4 cameras) | 1 s | 10.31 | 16.29 | -5.98 [-6.49, -5.51] (worse) |
| picture (4 cameras) | 2 s | 10.71 | 14.66 | -3.95 [-4.81, -3.12] (worse) |

(15 runs)

