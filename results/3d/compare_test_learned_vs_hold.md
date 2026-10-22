# learned minus hold (test; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Tracking from the true state (active PSNR, dB; paired over runs):

| measure | horizon | learned | hold | difference |
|---|---:|---:|---:|---|
| field: temperature | 8 frames | 22.57 | 21.01 | +1.56 [+1.15, +1.95] (better) |
| field: temperature | 1 s | 19.31 | 20.09 | -0.78 [-1.27, -0.29] (worse) |
| field: temperature | 2 s | 18.86 | 19.83 | -0.97 [-1.76, -0.20] (worse) |
| field: soot | 8 frames | 27.35 | 25.27 | +2.08 [+1.69, +2.45] (better) |
| field: soot | 1 s | 24.77 | 24.92 | -0.14 [-0.58, +0.31] (tie) |
| field: soot | 2 s | 24.22 | 24.71 | -0.49 [-1.15, +0.17] (tie) |
| picture (4 cameras) | 8 frames | 21.15 | 19.32 | +1.83 [+1.05, +2.61] (better) |
| picture (4 cameras) | 1 s | 16.98 | 17.68 | -0.71 [-1.67, +0.13] (tie) |
| picture (4 cameras) | 2 s | 17.00 | 18.04 | -1.04 [-2.06, -0.15] (worse) |

(18 runs)

