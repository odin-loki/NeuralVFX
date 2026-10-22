# A minus hold (val; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Tracking from the true state (active PSNR, dB; paired over runs):

| measure | horizon | A | hold | difference |
|---|---:|---:|---:|---|
| field: temperature | 8 frames | 22.54 | 21.22 | +1.31 [+0.50, +2.05] (better) |
| field: temperature | 1 s | 19.57 | 20.38 | -0.81 [-1.82, +0.08] (tie) |
| field: temperature | 2 s | 18.62 | 19.74 | -1.12 [-2.02, -0.25] (worse) |
| field: soot | 8 frames | 27.38 | 25.46 | +1.92 [+1.30, +2.50] (better) |
| field: soot | 1 s | 25.03 | 24.86 | +0.17 [-0.52, +0.79] (tie) |
| field: soot | 2 s | 24.14 | 24.69 | -0.55 [-1.41, +0.22] (tie) |
| picture (4 cameras) | 8 frames | 21.32 | 19.65 | +1.66 [+0.73, +2.46] (better) |
| picture (4 cameras) | 1 s | 17.37 | 18.06 | -0.69 [-2.10, +0.55] (tie) |
| picture (4 cameras) | 2 s | 17.20 | 18.00 | -0.81 [-2.36, +0.47] (tie) |

(12 runs)

