# B minus hold (val; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Tracking from the true state (active PSNR, dB; paired over runs):

| measure | horizon | B | hold | difference |
|---|---:|---:|---:|---|
| field: temperature | 8 frames | 22.26 | 21.22 | +1.03 [+0.26, +1.72] (better) |
| field: temperature | 1 s | 19.31 | 20.38 | -1.07 [-2.02, -0.23] (worse) |
| field: temperature | 2 s | 18.68 | 19.74 | -1.05 [-1.84, -0.27] (worse) |
| field: soot | 8 frames | 27.20 | 25.46 | +1.75 [+1.16, +2.28] (better) |
| field: soot | 1 s | 24.84 | 24.86 | -0.02 [-0.69, +0.58] (tie) |
| field: soot | 2 s | 24.21 | 24.69 | -0.48 [-1.22, +0.24] (tie) |
| picture (4 cameras) | 8 frames | 20.81 | 19.65 | +1.16 [+0.30, +1.90] (better) |
| picture (4 cameras) | 1 s | 17.08 | 18.06 | -0.98 [-2.38, +0.21] (tie) |
| picture (4 cameras) | 2 s | 17.51 | 18.00 | -0.50 [-1.69, +0.56] (tie) |

(12 runs)

