# hold minus coarse (test; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Tracking from the true state (active PSNR, dB; paired over runs):

| measure | horizon | hold | coarse | difference |
|---|---:|---:|---:|---|
| field: temperature | 8 frames | 18.75 | 19.80 | -1.05 [-1.41, -0.69] (worse) |
| field: temperature | 1 s | 17.78 | 18.17 | -0.39 [-0.92, +0.18] (tie) |
| field: temperature | 2 s | 18.33 | 17.74 | +0.59 [+0.07, +1.10] (better) |
| field: soot | 8 frames | 18.25 | 19.30 | -1.05 [-1.41, -0.69] (worse) |
| field: soot | 1 s | 17.29 | 17.67 | -0.39 [-0.91, +0.19] (tie) |
| field: soot | 2 s | 17.85 | 17.27 | +0.58 [+0.07, +1.10] (better) |
| picture (4 cameras) | 8 frames | 14.56 | 21.13 | -6.56 [-6.97, -6.18] (worse) |
| picture (4 cameras) | 1 s | 10.46 | 16.28 | -5.82 [-6.26, -5.37] (worse) |
| picture (4 cameras) | 2 s | 10.75 | 14.78 | -4.03 [-4.65, -3.36] (worse) |

(18 runs)

