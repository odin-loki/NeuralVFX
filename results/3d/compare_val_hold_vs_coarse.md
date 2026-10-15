# hold minus coarse (val; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Tracking from the true state (active PSNR, dB; paired over runs):

| measure | horizon | hold | coarse | difference |
|---|---:|---:|---:|---|
| field: temperature | 8 frames | 21.22 | 20.41 | +0.81 [-0.04, +1.80] (tie) |
| field: temperature | 1 s | 20.38 | 18.31 | +2.07 [+1.20, +2.90] (better) |
| field: temperature | 2 s | 19.74 | 16.85 | +2.89 [+1.88, +4.08] (better) |
| field: soot | 8 frames | 25.46 | 24.54 | +0.91 [+0.01, +1.97] (better) |
| field: soot | 1 s | 24.86 | 19.56 | +5.29 [+4.66, +5.88] (better) |
| field: soot | 2 s | 24.69 | 17.57 | +7.12 [+6.15, +8.20] (better) |
| picture (4 cameras) | 8 frames | 19.65 | 18.45 | +1.21 [+0.32, +2.28] (better) |
| picture (4 cameras) | 1 s | 18.06 | 14.78 | +3.28 [+2.10, +4.43] (better) |
| picture (4 cameras) | 2 s | 18.00 | 14.67 | +3.34 [+2.30, +4.53] (better) |

(12 runs)

