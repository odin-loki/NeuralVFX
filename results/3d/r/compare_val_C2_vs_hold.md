# C2 minus hold (val; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Tracking from the true state (active PSNR, dB; paired over runs):

| measure | horizon | C2 | hold | difference |
|---|---:|---:|---:|---|
| field: temperature | 8 frames | 20.41 | 18.12 | +2.30 [+2.12, +2.47] (better) |
| field: temperature | 1 s | 18.22 | 17.23 | +0.99 [+0.59, +1.39] (better) |
| field: temperature | 2 s | 18.29 | 17.94 | +0.35 [-0.20, +0.87] (tie) |
| field: soot | 8 frames | 19.87 | 17.61 | +2.25 [+2.07, +2.44] (better) |
| field: soot | 1 s | 17.62 | 16.72 | +0.89 [+0.47, +1.30] (better) |
| field: soot | 2 s | 17.71 | 17.44 | +0.26 [-0.31, +0.80] (tie) |
| picture (4 cameras) | 8 frames | 21.67 | 14.27 | +7.40 [+7.09, +7.70] (better) |
| picture (4 cameras) | 1 s | 17.65 | 10.31 | +7.34 [+6.96, +7.74] (better) |
| picture (4 cameras) | 2 s | 15.18 | 10.71 | +4.47 [+3.75, +5.15] (better) |

(15 runs)

