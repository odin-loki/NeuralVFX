# C3 minus hold (val; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Tracking from the true state (active PSNR, dB; paired over runs):

| measure | horizon | C3 | hold | difference |
|---|---:|---:|---:|---|
| field: temperature | 8 frames | 20.30 | 18.12 | +2.19 [+2.01, +2.37] (better) |
| field: temperature | 1 s | 18.13 | 17.23 | +0.90 [+0.52, +1.27] (better) |
| field: temperature | 2 s | 18.25 | 17.94 | +0.32 [-0.20, +0.80] (tie) |
| field: soot | 8 frames | 19.74 | 17.61 | +2.12 [+1.94, +2.31] (better) |
| field: soot | 1 s | 17.51 | 16.72 | +0.78 [+0.38, +1.16] (better) |
| field: soot | 2 s | 17.65 | 17.44 | +0.21 [-0.33, +0.70] (tie) |
| picture (4 cameras) | 8 frames | 21.74 | 14.27 | +7.47 [+7.15, +7.79] (better) |
| picture (4 cameras) | 1 s | 17.63 | 10.31 | +7.32 [+6.96, +7.70] (better) |
| picture (4 cameras) | 2 s | 15.26 | 10.71 | +4.55 [+3.77, +5.29] (better) |

(15 runs)

