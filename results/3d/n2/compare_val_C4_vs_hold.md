# C4 minus hold (val; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Tracking from the true state (active PSNR, dB; paired over runs):

| measure | horizon | C4 | hold | difference |
|---|---:|---:|---:|---|
| field: temperature | 8 frames | 22.14 | 20.84 | +1.30 [+0.76, +1.81] (better) |
| field: temperature | 1 s | 20.56 | 20.36 | +0.21 [-0.37, +0.77] (tie) |
| field: temperature | 2 s | 20.89 | 20.15 | +0.75 [+0.10, +1.33] (better) |
| field: soot | 8 frames | 26.87 | 25.33 | +1.53 [+1.10, +1.96] (better) |
| field: soot | 1 s | 25.54 | 25.00 | +0.54 [+0.06, +1.02] (better) |
| field: soot | 2 s | 25.97 | 24.95 | +1.02 [+0.46, +1.56] (better) |
| picture (4 cameras) | 8 frames | 21.47 | 19.94 | +1.53 [+0.85, +2.15] (better) |
| picture (4 cameras) | 1 s | 19.21 | 18.54 | +0.67 [-0.07, +1.38] (tie) |
| picture (4 cameras) | 2 s | 19.62 | 18.94 | +0.68 [-0.23, +1.40] (tie) |

(15 runs)

