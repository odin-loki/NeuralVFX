# C2 minus coarse (val; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Tracking from the true state (active PSNR, dB; paired over runs):

| measure | horizon | C2 | coarse | difference |
|---|---:|---:|---:|---|
| field: temperature | 8 frames | 20.41 | 19.35 | +1.06 [+0.71, +1.41] (better) |
| field: temperature | 1 s | 18.22 | 17.54 | +0.68 [+0.21, +1.13] (better) |
| field: temperature | 2 s | 18.29 | 17.61 | +0.68 [+0.01, +1.31] (better) |
| field: soot | 8 frames | 19.87 | 18.84 | +1.02 [+0.67, +1.37] (better) |
| field: soot | 1 s | 17.62 | 17.03 | +0.58 [+0.09, +1.06] (better) |
| field: soot | 2 s | 17.71 | 17.11 | +0.59 [-0.11, +1.25] (tie) |
| picture (4 cameras) | 8 frames | 21.67 | 21.23 | +0.44 [+0.22, +0.61] (better) |
| picture (4 cameras) | 1 s | 17.65 | 16.29 | +1.36 [+0.94, +1.81] (better) |
| picture (4 cameras) | 2 s | 15.18 | 14.66 | +0.52 [+0.06, +0.98] (better) |

(15 runs)

Endless statistics against real runs (5 settings, 2 runs pooled per setting; lower is better but for PSNR and the motion ratio, whose ideal is 1):

| statistic | C2 | coarse | difference |
|---|---:|---:|---|
| 3D spectrum distance | 0.511 | 0.710 | -0.199 [-0.289, -0.119] (better) |
| \|ln motion ratio\| | 0.674 | 0.605 | +0.069 [+0.009, +0.130] (worse) |
| motion ratio | 0.534 | 0.564 | -0.030 [-0.054, -0.006] (context) |
| coverage distance | 0.0291 | 0.0130 | +0.0162 [+0.0035, +0.0298] (worse) |
| light distance | 0.0247 | 0.0121 | +0.0126 [+0.0020, +0.0239] (worse) |
| mean picture PSNR (dB) | 22.03 | 23.57 | -1.54 [-2.41, -0.56] (worse) |

Per setting (C2 / coarse):

| setting (intensity, wind, turbulence) | motion ratio | coverage distance | light distance | 3D spectrum distance | mean picture PSNR (dB) |
|---|---|---|---|---|---|
| 0.55, 0.10, 0.45 | 0.30 / 0.36 | 0.0275 / 0.0202 | 0.0292 / 0.0188 | 0.510 / 0.629 | 22.21 / 21.91 |
| 0.35, 0.85, 0.60 | 0.56 / 0.55 | 0.0150 / 0.0139 | 0.0183 / 0.0137 | 0.518 / 0.601 | 23.98 / 25.18 |
| 0.70, 0.35, 0.75 | 0.55 / 0.58 | 0.0366 / 0.0064 | 0.0266 / 0.0051 | 0.521 / 0.764 | 20.85 / 23.38 |
| 0.45, 0.55, 0.30 | 0.79 / 0.80 | 0.0144 / 0.0104 | 0.0103 / 0.0150 | 0.507 / 0.698 | 23.04 / 24.58 |
| 0.80, 0.70, 0.55 | 0.47 / 0.54 | 0.0522 / 0.0140 | 0.0392 / 0.0080 | 0.499 / 0.857 | 20.05 / 22.79 |
