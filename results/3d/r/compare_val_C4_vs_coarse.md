# C4 minus coarse (val; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Tracking from the true state (active PSNR, dB; paired over runs):

| measure | horizon | C4 | coarse | difference |
|---|---:|---:|---:|---|
| field: temperature | 8 frames | 20.81 | 19.35 | +1.46 [+1.10, +1.80] (better) |
| field: temperature | 1 s | 19.30 | 17.54 | +1.76 [+1.40, +2.08] (better) |
| field: temperature | 2 s | 20.42 | 17.61 | +2.81 [+2.15, +3.51] (better) |
| field: soot | 8 frames | 20.32 | 18.84 | +1.48 [+1.12, +1.82] (better) |
| field: soot | 1 s | 18.82 | 17.03 | +1.78 [+1.42, +2.11] (better) |
| field: soot | 2 s | 19.93 | 17.11 | +2.82 [+2.16, +3.52] (better) |
| picture (4 cameras) | 8 frames | 22.63 | 21.23 | +1.40 [+0.89, +1.91] (better) |
| picture (4 cameras) | 1 s | 17.83 | 16.29 | +1.54 [+0.96, +2.15] (better) |
| picture (4 cameras) | 2 s | 16.15 | 14.66 | +1.49 [+1.00, +1.99] (better) |

(15 runs)

Endless statistics against real runs (5 settings, 2 runs pooled per setting; lower is better but for PSNR and the motion ratio, whose ideal is 1):

| statistic | C4 | coarse | difference |
|---|---:|---:|---|
| 3D spectrum distance | 0.512 | 0.710 | -0.198 [-0.278, -0.123] (better) |
| \|ln motion ratio\| | 0.531 | 0.605 | -0.074 [-0.311, +0.119] (tie) |
| motion ratio | 0.597 | 0.564 | +0.033 [-0.067, +0.147] (context) |
| coverage distance | 0.0167 | 0.0130 | +0.0037 [+0.0002, +0.0072] (worse) |
| light distance | 0.0304 | 0.0121 | +0.0183 [+0.0090, +0.0275] (worse) |
| mean picture PSNR (dB) | 22.80 | 23.57 | -0.77 [-3.47, +1.94] (tie) |

Per setting (C4 / coarse):

| setting (intensity, wind, turbulence) | motion ratio | coverage distance | light distance | 3D spectrum distance | mean picture PSNR (dB) |
|---|---|---|---|---|---|
| 0.55, 0.10, 0.45 | 0.57 / 0.36 | 0.0176 / 0.0202 | 0.0217 / 0.0188 | 0.517 / 0.629 | 24.01 / 21.91 |
| 0.35, 0.85, 0.60 | 0.69 / 0.55 | 0.0165 / 0.0139 | 0.0423 / 0.0137 | 0.510 / 0.601 | 21.15 / 25.18 |
| 0.70, 0.35, 0.75 | 0.49 / 0.58 | 0.0109 / 0.0064 | 0.0281 / 0.0051 | 0.527 / 0.764 | 23.34 / 23.38 |
| 0.45, 0.55, 0.30 | 0.75 / 0.80 | 0.0148 / 0.0104 | 0.0436 / 0.0150 | 0.487 / 0.698 | 19.96 / 24.58 |
| 0.80, 0.70, 0.55 | 0.49 / 0.54 | 0.0237 / 0.0140 | 0.0161 / 0.0080 | 0.517 / 0.857 | 25.54 / 22.79 |
