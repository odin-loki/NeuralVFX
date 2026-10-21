# C3 minus coarse (val; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Tracking from the true state (active PSNR, dB; paired over runs):

| measure | horizon | C3 | coarse | difference |
|---|---:|---:|---:|---|
| field: temperature | 8 frames | 20.30 | 19.35 | +0.96 [+0.60, +1.30] (better) |
| field: temperature | 1 s | 18.13 | 17.54 | +0.59 [+0.14, +1.04] (better) |
| field: temperature | 2 s | 18.25 | 17.61 | +0.65 [+0.01, +1.26] (better) |
| field: soot | 8 frames | 19.74 | 18.84 | +0.90 [+0.53, +1.24] (better) |
| field: soot | 1 s | 17.51 | 17.03 | +0.47 [-0.01, +0.95] (tie) |
| field: soot | 2 s | 17.65 | 17.11 | +0.54 [-0.13, +1.19] (tie) |
| picture (4 cameras) | 8 frames | 21.74 | 21.23 | +0.51 [+0.32, +0.68] (better) |
| picture (4 cameras) | 1 s | 17.63 | 16.29 | +1.34 [+0.91, +1.78] (better) |
| picture (4 cameras) | 2 s | 15.26 | 14.66 | +0.60 [+0.11, +1.10] (better) |

(15 runs)

Endless statistics against real runs (5 settings, 2 runs pooled per setting; lower is better but for PSNR and the motion ratio, whose ideal is 1):

| statistic | C3 | coarse | difference |
|---|---:|---:|---|
| 3D spectrum distance | 0.535 | 0.710 | -0.175 [-0.266, -0.093] (better) |
| \|ln motion ratio\| | 0.565 | 0.605 | -0.040 [-0.118, +0.061] (tie) |
| motion ratio | 0.601 | 0.564 | +0.037 [-0.015, +0.082] (context) |
| coverage distance | 0.0240 | 0.0130 | +0.0110 [-0.0009, +0.0230] (tie) |
| light distance | 0.0214 | 0.0121 | +0.0093 [+0.0001, +0.0185] (worse) |
| mean picture PSNR (dB) | 22.66 | 23.57 | -0.91 [-1.77, -0.06] (worse) |

Per setting (C3 / coarse):

| setting (intensity, wind, turbulence) | motion ratio | coverage distance | light distance | 3D spectrum distance | mean picture PSNR (dB) |
|---|---|---|---|---|---|
| 0.55, 0.10, 0.45 | 0.31 / 0.36 | 0.0254 / 0.0202 | 0.0284 / 0.0188 | 0.531 / 0.629 | 22.28 / 21.91 |
| 0.35, 0.85, 0.60 | 0.62 / 0.55 | 0.0116 / 0.0139 | 0.0149 / 0.0137 | 0.544 / 0.601 | 23.25 / 25.18 |
| 0.70, 0.35, 0.75 | 0.57 / 0.58 | 0.0372 / 0.0064 | 0.0262 / 0.0051 | 0.530 / 0.764 | 21.26 / 23.38 |
| 0.45, 0.55, 0.30 | 0.91 / 0.80 | 0.0078 / 0.0104 | 0.0091 / 0.0150 | 0.545 / 0.698 | 24.43 / 24.58 |
| 0.80, 0.70, 0.55 | 0.59 / 0.54 | 0.0382 / 0.0140 | 0.0283 / 0.0080 | 0.524 / 0.857 | 22.06 / 22.79 |
