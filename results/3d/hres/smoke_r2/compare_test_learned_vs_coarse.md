# learned minus coarse (test; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Tracking from the true state (active PSNR, dB; paired over runs):

| measure | horizon | learned | coarse | difference |
|---|---:|---:|---:|---|
| field: temperature | 8 frames | 26.34 | 25.70 | +0.64 [+0.42, +0.86] (better) |
| field: temperature | 1 s | 22.79 | 22.70 | +0.09 [-0.36, +0.55] (tie) |
| field: temperature | 2 s | 23.11 | 23.28 | -0.18 [-0.46, +0.09] (tie) |
| field: soot | 8 frames | 25.83 | 25.21 | +0.62 [+0.40, +0.83] (better) |
| field: soot | 1 s | 22.28 | 22.22 | +0.06 [-0.38, +0.52] (tie) |
| field: soot | 2 s | 22.60 | 22.82 | -0.22 [-0.51, +0.06] (tie) |
| picture (4 cameras) | 8 frames | 26.96 | 27.32 | -0.36 [-0.60, -0.13] (worse) |
| picture (4 cameras) | 1 s | 19.94 | 21.00 | -1.06 [-1.45, -0.61] (worse) |
| picture (4 cameras) | 2 s | 17.19 | 18.35 | -1.16 [-1.65, -0.70] (worse) |

(18 runs)

Endless statistics against real runs (6 settings, 3 runs pooled per setting; lower is better but for PSNR and the motion ratio, whose ideal is 1):

| statistic | learned | coarse | floor (other real runs) | difference |
|---|---:|---:|---:|---|
| 3D spectrum distance | 0.394 | 0.325 | 0.044 | +0.069 [+0.023, +0.141] (worse) |
| \|ln motion ratio\| | 0.606 | 0.299 | 0.054 | +0.307 [+0.149, +0.532] (worse) |
| motion ratio | 0.563 | 0.745 | 0.973 | -0.182 [-0.273, -0.102] (context) |
| coverage distance | 0.0343 | 0.0223 | 0.0079 | +0.0120 [+0.0054, +0.0197] (worse) |
| emission distance | 0.0139 | 0.0144 | 0.0023 | -0.0004 [-0.0022, +0.0014] (tie) |
| mean picture PSNR (dB) | 21.67 | 23.43 | 32.57 | -1.76 [-2.53, -1.10] (worse) |

Per setting (learned / coarse):

| setting (intensity, wind, turbulence) | motion ratio | coverage distance | emission distance | 3D spectrum distance | mean picture PSNR (dB) |
|---|---|---|---|---|---|
| 0.68, 0.07, 0.42 | 0.59 / 0.62 | 0.0269 / 0.0220 | 0.0043 / 0.0051 | 0.366 / 0.315 | 23.21 / 24.72 |
| 0.52, 0.93, 0.52 | 0.65 / 0.77 | 0.0321 / 0.0300 | 0.0116 / 0.0145 | 0.333 / 0.300 | 22.23 / 23.44 |
| 0.42, 0.22, 0.68 | 0.30 / 0.68 | 0.0545 / 0.0271 | 0.0105 / 0.0112 | 0.561 / 0.319 | 19.40 / 22.77 |
| 0.78, 0.48, 0.32 | 0.62 / 0.79 | 0.0291 / 0.0153 | 0.0247 / 0.0235 | 0.362 / 0.311 | 20.52 / 22.05 |
| 0.33, 0.72, 0.42 | 0.63 / 0.82 | 0.0334 / 0.0288 | 0.0126 / 0.0155 | 0.366 / 0.368 | 23.50 / 24.06 |
| 0.63, 0.38, 0.88 | 0.60 / 0.80 | 0.0300 / 0.0105 | 0.0200 / 0.0164 | 0.378 / 0.337 | 21.17 / 23.53 |
