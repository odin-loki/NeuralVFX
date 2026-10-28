# C2 minus coarse (val; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Tracking from the true state (active PSNR, dB; paired over runs):

| measure | horizon | C2 | coarse | difference |
|---|---:|---:|---:|---|
| field: temperature | 8 frames | 27.76 | 27.01 | +0.75 [+0.60, +0.90] (better) |
| field: temperature | 1 s | 23.91 | 23.86 | +0.05 [-0.29, +0.39] (tie) |
| field: temperature | 2 s | 22.99 | 23.54 | -0.55 [-0.94, -0.16] (worse) |
| field: soot | 8 frames | 27.34 | 26.59 | +0.74 [+0.59, +0.89] (better) |
| field: soot | 1 s | 23.43 | 23.45 | -0.02 [-0.36, +0.34] (tie) |
| field: soot | 2 s | 22.53 | 23.14 | -0.60 [-1.01, -0.19] (worse) |
| picture (4 cameras) | 8 frames | 27.15 | 27.77 | -0.62 [-0.98, -0.26] (worse) |
| picture (4 cameras) | 1 s | 19.81 | 21.17 | -1.35 [-1.84, -0.90] (worse) |
| picture (4 cameras) | 2 s | 17.02 | 18.84 | -1.82 [-2.49, -1.18] (worse) |

(15 runs)

Endless statistics against real runs (5 settings, 2 runs pooled per setting; lower is better but for PSNR and the motion ratio, whose ideal is 1):

| statistic | C2 | coarse | difference |
|---|---:|---:|---|
| 3D spectrum distance | 0.456 | 0.328 | +0.128 [+0.037, +0.234] (worse) |
| \|ln motion ratio\| | 0.577 | 0.251 | +0.326 [+0.271, +0.387] (worse) |
| motion ratio | 0.564 | 0.780 | -0.216 [-0.253, -0.181] (context) |
| coverage distance | 0.0351 | 0.0212 | +0.0139 [+0.0048, +0.0232] (worse) |
| emission distance | 0.0129 | 0.0131 | -0.0002 [-0.0027, +0.0020] (tie) |
| mean picture PSNR (dB) | 21.24 | 23.72 | -2.47 [-4.09, -0.86] (worse) |

Per setting (C2 / coarse):

| setting (intensity, wind, turbulence) | motion ratio | coverage distance | emission distance | 3D spectrum distance | mean picture PSNR (dB) |
|---|---|---|---|---|---|
| 0.62, 0.12, 0.48 | 0.52 / 0.70 | 0.0297 / 0.0161 | 0.0065 / 0.0055 | 0.410 / 0.316 | 23.13 / 25.92 |
| 0.38, 0.88, 0.62 | 0.57 / 0.74 | 0.0375 / 0.0323 | 0.0112 / 0.0156 | 0.516 / 0.322 | 21.40 / 22.40 |
| 0.72, 0.28, 0.38 | 0.55 / 0.81 | 0.0494 / 0.0193 | 0.0129 / 0.0095 | 0.393 / 0.337 | 19.43 / 24.07 |
| 0.48, 0.52, 0.78 | 0.66 / 0.86 | 0.0222 / 0.0221 | 0.0204 / 0.0204 | 0.321 / 0.331 | 22.69 / 22.43 |
| 0.28, 0.68, 0.22 | 0.51 / 0.79 | 0.0369 / 0.0162 | 0.0137 / 0.0148 | 0.639 / 0.332 | 19.55 / 23.75 |
