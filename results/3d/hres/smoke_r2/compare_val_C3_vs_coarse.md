# C3 minus coarse (val; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Tracking from the true state (active PSNR, dB; paired over runs):

| measure | horizon | C3 | coarse | difference |
|---|---:|---:|---:|---|
| field: temperature | 8 frames | 27.56 | 27.01 | +0.54 [+0.37, +0.71] (better) |
| field: temperature | 1 s | 23.75 | 23.86 | -0.11 [-0.44, +0.21] (tie) |
| field: temperature | 2 s | 22.96 | 23.54 | -0.58 [-0.96, -0.18] (worse) |
| field: soot | 8 frames | 27.10 | 26.59 | +0.51 [+0.34, +0.67] (better) |
| field: soot | 1 s | 23.25 | 23.45 | -0.20 [-0.54, +0.13] (tie) |
| field: soot | 2 s | 22.49 | 23.14 | -0.65 [-1.04, -0.22] (worse) |
| picture (4 cameras) | 8 frames | 27.20 | 27.77 | -0.57 [-0.95, -0.19] (worse) |
| picture (4 cameras) | 1 s | 20.04 | 21.17 | -1.13 [-1.65, -0.66] (worse) |
| picture (4 cameras) | 2 s | 17.28 | 18.84 | -1.56 [-2.18, -0.95] (worse) |

(15 runs)

Endless statistics against real runs (5 settings, 2 runs pooled per setting; lower is better but for PSNR and the motion ratio, whose ideal is 1):

| statistic | C3 | coarse | difference |
|---|---:|---:|---|
| 3D spectrum distance | 0.419 | 0.328 | +0.091 [+0.036, +0.150] (worse) |
| \|ln motion ratio\| | 0.485 | 0.251 | +0.234 [+0.165, +0.305] (worse) |
| motion ratio | 0.618 | 0.780 | -0.162 [-0.210, -0.116] (context) |
| coverage distance | 0.0312 | 0.0212 | +0.0099 [+0.0021, +0.0202] (worse) |
| emission distance | 0.0151 | 0.0131 | +0.0019 [-0.0008, +0.0048] (tie) |
| mean picture PSNR (dB) | 21.49 | 23.72 | -2.22 [-3.46, -1.00] (worse) |

Per setting (C3 / coarse):

| setting (intensity, wind, turbulence) | motion ratio | coverage distance | emission distance | 3D spectrum distance | mean picture PSNR (dB) |
|---|---|---|---|---|---|
| 0.62, 0.12, 0.48 | 0.63 / 0.70 | 0.0232 / 0.0161 | 0.0082 / 0.0055 | 0.386 / 0.316 | 23.62 / 25.92 |
| 0.38, 0.88, 0.62 | 0.59 / 0.74 | 0.0380 / 0.0323 | 0.0130 / 0.0156 | 0.500 / 0.322 | 21.02 / 22.40 |
| 0.72, 0.28, 0.38 | 0.60 / 0.81 | 0.0487 / 0.0193 | 0.0165 / 0.0095 | 0.400 / 0.337 | 19.53 / 24.07 |
| 0.48, 0.52, 0.78 | 0.71 / 0.86 | 0.0201 / 0.0221 | 0.0219 / 0.0204 | 0.324 / 0.331 | 22.29 / 22.43 |
| 0.28, 0.68, 0.22 | 0.56 / 0.79 | 0.0258 / 0.0162 | 0.0157 / 0.0148 | 0.483 / 0.332 | 21.00 / 23.75 |
