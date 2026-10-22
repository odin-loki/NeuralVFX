# C3 minus N (val; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Tracking from the true state (active PSNR, dB; paired over runs):

| measure | horizon | C3 | N | difference |
|---|---:|---:|---:|---|
| field: temperature | 8 frames | 21.87 | 22.08 | -0.21 [-0.31, -0.12] (worse) |
| field: temperature | 1 s | 18.88 | 19.41 | -0.52 [-0.90, -0.12] (worse) |
| field: temperature | 2 s | 18.76 | 19.09 | -0.33 [-0.62, -0.04] (worse) |
| field: soot | 8 frames | 26.90 | 26.99 | -0.09 [-0.19, +0.01] (tie) |
| field: soot | 1 s | 24.27 | 24.49 | -0.22 [-0.59, +0.21] (tie) |
| field: soot | 2 s | 24.25 | 24.19 | +0.05 [-0.26, +0.40] (tie) |
| picture (4 cameras) | 8 frames | 20.89 | 21.30 | -0.40 [-0.68, -0.13] (worse) |
| picture (4 cameras) | 1 s | 17.28 | 17.81 | -0.53 [-1.13, +0.10] (tie) |
| picture (4 cameras) | 2 s | 17.86 | 18.03 | -0.17 [-0.74, +0.41] (tie) |

(15 runs)

Endless statistics against real runs (5 settings, 2 runs pooled per setting; lower is better but for PSNR and the motion ratio, whose ideal is 1):

| statistic | C3 | N | difference |
|---|---:|---:|---|
| 3D spectrum distance | 0.301 | 0.774 | -0.473 [-0.926, -0.046] (better) |
| \|ln motion ratio\| | 0.485 | 0.440 | +0.045 [-0.489, +0.581] (tie) |
| motion ratio | 0.652 | 1.588 | -0.936 [-1.935, -0.116] (context) |
| coverage distance | 0.0046 | 0.0285 | -0.0239 [-0.0549, -0.0015] (better) |
| emission distance | 0.0043 | 0.0271 | -0.0229 [-0.0523, -0.0031] (better) |
| mean picture PSNR (dB) | 28.91 | 24.15 | +4.75 [-0.70, +10.83] (tie) |

Per setting (C3 / N):

| setting (intensity, wind, turbulence) | motion ratio | coverage distance | emission distance | 3D spectrum distance | mean picture PSNR (dB) |
|---|---|---|---|---|---|
| 0.60, 0.08, 0.50 | 0.68 / 3.46 | 0.0039 / 0.0832 | 0.0041 / 0.0801 | 0.241 / 1.502 | 29.69 / 14.35 |
| 0.40, 0.92, 0.65 | 0.99 / 0.74 | 0.0013 / 0.0012 | 0.0036 / 0.0025 | 0.432 / 0.366 | 28.17 / 32.29 |
| 0.75, 0.30, 0.35 | 0.33 / 1.15 | 0.0119 / 0.0463 | 0.0072 / 0.0350 | 0.300 / 1.130 | 25.92 / 19.48 |
| 0.50, 0.50, 0.80 | 0.65 / 0.97 | 0.0041 / 0.0076 | 0.0032 / 0.0091 | 0.224 / 0.244 | 29.27 / 27.61 |
| 0.30, 0.65, 0.25 | 0.62 / 1.62 | 0.0020 / 0.0042 | 0.0032 / 0.0091 | 0.307 / 0.628 | 31.48 / 27.05 |
