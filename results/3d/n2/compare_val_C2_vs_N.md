# C2 minus N (val; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Tracking from the true state (active PSNR, dB; paired over runs):

| measure | horizon | C2 | N | difference |
|---|---:|---:|---:|---|
| field: temperature | 8 frames | 22.11 | 22.08 | +0.03 [-0.04, +0.11] (tie) |
| field: temperature | 1 s | 19.22 | 19.41 | -0.19 [-0.51, +0.17] (tie) |
| field: temperature | 2 s | 18.90 | 19.09 | -0.19 [-0.50, +0.12] (tie) |
| field: soot | 8 frames | 27.09 | 26.99 | +0.10 [+0.01, +0.19] (better) |
| field: soot | 1 s | 24.55 | 24.49 | +0.07 [-0.26, +0.46] (tie) |
| field: soot | 2 s | 24.30 | 24.19 | +0.11 [-0.23, +0.46] (tie) |
| picture (4 cameras) | 8 frames | 21.34 | 21.30 | +0.05 [-0.14, +0.23] (tie) |
| picture (4 cameras) | 1 s | 17.74 | 17.81 | -0.07 [-0.58, +0.44] (tie) |
| picture (4 cameras) | 2 s | 18.03 | 18.03 | -0.00 [-0.44, +0.47] (tie) |

(15 runs)

Endless statistics against real runs (5 settings, 2 runs pooled per setting; lower is better but for PSNR and the motion ratio, whose ideal is 1):

| statistic | C2 | N | difference |
|---|---:|---:|---|
| 3D spectrum distance | 0.316 | 0.774 | -0.458 [-0.905, -0.044] (better) |
| \|ln motion ratio\| | 0.424 | 0.440 | -0.016 [-0.568, +0.536] (tie) |
| motion ratio | 0.721 | 1.588 | -0.867 [-1.838, -0.115] (context) |
| coverage distance | 0.0044 | 0.0285 | -0.0241 [-0.0554, -0.0019] (better) |
| emission distance | 0.0045 | 0.0271 | -0.0226 [-0.0518, -0.0025] (better) |
| mean picture PSNR (dB) | 28.84 | 24.15 | +4.69 [-0.26, +11.01] (tie) |

Per setting (C2 / N):

| setting (intensity, wind, turbulence) | motion ratio | coverage distance | emission distance | 3D spectrum distance | mean picture PSNR (dB) |
|---|---|---|---|---|---|
| 0.60, 0.08, 0.50 | 0.81 / 3.46 | 0.0034 / 0.0832 | 0.0048 / 0.0801 | 0.278 / 1.502 | 29.99 / 14.35 |
| 0.40, 0.92, 0.65 | 1.08 / 0.74 | 0.0008 / 0.0012 | 0.0045 / 0.0025 | 0.417 / 0.366 | 29.15 / 32.29 |
| 0.75, 0.30, 0.35 | 0.36 / 1.15 | 0.0116 / 0.0463 | 0.0067 / 0.0350 | 0.302 / 1.130 | 25.72 / 19.48 |
| 0.50, 0.50, 0.80 | 0.55 / 0.97 | 0.0049 / 0.0076 | 0.0042 / 0.0091 | 0.215 / 0.244 | 29.47 / 27.61 |
| 0.30, 0.65, 0.25 | 0.81 / 1.62 | 0.0012 / 0.0042 | 0.0024 / 0.0091 | 0.366 / 0.628 | 29.88 / 27.05 |
