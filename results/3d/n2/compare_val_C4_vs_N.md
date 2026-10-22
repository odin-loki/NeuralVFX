# C4 minus N (val; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Tracking from the true state (active PSNR, dB; paired over runs):

| measure | horizon | C4 | N | difference |
|---|---:|---:|---:|---|
| field: temperature | 8 frames | 22.14 | 22.08 | +0.05 [-0.21, +0.29] (tie) |
| field: temperature | 1 s | 20.56 | 19.41 | +1.16 [+0.72, +1.59] (better) |
| field: temperature | 2 s | 20.89 | 19.09 | +1.80 [+1.31, +2.25] (better) |
| field: soot | 8 frames | 26.87 | 26.99 | -0.13 [-0.41, +0.13] (tie) |
| field: soot | 1 s | 25.54 | 24.49 | +1.05 [+0.56, +1.55] (better) |
| field: soot | 2 s | 25.97 | 24.19 | +1.78 [+1.34, +2.21] (better) |
| picture (4 cameras) | 8 frames | 21.47 | 21.30 | +0.17 [-0.31, +0.65] (tie) |
| picture (4 cameras) | 1 s | 19.21 | 17.81 | +1.40 [+0.49, +2.26] (better) |
| picture (4 cameras) | 2 s | 19.62 | 18.03 | +1.59 [+1.11, +2.05] (better) |

(15 runs)

Endless statistics against real runs (5 settings, 2 runs pooled per setting; lower is better but for PSNR and the motion ratio, whose ideal is 1):

| statistic | C4 | N | difference |
|---|---:|---:|---|
| 3D spectrum distance | 0.262 | 0.774 | -0.512 [-0.948, -0.109] (better) |
| \|ln motion ratio\| | 0.373 | 0.440 | -0.067 [-0.543, +0.382] (tie) |
| motion ratio | 0.745 | 1.588 | -0.843 [-1.857, -0.051] (context) |
| coverage distance | 0.0028 | 0.0285 | -0.0257 [-0.0570, -0.0026] (better) |
| emission distance | 0.0040 | 0.0271 | -0.0231 [-0.0518, -0.0027] (better) |
| mean picture PSNR (dB) | 31.56 | 24.15 | +7.41 [+2.59, +13.60] (better) |

Per setting (C4 / N):

| setting (intensity, wind, turbulence) | motion ratio | coverage distance | emission distance | 3D spectrum distance | mean picture PSNR (dB) |
|---|---|---|---|---|---|
| 0.60, 0.08, 0.50 | 0.76 / 3.46 | 0.0022 / 0.0832 | 0.0060 / 0.0801 | 0.221 / 1.502 | 32.39 / 14.35 |
| 0.40, 0.92, 0.65 | 1.10 / 0.74 | 0.0005 / 0.0012 | 0.0042 / 0.0025 | 0.306 / 0.366 | 32.05 / 32.29 |
| 0.75, 0.30, 0.35 | 0.46 / 1.15 | 0.0076 / 0.0463 | 0.0035 / 0.0350 | 0.292 / 1.130 | 29.34 / 19.48 |
| 0.50, 0.50, 0.80 | 0.64 / 0.97 | 0.0029 / 0.0076 | 0.0025 / 0.0091 | 0.200 / 0.244 | 32.96 / 27.61 |
| 0.30, 0.65, 0.25 | 0.77 / 1.62 | 0.0009 / 0.0042 | 0.0039 / 0.0091 | 0.292 / 0.628 | 31.08 / 27.05 |
