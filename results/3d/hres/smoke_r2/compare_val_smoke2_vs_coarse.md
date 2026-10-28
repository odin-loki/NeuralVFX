# smoke2 minus coarse (val; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Endless statistics against real runs (5 settings, 2 runs pooled per setting; lower is better but for PSNR and the motion ratio, whose ideal is 1):

| statistic | smoke2 | coarse | difference |
|---|---:|---:|---|
| 3D spectrum distance | 0.488 | 0.328 | +0.160 [+0.134, +0.187] (worse) |
| \|ln motion ratio\| | 0.798 | 0.251 | +0.547 [+0.470, +0.626] (worse) |
| motion ratio | 0.451 | 0.780 | -0.328 [-0.375, -0.278] (context) |
| coverage distance | 0.0249 | 0.0212 | +0.0037 [-0.0087, +0.0176] (tie) |
| emission distance | 0.0056 | 0.0131 | -0.0075 [-0.0112, -0.0034] (better) |
| mean picture PSNR (dB) | 21.05 | 23.72 | -2.67 [-4.16, -0.90] (worse) |

Per setting (smoke2 / coarse):

| setting (intensity, wind, turbulence) | motion ratio | coverage distance | emission distance | 3D spectrum distance | mean picture PSNR (dB) |
|---|---|---|---|---|---|
| 0.62, 0.12, 0.48 | 0.46 / 0.70 | 0.0127 / 0.0161 | 0.0044 / 0.0055 | 0.518 / 0.316 | 23.02 / 25.92 |
| 0.38, 0.88, 0.62 | 0.46 / 0.74 | 0.0177 / 0.0323 | 0.0068 / 0.0156 | 0.449 / 0.322 | 20.44 / 22.40 |
| 0.72, 0.28, 0.38 | 0.41 / 0.81 | 0.0453 / 0.0193 | 0.0054 / 0.0095 | 0.487 / 0.337 | 19.10 / 24.07 |
| 0.48, 0.52, 0.78 | 0.51 / 0.86 | 0.0167 / 0.0221 | 0.0064 / 0.0204 | 0.521 / 0.331 | 22.90 / 22.43 |
| 0.28, 0.68, 0.22 | 0.43 / 0.79 | 0.0320 / 0.0162 | 0.0051 / 0.0148 | 0.465 / 0.332 | 19.78 / 23.75 |
