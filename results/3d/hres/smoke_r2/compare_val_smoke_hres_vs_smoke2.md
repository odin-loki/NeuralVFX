# smoke_hres minus smoke2 (val; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Endless statistics against real runs (5 settings, 2 runs pooled per setting; lower is better but for PSNR and the motion ratio, whose ideal is 1):

| statistic | smoke_hres | smoke2 | difference |
|---|---:|---:|---|
| 3D spectrum distance | 0.477 | 0.488 | -0.011 [-0.078, +0.055] (tie) |
| \|ln motion ratio\| | 0.408 | 0.798 | -0.390 [-0.489, -0.268] (better) |
| motion ratio | 0.669 | 0.451 | +0.218 [+0.146, +0.271] (context) |
| coverage distance | 0.0286 | 0.0249 | +0.0037 [-0.0027, +0.0100] (tie) |
| emission distance | 0.0096 | 0.0056 | +0.0040 [+0.0032, +0.0046] (worse) |
| mean picture PSNR (dB) | 21.85 | 21.05 | +0.80 [-0.11, +1.88] (tie) |

Per setting (smoke_hres / smoke2):

| setting (intensity, wind, turbulence) | motion ratio | coverage distance | emission distance | 3D spectrum distance | mean picture PSNR (dB) |
|---|---|---|---|---|---|
| 0.62, 0.12, 0.48 | 0.53 / 0.46 | 0.0202 / 0.0127 | 0.0089 / 0.0044 | 0.455 / 0.518 | 22.44 / 23.02 |
| 0.38, 0.88, 0.62 | 0.68 / 0.46 | 0.0123 / 0.0177 | 0.0116 / 0.0068 | 0.537 / 0.449 | 22.96 / 20.44 |
| 0.72, 0.28, 0.38 | 0.66 / 0.41 | 0.0530 / 0.0453 | 0.0094 / 0.0054 | 0.534 / 0.487 | 19.30 / 19.10 |
| 0.48, 0.52, 0.78 | 0.75 / 0.51 | 0.0303 / 0.0167 | 0.0088 / 0.0064 | 0.388 / 0.521 | 23.11 / 22.90 |
| 0.28, 0.68, 0.22 | 0.72 / 0.43 | 0.0270 / 0.0320 | 0.0093 / 0.0051 | 0.470 / 0.465 | 21.44 / 19.78 |
