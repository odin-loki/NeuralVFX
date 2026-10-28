# C3 minus smoke2 (val; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Endless statistics against real runs (5 settings, 2 runs pooled per setting; lower is better but for PSNR and the motion ratio, whose ideal is 1):

| statistic | C3 | smoke2 | difference |
|---|---:|---:|---|
| 3D spectrum distance | 0.419 | 0.488 | -0.070 [-0.149, +0.010] (tie) |
| \|ln motion ratio\| | 0.485 | 0.798 | -0.313 [-0.354, -0.273] (better) |
| motion ratio | 0.618 | 0.451 | +0.166 [+0.141, +0.192] (context) |
| coverage distance | 0.0312 | 0.0249 | +0.0063 [-0.0009, +0.0150] (tie) |
| emission distance | 0.0151 | 0.0056 | +0.0095 [+0.0058, +0.0128] (worse) |
| mean picture PSNR (dB) | 21.49 | 21.05 | +0.45 [-0.13, +0.93] (tie) |

Per setting (C3 / smoke2):

| setting (intensity, wind, turbulence) | motion ratio | coverage distance | emission distance | 3D spectrum distance | mean picture PSNR (dB) |
|---|---|---|---|---|---|
| 0.62, 0.12, 0.48 | 0.63 / 0.46 | 0.0232 / 0.0127 | 0.0082 / 0.0044 | 0.386 / 0.518 | 23.62 / 23.02 |
| 0.38, 0.88, 0.62 | 0.59 / 0.46 | 0.0380 / 0.0177 | 0.0130 / 0.0068 | 0.500 / 0.449 | 21.02 / 20.44 |
| 0.72, 0.28, 0.38 | 0.60 / 0.41 | 0.0487 / 0.0453 | 0.0165 / 0.0054 | 0.400 / 0.487 | 19.53 / 19.10 |
| 0.48, 0.52, 0.78 | 0.71 / 0.51 | 0.0201 / 0.0167 | 0.0219 / 0.0064 | 0.324 / 0.521 | 22.29 / 22.90 |
| 0.28, 0.68, 0.22 | 0.56 / 0.43 | 0.0258 / 0.0320 | 0.0157 / 0.0051 | 0.483 / 0.465 | 21.00 / 19.78 |
