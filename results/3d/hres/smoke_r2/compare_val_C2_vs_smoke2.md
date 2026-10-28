# C2 minus smoke2 (val; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Endless statistics against real runs (5 settings, 2 runs pooled per setting; lower is better but for PSNR and the motion ratio, whose ideal is 1):

| statistic | C2 | smoke2 | difference |
|---|---:|---:|---|
| 3D spectrum distance | 0.456 | 0.488 | -0.033 [-0.142, +0.096] (tie) |
| \|ln motion ratio\| | 0.577 | 0.798 | -0.221 [-0.270, -0.168] (better) |
| motion ratio | 0.564 | 0.451 | +0.112 [+0.084, +0.140] (context) |
| coverage distance | 0.0351 | 0.0249 | +0.0103 [+0.0047, +0.0164] (worse) |
| emission distance | 0.0129 | 0.0056 | +0.0073 [+0.0039, +0.0108] (worse) |
| mean picture PSNR (dB) | 21.24 | 21.05 | +0.19 [-0.16, +0.60] (tie) |

Per setting (C2 / smoke2):

| setting (intensity, wind, turbulence) | motion ratio | coverage distance | emission distance | 3D spectrum distance | mean picture PSNR (dB) |
|---|---|---|---|---|---|
| 0.62, 0.12, 0.48 | 0.52 / 0.46 | 0.0297 / 0.0127 | 0.0065 / 0.0044 | 0.410 / 0.518 | 23.13 / 23.02 |
| 0.38, 0.88, 0.62 | 0.57 / 0.46 | 0.0375 / 0.0177 | 0.0112 / 0.0068 | 0.516 / 0.449 | 21.40 / 20.44 |
| 0.72, 0.28, 0.38 | 0.55 / 0.41 | 0.0494 / 0.0453 | 0.0129 / 0.0054 | 0.393 / 0.487 | 19.43 / 19.10 |
| 0.48, 0.52, 0.78 | 0.66 / 0.51 | 0.0222 / 0.0167 | 0.0204 / 0.0064 | 0.321 / 0.521 | 22.69 / 22.90 |
| 0.28, 0.68, 0.22 | 0.51 / 0.43 | 0.0369 / 0.0320 | 0.0137 / 0.0051 | 0.639 / 0.465 | 19.55 / 19.78 |
