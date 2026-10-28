# learned minus smoke2 (test; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Endless statistics against real runs (6 settings, 3 runs pooled per setting; lower is better but for PSNR and the motion ratio, whose ideal is 1):

| statistic | learned | smoke2 | floor (other real runs) | difference |
|---|---:|---:|---:|---|
| 3D spectrum distance | 0.394 | 0.470 | 0.044 | -0.076 [-0.112, -0.023] (better) |
| \|ln motion ratio\| | 0.606 | 0.798 | 0.054 | -0.192 [-0.358, +0.073] (tie) |
| motion ratio | 0.563 | 0.451 | 0.973 | +0.112 [-0.004, +0.188] (context) |
| coverage distance | 0.0343 | 0.0265 | 0.0079 | +0.0079 [-0.0025, +0.0192] (tie) |
| emission distance | 0.0139 | 0.0089 | 0.0023 | +0.0051 [+0.0016, +0.0087] (worse) |
| mean picture PSNR (dB) | 21.67 | 20.12 | 32.57 | +1.54 [+0.64, +2.40] (better) |

Per setting (learned / smoke2):

| setting (intensity, wind, turbulence) | motion ratio | coverage distance | emission distance | 3D spectrum distance | mean picture PSNR (dB) |
|---|---|---|---|---|---|
| 0.68, 0.07, 0.42 | 0.59 / 0.43 | 0.0269 / 0.0194 | 0.0043 / 0.0040 | 0.366 / 0.496 | 23.21 / 23.42 |
| 0.52, 0.93, 0.52 | 0.65 / 0.43 | 0.0321 / 0.0154 | 0.0116 / 0.0095 | 0.333 / 0.443 | 22.23 / 20.40 |
| 0.42, 0.22, 0.68 | 0.30 / 0.46 | 0.0545 / 0.0229 | 0.0105 / 0.0090 | 0.561 / 0.510 | 19.40 / 19.09 |
| 0.78, 0.48, 0.32 | 0.62 / 0.49 | 0.0291 / 0.0269 | 0.0247 / 0.0137 | 0.362 / 0.433 | 20.52 / 18.54 |
| 0.33, 0.72, 0.42 | 0.63 / 0.43 | 0.0334 / 0.0321 | 0.0126 / 0.0090 | 0.366 / 0.464 | 23.50 / 21.14 |
| 0.63, 0.38, 0.88 | 0.60 / 0.45 | 0.0300 / 0.0420 | 0.0200 / 0.0081 | 0.378 / 0.473 | 21.17 / 18.16 |
