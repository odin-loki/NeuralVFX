# B minus coarse (val; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Tracking from the true state (active PSNR, dB; paired over runs):

| measure | horizon | B | coarse | difference |
|---|---:|---:|---:|---|
| field: temperature | 8 frames | 22.26 | 20.41 | +1.85 [+1.59, +2.13] (better) |
| field: temperature | 1 s | 19.31 | 18.31 | +1.00 [+0.04, +1.91] (better) |
| field: temperature | 2 s | 18.68 | 16.85 | +1.83 [+1.27, +2.44] (better) |
| field: soot | 8 frames | 27.20 | 24.54 | +2.66 [+2.21, +3.17] (better) |
| field: soot | 1 s | 24.84 | 19.56 | +5.27 [+4.50, +6.03] (better) |
| field: soot | 2 s | 24.21 | 17.57 | +6.64 [+5.96, +7.36] (better) |
| picture (4 cameras) | 8 frames | 20.81 | 18.45 | +2.36 [+1.84, +2.89] (better) |
| picture (4 cameras) | 1 s | 17.08 | 14.78 | +2.30 [+0.64, +3.91] (better) |
| picture (4 cameras) | 2 s | 17.51 | 14.67 | +2.84 [+1.89, +3.67] (better) |

(12 runs)

Endless statistics against real runs (4 settings, 2 runs pooled per setting; lower is better but for PSNR and the motion ratio, whose ideal is 1):

| statistic | B | coarse | difference |
|---|---:|---:|---|
| 3D spectrum distance | 0.358 | 0.778 | -0.420 [-0.514, -0.301] (better) |
| \|ln motion ratio\| | 0.491 | 0.233 | +0.258 [+0.047, +0.469] (worse) |
| motion ratio | 0.831 | 1.099 | -0.268 [-0.452, -0.044] (context) |
| coverage distance | 0.0054 | 0.0037 | +0.0017 [+0.0005, +0.0027] (worse) |
| emission distance | 0.0049 | 0.0136 | -0.0088 [-0.0129, -0.0058] (better) |
| mean picture PSNR (dB) | 27.50 | 23.55 | +3.95 [+2.24, +5.38] (better) |

Per setting (B / coarse):

| setting (intensity, wind, turbulence) | motion ratio | coverage distance | emission distance | 3D spectrum distance | mean picture PSNR (dB) |
|---|---|---|---|---|---|
| 0.35, 0.30, 0.40 | 1.49 / 1.44 | 0.0042 / 0.0011 | 0.0078 / 0.0127 | 0.564 / 0.810 | 26.77 / 25.39 |
| 0.65, 0.70, 0.60 | 0.66 / 0.96 | 0.0037 / 0.0038 | 0.0033 / 0.0117 | 0.304 / 0.771 | 26.74 / 23.07 |
| 0.50, 0.15, 0.75 | 0.75 / 1.25 | 0.0031 / 0.0016 | 0.0044 / 0.0110 | 0.268 / 0.812 | 29.09 / 23.21 |
| 0.80, 0.55, 0.25 | 0.42 / 0.74 | 0.0106 / 0.0083 | 0.0040 / 0.0191 | 0.295 / 0.719 | 27.39 / 22.52 |
