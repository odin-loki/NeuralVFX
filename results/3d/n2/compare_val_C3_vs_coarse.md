# C3 minus coarse (val; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Tracking from the true state (active PSNR, dB; paired over runs):

| measure | horizon | C3 | coarse | difference |
|---|---:|---:|---:|---|
| field: temperature | 8 frames | 21.87 | 19.55 | +2.32 [+1.71, +2.97] (better) |
| field: temperature | 1 s | 18.88 | 17.42 | +1.46 [+0.80, +2.13] (better) |
| field: temperature | 2 s | 18.76 | 17.50 | +1.26 [+0.78, +1.84] (better) |
| field: soot | 8 frames | 26.90 | 23.82 | +3.08 [+2.48, +3.70] (better) |
| field: soot | 1 s | 24.27 | 19.22 | +5.05 [+4.37, +5.63] (better) |
| field: soot | 2 s | 24.25 | 17.59 | +6.66 [+6.00, +7.34] (better) |
| picture (4 cameras) | 8 frames | 20.89 | 16.38 | +4.52 [+3.25, +5.85] (better) |
| picture (4 cameras) | 1 s | 17.28 | 13.61 | +3.67 [+2.34, +4.89] (better) |
| picture (4 cameras) | 2 s | 17.86 | 14.93 | +2.93 [+1.90, +3.98] (better) |

(15 runs)

Endless statistics against real runs (5 settings, 2 runs pooled per setting; lower is better but for PSNR and the motion ratio, whose ideal is 1):

| statistic | C3 | coarse | difference |
|---|---:|---:|---|
| 3D spectrum distance | 0.301 | 0.772 | -0.471 [-0.577, -0.356] (better) |
| \|ln motion ratio\| | 0.485 | 0.316 | +0.170 [-0.130, +0.555] (tie) |
| motion ratio | 0.652 | 1.336 | -0.684 [-1.059, -0.347] (context) |
| coverage distance | 0.0046 | 0.0035 | +0.0011 [+0.0001, +0.0026] (worse) |
| emission distance | 0.0043 | 0.0173 | -0.0130 [-0.0202, -0.0065] (better) |
| mean picture PSNR (dB) | 28.91 | 23.62 | +5.29 [+3.30, +7.51] (better) |

Per setting (C3 / coarse):

| setting (intensity, wind, turbulence) | motion ratio | coverage distance | emission distance | 3D spectrum distance | mean picture PSNR (dB) |
|---|---|---|---|---|---|
| 0.60, 0.08, 0.50 | 0.68 / 1.94 | 0.0039 / 0.0038 | 0.0041 / 0.0296 | 0.241 / 0.891 | 29.69 / 20.04 |
| 0.40, 0.92, 0.65 | 0.99 / 1.06 | 0.0013 / 0.0016 | 0.0036 / 0.0061 | 0.432 / 0.695 | 28.17 / 25.55 |
| 0.75, 0.30, 0.35 | 0.33 / 0.84 | 0.0119 / 0.0079 | 0.0072 / 0.0196 | 0.300 / 0.720 | 25.92 / 22.88 |
| 0.50, 0.50, 0.80 | 0.65 / 1.21 | 0.0041 / 0.0030 | 0.0032 / 0.0179 | 0.224 / 0.719 | 29.27 / 23.73 |
| 0.30, 0.65, 0.25 | 0.62 / 1.63 | 0.0020 / 0.0013 | 0.0032 / 0.0131 | 0.307 / 0.836 | 31.48 / 25.91 |
