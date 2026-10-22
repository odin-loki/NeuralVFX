# Nfix minus coarse (val; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Tracking from the true state (active PSNR, dB; paired over runs):

| measure | horizon | Nfix | coarse | difference |
|---|---:|---:|---:|---|
| field: temperature | 8 frames | 22.08 | 19.55 | +2.53 [+1.92, +3.19] (better) |
| field: temperature | 1 s | 19.38 | 17.42 | +1.96 [+1.27, +2.60] (better) |
| field: temperature | 2 s | 19.06 | 17.50 | +1.56 [+1.03, +2.20] (better) |
| field: soot | 8 frames | 26.99 | 23.82 | +3.17 [+2.55, +3.80] (better) |
| field: soot | 1 s | 24.47 | 19.22 | +5.25 [+4.45, +5.97] (better) |
| field: soot | 2 s | 24.15 | 17.59 | +6.55 [+5.70, +7.42] (better) |
| picture (4 cameras) | 8 frames | 21.30 | 16.38 | +4.92 [+3.57, +6.34] (better) |
| picture (4 cameras) | 1 s | 17.81 | 13.61 | +4.19 [+2.71, +5.50] (better) |
| picture (4 cameras) | 2 s | 18.00 | 14.93 | +3.07 [+1.92, +4.23] (better) |

(15 runs)

Endless statistics against real runs (5 settings, 2 runs pooled per setting; lower is better but for PSNR and the motion ratio, whose ideal is 1):

| statistic | Nfix | coarse | difference |
|---|---:|---:|---|
| 3D spectrum distance | 0.353 | 0.772 | -0.420 [-0.481, -0.359] (better) |
| \|ln motion ratio\| | 0.379 | 0.316 | +0.063 [-0.320, +0.446] (tie) |
| motion ratio | 0.789 | 1.336 | -0.547 [-0.717, -0.371] (context) |
| coverage distance | 0.0040 | 0.0035 | +0.0005 [-0.0004, +0.0018] (tie) |
| emission distance | 0.0052 | 0.0173 | -0.0120 [-0.0167, -0.0072] (better) |
| mean picture PSNR (dB) | 28.05 | 23.62 | +4.43 [+2.90, +5.95] (better) |

Per setting (Nfix / coarse):

| setting (intensity, wind, turbulence) | motion ratio | coverage distance | emission distance | 3D spectrum distance | mean picture PSNR (dB) |
|---|---|---|---|---|---|
| 0.60, 0.08, 0.50 | 1.22 / 1.94 | 0.0035 / 0.0038 | 0.0099 / 0.0296 | 0.398 / 0.891 | 26.24 / 20.04 |
| 0.40, 0.92, 0.65 | 0.72 / 1.06 | 0.0013 / 0.0016 | 0.0025 / 0.0061 | 0.365 / 0.695 | 32.11 / 25.55 |
| 0.75, 0.30, 0.35 | 0.57 / 0.84 | 0.0084 / 0.0079 | 0.0050 / 0.0196 | 0.370 / 0.720 | 25.28 / 22.88 |
| 0.50, 0.50, 0.80 | 0.45 / 1.21 | 0.0060 / 0.0030 | 0.0053 / 0.0179 | 0.227 / 0.719 | 27.99 / 23.73 |
| 0.30, 0.65, 0.25 | 0.98 / 1.63 | 0.0008 / 0.0013 | 0.0033 / 0.0131 | 0.404 / 0.836 | 28.62 / 25.91 |
