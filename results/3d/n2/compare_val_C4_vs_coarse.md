# C4 minus coarse (val; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Tracking from the true state (active PSNR, dB; paired over runs):

| measure | horizon | C4 | coarse | difference |
|---|---:|---:|---:|---|
| field: temperature | 8 frames | 22.14 | 19.55 | +2.59 [+1.98, +3.26] (better) |
| field: temperature | 1 s | 20.56 | 17.42 | +3.14 [+2.43, +3.80] (better) |
| field: temperature | 2 s | 20.89 | 17.50 | +3.39 [+2.87, +3.96] (better) |
| field: soot | 8 frames | 26.87 | 23.82 | +3.05 [+2.45, +3.70] (better) |
| field: soot | 1 s | 25.54 | 19.22 | +6.32 [+5.52, +7.09] (better) |
| field: soot | 2 s | 25.97 | 17.59 | +8.38 [+7.66, +9.11] (better) |
| picture (4 cameras) | 8 frames | 21.47 | 16.38 | +5.09 [+3.76, +6.59] (better) |
| picture (4 cameras) | 1 s | 19.21 | 13.61 | +5.60 [+4.33, +6.75] (better) |
| picture (4 cameras) | 2 s | 19.62 | 14.93 | +4.69 [+3.85, +5.58] (better) |

(15 runs)

Endless statistics against real runs (5 settings, 2 runs pooled per setting; lower is better but for PSNR and the motion ratio, whose ideal is 1):

| statistic | C4 | coarse | difference |
|---|---:|---:|---|
| 3D spectrum distance | 0.262 | 0.772 | -0.510 [-0.596, -0.428] (better) |
| \|ln motion ratio\| | 0.373 | 0.316 | +0.057 [-0.238, +0.372] (tie) |
| motion ratio | 0.745 | 1.336 | -0.591 [-0.954, -0.225] (context) |
| coverage distance | 0.0028 | 0.0035 | -0.0007 [-0.0012, -0.0002] (better) |
| emission distance | 0.0040 | 0.0173 | -0.0132 [-0.0192, -0.0063] (better) |
| mean picture PSNR (dB) | 31.56 | 23.62 | +7.94 [+5.96, +10.48] (better) |

Per setting (C4 / coarse):

| setting (intensity, wind, turbulence) | motion ratio | coverage distance | emission distance | 3D spectrum distance | mean picture PSNR (dB) |
|---|---|---|---|---|---|
| 0.60, 0.08, 0.50 | 0.76 / 1.94 | 0.0022 / 0.0038 | 0.0060 / 0.0296 | 0.221 / 0.891 | 32.39 / 20.04 |
| 0.40, 0.92, 0.65 | 1.10 / 1.06 | 0.0005 / 0.0016 | 0.0042 / 0.0061 | 0.306 / 0.695 | 32.05 / 25.55 |
| 0.75, 0.30, 0.35 | 0.46 / 0.84 | 0.0076 / 0.0079 | 0.0035 / 0.0196 | 0.292 / 0.720 | 29.34 / 22.88 |
| 0.50, 0.50, 0.80 | 0.64 / 1.21 | 0.0029 / 0.0030 | 0.0025 / 0.0179 | 0.200 / 0.719 | 32.96 / 23.73 |
| 0.30, 0.65, 0.25 | 0.77 / 1.63 | 0.0009 / 0.0013 | 0.0039 / 0.0131 | 0.292 / 0.836 | 31.08 / 25.91 |
