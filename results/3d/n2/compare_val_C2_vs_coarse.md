# C2 minus coarse (val; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Tracking from the true state (active PSNR, dB; paired over runs):

| measure | horizon | C2 | coarse | difference |
|---|---:|---:|---:|---|
| field: temperature | 8 frames | 22.11 | 19.55 | +2.57 [+1.95, +3.22] (better) |
| field: temperature | 1 s | 19.22 | 17.42 | +1.79 [+1.14, +2.45] (better) |
| field: temperature | 2 s | 18.90 | 17.50 | +1.40 [+0.99, +1.93] (better) |
| field: soot | 8 frames | 27.09 | 23.82 | +3.27 [+2.68, +3.88] (better) |
| field: soot | 1 s | 24.55 | 19.22 | +5.33 [+4.67, +5.93] (better) |
| field: soot | 2 s | 24.30 | 17.59 | +6.71 [+6.05, +7.41] (better) |
| picture (4 cameras) | 8 frames | 21.34 | 16.38 | +4.96 [+3.61, +6.39] (better) |
| picture (4 cameras) | 1 s | 17.74 | 13.61 | +4.13 [+2.79, +5.37] (better) |
| picture (4 cameras) | 2 s | 18.03 | 14.93 | +3.10 [+2.20, +4.02] (better) |

(15 runs)

Endless statistics against real runs (5 settings, 2 runs pooled per setting; lower is better but for PSNR and the motion ratio, whose ideal is 1):

| statistic | C2 | coarse | difference |
|---|---:|---:|---|
| 3D spectrum distance | 0.316 | 0.772 | -0.457 [-0.548, -0.355] (better) |
| \|ln motion ratio\| | 0.424 | 0.316 | +0.109 [-0.289, +0.519] (tie) |
| motion ratio | 0.721 | 1.336 | -0.615 [-0.938, -0.256] (context) |
| coverage distance | 0.0044 | 0.0035 | +0.0009 [-0.0005, +0.0025] (tie) |
| emission distance | 0.0045 | 0.0173 | -0.0127 [-0.0197, -0.0061] (better) |
| mean picture PSNR (dB) | 28.84 | 23.62 | +5.22 [+3.37, +7.69] (better) |

Per setting (C2 / coarse):

| setting (intensity, wind, turbulence) | motion ratio | coverage distance | emission distance | 3D spectrum distance | mean picture PSNR (dB) |
|---|---|---|---|---|---|
| 0.60, 0.08, 0.50 | 0.81 / 1.94 | 0.0034 / 0.0038 | 0.0048 / 0.0296 | 0.278 / 0.891 | 29.99 / 20.04 |
| 0.40, 0.92, 0.65 | 1.08 / 1.06 | 0.0008 / 0.0016 | 0.0045 / 0.0061 | 0.417 / 0.695 | 29.15 / 25.55 |
| 0.75, 0.30, 0.35 | 0.36 / 0.84 | 0.0116 / 0.0079 | 0.0067 / 0.0196 | 0.302 / 0.720 | 25.72 / 22.88 |
| 0.50, 0.50, 0.80 | 0.55 / 1.21 | 0.0049 / 0.0030 | 0.0042 / 0.0179 | 0.215 / 0.719 | 29.47 / 23.73 |
| 0.30, 0.65, 0.25 | 0.81 / 1.63 | 0.0012 / 0.0013 | 0.0024 / 0.0131 | 0.366 / 0.836 | 29.88 / 25.91 |
