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
