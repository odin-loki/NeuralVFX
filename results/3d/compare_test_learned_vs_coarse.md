# learned minus coarse (test; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Tracking from the true state (active PSNR, dB; paired over runs):

| measure | horizon | learned | coarse | difference |
|---|---:|---:|---:|---|
| field: temperature | 8 frames | 22.57 | 20.58 | +1.99 [+1.73, +2.27] (better) |
| field: temperature | 1 s | 19.31 | 18.23 | +1.08 [+0.74, +1.39] (better) |
| field: temperature | 2 s | 18.86 | 17.77 | +1.09 [+0.67, +1.49] (better) |
| field: soot | 8 frames | 27.35 | 24.78 | +2.57 [+2.26, +2.92] (better) |
| field: soot | 1 s | 24.77 | 19.22 | +5.55 [+4.84, +6.21] (better) |
| field: soot | 2 s | 24.22 | 17.79 | +6.43 [+5.92, +6.92] (better) |
| picture (4 cameras) | 8 frames | 21.15 | 17.90 | +3.25 [+2.69, +3.81] (better) |
| picture (4 cameras) | 1 s | 16.98 | 14.78 | +2.20 [+1.09, +3.12] (better) |
| picture (4 cameras) | 2 s | 17.00 | 14.89 | +2.11 [+1.25, +2.98] (better) |

(18 runs)

Endless statistics against real runs (6 settings, 3 runs pooled per setting; lower is better but for PSNR and the motion ratio, whose ideal is 1):

| statistic | learned | coarse | floor (other real runs) | difference |
|---|---:|---:|---:|---|
| 3D spectrum distance | 0.451 | 0.769 | 0.034 | -0.318 [-0.477, -0.053] (better) |
| \|ln motion ratio\| | 0.409 | 0.181 | 0.064 | +0.229 [+0.063, +0.402] (worse) |
| motion ratio | 1.063 | 1.071 | 1.015 | -0.008 [-0.323, +0.421] (context) |
| coverage distance | 0.0089 | 0.0040 | 0.0016 | +0.0049 [+0.0006, +0.0116] (worse) |
| emission distance | 0.0101 | 0.0159 | 0.0011 | -0.0059 [-0.0128, +0.0037] (tie) |
| mean picture PSNR (dB) | 27.18 | 23.91 | 42.52 | +3.27 [+0.54, +5.20] (better) |
