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

Per setting (learned / coarse):

| setting (intensity, wind, turbulence) | motion ratio | coverage distance | emission distance | 3D spectrum distance | mean picture PSNR (dB) |
|---|---|---|---|---|---|
| 0.30, 0.60, 0.55 | 0.77 / 1.22 | 0.0013 / 0.0019 | 0.0023 / 0.0077 | 0.320 / 0.769 | 29.68 / 27.13 |
| 0.45, 0.25, 0.30 | 2.40 / 1.44 | 0.0222 / 0.0013 | 0.0298 / 0.0138 | 1.144 / 0.821 | 22.23 / 25.13 |
| 0.55, 0.80, 0.70 | 1.15 / 1.11 | 0.0044 / 0.0031 | 0.0056 / 0.0114 | 0.327 / 0.694 | 28.46 / 24.19 |
| 0.70, 0.40, 0.85 | 0.53 / 0.85 | 0.0075 / 0.0068 | 0.0085 / 0.0178 | 0.254 / 0.743 | 26.68 / 22.08 |
| 0.85, 0.65, 0.45 | 0.82 / 0.79 | 0.0124 / 0.0073 | 0.0099 / 0.0259 | 0.344 / 0.763 | 26.34 / 21.54 |
| 0.60, 0.50, 0.20 | 0.71 / 1.02 | 0.0054 / 0.0033 | 0.0044 / 0.0190 | 0.315 / 0.821 | 29.68 / 23.39 |
