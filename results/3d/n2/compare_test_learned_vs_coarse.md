# learned minus coarse (test; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Tracking from the true state (active PSNR, dB; paired over runs):

| measure | horizon | learned | coarse | difference |
|---|---:|---:|---:|---|
| field: temperature | 8 frames | 21.90 | 19.73 | +2.17 [+1.83, +2.52] (better) |
| field: temperature | 1 s | 20.80 | 18.00 | +2.79 [+2.20, +3.36] (better) |
| field: temperature | 2 s | 21.11 | 17.55 | +3.56 [+3.06, +4.09] (better) |
| field: soot | 8 frames | 26.53 | 23.82 | +2.71 [+2.33, +3.23] (better) |
| field: soot | 1 s | 25.67 | 19.29 | +6.38 [+5.58, +7.18] (better) |
| field: soot | 2 s | 26.23 | 18.33 | +7.90 [+7.34, +8.54] (better) |
| picture (4 cameras) | 8 frames | 21.19 | 16.89 | +4.30 [+3.21, +5.47] (better) |
| picture (4 cameras) | 1 s | 19.40 | 14.24 | +5.15 [+3.88, +6.38] (better) |
| picture (4 cameras) | 2 s | 19.93 | 14.80 | +5.13 [+4.31, +5.89] (better) |

(18 runs)

Endless statistics against real runs (6 settings, 3 runs pooled per setting; lower is better but for PSNR and the motion ratio, whose ideal is 1):

| statistic | learned | coarse | floor (other real runs) | difference |
|---|---:|---:|---:|---|
| 3D spectrum distance | 0.258 | 0.750 | 0.044 | -0.492 [-0.550, -0.424] (better) |
| \|ln motion ratio\| | 0.392 | 0.227 | 0.099 | +0.165 [+0.007, +0.329] (worse) |
| motion ratio | 0.682 | 1.171 | 0.925 | -0.489 [-0.716, -0.264] (context) |
| coverage distance | 0.0032 | 0.0036 | 0.0019 | -0.0004 [-0.0019, +0.0010] (tie) |
| emission distance | 0.0049 | 0.0189 | 0.0012 | -0.0140 [-0.0217, -0.0071] (better) |
| mean picture PSNR (dB) | 31.89 | 22.81 | 41.51 | +9.08 [+7.85, +10.59] (better) |

Per setting (learned / coarse):

| setting (intensity, wind, turbulence) | motion ratio | coverage distance | emission distance | 3D spectrum distance | mean picture PSNR (dB) |
|---|---|---|---|---|---|
| 0.70, 0.05, 0.40 | 0.57 / 1.48 | 0.0047 / 0.0026 | 0.0054 / 0.0339 | 0.231 / 0.825 | 31.73 / 19.28 |
| 0.55, 0.95, 0.55 | 0.74 / 0.84 | 0.0022 / 0.0045 | 0.0052 / 0.0082 | 0.296 / 0.640 | 31.99 / 24.02 |
| 0.45, 0.20, 0.70 | 0.72 / 1.30 | 0.0017 / 0.0017 | 0.0045 / 0.0121 | 0.257 / 0.754 | 33.64 / 24.34 |
| 0.80, 0.50, 0.30 | 0.54 / 0.91 | 0.0070 / 0.0056 | 0.0045 / 0.0278 | 0.269 / 0.825 | 31.26 / 21.81 |
| 0.35, 0.75, 0.45 | 0.76 / 1.52 | 0.0010 / 0.0015 | 0.0038 / 0.0117 | 0.314 / 0.769 | 32.04 / 25.16 |
| 0.65, 0.40, 0.90 | 0.76 / 0.98 | 0.0027 / 0.0059 | 0.0063 / 0.0198 | 0.184 / 0.687 | 30.69 / 22.26 |
