# N minus coarse (test; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Tracking from the true state (active PSNR, dB; paired over runs):

| measure | horizon | N | coarse | difference |
|---|---:|---:|---:|---|
| field: temperature | 8 frames | 21.81 | 19.73 | +2.08 [+1.64, +2.51] (better) |
| field: temperature | 1 s | 19.56 | 18.00 | +1.56 [+0.93, +2.15] (better) |
| field: temperature | 2 s | 18.95 | 17.55 | +1.40 [+0.96, +1.86] (better) |
| field: soot | 8 frames | 26.58 | 23.82 | +2.76 [+2.30, +3.32] (better) |
| field: soot | 1 s | 24.70 | 19.29 | +5.41 [+4.68, +6.17] (better) |
| field: soot | 2 s | 24.05 | 18.33 | +5.72 [+5.11, +6.37] (better) |
| picture (4 cameras) | 8 frames | 20.83 | 16.89 | +3.94 [+2.70, +5.24] (better) |
| picture (4 cameras) | 1 s | 17.45 | 14.24 | +3.20 [+1.86, +4.50] (better) |
| picture (4 cameras) | 2 s | 17.57 | 14.80 | +2.77 [+1.88, +3.65] (better) |

(18 runs)

Endless statistics against real runs (6 settings, 3 runs pooled per setting; lower is better but for PSNR and the motion ratio, whose ideal is 1):

| statistic | N | coarse | floor (other real runs) | difference |
|---|---:|---:|---:|---|
| 3D spectrum distance | 0.626 | 0.750 | 0.044 | -0.124 [-0.378, +0.194] (tie) |
| \|ln motion ratio\| | 0.433 | 0.227 | 0.099 | +0.206 [+0.005, +0.403] (worse) |
| motion ratio | 1.480 | 1.171 | 0.925 | +0.309 [-0.034, +0.690] (context) |
| coverage distance | 0.0272 | 0.0036 | 0.0019 | +0.0236 [+0.0018, +0.0603] (worse) |
| emission distance | 0.0238 | 0.0189 | 0.0012 | +0.0049 [-0.0108, +0.0269] (tie) |
| mean picture PSNR (dB) | 24.41 | 22.81 | 41.51 | +1.60 [-1.97, +4.46] (tie) |

Per setting (N / coarse):

| setting (intensity, wind, turbulence) | motion ratio | coverage distance | emission distance | 3D spectrum distance | mean picture PSNR (dB) |
|---|---|---|---|---|---|
| 0.70, 0.05, 0.40 | 2.58 / 1.48 | 0.1139 / 0.0026 | 0.0885 / 0.0339 | 1.432 / 0.825 | 12.72 / 19.28 |
| 0.55, 0.95, 0.55 | 0.99 / 0.84 | 0.0027 / 0.0045 | 0.0071 / 0.0082 | 0.321 / 0.640 | 27.56 / 24.02 |
| 0.45, 0.20, 0.70 | 2.00 / 1.30 | 0.0200 / 0.0017 | 0.0170 / 0.0121 | 0.686 / 0.754 | 24.59 / 24.34 |
| 0.80, 0.50, 0.30 | 0.67 / 0.91 | 0.0063 / 0.0056 | 0.0054 / 0.0278 | 0.288 / 0.825 | 28.12 / 21.81 |
| 0.35, 0.75, 0.45 | 1.48 / 1.52 | 0.0031 / 0.0015 | 0.0054 / 0.0117 | 0.393 / 0.769 | 29.39 / 25.16 |
| 0.65, 0.40, 0.90 | 1.16 / 0.98 | 0.0172 / 0.0059 | 0.0195 / 0.0198 | 0.637 / 0.687 | 24.07 / 22.26 |
