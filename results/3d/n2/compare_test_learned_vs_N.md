# learned minus N (test; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Tracking from the true state (active PSNR, dB; paired over runs):

| measure | horizon | learned | N | difference |
|---|---:|---:|---:|---|
| field: temperature | 8 frames | 21.90 | 21.81 | +0.09 [-0.16, +0.32] (tie) |
| field: temperature | 1 s | 20.80 | 19.56 | +1.24 [+0.89, +1.56] (better) |
| field: temperature | 2 s | 21.11 | 18.95 | +2.16 [+1.77, +2.57] (better) |
| field: soot | 8 frames | 26.53 | 26.58 | -0.05 [-0.29, +0.17] (tie) |
| field: soot | 1 s | 25.67 | 24.70 | +0.97 [+0.67, +1.26] (better) |
| field: soot | 2 s | 26.23 | 24.05 | +2.18 [+1.78, +2.63] (better) |
| picture (4 cameras) | 8 frames | 21.19 | 20.83 | +0.36 [-0.04, +0.74] (tie) |
| picture (4 cameras) | 1 s | 19.40 | 17.45 | +1.95 [+1.30, +2.64] (better) |
| picture (4 cameras) | 2 s | 19.93 | 17.57 | +2.36 [+1.85, +2.92] (better) |

(18 runs)

Endless statistics against real runs (6 settings, 3 runs pooled per setting; lower is better but for PSNR and the motion ratio, whose ideal is 1):

| statistic | learned | N | floor (other real runs) | difference |
|---|---:|---:|---:|---|
| 3D spectrum distance | 0.258 | 0.626 | 0.044 | -0.368 [-0.751, -0.098] (better) |
| \|ln motion ratio\| | 0.392 | 0.433 | 0.099 | -0.041 [-0.254, +0.171] (tie) |
| motion ratio | 0.682 | 1.480 | 0.925 | -0.798 [-1.358, -0.314] (context) |
| coverage distance | 0.0032 | 0.0272 | 0.0019 | -0.0240 [-0.0600, -0.0027] (better) |
| emission distance | 0.0049 | 0.0238 | 0.0012 | -0.0189 [-0.0460, -0.0032] (better) |
| mean picture PSNR (dB) | 31.89 | 24.41 | 41.51 | +7.49 [+3.86, +12.56] (better) |

Per setting (learned / N):

| setting (intensity, wind, turbulence) | motion ratio | coverage distance | emission distance | 3D spectrum distance | mean picture PSNR (dB) |
|---|---|---|---|---|---|
| 0.70, 0.05, 0.40 | 0.57 / 2.58 | 0.0047 / 0.1139 | 0.0054 / 0.0885 | 0.231 / 1.432 | 31.73 / 12.72 |
| 0.55, 0.95, 0.55 | 0.74 / 0.99 | 0.0022 / 0.0027 | 0.0052 / 0.0071 | 0.296 / 0.321 | 31.99 / 27.56 |
| 0.45, 0.20, 0.70 | 0.72 / 2.00 | 0.0017 / 0.0200 | 0.0045 / 0.0170 | 0.257 / 0.686 | 33.64 / 24.59 |
| 0.80, 0.50, 0.30 | 0.54 / 0.67 | 0.0070 / 0.0063 | 0.0045 / 0.0054 | 0.269 / 0.288 | 31.26 / 28.12 |
| 0.35, 0.75, 0.45 | 0.76 / 1.48 | 0.0010 / 0.0031 | 0.0038 / 0.0054 | 0.314 / 0.393 | 32.04 / 29.39 |
| 0.65, 0.40, 0.90 | 0.76 / 1.16 | 0.0027 / 0.0172 | 0.0063 / 0.0195 | 0.184 / 0.637 | 30.69 / 24.07 |
