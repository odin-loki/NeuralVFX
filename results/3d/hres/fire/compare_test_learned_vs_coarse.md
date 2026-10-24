# learned minus coarse (test; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Tracking from the true state (active PSNR, dB; paired over runs):

| measure | horizon | learned | coarse | difference |
|---|---:|---:|---:|---|
| field: temperature | 8 frames | 26.15 | 23.83 | +2.32 [+1.93, +2.72] (better) |
| field: temperature | 1 s | 22.07 | 20.78 | +1.29 [+0.98, +1.60] (better) |
| field: temperature | 2 s | 20.60 | 19.76 | +0.84 [+0.14, +1.53] (better) |
| field: soot | 8 frames | 30.68 | 27.32 | +3.35 [+2.90, +3.83] (better) |
| field: soot | 1 s | 27.28 | 22.44 | +4.84 [+4.36, +5.35] (better) |
| field: soot | 2 s | 25.63 | 20.44 | +5.19 [+4.15, +6.30] (better) |
| picture (4 cameras) | 8 frames | 25.10 | 22.53 | +2.58 [+1.78, +3.39] (better) |
| picture (4 cameras) | 1 s | 21.32 | 19.63 | +1.69 [+1.07, +2.32] (better) |
| picture (4 cameras) | 2 s | 20.07 | 19.08 | +0.99 [-0.34, +2.15] (tie) |

(18 runs)

Endless statistics against real runs (6 settings, 3 runs pooled per setting; lower is better but for PSNR and the motion ratio, whose ideal is 1):

| statistic | learned | coarse | floor (other real runs) | difference |
|---|---:|---:|---:|---|
| 3D spectrum distance | 0.183 | 0.460 | 0.045 | -0.277 [-0.361, -0.202] (better) |
| \|ln motion ratio\| | 0.105 | 0.420 | 0.091 | -0.315 [-0.419, -0.223] (better) |
| motion ratio | 0.988 | 0.660 | 0.995 | +0.328 [+0.223, +0.429] (context) |
| coverage distance | 0.0021 | 0.0025 | 0.0020 | -0.0004 [-0.0011, +0.0002] (tie) |
| emission distance | 0.0030 | 0.0031 | 0.0012 | -0.0000 [-0.0006, +0.0006] (tie) |
| mean picture PSNR (dB) | 32.98 | 31.26 | 41.07 | +1.73 [-0.29, +3.87] (tie) |

Per setting (learned / coarse):

| setting (intensity, wind, turbulence) | motion ratio | coverage distance | emission distance | 3D spectrum distance | mean picture PSNR (dB) |
|---|---|---|---|---|---|
| 0.68, 0.07, 0.42 | 0.84 / 0.72 | 0.0017 / 0.0026 | 0.0019 / 0.0028 | 0.134 / 0.589 | 33.64 / 27.52 |
| 0.52, 0.93, 0.52 | 0.99 / 0.57 | 0.0007 / 0.0020 | 0.0016 / 0.0023 | 0.188 / 0.345 | 34.97 / 33.33 |
| 0.42, 0.22, 0.68 | 1.07 / 0.72 | 0.0015 / 0.0011 | 0.0018 / 0.0016 | 0.207 / 0.513 | 33.71 / 32.31 |
| 0.78, 0.48, 0.32 | 1.01 / 0.68 | 0.0020 / 0.0036 | 0.0057 / 0.0064 | 0.161 / 0.481 | 30.75 / 29.61 |
| 0.33, 0.72, 0.42 | 1.19 / 0.66 | 0.0013 / 0.0011 | 0.0017 / 0.0012 | 0.230 / 0.418 | 32.35 / 34.64 |
| 0.63, 0.38, 0.88 | 0.82 / 0.60 | 0.0053 / 0.0048 | 0.0053 / 0.0040 | 0.179 / 0.416 | 32.50 / 30.14 |
