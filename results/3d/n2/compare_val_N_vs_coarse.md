# N minus coarse (val; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Tracking from the true state (active PSNR, dB; paired over runs):

| measure | horizon | N | coarse | difference |
|---|---:|---:|---:|---|
| field: temperature | 8 frames | 22.08 | 19.55 | +2.54 [+1.93, +3.20] (better) |
| field: temperature | 1 s | 19.41 | 17.42 | +1.98 [+1.29, +2.63] (better) |
| field: temperature | 2 s | 19.09 | 17.50 | +1.59 [+1.05, +2.23] (better) |
| field: soot | 8 frames | 26.99 | 23.82 | +3.17 [+2.55, +3.80] (better) |
| field: soot | 1 s | 24.49 | 19.22 | +5.27 [+4.46, +5.98] (better) |
| field: soot | 2 s | 24.19 | 17.59 | +6.60 [+5.74, +7.46] (better) |
| picture (4 cameras) | 8 frames | 21.30 | 16.38 | +4.92 [+3.57, +6.34] (better) |
| picture (4 cameras) | 1 s | 17.81 | 13.61 | +4.20 [+2.71, +5.51] (better) |
| picture (4 cameras) | 2 s | 18.03 | 14.93 | +3.10 [+1.95, +4.25] (better) |

(15 runs)

Endless statistics against real runs (5 settings, 2 runs pooled per setting; lower is better but for PSNR and the motion ratio, whose ideal is 1):

| statistic | N | coarse | difference |
|---|---:|---:|---|
| 3D spectrum distance | 0.774 | 0.772 | +0.002 [-0.363, +0.382] (tie) |
| \|ln motion ratio\| | 0.440 | 0.316 | +0.124 [-0.080, +0.379] (tie) |
| motion ratio | 1.588 | 1.336 | +0.252 [-0.229, +0.928] (context) |
| coverage distance | 0.0285 | 0.0035 | +0.0250 [+0.0019, +0.0559] (worse) |
| emission distance | 0.0271 | 0.0173 | +0.0099 [-0.0058, +0.0326] (tie) |
| mean picture PSNR (dB) | 24.15 | 23.62 | +0.53 [-3.41, +4.48] (tie) |

Per setting (N / coarse):

| setting (intensity, wind, turbulence) | motion ratio | coverage distance | emission distance | 3D spectrum distance | mean picture PSNR (dB) |
|---|---|---|---|---|---|
| 0.60, 0.08, 0.50 | 3.46 / 1.94 | 0.0832 / 0.0038 | 0.0801 / 0.0296 | 1.502 / 0.891 | 14.35 / 20.04 |
| 0.40, 0.92, 0.65 | 0.74 / 1.06 | 0.0012 / 0.0016 | 0.0025 / 0.0061 | 0.366 / 0.695 | 32.29 / 25.55 |
| 0.75, 0.30, 0.35 | 1.15 / 0.84 | 0.0463 / 0.0079 | 0.0350 / 0.0196 | 1.130 / 0.720 | 19.48 / 22.88 |
| 0.50, 0.50, 0.80 | 0.97 / 1.21 | 0.0076 / 0.0030 | 0.0091 / 0.0179 | 0.244 / 0.719 | 27.61 / 23.73 |
| 0.30, 0.65, 0.25 | 1.62 / 1.63 | 0.0042 / 0.0013 | 0.0091 / 0.0131 | 0.628 / 0.836 | 27.05 / 25.91 |
