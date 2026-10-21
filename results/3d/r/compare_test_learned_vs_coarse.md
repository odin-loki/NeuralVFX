# learned minus coarse (test; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Tracking from the true state (active PSNR, dB; paired over runs):

| measure | horizon | learned | coarse | difference |
|---|---:|---:|---:|---|
| field: temperature | 8 frames | 21.31 | 19.80 | +1.51 [+1.29, +1.74] (better) |
| field: temperature | 1 s | 20.02 | 18.17 | +1.85 [+1.45, +2.27] (better) |
| field: temperature | 2 s | 20.62 | 17.74 | +2.88 [+2.38, +3.39] (better) |
| field: soot | 8 frames | 20.83 | 19.30 | +1.53 [+1.31, +1.76] (better) |
| field: soot | 1 s | 19.56 | 17.67 | +1.89 [+1.47, +2.32] (better) |
| field: soot | 2 s | 20.16 | 17.27 | +2.90 [+2.40, +3.41] (better) |
| picture (4 cameras) | 8 frames | 22.74 | 21.13 | +1.62 [+1.16, +2.10] (better) |
| picture (4 cameras) | 1 s | 17.87 | 16.28 | +1.59 [+1.02, +2.20] (better) |
| picture (4 cameras) | 2 s | 16.39 | 14.78 | +1.62 [+1.08, +2.19] (better) |

(18 runs)

Endless statistics against real runs (6 settings, 3 runs pooled per setting; lower is better but for PSNR and the motion ratio, whose ideal is 1):

| statistic | learned | coarse | floor (other real runs) | difference |
|---|---:|---:|---:|---|
| 3D spectrum distance | 0.504 | 0.686 | 0.038 | -0.182 [-0.230, -0.140] (better) |
| \|ln motion ratio\| | 0.552 | 0.602 | 0.065 | -0.051 [-0.182, +0.084] (tie) |
| motion ratio | 0.582 | 0.549 | 1.017 | +0.033 [-0.043, +0.107] (context) |
| coverage distance | 0.0201 | 0.0152 | 0.0058 | +0.0049 [+0.0002, +0.0095] (worse) |
| light distance | 0.0363 | 0.0131 | 0.0039 | +0.0232 [+0.0126, +0.0326] (worse) |
| mean picture PSNR (dB) | 21.70 | 24.22 | 34.07 | -2.52 [-3.79, -1.29] (worse) |

Per setting (learned / coarse):

| setting (intensity, wind, turbulence) | motion ratio | coverage distance | light distance | 3D spectrum distance | mean picture PSNR (dB) |
|---|---|---|---|---|---|
| 0.60, 0.06, 0.65 | 0.63 / 0.47 | 0.0347 / 0.0227 | 0.0584 / 0.0210 | 0.525 / 0.627 | 20.18 / 23.64 |
| 0.40, 0.94, 0.40 | 0.71 / 0.57 | 0.0221 / 0.0108 | 0.0499 / 0.0138 | 0.505 / 0.716 | 20.53 / 25.40 |
| 0.75, 0.25, 0.30 | 0.48 / 0.52 | 0.0127 / 0.0170 | 0.0359 / 0.0125 | 0.525 / 0.679 | 22.04 / 23.54 |
| 0.30, 0.45, 0.80 | 0.60 / 0.58 | 0.0258 / 0.0224 | 0.0191 / 0.0186 | 0.470 / 0.673 | 20.43 / 21.34 |
| 0.50, 0.70, 0.15 | 0.59 / 0.57 | 0.0148 / 0.0077 | 0.0314 / 0.0060 | 0.522 / 0.668 | 23.92 / 24.54 |
| 0.85, 0.60, 0.50 | 0.48 / 0.59 | 0.0104 / 0.0105 | 0.0233 / 0.0069 | 0.479 / 0.755 | 23.09 / 26.84 |
