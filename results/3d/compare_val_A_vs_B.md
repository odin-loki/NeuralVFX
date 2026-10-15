# A minus B (val; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Tracking from the true state (active PSNR, dB; paired over runs):

| measure | horizon | A | B | difference |
|---|---:|---:|---:|---|
| field: temperature | 8 frames | 22.54 | 22.26 | +0.28 [+0.19, +0.37] (better) |
| field: temperature | 1 s | 19.57 | 19.31 | +0.26 [+0.12, +0.39] (better) |
| field: temperature | 2 s | 18.62 | 18.68 | -0.06 [-0.35, +0.18] (tie) |
| field: soot | 8 frames | 27.38 | 27.20 | +0.17 [+0.10, +0.25] (better) |
| field: soot | 1 s | 25.03 | 24.84 | +0.19 [+0.07, +0.31] (better) |
| field: soot | 2 s | 24.14 | 24.21 | -0.07 [-0.27, +0.11] (tie) |
| picture (4 cameras) | 8 frames | 21.32 | 20.81 | +0.51 [+0.29, +0.76] (better) |
| picture (4 cameras) | 1 s | 17.37 | 17.08 | +0.29 [+0.01, +0.58] (better) |
| picture (4 cameras) | 2 s | 17.20 | 17.51 | -0.31 [-0.84, +0.10] (tie) |

(12 runs)

Endless statistics against real runs (4 settings, 2 runs pooled per setting; lower is better but for PSNR and the motion ratio, whose ideal is 1):

| statistic | A | B | difference |
|---|---:|---:|---|
| 3D spectrum distance | 0.842 | 0.358 | +0.484 [-0.009, +0.977] (tie) |
| \|ln motion ratio\| | 0.890 | 0.491 | +0.399 [+0.058, +0.740] (worse) |
| motion ratio | 1.730 | 0.831 | +0.899 [-0.035, +1.834] (context) |
| coverage distance | 0.0262 | 0.0054 | +0.0208 [-0.0000, +0.0417] (tie) |
| emission distance | 0.0270 | 0.0049 | +0.0221 [+0.0002, +0.0439] (worse) |
| mean picture PSNR (dB) | 22.62 | 27.50 | -4.87 [-10.09, +0.34] (tie) |

Per setting (A / B):

| setting (intensity, wind, turbulence) | motion ratio | coverage distance | emission distance | 3D spectrum distance | mean picture PSNR (dB) |
|---|---|---|---|---|---|
| 0.35, 0.30, 0.40 | 3.08 / 1.49 | 0.0363 / 0.0042 | 0.0410 / 0.0078 | 1.369 / 0.564 | 18.38 / 26.77 |
| 0.65, 0.70, 0.60 | 0.59 / 0.66 | 0.0038 / 0.0037 | 0.0034 / 0.0033 | 0.283 / 0.304 | 27.52 / 26.74 |
| 0.50, 0.15, 0.75 | 2.83 / 0.75 | 0.0543 / 0.0031 | 0.0590 / 0.0044 | 1.417 / 0.268 | 17.30 / 29.09 |
| 0.80, 0.55, 0.25 | 0.42 / 0.42 | 0.0105 / 0.0106 | 0.0044 / 0.0040 | 0.298 / 0.295 | 27.30 / 27.39 |
