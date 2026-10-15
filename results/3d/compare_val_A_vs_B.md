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
