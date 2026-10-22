# A minus coarse (val; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Tracking from the true state (active PSNR, dB; paired over runs):

| measure | horizon | A | coarse | difference |
|---|---:|---:|---:|---|
| field: temperature | 8 frames | 22.54 | 20.41 | +2.13 [+1.89, +2.36] (better) |
| field: temperature | 1 s | 19.57 | 18.31 | +1.26 [+0.33, +2.14] (better) |
| field: temperature | 2 s | 18.62 | 16.85 | +1.77 [+1.23, +2.33] (better) |
| field: soot | 8 frames | 27.38 | 24.54 | +2.84 [+2.39, +3.32] (better) |
| field: soot | 1 s | 25.03 | 19.56 | +5.47 [+4.71, +6.21] (better) |
| field: soot | 2 s | 24.14 | 17.57 | +6.57 [+5.82, +7.34] (better) |
| picture (4 cameras) | 8 frames | 21.32 | 18.45 | +2.87 [+2.43, +3.28] (better) |
| picture (4 cameras) | 1 s | 17.37 | 14.78 | +2.60 [+1.02, +4.14] (better) |
| picture (4 cameras) | 2 s | 17.20 | 14.67 | +2.53 [+1.38, +3.63] (better) |

(12 runs)

Endless statistics against real runs (4 settings, 2 runs pooled per setting; lower is better but for PSNR and the motion ratio, whose ideal is 1):

| statistic | A | coarse | difference |
|---|---:|---:|---|
| 3D spectrum distance | 0.842 | 0.778 | +0.064 [-0.454, +0.582] (tie) |
| \|ln motion ratio\| | 0.890 | 0.233 | +0.657 [+0.527, +0.786] (worse) |
| motion ratio | 1.730 | 1.099 | +0.631 [-0.345, +1.606] (context) |
| coverage distance | 0.0262 | 0.0037 | +0.0225 [+0.0011, +0.0440] (worse) |
| emission distance | 0.0270 | 0.0136 | +0.0133 [-0.0115, +0.0382] (tie) |
| mean picture PSNR (dB) | 22.62 | 23.55 | -0.93 [-6.46, +4.61] (tie) |

Per setting (A / coarse):

| setting (intensity, wind, turbulence) | motion ratio | coverage distance | emission distance | 3D spectrum distance | mean picture PSNR (dB) |
|---|---|---|---|---|---|
| 0.35, 0.30, 0.40 | 3.08 / 1.44 | 0.0363 / 0.0011 | 0.0410 / 0.0127 | 1.369 / 0.810 | 18.38 / 25.39 |
| 0.65, 0.70, 0.60 | 0.59 / 0.96 | 0.0038 / 0.0038 | 0.0034 / 0.0117 | 0.283 / 0.771 | 27.52 / 23.07 |
| 0.50, 0.15, 0.75 | 2.83 / 1.25 | 0.0543 / 0.0016 | 0.0590 / 0.0110 | 1.417 / 0.812 | 17.30 / 23.21 |
| 0.80, 0.55, 0.25 | 0.42 / 0.74 | 0.0105 / 0.0083 | 0.0044 / 0.0191 | 0.298 / 0.719 | 27.30 / 22.52 |
