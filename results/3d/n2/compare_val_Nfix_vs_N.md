# Nfix minus N (val; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Tracking from the true state (active PSNR, dB; paired over runs):

| measure | horizon | Nfix | N | difference |
|---|---:|---:|---:|---|
| field: temperature | 8 frames | 22.08 | 22.08 | -0.00 [-0.01, -0.00] (worse) |
| field: temperature | 1 s | 19.38 | 19.41 | -0.03 [-0.06, -0.00] (worse) |
| field: temperature | 2 s | 19.06 | 19.09 | -0.03 [-0.07, -0.00] (worse) |
| field: soot | 8 frames | 26.99 | 26.99 | +0.00 [-0.00, +0.01] (tie) |
| field: soot | 1 s | 24.47 | 24.49 | -0.02 [-0.03, -0.00] (worse) |
| field: soot | 2 s | 24.15 | 24.19 | -0.05 [-0.11, -0.01] (worse) |
| picture (4 cameras) | 8 frames | 21.30 | 21.30 | +0.00 [-0.00, +0.00] (tie) |
| picture (4 cameras) | 1 s | 17.81 | 17.81 | -0.01 [-0.02, +0.01] (tie) |
| picture (4 cameras) | 2 s | 18.00 | 18.03 | -0.03 [-0.06, -0.01] (worse) |

(15 runs)

Endless statistics against real runs (5 settings, 2 runs pooled per setting; lower is better but for PSNR and the motion ratio, whose ideal is 1):

| statistic | Nfix | N | difference |
|---|---:|---:|---|
| 3D spectrum distance | 0.353 | 0.774 | -0.422 [-0.818, -0.053] (better) |
| \|ln motion ratio\| | 0.379 | 0.440 | -0.061 [-0.635, +0.476] (tie) |
| motion ratio | 0.789 | 1.588 | -0.799 [-1.577, -0.241] (context) |
| coverage distance | 0.0040 | 0.0285 | -0.0245 [-0.0557, -0.0013] (better) |
| emission distance | 0.0052 | 0.0271 | -0.0219 [-0.0488, -0.0026] (better) |
| mean picture PSNR (dB) | 28.05 | 24.15 | +3.90 [+0.40, +8.37] (better) |

Per setting (Nfix / N):

| setting (intensity, wind, turbulence) | motion ratio | coverage distance | emission distance | 3D spectrum distance | mean picture PSNR (dB) |
|---|---|---|---|---|---|
| 0.60, 0.08, 0.50 | 1.22 / 3.46 | 0.0035 / 0.0832 | 0.0099 / 0.0801 | 0.398 / 1.502 | 26.24 / 14.35 |
| 0.40, 0.92, 0.65 | 0.72 / 0.74 | 0.0013 / 0.0012 | 0.0025 / 0.0025 | 0.365 / 0.366 | 32.11 / 32.29 |
| 0.75, 0.30, 0.35 | 0.57 / 1.15 | 0.0084 / 0.0463 | 0.0050 / 0.0350 | 0.370 / 1.130 | 25.28 / 19.48 |
| 0.50, 0.50, 0.80 | 0.45 / 0.97 | 0.0060 / 0.0076 | 0.0053 / 0.0091 | 0.227 / 0.244 | 27.99 / 27.61 |
| 0.30, 0.65, 0.25 | 0.98 / 1.62 | 0.0008 / 0.0042 | 0.0033 / 0.0091 | 0.404 / 0.628 | 28.62 / 27.05 |
