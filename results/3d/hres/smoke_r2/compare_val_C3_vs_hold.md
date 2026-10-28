# C3 minus hold (val; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Tracking from the true state (active PSNR, dB; paired over runs):

| measure | horizon | C3 | hold | difference |
|---|---:|---:|---:|---|
| field: temperature | 8 frames | 27.56 | 23.49 | +4.06 [+3.66, +4.46] (better) |
| field: temperature | 1 s | 23.75 | 21.89 | +1.86 [+1.43, +2.23] (better) |
| field: temperature | 2 s | 22.96 | 21.98 | +0.98 [+0.43, +1.58] (better) |
| field: soot | 8 frames | 27.10 | 23.07 | +4.04 [+3.64, +4.43] (better) |
| field: soot | 1 s | 23.25 | 21.47 | +1.78 [+1.33, +2.16] (better) |
| field: soot | 2 s | 22.49 | 21.57 | +0.92 [+0.35, +1.54] (better) |
| picture (4 cameras) | 8 frames | 27.20 | 17.45 | +9.75 [+9.15, +10.30] (better) |
| picture (4 cameras) | 1 s | 20.04 | 12.96 | +7.08 [+6.57, +7.52] (better) |
| picture (4 cameras) | 2 s | 17.28 | 12.31 | +4.97 [+4.12, +5.78] (better) |

(15 runs)

