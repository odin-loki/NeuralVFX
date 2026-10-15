# hold minus coarse (test; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Tracking from the true state (active PSNR, dB; paired over runs):

| measure | horizon | hold | coarse | difference |
|---|---:|---:|---:|---|
| field: temperature | 8 frames | 21.01 | 20.58 | +0.43 [-0.07, +0.95] (tie) |
| field: temperature | 1 s | 20.09 | 18.23 | +1.86 [+1.36, +2.36] (better) |
| field: temperature | 2 s | 19.83 | 17.77 | +2.06 [+1.13, +3.03] (better) |
| field: soot | 8 frames | 25.27 | 24.78 | +0.49 [+0.09, +0.87] (better) |
| field: soot | 1 s | 24.92 | 19.22 | +5.69 [+4.94, +6.40] (better) |
| field: soot | 2 s | 24.71 | 17.79 | +6.92 [+6.21, +7.71] (better) |
| picture (4 cameras) | 8 frames | 19.32 | 17.90 | +1.42 [+0.42, +2.48] (better) |
| picture (4 cameras) | 1 s | 17.68 | 14.78 | +2.90 [+2.12, +3.71] (better) |
| picture (4 cameras) | 2 s | 18.04 | 14.89 | +3.15 [+1.90, +4.44] (better) |

(18 runs)

