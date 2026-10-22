# hold minus coarse (val; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Tracking from the true state (active PSNR, dB; paired over runs):

| measure | horizon | hold | coarse | difference |
|---|---:|---:|---:|---|
| field: temperature | 8 frames | 20.84 | 19.55 | +1.29 [+0.69, +2.06] (better) |
| field: temperature | 1 s | 20.36 | 17.42 | +2.93 [+2.39, +3.58] (better) |
| field: temperature | 2 s | 20.15 | 17.50 | +2.65 [+2.00, +3.28] (better) |
| field: soot | 8 frames | 25.33 | 23.82 | +1.51 [+0.82, +2.25] (better) |
| field: soot | 1 s | 25.00 | 19.22 | +5.78 [+5.17, +6.48] (better) |
| field: soot | 2 s | 24.95 | 17.59 | +7.36 [+6.53, +8.25] (better) |
| picture (4 cameras) | 8 frames | 19.94 | 16.38 | +3.57 [+2.33, +5.04] (better) |
| picture (4 cameras) | 1 s | 18.54 | 13.61 | +4.93 [+3.84, +6.02] (better) |
| picture (4 cameras) | 2 s | 18.94 | 14.93 | +4.01 [+3.01, +5.01] (better) |

(15 runs)

