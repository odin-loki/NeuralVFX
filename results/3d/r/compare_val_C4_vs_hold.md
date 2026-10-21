# C4 minus hold (val; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Tracking from the true state (active PSNR, dB; paired over runs):

| measure | horizon | C4 | hold | difference |
|---|---:|---:|---:|---|
| field: temperature | 8 frames | 20.81 | 18.12 | +2.69 [+2.49, +2.88] (better) |
| field: temperature | 1 s | 19.30 | 17.23 | +2.07 [+1.59, +2.54] (better) |
| field: temperature | 2 s | 20.42 | 17.94 | +2.48 [+1.87, +3.09] (better) |
| field: soot | 8 frames | 20.32 | 17.61 | +2.71 [+2.51, +2.89] (better) |
| field: soot | 1 s | 18.82 | 16.72 | +2.09 [+1.62, +2.56] (better) |
| field: soot | 2 s | 19.93 | 17.44 | +2.49 [+1.88, +3.09] (better) |
| picture (4 cameras) | 8 frames | 22.63 | 14.27 | +8.36 [+8.11, +8.64] (better) |
| picture (4 cameras) | 1 s | 17.83 | 10.31 | +7.52 [+7.11, +7.91] (better) |
| picture (4 cameras) | 2 s | 16.15 | 10.71 | +5.44 [+4.92, +5.90] (better) |

(15 runs)

