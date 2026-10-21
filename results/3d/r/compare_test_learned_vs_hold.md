# learned minus hold (test; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Tracking from the true state (active PSNR, dB; paired over runs):

| measure | horizon | learned | hold | difference |
|---|---:|---:|---:|---|
| field: temperature | 8 frames | 21.31 | 18.75 | +2.56 [+2.23, +2.87] (better) |
| field: temperature | 1 s | 20.02 | 17.78 | +2.24 [+1.91, +2.58] (better) |
| field: temperature | 2 s | 20.62 | 18.33 | +2.29 [+1.92, +2.65] (better) |
| field: soot | 8 frames | 20.83 | 18.25 | +2.58 [+2.24, +2.89] (better) |
| field: soot | 1 s | 19.56 | 17.29 | +2.27 [+1.93, +2.61] (better) |
| field: soot | 2 s | 20.16 | 17.85 | +2.32 [+1.93, +2.69] (better) |
| picture (4 cameras) | 8 frames | 22.74 | 14.56 | +8.18 [+7.86, +8.49] (better) |
| picture (4 cameras) | 1 s | 17.87 | 10.46 | +7.42 [+7.08, +7.74] (better) |
| picture (4 cameras) | 2 s | 16.39 | 10.75 | +5.64 [+5.30, +6.01] (better) |

(18 runs)

