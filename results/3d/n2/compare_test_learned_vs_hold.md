# learned minus hold (test; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Tracking from the true state (active PSNR, dB; paired over runs):

| measure | horizon | learned | hold | difference |
|---|---:|---:|---:|---|
| field: temperature | 8 frames | 21.90 | 20.41 | +1.49 [+1.15, +1.85] (better) |
| field: temperature | 1 s | 20.80 | 19.94 | +0.85 [+0.12, +1.58] (better) |
| field: temperature | 2 s | 21.11 | 19.72 | +1.39 [+0.60, +2.20] (better) |
| field: soot | 8 frames | 26.53 | 24.74 | +1.79 [+1.49, +2.08] (better) |
| field: soot | 1 s | 25.67 | 24.55 | +1.12 [+0.54, +1.69] (better) |
| field: soot | 2 s | 26.23 | 24.56 | +1.67 [+1.03, +2.34] (better) |
| picture (4 cameras) | 8 frames | 21.19 | 19.67 | +1.52 [+0.81, +2.26] (better) |
| picture (4 cameras) | 1 s | 19.40 | 18.08 | +1.32 [+0.43, +2.28] (better) |
| picture (4 cameras) | 2 s | 19.93 | 18.38 | +1.55 [+0.76, +2.38] (better) |

(18 runs)

