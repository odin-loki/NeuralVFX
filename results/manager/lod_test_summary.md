# Effect manager: quality of the levels of detail (test)

Written by `nvfx_lod test` (docs/MANAGER.md §5). Every number is a mean over the effects' ten settings; intervals are paired bootstrap 95% intervals over settings (10,000 resamples). Active PSNR per frame in dB.

## Rate: frames between keyframes against the full-rate run at the same size

| size | rate | interpolation | active PSNR per frame | spectrum L1 | motion ratio | mean-frame PSNR |
|---:|---:|---|---:|---:|---:|---:|
| 32 | 15.0 Hz | hold | 30.05 | 0.002 | 0.959 | 68.31 |
| 32 | 15.0 Hz | crossfade | 39.98 | 0.019 | 0.961 | 69.74 |
| 32 | 15.0 Hz | motion | 32.19 | 0.076 | 1.678 | 53.09 |
| 32 | 10.0 Hz | hold | 27.44 | 0.003 | 0.931 | 64.49 |
| 32 | 10.0 Hz | crossfade | 36.46 | 0.040 | 0.933 | 70.99 |
| 32 | 10.0 Hz | motion | 30.73 | 0.122 | 1.679 | 48.39 |
| 32 | 7.5 Hz | hold | 25.56 | 0.006 | 0.892 | 60.00 |
| 32 | 7.5 Hz | crossfade | 33.60 | 0.056 | 0.895 | 63.94 |
| 32 | 7.5 Hz | motion | 29.57 | 0.153 | 1.754 | 45.50 |
| 64 | 15.0 Hz | hold | 28.68 | 0.002 | 0.961 | 67.40 |
| 64 | 15.0 Hz | crossfade | 38.19 | 0.021 | 0.963 | 68.58 |
| 64 | 15.0 Hz | motion | 31.65 | 0.083 | 1.468 | 53.67 |
| 64 | 10.0 Hz | hold | 26.05 | 0.003 | 0.925 | 63.18 |
| 64 | 10.0 Hz | crossfade | 34.43 | 0.041 | 0.927 | 68.05 |
| 64 | 10.0 Hz | motion | 29.71 | 0.134 | 1.583 | 48.45 |
| 64 | 7.5 Hz | hold | 24.22 | 0.004 | 0.883 | 58.78 |
| 64 | 7.5 Hz | crossfade | 31.67 | 0.058 | 0.886 | 62.10 |
| 64 | 7.5 Hz | motion | 28.19 | 0.171 | 1.694 | 45.26 |
| 128 | 15.0 Hz | hold | 27.05 | 0.001 | 0.944 | 66.27 |
| 128 | 15.0 Hz | crossfade | 34.99 | 0.035 | 0.945 | 67.38 |
| 128 | 15.0 Hz | motion | 31.93 | 0.097 | 1.341 | 55.52 |
| 128 | 10.0 Hz | hold | 24.57 | 0.002 | 0.891 | 61.52 |
| 128 | 10.0 Hz | crossfade | 31.49 | 0.062 | 0.893 | 64.77 |
| 128 | 10.0 Hz | motion | 29.70 | 0.152 | 1.473 | 49.84 |
| 128 | 7.5 Hz | hold | 22.88 | 0.004 | 0.836 | 56.77 |
| 128 | 7.5 Hz | crossfade | 29.11 | 0.082 | 0.839 | 58.93 |
| 128 | 7.5 Hz | motion | 28.05 | 0.190 | 1.550 | 46.33 |

## The comparisons of the rule (docs/MANAGER.md §5): mean over 64 and 128 px at 15, 10 and 7.5 Hz

| comparison | effect | active PSNR per frame | motion error abs(ln ratio) | spectrum L1 |
|---|---|---|---|---|
| motion - crossfade | fire | -1.73 [-2.51, -0.87] | +0.023 [-0.009, +0.053] (tie) | +0.098 [+0.087, +0.108] |
| motion - crossfade | smoke | -0.88 [-1.83, -0.04] | +0.020 [-0.002, +0.045] (tie) | +0.075 [+0.067, +0.084] |
| motion - crossfade | explosion | +1.20 [-0.21, +2.68] (tie) | -0.014 [-0.025, -0.005] | +0.022 [+0.006, +0.037] |
| motion - crossfade | steam | -1.41 [-2.71, -0.02] | +0.013 [-0.016, +0.041] (tie) | +0.076 [+0.069, +0.083] |
| motion - crossfade | magic | -14.38 [-15.03, -13.70] | +1.012 [+0.922, +1.101] | +0.170 [+0.163, +0.175] |
| motion - crossfade | pooled | -3.44 [-5.11, -1.87] | +0.211 [+0.105, +0.327] | +0.088 [+0.074, +0.102] |
| crossfade - hold | fire | +6.25 [+5.83, +6.64] | -0.004 [-0.005, -0.002] | +0.054 [+0.048, +0.060] |
| crossfade - hold | smoke | +6.78 [+6.42, +7.13] | -0.003 [-0.004, -0.001] | +0.049 [+0.044, +0.054] |
| crossfade - hold | explosion | +10.47 [+9.82, +11.08] | -0.000 [-0.000, +0.000] (tie) | +0.053 [+0.048, +0.057] |
| crossfade - hold | steam | +7.12 [+6.69, +7.50] | -0.003 [-0.004, -0.002] | +0.047 [+0.041, +0.053] |
| crossfade - hold | magic | +8.07 [+7.90, +8.25] | -0.004 [-0.005, -0.003] | +0.034 [+0.033, +0.035] |
| crossfade - hold | pooled | +7.74 [+7.31, +8.21] | -0.003 [-0.003, -0.002] | +0.047 [+0.045, +0.050] |
| motion - hold | fire | +4.52 [+4.01, +5.04] | +0.019 [-0.012, +0.049] (tie) | +0.152 [+0.138, +0.164] |
| motion - hold | smoke | +5.90 [+5.25, +6.45] | +0.018 [-0.004, +0.042] (tie) | +0.124 [+0.114, +0.135] |
| motion - hold | explosion | +11.67 [+10.81, +12.54] | -0.014 [-0.025, -0.005] | +0.074 [+0.056, +0.093] |
| motion - hold | steam | +5.71 [+4.74, +6.71] | +0.010 [-0.020, +0.039] (tie) | +0.122 [+0.114, +0.131] |
| motion - hold | magic | -6.31 [-7.12, -5.46] | +1.009 [+0.919, +1.097] | +0.204 [+0.197, +0.210] |
| motion - hold | pooled | +4.30 [+2.59, +5.91] | +0.208 [+0.103, +0.324] | +0.135 [+0.122, +0.148] |

## Size: 32 and 64 px at 30 Hz against 128 px

| size | shown at | active PSNR per frame | spectrum L1 | motion ratio | mean-frame PSNR |
|---:|---|---:|---:|---:|---:|
| 32 | 32 px (128 px averaged down) | 30.69 | 0.034 | 0.975 | 44.41 |
| 32 | 128 px (bilinear up) | 23.40 | 1.170 | 0.694 | 40.57 |
| 64 | 64 px (128 px averaged down) | 28.28 | 0.068 | 0.930 | 47.10 |
| 64 | 128 px (bilinear up) | 26.29 | 0.708 | 0.801 | 46.04 |

## Proxies: 64 px, 7.5 Hz, snapped controls, another seed; against 128 px averaged to 64 (the floor: another seed at the same controls, 64 px, 30 Hz)

| effect | method | spectrum L1 | motion ratio | mean-frame PSNR | proxy - floor: spectrum L1 | abs(ln motion) | mean-frame PSNR |
|---|---|---:|---:|---:|---|---|---|
| fire | proxy, crossfade | 0.122 | 1.329 | 33.49 | -0.005 [-0.062, +0.059] (tie) | +0.028 [-0.222, +0.274] (tie) | -0.89 [-3.39, +1.89] (tie) |
| fire | proxy, motion | 0.165 | 1.443 | 33.49 | +0.037 [-0.065, +0.146] (tie) | +0.110 [-0.107, +0.330] (tie) | -0.89 [-3.35, +1.86] (tie) |
| fire | floor | 0.127 | 1.292 | 34.39 | | | |
| smoke | proxy, crossfade | 0.145 | 1.364 | 28.91 | +0.012 [-0.022, +0.046] (tie) | +0.102 [-0.029, +0.239] (tie) | -0.07 [-1.43, +1.08] (tie) |
| smoke | proxy, motion | 0.112 | 1.179 | 29.08 | -0.020 [-0.069, +0.035] (tie) | -0.044 [-0.201, +0.118] (tie) | +0.10 [-1.30, +1.24] (tie) |
| smoke | floor | 0.133 | 1.232 | 28.98 | | | |
| explosion | proxy, crossfade | 0.154 | 1.354 | 24.23 | +0.062 [-0.007, +0.131] (tie) | +0.131 [-0.176, +0.426] (tie) | -2.41 [-5.99, +1.56] (tie) |
| explosion | proxy, motion | 0.132 | 1.356 | 24.28 | +0.040 [-0.032, +0.115] (tie) | +0.132 [-0.174, +0.438] (tie) | -2.35 [-5.94, +1.63] (tie) |
| explosion | floor | 0.092 | 1.188 | 26.63 | | | |
| steam | proxy, crossfade | 0.151 | 1.311 | 27.95 | -0.004 [-0.027, +0.022] (tie) | +0.102 [-0.020, +0.225] (tie) | -2.59 [-4.36, -0.67] |
| steam | proxy, motion | 0.125 | 1.253 | 28.01 | -0.031 [-0.062, +0.004] (tie) | +0.056 [-0.010, +0.123] (tie) | -2.53 [-4.31, -0.61] |
| steam | floor | 0.155 | 1.185 | 30.55 | | | |
| magic | proxy, crossfade | 0.146 | 1.263 | 28.06 | +0.063 [+0.040, +0.085] | +0.108 [+0.052, +0.163] | -3.29 [-5.52, -0.93] |
| magic | proxy, motion | 0.339 | 3.476 | 28.32 | +0.256 [+0.208, +0.301] | +1.120 [+1.009, +1.219] | -3.02 [-5.32, -0.76] |
| magic | floor | 0.083 | 1.134 | 31.35 | | | |
