# Effect manager: quality of the levels of detail (val)

Written by `nvfx_lod val` (docs/MANAGER.md §5). Every number is a mean over the effects' ten settings; intervals are paired bootstrap 95% intervals over settings (10,000 resamples). Active PSNR per frame in dB.

## Rate: frames between keyframes against the full-rate run at the same size

| size | rate | interpolation | active PSNR per frame | spectrum L1 | motion ratio | mean-frame PSNR |
|---:|---:|---|---:|---:|---:|---:|
| 32 | 15.0 Hz | hold | 29.84 | 0.002 | 0.962 | 68.46 |
| 32 | 15.0 Hz | crossfade | 39.81 | 0.019 | 0.963 | 69.81 |
| 32 | 15.0 Hz | motion | 31.89 | 0.080 | 1.694 | 53.00 |
| 32 | 10.0 Hz | hold | 27.20 | 0.003 | 0.934 | 64.85 |
| 32 | 10.0 Hz | crossfade | 36.19 | 0.041 | 0.936 | 71.19 |
| 32 | 10.0 Hz | motion | 30.46 | 0.127 | 1.697 | 48.31 |
| 32 | 7.5 Hz | hold | 25.34 | 0.005 | 0.896 | 60.09 |
| 32 | 7.5 Hz | crossfade | 33.33 | 0.058 | 0.900 | 64.06 |
| 32 | 7.5 Hz | motion | 29.36 | 0.159 | 1.777 | 45.47 |
| 64 | 15.0 Hz | hold | 28.52 | 0.002 | 0.962 | 67.55 |
| 64 | 15.0 Hz | crossfade | 38.04 | 0.022 | 0.963 | 68.76 |
| 64 | 15.0 Hz | motion | 31.61 | 0.087 | 1.471 | 53.58 |
| 64 | 10.0 Hz | hold | 25.87 | 0.003 | 0.926 | 63.35 |
| 64 | 10.0 Hz | crossfade | 34.24 | 0.043 | 0.929 | 68.06 |
| 64 | 10.0 Hz | motion | 29.71 | 0.139 | 1.585 | 48.39 |
| 64 | 7.5 Hz | hold | 24.05 | 0.004 | 0.884 | 58.96 |
| 64 | 7.5 Hz | crossfade | 31.49 | 0.060 | 0.888 | 62.18 |
| 64 | 7.5 Hz | motion | 28.23 | 0.176 | 1.695 | 45.25 |
| 128 | 15.0 Hz | hold | 26.91 | 0.001 | 0.944 | 66.20 |
| 128 | 15.0 Hz | crossfade | 34.83 | 0.037 | 0.946 | 67.23 |
| 128 | 15.0 Hz | motion | 32.28 | 0.095 | 1.340 | 55.74 |
| 128 | 10.0 Hz | hold | 24.41 | 0.002 | 0.892 | 61.69 |
| 128 | 10.0 Hz | crossfade | 31.32 | 0.064 | 0.894 | 64.79 |
| 128 | 10.0 Hz | motion | 30.05 | 0.148 | 1.470 | 50.02 |
| 128 | 7.5 Hz | hold | 22.73 | 0.004 | 0.837 | 56.85 |
| 128 | 7.5 Hz | crossfade | 28.96 | 0.083 | 0.841 | 58.88 |
| 128 | 7.5 Hz | motion | 28.41 | 0.183 | 1.544 | 46.46 |

## The comparisons of the rule (docs/MANAGER.md §5): mean over 64 and 128 px at 15, 10 and 7.5 Hz

| comparison | effect | active PSNR per frame | motion error abs(ln ratio) | spectrum L1 |
|---|---|---|---|---|
| motion - crossfade | fire | -1.10 [-2.05, -0.30] | +0.009 [-0.019, +0.037] (tie) | +0.096 [+0.087, +0.105] |
| motion - crossfade | smoke | -0.68 [-1.34, -0.12] | +0.013 [-0.001, +0.027] (tie) | +0.076 [+0.067, +0.086] |
| motion - crossfade | explosion | +2.19 [+1.21, +3.03] | -0.019 [-0.025, -0.013] | +0.009 [+0.000, +0.018] |
| motion - crossfade | steam | -1.46 [-2.61, -0.45] | +0.020 [-0.009, +0.053] (tie) | +0.081 [+0.071, +0.091] |
| motion - crossfade | magic | -14.46 [-15.28, -13.67] | +1.015 [+0.912, +1.121] | +0.171 [+0.163, +0.178] |
| motion - crossfade | pooled | -3.10 [-4.85, -1.49] | +0.208 [+0.101, +0.326] | +0.087 [+0.072, +0.102] |
| crossfade - hold | fire | +6.22 [+5.87, +6.58] | -0.003 [-0.005, -0.002] | +0.059 [+0.054, +0.063] |
| crossfade - hold | smoke | +6.77 [+6.49, +7.05] | -0.004 [-0.006, -0.003] | +0.049 [+0.046, +0.051] |
| crossfade - hold | explosion | +10.41 [+10.06, +10.77] | -0.000 [-0.000, +0.000] (tie) | +0.055 [+0.048, +0.061] |
| crossfade - hold | steam | +7.25 [+6.93, +7.60] | -0.003 [-0.004, -0.002] | +0.048 [+0.043, +0.052] |
| crossfade - hold | magic | +8.02 [+7.84, +8.19] | -0.003 [-0.005, -0.002] | +0.034 [+0.033, +0.035] |
| crossfade - hold | pooled | +7.73 [+7.31, +8.17] | -0.003 [-0.003, -0.002] | +0.049 [+0.046, +0.052] |
| motion - hold | fire | +5.12 [+4.47, +5.68] | +0.005 [-0.022, +0.035] (tie) | +0.155 [+0.145, +0.165] |
| motion - hold | smoke | +6.09 [+5.67, +6.49] | +0.009 [-0.006, +0.023] (tie) | +0.125 [+0.115, +0.136] |
| motion - hold | explosion | +12.60 [+11.78, +13.28] | -0.019 [-0.025, -0.013] | +0.063 [+0.051, +0.076] |
| motion - hold | steam | +5.79 [+4.98, +6.50] | +0.017 [-0.012, +0.050] (tie) | +0.129 [+0.119, +0.138] |
| motion - hold | magic | -6.44 [-7.43, -5.48] | +1.012 [+0.908, +1.117] | +0.204 [+0.196, +0.212] |
| motion - hold | pooled | +4.63 [+2.82, +6.34] | +0.205 [+0.098, +0.323] | +0.135 [+0.122, +0.149] |

## Size: 32 and 64 px at 30 Hz against 128 px

| size | shown at | active PSNR per frame | spectrum L1 | motion ratio | mean-frame PSNR |
|---:|---|---:|---:|---:|---:|
| 32 | 32 px (128 px averaged down) | 31.02 | 0.034 | 0.972 | 44.89 |
| 32 | 128 px (bilinear up) | 23.66 | 1.118 | 0.698 | 40.80 |
| 64 | 64 px (128 px averaged down) | 28.70 | 0.069 | 0.931 | 47.21 |
| 64 | 128 px (bilinear up) | 26.66 | 0.669 | 0.805 | 46.36 |

## Proxies: 64 px, 7.5 Hz, snapped controls, another seed; against 128 px averaged to 64 (the floor: another seed at the same controls, 64 px, 30 Hz)

| effect | method | spectrum L1 | motion ratio | mean-frame PSNR | proxy - floor: spectrum L1 | abs(ln motion) | mean-frame PSNR |
|---|---|---:|---:|---:|---|---|---|
| fire | proxy, crossfade | 0.150 | 1.444 | 33.02 | +0.038 [-0.023, +0.104] (tie) | +0.113 [-0.062, +0.298] (tie) | -3.16 [-5.98, -0.66] |
| fire | proxy, motion | 0.240 | 1.300 | 33.07 | +0.128 [+0.066, +0.196] | +0.008 [-0.180, +0.212] (tie) | -3.11 [-5.91, -0.66] |
| fire | floor | 0.112 | 1.290 | 36.18 | | | |
| smoke | proxy, crossfade | 0.144 | 1.218 | 28.39 | -0.003 [-0.035, +0.023] (tie) | -0.025 [-0.141, +0.097] (tie) | -2.73 [-4.64, -0.85] |
| smoke | proxy, motion | 0.103 | 1.201 | 28.55 | -0.044 [-0.084, -0.006] | -0.039 [-0.213, +0.124] (tie) | -2.56 [-4.51, -0.63] |
| smoke | floor | 0.147 | 1.249 | 31.11 | | | |
| explosion | proxy, crossfade | 0.181 | 1.484 | 23.45 | +0.022 [-0.066, +0.117] (tie) | +0.042 [-0.216, +0.305] (tie) | -1.54 [-5.30, +2.21] (tie) |
| explosion | proxy, motion | 0.179 | 1.467 | 23.49 | +0.020 [-0.076, +0.123] (tie) | +0.030 [-0.227, +0.295] (tie) | -1.50 [-5.27, +2.26] (tie) |
| explosion | floor | 0.159 | 1.423 | 25.00 | | | |
| steam | proxy, crossfade | 0.163 | 1.391 | 27.35 | +0.035 [-0.002, +0.067] (tie) | +0.104 [-0.043, +0.254] (tie) | -3.14 [-5.04, -1.41] |
| steam | proxy, motion | 0.158 | 1.368 | 27.37 | +0.030 [-0.021, +0.082] (tie) | +0.087 [+0.019, +0.143] | -3.12 [-5.01, -1.41] |
| steam | floor | 0.128 | 1.254 | 30.49 | | | |
| magic | proxy, crossfade | 0.163 | 1.319 | 27.75 | +0.068 [+0.036, +0.104] | +0.119 [+0.083, +0.154] | -3.19 [-6.12, -0.55] |
| magic | proxy, motion | 0.360 | 3.144 | 27.90 | +0.265 [+0.222, +0.309] | +0.987 [+0.830, +1.138] | -3.04 [-5.95, -0.16] |
| magic | floor | 0.095 | 1.171 | 30.94 | | | |
