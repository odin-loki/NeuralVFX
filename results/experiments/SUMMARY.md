# Experiment summary (generated)

Status: **generated** by `nvfx_experiment report` from the CSVs in this folder; rerun it rather than editing. 95% paired bootstrap intervals in square brackets (10,000 resamples over clips); an interval covering zero is reported as a tie. PSNR in dB; "active" = PSNR over pixels visible in either clip.

## A. Compression: one model per clip against flipbooks of the same clip

Means over 12 clips (fire, smoke, explosion; 128 x 128, 64 frames). ms = median per frame through the runtime on one AVX2 core, measured on a quiet machine (timing step).

Spectrum = mean |log power difference| of the radially averaged luminance spectrum against the reference (0 = same sharpness; blur raises it); motion = frame-to-frame change relative to the reference (1 = same).

| method | family | KB | PSNR | active PSNR | SSIM | temporal PSNR | flicker | spectrum | motion | ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bc3 4f 32px | flipbook_bc3 | 4.0 | 22.16 | 17.25 | 0.7967 | 29.48 | 0.01 | 1.096 | 0.37 | - |
| bc3 8f 32px | flipbook_bc3 | 8.0 | 25.16 | 19.98 | 0.8506 | 29.97 | 0.04 | 1.043 | 0.52 | - |
| raw 4f 32px | flipbook_raw | 16.0 | 22.21 | 17.21 | 0.8007 | 29.49 | 0.01 | 1.095 | 0.37 | - |
| bc3 16f 32px | flipbook_bc3 | 16.0 | 27.24 | 21.89 | 0.8831 | 30.80 | 0.09 | 1.021 | 0.63 | - |
| bc3 4f 64px | flipbook_bc3 | 16.0 | 22.17 | 16.88 | 0.8191 | 29.51 | 0.01 | 0.748 | 0.40 | - |
| bc3 4f 64px +mv16 | flipbook_mv | 18.0 | 22.84 | 17.29 | 0.8493 | 29.69 | 0.04 | 0.584 | 0.49 | - |
| raw 8f 32px | flipbook_raw | 32.0 | 25.29 | 20.01 | 0.8564 | 29.98 | 0.04 | 1.042 | 0.52 | - |
| bc3 32f 32px | flipbook_bc3 | 32.0 | 28.21 | 22.76 | 0.8973 | 31.67 | 0.17 | 0.986 | 0.71 | - |
| bc3 8f 64px | flipbook_bc3 | 32.0 | 25.88 | 20.27 | 0.8877 | 30.09 | 0.05 | 0.709 | 0.57 | - |
| bc3 8f 64px +mv16 | flipbook_mv | 36.0 | 28.65 | 22.79 | 0.9282 | 31.60 | 0.15 | 0.567 | 0.71 | - |
| raw 16f 32px | flipbook_raw | 64.0 | 27.49 | 22.02 | 0.8906 | 30.87 | 0.09 | 1.016 | 0.62 | - |
| raw 4f 64px | flipbook_raw | 64.0 | 22.19 | 16.87 | 0.8215 | 29.51 | 0.01 | 0.788 | 0.40 | - |
| bc3 64f 32px | flipbook_bc3 | 64.0 | 28.49 | 23.03 | 0.9016 | 31.86 | 0.41 | 0.963 | 0.79 | - |
| bc3 4f 128px | flipbook_bc3 | 64.0 | 21.89 | 16.36 | 0.8234 | 29.52 | 0.01 | 0.152 | 0.42 | - |
| bc3 16f 64px | flipbook_bc3 | 64.0 | 29.61 | 23.76 | 0.9352 | 31.34 | 0.15 | 0.688 | 0.71 | - |
| conv_s|8 | neural_conv | 69.0 | 35.79 | 29.45 | 0.9658 | 36.18 | 0.63 | 0.262 | 0.94 | 0.567 |
| bc3 4f 128px +mv32 | flipbook_mv | 72.0 | 22.25 | 16.53 | 0.8453 | 29.55 | 0.05 | 0.221 | 0.49 | - |
| bc3 16f 64px +mv16 | flipbook_mv | 72.0 | 32.71 | 26.69 | 0.9618 | 34.50 | 0.32 | 0.574 | 0.83 | - |
| grid_s|8 | neural_grid | 73.0 | 34.32 | 28.07 | 0.9627 | 34.97 | 0.64 | 0.171 | 0.95 | 0.383 |
| bc3 8f 128px | flipbook_bc3 | 128.0 | 25.82 | 19.91 | 0.8990 | 30.14 | 0.06 | 0.141 | 0.61 | - |
| bc3 32f 64px | flipbook_bc3 | 128.0 | 32.38 | 26.39 | 0.9593 | 33.46 | 0.32 | 0.647 | 0.81 | - |
| raw 32f 32px | flipbook_raw | 128.0 | 28.56 | 22.98 | 0.9065 | 31.93 | 0.14 | 0.979 | 0.67 | - |
| raw 8f 64px | flipbook_raw | 128.0 | 25.96 | 20.30 | 0.8914 | 30.11 | 0.05 | 0.747 | 0.57 | - |
| grid_m|8 | neural_grid | 131.5 | 39.10 | 32.62 | 0.9873 | 39.83 | 0.87 | 0.096 | 0.98 | 1.060 |
| conv_s|16 | neural_conv | 132.0 | 35.79 | 29.45 | 0.9658 | 36.18 | 0.63 | 0.262 | 0.94 | 0.565 |
| conv_m|8 | neural_conv | 142.0 | 37.72 | 31.33 | 0.9733 | 37.61 | 0.67 | 0.191 | 0.95 | 0.760 |
| bc3 8f 128px +mv32 | flipbook_mv | 144.0 | 28.07 | 21.99 | 0.9337 | 31.19 | 0.24 | 0.189 | 0.74 | - |
| grid_s|16 | neural_grid | 144.5 | 34.33 | 28.07 | 0.9631 | 34.97 | 0.64 | 0.171 | 0.95 | 0.394 |
| bc3 16f 128px | flipbook_bc3 | 256.0 | 30.67 | 24.44 | 0.9548 | 31.57 | 0.23 | 0.124 | 0.79 | - |
| bc3 64f 64px | flipbook_bc3 | 256.0 | 33.63 | 27.59 | 0.9671 | 35.60 | 0.56 | 0.602 | 0.89 | - |
| raw 64f 32px | flipbook_raw | 256.0 | 28.91 | 23.30 | 0.9119 | 32.70 | 0.14 | 0.958 | 0.69 | - |
| raw 16f 64px | flipbook_raw | 256.0 | 29.87 | 23.97 | 0.9406 | 31.40 | 0.15 | 0.721 | 0.71 | - |
| grid_m|16 | neural_grid | 259.0 | 39.12 | 32.64 | 0.9874 | 39.85 | 0.87 | 0.096 | 0.98 | 1.105 |
| grid_mt|8 | neural_grid | 260.0 | 40.26 | 33.76 | 0.9897 | 41.22 | 0.96 | 0.089 | 1.01 | 1.091 |
| conv_m|16 | neural_conv | 268.0 | 37.73 | 31.33 | 0.9734 | 37.61 | 0.67 | 0.191 | 0.95 | 0.790 |
| bc3 16f 128px +mv32 | flipbook_mv | 288.0 | 35.61 | 29.25 | 0.9811 | 35.20 | 0.63 | 0.123 | 0.93 | - |
| grid_l|8 | neural_grid | 291.5 | 42.73 | 36.23 | 0.9930 | 41.42 | 0.90 | 0.039 | 0.98 | 1.091 |
| raw 32f 64px | flipbook_raw | 512.0 | 32.99 | 26.94 | 0.9663 | 33.77 | 0.31 | 0.677 | 0.80 | - |
| bc3 32f 128px | flipbook_bc3 | 512.0 | 36.36 | 29.92 | 0.9838 | 34.50 | 0.61 | 0.103 | 0.94 | - |
| grid_mt|16 | neural_grid | 515.0 | 40.28 | 33.77 | 0.9897 | 41.24 | 0.96 | 0.089 | 1.01 | 1.100 |
| grid_l|16 | neural_grid | 579.0 | 42.78 | 36.28 | 0.9931 | 41.44 | 0.90 | 0.039 | 0.98 | 1.124 |
| bc3 64f 128px | flipbook_bc3 | 1024.0 | 41.02 | 34.48 | 0.9924 | 39.39 | 1.23 | 0.102 | 1.03 | - |
| raw 64f 64px | flipbook_raw | 1024.0 | 34.62 | 28.50 | 0.9756 | 37.30 | 0.39 | 0.636 | 0.84 | - |

### Matched memory

Within each budget, the neural configuration and the flipbook configuration with the best mean active PSNR (chosen on the same clips they are scored on, which favours neither), compared clip by clip.

| budget | neural | KB | flipbook | KB | delta active PSNR | delta PSNR | delta SSIM |
|---|---|---:|---|---:|---:|---:|---:|
| 128 KB | conv_s|8 | 69 | bc3 16f 64px +mv16 | 72 | +2.75 [+2.28, +3.29] | +3.08 [+2.59, +3.61] | +0.0040 [-0.0010, +0.0090] (tie) |
| 160 KB | grid_m|8 | 132 | bc3 16f 64px +mv16 | 72 | +5.93 [+4.99, +7.01] | +6.39 [+5.36, +7.54] | +0.0255 [+0.0202, +0.0307] |
| 256 KB | grid_mt|8 | 260 | bc3 64f 64px | 256 | +6.16 [+4.82, +7.50] | +6.63 [+5.15, +8.11] | +0.0226 [+0.0182, +0.0271] |
| 320 KB | grid_l|8 | 292 | bc3 16f 128px +mv32 | 288 | +6.98 [+5.70, +8.22] | +7.12 [+5.81, +8.38] | +0.0119 [+0.0083, +0.0158] |
| 512 KB | grid_l|8 | 292 | bc3 32f 128px | 512 | +6.30 [+4.87, +7.70] | +6.37 [+4.92, +7.77] | +0.0092 [+0.0065, +0.0123] |

### Memory at equal quality

For each neural configuration: the flipbook memory needed for the same mean active PSNR (log-linear interpolation along the best-flipbook-at-each-size envelope; ">" when no flipbook up to 1 MB reaches it).

| neural | KB | active PSNR | flipbook KB for equal quality | ratio |
|---|---:|---:|---:|---:|
| conv_s|8 | 69 | 29.45 | 512 | 7.4x |
| grid_s|8 | 73 | 28.07 | 265 | 3.6x |
| grid_m|8 | 132 | 32.62 | 772 | 5.9x |
| conv_s|16 | 132 | 29.45 | 512 | 3.9x |
| conv_m|8 | 142 | 31.33 | 634 | 4.5x |
| grid_s|16 | 144 | 28.07 | 265 | 1.8x |
| grid_m|16 | 259 | 32.64 | 774 | 3.0x |
| grid_mt|8 | 260 | 33.76 | 917 | 3.5x |
| conv_m|16 | 268 | 31.33 | 634 | 2.4x |
| grid_l|8 | 292 | 36.23 | > 1024 | > 3.5x |
| grid_mt|16 | 515 | 33.77 | 920 | 1.8x |
| grid_l|16 | 579 | 36.28 | > 1024 | > 1.8x |

### By effect (grid_m 8-bit against BC3 16 frames 128 px and BC3 all frames)

| effect | grid_m 8-bit active | bc3 16f 128px active | bc3 64f 128px active |
|---|---:|---:|---:|
| fire | 31.47 | 22.77 | 30.44 |
| smoke | 31.86 | 26.13 | 36.48 |
| explosion | 34.53 | 24.43 | 36.51 |

### Frame interpolation (only even frames available; odd frames scored)

| method | KB | mean active PSNR on odd frames | method minus neural |
|---|---:|---:|---:|
| bc3 even frames, 128px | 512 | 27.99 | -1.54 [-2.93, -0.29] |
| bc3 even frames, 128px +mv32 | 576 | 33.45 | +3.92 [+2.40, +5.22] |
| grid_m 8-bit (trained on the even frames) | 132 | 29.53 | - |
| raw even frames, 128px | 2048 | 28.66 | -0.87 [-2.22, +0.36] (tie) |

## B. Controls: held-out control settings

One model per effect trained on 45 settings (3 intensity x 5 wind x 3 turbulence, seed 1); scored on 10 off-grid settings per effect. Baselines are libraries of 45 BC3 flipbooks (one per training setting).

| method | KB per effect | mean active PSNR | mean PSNR | mean SSIM | spectrum | motion | ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| blend2 | 46080 | 16.13 | 21.81 | 0.7531 | 0.220 | 1.05 | - |
| grid_k16 | 2063 | 17.28 | 23.42 | 0.7759 | 0.239 | 0.99 | 1.836 |
| grid_k8 | 1031 | 17.12 | 23.37 | 0.7750 | 0.218 | 0.99 | 1.030 |
| nearest | 46080 | 14.64 | 20.73 | 0.7687 | 0.184 | 1.14 | - |
| oracle | 184320 | 15.09 | - | - | - | - | - |

| comparison (active PSNR, paired over 30 held-out settings) | difference |
|---|---:|
| grid_k8 - nearest | +2.49 [+2.22, +2.77] |
| grid_k8 - blend2 | +0.99 [+0.65, +1.36] |
| grid_k8 - oracle | +2.03 [+1.72, +2.37] |
| grid_k16 - nearest | +2.64 [+2.32, +2.98] |
| grid_k16 - blend2 | +1.15 [+0.72, +1.60] |
| grid_k16 - oracle | +2.19 [+1.81, +2.59] |

| effect | grid_k8 | grid_k16 | nearest | blend2 | oracle |
|---|---:|---:|---:|---:|---:|
| fire | 19.16 | 19.40 | 16.70 | 18.23 | 17.14 |
| smoke | 17.06 | 16.86 | 14.92 | 16.96 | 15.70 |
| explosion | 15.15 | 15.57 | 12.29 | 13.21 | 12.43 |

Training-setting fit (fire|grid_k8|fit): mean active PSNR 25.10 on every 4th training setting.

Training-setting fit (fire|grid_k16|fit): mean active PSNR 25.42 on every 4th training setting.

Training-setting fit (smoke|grid_k8|fit): mean active PSNR 23.74 on every 4th training setting.

Training-setting fit (smoke|grid_k16|fit): mean active PSNR 24.17 on every 4th training setting.

Training-setting fit (explosion|grid_k8|fit): mean active PSNR 26.54 on every 4th training setting.

Training-setting fit (explosion|grid_k16|fit): mean active PSNR 26.38 on every 4th training setting.

## C. Variation: new seeds against held-out real clips

Per effect: models with 8-dimensional variation codes trained on 24 simulated seeds at fixed controls; k8 shares 8 feature volumes, k24 has one per training seed. 8 new seeds are generated through the runtime (each a random point between two training codes) and compared with 8 held-out simulated seeds. Distances are means over pairs; smaller is closer. The "real" rows are the natural spread between simulated seeds: a generator should match them, not beat them. Spectrum = |log power difference| (blur raises it); motion = frame-to-frame change relative to the first clip of the pair; reconstruction = a training seed replayed from its own code.

| effect | model | pairs | coverage L1 | spectrum L1 | mean-frame PSNR | motion ratio | active PSNR |
|---|---|---|---:|---:|---:|---:|---:|
| fire | real | heldout_vs_heldout | 0.0034 | 0.130 | 28.95 | 1.05 | 12.89 |
| fire | real | training_vs_heldout | 0.0039 | 0.118 | 29.18 | 1.11 | 12.63 |
| fire | real | heldout_nearest_training | 0.0022 | 0.125 | 31.20 | 1.08 | 14.11 |
| fire | k8 | generated_vs_heldout | 0.0043 | 0.330 | 28.92 | 0.86 | 13.30 |
| fire | k8 | generated_vs_generated | 0.0046 | 0.218 | 29.65 | 1.15 | 13.74 |
| fire | k8 | generated_nearest_training | 0.0014 | 0.354 | 38.96 | 0.74 | 20.52 |
| fire | k8 | reconstruction | 0.0004 | 0.219 | 47.00 | 0.87 | 24.89 |
| fire | k24 | generated_vs_heldout | 0.0038 | 0.282 | 29.30 | 0.87 | 13.35 |
| fire | k24 | generated_vs_generated | 0.0039 | 0.178 | 30.39 | 1.15 | 13.89 |
| fire | k24 | generated_nearest_training | 0.0015 | 0.308 | 39.60 | 0.76 | 20.69 |
| fire | k24 | reconstruction | 0.0004 | 0.185 | 47.60 | 0.88 | 25.47 |
| smoke | real | heldout_vs_heldout | 0.0223 | 0.133 | 23.21 | 1.01 | 13.68 |
| smoke | real | training_vs_heldout | 0.0228 | 0.109 | 23.41 | 0.97 | 13.42 |
| smoke | real | heldout_nearest_training | 0.0148 | 0.114 | 24.94 | 1.28 | 14.88 |
| smoke | k8 | generated_vs_heldout | 0.0191 | 0.421 | 24.25 | 0.85 | 14.41 |
| smoke | k8 | generated_vs_generated | 0.0150 | 0.095 | 25.15 | 0.96 | 15.30 |
| smoke | k8 | generated_nearest_training | 0.0123 | 0.442 | 32.98 | 1.02 | 21.12 |
| smoke | k8 | reconstruction | 0.0012 | 0.341 | 40.31 | 0.95 | 26.10 |
| smoke | k24 | generated_vs_heldout | 0.0195 | 0.425 | 24.35 | 0.85 | 14.52 |
| smoke | k24 | generated_vs_generated | 0.0152 | 0.135 | 25.48 | 1.07 | 15.68 |
| smoke | k24 | generated_nearest_training | 0.0145 | 0.449 | 32.55 | 0.99 | 20.93 |
| smoke | k24 | reconstruction | 0.0012 | 0.310 | 40.48 | 0.97 | 26.24 |
| explosion | real | heldout_vs_heldout | 0.0138 | 0.068 | 25.51 | 1.00 | 13.89 |
| explosion | real | training_vs_heldout | 0.0207 | 0.084 | 25.27 | 0.94 | 13.80 |
| explosion | real | heldout_nearest_training | 0.0104 | 0.058 | 27.25 | 1.03 | 15.34 |
| explosion | k8 | generated_vs_heldout | 0.0151 | 0.208 | 26.09 | 0.94 | 14.58 |
| explosion | k8 | generated_vs_generated | 0.0126 | 0.074 | 27.30 | 1.02 | 15.83 |
| explosion | k8 | generated_nearest_training | 0.0026 | 0.184 | 37.49 | 0.96 | 24.14 |
| explosion | k8 | reconstruction | 0.0008 | 0.121 | 44.73 | 1.00 | 29.78 |
| explosion | k24 | generated_vs_heldout | 0.0147 | 0.213 | 26.16 | 0.93 | 14.69 |
| explosion | k24 | generated_vs_generated | 0.0139 | 0.055 | 27.32 | 1.02 | 15.98 |
| explosion | k24 | generated_nearest_training | 0.0048 | 0.197 | 37.27 | 0.96 | 24.11 |
| explosion | k24 | reconstruction | 0.0007 | 0.118 | 45.33 | 1.00 | 30.32 |

| effect | model | KB stored | training s |
|---|---|---:|---:|
| fire | fire|k8 | 1033 | 185.7 |
| fire | fire|k24 | 3089 | 359.9 |
| smoke | smoke|k8 | 1033 | 181.5 |
| smoke | smoke|k24 | 3089 | 351.2 |
| explosion | explosion|k8 | 1033 | 180.3 |
| explosion | explosion|k24 | 3089 | 356.3 |

## Runtime cost

Median (90th percentile) ms per frame through `nvfx_render`, one pinned core, nothing else running. Models trained on the first fire clip (A), the fire control model (B) and the fire variation model (C). The simulation row is the solver plus its renderer for one 128 x 128 output frame.

| model | KB stored / resident | 32 px | 64 px | 128 px | 256 px | avx512 128 | baseline 128 | MAC/px |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| b:fire_grid_k16 | 2062.7 / 2077.4 | 0.145 (0.179) | 0.472 (0.533) | 1.750 (1.837) | 6.816 (7.135) | 2.239 (3.482) | 4.467 (5.297) | 2912 |
| b:fire_grid_k8 | 1031.4 / 1038.9 | 0.087 (0.109) | 0.291 (0.316) | 1.069 (1.116) | 4.113 (7.025) | 1.353 (1.408) | 2.280 (2.403) | 1432 |
| c:fire_variation_k24 | 3088.8 / 3105.7 | 0.129 (0.157) | 0.348 (0.424) | 1.128 (1.166) | 4.320 (4.683) | 1.449 (1.536) | 2.492 (3.106) | 1448 |
| c:fire_variation_k8 | 1032.6 / 1041.1 | 0.093 (0.122) | 0.304 (0.342) | 1.148 (1.500) | 4.227 (4.374) | 1.456 (1.653) | 2.421 (3.249) | 1432 |
| conv_m16 | 268.0 / 279.9 | 0.764 (1.070) | 0.821 (0.895) | 0.790 (0.862) | - | 0.856 (1.155) | 2.076 (2.554) | 865 |
| conv_m8 | 142.0 / 155.9 | 0.702 (0.768) | 0.783 (1.367) | 0.760 (0.818) | - | 0.770 (0.836) | 1.917 (2.044) | 865 |
| conv_s16 | 132.0 / 136.1 | 0.469 (0.498) | 0.551 (0.602) | 0.565 (0.623) | - | 0.655 (0.710) | 1.300 (1.388) | 504 |
| conv_s8 | 69.0 / 74.1 | 0.459 (0.517) | 0.617 (0.675) | 0.567 (0.621) | - | 0.650 (0.707) | 1.251 (1.331) | 504 |
| grid_l16 | 579.0 / 582.0 | 0.121 (0.142) | 0.330 (0.363) | 1.124 (1.162) | 4.275 (4.441) | 1.463 (2.424) | 2.559 (4.705) | 1426 |
| grid_l8 | 291.5 / 295.0 | 0.072 (0.091) | 0.274 (0.299) | 1.091 (1.186) | 4.314 (6.630) | 1.421 (1.559) | 2.490 (2.792) | 1426 |
| grid_m16 | 259.0 / 262.0 | 0.096 (0.117) | 0.301 (0.355) | 1.105 (1.891) | 4.190 (6.611) | 1.413 (1.462) | 2.386 (2.454) | 1425 |
| grid_m8 | 131.5 / 135.0 | 0.075 (0.082) | 0.273 (0.299) | 1.060 (1.107) | 4.121 (4.322) | 1.364 (1.430) | 2.302 (2.541) | 1425 |
| grid_mt16 | 515.0 / 518.0 | 0.099 (0.118) | 0.306 (0.337) | 1.100 (1.168) | 4.263 (4.595) | 1.505 (2.565) | 2.438 (2.669) | 1425 |
| grid_mt8 | 260.0 / 264.0 | 0.078 (0.102) | 0.283 (0.309) | 1.091 (1.170) | 4.243 (4.755) | 1.433 (1.550) | 2.354 (3.189) | 1425 |
| grid_s16 | 144.5 / 145.0 | 0.040 (0.045) | 0.115 (0.159) | 0.394 (0.431) | 1.627 (2.335) | 0.599 (0.893) | 0.635 (0.750) | 209 |
| grid_s8 | 73.0 / 74.0 | 0.029 (0.053) | 0.103 (0.138) | 0.383 (0.426) | 1.492 (2.032) | 0.549 (0.608) | 0.624 (0.716) | 209 |
| simulation:fire | - | - | - | - | - | - | 8.825 (13.094) | - |
| simulation:smoke | - | - | - | - | - | - | 11.662 (14.008) | - |
| simulation:explosion | - | - | - | - | - | - | 10.656 (11.589) | - |
