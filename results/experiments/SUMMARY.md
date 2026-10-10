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
| conv_s|8 | neural_conv | 69.0 | 35.79 | 29.45 | 0.9658 | 36.18 | 0.63 | 0.262 | 0.94 | 0.430 |
| bc3 4f 128px +mv32 | flipbook_mv | 72.0 | 22.25 | 16.53 | 0.8453 | 29.55 | 0.05 | 0.221 | 0.49 | - |
| bc3 16f 64px +mv16 | flipbook_mv | 72.0 | 32.71 | 26.69 | 0.9618 | 34.50 | 0.32 | 0.574 | 0.83 | - |
| grid_s|8 | neural_grid | 73.0 | 34.32 | 28.07 | 0.9627 | 34.97 | 0.64 | 0.171 | 0.95 | 0.247 |
| bc3 8f 128px | flipbook_bc3 | 128.0 | 25.82 | 19.91 | 0.8990 | 30.14 | 0.06 | 0.141 | 0.61 | - |
| bc3 32f 64px | flipbook_bc3 | 128.0 | 32.38 | 26.39 | 0.9593 | 33.46 | 0.32 | 0.647 | 0.81 | - |
| raw 32f 32px | flipbook_raw | 128.0 | 28.56 | 22.98 | 0.9065 | 31.93 | 0.14 | 0.979 | 0.67 | - |
| raw 8f 64px | flipbook_raw | 128.0 | 25.96 | 20.30 | 0.8914 | 30.11 | 0.05 | 0.747 | 0.57 | - |
| grid_m|8 | neural_grid | 131.5 | 39.10 | 32.62 | 0.9873 | 39.83 | 0.87 | 0.096 | 0.98 | 0.778 |
| conv_s|16 | neural_conv | 132.0 | 35.79 | 29.45 | 0.9658 | 36.18 | 0.63 | 0.262 | 0.94 | 0.478 |
| conv_m|8 | neural_conv | 142.0 | 37.72 | 31.33 | 0.9733 | 37.61 | 0.67 | 0.191 | 0.95 | 0.595 |
| bc3 8f 128px +mv32 | flipbook_mv | 144.0 | 28.07 | 21.99 | 0.9337 | 31.19 | 0.24 | 0.189 | 0.74 | - |
| grid_s|16 | neural_grid | 144.5 | 34.33 | 28.07 | 0.9631 | 34.97 | 0.64 | 0.171 | 0.95 | 0.267 |
| bc3 16f 128px | flipbook_bc3 | 256.0 | 30.67 | 24.44 | 0.9548 | 31.57 | 0.23 | 0.124 | 0.79 | - |
| bc3 64f 64px | flipbook_bc3 | 256.0 | 33.63 | 27.59 | 0.9671 | 35.60 | 0.56 | 0.602 | 0.89 | - |
| raw 64f 32px | flipbook_raw | 256.0 | 28.91 | 23.30 | 0.9119 | 32.70 | 0.14 | 0.958 | 0.69 | - |
| raw 16f 64px | flipbook_raw | 256.0 | 29.87 | 23.97 | 0.9406 | 31.40 | 0.15 | 0.721 | 0.71 | - |
| grid_m|16 | neural_grid | 259.0 | 39.12 | 32.64 | 0.9874 | 39.85 | 0.87 | 0.096 | 0.98 | 0.771 |
| grid_mt|8 | neural_grid | 260.0 | 40.26 | 33.76 | 0.9897 | 41.22 | 0.96 | 0.089 | 1.01 | 0.817 |
| conv_m|16 | neural_conv | 268.0 | 37.73 | 31.33 | 0.9734 | 37.61 | 0.67 | 0.191 | 0.95 | 0.655 |
| bc3 16f 128px +mv32 | flipbook_mv | 288.0 | 35.61 | 29.25 | 0.9811 | 35.20 | 0.63 | 0.123 | 0.93 | - |
| grid_l|8 | neural_grid | 291.5 | 42.73 | 36.23 | 0.9930 | 41.42 | 0.90 | 0.039 | 0.98 | 0.866 |
| raw 32f 64px | flipbook_raw | 512.0 | 32.99 | 26.94 | 0.9663 | 33.77 | 0.31 | 0.677 | 0.80 | - |
| bc3 32f 128px | flipbook_bc3 | 512.0 | 36.36 | 29.92 | 0.9838 | 34.50 | 0.61 | 0.103 | 0.94 | - |
| grid_mt|16 | neural_grid | 515.0 | 40.28 | 33.77 | 0.9897 | 41.24 | 0.96 | 0.089 | 1.01 | 0.818 |
| grid_l|16 | neural_grid | 579.0 | 42.78 | 36.28 | 0.9931 | 41.44 | 0.90 | 0.039 | 0.98 | 0.884 |
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
| conv_s|8 | 69 | 29.45 | 341 | 4.9x |
| grid_s|8 | 73 | 28.07 | 265 | 3.6x |
| grid_m|8 | 132 | 32.62 | 772 | 5.9x |
| conv_s|16 | 132 | 29.45 | 343 | 2.6x |
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
| b:fire_grid_k16 | 2062.7 / 2077.4 | 0.114 (0.145) | 0.359 (0.410) | 1.475 (2.449) | 5.341 (5.687) | 2.416 (2.643) | 3.985 (4.482) | 2912 |
| b:fire_grid_k8 | 1031.4 / 1038.9 | 0.064 (0.070) | 0.210 (0.237) | 0.785 (0.822) | 3.081 (3.719) | 1.492 (1.977) | 2.103 (2.512) | 1432 |
| c:fire_variation_k24 | 3088.8 / 3105.7 | 0.096 (0.144) | 0.262 (0.305) | 0.886 (1.475) | 3.262 (3.996) | 1.530 (1.664) | 2.234 (2.977) | 1448 |
| c:fire_variation_k8 | 1032.6 / 1041.1 | 0.116 (0.152) | 0.241 (0.353) | 0.804 (0.857) | 3.081 (5.420) | 1.538 (1.670) | 2.092 (4.192) | 1432 |
| conv_m16 | 268.0 / 279.9 | 0.584 (0.935) | 0.639 (0.679) | 0.655 (0.848) | - | 0.829 (1.341) | 1.476 (2.155) | 865 |
| conv_m8 | 142.0 / 155.9 | 0.569 (0.615) | 0.628 (0.669) | 0.595 (0.636) | - | 0.784 (0.960) | 1.359 (1.498) | 865 |
| conv_s16 | 132.0 / 136.1 | 0.354 (0.434) | 0.454 (0.655) | 0.478 (0.630) | - | 0.582 (0.618) | 0.907 (0.951) | 504 |
| conv_s8 | 69.0 / 74.1 | 0.344 (0.378) | 0.421 (0.484) | 0.430 (0.571) | - | 0.609 (0.757) | 0.863 (0.945) | 504 |
| grid_l16 | 579.0 / 582.0 | 0.076 (0.104) | 0.216 (0.257) | 0.884 (1.537) | 3.145 (4.032) | 1.605 (1.864) | 2.350 (2.769) | 1426 |
| grid_l8 | 291.5 / 295.0 | 0.093 (0.130) | 0.279 (0.400) | 0.866 (1.310) | 3.042 (3.229) | 1.494 (1.542) | 2.066 (2.140) | 1426 |
| grid_m16 | 259.0 / 262.0 | 0.065 (0.080) | 0.212 (0.242) | 0.771 (0.873) | 3.171 (3.450) | 1.438 (1.633) | 2.226 (3.813) | 1425 |
| grid_m8 | 131.5 / 135.0 | 0.096 (0.123) | 0.331 (0.402) | 0.778 (0.846) | 3.147 (4.594) | 1.411 (1.517) | 2.090 (2.335) | 1425 |
| grid_mt16 | 515.0 / 518.0 | 0.072 (0.120) | 0.225 (0.256) | 0.818 (0.911) | 3.087 (3.991) | 1.496 (1.583) | 2.221 (2.925) | 1425 |
| grid_mt8 | 260.0 / 264.0 | 0.059 (0.076) | 0.227 (0.431) | 0.817 (0.856) | 3.210 (4.621) | 1.500 (1.551) | 2.060 (2.139) | 1425 |
| grid_s16 | 144.5 / 145.0 | 0.024 (0.025) | 0.076 (0.091) | 0.267 (0.289) | 0.951 (1.003) | 0.482 (0.677) | 0.529 (0.842) | 209 |
| grid_s8 | 73.0 / 74.0 | 0.019 (0.020) | 0.068 (0.086) | 0.247 (0.274) | 0.998 (1.064) | 0.455 (0.475) | 0.449 (0.473) | 209 |
| simulation:fire | - | - | - | - | - | - | 6.944 (8.143) | - |
| simulation:smoke | - | - | - | - | - | - | 9.356 (10.869) | - |
| simulation:explosion | - | - | - | - | - | - | 9.105 (10.872) | - |

## D: start points and learned dynamics

### Chaos horizon (the simulation itself)

Active PSNR against a run of the simulation from a stored start point, by frames after it (30 per second), mean over runs. Perturbations are relative to the velocity RMS.

| effect | case | 1 | 4 | 8 | 15 | 30 | 60 | 120 | 240 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fire | start+1e-3_same_seed | 73.94 | 71.32 | 69.92 | 66.74 | 61.86 | 56.65 | 47.81 | 34.79 |
| fire | start+1e-1_same_seed | 50.21 | 45.74 | 43.45 | 40.17 | 33.44 | 27.21 | 26.08 | 26.05 |
| fire | same_start_other_seed | 22.77 | 17.75 | 15.76 | 13.57 | 12.76 | 12.41 | 13.29 | 12.90 |
| fire | other_start_same_seed | 15.19 | 16.03 | 16.78 | 17.14 | 19.07 | 19.81 | 19.80 | 22.14 |
| smoke | start+1e-3_same_seed | 71.43 | 68.15 | 66.27 | 64.85 | 61.60 | 52.93 | 45.30 | 31.70 |
| smoke | start+1e-1_same_seed | 48.05 | 43.67 | 40.83 | 37.82 | 32.23 | 23.92 | 20.58 | 18.21 |
| smoke | same_start_other_seed | 25.08 | 23.02 | 21.08 | 18.50 | 15.01 | 13.28 | 13.63 | 12.91 |
| smoke | other_start_same_seed | 12.54 | 12.71 | 12.88 | 13.22 | 13.56 | 14.66 | 15.83 | 17.24 |
| explosion | start+1e-3_same_seed | 72.24 | 69.86 | 69.04 | 67.92 | 64.71 | 60.42 | - | - |
| explosion | start+1e-1_same_seed | 48.07 | 44.59 | 42.84 | 39.78 | 33.64 | 27.72 | - | - |
| explosion | same_start_other_seed | 54.22 | 36.74 | 28.10 | 20.86 | 16.09 | 13.77 | - | - |
| explosion | other_start_same_seed | 16.64 | 17.07 | 17.33 | 16.85 | 14.49 | 13.55 | - | - |

### Training

Stepper loss: stages 1-2 (normalised squared error plus profiles) and stage 3 (plus activity). Renderer PSNR on its training samples (true fields). Detail constants as calibrated.

| effect | runs x frames | minutes of simulation | stepper loss 1-2 | stepper s | stage 3 loss | stage 3 s | renderer dB | contrast | swirl | grow | start points | KB stored |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fire | 160 x 240 | 21.33 | 0.06687 | 2533 | 0.08811 | 1080 | 54.96 | 1 | 0.5 | 1.5 | 8 | 81.6 |
| smoke | 160 x 240 | 21.33 | 0.07107 | 3205 | 0.08316 | 1057 | 34.66 | 1 | 0.5 | 4 | 8 | 145.7 |
| explosion | 240 x 90 | 12.00 | 0.01317 | 3316 | 0.02040 | 1221 | 34.67 | 0.5 | 0.5 | 1 | 16 | 274.2 |

Renderer, detail constants and start points from `d-finish` on the trained steppers; stage 3 from `d-tune`.

### Tracking a held-out run from its true start point

Active PSNR against the true run, mean over held-out runs (other seeds and settings), by frames after the start point. `neural`: the rollout effect through the runtime, started from the true state (coarse and fine) with the run's noise seed. `coarse_sim_detail`: the simulation on the same 32-cell grid with the same detail layer, drawn by the simulation's own renderer. `neural_dynamics_true_renderer`: the neural dynamics drawn by that renderer too (separates dynamics from rendering). `renderer_on_true_fields`: the learned renderer on the true fields (frames 1-30, then their mean).

**explosion**

| method | 1 | 4 | 8 | 16 | 30 | 60 | 120 | 240 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| neural | 20.35 | 18.03 | 20.16 | 20.49 | 18.79 | 18.04 | - | - |
| neural_dynamics_true_renderer | 26.13 | 20.41 | 22.09 | 20.55 | 18.77 | 18.16 | - | - |
| coarse_sim_detail | 26.56 | 18.43 | 16.38 | 15.38 | 14.07 | 15.50 | - | - |
| coarse_sim | 22.66 | 19.51 | 17.46 | 16.10 | 14.73 | 16.14 | - | - |
| frozen | 11.17 | 6.56 | 5.77 | 5.75 | 5.75 | 7.21 | - | - |
| renderer_on_true_fields | 20.89 | 21.38 | 24.12 | 31.41 | 34.20 | 29.22 | 29.22 | 29.22 |

neural - coarse_sim_detail, paired over runs: frame 1: -6.21 [-7.35, -5.08]; frame 8: +3.78 [+2.49, +5.09]; frame 30: +4.72 [+3.78, +5.67]; frame 60: +2.54 [+1.81, +3.33];

neural_dynamics_true_renderer - coarse_sim_detail, paired over runs: frame 1: -0.43 [-2.17, +1.23] (tie); frame 8: +5.71 [+4.05, +7.28]; frame 30: +4.70 [+3.76, +5.64]; frame 60: +2.66 [+1.97, +3.45];

**fire**

| method | 1 | 4 | 8 | 16 | 30 | 60 | 120 | 240 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| neural | 25.49 | 22.00 | 20.17 | 18.76 | 17.53 | 16.40 | 16.89 | 14.83 |
| neural_dynamics_true_renderer | 25.53 | 22.06 | 20.24 | 18.81 | 17.55 | 16.41 | 16.95 | 14.84 |
| coarse_sim_detail | 23.01 | 19.30 | 17.34 | 16.24 | 15.38 | 14.70 | 16.41 | 14.76 |
| coarse_sim | 19.90 | 18.38 | 17.68 | 16.63 | 16.23 | 15.23 | 17.14 | 15.60 |
| frozen | 21.06 | 14.85 | 13.84 | 12.05 | 12.48 | 12.45 | 12.68 | 11.51 |
| renderer_on_true_fields | 49.57 | 48.64 | 48.89 | 48.95 | 49.16 | 48.84 | 48.84 | 48.84 |

neural - coarse_sim_detail, paired over runs: frame 1: +2.48 [+0.92, +4.04]; frame 8: +2.83 [+2.23, +3.37]; frame 30: +2.14 [+1.68, +2.64]; frame 60: +1.70 [+1.18, +2.19]; frame 240: +0.08 [-1.21, +1.08] (tie);

neural_dynamics_true_renderer - coarse_sim_detail, paired over runs: frame 1: +2.53 [+0.92, +4.11]; frame 8: +2.89 [+2.30, +3.44]; frame 30: +2.17 [+1.69, +2.68]; frame 60: +1.71 [+1.18, +2.19]; frame 240: +0.09 [-1.20, +1.11] (tie);

**smoke**

| method | 1 | 4 | 8 | 16 | 30 | 60 | 120 | 240 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| neural | 26.83 | 23.13 | 20.76 | 18.74 | 17.08 | 15.20 | 14.09 | 12.58 |
| neural_dynamics_true_renderer | 29.52 | 24.04 | 21.15 | 18.90 | 17.16 | 15.20 | 14.06 | 12.58 |
| coarse_sim_detail | 26.80 | 20.98 | 18.20 | 16.15 | 14.99 | 14.52 | 14.84 | 13.98 |
| coarse_sim | 21.20 | 19.75 | 18.90 | 17.93 | 16.98 | 16.14 | 16.16 | 15.00 |
| frozen | 23.33 | 15.39 | 13.76 | 12.47 | 11.83 | 11.90 | 11.93 | 12.12 |
| renderer_on_true_fields | 30.67 | 30.67 | 31.06 | 31.15 | 31.44 | 31.01 | 31.01 | 31.01 |

neural - coarse_sim_detail, paired over runs: frame 1: +0.03 [-1.76, +1.61] (tie); frame 8: +2.56 [+2.05, +3.08]; frame 30: +2.09 [+1.65, +2.52]; frame 60: +0.68 [+0.28, +1.10]; frame 240: -1.41 [-1.69, -1.08];

neural_dynamics_true_renderer - coarse_sim_detail, paired over runs: frame 1: +2.72 [+1.64, +3.79]; frame 8: +2.95 [+2.51, +3.43]; frame 30: +2.17 [+1.67, +2.66]; frame 60: +0.68 [+0.29, +1.08]; frame 240: -1.40 [-1.69, -1.09];

### Endless runs at held-out settings

The ten held-out settings of study B, a new seed each, 10 s per run (explosions: their 3 s). Distances of each method's frame statistics to a real run of the simulation at that setting; `real_other_seed` is a second real run with another seed, the floor. Lower is better except motion ratio (1 is right) and mean-frame PSNR (higher is closer).

**explosion**

| method | spectrum L1 | motion ratio | coverage L1 | emission L1 | mean-frame PSNR |
|---|---:|---:|---:|---:|---:|
| real_other_seed | 0.116 | 1.07 | 0.0195 | 0.0000 | 26.76 |
| neural | 0.200 | 1.03 | 0.0517 | 0.0000 | 24.69 |
| grid_k8 | 0.203 | 1.03 | 0.0257 | 0.0000 | 25.81 |
| flipbook_nearest | 0.141 | 0.97 | 0.0548 | 0.0000 | 23.22 |
| coarse_sim_detail | 0.261 | 0.82 | 0.0163 | 0.0000 | 26.02 |
| coarse_sim | 0.782 | 0.74 | 0.0179 | 0.0000 | 26.29 |

Spectrum distance, paired over settings: neural - grid_k8 -0.004 [-0.078, +0.073] (tie); neural - flipbook_nearest +0.059 [-0.021, +0.149] (tie); neural - coarse_sim_detail -0.061 [-0.174, +0.035] (tie); neural - real_other_seed +0.083 [+0.014, +0.168].

**fire**

| method | spectrum L1 | motion ratio | coverage L1 | emission L1 | mean-frame PSNR |
|---|---:|---:|---:|---:|---:|
| real_other_seed | 0.080 | 1.06 | 0.0035 | 0.0030 | 36.39 |
| neural | 0.195 | 0.89 | 0.0052 | 0.0043 | 33.78 |
| grid_k8 | 0.250 | 0.83 | 0.0032 | 0.0026 | 31.66 |
| flipbook_nearest | 0.158 | 1.13 | 0.0043 | 0.0035 | 30.25 |
| coarse_sim_detail | 0.229 | 0.94 | 0.0055 | 0.0044 | 31.66 |
| coarse_sim | 0.907 | 0.59 | 0.0055 | 0.0043 | 31.78 |

Spectrum distance, paired over settings: neural - grid_k8 -0.055 [-0.156, +0.051] (tie); neural - flipbook_nearest +0.038 [-0.057, +0.129] (tie); neural - coarse_sim_detail -0.034 [-0.099, +0.031] (tie); neural - real_other_seed +0.115 [+0.050, +0.179].

**smoke**

| method | spectrum L1 | motion ratio | coverage L1 | emission L1 | mean-frame PSNR |
|---|---:|---:|---:|---:|---:|
| real_other_seed | 0.076 | 1.04 | 0.0208 | 0.0000 | 29.79 |
| neural | 0.143 | 0.87 | 0.0247 | 0.0001 | 28.20 |
| grid_k8 | 0.409 | 1.02 | 0.0175 | 0.0000 | 27.47 |
| flipbook_nearest | 0.136 | 1.09 | 0.0236 | 0.0000 | 24.91 |
| coarse_sim_detail | 0.268 | 0.84 | 0.0252 | 0.0000 | 28.07 |
| coarse_sim | 1.078 | 0.45 | 0.0285 | 0.0000 | 27.62 |

Spectrum distance, paired over settings: neural - grid_k8 -0.266 [-0.326, -0.192]; neural - flipbook_nearest +0.008 [-0.063, +0.072] (tie); neural - coarse_sim_detail -0.125 [-0.171, -0.080]; neural - real_other_seed +0.067 [+0.036, +0.099].

### One minute of play

Statistics of each 10 s window of a 60 s neural run (shards of 6 s from start points, the default) against a real 10 s run at the same setting (two held-out settings): if the effect drifted, the distances would grow window by window.

| effect | setting | window | spectrum L1 | motion ratio | coverage L1 | mean-frame PSNR |
|---|---:|---:|---:|---:|---:|---:|
| fire | 0 | 0 | 0.2091 | 1.1495 | 0.0032 | 37.355 |
| fire | 0 | 1 | 0.2015 | 0.9450 | 0.0035 | 38.966 |
| fire | 0 | 2 | 0.1503 | 1.0114 | 0.0038 | 37.567 |
| fire | 0 | 3 | 0.1548 | 0.9905 | 0.0041 | 39.132 |
| fire | 0 | 4 | 0.1856 | 0.6597 | 0.0032 | 34.609 |
| fire | 0 | 5 | 0.1996 | 0.9728 | 0.0040 | 37.753 |
| fire | 1 | 0 | 0.1601 | 0.8531 | 0.0045 | 37.981 |
| fire | 1 | 1 | 0.1376 | 0.8102 | 0.0034 | 38.437 |
| fire | 1 | 2 | 0.1567 | 0.7084 | 0.0022 | 38.681 |
| fire | 1 | 3 | 0.0984 | 0.7033 | 0.0032 | 35.378 |
| fire | 1 | 4 | 0.2739 | 0.9982 | 0.0028 | 34.827 |
| fire | 1 | 5 | 0.2429 | 0.8271 | 0.0034 | 36.326 |
| smoke | 0 | 0 | 0.1684 | 0.9365 | 0.0286 | 30.306 |
| smoke | 0 | 1 | 0.1520 | 0.7690 | 0.0220 | 31.095 |
| smoke | 0 | 2 | 0.1364 | 0.7796 | 0.0170 | 31.855 |
| smoke | 0 | 3 | 0.1545 | 0.9239 | 0.0183 | 30.099 |
| smoke | 0 | 4 | 0.1399 | 0.7545 | 0.0291 | 30.451 |
| smoke | 0 | 5 | 0.1220 | 0.8486 | 0.0154 | 30.956 |
| smoke | 1 | 0 | 0.1062 | 1.1530 | 0.0306 | 27.198 |
| smoke | 1 | 1 | 0.1000 | 1.1626 | 0.0211 | 30.539 |
| smoke | 1 | 2 | 0.1050 | 1.0678 | 0.0208 | 30.707 |
| smoke | 1 | 3 | 0.1086 | 1.0567 | 0.0273 | 28.821 |
| smoke | 1 | 4 | 0.0849 | 0.9651 | 0.0178 | 30.048 |
| smoke | 1 | 5 | 0.0848 | 0.7958 | 0.0271 | 28.413 |

### Cost

Median (90th percentile) ms per frame through `nvfx_render` on one pinned core; `restart` is a seek (a restart from a start point, including any warm-up). The coarse simulation row is the 32-cell solver plus its renderer at 128 px (scalar code).

| model | ISA | size | ms | restart ms | MAC/px | KB stored / resident | KB scratch |
|---|---|---:|---:|---:|---:|---|---:|
| rollout_fire | avx2 | 64 | 0.4735 (0.9375) | 11.06 | 2510 | 81.6 / 162.8 | 2307.3 |
| rollout_fire | avx2 | 128 | 0.9146 (1.2453) | 14.27 | 1012 | 81.6 / 162.8 | 4127.8 |
| rollout_fire | avx2 | 256 | 2.3666 (4.2355) | 25.05 | 637 | 81.6 / 162.8 | 11320.8 |
| rollout_fire | baseline | 128 | 2.2795 (3.4795) | 54.61 | 1012 | 81.6 / 162.8 | 4127.8 |
| rollout_fire | avx512 | 128 | 1.2386 (1.9197) | 20.25 | 1012 | 81.6 / 162.8 | 4127.8 |
| coarse_sim_fire | scalar | 128 | 1.3628 (1.7019) | - | - | - | - |
| rollout_smoke | avx2 | 64 | 0.4790 (0.7015) | 0.46 | 2510 | 145.7 / 418.8 | 2307.3 |
| rollout_smoke | avx2 | 128 | 0.8529 (1.1368) | 0.41 | 1012 | 145.7 / 418.8 | 4127.8 |
| rollout_smoke | avx2 | 256 | 2.3514 (2.9944) | 1.52 | 637 | 145.7 / 418.8 | 11320.8 |
| rollout_smoke | baseline | 128 | 1.9126 (2.2075) | 1.13 | 1012 | 145.7 / 418.8 | 4127.8 |
| rollout_smoke | avx512 | 128 | 1.1480 (1.4139) | 0.65 | 1012 | 145.7 / 418.8 | 4127.8 |
| coarse_sim_smoke | scalar | 128 | 3.4053 (5.3575) | - | - | - | - |
| rollout_explosion | avx2 | 64 | 0.4442 (0.5027) | 0.15 | 2510 | 274.2 / 803.6 | 2307.3 |
| rollout_explosion | avx2 | 128 | 0.7766 (0.8540) | 0.41 | 1012 | 274.2 / 803.6 | 4127.8 |
| rollout_explosion | avx2 | 256 | 1.7600 (2.5153) | 1.45 | 637 | 274.2 / 803.6 | 11320.8 |
| rollout_explosion | baseline | 128 | 1.9237 (3.1724) | 0.92 | 1012 | 274.2 / 803.6 | 4127.8 |
| rollout_explosion | avx512 | 128 | 1.0671 (1.1780) | 0.66 | 1012 | 274.2 / 803.6 | 4127.8 |
| coarse_sim_explosion | scalar | 128 | 2.5312 (2.6757) | - | - | - | - |

## G1: DCM-fine (docs/DCM.md)

### G1 pilot: the default mixer against the v1 lock as a predictor

Held-out bits per active pixel on the validation runs (salt 3), each pixel coded in a bin of s_q / 256; the v1 lock with a Laplace scale fitted per channel and heat-level bin. Differences paired over validation runs.

| effect | v1 lock | default mixer (linear) | default mixer (log) | linear - v1 | log - linear |
|---|---:|---:|---:|---|---|
| fire | 3.417 | 2.678 | 3.235 | -0.739 [-0.784, -0.692] | +0.556 [+0.528, +0.584] |
| smoke | 4.149 | 3.403 | 3.828 | -0.746 [-0.825, -0.664] | +0.425 [+0.381, +0.466] |
| explosion | 3.808 | 3.082 | 3.632 | -0.726 [-0.802, -0.649] | +0.550 [+0.496, +0.608] |

### G1 search: nested held-out bits per active pixel

Nested leave-one-control-bin-out search (5 k-means bins of the 48 training runs' controls), 200 configurations plus 2 refinement rounds, objective Laplace bits, cost budget +1 ms per 128 x 128 frame. `nested` is the honest held-out score of the whole search procedure (mean over training runs, each predicted by a mixer chosen and trained without its bin); `global #0` is the released configuration (chosen with every bin, so its search score is optimistic), with its bits on the validation runs.

| effect | family | seed | nested bits | global #0 validation bits | cost ms | global #0 configuration |
|---|---|---:|---:|---:|---:|---|
| explosion | hand+macro | 0 | 2.7248 | 2.9086 | 0.950 | inputs A+A_sl+A-A_sl+a_phi1+a_phi2+a_phi3 | mixers -,channel | avm - 0.15 | scale ratio | loss squared | lr 0.002/0.05 | ep 8 |
| fire | hand+macro | 0 | 2.6264 | 2.6449 | 0.993 | inputs A+A_sl+A-A_sl+bias | mixers - | avm channel 0.15 | scale height | loss laplace | lr 0.02/0.002 | ep 8 |
| fire | hand+macro | 1 | 2.6383 | 2.6421 | 0.917 | inputs A+A_sl+A-A_sl+prev | mixers - | avm off | scale height | loss laplace | lr 0.002/0.005 | ep 8 |
| fire | hand+macro | 2 | 2.6269 | 2.6476 | 0.993 | inputs A+A_sl+A-A_sl+bias | mixers - | avm channel 0.15 | scale height | loss laplace | lr 0.002/0.002 | ep 6 |
| fire | hand+macro/free | 0 | 2.6611 | 2.6769 | 1.321 | inputs A+A_sl+A-A_sl+L+rA+a_up+C_up+bres+lap+grad | mixers -,lvl | avm off | scale height | loss laplace | lr 0.02/0.05 | ep 8 |
| fire | hand | 0 | 2.6421 | 2.6718 | 0.906 | inputs A+A_sl+A-A_sl+bias | mixers - | avm off | scale height | loss laplace | lr 0.002/0.002 | ep 3 |
| fire | hand/free | 0 | 2.6324 | 2.6363 | 1.648 | inputs A+A_sl+A-A_sl+L+rA+a_up+a_phi1+a_phi2+a_phi3+C_up+bres+lap+grad+prev+bias | mixers -,flow,ctrl,channel | avm flow 0.15 | scale height | loss laplace | lr 0.05/0.005 | ep 8 |
| fire | none | 0 | 2.6930 | 2.6655 | 0.977 | inputs A+A_sl+A-A_sl+L+rA+a_up+a_phi1+a_phi2+a_phi3+prev+bias | mixers - | avm off | scale - | loss laplace | lr 0.005/0.002 | ep 4 |
| fire | none/free | 0 | 2.6898 | 2.6681 | 1.083 | inputs A+A_sl+A-A_sl+L+rA+a_up+a_phi1+a_phi2+a_phi3+C_up+bres+prev+bias | mixers - | avm - 0.15 | scale - | loss laplace | lr 0.02/0.002 | ep 8 |
| smoke | hand+macro | 0 | 3.3931 | 3.3951 | 0.947 | inputs A+A_sl+A-A_sl+L+rA+a_up+prev+bias | mixers - | avm - 0.5 | scale - | loss laplace | lr 0.01/0.005 | ep 8 |

Seed noise of the search (fire, seeds 0, 1, 2; nested bits paired over training runs): 1 - 0: +0.0119 [+0.0086, +0.0158]; 2 - 0: +0.0004 [-0.0023, +0.0031] (tie); 2 - 1: -0.0115 [-0.0140, -0.0091].

Context families (fire, within the cost budget, nested bits paired over training runs): hand - none -0.0509 [-0.0693, -0.0333]; hand+macro - hand -0.0156 [-0.0194, -0.0117]; hand+macro - none -0.0665 [-0.0856, -0.0485].

Context families (fire, without a cost budget (60 configurations, 1 refinement round), nested bits paired over training runs): hand - none -0.0574 [-0.0699, -0.0451]; hand+macro - hand +0.0288 [+0.0192, +0.0374]; hand+macro - none -0.0286 [-0.0476, -0.0104].

### G1 generation on validation settings

The calibration score of `calibrate_detail` (detail spectrum distance + |log motion ratio| + log ratios of light and cover; lower is better), mean over the 10 validation settings, single shards from the nearest start point. Candidates are the search's global top 10; pass 1 is trained on the one-step rows, pass 2 adds the own-rollout windows.

| effect | generator | score | - v1, paired over settings |
|---|---|---:|---|
| fire | cand0_pass1 tau 0.00 relock 2 | 0.7720 | -0.043 [-0.054, -0.033] |
| fire | cand0_pass2 tau 0.00 relock 0 | 15.2364 | +14.422 [+13.612, +15.293] |
| fire | cand0_pass2 tau 0.00 relock 1 | 1.2759 | +0.461 [+0.340, +0.590] |
| fire | cand0_pass2 tau 0.00 relock 2 | 0.7660 | -0.049 [-0.061, -0.039] |
| fire | cand0_pass2 tau 0.50 relock 0 | 8.8664 | +8.052 [+7.257, +8.852] |
| fire | cand0_pass2 tau 0.50 relock 1 | 2.1789 | +1.364 [+1.159, +1.567] |
| fire | cand0_pass2 tau 0.50 relock 2 | 1.1007 | +0.286 [+0.226, +0.349] |
| fire | cand0_pass2 tau 1.00 relock 0 | 10.0451 | +9.230 [+8.241, +10.235] |
| fire | cand0_pass2 tau 1.00 relock 1 | 2.5368 | +1.722 [+1.439, +1.992] |
| fire | cand0_pass2 tau 1.00 relock 2 | 1.3958 | +0.581 [+0.452, +0.706] |
| fire | cand1_pass2 tau 0.00 relock 2 | 0.7418 | -0.073 [-0.097, -0.052] |
| fire | cand2_pass2 tau 0.00 relock 2 | 0.7440 | -0.071 [-0.093, -0.052] |
| fire | cand3_pass2 tau 0.00 relock 2 | 0.7430 | -0.072 [-0.095, -0.052] |
| fire | cand4_pass2 tau 0.00 relock 2 | 0.7352 | -0.080 [-0.109, -0.053] |
| fire | cand5_pass2 tau 0.00 relock 2 | 0.7425 | -0.072 [-0.095, -0.053] |
| fire | cand6_pass2 tau 0.00 relock 2 | 0.7308 | -0.084 [-0.115, -0.054] |
| fire | cand7_pass2 tau 0.00 relock 2 | 0.7204 | -0.094 [-0.128, -0.059] |
| fire | cand7_pass2 tau 0.50 relock 2 | 0.9519 | +0.137 [+0.093, +0.179] |
| fire | cand7_pass2 tau 1.00 relock 2 | 1.2655 | +0.451 [+0.348, +0.551] |
| fire | cand8_pass2 tau 0.00 relock 2 | 0.7440 | -0.071 [-0.092, -0.053] |
| fire | cand9_pass2 tau 0.00 relock 2 | 0.7526 | -0.062 [-0.081, -0.044] |
| fire | v1 tau 0.00 relock 0 | 0.8148 |  |
| smoke | cand0_pass1 tau 0.00 relock 2 | 0.5831 | +0.002 [-0.070, +0.055] (tie) |
| smoke | cand0_pass2 tau 0.00 relock 0 | 1.0304 | +0.449 [+0.293, +0.590] |
| smoke | cand0_pass2 tau 0.00 relock 1 | 0.5973 | +0.016 [-0.083, +0.098] (tie) |
| smoke | cand0_pass2 tau 0.00 relock 2 | 0.5941 | +0.013 [-0.086, +0.096] (tie) |
| smoke | cand0_pass2 tau 0.50 relock 0 | 4.0815 | +3.501 [+3.207, +3.795] |
| smoke | cand0_pass2 tau 0.50 relock 1 | 0.8054 | +0.224 [+0.187, +0.256] |
| smoke | cand0_pass2 tau 0.50 relock 2 | 0.7740 | +0.193 [+0.164, +0.220] |
| smoke | cand0_pass2 tau 1.00 relock 0 | 4.7775 | +4.197 [+3.878, +4.503] |
| smoke | cand0_pass2 tau 1.00 relock 1 | 1.1647 | +0.584 [+0.464, +0.706] |
| smoke | cand0_pass2 tau 1.00 relock 2 | 0.9395 | +0.359 [+0.309, +0.414] |
| smoke | cand1_pass2 tau 0.00 relock 2 | 0.5707 | -0.010 [-0.082, +0.045] (tie) |
| smoke | cand2_pass2 tau 0.00 relock 2 | 0.5674 | -0.014 [-0.092, +0.049] (tie) |
| smoke | cand2_pass2 tau 0.50 relock 2 | 0.8066 | +0.226 [+0.189, +0.259] |
| smoke | cand2_pass2 tau 1.00 relock 2 | 1.0152 | +0.434 [+0.353, +0.517] |
| smoke | cand3_pass2 tau 0.00 relock 2 | 0.5830 | +0.002 [-0.088, +0.078] (tie) |
| smoke | cand4_pass2 tau 0.00 relock 2 | 0.5770 | -0.004 [-0.091, +0.068] (tie) |
| smoke | cand5_pass2 tau 0.00 relock 2 | 0.5714 | -0.010 [-0.094, +0.058] (tie) |
| smoke | cand6_pass2 tau 0.00 relock 2 | 0.5848 | +0.004 [-0.088, +0.082] (tie) |
| smoke | cand7_pass2 tau 0.00 relock 2 | 0.5830 | +0.002 [-0.090, +0.080] (tie) |
| smoke | cand8_pass2 tau 0.00 relock 2 | 0.5684 | -0.013 [-0.118, +0.080] (tie) |
| smoke | cand9_pass2 tau 0.00 relock 2 | 0.6587 | +0.078 [-0.044, +0.181] (tie) |
| smoke | v1 tau 0.00 relock 0 | 0.5809 |  |
| explosion | cand0_pass1 tau 0.00 relock 0 | 0.6764 | -0.034 [-0.149, +0.082] (tie) |
| explosion | cand0_pass2 tau 0.00 relock 0 | 0.6699 | -0.040 [-0.143, +0.061] (tie) |
| explosion | cand0_pass2 tau 0.00 relock 1 | 0.7152 | +0.005 [-0.094, +0.108] (tie) |
| explosion | cand0_pass2 tau 0.00 relock 2 | 0.7520 | +0.042 [-0.001, +0.081] (tie) |
| explosion | cand0_pass2 tau 0.50 relock 0 | 3.8421 | +3.132 [+2.567, +3.633] |
| explosion | cand0_pass2 tau 0.50 relock 1 | 2.2786 | +1.569 [+1.330, +1.780] |
| explosion | cand0_pass2 tau 0.50 relock 2 | 0.7270 | +0.017 [-0.025, +0.064] (tie) |
| explosion | cand0_pass2 tau 1.00 relock 0 | 4.0650 | +3.355 [+2.802, +3.847] |
| explosion | cand0_pass2 tau 1.00 relock 1 | 2.8805 | +2.171 [+1.935, +2.383] |
| explosion | cand0_pass2 tau 1.00 relock 2 | 0.8424 | +0.132 [-0.006, +0.271] (tie) |
| explosion | cand1_pass2 tau 0.00 relock 0 | 0.6976 | -0.012 [-0.160, +0.136] (tie) |
| explosion | cand2_pass2 tau 0.00 relock 0 | 0.7103 | +0.000 [-0.162, +0.163] (tie) |
| explosion | cand3_pass2 tau 0.00 relock 0 | 0.6968 | -0.013 [-0.161, +0.135] (tie) |
| explosion | cand4_pass2 tau 0.00 relock 0 | 0.7318 | +0.022 [-0.161, +0.207] (tie) |
| explosion | cand5_pass2 tau 0.00 relock 0 | 0.6976 | -0.012 [-0.160, +0.136] (tie) |
| explosion | cand6_pass2 tau 0.00 relock 0 | 0.6655 | -0.044 [-0.135, +0.043] (tie) |
| explosion | cand6_pass2 tau 0.50 relock 0 | 3.8243 | +3.114 [+2.554, +3.612] |
| explosion | cand6_pass2 tau 1.00 relock 0 | 4.0450 | +3.335 [+2.782, +3.827] |
| explosion | cand7_pass2 tau 0.00 relock 0 | 0.6976 | -0.012 [-0.160, +0.136] (tie) |
| explosion | cand8_pass2 tau 0.00 relock 0 | 0.6976 | -0.012 [-0.160, +0.136] (tie) |
| explosion | cand9_pass2 tau 0.00 relock 0 | 0.6976 | -0.012 [-0.160, +0.136] (tie) |
| explosion | v1 tau 0.00 relock 0 | 0.7099 |  |

### G1 test (once): B's 10 held-out settings, new seeds

Single 6 s shards (explosions: 3 s) from the nearest start point, v1 and DCM-fine through the reference with the same stepper and seeds (only the detail layer differs), against a real 10 s run (explosions: 3 s). Means over settings; differences DCM-fine - v1 paired over settings.

| effect | method | spectrum L1 | motion ratio | coverage L1 | emission L1 | mean-frame PSNR |
|---|---|---:|---:|---:|---:|---:|
| fire | real_other_seed | 0.0803 | 1.058 | 0.0035 | 0.0030 | 36.39 |
| fire | v1 | 0.3652 | 0.959 | 0.0044 | 0.0039 | 33.59 |
| fire | dcm_fine | 0.1914 | 0.897 | 0.0044 | 0.0039 | 33.72 |
| smoke | real_other_seed | 0.0763 | 1.042 | 0.0208 | 0.0000 | 29.79 |
| smoke | v1 | 0.2538 | 0.850 | 0.0239 | 0.0001 | 28.31 |
| smoke | dcm_fine | 0.1252 | 0.718 | 0.0239 | 0.0001 | 28.27 |
| explosion | real_other_seed | 0.1164 | 1.067 | 0.0195 | 0.0000 | 26.76 |
| explosion | v1 | 0.1993 | 0.922 | 0.0305 | 0.0000 | 26.16 |
| explosion | dcm_fine | 0.1023 | 1.045 | 0.0306 | 0.0000 | 25.86 |

| effect | spectrum L1 | abs log motion ratio | coverage L1 | mean-frame PSNR | calibration score |
|---|---|---|---|---|---|
| fire | -0.174 [-0.186, -0.161] | +0.009 [-0.033, +0.053] (tie) | -0.0000 [-0.0000, +0.0000] (tie) | +0.13 [+0.07, +0.20] | -0.169 [-0.221, -0.119] |
| smoke | -0.129 [-0.176, -0.068] | +0.160 [+0.135, +0.180] | +0.0000 [-0.0001, +0.0001] (tie) | -0.05 [-0.07, -0.02] | +0.032 [-0.021, +0.093] (tie) |
| explosion | -0.097 [-0.151, -0.047] | -0.033 [-0.112, +0.049] (tie) | +0.0001 [-0.0034, +0.0038] (tie) | -0.30 [-0.73, +0.11] (tie) | -0.121 [-0.234, -0.025] |

**Decision (rule fixed in advance):** spectrum distance lower than v1 with an interval excluding zero on 3 of 3 effects (2 needed); an effect is worse; cost within +1 ms. G1 **does not pass**. Notes: smoke: motion further from 1; smoke: mean-frame PSNR worse;

### G1c: the first second

Calibration score and spectrum distance of the first 30 frames against the real run (looping effects: its 10 s statistics; explosions: its own first second). `usual`: v1's start (fire: a 30-frame warm-up; smoke and explosions: stored fine fields); `cold`: from the coarse state alone, no warm-up and no stored fine fields.

| effect | v1 usual | v1 cold | DCM-fine usual | DCM-fine cold | DCM cold - v1 usual (score) | DCM cold - v1 usual (spectrum) |
|---|---:|---:|---:|---:|---|---|
| fire | 1.693 | 2.185 | 1.552 | 2.018 | +0.325 [-0.161, +0.911] (tie) | -0.179 [-0.339, -0.062] |
| smoke | 1.671 | 1.671 | 1.536 | 1.542 | -0.129 [-0.198, -0.067] | -0.188 [-0.206, -0.170] |
| explosion | 0.510 | 0.520 | 0.484 | 0.496 | -0.014 [-0.054, +0.025] (tie) | -0.016 [-0.048, +0.015] (tie) |

### G1 test: tracking 8 held-out runs from their true state

Active PSNR against the true run at 1, 8, 30 and 60 frames; DCM-fine - v1 paired over runs.

| effect | v1 1 / 8 / 30 / 60 | DCM-fine 1 / 8 / 30 / 60 | difference at 1 | 8 | 30 | 60 |
|---|---|---|---|---|---|---|
| fire | 25.49 / 20.17 / 17.53 / 16.40 | 25.59 / 20.52 / 17.79 / 16.87 | +0.10 [+0.07, +0.14] | +0.35 [+0.27, +0.45] | +0.27 [+0.14, +0.42] | +0.47 [+0.28, +0.67] |
| smoke | 26.83 / 20.77 / 17.08 / 15.20 | 27.14 / 21.33 / 17.61 / 15.48 | +0.31 [+0.20, +0.42] | +0.57 [+0.44, +0.69] | +0.53 [+0.43, +0.64] | +0.28 [+0.20, +0.36] |
| explosion | 20.35 / 20.16 / 18.79 / 18.04 | 19.90 / 18.39 / 16.84 / 16.66 | -0.45 [-0.71, -0.20] | -1.77 [-2.89, -0.37] | -1.95 [-2.40, -1.50] | -1.38 [-1.82, -1.00] |
