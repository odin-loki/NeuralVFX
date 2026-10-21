# The smoke plume against Morton, Taylor and Turner (1956)

Time means of each run, pooled over the runs (seeds); fits over heights 24.0 to 92.0 cells above the floor; 95% intervals by resampling the runs (10000 resamples, seed 2026). Gaussian widths are 1/e radii of the core (STUDY_R.md §1.3); alpha_G = beta / 1.2, alpha_T = sqrt(2) alpha_G.

| config | runs | controls | F (cells^4/s^3) | F spread | beta (b_w / z) | alpha_G | alpha_T | z0 (cells) | p_w (MTT -1/3) | p_g (MTT -5/3) | p_soot (-5/3) | p_F (0) | z0 from w^-3 | z0 from g^-3/5 | p_w, z0 free | lambda = b_g / b_w | C_w (MTT at alpha_G) | C_g (MTT at alpha_G) | Gamma, lowest / highest heights (pure plume 1) | dM/dz over buoyancy (1) |
|---|---:|---|---:|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v_i50b | 4 | 0.50, 0.50, 0.50 (64x96x64, span 128) | 48290 [41225, 53669] | 0.03 [0.02, 0.11] | 0.134 [0.118, 0.154] | 0.112 [0.098, 0.128] | 0.158 [0.139, 0.181] | -8.0 [-16.0, -2.3] | -0.447 [-0.521, -0.349] | -1.451 [-1.515, -1.415] | -1.451 [-1.515, -1.415] | 0.033 [-0.017, 0.107] | 6.9 [-18.6, 18.0] | -15.7 [-20.9, -10.0] | -0.395 [-3.362, -0.155] | 0.97 [0.96, 0.99] | 2.56 [2.35, 2.74] (3.76) | 8.13 [6.45, 10.42] (9.43) | 2.45 [1.80, 3.48] / 3.87 [3.37, 4.96] | 0.78 [0.74, 0.81] |
| v_t100b | 4 | 0.50, 0.50, 1.00 (64x96x64, span 128) | 48662 [45043, 53097] | 0.04 [0.03, 0.08] | 0.139 [0.130, 0.148] | 0.116 [0.108, 0.124] | 0.164 [0.153, 0.175] | -13.9 [-17.7, -10.8] | -0.365 [-0.424, -0.300] | -1.741 [-1.828, -1.672] | -1.741 [-1.828, -1.672] | -0.034 [-0.049, -0.023] | -14.8 [-33.3, -4.2] | -10.9 [-12.1, -9.9] | -0.168 [-0.234, -0.109] | 1.02 [0.99, 1.07] | 2.32 [2.24, 2.41] (3.66) | 7.93 [7.24, 8.72] (8.94) | 3.43 [3.12, 3.77] / 3.66 [3.37, 3.89] | 0.72 [0.69, 0.74] |

The buoyancy budget from 24 to 92 cells (each as a fraction of F at 24; 95% intervals by resampling the runs): the planes' flux lost, what leaves through the four sides in between, what the box stores there, and the rest (made or lost by the numerics):

| config | runs | F at lo | F at hi | lost, (F(lo) - F(hi)) / F(lo) | out through the sides | stored | the rest |
|---|---:|---:|---:|---|---|---|---|
| v_i50b | 4 | 47267 | 48369 | -0.023 [-0.085, 0.020] | 0.005 [0.000, 0.013] | -0.026 [-0.086, 0.011] | -0.003 [-0.004, -0.000] |
| v_t100b | 4 | 49228 | 47556 | 0.034 [0.014, 0.053] | 0.003 [0.001, 0.004] | 0.034 [0.015, 0.053] | -0.003 [-0.003, -0.003] |
