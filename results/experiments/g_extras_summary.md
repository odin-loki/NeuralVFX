# Study G extras (stage S8): tables

Written by `nvfx_g_extras summary` from `results/experiments/g_extras_*.csv`; rules in docs/DCM.md §10. Intervals: 95% paired
bootstrap, 10,000 resamples, over settings; (tie): the interval covers zero.

#### G4a pilot: each renderer alone (validation tracking, active PSNR in dB, means over 10 settings)

| effect | renderer | frames 1-30 | frames 31-60 (89) | minus learned, frames 1-30 |
|---|---|---:|---:|---|
| explosion | learned | 21.18 | 17.96 |  |
| explosion | other1 | 11.82 | 15.24 | -9.35 [-10.21, -8.49] |
| explosion | other2 | 14.73 | 15.57 | -6.45 [-7.08, -5.76] |
| explosion | shader | 17.34 | 17.18 | -3.83 [-4.41, -3.19] |
| explosion | sim | 22.16 | 18.02 | +0.99 [+0.84, +1.15] |
| fire | learned | 20.44 | 18.12 |  |
| fire | other1 | 14.72 | 14.72 | -5.73 [-6.49, -5.06] |
| fire | other2 | 17.50 | 17.38 | -2.94 [-3.66, -2.38] |
| fire | shader | 18.73 | 19.21 | -1.72 [-2.61, -0.99] |
| fire | sim | 20.50 | 18.16 | +0.06 [+0.04, +0.09] |
| smoke | learned | 20.19 | 16.59 |  |
| smoke | other1 | 15.80 | 14.92 | -4.39 [-4.77, -4.01] |
| smoke | other2 | 18.19 | 16.26 | -2.00 [-2.27, -1.74] |
| smoke | shader | 18.83 | 16.93 | -1.36 [-1.72, -0.99] |
| smoke | sim | 20.50 | 16.61 | +0.31 [+0.16, +0.47] |

#### G4a val: v1's learned renderer and the mixers (means over settings; differences paired over settings)

| effect | method | frames 1-30 | 31-60 (89) | score | spectrum | \|ln motion\| | coverage L1 | mean-frame PSNR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| explosion | mix_learned_sim | 21.9990 | 18.1491 | 0.8581 | 0.2528 | 0.2862 | 0.0473 | 25.0102 |
| explosion | mix_learned_sim_shader | 22.0798 | 18.2351 | 0.8651 | 0.2620 | 0.2840 | 0.0472 | 24.9751 |
| explosion | mix_learned_sim_shader_other1_other2 | 22.1852 | 18.3446 | 0.8916 | 0.2910 | 0.2816 | 0.0477 | 24.9841 |
| explosion | v1 | 21.1756 | 17.9626 | 0.8009 | 0.2102 | 0.2738 | 0.0479 | 24.7891 |
| fire | mix_learned_sim | 20.7995 | 18.5484 | 0.8865 | 0.1902 | 0.2238 | 0.0038 | 34.3276 |
| fire | mix_learned_sim_shader | 20.8821 | 18.8766 | 0.8253 | 0.1506 | 0.2124 | 0.0038 | 34.2685 |
| fire | mix_learned_sim_shader_other1_other2 | 22.3798 | 21.1939 | 0.7474 | 0.1528 | 0.1902 | 0.0037 | 34.2391 |
| fire | v1 | 20.4440 | 18.1211 | 0.9021 | 0.2707 | 0.1895 | 0.0039 | 34.2572 |
| smoke | mix_learned_sim | 20.7609 | 17.0124 | 0.6664 | 0.1538 | 0.2492 | 0.0197 | 29.4090 |
| smoke | mix_learned_sim_shader | 20.8401 | 17.2078 | 0.6553 | 0.1370 | 0.2601 | 0.0195 | 29.4529 |
| smoke | mix_learned_sim_shader_other1_other2 | 20.8642 | 17.3938 | 0.6818 | 0.1348 | 0.2766 | 0.0198 | 29.4748 |
| smoke | v1 | 20.1912 | 16.5915 | 0.6383 | 0.2159 | 0.1825 | 0.0194 | 29.3984 |

| effect | method - v1 | frames 1-30 | 31-60 (89) | score | spectrum | \|ln motion\| | coverage L1 | mean-frame PSNR | worse in a rule measure |
|---|---|---|---|---|---|---|---|---|---|
| explosion | mix_learned_sim | +0.82 [+0.63, +1.01] | +0.19 [+0.10, +0.28] | +0.0572 [-0.0373, +0.1458] (tie) | +0.0427 [+0.0078, +0.0734] | +0.0124 [-0.0254, +0.0497] (tie) | -0.0006 [-0.0052, +0.0044] (tie) | +0.22 [-0.11, +0.54] (tie) | no |
| explosion | mix_learned_sim_shader | +0.90 [+0.71, +1.09] | +0.27 [+0.18, +0.37] | +0.0641 [-0.0238, +0.1474] (tie) | +0.0519 [+0.0176, +0.0812] | +0.0102 [-0.0236, +0.0435] (tie) | -0.0006 [-0.0053, +0.0043] (tie) | +0.19 [-0.06, +0.43] (tie) | no |
| explosion | mix_learned_sim_shader_other1_other2 | +1.01 [+0.90, +1.11] | +0.38 [+0.28, +0.49] | +0.0906 [+0.0062, +0.1676] | +0.0809 [+0.0338, +0.1178] | +0.0078 [-0.0203, +0.0351] (tie) | -0.0002 [-0.0034, +0.0033] (tie) | +0.20 [-0.00, +0.39] (tie) | **yes** |
| fire | mix_learned_sim | +0.36 [+0.24, +0.47] | +0.43 [+0.35, +0.52] | -0.0156 [-0.1510, +0.1182] (tie) | -0.0805 [-0.0959, -0.0601] | +0.0344 [-0.0136, +0.0758] (tie) | -0.0001 [-0.0003, +0.0001] (tie) | +0.07 [-0.52, +0.69] (tie) | no |
| fire | mix_learned_sim_shader | +0.44 [+0.25, +0.62] | +0.76 [+0.59, +0.94] | -0.0768 [-0.2321, +0.0705] (tie) | -0.1201 [-0.1491, -0.0822] | +0.0229 [-0.0192, +0.0609] (tie) | -0.0001 [-0.0004, +0.0001] (tie) | +0.01 [-0.79, +0.89] (tie) | no |
| fire | mix_learned_sim_shader_other1_other2 | +1.94 [+1.70, +2.18] | +3.07 [+2.57, +3.59] | -0.1546 [-0.2323, -0.0812] | -0.1178 [-0.1453, -0.0815] | +0.0007 [-0.0288, +0.0291] (tie) | -0.0003 [-0.0004, -0.0001] | -0.02 [-0.73, +0.77] (tie) | no |
| smoke | mix_learned_sim | +0.57 [+0.43, +0.74] | +0.42 [+0.33, +0.52] | +0.0280 [-0.0109, +0.0609] (tie) | -0.0621 [-0.0789, -0.0450] | +0.0667 [+0.0619, +0.0714] | +0.0003 [-0.0007, +0.0012] (tie) | +0.01 [-0.23, +0.27] (tie) | no |
| smoke | mix_learned_sim_shader | +0.65 [+0.49, +0.83] | +0.62 [+0.49, +0.74] | +0.0169 [-0.0255, +0.0552] (tie) | -0.0790 [-0.1038, -0.0518] | +0.0776 [+0.0701, +0.0849] | +0.0002 [-0.0007, +0.0009] (tie) | +0.05 [-0.17, +0.29] (tie) | no |
| smoke | mix_learned_sim_shader_other1_other2 | +0.67 [+0.51, +0.85] | +0.80 [+0.65, +0.96] | +0.0434 [-0.0106, +0.0889] (tie) | -0.0811 [-0.1084, -0.0512] | +0.0941 [+0.0833, +0.1041] | +0.0004 [-0.0008, +0.0015] (tie) | +0.08 [-0.22, +0.39] (tie) | no |

**G4a choice (validation):** `mix_learned_sim_shader`, mean first-second gain over the effects +0.66 dB.

#### G4a test: v1's learned renderer and the mixers (means over settings; differences paired over settings)

| effect | method | frames 1-30 | 31-60 (89) | score | spectrum | \|ln motion\| | coverage L1 | mean-frame PSNR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| explosion | mix_learned_sim_shader | 20.7021 | 17.8710 | 0.5911 | 0.2484 | 0.1959 | 0.0288 | 26.4848 |
| explosion | v1 | 19.6785 | 17.4989 | 0.5107 | 0.2003 | 0.1677 | 0.0299 | 26.1740 |
| fire | mix_learned_sim_shader | 20.5004 | 19.0034 | 1.1050 | 0.2386 | 0.3283 | 0.0041 | 34.2511 |
| fire | v1 | 19.8172 | 18.0105 | 1.1490 | 0.3662 | 0.2909 | 0.0040 | 34.0921 |
| smoke | mix_learned_sim_shader | 20.6453 | 16.6140 | 0.7068 | 0.1317 | 0.3161 | 0.0269 | 27.5424 |
| smoke | v1 | 20.0200 | 16.0913 | 0.6898 | 0.2135 | 0.2370 | 0.0268 | 27.4750 |

| effect | method - v1 | frames 1-30 | 31-60 (89) | score | spectrum | \|ln motion\| | coverage L1 | mean-frame PSNR | worse in a rule measure |
|---|---|---|---|---|---|---|---|---|---|
| explosion | mix_learned_sim_shader | +1.02 [+0.80, +1.25] | +0.37 [+0.32, +0.43] | +0.0804 [+0.0200, +0.1391] | +0.0481 [+0.0169, +0.0743] | +0.0282 [-0.0002, +0.0521] (tie) | -0.0011 [-0.0047, +0.0024] (tie) | +0.31 [+0.16, +0.44] | **yes** |
| fire | mix_learned_sim_shader | +0.68 [+0.53, +0.82] | +0.99 [+0.76, +1.22] | -0.0440 [-0.1904, +0.0993] (tie) | -0.1277 [-0.1486, -0.1046] | +0.0374 [-0.0054, +0.0774] (tie) | +0.0001 [-0.0001, +0.0002] (tie) | +0.16 [-0.51, +0.86] (tie) | no |
| smoke | mix_learned_sim_shader | +0.63 [+0.57, +0.69] | +0.52 [+0.49, +0.55] | +0.0170 [-0.0184, +0.0485] (tie) | -0.0818 [-0.1091, -0.0537] | +0.0791 [+0.0690, +0.0901] | +0.0001 [-0.0011, +0.0013] (tie) | +0.07 [-0.15, +0.29] (tie) | no |

**G4a decision (test), `mix_learned_sim_shader`:** explosion first second +1.02 [+0.80, +1.25] dB; an effect worse in a rule measure: yes. **Not kept.**

