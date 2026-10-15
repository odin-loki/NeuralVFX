Side heat, mean over 5 to 15 s / mean over 20 to 30 s (largest over 5 to 30 s; the simulation's largest):

| model | wind | x = 0 | x = max | z = 0 | z = max | top | verdict |
|---|---:|---|---|---|---|---|---|
| C2 | 0.05 | 2.12 / 5.01 (6.91; 8.50) | 0.43 / 0.31 (1.55; 0.00) | 0.57 / 1.64 (4.77; 0.05) | 0.08 / 0.14 (0.23; 0.04) | 0.02 / 2.64 (4.25; 0.44) | **runs away**: x = 0 grows; x = max holds too much; z = 0 grows and holds too much; top grows and holds too much |
| C2 | 0.25 | 1.44 / 1.51 (3.50; 5.32) | 0.09 / 1.92 (3.58; 0.00) | 0.09 / 1.32 (2.89; 0.11) | 0.05 / 0.07 (0.22; 0.46) | 0.04 / 0.82 (3.01; 2.56) | **runs away**: x = max grows and holds too much; z = 0 grows and holds too much; top grows |
| C2 | 0.50 | 0.05 / 0.81 (2.18; 0.01) | 0.02 / 3.03 (5.35; 0.00) | 0.02 / 1.44 (2.85; 0.01) | 0.05 / 0.05 (0.11; 0.03) | 0.00 / 0.60 (1.95; 3.31) | **runs away**: x = 0 grows and holds too much; x = max grows and holds too much; z = 0 grows and holds too much; top grows |
| C2 | 0.75 | 0.06 / 0.44 (0.74; 0.00) | 1.00 / 4.37 (8.98; 2.06) | 0.01 / 0.64 (1.60; 0.20) | 0.03 / 0.04 (0.09; 0.04) | 0.00 / 0.01 (0.11; 1.08) | **runs away**: x = max grows and holds too much; z = 0 grows and holds too much |
| C2 | 0.95 | 0.08 / 0.35 (0.46; 0.00) | 2.16 / 14.98 (17.79; 4.10) | 0.18 / 0.50 (1.02; 0.05) | 0.02 / 0.02 (0.09; 0.03) | 0.01 / 0.10 (0.55; 0.58) | **runs away**: x = max grows and holds too much |
| C3 | 0.05 | 2.68 / 3.01 (4.93; 8.50) | 0.00 / 0.00 (0.02; 0.00) | 0.00 / 0.00 (0.02; 0.05) | 0.08 / 0.07 (0.16; 0.04) | 0.01 / 0.00 (0.03; 0.44) | no runaway |
| C3 | 0.25 | 0.48 / 0.25 (1.04; 5.32) | 0.01 / 0.02 (0.05; 0.00) | 0.01 / 0.02 (0.05; 0.11) | 0.05 / 0.05 (0.18; 0.46) | 0.00 / 0.00 (0.01; 2.56) | no runaway |
| C3 | 0.50 | 0.12 / 0.69 (1.23; 0.01) | 0.00 / 0.00 (0.00; 0.00) | 0.00 / 0.00 (0.00; 0.01) | 0.05 / 0.02 (0.09; 0.03) | 0.00 / 0.03 (0.21; 3.31) | **runs away**: x = 0 holds too much |
| C3 | 0.75 | 0.06 / 0.66 (1.70; 0.00) | 1.43 / 1.90 (3.47; 2.06) | 0.00 / 0.00 (0.02; 0.20) | 0.02 / 0.01 (0.04; 0.04) | 0.01 / 0.01 (0.10; 1.08) | **runs away**: x = 0 grows and holds too much |
| C3 | 0.95 | 0.09 / 0.05 (0.39; 0.00) | 0.70 / 0.60 (1.61; 4.10) | 0.00 / 0.00 (0.03; 0.05) | 0.01 / 0.03 (0.17; 0.03) | 0.00 / 0.00 (0.00; 0.58) | no runaway |
| C4 | 0.05 | 5.65 / 6.25 (8.53; 8.50) | 0.05 / 0.02 (0.09; 0.00) | 0.01 / 0.01 (0.03; 0.05) | 0.01 / 0.00 (0.02; 0.04) | 0.13 / 0.13 (0.20; 0.44) | no runaway |
| C4 | 0.25 | 4.18 / 2.52 (4.45; 5.32) | 0.02 / 0.02 (0.05; 0.00) | 0.00 / 0.00 (0.01; 0.11) | 0.02 / 0.01 (0.05; 0.46) | 0.68 / 0.32 (0.95; 2.56) | no runaway |
| C4 | 0.50 | 0.05 / 0.05 (0.16; 0.01) | 0.01 / 0.01 (0.05; 0.00) | 0.00 / 0.00 (0.01; 0.01) | 0.01 / 0.00 (0.06; 0.03) | 0.38 / 0.49 (0.73; 3.31) | no runaway |
| C4 | 0.75 | 0.01 / 0.02 (0.05; 0.00) | 3.40 / 4.37 (5.29; 2.06) | 0.00 / 0.08 (0.17; 0.20) | 0.00 / 0.01 (0.06; 0.04) | 0.21 / 0.47 (0.60; 1.08) | **runs away**: x = max holds too much |
| C4 | 0.95 | 0.01 / 0.01 (0.07; 0.00) | 2.79 / 2.28 (3.57; 4.10) | 0.00 / 0.00 (0.01; 0.05) | 0.00 / 0.01 (0.03; 0.03) | 0.04 / 0.01 (0.08; 0.58) | no runaway |
| N | 0.05 | 60.40 / 627.68 (641.64; 8.50) | 61.90 / 578.01 (578.87; 0.00) | 387.20 / 626.22 (685.55; 0.05) | 58.77 / 626.13 (629.95; 0.04) | 49.85 / 332.21 (333.37; 0.44) | **runs away**: x = 0 grows and holds too much; x = max grows and holds too much; z = 0 holds too much; z = max grows and holds too much; top grows and holds too much |
| N | 0.25 | 35.35 / 635.76 (637.92; 5.32) | 56.27 / 574.29 (575.46; 0.00) | 304.46 / 622.55 (682.52; 0.11) | 47.17 / 631.85 (643.74; 0.46) | 37.03 / 334.62 (334.95; 2.56) | **runs away**: x = 0 grows and holds too much; x = max grows and holds too much; z = 0 grows and holds too much; z = max grows and holds too much; top grows and holds too much |
| N | 0.50 | 1.40 / 634.54 (634.96; 0.01) | 1.64 / 572.28 (573.20; 0.00) | 3.85 / 619.29 (620.81; 0.01) | 2.20 / 633.69 (634.00; 0.03) | 0.73 / 336.11 (336.52; 3.31) | **runs away**: x = 0 grows and holds too much; x = max grows and holds too much; z = 0 grows and holds too much; z = max grows and holds too much; top grows and holds too much |
| N | 0.75 | 1.47 / 629.15 (630.41; 0.00) | 1.05 / 570.63 (571.47; 2.06) | 0.00 / 615.71 (617.85; 0.20) | 11.81 / 633.15 (633.67; 0.04) | 2.55 / 335.13 (335.58; 1.08) | **runs away**: x = 0 grows and holds too much; x = max grows and holds too much; z = 0 grows and holds too much; z = max grows and holds too much; top grows and holds too much |
| N | 0.95 | 1.47 / 615.19 (625.51; 0.00) | 8.65 / 544.71 (570.60; 4.10) | 0.00 / 577.50 (612.38; 0.05) | 21.71 / 630.14 (635.53; 0.03) | 3.80 / 325.26 (335.48; 0.58) | **runs away**: x = 0 grows and holds too much; x = max grows and holds too much; z = 0 grows and holds too much; z = max grows and holds too much; top grows and holds too much |
| Nfix | 0.05 | 1.06 / 1.94 (6.00; 8.50) | 4.20 / 9.80 (14.57; 0.00) | 12.61 / 3.08 (27.01; 0.05) | 9.08 / 3.84 (16.81; 0.04) | 0.72 / 0.75 (2.60; 0.44) | **runs away**: x = max grows and holds too much; z = 0 holds too much; z = max holds too much; top holds too much |
| Nfix | 0.25 | 0.66 / 4.15 (6.81; 5.32) | 0.20 / 6.25 (13.91; 0.00) | 1.42 / 7.93 (11.07; 0.11) | 0.46 / 6.68 (11.83; 0.46) | 0.32 / 2.02 (5.11; 2.56) | **runs away**: x = 0 grows; x = max grows and holds too much; z = 0 grows and holds too much; z = max grows and holds too much; top grows |
| Nfix | 0.50 | 0.01 / 0.00 (0.09; 0.01) | 0.00 / 1.17 (6.91; 0.00) | 0.00 / 3.44 (8.54; 0.01) | 0.01 / 1.16 (3.09; 0.03) | 0.04 / 0.08 (0.86; 3.31) | **runs away**: x = max grows and holds too much; z = 0 grows and holds too much; z = max grows and holds too much |
| Nfix | 0.75 | 0.05 / 0.03 (0.25; 0.00) | 0.02 / 0.30 (0.56; 2.06) | 0.00 / 0.00 (0.00; 0.20) | 0.05 / 0.04 (0.25; 0.04) | 0.02 / 0.01 (0.22; 1.08) | no runaway |
| Nfix | 0.95 | 0.06 / 0.05 (0.24; 0.00) | 0.34 / 0.52 (1.12; 4.10) | 0.00 / 0.00 (0.00; 0.05) | 0.06 / 0.05 (0.25; 0.03) | 0.00 / 0.00 (0.08; 0.58) | no runaway |

C2: FAILS the edge check
C3: FAILS the edge check
C4: FAILS the edge check
N: FAILS the edge check
Nfix: FAILS the edge check
