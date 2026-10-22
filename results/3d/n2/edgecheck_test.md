Side heat, mean over 5 to 15 s / mean over 20 to 30 s (largest over 5 to 30 s; the simulation's largest):

| model | wind | x = 0 | x = max | z = 0 | z = max | top | verdict |
|---|---:|---|---|---|---|---|---|
| N | 0.05 | 284.37 / 626.00 (644.40; 8.50) | 237.76 / 578.52 (578.99; 0.00) | 570.32 / 625.94 (688.15; 0.05) | 264.64 / 624.33 (629.10; 0.04) | 167.89 / 332.10 (333.32; 0.44) | **runs away**: x = 0 grows and holds too much; x = max grows and holds too much; z = 0 holds too much; z = max grows and holds too much; top holds too much |
| N | 0.25 | 40.57 / 635.14 (651.15; 5.32) | 44.20 / 574.64 (575.34; 0.00) | 348.30 / 622.72 (680.30; 0.11) | 34.91 / 632.54 (639.74; 0.46) | 37.28 / 334.70 (335.21; 2.56) | **runs away**: x = 0 grows and holds too much; x = max grows and holds too much; z = 0 holds too much; z = max grows and holds too much; top grows and holds too much |
| N | 0.50 | 1.32 / 548.39 (593.24; 0.01) | 25.54 / 426.98 (565.03; 0.00) | 10.29 / 496.52 (614.38; 0.01) | 39.07 / 477.40 (571.44; 0.03) | 7.46 / 255.28 (309.72; 3.31) | **runs away**: x = 0 grows and holds too much; x = max grows and holds too much; z = 0 grows and holds too much; z = max grows and holds too much; top grows and holds too much |
| N | 0.75 | 1.11 / 606.69 (629.15; 0.00) | 1.87 / 547.44 (571.23; 2.06) | 0.00 / 558.10 (618.50; 0.20) | 17.26 / 635.01 (662.96; 0.04) | 2.28 / 325.63 (335.42; 1.08) | **runs away**: x = 0 grows and holds too much; x = max grows and holds too much; z = 0 grows and holds too much; z = max grows and holds too much; top grows and holds too much |
| N | 0.95 | 7.87 / 625.00 (625.51; 0.00) | 4.87 / 569.40 (570.50; 4.10) | 11.10 / 610.40 (615.11; 0.05) | 4.31 / 634.81 (647.46; 0.03) | 1.86 / 334.85 (335.42; 0.58) | **runs away**: x = 0 grows and holds too much; x = max grows and holds too much; z = 0 grows and holds too much; z = max grows and holds too much; top grows and holds too much |
| learned | 0.05 | 9.88 / 7.92 (13.29; 8.50) | 0.05 / 0.05 (0.11; 0.00) | 0.04 / 0.01 (0.14; 0.05) | 0.01 / 0.01 (0.04; 0.04) | 0.45 / 0.11 (0.91; 0.44) | no runaway |
| learned | 0.25 | 4.22 / 4.02 (5.69; 5.32) | 0.02 / 0.03 (0.05; 0.00) | 0.00 / 0.00 (0.01; 0.11) | 0.01 / 0.01 (0.03; 0.46) | 0.47 / 0.22 (0.59; 2.56) | no runaway |
| learned | 0.50 | 0.02 / 0.02 (0.09; 0.01) | 0.01 / 0.02 (0.04; 0.00) | 0.00 / 0.00 (0.00; 0.01) | 0.01 / 0.00 (0.03; 0.03) | 0.33 / 0.34 (0.77; 3.31) | no runaway |
| learned | 0.75 | 0.06 / 0.01 (0.18; 0.00) | 1.80 / 2.18 (2.37; 2.06) | 0.00 / 0.00 (0.01; 0.20) | 0.01 / 0.01 (0.04; 0.04) | 0.12 / 0.18 (0.23; 1.08) | no runaway |
| learned | 0.95 | 0.02 / 0.01 (0.08; 0.00) | 3.04 / 3.79 (4.73; 4.10) | 0.00 / 0.00 (0.01; 0.05) | 0.02 / 0.00 (0.05; 0.03) | 0.06 / 0.00 (0.11; 0.58) | no runaway |

N: FAILS the edge check
learned: passes the edge check
