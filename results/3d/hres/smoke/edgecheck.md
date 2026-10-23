Side heat, mean over 5 to 15 s / mean over 20 to 30 s (largest over 5 to 30 s; the simulation's largest):

| model | wind | x = 0 | x = max | z = 0 | z = max | top | verdict |
|---|---:|---|---|---|---|---|---|
| smoke2 | 0.05 | 2.28 / 2.09 (2.59; 24.44) | 0.11 / 0.84 (1.61; 0.00) | 0.04 / 0.15 (0.21; 0.00) | 0.33 / 2.48 (2.84; 0.00) | 0.42 / 1.73 (1.96; 0.80) | **runs away**: x = max grows and holds too much; z = max grows and holds too much; top grows |
| smoke2 | 0.25 | 0.64 / 0.52 (1.37; 19.13) | 0.21 / 1.73 (2.91; 0.00) | 0.00 / 0.01 (0.02; 0.18) | 0.79 / 2.68 (3.09; 0.38) | 0.69 / 0.95 (1.08; 2.02) | **runs away**: x = max grows and holds too much; z = max grows and holds too much |
| smoke2 | 0.50 | 0.01 / 0.01 (0.08; 0.01) | 0.08 / 1.14 (1.44; 0.00) | 0.00 / 0.01 (0.02; 0.00) | 0.64 / 2.25 (2.48; 0.00) | 0.46 / 0.72 (1.10; 8.02) | **runs away**: x = max grows and holds too much; z = max grows and holds too much |
| smoke2 | 0.75 | 0.01 / 0.42 (0.60; 0.00) | 0.84 / 2.35 (2.83; 4.52) | 0.00 / 0.00 (0.01; 0.16) | 0.95 / 4.35 (4.94; 0.19) | 0.42 / 0.21 (0.64; 1.78) | **runs away**: x = max grows; z = max grows and holds too much |
| smoke2 | 0.95 | 0.00 / 0.00 (0.01; 0.00) | 1.67 / 3.39 (4.65; 8.04) | 0.01 / 0.02 (0.05; 0.30) | 0.09 / 0.33 (0.45; 0.34) | 0.09 / 0.11 (0.33; 0.63) | no runaway |
| smoke_hres | 0.05 | 18.63 / 20.36 (21.90; 24.44) | 0.02 / 0.06 (0.09; 0.00) | 0.01 / 0.03 (0.09; 0.00) | 0.00 / 0.01 (0.02; 0.00) | 0.43 / 0.80 (1.03; 0.80) | no runaway |
| smoke_hres | 0.25 | 8.09 / 10.02 (11.04; 19.13) | 0.04 / 0.12 (0.24; 0.00) | 0.03 / 0.03 (0.10; 0.18) | 0.00 / 0.03 (0.13; 0.38) | 0.95 / 1.10 (1.59; 2.02) | no runaway |
| smoke_hres | 0.50 | 0.31 / 1.15 (1.71; 0.01) | 0.03 / 0.03 (0.06; 0.00) | 0.02 / 0.02 (0.04; 0.00) | 0.00 / 0.00 (0.01; 0.00) | 0.85 / 1.51 (2.02; 8.02) | **runs away**: x = 0 grows and holds too much |
| smoke_hres | 0.75 | 0.73 / 2.69 (3.09; 0.00) | 2.43 / 2.83 (3.21; 4.52) | 0.02 / 0.03 (0.06; 0.16) | 0.00 / 0.03 (0.11; 0.19) | 0.89 / 1.84 (2.48; 1.78) | **runs away**: x = 0 grows and holds too much |
| smoke_hres | 0.95 | 2.95 / 4.87 (5.30; 0.00) | 4.65 / 5.69 (6.87; 8.04) | 0.03 / 0.15 (0.24; 0.30) | 0.05 / 0.04 (0.12; 0.34) | 0.77 / 1.90 (2.89; 0.63) | **runs away**: x = 0 holds too much; top holds too much |

smoke2: FAILS the edge check
smoke_hres: FAILS the edge check
