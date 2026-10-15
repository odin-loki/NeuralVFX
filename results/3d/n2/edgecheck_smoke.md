Side heat, mean over 5 to 15 s / mean over 20 to 30 s (largest over 5 to 30 s; the simulation's largest):

| model | wind | x = 0 | x = max | z = 0 | z = max | top | verdict |
|---|---:|---|---|---|---|---|---|
| smoke2 | 0.05 | 2.39 / 1.50 (2.52; 3.01) | 0.61 / 3.13 (3.79; 0.00) | 0.05 / 0.05 (0.10; 0.00) | 0.43 / 1.03 (1.85; 0.00) | 0.31 / 1.47 (1.68; 0.00) | **runs away**: x = max grows and holds too much; z = max holds too much; top grows and holds too much |
| smoke2 | 0.25 | 0.94 / 2.09 (2.51; 3.87) | 0.13 / 0.59 (0.95; 0.00) | 0.00 / 0.03 (0.03; 0.02) | 1.06 / 4.68 (5.22; 0.09) | 0.49 / 1.33 (1.62; 1.24) | **runs away**: z = max grows and holds too much |
| smoke2 | 0.50 | 0.01 / 0.07 (0.12; 0.00) | 0.35 / 0.99 (1.31; 0.00) | 0.00 / 0.01 (0.01; 0.00) | 0.08 / 1.57 (2.33; 0.09) | 0.48 / 0.87 (1.12; 1.35) | **runs away**: x = max holds too much; z = max grows and holds too much |
| smoke2 | 0.75 | 0.10 / 0.20 (0.25; 0.00) | 0.95 / 2.68 (3.24; 1.48) | 0.00 / 0.02 (0.05; 0.02) | 1.49 / 4.68 (5.02; 0.02) | 0.48 / 0.20 (0.73; 0.69) | **runs away**: x = max grows; z = max grows and holds too much |
| smoke2 | 0.95 | 0.00 / 0.36 (0.78; 0.00) | 1.62 / 2.24 (3.25; 2.33) | 0.06 / 0.01 (0.11; 0.05) | 0.33 / 2.82 (3.94; 0.05) | 0.04 / 0.01 (0.09; 0.24) | **runs away**: z = max grows and holds too much |
| smokeN | 0.05 | 1.68 / 72.11 (98.73; 3.01) | 0.00 / 58.29 (76.27; 0.00) | 0.02 / 2.07 (7.65; 0.00) | 0.10 / 189.51 (209.66; 0.00) | 0.21 / 65.04 (94.94; 0.00) | **runs away**: x = 0 grows and holds too much; x = max grows and holds too much; z = 0 grows and holds too much; z = max grows and holds too much; top grows and holds too much |
| smokeN | 0.25 | 19.15 / 113.01 (188.86; 3.87) | 21.44 / 135.06 (206.16; 0.00) | 0.00 / 97.33 (213.06; 0.02) | 114.07 / 183.87 (213.33; 0.09) | 16.64 / 86.79 (123.22; 1.24) | **runs away**: x = 0 grows and holds too much; x = max grows and holds too much; z = 0 grows and holds too much; z = max holds too much; top grows and holds too much |
| smokeN | 0.50 | 0.00 / 0.69 (5.57; 0.00) | 0.02 / 0.31 (0.77; 0.00) | 0.00 / 0.00 (0.01; 0.00) | 0.00 / 1.69 (13.86; 0.09) | 0.29 / 0.30 (0.87; 1.35) | **runs away**: x = 0 grows and holds too much; z = max grows and holds too much |
| smokeN | 0.75 | 0.00 / 5.97 (34.66; 0.00) | 1.05 / 9.31 (50.35; 1.48) | 0.00 / 0.30 (3.61; 0.02) | 0.00 / 46.88 (202.49; 0.02) | 0.10 / 5.39 (32.29; 0.69) | **runs away**: x = 0 grows and holds too much; x = max grows and holds too much; z = 0 holds too much; z = max grows and holds too much; top grows and holds too much |
| smokeN | 0.95 | 0.00 / 0.01 (0.05; 0.00) | 0.94 / 5.88 (15.28; 2.33) | 0.00 / 0.26 (1.16; 0.05) | 0.00 / 2.93 (15.13; 0.05) | 0.00 / 1.65 (8.34; 0.24) | **runs away**: x = max grows and holds too much; z = 0 holds too much; z = max grows and holds too much; top grows and holds too much |

smoke2: FAILS the edge check
smokeN: FAILS the edge check
