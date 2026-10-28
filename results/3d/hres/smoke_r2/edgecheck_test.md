Side heat, mean over 5 to 15 s / mean over 20 to 30 s (largest over 5 to 30 s; the calibrated bound on the largest):

| model | wind | x = 0 | x = max | z = 0 | z = max | top | verdict |
|---|---:|---|---|---|---|---|---|
| learned | 0.05 | 0.14 / 0.27 (1.55; bound 33.38) | 0.08 / 0.09 (0.18; bound 4.86) | 0.01 / 0.11 (0.55; bound 8.19) | 0.38 / 0.10 (0.99; bound 8.10) | 0.27 / 0.90 (3.11; bound 11.12) | no runaway |
| learned | 0.25 | 0.02 / 0.01 (0.09; bound 26.80) | 0.11 / 0.15 (0.22; bound 4.38) | 0.05 / 0.03 (0.12; bound 7.67) | 0.15 / 0.04 (0.80; bound 10.06) | 0.47 / 0.25 (1.21; bound 18.39) | no runaway |
| learned | 0.50 | 0.01 / 0.01 (0.03; bound 4.29) | 0.16 / 0.17 (0.24; bound 4.32) | 0.10 / 0.06 (0.22; bound 5.24) | 0.01 / 0.15 (1.53; bound 5.34) | 0.50 / 0.31 (1.45; bound 23.01) | no runaway |
| learned | 0.75 | 0.00 / 0.02 (0.04; bound 4.74) | 0.31 / 0.40 (1.05; bound 25.43) | 0.05 / 0.07 (0.12; bound 8.76) | 0.02 / 0.10 (0.22; bound 8.61) | 0.43 / 0.16 (1.45; bound 18.19) | no runaway |
| learned | 0.95 | 0.02 / 0.00 (0.05; bound 4.48) | 0.33 / 1.03 (2.34; bound 35.60) | 0.03 / 0.04 (0.11; bound 5.86) | 0.00 / 0.17 (0.38; bound 6.74) | 0.33 / 0.24 (1.21; bound 9.64) | no runaway |
| smoke2 | 0.05 | 2.36 / 2.28 (2.85; bound 33.38) | 0.09 / 0.76 (1.71; bound 4.86) | 0.12 / 0.42 (0.73; bound 8.19) | 0.09 / 1.61 (2.40; bound 8.10) | 0.28 / 1.48 (2.03; bound 11.12) | no runaway |
| smoke2 | 0.25 | 0.23 / 0.76 (1.04; bound 26.80) | 0.17 / 1.22 (2.28; bound 4.38) | 0.00 / 0.05 (0.07; bound 7.67) | 1.06 / 2.91 (3.17; bound 10.06) | 0.58 / 1.29 (1.76; bound 18.39) | no runaway |
| smoke2 | 0.50 | 0.01 / 0.18 (0.40; bound 4.29) | 0.53 / 3.02 (4.15; bound 4.32) | 0.00 / 0.01 (0.02; bound 5.24) | 0.80 / 1.19 (1.76; bound 5.34) | 0.49 / 0.76 (1.25; bound 23.01) | no runaway |
| smoke2 | 0.75 | 0.00 / 0.00 (0.01; bound 4.74) | 1.29 / 2.90 (3.92; bound 25.43) | 0.01 / 0.01 (0.04; bound 8.76) | 0.27 / 0.70 (1.14; bound 8.61) | 0.47 / 0.23 (0.60; bound 18.19) | no runaway |
| smoke2 | 0.95 | 0.04 / 1.19 (2.27; bound 4.48) | 1.59 / 1.87 (2.45; bound 35.60) | 0.01 / 0.01 (0.04; bound 5.86) | 0.48 / 3.26 (3.68; bound 6.74) | 0.03 / 0.18 (0.29; bound 9.64) | **runs away**: z = max grows |

learned: passes the edge check
smoke2: FAILS the edge check
