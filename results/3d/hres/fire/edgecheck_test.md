Side heat, mean over 5 to 15 s / mean over 20 to 30 s (largest over 5 to 30 s; the calibrated bound on the largest):

| model | wind | x = 0 | x = max | z = 0 | z = max | top | verdict |
|---|---:|---|---|---|---|---|---|
| N2 | 0.05 | 5.18 / 5.70 (6.97; bound 115.76) | 0.06 / 0.03 (0.10; bound 4.00) | 0.02 / 0.01 (0.04; bound 4.84) | 0.03 / 0.02 (0.11; bound 7.75) | 0.17 / 0.21 (0.34; bound 7.73) | no runaway |
| N2 | 0.25 | 3.05 / 4.70 (6.69; bound 77.54) | 0.02 / 0.02 (0.04; bound 4.00) | 0.00 / 0.00 (0.01; bound 5.30) | 0.00 / 0.01 (0.05; bound 4.53) | 0.27 / 0.26 (0.43; bound 18.97) | no runaway |
| N2 | 0.50 | 0.01 / 0.02 (0.07; bound 4.95) | 0.01 / 0.01 (0.03; bound 4.01) | 0.00 / 0.00 (0.00; bound 6.91) | 0.02 / 0.00 (0.10; bound 5.10) | 0.41 / 0.31 (0.58; bound 38.15) | no runaway |
| N2 | 0.75 | 0.02 / 0.03 (0.10; bound 4.00) | 1.80 / 2.12 (2.71; bound 21.13) | 0.00 / 0.00 (0.01; bound 6.38) | 0.01 / 0.01 (0.09; bound 6.35) | 0.22 / 0.29 (0.72; bound 19.17) | no runaway |
| N2 | 0.95 | 0.00 / 0.02 (0.09; bound 4.00) | 2.64 / 2.54 (3.48; bound 51.91) | 0.00 / 0.00 (0.00; bound 4.73) | 0.00 / 0.01 (0.03; bound 6.22) | 0.07 / 0.00 (0.13; bound 19.96) | no runaway |
| learned | 0.05 | 33.39 / 31.76 (52.11; bound 115.76) | 0.01 / 0.00 (0.03; bound 4.00) | 0.00 / 0.00 (0.00; bound 4.84) | 0.00 / 0.00 (0.01; bound 7.75) | 0.33 / 0.18 (1.17; bound 7.73) | no runaway |
| learned | 0.25 | 17.07 / 15.40 (27.96; bound 77.54) | 0.00 / 0.00 (0.02; bound 4.00) | 0.00 / 0.00 (0.00; bound 5.30) | 0.00 / 0.00 (0.00; bound 4.53) | 1.03 / 0.59 (3.07; bound 18.97) | no runaway |
| learned | 0.50 | 0.82 / 0.08 (3.43; bound 4.95) | 0.00 / 0.00 (0.00; bound 4.01) | 0.00 / 0.00 (0.00; bound 6.91) | 0.00 / 0.00 (0.00; bound 5.10) | 1.54 / 1.83 (8.01; bound 38.15) | no runaway |
| learned | 0.75 | 0.00 / 0.00 (0.01; bound 4.00) | 1.01 / 0.65 (3.43; bound 21.13) | 0.00 / 0.00 (0.00; bound 6.38) | 0.00 / 0.00 (0.00; bound 6.35) | 0.32 / 0.12 (1.92; bound 19.17) | no runaway |
| learned | 0.95 | 0.56 / 0.02 (1.05; bound 4.00) | 6.31 / 6.65 (14.57; bound 51.91) | 0.27 / 0.05 (2.36; bound 4.73) | 0.00 / 0.00 (0.00; bound 6.22) | 0.58 / 0.24 (3.31; bound 19.96) | no runaway |

N2: passes the edge check
learned: passes the edge check
