Side heat, mean over 5 to 15 s / mean over 20 to 30 s (largest over 5 to 30 s; the calibrated bound on the largest):

| model | wind | x = 0 | x = max | z = 0 | z = max | top | verdict |
|---|---:|---|---|---|---|---|---|
| learned | 0.05 | 1.93 / 3.55 (4.56; bound 9.35) | 0.36 / 0.04 (0.47; bound 1.02) | 0.15 / 0.27 (0.72; bound 1.12) | 0.23 / 0.51 (0.79; bound 1.25) | 0.46 / 0.53 (1.32; bound 2.44) | no runaway |
| learned | 0.25 | 1.59 / 3.17 (3.80; bound 7.15) | 0.31 / 0.43 (0.52; bound 1.00) | 0.07 / 0.49 (0.88; bound 1.58) | 0.01 / 0.37 (0.45; bound 1.85) | 0.82 / 1.09 (1.47; bound 4.69) | no runaway |
| learned | 0.50 | 0.44 / 0.67 (1.00; bound 1.01) | 0.69 / 0.40 (0.78; bound 1.03) | 0.23 / 0.42 (0.50; bound 2.04) | 0.03 / 0.09 (0.12; bound 1.34) | 1.07 / 1.31 (1.49; bound 5.56) | no runaway |
| learned | 0.75 | 0.39 / 0.52 (0.67; bound 1.04) | 1.17 / 1.98 (3.02; bound 6.95) | 0.15 / 0.17 (0.23; bound 1.69) | 0.02 / 0.15 (0.25; bound 1.41) | 0.77 / 1.22 (1.37; bound 5.37) | no runaway |
| learned | 0.95 | 0.40 / 0.40 (0.61; bound 1.03) | 1.52 / 1.57 (1.82; bound 8.14) | 0.09 / 0.26 (0.30; bound 1.04) | 0.02 / 0.12 (0.15; bound 6.95) | 0.49 / 1.01 (1.20; bound 1.50) | no runaway |

learned: passes the edge check
