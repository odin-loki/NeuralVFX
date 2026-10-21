## 3D volume step: new / old (median over rounds [95% bootstrap over rounds]; thread CPU time on 1 thread, wall time on more)

| threads | ISA | old step ms | new step ms | new / old | frames compared | differing |
|---:|---|---|---|---|---:|---:|
| 1 | avx2 | 64.60 [61.71, 69.49] | 25.63 [24.71, 26.80] | 0.396 [0.389, 0.402] | 60 | 0 |
| 1 | avx512 | 65.61 [64.28, 75.48] | 24.60 [24.05, 29.06] | 0.375 [0.365, 0.384] | 60 | 0 |
| 4 | avx2 | 20.58 [20.31, 22.39] | 9.65 [9.09, 10.40] | 0.460 [0.448, 0.474] | 60 | 0 |
| 4 | avx512 | 25.15 [23.83, 27.20] | 11.28 [10.96, 11.86] | 0.451 [0.431, 0.464] | 60 | 0 |
