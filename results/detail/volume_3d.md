## 3D volume step: new / old (median over rounds [95% bootstrap over rounds]; thread CPU time on 1 thread, wall time on more)

| threads | ISA | old step ms | new step ms | new / old | frames compared | differing |
|---:|---|---|---|---|---:|---:|
| 1 | avx2 | 86.45 [81.91, 91.95] | 26.78 [26.18, 29.34] | 0.319 [0.300, 0.326] | 60 | 0 |
| 1 | avx512 | 83.10 [81.64, 84.44] | 25.76 [25.55, 27.19] | 0.309 [0.306, 0.324] | 60 | 0 |
| 4 | avx2 | 197.60 [185.61, 225.51] | 35.31 [28.79, 55.82] | 0.184 [0.133, 0.300] | 60 | 0 |
| 4 | avx512 | 172.36 [162.27, 198.89] | 53.33 [28.24, 74.07] | 0.303 [0.152, 0.372] | 60 | 0 |
