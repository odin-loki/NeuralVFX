## 3D detail layer alone: the volume runtime's / rollout3d::detail_step (median over rounds [95% bootstrap]; thread CPU time on 1 thread, wall time on more)

| threads | ISA | reference ms | runtime ms | runtime / reference | steps compared | differing cells |
|---:|---|---|---|---|---:|---:|
| 1 | avx2 | 63.39 [62.06, 64.22] | 11.02 [10.81, 11.41] | 0.176 [0.173, 0.179] | 60 | 0 |
| 1 | avx512 | 63.99 [63.32, 64.96] | 9.97 [9.78, 10.08] | 0.155 [0.153, 0.158] | 60 | 0 |
| 4 | avx2 | 145.59 [119.48, 167.78] | 23.67 [16.72, 38.94] | 0.174 [0.132, 0.216] | 60 | 0 |
| 4 | avx512 | 121.70 [111.60, 145.86] | 23.09 [15.97, 26.18] | 0.181 [0.126, 0.211] | 60 | 0 |
