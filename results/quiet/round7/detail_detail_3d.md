## 3D detail layer alone: the volume runtime's / rollout3d::detail_step (median over rounds [95% bootstrap]; thread CPU time on 1 thread, wall time on more)

| threads | ISA | reference ms | runtime ms | runtime / reference | steps compared | differing cells |
|---:|---|---|---|---|---:|---:|
| 1 | avx2 | 47.28 [44.88, 56.65] | 10.42 [10.11, 11.99] | 0.223 [0.204, 0.227] | 60 | 0 |
| 1 | avx512 | 49.75 [48.44, 50.26] | 9.99 [9.77, 10.26] | 0.204 [0.199, 0.207] | 60 | 0 |
| 4 | avx2 | 16.90 [16.58, 17.53] | 4.13 [4.05, 4.23] | 0.242 [0.239, 0.251] | 60 | 0 |
| 4 | avx512 | 16.28 [16.00, 17.16] | 3.66 [3.55, 3.77] | 0.224 [0.218, 0.229] | 60 | 0 |
