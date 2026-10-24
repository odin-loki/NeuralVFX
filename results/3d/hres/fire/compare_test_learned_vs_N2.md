# learned minus N2 (test; paired, 95% bootstrap, 10,000 resamples, seed 2026)

Endless statistics against real runs (6 settings, 3 runs pooled per setting; lower is better but for PSNR and the motion ratio, whose ideal is 1):

| statistic | learned | N2 | floor (other real runs) | difference |
|---|---:|---:|---:|---|
| 3D spectrum distance | 0.183 | 0.264 | 0.045 | -0.080 [-0.101, -0.056] (better) |
| \|ln motion ratio\| | 0.105 | 0.446 | 0.091 | -0.340 [-0.553, -0.131] (better) |
| motion ratio | 0.988 | 0.660 | 0.995 | +0.328 [+0.182, +0.473] (context) |
| coverage distance | 0.0021 | 0.0037 | 0.0020 | -0.0016 [-0.0038, +0.0003] (tie) |
| emission distance | 0.0030 | 0.0047 | 0.0012 | -0.0017 [-0.0037, +0.0003] (tie) |
| mean picture PSNR (dB) | 32.98 | 30.78 | 41.07 | +2.20 [+1.28, +3.14] (better) |

Per setting (learned / N2):

| setting (intensity, wind, turbulence) | motion ratio | coverage distance | emission distance | 3D spectrum distance | mean picture PSNR (dB) |
|---|---|---|---|---|---|
| 0.68, 0.07, 0.42 | 0.84 / 0.57 | 0.0017 / 0.0045 | 0.0019 / 0.0039 | 0.134 / 0.233 | 33.64 / 31.95 |
| 0.52, 0.93, 0.52 | 0.99 / 0.91 | 0.0007 / 0.0007 | 0.0016 / 0.0068 | 0.188 / 0.293 | 34.97 / 31.17 |
| 0.42, 0.22, 0.68 | 1.07 / 0.52 | 0.0015 / 0.0035 | 0.0018 / 0.0055 | 0.207 / 0.258 | 33.71 / 31.52 |
| 0.78, 0.48, 0.32 | 1.01 / 0.46 | 0.0020 / 0.0084 | 0.0057 / 0.0034 | 0.161 / 0.256 | 30.75 / 30.20 |
| 0.33, 0.72, 0.42 | 1.19 / 0.83 | 0.0013 / 0.0005 | 0.0017 / 0.0039 | 0.230 / 0.331 | 32.35 / 30.93 |
| 0.63, 0.38, 0.88 | 0.82 / 0.67 | 0.0053 / 0.0042 | 0.0053 / 0.0046 | 0.179 / 0.210 | 32.50 / 28.93 |
