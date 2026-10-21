# Statistics for the quiet re-timing's summaries (tools/quiet/summary7.sh), as an awk library: awk -f stats.awk -f ...
#
# mci(a, n): the median of a[1..n] (sorted in place) and a distribution-free confidence interval for it from order
# statistics: (x_k, x_{n+1-k}) with the largest k whose coverage, 1 - 2 P(Bin(n, 1/2) <= k - 1), is at least 95%; with
# fewer than six values no such k exists and the interval is the range, its coverage below 95% (n = 5: 93.8%). Sets
# MED, LO, HI and COV (the interval's coverage).
# gmean_ci(a, n, B, seed): the geometric mean of a[1..n] and its 95% percentile bootstrap interval over the values (B
# resamples). Sets GM, GLO, GHI. mean_ci(a, n, B, seed) likewise for the arithmetic mean: MEAN, MLO, MHI.

function sortn(a, n,   i, t, end) {  # heapsort, ascending
  for (i = int(n / 2); i >= 1; i--) sift(a, i, n)
  for (end = n; end > 1; end--) { t = a[1]; a[1] = a[end]; a[end] = t; sift(a, 1, end - 1) }
}

function sift(a, i, n,   c, t) {
  while (2 * i <= n) {
    c = 2 * i
    if (c < n && a[c + 1] > a[c]) c++
    if (a[i] >= a[c]) return
    t = a[i]; a[i] = a[c]; a[c] = t; i = c
  }
}

function binom_cdf(k, n,   i, p, s) {  # P(X <= k), X ~ Bin(n, 1/2)
  if (k < 0) return 0
  p = 2 ^ -n; s = 0
  for (i = 0; i <= k; i++) { s += p; p = p * (n - i) / (i + 1) }
  return s
}

function mci(a, n,   k, best) {
  sortn(a, n)
  MED = (n % 2) ? a[(n + 1) / 2] : (a[n / 2] + a[n / 2 + 1]) / 2
  best = 1
  for (k = 1; k <= n / 2; k++) if (1 - 2 * binom_cdf(k - 1, n) >= 0.95) best = k
  LO = a[best]; HI = a[n + 1 - best]; COV = 1 - 2 * binom_cdf(best - 1, n)
}

function lcg() { RNG = (RNG * 16807) % 2147483647; return (RNG - 1) / 2147483646 }  # Park and Miller, exact in doubles

function boot(a, n, B, seed, geo,   b, i, s, m, r, cnt) {
  RNG = seed % 2147483646 + 1
  for (b = 1; b <= B; b++) {
    s = 0
    for (i = 1; i <= n; i++) { r = a[int(lcg() * n) + 1]; s += geo ? log(r) : r }
    m[b] = geo ? exp(s / n) : s / n
  }
  sortn(m, B)
  BLO = m[int(0.025 * B) + 1]; BHI = m[int(0.975 * B)]
}

function gmean_ci(a, n, B, seed,   i, s) {
  s = 0
  for (i = 1; i <= n; i++) s += log(a[i])
  GM = exp(s / n)
  boot(a, n, B, seed, 1); GLO = BLO; GHI = BHI
}

function mean_ci(a, n, B, seed,   i, s) {
  s = 0
  for (i = 1; i <= n; i++) s += a[i]
  MEAN = s / n
  boot(a, n, B, seed, 0); MLO = BLO; MHI = BHI
}
