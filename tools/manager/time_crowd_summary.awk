NR == 1 { for (i = 1; i <= NF; i++) col[$i] = i; next }
{
  l = $1; if (!(l in n)) { order[++labels] = l; n[l] = 0 }
  k = ++n[l]
  split("cpu_mean cpu_median cpu_p99 wall_median wall_p99 plan_mean overhead_mean sims proxied memory_mb rss_mb", f, " ")
  for (j in f) v[l, f[j], k] = $(col[f[j]])
  budget[l] = $(col["budget_ms"])
}
function stat(l, name,   m, i, j, t, a) {
  m = n[l]
  for (i = 1; i <= m; i++) a[i] = v[l, name, i] + 0
  for (i = 2; i <= m; i++) { t = a[i]; for (j = i - 1; j >= 1 && a[j] > t; j--) a[j + 1] = a[j]; a[j + 1] = t }
  return sprintf("%.2f (%.2f to %.2f)", (m % 2) ? a[(m + 1) / 2] : (a[m / 2] + a[m / 2 + 1]) / 2, a[1], a[m])
}
END {
  print "label,budget_ms,rounds,cpu_mean_ms,cpu_median_ms,cpu_p99_ms,wall_median_ms,wall_p99_ms,plan_ms,overhead_ms,sims,proxied,memory_mb,rss_mb"
  for (o = 1; o <= labels; o++) {
    l = order[o]
    printf "%s,%s,%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", l, budget[l], n[l], stat(l, "cpu_mean"), stat(l, "cpu_median"), stat(l, "cpu_p99"),
           stat(l, "wall_median"), stat(l, "wall_p99"), stat(l, "plan_mean"), stat(l, "overhead_mean"), stat(l, "sims"), stat(l, "proxied"), stat(l, "memory_mb"), stat(l, "rss_mb")
  }
}
