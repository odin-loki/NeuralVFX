// ROC-AUC for the DCM search (declared in include/neuralfx/dcm/search.hpp). Ported from the owner's CameraDetector,
// cabinlab/src/diffusion/probe.cpp (roc_auc only; the rest of that file needs LibTorch).
#include <neuralfx/dcm/search.hpp>

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace nfx::dcm {

double roc_auc(std::span<const float> scores, std::span<const int> y) {
  if (scores.size() != y.size()) throw std::invalid_argument("dcm: roc_auc: sizes differ");
  std::vector<std::size_t> order(scores.size());
  std::iota(order.begin(), order.end(), 0);
  std::ranges::sort(order, [&](std::size_t a, std::size_t b) { return scores[a] < scores[b]; });
  double rank_sum = 0.0;
  std::size_t pos = 0, neg = 0;
  std::size_t i = 0;
  while (i < order.size()) {  // tied scores share their mean rank
    std::size_t j = i;
    while (j + 1 < order.size() && scores[order[j + 1]] == scores[order[i]]) ++j;
    const double rank = (static_cast<double>(i) + static_cast<double>(j)) / 2.0 + 1.0;
    for (std::size_t k = i; k <= j; ++k) {
      if (y[order[k]]) {
        rank_sum += rank;
        ++pos;
      } else {
        ++neg;
      }
    }
    i = j + 1;
  }
  if (!pos || !neg) return std::numeric_limits<double>::quiet_NaN();
  return (rank_sum - static_cast<double>(pos) * (static_cast<double>(pos) + 1.0) / 2.0) /
         (static_cast<double>(pos) * static_cast<double>(neg));
}

}  // namespace nfx::dcm
