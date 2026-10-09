// Macro contexts of diffusion-context mixing (DCM): features of a scene (in NeuralVFX: of a small denoiser run on the
// coarse state, or plain coarse statistics) are reduced by PCA and clustered without labels, and the cluster ids select
// the weight sets of a context mixer (mixer.hpp). Ported from the owner's CameraDetector, cabinlab/src/diffusion
// (Phase 12, kmeans.hpp), and rewritten on std::vector<double> because this project does not use LibTorch: the
// original takes the PCA from an SVD, this one from the covariance matrix by cyclic Jacobi rotations (O(d³) per sweep,
// meant for d up to a few hundred). The contracts are the original's.
//
// Everything is deterministic for a seed: PCA fixes the sign of each component, k-means draws its k-means++ seeds from
// its own generator, Lloyd iterations and empty-cluster repairs are sequential, and clusters are numbered by decreasing
// size. The adjusted Rand index and normalised mutual information compare labellings (stability between seeds,
// agreement with hand-made contexts).
//
// Matrices are row-major std::vector<double> (or spans of them) of n rows and d columns, d passed alongside.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace nfx::dcm {

// Principal components of a set of rows.
struct Pca {
  std::size_t dims = 0;             // d
  std::vector<double> mean;         // d
  std::vector<double> components;   // k x d: orthonormal rows by decreasing variance; each row's largest-magnitude
                                    // entry is positive (the sign convention that makes the fit deterministic)
  std::vector<double> variance;     // k: variance of the fitting rows along each component
  [[nodiscard]] std::size_t count() const noexcept { return variance.size(); }
  // Coordinates (n x k) of rows x (n x d) in the component basis. Throws std::invalid_argument when d differs from
  // dims or x is not a whole number of rows.
  [[nodiscard]] std::vector<double> project(std::span<const double> x, std::size_t d) const;
};

// PCA of the rows of x (n x d): the eigenvectors of the covariance matrix (divided by n − 1) by cyclic Jacobi
// rotations; keeps min(k, n, d) components. Throws std::invalid_argument for k < 1, d < 1, x not a whole number of
// rows, fewer than two rows or a non-finite entry.
[[nodiscard]] Pca fit_pca(std::span<const double> x, std::size_t d, int k);

struct KMeansOptions {
  int k = 8;
  int restarts = 8;          // independent k-means++ starts; the lowest inertia is kept
  int max_iter = 100;        // Lloyd iterations per start (stops earlier when no assignment changes)
  std::uint64_t seed = 0;    // start r draws from a generator seeded with seed * 1000003 + r
};

struct KMeans {
  std::size_t dims = 0;           // d
  std::vector<double> centroids;  // k x d; cluster 0 is the largest on the fitting rows
  std::vector<int> labels;        // cluster of each fitting row
  double inertia = 0.0;           // sum of squared distances of the fitting rows to their centroids
  int iterations = 0;             // Lloyd iterations of the kept start
  [[nodiscard]] int clusters() const noexcept { return dims ? static_cast<int>(centroids.size() / dims) : 0; }
  // Nearest centroid (squared Euclidean, ties: the lower cluster id) of each row of x (n x d). Throws
  // std::invalid_argument for a different number of columns or a non-finite entry.
  [[nodiscard]] std::vector<int> assign(std::span<const double> x, std::size_t d) const;
};

// k-means of the rows of x (n x d): k-means++ initialisation (Arthur & Vassilvitskii 2007), Lloyd iterations (a row
// moves only to a strictly nearer centroid), an empty cluster takes the row farthest from its centroid, several starts
// keeping the lowest inertia. Clusters are renumbered by decreasing size (ties: the cluster of the earlier row first).
// Throws std::invalid_argument when k < 1, k > n, restarts < 1, max_iter < 1, x is not a whole number of rows or an
// entry of x is not finite.
[[nodiscard]] KMeans fit_kmeans(std::span<const double> x, std::size_t d, const KMeansOptions& options);

// Adjusted Rand index (Hubert & Arabie 1985) of two labellings of the same items: 1 for the same partition up to
// renaming, about 0 for independent ones (it can be negative). 1 when both labellings are trivial (one cluster or one
// item per cluster) and agree. Throws std::invalid_argument when the sizes differ.
[[nodiscard]] double adjusted_rand_index(std::span<const int> a, std::span<const int> b);

// Normalised mutual information I(a; b) / ((H(a) + H(b)) / 2) of two labellings, in [0, 1]; 1 when both have a single
// cluster. Unlike the ARI it does not penalise a different number of clusters, so it suits comparing K clusters with a
// fixed set of hand-made contexts. Throws std::invalid_argument when the sizes differ.
[[nodiscard]] double normalized_mutual_info(std::span<const int> a, std::span<const int> b);

}  // namespace nfx::dcm
