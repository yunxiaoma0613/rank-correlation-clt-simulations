// [[Rcpp::depends(independence)]]
// [[Rcpp::plugins(openmp)]]

#include <Rcpp.h>
#include <R_ext/BLAS.h>
#include <R_ext/RS.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

double calc_hoeffding(const std::vector<unsigned long>& permutation);
double calc_refined(const std::vector<unsigned long>& permutation);
double calc_taustar(const std::vector<unsigned long>& permutation);

extern "C" {
void F77_NAME(sgemm)(const char* transa, const char* transb, const int* m,
                     const int* n, const int* k, const float* alpha,
                     const float* a, const int* lda, const float* b,
                     const int* ldb, const float* beta, float* c,
                     const int* ldc FCLEN FCLEN);
}

namespace {

class PrefixSumTree {
 public:
  explicit PrefixSumTree(unsigned long length)
      : length_(length), tree_(length + 1, 0) {}

  void reset() {
    std::fill(tree_.begin(), tree_.end(), 0);
  }

  void add(unsigned long index) {
    ++tree_[index];
    while (index) {
      index -= index & -index;
      ++tree_[index];
    }
  }

  std::int64_t prefix_sum(unsigned long index) const {
    std::int64_t sum = tree_[0];
    do {
      sum -= tree_[index];
    } while (index && ((index += index & -index) < length_));
    return sum;
  }

 private:
  unsigned long length_;
  std::vector<std::int64_t> tree_;
};

// This is the O(n log n) count used by independence 1.0.1, with the
// worker-thread-unsafe R interrupt check removed.
double calc_hoeffding_parallel_safe(
    const std::vector<unsigned long>& permutation,
    PrefixSumTree& tree) {
  const unsigned long n = permutation.size();
  tree.reset();
  std::int64_t sum = 0;
  for (unsigned long i = 0; i < n; ++i) {
    tree.add(permutation[i]);
    const std::int64_t a = i;
    const std::int64_t b = permutation[i];
    const std::int64_t c = tree.prefix_sum(permutation[i]);
    sum += a * (a - 1) / 2 * b * (b - 1)
           - (a - 1) * (b - 1) * c * (n - 2)
           + c * (c - 1) / 2 * (n - 2) * (n - 3);
  }
  return static_cast<double>(sum) * 2.0 / n / (n - 1) / (n - 2)
         / (n - 3) / (n - 4);
}

template <typename Element>
class WeightedSumTree {
 public:
  explicit WeightedSumTree(unsigned long length)
      : length_(length), tree_(length + 1, 0) {}

  void reset() {
    std::fill(tree_.begin(), tree_.end(), 0);
  }

  void add(unsigned long index, Element value) {
    tree_[index] += value;
    while (index) {
      index -= index & -index;
      tree_[index] += value;
    }
  }

  Element suffix_sum(unsigned long index) const {
    Element sum = 0;
    do {
      sum += tree_[index];
    } while (index && ((index += index & -index) < length_));
    return sum;
  }

  Element prefix_sum(unsigned long index) const {
    Element sum = tree_[0];
    do {
      sum -= tree_[index];
    } while (index && ((index += index & -index) < length_));
    return sum;
  }

 private:
  unsigned long length_;
  std::vector<Element> tree_;
};

struct TauStarWorkspace {
  explicit TauStarWorkspace(unsigned long length)
      : mirror(length),
        counts(length),
        up_counts(length),
        down_counts(length),
        up_down_counts(length) {}

  std::vector<unsigned long> mirror;
  WeightedSumTree<std::uint64_t> counts;
  WeightedSumTree<std::uint64_t> up_counts;
  WeightedSumTree<std::uint64_t> down_counts;
  WeightedSumTree<std::uint64_t> up_down_counts;
};

// This is the O(n log n) Tau* count from independence 1.0.1, with the
// worker-thread-unsafe interrupt checks removed.
std::uint64_t taustar_subroutine_parallel_safe(
    const std::vector<unsigned long>& permutation,
    TauStarWorkspace& workspace) {
  const unsigned long n = permutation.size();
  workspace.counts.reset();
  workspace.up_counts.reset();
  workspace.down_counts.reset();
  workspace.up_down_counts.reset();

  std::uint64_t sum = 0;
  for (unsigned long i = 0; i < n; ++i) {
    const std::uint64_t up =
        workspace.counts.prefix_sum(permutation[i]);
    const std::uint64_t down =
        workspace.counts.suffix_sum(permutation[i] + 1);
    const std::uint64_t down_up =
        workspace.down_counts.prefix_sum(permutation[i]);
    const std::uint64_t up_down =
        workspace.up_counts.suffix_sum(permutation[i] + 1);
    const std::uint64_t up_down_up =
        workspace.up_down_counts.prefix_sum(permutation[i]);

    workspace.counts.add(permutation[i], 1);
    workspace.up_counts.add(permutation[i], up);
    workspace.down_counts.add(permutation[i], down);
    workspace.up_down_counts.add(permutation[i], up_down);
    sum += 2 * up_down_up - down * down_up - up * up_down
           + up * down * (i - 1);
  }
  return sum;
}

std::uint64_t taustar_count_parallel_safe(
    const std::vector<unsigned long>& permutation,
    TauStarWorkspace& workspace) {
  const unsigned long n = permutation.size();
  std::uint64_t sum =
      taustar_subroutine_parallel_safe(permutation, workspace);

  for (unsigned long i = 0; i < n; ++i) {
    workspace.mirror[i] = permutation[n - 1 - i];
  }
  sum += taustar_subroutine_parallel_safe(workspace.mirror, workspace);

  for (unsigned long i = 0; i < n; ++i) {
    workspace.mirror[permutation[i]] = i;
  }
  sum += taustar_subroutine_parallel_safe(workspace.mirror, workspace);

  for (unsigned long i = 0; i < n; ++i) {
    workspace.mirror[n - 1 - permutation[i]] = i;
  }
  sum += taustar_subroutine_parallel_safe(workspace.mirror, workspace);

  const std::uint64_t total =
      static_cast<std::uint64_t>(n) * (n - 1) / 2 * (n - 2) / 3
      * (n - 3) / 4;
  return total * 2 - (sum / 4) * 3;
}

double calc_taustar_parallel_safe(
    const std::vector<unsigned long>& permutation,
    TauStarWorkspace& workspace) {
  const unsigned long n = permutation.size();
  const std::int64_t count = static_cast<std::int64_t>(
      taustar_count_parallel_safe(permutation, workspace));
  return static_cast<double>(count) * 8.0 / n / (n - 1) / (n - 2)
         / (n - 3);
}

std::vector<double> kpm_moments_float(
    const float* matrix,
    int p,
    int probes,
    int degree,
    double spectral_bound,
    int seed) {
  const size_t block_size =
      static_cast<size_t>(p) * static_cast<size_t>(probes);
  std::vector<float> reference(block_size);
  std::vector<float> previous(block_size);
  std::vector<float> current(block_size);
  std::vector<float> next(block_size);
  std::mt19937_64 generator(static_cast<std::uint64_t>(seed));
  for (size_t index = 0; index < block_size; ++index) {
    reference[index] = (generator() & 1ULL) ? 1.0F : -1.0F;
  }
  previous = reference;

  const char no_transpose = 'N';
  const float coefficient =
      static_cast<float>(1.0 / spectral_bound);
  const float zero = 0.0F;
  auto apply_scaled_matrix = [&](const std::vector<float>& input,
                                 std::vector<float>& output) {
    F77_CALL(sgemm)(
        &no_transpose,
        &no_transpose,
        &p,
        &probes,
        &p,
        &coefficient,
        matrix,
        &p,
        input.data(),
        &p,
        &zero,
        output.data(),
        &p
        FCONE
        FCONE);
  };
  auto trace_moment = [&](const std::vector<float>& values) {
    long double sum = 0.0L;
    for (size_t index = 0; index < block_size; ++index) {
      sum += static_cast<long double>(reference[index])
             * static_cast<long double>(values[index]);
    }
    return static_cast<double>(
        sum / static_cast<long double>(block_size));
  };

  std::vector<double> moments(static_cast<size_t>(degree) + 1);
  moments[0] = 1.0;
  apply_scaled_matrix(previous, current);
  moments[1] = trace_moment(current);
  for (int order = 2; order <= degree; ++order) {
    apply_scaled_matrix(current, next);
    for (size_t index = 0; index < block_size; ++index) {
      next[index] = 2.0F * next[index] - previous[index];
    }
    moments[static_cast<size_t>(order)] = trace_moment(next);
    previous.swap(current);
    current.swap(next);
  }
  return moments;
}

}  // namespace

// [[Rcpp::export]]
Rcpp::NumericMatrix calculate_hoeffding_matrix_cpp(
    const Rcpp::IntegerMatrix& rank_matrix,
    const Rcpp::IntegerMatrix& order_matrix) {
  const int n = rank_matrix.nrow();
  const int p = rank_matrix.ncol();
  if (order_matrix.nrow() != n || order_matrix.ncol() != p) {
    Rcpp::stop("rank and order matrices must have the same dimensions");
  }

  Rcpp::NumericMatrix hoeffding_d(p, p);
  for (int column = 0; column < p; ++column) {
    hoeffding_d(column, column) = 1.0;
  }

  const int* ranks = rank_matrix.begin();
  const int* orders = order_matrix.begin();
  double* output = hoeffding_d.begin();
#ifdef _OPENMP
  const int thread_count = omp_get_max_threads();
#else
  const int thread_count = 1;
#endif
  Rcpp::Rcout << "parallel Hoeffding D threads=" << thread_count << "\n";

#pragma omp parallel
  {
    std::vector<unsigned long> relative_permutation(
        static_cast<size_t>(n));
    PrefixSumTree tree(static_cast<unsigned long>(n));
#pragma omp for schedule(dynamic, 4)
    for (int left = 0; left < p - 1; ++left) {
      for (int right = left + 1; right < p; ++right) {
        for (int row = 0; row < n; ++row) {
          const int ordered_row = orders[row + n * left] - 1;
          relative_permutation[static_cast<size_t>(row)] =
              static_cast<unsigned long>(
                  ranks[ordered_row + n * right] - 1);
        }

        const double value_d =
            30.0 * calc_hoeffding_parallel_safe(
                relative_permutation, tree);
        output[left + p * right] = value_d;
        output[right + p * left] = value_d;
      }
    }
  }
  Rcpp::checkUserInterrupt();

  return hoeffding_d;
}

// [[Rcpp::export]]
Rcpp::NumericMatrix calculate_selected_rank_matrix_cpp(
    const Rcpp::IntegerMatrix& rank_matrix,
    const Rcpp::IntegerMatrix& order_matrix,
    std::string statistic) {
  const int n = rank_matrix.nrow();
  const int p = rank_matrix.ncol();
  if (order_matrix.nrow() != n || order_matrix.ncol() != p) {
    Rcpp::stop("rank and order matrices must have the same dimensions");
  }
  if (statistic != "bkr_r" && statistic != "bdy_tau_star") {
    Rcpp::stop("the selected statistic must be bkr_r or bdy_tau_star");
  }
  if (n < 5 || n > 14081) {
    Rcpp::stop("the selected rank-statistic backend requires 5 <= n <= 14081");
  }

  Rcpp::NumericMatrix result(p, p);
  for (int column = 0; column < p; ++column) {
    result(column, column) = 1.0;
  }

  const int* ranks = rank_matrix.begin();
  const int* orders = order_matrix.begin();
  double* output = result.begin();
  const bool compute_bkr = statistic == "bkr_r";
#ifdef _OPENMP
  const int thread_count = omp_get_max_threads();
#else
  const int thread_count = 1;
#endif
  Rcpp::Rcout << "parallel " << statistic
              << " threads=" << thread_count << "\n";

#pragma omp parallel
  {
    std::vector<unsigned long> relative_permutation(
        static_cast<size_t>(n));
    PrefixSumTree hoeffding_tree(static_cast<unsigned long>(n));
    TauStarWorkspace taustar_workspace(static_cast<unsigned long>(n));
#pragma omp for schedule(dynamic, 2)
    for (int left = 0; left < p - 1; ++left) {
      for (int right = left + 1; right < p; ++right) {
        for (int row = 0; row < n; ++row) {
          const int ordered_row = orders[row + n * left] - 1;
          relative_permutation[static_cast<size_t>(row)] =
              static_cast<unsigned long>(
                  ranks[ordered_row + n * right] - 1);
        }

        const double tau_star = calc_taustar_parallel_safe(
            relative_permutation, taustar_workspace);
        double value = 1.5 * tau_star;
        if (compute_bkr) {
          const double hoeffding = calc_hoeffding_parallel_safe(
              relative_permutation, hoeffding_tree);
          value = 90.0 * (-0.5 * hoeffding + tau_star / 24.0);
        }
        output[left + p * right] = value;
        output[right + p * left] = value;
      }
    }
  }
  Rcpp::checkUserInterrupt();
  return result;
}

// [[Rcpp::export]]
Rcpp::List calculate_bkr_tau_matrices_cpp(
    const Rcpp::IntegerMatrix& rank_matrix,
    const Rcpp::IntegerMatrix& order_matrix) {
  const int n = rank_matrix.nrow();
  const int p = rank_matrix.ncol();
  if (order_matrix.nrow() != n || order_matrix.ncol() != p) {
    Rcpp::stop("rank and order matrices must have the same dimensions");
  }
  if (n < 5 || n > 14081) {
    Rcpp::stop("the paired rank-statistic backend requires 5 <= n <= 14081");
  }

  Rcpp::NumericMatrix bkr_r(p, p);
  Rcpp::NumericMatrix bdy_tau_star(p, p);
  for (int column = 0; column < p; ++column) {
    bkr_r(column, column) = 1.0;
    bdy_tau_star(column, column) = 1.0;
  }

  const int* ranks = rank_matrix.begin();
  const int* orders = order_matrix.begin();
  double* bkr_output = bkr_r.begin();
  double* tau_output = bdy_tau_star.begin();
#ifdef _OPENMP
  const int thread_count = omp_get_max_threads();
#else
  const int thread_count = 1;
#endif
  Rcpp::Rcout << "parallel BKR R and BDY tau* threads="
              << thread_count << "\n";

#pragma omp parallel
  {
    std::vector<unsigned long> relative_permutation(
        static_cast<size_t>(n));
    PrefixSumTree hoeffding_tree(static_cast<unsigned long>(n));
    TauStarWorkspace taustar_workspace(static_cast<unsigned long>(n));
#pragma omp for schedule(dynamic, 2)
    for (int left = 0; left < p - 1; ++left) {
      for (int right = left + 1; right < p; ++right) {
        for (int row = 0; row < n; ++row) {
          const int ordered_row = orders[row + n * left] - 1;
          relative_permutation[static_cast<size_t>(row)] =
              static_cast<unsigned long>(
                  ranks[ordered_row + n * right] - 1);
        }

        const double hoeffding = calc_hoeffding_parallel_safe(
            relative_permutation, hoeffding_tree);
        const double tau_star = calc_taustar_parallel_safe(
            relative_permutation, taustar_workspace);
        const double value_r =
            90.0 * (-0.5 * hoeffding + tau_star / 24.0);
        const double value_tau = 1.5 * tau_star;
        bkr_output[left + p * right] = value_r;
        bkr_output[right + p * left] = value_r;
        tau_output[left + p * right] = value_tau;
        tau_output[right + p * left] = value_tau;
      }
    }
  }
  Rcpp::checkUserInterrupt();

  return Rcpp::List::create(
      Rcpp::Named("bkr_r") = bkr_r,
      Rcpp::Named("bdy_tau_star") = bdy_tau_star);
}

// [[Rcpp::export]]
Rcpp::List calculate_rank_matrices_parallel_cpp(
    const Rcpp::IntegerMatrix& rank_matrix,
    const Rcpp::IntegerMatrix& order_matrix) {
  const int n = rank_matrix.nrow();
  const int p = rank_matrix.ncol();
  if (order_matrix.nrow() != n || order_matrix.ncol() != p) {
    Rcpp::stop("rank and order matrices must have the same dimensions");
  }
  if (n < 5 || n > 14081) {
    Rcpp::stop("the joint rank-statistic backend requires 5 <= n <= 14081");
  }

  Rcpp::NumericMatrix hoeffding_d(p, p);
  Rcpp::NumericMatrix bkr_r(p, p);
  Rcpp::NumericMatrix bdy_tau_star(p, p);
  for (int column = 0; column < p; ++column) {
    hoeffding_d(column, column) = 1.0;
    bkr_r(column, column) = 1.0;
    bdy_tau_star(column, column) = 1.0;
  }

  const int* ranks = rank_matrix.begin();
  const int* orders = order_matrix.begin();
  double* d_output = hoeffding_d.begin();
  double* r_output = bkr_r.begin();
  double* tau_output = bdy_tau_star.begin();
#ifdef _OPENMP
  const int thread_count = omp_get_max_threads();
#else
  const int thread_count = 1;
#endif
  Rcpp::Rcout << "parallel D, BKR R, and BDY tau* threads="
              << thread_count << "\n";

#pragma omp parallel
  {
    std::vector<unsigned long> relative_permutation(
        static_cast<size_t>(n));
    PrefixSumTree hoeffding_tree(static_cast<unsigned long>(n));
    TauStarWorkspace taustar_workspace(static_cast<unsigned long>(n));
#pragma omp for schedule(dynamic, 2)
    for (int left = 0; left < p - 1; ++left) {
      for (int right = left + 1; right < p; ++right) {
        for (int row = 0; row < n; ++row) {
          const int ordered_row = orders[row + n * left] - 1;
          relative_permutation[static_cast<size_t>(row)] =
              static_cast<unsigned long>(
                  ranks[ordered_row + n * right] - 1);
        }

        const double hoeffding = calc_hoeffding_parallel_safe(
            relative_permutation, hoeffding_tree);
        const double tau_star = calc_taustar_parallel_safe(
            relative_permutation, taustar_workspace);
        const double value_d = 30.0 * hoeffding;
        const double value_r =
            90.0 * (-0.5 * hoeffding + tau_star / 24.0);
        const double value_tau = 1.5 * tau_star;
        d_output[left + p * right] = value_d;
        d_output[right + p * left] = value_d;
        r_output[left + p * right] = value_r;
        r_output[right + p * left] = value_r;
        tau_output[left + p * right] = value_tau;
        tau_output[right + p * left] = value_tau;
      }
    }
  }
  Rcpp::checkUserInterrupt();

  return Rcpp::List::create(
      Rcpp::Named("hoeffding_d") = hoeffding_d,
      Rcpp::Named("bkr_r") = bkr_r,
      Rcpp::Named("bdy_tau_star") = bdy_tau_star);
}

// [[Rcpp::export]]
Rcpp::List calculate_bkr_tau_kpm_moments_cpp(
    const Rcpp::IntegerMatrix& rank_matrix,
    const Rcpp::IntegerMatrix& order_matrix,
    int probes,
    int degree,
    double spectral_bound,
    double bkr_scale,
    double tau_scale,
    int seed) {
  const int n = rank_matrix.nrow();
  const int p = rank_matrix.ncol();
  if (order_matrix.nrow() != n || order_matrix.ncol() != p) {
    Rcpp::stop("rank and order matrices must have the same dimensions");
  }
  if (n < 5 || n > 14081) {
    Rcpp::stop("the paired rank-statistic backend requires 5 <= n <= 14081");
  }
  if (probes < 1 || degree < 1 || spectral_bound <= 0.0 ||
      bkr_scale <= 0.0 || tau_scale <= 0.0) {
    Rcpp::stop("invalid compact KPM parameters");
  }

  const size_t matrix_size =
      static_cast<size_t>(p) * static_cast<size_t>(p);
  const size_t packed_size =
      static_cast<size_t>(p) * static_cast<size_t>(p + 1) / 2;
  std::unique_ptr<float[]> bkr_matrix(new float[matrix_size]);
  std::unique_ptr<float[]> tau_packed(new float[packed_size]);
  for (int column = 0; column < p; ++column) {
    bkr_matrix[static_cast<size_t>(column) * p + column] = 0.0F;
    tau_packed[
        static_cast<size_t>(column) * (column + 1) / 2 + column] = 0.0F;
  }

  const int* ranks = rank_matrix.begin();
  const int* orders = order_matrix.begin();
  const auto build_started = std::chrono::steady_clock::now();
#ifdef _OPENMP
  const int thread_count = omp_get_max_threads();
#else
  const int thread_count = 1;
#endif
  Rcpp::Rcout << "parallel compact BKR R and BDY tau* threads="
              << thread_count << "\n";

#pragma omp parallel
  {
    std::vector<unsigned long> relative_permutation(
        static_cast<size_t>(n));
    PrefixSumTree hoeffding_tree(static_cast<unsigned long>(n));
    TauStarWorkspace taustar_workspace(static_cast<unsigned long>(n));
#pragma omp for schedule(dynamic, 2)
    for (int left = 0; left < p - 1; ++left) {
      for (int right = left + 1; right < p; ++right) {
        for (int row = 0; row < n; ++row) {
          const int ordered_row = orders[row + n * left] - 1;
          relative_permutation[static_cast<size_t>(row)] =
              static_cast<unsigned long>(
                  ranks[ordered_row + n * right] - 1);
        }

        const double hoeffding = calc_hoeffding_parallel_safe(
            relative_permutation, hoeffding_tree);
        const double tau_star = calc_taustar_parallel_safe(
            relative_permutation, taustar_workspace);
        const double value_r =
            90.0 * (-0.5 * hoeffding + tau_star / 24.0);
        const double value_tau = 1.5 * tau_star;
        bkr_matrix[static_cast<size_t>(left) +
                   static_cast<size_t>(p) * right] =
            static_cast<float>(bkr_scale * value_r);
        bkr_matrix[static_cast<size_t>(right) +
                   static_cast<size_t>(p) * left] =
            static_cast<float>(bkr_scale * value_r);
        tau_packed[
            static_cast<size_t>(right) * (right + 1) / 2 + left] =
            static_cast<float>(tau_scale * value_tau);
      }
    }
  }
  Rcpp::checkUserInterrupt();
  const auto build_finished = std::chrono::steady_clock::now();

  const auto bkr_kpm_started = std::chrono::steady_clock::now();
  const std::vector<double> bkr_moments = kpm_moments_float(
      bkr_matrix.get(),
      p,
      probes,
      degree,
      spectral_bound,
      seed);
  const auto bkr_kpm_finished = std::chrono::steady_clock::now();
  bkr_matrix.reset();

  std::unique_ptr<float[]> tau_matrix(new float[matrix_size]);
#pragma omp parallel for schedule(static)
  for (int right = 0; right < p; ++right) {
    for (int left = 0; left <= right; ++left) {
      const float value =
          tau_packed[
              static_cast<size_t>(right) * (right + 1) / 2 + left];
      tau_matrix[static_cast<size_t>(left) +
                 static_cast<size_t>(p) * right] = value;
      tau_matrix[static_cast<size_t>(right) +
                 static_cast<size_t>(p) * left] = value;
    }
  }
  tau_packed.reset();

  const auto tau_kpm_started = std::chrono::steady_clock::now();
  const std::vector<double> tau_moments = kpm_moments_float(
      tau_matrix.get(),
      p,
      probes,
      degree,
      spectral_bound,
      seed + 1);
  const auto tau_kpm_finished = std::chrono::steady_clock::now();
  tau_matrix.reset();

  const auto seconds_between = [](
      const std::chrono::steady_clock::time_point& first,
      const std::chrono::steady_clock::time_point& second) {
    return std::chrono::duration<double>(second - first).count();
  };
  return Rcpp::List::create(
      Rcpp::Named("bkr_r") = Rcpp::wrap(bkr_moments),
      Rcpp::Named("bdy_tau_star") = Rcpp::wrap(tau_moments),
      Rcpp::Named("build_seconds") =
          seconds_between(build_started, build_finished),
      Rcpp::Named("bkr_kpm_seconds") =
          seconds_between(bkr_kpm_started, bkr_kpm_finished),
      Rcpp::Named("tau_kpm_seconds") =
          seconds_between(tau_kpm_started, tau_kpm_finished));
}

// [[Rcpp::export]]
Rcpp::NumericVector hoeffding_kpm_moments_cpp(
    const Rcpp::NumericMatrix& hoeffding_d,
    int probes,
    int degree,
    double spectral_bound,
    double standardization_scale,
    int seed) {
  const int p = hoeffding_d.nrow();
  if (hoeffding_d.ncol() != p) {
    Rcpp::stop("the Hoeffding matrix must be square");
  }
  if (probes < 1 || degree < 1) {
    Rcpp::stop("probes and degree must be positive");
  }
  if (!(spectral_bound > 0.0) || !(standardization_scale > 0.0)) {
    Rcpp::stop("spectral bound and standardization scale must be positive");
  }

  const size_t block_size =
      static_cast<size_t>(p) * static_cast<size_t>(probes);
  std::vector<double> reference(block_size);
  std::vector<double> previous(block_size);
  std::vector<double> current(block_size);
  std::vector<double> next(block_size);
  std::mt19937_64 generator(static_cast<std::uint64_t>(seed));
  for (size_t index = 0; index < block_size; ++index) {
    reference[index] = (generator() & 1ULL) ? 1.0 : -1.0;
  }
  previous = reference;

  const double scaled_coefficient =
      standardization_scale / spectral_bound;
  const char side = 'L';
  const char triangle = 'U';
  const double zero = 0.0;
  auto apply_scaled_matrix = [&](const std::vector<double>& input,
                                 std::vector<double>& output) {
    F77_CALL(dsymm)(
        &side,
        &triangle,
        &p,
        &probes,
        &scaled_coefficient,
        hoeffding_d.begin(),
        &p,
        input.data(),
        &p,
        &zero,
        output.data(),
        &p
        FCONE
        FCONE);
    for (size_t index = 0; index < block_size; ++index) {
      output[index] -= scaled_coefficient * input[index];
    }
  };
  auto trace_moment = [&](const std::vector<double>& values) {
    long double sum = 0.0L;
    for (size_t index = 0; index < block_size; ++index) {
      sum += static_cast<long double>(reference[index])
             * static_cast<long double>(values[index]);
    }
    return static_cast<double>(
        sum / static_cast<long double>(block_size));
  };

  Rcpp::NumericVector moments(degree + 1);
  moments[0] = 1.0;
  apply_scaled_matrix(previous, current);
  moments[1] = trace_moment(current);
  for (int order = 2; order <= degree; ++order) {
    apply_scaled_matrix(current, next);
    for (size_t index = 0; index < block_size; ++index) {
      next[index] = 2.0 * next[index] - previous[index];
    }
    moments[order] = trace_moment(next);
    previous.swap(current);
    current.swap(next);
  }
  return moments;
}

// [[Rcpp::export]]
Rcpp::List calculate_rank_matrices_cpp(
    const Rcpp::IntegerMatrix& rank_matrix,
    const Rcpp::IntegerMatrix& order_matrix) {
  const int n = rank_matrix.nrow();
  const int p = rank_matrix.ncol();
  if (order_matrix.nrow() != n || order_matrix.ncol() != p) {
    Rcpp::stop("rank and order matrices must have the same dimensions");
  }

  Rcpp::NumericMatrix hoeffding_d(p, p);
  Rcpp::NumericMatrix bkr_r(p, p);
  Rcpp::NumericMatrix bdy_tau_star(p, p);
  for (int column = 0; column < p; ++column) {
    hoeffding_d(column, column) = 1.0;
    bkr_r(column, column) = 1.0;
    bdy_tau_star(column, column) = 1.0;
  }

  std::vector<unsigned long> relative_permutation(static_cast<size_t>(n));
  for (int left = 0; left < p - 1; ++left) {
    for (int right = left + 1; right < p; ++right) {
      for (int row = 0; row < n; ++row) {
        const int ordered_row = order_matrix(row, left) - 1;
        relative_permutation[static_cast<size_t>(row)] =
            static_cast<unsigned long>(
                rank_matrix(ordered_row, right) - 1);
      }

      const double value_d = 30.0 * calc_hoeffding(relative_permutation);
      const double value_r = 90.0 * calc_refined(relative_permutation);
      const double value_tau = 1.5 * calc_taustar(relative_permutation);

      hoeffding_d(left, right) = hoeffding_d(right, left) = value_d;
      bkr_r(left, right) = bkr_r(right, left) = value_r;
      bdy_tau_star(left, right) = bdy_tau_star(right, left) = value_tau;
    }
    if ((left + 1) % 100 == 0) {
      Rcpp::Rcout << "completed " << left + 1 << " of " << p - 1
                  << " matrix columns\n";
    }
    if ((left + 1) % 16 == 0) {
      Rcpp::checkUserInterrupt();
    }
  }

  return Rcpp::List::create(
      Rcpp::Named("hoeffding_d") = hoeffding_d,
      Rcpp::Named("bkr_r") = bkr_r,
      Rcpp::Named("bdy_tau_star") = bdy_tau_star);
}
