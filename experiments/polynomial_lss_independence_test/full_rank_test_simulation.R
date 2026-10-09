args <- commandArgs(trailingOnly = TRUE)
if (!(length(args) %in% c(8L, 9L, 10L))) {
  stop(
    paste(
      "usage: Rscript full_rank_test_simulation.R",
      "<n> <p> <replications> <distribution> <dependence>",
      "<master-seed> <checkpoint-every> <output.csv>",
      "[all|hoeffding_d|bkr_r|bdy_tau_star] [signal-scale]"
    )
  )
}

n <- as.integer(args[[1L]])
p <- as.integer(args[[2L]])
replications <- as.integer(args[[3L]])
distribution <- args[[4L]]
dependence <- args[[5L]]
master_seed <- as.integer(args[[6L]])
checkpoint_every <- as.integer(args[[7L]])
output_path <- args[[8L]]
selected_statistic <- if (length(args) >= 9L) args[[9L]] else "all"
signal_scale <- if (length(args) == 10L) as.numeric(args[[10L]]) else 1

if (n < 6L || p < 2L || replications < 1L || checkpoint_every < 1L) {
  stop("n>=6, p>=2, replications>=1, and checkpoint-every>=1 are required")
}
if (!distribution %in% c("I", "II", "III")) {
  stop("distribution must be I, II, or III")
}
if (!dependence %in% c("null", "toeplitz", "nonlinear")) {
  stop("dependence must be null, toeplitz, or nonlinear")
}
if (!selected_statistic %in% c(
  "all", "hoeffding_d", "bkr_r", "bdy_tau_star"
)) {
  stop("unsupported statistic")
}
if (!is.finite(signal_scale) || signal_scale <= 0) {
  stop("signal-scale must be positive")
}
if (dependence == "nonlinear" && distribution == "I") {
  stop("Li et al. (2021) specify the nonlinear alternative only for Models II and III")
}

if (!requireNamespace("independence", quietly = TRUE)) {
  stop("R package 'independence' is unavailable")
}
if (as.character(packageVersion("independence")) != "1.0.1") {
  stop("the validated backend requires independence version 1.0.1")
}
if (!requireNamespace("Rcpp", quietly = TRUE)) {
  stop("R package 'Rcpp' is unavailable")
}

script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- dirname(normalizePath(sub("^--file=", "", script_argument[[1L]])))
backend_dir <- normalizePath(file.path(script_dir, "..", "second_order_lsd"))
cpp_path <- file.path(backend_dir, "src", "fast_full_rank_matrix.cpp")
cache_dir <- file.path(backend_dir, ".rcpp_cache", R.version$arch)
dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)

macos_cpp_headers <- file.path(
  system("xcrun --show-sdk-path", intern = TRUE),
  "usr", "include", "c++", "v1"
)
if (dir.exists(macos_cpp_headers)) {
  Sys.setenv(
    PKG_CPPFLAGS = paste(
      Sys.getenv("PKG_CPPFLAGS"), "-isystem", shQuote(macos_cpp_headers)
    )
  )
}
architecture_flag <- if (R.version$arch == "aarch64") "-arch arm64" else "-arch x86_64"
Sys.setenv(
  PKG_CXXFLAGS = paste(Sys.getenv("PKG_CXXFLAGS"), architecture_flag, "-O3 -DNDEBUG")
)
independence_library <- system.file(
  "libs", paste0("independence", .Platform$dynlib.ext), package = "independence"
)
if (!nzchar(independence_library)) {
  stop("unable to locate the independence shared library")
}
Sys.setenv(
  PKG_LIBS = paste(
    architecture_flag, Sys.getenv("PKG_LIBS"), shQuote(independence_library)
  )
)
cache_lock <- paste0(cache_dir, ".sourceCpp-lock")
while (!dir.create(cache_lock, showWarnings = FALSE)) {
  Sys.sleep(0.1)
}
lock_held <- TRUE
on.exit(if (lock_held) unlink(cache_lock, recursive = TRUE), add = TRUE)
Rcpp::sourceCpp(cpp_path, cacheDir = cache_dir, rebuild = FALSE, showOutput = FALSE)
unlink(cache_lock, recursive = TRUE)
lock_held <- FALSE

generate_independent <- function(sample_size, dimension, model, reference_dimension) {
  if (model == "III") {
    return(matrix(rcauchy(sample_size * dimension), nrow = sample_size))
  }

  split <- floor(reference_dimension / 2L)
  first_dimension <- min(split, dimension)
  second_dimension <- dimension - first_dimension
  first <- if (model == "I") {
    matrix(rgamma(sample_size * first_dimension, shape = 4, scale = 0.5),
           nrow = sample_size)
  } else {
    matrix(rcauchy(sample_size * first_dimension), nrow = sample_size)
  }
  if (second_dimension == 0L) {
    return(first)
  }
  second <- matrix(rt(sample_size * second_dimension, df = 5), nrow = sample_size)
  cbind(first, second)
}

apply_toeplitz <- function(z, rho) {
  dimension <- ncol(z)
  bandwidth <- max(1L, floor(dimension / 100L))
  x <- z
  for (lag in seq_len(bandwidth)) {
    coefficient <- rho^lag
    left <- seq_len(dimension - lag)
    right <- left + lag
    x[, left] <- x[, left, drop = FALSE] + coefficient * z[, right, drop = FALSE]
    x[, right] <- x[, right, drop = FALSE] + coefficient * z[, left, drop = FALSE]
  }
  x
}

generate_data <- function(sample_size, dimension, model, structure) {
  if (structure == "null") {
    return(generate_independent(sample_size, dimension, model, dimension))
  }
  if (structure == "toeplitz") {
    rho <- signal_scale * c(I = 0.06, II = 0.03, III = 0.02)[[model]]
    if (rho >= 1) {
      stop("the scaled Toeplitz coefficient must be less than one")
    }
    z <- generate_independent(sample_size, dimension, model, dimension)
    return(apply_toeplitz(z, rho))
  }

  coefficients <- c(0.01, 0.02, 0.006, 0.5)
  coefficients[seq_len(3L)] <- signal_scale * coefficients[seq_len(3L)]
  z <- generate_independent(sample_size, dimension + 2L, model, dimension)
  noise <- matrix(rnorm(sample_size * dimension), nrow = sample_size)
  coefficients[[1L]] * z[, seq_len(dimension), drop = FALSE] +
    coefficients[[2L]] * z[, 1L + seq_len(dimension), drop = FALSE]^2 +
    coefficients[[3L]] * z[, 2L + seq_len(dimension), drop = FALSE]^2 +
    coefficients[[4L]] * noise
}

compute_rows <- function(replication, replication_seed) {
  set.seed(replication_seed)
  data <- generate_data(n, p, distribution, dependence)
  rank_matrix <- apply(data, 2L, rank, ties.method = "first")
  order_matrix <- apply(data, 2L, order)
  storage.mode(rank_matrix) <- "integer"
  storage.mode(order_matrix) <- "integer"

  rank_started <- proc.time()[["elapsed"]]
  invisible(capture.output({
    matrices <- if (selected_statistic == "all") {
      calculate_rank_matrices_parallel_cpp(rank_matrix, order_matrix)
    } else if (selected_statistic == "hoeffding_d") {
      list(hoeffding_d = calculate_hoeffding_matrix_cpp(rank_matrix, order_matrix))
    } else {
      matrix_selected <- calculate_selected_rank_matrix_cpp(
        rank_matrix, order_matrix, selected_statistic
      )
      setNames(list(matrix_selected), selected_statistic)
    }
  }))
  rank_seconds <- proc.time()[["elapsed"]] - rank_started

  statistics <- if (selected_statistic == "all") {
    c("hoeffding_d", "bkr_r", "bdy_tau_star")
  } else {
    selected_statistic
  }
  rows <- vector("list", length(statistics))
  for (index in seq_along(statistics)) {
    statistic <- statistics[[index]]
    matrix_s <- matrices[[statistic]]
    matrix_squared <- crossprod(matrix_s)
    rows[[index]] <- data.frame(
      replication = replication,
      replication_seed = replication_seed,
      distribution = distribution,
      dependence = dependence,
      statistic = statistic,
      q2 = sum(matrix_s * matrix_s),
      q4 = sum(matrix_squared * matrix_squared),
      rank_seconds = rank_seconds,
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, rows)
}

RNGkind("Mersenne-Twister", "Inversion", "Rejection")
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
if (file.exists(output_path)) {
  results <- read.csv(output_path, stringsAsFactors = FALSE)
  completed <- unique(results$replication)
} else {
  results <- NULL
  completed <- integer(0L)
}

started <- proc.time()[["elapsed"]]
pending <- setdiff(seq_len(replications), completed)
for (position in seq_along(pending)) {
  replication <- pending[[position]]
  new_rows <- compute_rows(replication, master_seed + replication - 1L)
  results <- rbind(results, new_rows)

  if (position %% checkpoint_every == 0L || position == length(pending)) {
    results <- results[order(results$replication, results$statistic), ]
    write.csv(results, output_path, row.names = FALSE)
    cat(
      sprintf(
        "%s/%s n=%d p=%d completed %d/%d in %.2f seconds\n",
        distribution, dependence, n, p, length(unique(results$replication)),
        replications, proc.time()[["elapsed"]] - started
      )
    )
    flush.console()
  }
}
