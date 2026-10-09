args <- commandArgs(trailingOnly = TRUE)
if (!(length(args) %in% c(2L, 3L, 4L))) {
  stop(
    paste(
      "usage: Rscript fast_full_rank_matrix.R",
      "<uniforms.csv> <output-directory> [matrices|spectra|kpm]",
      "[all|hoeffding_d]"
    )
  )
}

if (!requireNamespace("independence", quietly = TRUE)) {
  stop("R package 'independence' is unavailable; run setup_fast_rank_backend.R")
}
if (as.character(packageVersion("independence")) != "1.0.1") {
  stop("the validated backend requires independence version 1.0.1")
}

input_path <- args[[1L]]
output_dir <- args[[2L]]
output_mode <- if (length(args) >= 3L) args[[3L]] else "matrices"
if (!(output_mode %in% c("matrices", "spectra", "kpm"))) {
  stop("output mode must be 'matrices', 'spectra', or 'kpm'")
}
selected_statistic <- if (length(args) == 4L) args[[4L]] else "all"
valid_statistics <- c("hoeffding_d", "bkr_r", "bdy_tau_star")
if (!(selected_statistic %in% c("all", "bkr_tau_pair", valid_statistics))) {
  stop("unknown selected statistic")
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

if (input_path == "__generate__") {
  n_generated <- as.integer(Sys.getenv("KENDALL_GENERATE_N"))
  p_generated <- as.integer(Sys.getenv("KENDALL_GENERATE_P"))
  seed_generated <- as.integer(Sys.getenv("KENDALL_GENERATE_SEED"))
  if (is.na(n_generated) || is.na(p_generated) || is.na(seed_generated)) {
    stop("generated input requires n, p, and seed environment variables")
  }
  set.seed(seed_generated)
  generated_size <- n_generated * p_generated
  uniforms <- matrix(
    runif(generated_size) + 2^(-32) * runif(generated_size),
    nrow = n_generated
  )
} else {
  uniforms <- as.matrix(
    read.csv(input_path, header = FALSE, check.names = FALSE)
  )
}
storage.mode(uniforms) <- "double"
n <- nrow(uniforms)
p <- ncol(uniforms)
if (n < 6L || p < 2L) {
  stop("the joint D/R/tau* backend requires n>=6 and p>=2")
}

has_ties <- any(vapply(
  seq_len(p),
  function(column) anyDuplicated(uniforms[, column]) > 0L,
  logical(1L)
))
if (has_ties) {
  stop("the fast backend is validated only for samples without ties")
}

rank_matrix <- apply(uniforms, 2L, rank, ties.method = "first")
order_matrix <- apply(uniforms, 2L, order)
storage.mode(rank_matrix) <- "integer"
storage.mode(order_matrix) <- "integer"

start_time <- proc.time()[["elapsed"]]
engine <- Sys.getenv("KENDALL_FAST_RANK_ENGINE", "cpp")
compact_pair_kpm <- NULL
use_compact_pair_kpm <- Sys.getenv("KENDALL_COMPACT_PAIR_KPM", "1") != "0"
if (engine == "cpp") {
  script_argument <- grep(
    "^--file=", commandArgs(trailingOnly = FALSE), value = TRUE
  )
  if (length(script_argument) != 1L) {
    stop("unable to locate the fast backend source directory")
  }
  script_dir <- dirname(
    normalizePath(sub("^--file=", "", script_argument[[1L]]))
  )
  cpp_path <- file.path(script_dir, "fast_full_rank_matrix.cpp")
  cache_dir <- file.path(script_dir, "..", ".rcpp_cache", R.version$arch)
  dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)

  macos_cpp_headers <- file.path(
    system("xcrun --show-sdk-path", intern = TRUE),
    "usr", "include", "c++", "v1"
  )
  if (dir.exists(macos_cpp_headers)) {
    Sys.setenv(
      PKG_CPPFLAGS = paste(
        Sys.getenv("PKG_CPPFLAGS"),
        "-isystem",
        shQuote(macos_cpp_headers)
      )
    )
  }
  architecture_flag <- if (R.version$arch == "aarch64") {
    "-arch arm64"
  } else {
    "-arch x86_64"
  }
  Sys.setenv(
    PKG_CXXFLAGS = paste(
      Sys.getenv("PKG_CXXFLAGS"),
      architecture_flag,
      "-O3 -DNDEBUG"
    )
  )
  independence_library <- system.file(
    "libs",
    paste0("independence", .Platform$dynlib.ext),
    package = "independence"
  )
  if (!nzchar(independence_library)) {
    stop("unable to locate the independence shared library")
  }
  Sys.setenv(
    PKG_LIBS = paste(
      architecture_flag,
      Sys.getenv("PKG_LIBS"),
      shQuote(independence_library)
    )
  )
  Rcpp::sourceCpp(
    cpp_path,
    cacheDir = cache_dir,
    rebuild = identical(Sys.getenv("KENDALL_FAST_RANK_REBUILD"), "1"),
    showOutput = FALSE,
    verbose = FALSE
  )
  if (selected_statistic == "bkr_tau_pair" &&
      output_mode == "kpm" &&
      use_compact_pair_kpm) {
    probes <- as.integer(Sys.getenv("KENDALL_KPM_PROBES", "16"))
    degree <- as.integer(Sys.getenv("KENDALL_KPM_DEGREE", "160"))
    spectral_bound <- as.numeric(
      Sys.getenv("KENDALL_KPM_SPECTRAL_BOUND", "2.6")
    )
    kpm_seed <- as.integer(Sys.getenv("KENDALL_KPM_SEED", "20260730"))
    bkr_scale <- sqrt((n * (n - 1) / 2) / p)
    tau_scale <- bkr_scale / (3 / 5)
    compact_pair_kpm <- calculate_bkr_tau_kpm_moments_cpp(
      rank_matrix,
      order_matrix,
      probes,
      degree,
      spectral_bound,
      bkr_scale,
      tau_scale,
      kpm_seed
    )
    rank_matrices <- list()
  } else if (selected_statistic == "all") {
    rank_matrices <- calculate_rank_matrices_parallel_cpp(
      rank_matrix,
      order_matrix
    )
  } else if (selected_statistic == "bkr_tau_pair") {
    rank_matrices <- calculate_bkr_tau_matrices_cpp(rank_matrix, order_matrix)
  } else if (selected_statistic == "hoeffding_d") {
    rank_matrices <- list(
      hoeffding_d = calculate_hoeffding_matrix_cpp(
        rank_matrix,
        order_matrix
      )
    )
  } else {
    selected_matrix <- calculate_selected_rank_matrix_cpp(
      rank_matrix,
      order_matrix,
      selected_statistic
    )
    rank_matrices <- setNames(list(selected_matrix), selected_statistic)
  }
} else if (engine == "r") {
  if (selected_statistic == "bkr_tau_pair") {
    stop("the paired selection requires the C++ engine")
  }
  statistics_to_compute <- if (selected_statistic == "all") {
    valid_statistics
  } else {
    selected_statistic
  }
  rank_matrices <- setNames(
    lapply(statistics_to_compute, function(statistic) diag(1.0, p)),
    statistics_to_compute
  )
  calc_d <- getFromNamespace(".calc.hoeffding", "independence")
  calc_r <- getFromNamespace(".calc.refined", "independence")
  calc_tau <- getFromNamespace(".calc.taustar", "independence")
  for (left in seq_len(p - 1L)) {
    left_order <- order_matrix[, left]
    for (right in seq.int(left + 1L, p)) {
      relative_permutation <- as.integer(rank_matrix[left_order, right] - 1L)

      # These constants map independence 1.0.1 to the normalized symmetric
      # U-statistic kernels audited in full_rank_statistics.py.
      if ("hoeffding_d" %in% statistics_to_compute) {
        value_d <- 30.0 * calc_d(relative_permutation)
        rank_matrices$hoeffding_d[left, right] <-
          rank_matrices$hoeffding_d[right, left] <- value_d
      }
      if ("bkr_r" %in% statistics_to_compute) {
        value_r <- 90.0 * calc_r(relative_permutation)
        rank_matrices$bkr_r[left, right] <-
          rank_matrices$bkr_r[right, left] <- value_r
      }
      if ("bdy_tau_star" %in% statistics_to_compute) {
        value_tau <- 1.5 * calc_tau(relative_permutation)
        rank_matrices$bdy_tau_star[left, right] <-
          rank_matrices$bdy_tau_star[right, left] <- value_tau
      }
    }
    if (left %% 100L == 0L) {
      cat("completed", left, "of", p - 1L, "matrix columns\n")
      flush.console()
    }
  }
} else {
  stop("KENDALL_FAST_RANK_ENGINE must be 'cpp' or 'r'")
}
elapsed <- proc.time()[["elapsed"]] - start_time

if (output_mode == "matrices") {
  for (statistic in names(rank_matrices)) {
    write.table(
      rank_matrices[[statistic]],
      file.path(output_dir, paste0("full_", statistic, ".csv")),
      row.names = FALSE,
      col.names = FALSE,
      sep = ","
    )
  }
} else if (output_mode == "spectra") {
  spectra <- lapply(
    rank_matrices,
    function(matrix) eigen(matrix, symmetric = TRUE, only.values = TRUE)$values
  )
  for (statistic in names(spectra)) {
    write.table(
      spectra[[statistic]],
      file.path(output_dir, paste0("eigenvalues_", statistic, ".csv")),
      row.names = FALSE,
      col.names = FALSE,
      sep = ","
    )
  }
} else {
  if (selected_statistic == "all") {
    stop("KPM mode requires one statistic or the BKR/tau* pair")
  }
  probes <- as.integer(Sys.getenv("KENDALL_KPM_PROBES", "16"))
  degree <- as.integer(Sys.getenv("KENDALL_KPM_DEGREE", "160"))
  spectral_bound <- as.numeric(
    Sys.getenv("KENDALL_KPM_SPECTRAL_BOUND", "2.6")
  )
  kpm_seed <- as.integer(Sys.getenv("KENDALL_KPM_SEED", "20260730"))
  tau_constants <- c(
    hoeffding_d = 1 / 3,
    bkr_r = 1,
    bdy_tau_star = 3 / 5
  )
  standardization_scales <- sqrt((n * (n - 1) / 2) / p) /
    tau_constants[names(rank_matrices)]
  requested_scale <- as.numeric(
    Sys.getenv("KENDALL_KPM_STANDARDIZATION_SCALE")
  )
  if (length(rank_matrices) == 1L && !is.na(requested_scale)) {
    standardization_scales[[1L]] <- requested_scale
  }
  kpm_start <- proc.time()[["elapsed"]]
  if (!is.null(compact_pair_kpm)) {
    standardization_scales <- c(
      bkr_r = sqrt((n * (n - 1) / 2) / p),
      bdy_tau_star = sqrt((n * (n - 1) / 2) / p) / (3 / 5)
    )
    for (statistic in c("bkr_r", "bdy_tau_star")) {
      write.table(
        compact_pair_kpm[[statistic]],
        file.path(
          output_dir,
          paste0("kpm_moments_", statistic, ".csv")
        ),
        row.names = FALSE,
        col.names = FALSE,
        sep = ","
      )
      cat(
        sprintf(
          paste0(
            "KPM statistic=%s probes=%d degree=%d spectral_bound=%.6f ",
            "standardization_scale=%.10f\n"
          ),
          statistic,
          probes,
          degree,
          spectral_bound,
          standardization_scales[[statistic]]
        )
      )
    }
    cat(
      sprintf(
        paste0(
          "compact build_seconds=%.6f bkr_kpm_seconds=%.6f ",
          "tau_kpm_seconds=%.6f\n"
        ),
        compact_pair_kpm[["build_seconds"]],
        compact_pair_kpm[["bkr_kpm_seconds"]],
        compact_pair_kpm[["tau_kpm_seconds"]]
      )
    )
  } else {
    for (statistic_index in seq_along(rank_matrices)) {
      statistic <- names(rank_matrices)[[statistic_index]]
      moments <- hoeffding_kpm_moments_cpp(
        rank_matrices[[statistic]],
        probes,
        degree,
        spectral_bound,
        standardization_scales[[statistic]],
        kpm_seed + statistic_index - 1L
      )
      write.table(
        moments,
        file.path(
          output_dir,
          paste0("kpm_moments_", statistic, ".csv")
        ),
        row.names = FALSE,
        col.names = FALSE,
        sep = ","
      )
      cat(
        sprintf(
          paste0(
            "KPM statistic=%s probes=%d degree=%d spectral_bound=%.6f ",
            "standardization_scale=%.10f\n"
          ),
          statistic,
          probes,
          degree,
          spectral_bound,
          standardization_scales[[statistic]]
        )
      )
    }
  }
  kpm_elapsed <- proc.time()[["elapsed"]] - kpm_start
  cat(
    sprintf(
      paste0(
        "KPM probes=%d degree=%d spectral_bound=%.6f ",
        "elapsed_seconds=%.6f\n"
      ),
      probes,
      degree,
      spectral_bound,
      kpm_elapsed
    )
  )
}

cat(
  sprintf(
    paste0(
      "FAST_RANK_BACKEND n=%d p=%d pairs=%d elapsed_seconds=%.6f ",
      "mode=%s statistic=%s engine=%s package=independence-1.0.1\n"
    ),
    n,
    p,
    p * (p - 1L) / 2L,
    elapsed,
    output_mode,
    selected_statistic,
    engine
  )
)
