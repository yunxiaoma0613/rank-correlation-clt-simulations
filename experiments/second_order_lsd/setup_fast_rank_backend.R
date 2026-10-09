args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) {
  stop("usage: Rscript setup_fast_rank_backend.R <library-directory>")
}

library_dir <- normalizePath(args[[1L]], mustWork = FALSE)
dir.create(library_dir, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(library_dir, .libPaths()))

if (!requireNamespace("Rcpp", quietly = TRUE)) {
  install.packages(
    "Rcpp",
    lib = library_dir,
    repos = "https://cloud.r-project.org",
    type = "source"
  )
}

if (!requireNamespace("independence", quietly = TRUE)) {
  install.packages(
    "https://cran.r-project.org/src/contrib/Archive/independence/independence_1.0.1.tar.gz",
    lib = library_dir,
    repos = NULL,
    type = "source"
  )
}

if (as.character(packageVersion("independence")) != "1.0.1") {
  stop("the validated backend requires independence version 1.0.1")
}

cat("installed independence 1.0.1 in", library_dir, "\n")

