# Rank-correlation matrices: numerical simulations

Simulation code accompanying *A Central Limit Theorem and a Spectral
Transition for a Class of Rank Correlation Matrices Beyond the Linear Regime*.

This source package implements the numerical designs in Section 3 of the
manuscript dated October 9, 2026. It constructs the complete Hoeffding D,
Blum--Kiefer--Rosenblatt R, and Bergsma--Dassios--Yanagimoto tau-star matrices.
The main experiments do not substitute a second-order Gram approximation.

## Contents

| Manuscript experiment | Program |
| --- | --- |
| Semicircle densities and intermediate-regime Wasserstein discrepancies | `experiments/second_order_lsd/run_intermediate_semicircle_grid.py` |
| Affine MP densities from a single realization | `experiments/second_order_lsd/run_full_density_grid.py` |
| Quadratic-regime Wasserstein discrepancies | `experiments/second_order_lsd/run_replicated_full_density_grid.py` |
| Null sizes, Toeplitz powers, and nonlinear powers of six LSS tests | `experiments/polynomial_lss_independence_test/run_experiment.py` |
| Finite centers and limiting means and variances for powers 2 and 4 | `experiments/polynomial_lss_clt_validation/theory.py` |

Only simulation source, tests, and documentation are included. No manuscript,
real patient data, credentials, local R installation, or binary build cache is
included. The numerical source files are unchanged copies of the project
source; `SOURCE_MANIFEST.sha256` records their checksums.

## Environment

The source was tested on macOS with Python 3.9.13, NumPy 1.23.1, SciPy 1.11.3,
Matplotlib 3.5.3, R 4.2.2, Rcpp 1.1.2, and independence 1.0.1. The R build
scripts currently contain macOS compiler flags; Windows and Linux portability
has not been verified. A working R/C++ toolchain is required.

Run all commands from this directory. Python 3.9 is recommended for the pinned
historical dependencies.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r experiments/second_order_lsd/requirements-reproducibility.txt
Rscript experiments/second_order_lsd/setup_fast_rank_backend.R experiments/second_order_lsd/.r_lib
export KENDALL_FAST_R_LIB="$PWD/experiments/second_order_lsd/.r_lib"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MPLCONFIGDIR="$PWD/.mplconfig"
```

The R installer pins independence to version 1.0.1 and installs Rcpp if needed.
It does not install a C++ compiler. Keep the directory structure intact because
the simulation drivers share the rank-matrix backend and theoretical formulas.

## Quick checks

```sh
python -m unittest discover -s experiments/second_order_lsd/tests -v
python -m unittest discover -s experiments/polynomial_lss_clt_validation -v
python -m unittest experiments.polynomial_lss_independence_test.test_experiment -v
python experiments/polynomial_lss_independence_test/run_experiment.py --n 8 --q 0.7 --replications 2 --jobs 1 --signal-scale 3 --scenario I:null --output-dir outputs/smoke
```

The optional R integration test compares all three fast matrices with directly
enumerated symmetrized kernels on a small sample. It is skipped unless
`KENDALL_FAST_R_LIB` is set. The smoke run is only an implementation check, not
an estimate of the rejection probabilities reported in the paper.

## Reproduce the main-text designs

### Semicircle experiment

The three dimension pairs are `(300,939)`, `(450,1527)`, and `(600,2157)`.
The experiment uses 50 independent replications and standardizes each matrix
by its corresponding tau-zero constant: 1/3 for D, 1 for R, and 3/5 for
tau-star.

```sh
python experiments/second_order_lsd/run_intermediate_semicircle_grid.py --exponent 1.2 --configurations 300:939 450:1527 600:2157 --replications 50 --workers 2 --seed 20260721 --output-dir outputs/semicircle
```

Use `--replications 1` and a different output directory for a single-realization
density plot. The replicated driver also saves individual spectra and summary
metrics.

### Quadratic-regime experiment

The dimension pairs are `(70,500)`, `(90,1000)`, and `(120,2000)`. The target
affine MP distributions here use the finite ratio `p / binom(n,2)`, as specified
in the LSD simulation section, not the ratio used for LSS test calibration.

```sh
python experiments/second_order_lsd/run_full_density_grid.py --configurations 70:500 90:1000 120:2000 --seed 20260711 --output-dir outputs/mp_single
python experiments/second_order_lsd/run_replicated_full_density_grid.py --configurations 70:500 90:1000 120:2000 --replications 100 --workers 2 --seed 20260718 --output-dir outputs/mp_replicated
```

Wasserstein discrepancies use the mean absolute difference between ordered
eigenvalues and limiting quantiles at `(i-1/2)/p`.

### Independence tests

```sh
python experiments/polynomial_lss_independence_test/run_experiment.py --n 50 70 100 --q 0.3 0.7 1.0 --replications 1000 --jobs 2 --seed 20260920 --signal-scale 3 --output-dir outputs/independence_tests
```

Omitting `--scenario` runs all eight scenarios: null and Toeplitz alternatives
for Models I, II, and III, and nonlinear alternatives for Models II and III.
Model I uses Gamma(4,0.5) and Student-t(5) margins, Model II uses Cauchy and
Student-t(5) margins, and Model III uses Cauchy margins throughout. The split
is at `floor(p/2)`.

The dimension rule is `p = round(q*n*(n-3)/2)`, with ties to even, as in Python.
The actual ratio `q_n = p/(binom(n,2)-n)` is used in every calibration formula.
`--signal-scale 3` gives Toeplitz coefficients `(0.18,0.09,0.06)` and nonlinear
coefficients `(0.03,0.06,0.018,0.5)`, matching the current manuscript. The
nonlinear generator draws `p+2` base coordinates and retains the split at
`floor(p/2)`.

For each of D, R, and tau-star, the program computes `Tr(H^2)` and `Tr(H^4)`.
It subtracts the finite center and limiting mean, divides by the limiting
standard deviation, and compares with the upper standard-normal 0.95 quantile.
No permutation calibration or empirical recentering is substituted.

Full runs are computationally expensive, particularly the fourth-power traces
at `p=4850`. The release checks do not rerun the full 1000-replication campaign.
The programs save raw traces, rejection-rate summaries, seeds, dimensions, and
run metadata. Use a fresh output directory for any changed configuration;
existing raw files can be resumed and are not automatically invalidated when
the user changes settings.

## Reproducibility scope

These commands reproduce the specified statistical designs. An exact
cell-by-cell reproduction of every printed table value has not been certified
in this packaging step. Historical raw runs and their complete seed histories
are not bundled. Floating-point libraries can also affect the last digits.

See `VALIDATION.md` for the checks actually performed, and
`THIRD_PARTY_NOTICES.md` for backend provenance and licensing information.

## License

Copyright (C) 2026 Yunxiao Ma and Wang Zhou, except for the upstream and adapted
code identified in `THIRD_PARTY_NOTICES.md`.

This code is distributed under the GNU General Public License, version 3 or
(at your option) any later version (SPDX: GPL-3.0-or-later). See `LICENSE`.
It is provided without warranty, including the implied warranties of
merchantability or fitness for a particular purpose.

## Repository

Public source repository:
https://github.com/yunxiaoma0613/rank-correlation-clt-simulations

The repository contains simulation code, not the manuscript or its proof
supplement. No DOI has been assigned to this code release.
