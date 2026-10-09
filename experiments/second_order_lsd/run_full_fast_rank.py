from __future__ import annotations

import argparse
import csv
from datetime import date
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.full_rank_statistics import full_rank_matrix  # noqa: E402
from src.spectral_simulation import (  # noqa: E402
    STATISTICS,
    affine_density,
    affine_quantiles,
    edge_features_from_uniforms,
    mp_grid,
    wasserstein_quantile_distance,
)


BACKEND = ROOT / "src" / "fast_full_rank_matrix.R"
STATISTIC_LABELS = {
    "hoeffding_d": "Hoeffding D",
    "bkr_r": "BKR R",
    "bdy_tau_star": "BDY tau*",
}


def parse_args() -> argparse.Namespace:
    default_r_lib = os.environ.get("KENDALL_FAST_R_LIB", str(ROOT / ".r_lib"))
    parser = argparse.ArgumentParser(
        description="Compare exact fast rank matrices with their second-order spectra."
    )
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--p", type=int, default=900)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--r-lib", type=Path, default=Path(default_r_lib))
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "full_n30_p900"
    )
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def run_fast_backend(
    uniforms: np.ndarray,
    work_dir: Path,
    rscript: str,
    r_lib: Path,
    verbose: bool = True,
) -> tuple[dict[str, np.ndarray], str, float]:
    input_path = work_dir / "uniforms.csv"
    matrix_dir = work_dir / "matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(input_path, uniforms, delimiter=",", fmt="%.17g")

    environment = os.environ.copy()
    environment["R_LIBS"] = str(r_lib)
    start = time.perf_counter()
    completed = subprocess.run(
        [rscript, str(BACKEND), str(input_path), str(matrix_dir)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    elapsed = time.perf_counter() - start
    if completed.returncode != 0:
        raise RuntimeError(
            "fast R backend failed:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    if completed.stdout and verbose:
        print(completed.stdout.strip(), flush=True)

    matrices = {
        statistic: np.loadtxt(matrix_dir / f"full_{statistic}.csv", delimiter=",")
        for statistic in STATISTICS
    }
    return matrices, completed.stdout.strip(), elapsed


def run_fast_spectrum_backend(
    uniforms: np.ndarray,
    work_dir: Path,
    rscript: str,
    r_lib: Path,
    verbose: bool = True,
    statistic: str | None = None,
) -> tuple[dict[str, np.ndarray], str, float]:
    input_path = work_dir / "uniforms.csv"
    spectrum_dir = work_dir / "spectra"
    spectrum_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(input_path, uniforms, delimiter=",", fmt="%.17g")

    environment = os.environ.copy()
    environment["R_LIBS"] = str(r_lib)
    start = time.perf_counter()
    command = [
        rscript,
        str(BACKEND),
        str(input_path),
        str(spectrum_dir),
        "spectra",
    ]
    if statistic is not None:
        if statistic not in STATISTICS:
            raise ValueError(f"unknown statistic: {statistic}")
        command.append(statistic)
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    elapsed = time.perf_counter() - start
    if completed.returncode != 0:
        raise RuntimeError(
            "fast R spectrum backend failed:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    if completed.stdout and verbose:
        print(completed.stdout.strip(), flush=True)

    selected_statistics = tuple(STATISTICS) if statistic is None else (statistic,)
    spectra = {
        selected: np.atleast_1d(
            np.loadtxt(
                spectrum_dir / f"eigenvalues_{selected}.csv",
                delimiter=",",
            )
        )
        for selected in selected_statistics
    }
    return spectra, completed.stdout.strip(), elapsed


def validate_backend(rscript: str, r_lib: Path) -> dict[str, float]:
    rng = np.random.default_rng(918_273)
    uniforms = rng.random((8, 5))
    with (
        tempfile.TemporaryDirectory(
            prefix="kendall-fast-rank-validation-"
        ) as matrix_tmp,
        tempfile.TemporaryDirectory(
            prefix="kendall-fast-spectrum-validation-"
        ) as spectrum_tmp,
    ):
        fast, _, _ = run_fast_backend(
            uniforms, Path(matrix_tmp), rscript, r_lib
        )
        fast_spectra, _, _ = run_fast_spectrum_backend(
            uniforms, Path(spectrum_tmp), rscript, r_lib
        )

    maximum_errors: dict[str, float] = {}
    for statistic in STATISTICS:
        reference = full_rank_matrix(uniforms, statistic)
        matrix_error = float(np.max(np.abs(fast[statistic] - reference)))
        spectrum_error = float(
            np.max(
                np.abs(
                    np.sort(fast_spectra[statistic])
                    - np.linalg.eigvalsh(reference)
                )
            )
        )
        error = max(matrix_error, spectrum_error)
        maximum_errors[statistic] = error
        if error > 1e-12:
            raise AssertionError(
                f"fast {statistic} backend differs from the enumerated "
                f"kernel by {error}"
            )
    return maximum_errors


def write_metrics(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_density_plot(
    output_dir: Path,
    n: int,
    p: int,
    rho: float,
    full_eigenvalues: dict[str, np.ndarray],
    leading_eigenvalues: dict[str, np.ndarray],
    rows: list[dict[str, float | int | str]],
) -> Path:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows_by_statistic = {str(row["statistic"]): row for row in rows}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for axis, (statistic, tau) in zip(axes, STATISTICS.items()):
        full = full_eigenvalues[statistic]
        leading = leading_eigenvalues[statistic]
        row = rows_by_statistic[statistic]
        shift = 1.0 - tau
        grid = mp_grid(rho)
        support_low = shift + tau * grid.support_low
        support_high = shift + tau * grid.support_high
        lower = min(float(np.min(full)), float(np.min(leading)), shift, support_low)
        upper = max(float(np.max(full)), float(np.max(leading)), support_high)
        padding = 0.05 * max(upper - lower, 0.5)
        bins = np.linspace(lower - padding, upper + padding, 76)
        x = np.linspace(lower - padding, upper + padding, 1_800)

        axis.hist(
            full,
            bins=bins,
            density=True,
            color="#6BA3B8",
            alpha=0.62,
            edgecolor="white",
            linewidth=0.3,
            label="full U-statistic",
        )
        axis.hist(
            leading,
            bins=bins,
            density=True,
            histtype="step",
            color="#D08A32",
            linewidth=1.6,
            label="second-order R^(1)",
        )
        axis.plot(
            x,
            affine_density(x, tau, rho),
            color="#B43A34",
            linewidth=1.9,
            label="affine-MP density",
        )
        if rho > 1.0:
            axis.axvline(
                shift,
                color="#39734A",
                linestyle="--",
                linewidth=1.4,
                label=f"limit atom ({grid.atom_mass:.2f})",
            )
        axis.set_title(
            f"{STATISTIC_LABELS[statistic]}, tau={tau:.3f}\n"
            f"W1(full, R^(1))={float(row['wasserstein_full_vs_leading']):.4f}"
        )
        axis.set_xlabel("eigenvalue")
        axis.set_ylabel("density")
        axis.grid(alpha=0.18)
    axes[-1].legend(fontsize=8, frameon=False)
    fig.suptitle(
        f"Complete rank matrices versus second-order approximation: "
        f"n={n}, p={p}, rho=p/C(n,2)={rho:.3f}",
        fontsize=14,
    )
    path = output_dir / f"full_vs_second_order_density_n{n}_p{p}.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    if args.n < 6 or args.p < 2:
        raise ValueError("the joint experiment requires n>=6 and p>=2")
    if not args.r_lib.exists():
        raise FileNotFoundError(
            f"R library {args.r_lib} does not exist; run setup_fast_rank_backend.R"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    validation_errors: dict[str, float] = {}
    if not args.skip_validation:
        print("validating fast backend against enumerated U-statistic kernels", flush=True)
        validation_errors = validate_backend(args.rscript, args.r_lib)
        print(f"validation maximum errors: {validation_errors}", flush=True)

    rng = np.random.default_rng(args.seed)
    uniforms = rng.random((args.n, args.p))
    m_edges = args.n * (args.n - 1) // 2
    rho = args.p / float(m_edges)

    with tempfile.TemporaryDirectory(prefix="kendall-fast-rank-full-") as tmp:
        print(f"computing complete matrices for n={args.n}, p={args.p}", flush=True)
        full_matrices, backend_log, backend_elapsed = run_fast_backend(
            uniforms, Path(tmp), args.rscript, args.r_lib
        )

    second_order_start = time.perf_counter()
    features = edge_features_from_uniforms(uniforms)
    d0_gram = (features.T @ features) / float(m_edges)
    del features
    np.fill_diagonal(d0_gram, 0.0)
    eigen_d0_gram = np.linalg.eigvalsh(d0_gram)
    second_order_elapsed = time.perf_counter() - second_order_start

    grid = mp_grid(rho)
    rows: list[dict[str, float | int | str]] = []
    full_eigenvalues: dict[str, np.ndarray] = {}
    leading_eigenvalues: dict[str, np.ndarray] = {}
    eigendecomposition_seconds: dict[str, float] = {}

    for statistic, tau in STATISTICS.items():
        start = time.perf_counter()
        full_eigen = np.linalg.eigvalsh(full_matrices[statistic])
        eigendecomposition_seconds[statistic] = time.perf_counter() - start
        leading = np.eye(args.p) + tau * d0_gram
        leading_eigen = 1.0 + tau * eigen_d0_gram
        theoretical_quantiles = affine_quantiles(args.p, tau, grid)
        difference = full_matrices[statistic] - leading

        full_eigenvalues[statistic] = full_eigen
        leading_eigenvalues[statistic] = leading_eigen
        rows.append(
            {
                "date": date.today().isoformat(),
                "seed": args.seed,
                "n": args.n,
                "M": m_edges,
                "p": args.p,
                "rho_actual": rho,
                "statistic": statistic,
                "tau": tau,
                "wasserstein_full_vs_leading": float(
                    np.mean(np.abs(full_eigen - leading_eigen))
                ),
                "wasserstein_full_vs_affine_mp": wasserstein_quantile_distance(
                    full_eigen, theoretical_quantiles
                ),
                "wasserstein_leading_vs_affine_mp": wasserstein_quantile_distance(
                    leading_eigen, theoretical_quantiles
                ),
                "normalized_frobenius_full_vs_leading": float(
                    np.sum(difference * difference) / args.p
                ),
                "full_mean": float(np.mean(full_eigen)),
                "leading_mean": float(np.mean(leading_eigen)),
                "full_variance": float(np.var(full_eigen)),
                "leading_variance": float(np.var(leading_eigen)),
                "affine_mp_variance": tau * tau * rho,
                "full_minimum": float(np.min(full_eigen)),
                "full_maximum": float(np.max(full_eigen)),
                "leading_minimum": float(np.min(leading_eigen)),
                "leading_maximum": float(np.max(leading_eigen)),
                "atom_location": 1.0 - tau,
                "atom_mass_theory": grid.atom_mass,
                "fast_backend_seconds_all_statistics": backend_elapsed,
                "full_eigendecomposition_seconds": eigendecomposition_seconds[statistic],
                "second_order_build_and_eigendecomposition_seconds": second_order_elapsed,
                "validation_max_abs_error": validation_errors.get(statistic, float("nan")),
                "fast_backend_package": "independence-1.0.1",
            }
        )

    metrics_path = args.output_dir / "full_vs_second_order_metrics.csv"
    write_metrics(metrics_path, rows)
    figure_path = save_density_plot(
        args.output_dir,
        args.n,
        args.p,
        rho,
        full_eigenvalues,
        leading_eigenvalues,
        rows,
    )

    artifact_payload: dict[str, np.ndarray] = {
        "uniforms": uniforms,
        "eigen_d0_gram": eigen_d0_gram,
    }
    for statistic in STATISTICS:
        artifact_payload[f"full_matrix_{statistic}"] = full_matrices[statistic]
        artifact_payload[f"full_eigenvalues_{statistic}"] = full_eigenvalues[statistic]
        artifact_payload[f"leading_eigenvalues_{statistic}"] = leading_eigenvalues[statistic]
    artifact_path = args.output_dir / "full_matrices_and_spectra.npz"
    np.savez_compressed(artifact_path, **artifact_payload)

    metadata = {
        "date": date.today().isoformat(),
        "seed": args.seed,
        "n": args.n,
        "p": args.p,
        "M": m_edges,
        "rho_actual": rho,
        "fast_backend_package": "independence-1.0.1",
        "backend_log": backend_log,
        "validation_max_abs_errors": validation_errors,
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"wrote {figure_path}", flush=True)
    print(f"wrote {metrics_path}", flush=True)
    print(f"wrote {artifact_path}", flush=True)
    print(f"wrote {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
