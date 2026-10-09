from __future__ import annotations

import argparse
import csv
from datetime import date
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from run_full_fast_rank import (  # noqa: E402
    run_fast_spectrum_backend,
    validate_backend,
)
from src.spectral_simulation import (  # noqa: E402
    STATISTICS,
    affine_density,
    affine_quantiles,
    mp_grid,
    wasserstein_quantile_distance,
)


DEFAULT_CONFIGURATIONS = ((50, 600), (60, 1000), (70, 2000))
STATISTIC_LABELS = {
    "hoeffding_d": "Hoeffding's D",
    "bkr_r": "BKR R",
    "bdy_tau_star": r"BDY $\tau^*$",
}

LIMIT_DENSITY_LABELS = {
    "hoeffding_d": r"$\frac{2}{3}+\frac{1}{3}\,\mathrm{MP}(q)$",
    "bkr_r": r"$\mathrm{MP}(q)$",
    "bdy_tau_star": r"$\frac{2}{5}+\frac{3}{5}\,\mathrm{MP}(q)$",
}


def parse_configuration(value: str) -> tuple[int, int]:
    try:
        n_text, p_text = value.split(":", maxsplit=1)
        n, p = int(n_text), int(p_text)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("configurations must have the form n:p") from error
    if n < 6 or p < 2:
        raise argparse.ArgumentTypeError("each configuration requires n>=6 and p>=2")
    return n, p


def parse_args() -> argparse.Namespace:
    default_r_lib = os.environ.get("KENDALL_FAST_R_LIB", str(ROOT / ".r_lib"))
    parser = argparse.ArgumentParser(
        description="Full rank-statistic ESD densities for three dimension settings."
    )
    parser.add_argument(
        "--configurations",
        nargs="+",
        type=parse_configuration,
        default=list(DEFAULT_CONFIGURATIONS),
    )
    parser.add_argument(
        "--q",
        type=float,
        default=None,
        help="Use one common limiting ratio instead of each configuration's p/M.",
    )
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--r-lib", type=Path, default=Path(default_r_lib))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "selected_full_density",
    )
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def actual_ratio(n: int, p: int) -> float:
    return p / float(n * (n - 1) // 2)


def spectrum_cache_path(output_dir: Path, n: int, p: int) -> Path:
    return output_dir / "spectra" / f"full_spectra_n{n}_p{p}.npz"


def load_spectra(
    path: Path, n: int, p: int, seed: int
) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as payload:
        if (
            int(payload["n"]) != n
            or int(payload["p"]) != p
            or int(payload["seed"]) != seed
        ):
            return None
        return {
            statistic: np.asarray(payload[f"eigenvalues_{statistic}"], dtype=float)
            for statistic in STATISTICS
        }


def load_existing_full_experiment(
    n: int, p: int, seed: int
) -> dict[str, np.ndarray] | None:
    directory = ROOT / "results" / f"full_n{n}_p{p}"
    artifact = directory / "full_matrices_and_spectra.npz"
    metadata = directory / "metadata.json"
    if not artifact.exists() or not metadata.exists():
        return None
    details = json.loads(metadata.read_text(encoding="utf-8"))
    if int(details["n"]) != n or int(details["p"]) != p or int(details["seed"]) != seed:
        return None
    with np.load(artifact, allow_pickle=False) as payload:
        return {
            statistic: np.asarray(
                payload[f"full_eigenvalues_{statistic}"], dtype=float
            )
            for statistic in STATISTICS
        }


def save_spectra(
    path: Path,
    spectra: dict[str, np.ndarray],
    n: int,
    p: int,
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray | int | float] = {
        "n": n,
        "p": p,
        "M": n * (n - 1) // 2,
        "q_actual": actual_ratio(n, p),
        "seed": seed,
    }
    for statistic, eigenvalues in spectra.items():
        payload[f"eigenvalues_{statistic}"] = eigenvalues
    np.savez_compressed(path, **payload)


def simulate_full_spectra(
    n: int,
    p: int,
    seed: int,
    rscript: str,
    r_lib: Path,
) -> tuple[dict[str, np.ndarray], float]:
    uniforms = np.random.default_rng(seed).random((n, p))
    with tempfile.TemporaryDirectory(prefix="fixed-q-full-rank-") as tmp:
        spectra, _, backend_seconds = run_fast_spectrum_backend(
            uniforms, Path(tmp), rscript, r_lib, verbose=False
        )
    return spectra, backend_seconds


def metric_rows(
    configurations: list[tuple[int, int]],
    spectra_by_configuration: dict[tuple[int, int], dict[str, np.ndarray]],
    q_target: float | None,
    seed: int,
    sources: dict[tuple[int, int], str],
    elapsed: dict[tuple[int, int], float],
) -> list[dict[str, int | float | str]]:
    rows: list[dict[str, int | float | str]] = []
    for n, p in configurations:
        m_edges = n * (n - 1) // 2
        q_actual = p / float(m_edges)
        q_theory = q_actual if q_target is None else q_target
        grid = mp_grid(q_theory)
        for statistic, tau in STATISTICS.items():
            eigenvalues = spectra_by_configuration[(n, p)][statistic]
            quantiles = affine_quantiles(p, tau, grid)
            rows.append(
                {
                    "date": date.today().isoformat(),
                    "seed": seed,
                    "n": n,
                    "M": m_edges,
                    "p": p,
                    "q_theory": q_theory,
                    "q_actual": q_actual,
                    "statistic": statistic,
                    "tau": tau,
                    "wasserstein_1_to_limit": wasserstein_quantile_distance(
                        eigenvalues, quantiles
                    ),
                    "empirical_mean": float(np.mean(eigenvalues)),
                    "empirical_variance": float(np.var(eigenvalues)),
                    "limit_variance": tau * tau * q_theory,
                    "minimum_eigenvalue": float(np.min(eigenvalues)),
                    "maximum_eigenvalue": float(np.max(eigenvalues)),
                    "spectrum_source": sources[(n, p)],
                    "backend_seconds": elapsed[(n, p)],
                    "replications": 1,
                }
            )
    return rows


def write_metrics(path: Path, rows: list[dict[str, int | float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_density_grid(
    output_dir: Path,
    configurations: list[tuple[int, int]],
    spectra_by_configuration: dict[tuple[int, int], dict[str, np.ndarray]],
    q_target: float | None,
) -> tuple[Path, Path]:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(STATISTICS),
        len(configurations),
        figsize=(13.8, 10.2),
        sharex="row",
        sharey="row",
        constrained_layout=True,
    )

    for row_index, (statistic, tau) in enumerate(STATISTICS.items()):
        row_spectra = [
            spectra_by_configuration[configuration][statistic]
            for configuration in configurations
        ]
        theoretical_supports = []
        for n, p in configurations:
            q_theory = actual_ratio(n, p) if q_target is None else q_target
            grid = mp_grid(q_theory)
            shift = 1.0 - tau
            theoretical_supports.append(
                (shift + tau * grid.support_low, shift + tau * grid.support_high)
            )
        row_low = min(
            *(support[0] for support in theoretical_supports),
            *(float(np.min(values)) for values in row_spectra),
        )
        row_high = max(
            *(support[1] for support in theoretical_supports),
            *(float(np.max(values)) for values in row_spectra),
        )
        padding = 0.055 * max(row_high - row_low, 0.25)
        bins = np.linspace(row_low - padding, row_high + padding, 58)
        x = np.linspace(row_low - padding, row_high + padding, 2_000)

        for column_index, (n, p) in enumerate(configurations):
            axis = axes[row_index, column_index]
            eigenvalues = spectra_by_configuration[(n, p)][statistic]
            q_theory = actual_ratio(n, p) if q_target is None else q_target
            limit_density = affine_density(x, tau, q_theory)
            axis.hist(
                eigenvalues,
                bins=bins,
                density=True,
                color="#83B7C9",
                edgecolor="white",
                linewidth=0.35,
                alpha=0.82,
            )
            limit_line, = axis.plot(
                x,
                limit_density,
                color="#B53A32",
                linewidth=2.0,
                label=LIMIT_DENSITY_LABELS[statistic],
            )
            if row_index == 0:
                axis.set_title(
                    rf"$n={n},\ p={p},\ q={actual_ratio(n, p):.2f}$",
                    fontsize=10,
                    pad=6,
                )
            axis.legend(
                handles=[limit_line],
                loc="upper right",
                frameon=False,
                fontsize=12,
                handlelength=2.2,
                handletextpad=0.55,
                borderaxespad=0.45,
            )
            axis.grid(color="#C9C9C9", alpha=0.32, linewidth=0.6)
            axis.tick_params(labelsize=9)
            if column_index == 0:
                axis.set_ylabel(
                    f"{STATISTIC_LABELS[statistic]}\nDensity", fontsize=11
                )
            if row_index == len(STATISTICS) - 1:
                axis.set_xlabel("Eigenvalue", fontsize=10)
            panel_index = row_index * len(configurations) + column_index
            axis.text(
                0.025,
                0.96,
                f"({chr(ord('a') + panel_index)})",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=10,
            )

    png_path = output_dir / "full_esd_density_grid.png"
    pdf_path = output_dir / "full_esd_density_grid.pdf"
    figure.savefig(png_path, dpi=240, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def main() -> None:
    args = parse_args()
    if args.q is not None and args.q <= 0.0:
        raise ValueError("q must be positive")
    configurations = list(dict.fromkeys(args.configurations))
    if len(configurations) != 3:
        raise ValueError("the 3 by 3 figure requires exactly three configurations")
    if not args.r_lib.exists():
        raise FileNotFoundError(
            f"R library {args.r_lib} does not exist; run setup_fast_rank_backend.R"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_validation:
        errors = validate_backend(args.rscript, args.r_lib)
        print(f"validation maximum errors: {errors}", flush=True)

    spectra_by_configuration: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    sources: dict[tuple[int, int], str] = {}
    elapsed: dict[tuple[int, int], float] = {}

    for n, p in configurations:
        cache = spectrum_cache_path(args.output_dir, n, p)
        spectra = None if args.force else load_spectra(cache, n, p, args.seed)
        source = "spectrum cache"
        seconds = 0.0
        if spectra is None and not args.force:
            spectra = load_existing_full_experiment(n, p, args.seed)
            source = "existing full experiment"
        if spectra is None:
            start = time.perf_counter()
            print(
                f"simulating complete matrices n={n}, p={p}, "
                f"q_n={actual_ratio(n, p):.6f}",
                flush=True,
            )
            spectra, backend_seconds = simulate_full_spectra(
                n, p, args.seed, args.rscript, args.r_lib
            )
            seconds = time.perf_counter() - start
            source = f"computed, R backend {backend_seconds:.3f}s"
        save_spectra(cache, spectra, n, p, args.seed)
        spectra_by_configuration[(n, p)] = spectra
        sources[(n, p)] = source
        elapsed[(n, p)] = seconds
        print(f"loaded n={n}, p={p} from {source}", flush=True)

    rows = metric_rows(
        configurations,
        spectra_by_configuration,
        args.q,
        args.seed,
        sources,
        elapsed,
    )
    metrics_path = args.output_dir / "full_esd_density_metrics.csv"
    write_metrics(metrics_path, rows)
    png_path, pdf_path = save_density_grid(
        args.output_dir, configurations, spectra_by_configuration, args.q
    )
    print(f"wrote {png_path}", flush=True)
    print(f"wrote {pdf_path}", flush=True)
    print(f"wrote {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
