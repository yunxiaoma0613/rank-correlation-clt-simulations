from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import date
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from run_full_density_grid import (  # noqa: E402
    STATISTIC_LABELS,
    load_spectra,
    parse_configuration,
    save_spectra,
    simulate_full_spectra,
    validate_backend,
)
from src.spectral_simulation import STATISTICS  # noqa: E402


DEFAULT_EXPONENT = 1.3
DEFAULT_SAMPLE_SIZES = (100, 160, 250)


def parse_args() -> argparse.Namespace:
    default_r_lib = os.environ.get("KENDALL_FAST_R_LIB", str(ROOT / ".r_lib"))
    parser = argparse.ArgumentParser(
        description="Theorem 2 semicircle simulation for complete rank matrices."
    )
    parser.add_argument("--exponent", type=float, default=DEFAULT_EXPONENT)
    parser.add_argument(
        "--sample-sizes",
        nargs="+",
        type=int,
        default=list(DEFAULT_SAMPLE_SIZES),
    )
    parser.add_argument(
        "--configurations",
        nargs="+",
        type=parse_configuration,
        default=None,
        help="Optional explicit n:p configurations.",
    )
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--r-lib", type=Path, default=Path(default_r_lib))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def default_configurations(
    sample_sizes: list[int], exponent: float
) -> list[tuple[int, int]]:
    if not 1.0 < exponent < 2.0:
        raise ValueError("the exponent must lie in (1, 2)")
    if len(sample_sizes) != 3:
        raise ValueError("the 3 by 3 figure requires exactly three sample sizes")
    if any(n < 6 for n in sample_sizes):
        raise ValueError("all sample sizes must be at least six")
    return [(n, int(round(n**exponent))) for n in sample_sizes]


def exponent_tag(exponent: float) -> str:
    return str(exponent).replace(".", "p")


def replication_seed(master_seed: int, n: int, p: int, replication: int) -> int:
    return master_seed + 10_000_000 * replication + 10_000 * n + p


def cache_path(output_dir: Path, n: int, p: int, replication: int) -> Path:
    return (
        output_dir
        / "spectra"
        / f"full_spectra_n{n}_p{p}_rep{replication + 1:02d}.npz"
    )


def standardize_spectrum(
    eigenvalues: np.ndarray, n: int, p: int, tau_0: float
) -> np.ndarray:
    m_edges = n * (n - 1) // 2
    return np.sqrt(m_edges / p) * (np.asarray(eigenvalues) - 1.0) / tau_0


def semicircle_density(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.sqrt(np.maximum(4.0 - values * values, 0.0)) / (2.0 * np.pi)


def semicircle_quantiles(size: int, grid_size: int = 200_001) -> np.ndarray:
    if size < 1:
        raise ValueError("size must be positive")
    grid = np.linspace(-2.0, 2.0, grid_size)
    root = np.sqrt(np.maximum(4.0 - grid * grid, 0.0))
    cdf = 0.5 + grid * root / (4.0 * np.pi) + np.arcsin(grid / 2.0) / np.pi
    probabilities = (np.arange(size, dtype=float) + 0.5) / size
    return np.interp(probabilities, cdf, grid)


def wasserstein_to_semicircle(values: np.ndarray) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    return float(np.mean(np.abs(ordered - semicircle_quantiles(ordered.size))))


def write_csv(path: Path, rows: list[dict[str, int | float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def quantile(values: list[float], probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), probability))


def compute_metrics(
    configurations: list[tuple[int, int]],
    spectra: dict[tuple[int, int], list[dict[str, np.ndarray]]],
    master_seed: int,
    exponent: float,
) -> tuple[
    list[dict[str, int | float | str]],
    list[dict[str, int | float | str]],
]:
    replicate_rows: list[dict[str, int | float | str]] = []
    summary_rows: list[dict[str, int | float | str]] = []
    for n, p in configurations:
        m_edges = n * (n - 1) // 2
        for statistic, tau_0 in STATISTICS.items():
            standardized_replicates = []
            distances = []
            for replication, replicate in enumerate(spectra[(n, p)]):
                values = standardize_spectrum(
                    replicate[statistic], n, p, tau_0
                )
                standardized_replicates.append(values)
                distance = wasserstein_to_semicircle(values)
                distances.append(distance)
                replicate_rows.append(
                    {
                        "date": date.today().isoformat(),
                        "master_seed": master_seed,
                        "replication": replication + 1,
                        "replication_seed": replication_seed(
                            master_seed, n, p, replication
                        ),
                        "a": exponent,
                        "n": n,
                        "M": m_edges,
                        "p": p,
                        "p_over_n": p / n,
                        "p_over_M": p / m_edges,
                        "statistic": statistic,
                        "tau_0": tau_0,
                        "wasserstein_1_to_semicircle": distance,
                        "mean": float(np.mean(values)),
                        "second_moment": float(np.mean(values**2)),
                        "third_moment": float(np.mean(values**3)),
                        "fourth_moment": float(np.mean(values**4)),
                        "support_leakage": float(np.mean(np.abs(values) > 2.0)),
                        "minimum": float(np.min(values)),
                        "maximum": float(np.max(values)),
                    }
                )

            pooled = np.concatenate(standardized_replicates)
            summary_rows.append(
                {
                    "date": date.today().isoformat(),
                    "master_seed": master_seed,
                    "replications": len(standardized_replicates),
                    "a": exponent,
                    "n": n,
                    "M": m_edges,
                    "p": p,
                    "p_over_n": p / n,
                    "p_over_M": p / m_edges,
                    "statistic": statistic,
                    "tau_0": tau_0,
                    "mean_wasserstein_1": float(np.mean(distances)),
                    "sd_wasserstein_1": float(np.std(distances, ddof=1)),
                    "q05_wasserstein_1": quantile(distances, 0.05),
                    "median_wasserstein_1": quantile(distances, 0.50),
                    "q95_wasserstein_1": quantile(distances, 0.95),
                    "pooled_wasserstein_1": wasserstein_to_semicircle(pooled),
                    "pooled_mean": float(np.mean(pooled)),
                    "pooled_second_moment": float(np.mean(pooled**2)),
                    "pooled_third_moment": float(np.mean(pooled**3)),
                    "pooled_fourth_moment": float(np.mean(pooled**4)),
                    "pooled_support_leakage": float(np.mean(np.abs(pooled) > 2.0)),
                    "pooled_minimum": float(np.min(pooled)),
                    "pooled_maximum": float(np.max(pooled)),
                }
            )
    return replicate_rows, summary_rows


def save_density_grid(
    output_dir: Path,
    configurations: list[tuple[int, int]],
    spectra: dict[tuple[int, int], list[dict[str, np.ndarray]]],
    replications: int,
    exponent: float,
) -> tuple[Path, Path]:
    mpl_config = output_dir / ".mplconfig"
    cache_dir = output_dir / ".cache"
    mpl_config.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots(
        len(STATISTICS),
        len(configurations),
        figsize=(4.8 * len(configurations), 10.0),
        sharex=True,
        sharey=True,
        constrained_layout=False,
        squeeze=False,
    )
    pooled: dict[tuple[tuple[int, int], str], np.ndarray] = {}
    for n, p in configurations:
        for statistic, tau_0 in STATISTICS.items():
            pooled[((n, p), statistic)] = np.concatenate(
                [
                    standardize_spectrum(rep[statistic], n, p, tau_0)
                    for rep in spectra[(n, p)]
                ]
            )

    empirical_low = min(float(np.min(values)) for values in pooled.values())
    empirical_high = max(float(np.max(values)) for values in pooled.values())
    low = min(-2.35, empirical_low - 0.08)
    high = max(2.35, empirical_high + 0.08)
    bins = np.linspace(low, high, 58 if replications == 1 else 76)
    x = np.linspace(low, high, 2_000)

    for row_index, statistic in enumerate(STATISTICS):
        for column_index, (n, p) in enumerate(configurations):
            axis = axes[row_index, column_index]
            values = pooled[((n, p), statistic)]
            axis.hist(
                values,
                bins=bins,
                density=True,
                color="#83B7C9",
                edgecolor="white",
                linewidth=0.35,
                alpha=0.84,
            )
            axis.plot(x, semicircle_density(x), color="#B53A32", linewidth=2.2)
            axis.axvline(-2.0, color="#777777", linewidth=0.7, alpha=0.45)
            axis.axvline(2.0, color="#777777", linewidth=0.7, alpha=0.45)
            axis.grid(color="#C9C9C9", alpha=0.30, linewidth=0.6)
            axis.tick_params(labelsize=10)
            if row_index == 0:
                m_edges = n * (n - 1) // 2
                axis.set_title(
                    rf"$n={n},\ p={p}$"
                    + "\n"
                    + rf"$a={exponent:.1f},\ p/M={p/m_edges:.3f}$",
                    fontsize=11,
                    pad=7,
                )
            if column_index == 0:
                axis.set_ylabel(
                    f"{STATISTIC_LABELS[statistic]}\nDensity", fontsize=12
                )
            if row_index == len(STATISTICS) - 1:
                axis.set_xlabel("eigenvalue", fontsize=11)
            panel_index = row_index * len(configurations) + column_index
            axis.text(
                0.025,
                0.96,
                f"({chr(ord('a') + panel_index)})",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=11,
            )

    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="#B53A32",
                linewidth=2.2,
                label=r"standard semicircle law $\mu_{\mathrm{sc}}$",
            )
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        frameon=False,
        fontsize=12,
    )
    figure.subplots_adjust(
        left=0.17 if len(configurations) == 1 else 0.075,
        right=0.985 if len(configurations) == 1 else 0.995,
        bottom=0.065,
        top=0.900,
        wspace=0.035 if len(configurations) > 1 else 0.0,
        hspace=0.070,
    )
    tag = exponent_tag(exponent)
    png_path = output_dir / f"semicircle_density_a{tag}_R{replications}.png"
    pdf_path = output_dir / f"semicircle_density_a{tag}_R{replications}.pdf"
    figure.savefig(png_path, dpi=260, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def simulate_task(
    output_dir: str,
    n: int,
    p: int,
    replication: int,
    master_seed: int,
    rscript: str,
    r_lib: str,
    force: bool,
) -> tuple[int, int, int, dict[str, np.ndarray], str, float]:
    seed = replication_seed(master_seed, n, p, replication)
    path = cache_path(Path(output_dir), n, p, replication)
    replicate = None if force else load_spectra(path, n, p, seed)
    if replicate is not None:
        return n, p, replication, replicate, "cache", 0.0
    replicate, backend_seconds = simulate_full_spectra(
        n, p, seed, rscript, Path(r_lib)
    )
    save_spectra(path, replicate, n, p, seed)
    return n, p, replication, replicate, "computed", backend_seconds


def peak_rss_megabytes() -> float:
    usages = [
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    ]
    scale = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return float(max(usages) / scale)


def write_metadata(
    path: Path,
    args: argparse.Namespace,
    configurations: list[tuple[int, int]],
    elapsed_seconds: float,
) -> None:
    metadata = {
        "date": date.today().isoformat(),
        "experiment": "intermediate_semicircle",
        "a": args.exponent,
        "configurations": [
            {
                "n": n,
                "M": n * (n - 1) // 2,
                "p": p,
                "p_over_n": p / n,
                "p_over_M": p / (n * (n - 1) // 2),
            }
            for n, p in configurations
        ],
        "dimension_rule": "p=round(n^a)",
        "replications": args.replications,
        "master_seed": args.seed,
        "seed_rule": "master_seed + 10000000*replication + 10000*n + p",
        "workers": args.workers,
        "elapsed_seconds": elapsed_seconds,
        "peak_rss_mb": peak_rss_megabytes(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "rscript": args.rscript,
        "r_library": str(args.r_lib),
        "fast_backend_package": "independence-1.0.1",
        "wasserstein_method": (
            "midpoint quantile rule: mean_i |lambda_(i)-"
            "F_sc^{-1}((i-1/2)/p)|"
        ),
        "command": " ".join(sys.argv),
    }
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.replications < 1:
        raise ValueError("replications must be positive")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    configurations = (
        list(dict.fromkeys(args.configurations))
        if args.configurations is not None
        else default_configurations(args.sample_sizes, args.exponent)
    )
    if not configurations:
        raise ValueError("at least one configuration is required")
    if not args.r_lib.exists():
        raise FileNotFoundError(f"R library does not exist: {args.r_lib}")
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            ROOT
            / "results"
            / f"theorem2_semicircle_a{exponent_tag(args.exponent)}_R{args.replications}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_validation:
        errors = validate_backend(args.rscript, args.r_lib)
        print(f"validation maximum errors: {errors}", flush=True)

    spectra_by_replication: dict[
        tuple[int, int], dict[int, dict[str, np.ndarray]]
    ] = {
        configuration: {} for configuration in configurations
    }
    total = len(configurations) * args.replications
    completed = 0
    started = time.perf_counter()
    task_arguments = [
        (
            str(output_dir),
            n,
            p,
            replication,
            args.seed,
            args.rscript,
            str(args.r_lib),
            args.force,
        )
        for n, p in configurations
        for replication in range(args.replications)
    ]
    if args.workers == 1:
        results = (simulate_task(*task) for task in task_arguments)
        for n, p, replication, replicate, source, backend_seconds in results:
            spectra_by_replication[(n, p)][replication] = replicate
            completed += 1
            detail = (
                source
                if source == "cache"
                else f"{source} in {backend_seconds:.2f}s"
            )
            print(
                f"[{completed}/{total}] n={n}, p={p}, "
                f"rep={replication + 1}, source={detail}, "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(simulate_task, *task) for task in task_arguments
            ]
            for future in as_completed(futures):
                n, p, replication, replicate, source, backend_seconds = future.result()
                spectra_by_replication[(n, p)][replication] = replicate
                completed += 1
                detail = (
                    source
                    if source == "cache"
                    else f"{source} in {backend_seconds:.2f}s"
                )
                print(
                    f"[{completed}/{total}] n={n}, p={p}, "
                    f"rep={replication + 1}, source={detail}, "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )

    spectra = {
        configuration: [
            spectra_by_replication[configuration][replication]
            for replication in range(args.replications)
        ]
        for configuration in configurations
    }

    replicate_rows, summary_rows = compute_metrics(
        configurations, spectra, args.seed, args.exponent
    )
    replicate_path = output_dir / "replicate_metrics.csv"
    summary_path = output_dir / "summary_metrics.csv"
    write_csv(replicate_path, replicate_rows)
    write_csv(summary_path, summary_rows)
    png_path, pdf_path = save_density_grid(
        output_dir,
        configurations,
        spectra,
        args.replications,
        args.exponent,
    )
    elapsed_seconds = time.perf_counter() - started
    metadata_path = output_dir / "run_metadata.json"
    write_metadata(metadata_path, args, configurations, elapsed_seconds)
    print(f"wrote {png_path}", flush=True)
    print(f"wrote {pdf_path}", flush=True)
    print(f"wrote {replicate_path}", flush=True)
    print(f"wrote {summary_path}", flush=True)
    print(f"wrote {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
