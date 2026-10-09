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
    LIMIT_DENSITY_LABELS,
    STATISTIC_LABELS,
    actual_ratio,
    load_spectra,
    parse_configuration,
    save_spectra,
    simulate_full_spectra,
    validate_backend,
)
from src.spectral_simulation import (  # noqa: E402
    STATISTICS,
    affine_density,
    affine_quantiles,
    mp_grid,
    wasserstein_quantile_distance,
)


TARGET_Q = 0.25
DEFAULT_SAMPLE_SIZES = (50, 70, 90)
DEFAULT_CONFIGURATIONS = tuple(
    (n, int(round((n * (n - 1) // 2) * TARGET_Q)))
    for n in DEFAULT_SAMPLE_SIZES
)


def parse_args() -> argparse.Namespace:
    default_r_lib = os.environ.get("KENDALL_FAST_R_LIB", str(ROOT / ".r_lib"))
    parser = argparse.ArgumentParser(
        description="Replicated full rank-statistic ESD density grids."
    )
    parser.add_argument(
        "--configurations",
        nargs="+",
        type=parse_configuration,
        default=list(DEFAULT_CONFIGURATIONS),
    )
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--r-lib", type=Path, default=Path(default_r_lib))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "results"
        / "quadratic_fixed_q025_R100",
    )
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def replication_seed(master_seed: int, n: int, p: int, replication: int) -> int:
    if replication < 0:
        raise ValueError("replication must be nonnegative")
    return master_seed + 10_000_000 * replication + 10_000 * n + p


def cache_path(output_dir: Path, n: int, p: int, replication: int) -> Path:
    return (
        output_dir
        / "spectra"
        / f"full_spectra_n{n}_p{p}_rep{replication + 1:02d}.npz"
    )


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
) -> tuple[
    list[dict[str, int | float | str]], list[dict[str, int | float | str]]
]:
    replicate_rows: list[dict[str, int | float | str]] = []
    summary_rows: list[dict[str, int | float | str]] = []
    for n, p in configurations:
        q_actual = actual_ratio(n, p)
        grid = mp_grid(q_actual)
        for statistic, tau in STATISTICS.items():
            quantiles = affine_quantiles(p, tau, grid)
            distances = []
            means = []
            variances = []
            for replication, replicate_spectra in enumerate(spectra[(n, p)]):
                eigenvalues = replicate_spectra[statistic]
                distance = wasserstein_quantile_distance(eigenvalues, quantiles)
                distances.append(distance)
                means.append(float(np.mean(eigenvalues)))
                variances.append(float(np.var(eigenvalues)))
                replicate_rows.append(
                    {
                        "date": date.today().isoformat(),
                        "master_seed": master_seed,
                        "replication": replication + 1,
                        "replication_seed": replication_seed(
                            master_seed, n, p, replication
                        ),
                        "n": n,
                        "M": n * (n - 1) // 2,
                        "p": p,
                        "q_actual": q_actual,
                        "statistic": statistic,
                        "tau": tau,
                        "wasserstein_1_to_limit": distance,
                        "empirical_mean": means[-1],
                        "empirical_variance": variances[-1],
                    }
                )

            pooled = np.concatenate(
                [replicate_spectra[statistic] for replicate_spectra in spectra[(n, p)]]
            )
            pooled_quantiles = affine_quantiles(pooled.size, tau, grid)
            summary_rows.append(
                {
                    "date": date.today().isoformat(),
                    "master_seed": master_seed,
                    "replications": len(distances),
                    "n": n,
                    "M": n * (n - 1) // 2,
                    "p": p,
                    "q_actual": q_actual,
                    "statistic": statistic,
                    "tau": tau,
                    "mean_wasserstein_1": float(np.mean(distances)),
                    "sd_wasserstein_1": float(np.std(distances, ddof=1)),
                    "q05_wasserstein_1": quantile(distances, 0.05),
                    "median_wasserstein_1": quantile(distances, 0.50),
                    "q95_wasserstein_1": quantile(distances, 0.95),
                    "pooled_wasserstein_1": wasserstein_quantile_distance(
                        pooled, pooled_quantiles
                    ),
                    "mean_empirical_mean": float(np.mean(means)),
                    "mean_empirical_variance": float(np.mean(variances)),
                    "limit_variance": tau * tau * q_actual,
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

    figure, axes = plt.subplots(
        len(STATISTICS),
        len(configurations),
        figsize=(13.8, 10.2),
        sharex="row",
        sharey="row",
        constrained_layout=True,
    )

    for row_index, (statistic, tau) in enumerate(STATISTICS.items()):
        pooled_by_configuration = {
            configuration: np.concatenate(
                [replicate[statistic] for replicate in spectra[configuration]]
            )
            for configuration in configurations
        }
        theoretical_supports = []
        for n, p in configurations:
            grid = mp_grid(actual_ratio(n, p))
            shift = 1.0 - tau
            theoretical_supports.append(
                (shift + tau * grid.support_low, shift + tau * grid.support_high)
            )
        row_low = min(
            *(support[0] for support in theoretical_supports),
            *(float(np.min(values)) for values in pooled_by_configuration.values()),
        )
        row_high = max(
            *(support[1] for support in theoretical_supports),
            *(float(np.max(values)) for values in pooled_by_configuration.values()),
        )
        padding = 0.055 * max(row_high - row_low, 0.25)
        bins = np.linspace(row_low - padding, row_high + padding, 72)
        x = np.linspace(row_low - padding, row_high + padding, 2_000)

        for column_index, (n, p) in enumerate(configurations):
            axis = axes[row_index, column_index]
            q_actual = actual_ratio(n, p)
            axis.hist(
                pooled_by_configuration[(n, p)],
                bins=bins,
                density=True,
                color="#83B7C9",
                edgecolor="white",
                linewidth=0.35,
                alpha=0.82,
            )
            limit_line, = axis.plot(
                x,
                affine_density(x, tau, q_actual),
                color="#B53A32",
                linewidth=2.0,
                label=LIMIT_DENSITY_LABELS[statistic],
            )
            axis.grid(color="#C9C9C9", alpha=0.32, linewidth=0.6)
            axis.tick_params(labelsize=9)
            if row_index == 0:
                axis.set_title(
                    rf"$n={n},\ p={p},\ q={q_actual:.2f}$",
                    fontsize=11.5,
                    pad=7,
                )
            if column_index == 0:
                axis.set_ylabel(
                    f"{STATISTIC_LABELS[statistic]}\nDensity", fontsize=11
                )
            if row_index == len(STATISTICS) - 1:
                axis.set_xlabel("eigenvalue", fontsize=10)
            axis.legend(
                handles=[limit_line],
                loc="upper right",
                frameon=False,
                fontsize=11.5,
                handlelength=2.2,
                handletextpad=0.55,
            )
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

    png_path = output_dir / f"quadratic_density_grid_R{replications}.png"
    pdf_path = output_dir / f"quadratic_density_grid_R{replications}.pdf"
    figure.savefig(png_path, dpi=240, bbox_inches="tight")
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
    replicate_spectra = None if force else load_spectra(path, n, p, seed)
    if replicate_spectra is not None:
        return n, p, replication, replicate_spectra, "cache", 0.0
    replicate_spectra, backend_seconds = simulate_full_spectra(
        n, p, seed, rscript, Path(r_lib)
    )
    save_spectra(path, replicate_spectra, n, p, seed)
    return (
        n,
        p,
        replication,
        replicate_spectra,
        "computed",
        backend_seconds,
    )


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
        "experiment": "quadratic_fixed_aspect",
        "target_q": TARGET_Q,
        "configurations": [
            {
                "n": n,
                "M": n * (n - 1) // 2,
                "p": p,
                "q_actual": actual_ratio(n, p),
            }
            for n, p in configurations
        ],
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
            "F_limit^{-1}((i-1/2)/p)|"
        ),
        "command": " ".join(sys.argv),
    }
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configurations = list(dict.fromkeys(args.configurations))
    if len(configurations) != 3:
        raise ValueError("the 3 by 3 figure requires exactly three configurations")
    if args.replications < 2:
        raise ValueError("replications must be at least two")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if not args.r_lib.exists():
        raise FileNotFoundError(f"R library does not exist: {args.r_lib}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
            str(args.output_dir),
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
        configurations, spectra, args.seed
    )
    replicate_path = args.output_dir / "replicate_metrics.csv"
    summary_path = args.output_dir / "summary_metrics.csv"
    write_csv(replicate_path, replicate_rows)
    write_csv(summary_path, summary_rows)
    png_path, pdf_path = save_density_grid(
        args.output_dir, configurations, spectra, args.replications
    )
    elapsed_seconds = time.perf_counter() - started
    metadata_path = args.output_dir / "run_metadata.json"
    write_metadata(metadata_path, args, configurations, elapsed_seconds)
    print(f"wrote {png_path}", flush=True)
    print(f"wrote {pdf_path}", flush=True)
    print(f"wrote {replicate_path}", flush=True)
    print(f"wrote {summary_path}", flush=True)
    print(f"wrote {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
