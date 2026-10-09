from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import date
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from statistics import NormalDist


ROOT = Path(__file__).resolve().parent
R_SCRIPT = ROOT / "full_rank_test_simulation.R"
THEORY_DIR = ROOT.parent / "polynomial_lss_clt_validation"
DEFAULT_R_LIB = ROOT.parent / "second_order_lsd" / ".r_lib"
sys.path.insert(0, str(THEORY_DIR))

from theory import DISPLAY_NAMES, POWERS, STATISTICS, limit_parameters  # noqa: E402


SCENARIOS = (
    ("I", "null"),
    ("II", "null"),
    ("III", "null"),
    ("I", "toeplitz"),
    ("II", "toeplitz"),
    ("III", "toeplitz"),
    ("II", "nonlinear"),
    ("III", "nonlinear"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Size and power of polynomial LSS tests based on complete rank matrices."
    )
    parser.add_argument("--n", type=int, nargs="+", default=[50, 70, 100])
    parser.add_argument("--q", type=float, nargs="+", default=[0.25])
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--statistic",
        choices=("all", "hoeffding_d", "bkr_r", "bdy_tau_star"),
        default="all",
    )
    parser.add_argument("--signal-scale", type=float, default=1.0)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--r-lib", type=Path, default=DEFAULT_R_LIB)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "main")
    parser.add_argument(
        "--scenario",
        action="append",
        help="restrict runs to DISTRIBUTION:DEPENDENCE, for example I:null",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def selected_scenarios(values: list[str] | None) -> tuple[tuple[str, str], ...]:
    if not values:
        return SCENARIOS
    selected = []
    for value in values:
        try:
            distribution, dependence = value.split(":", maxsplit=1)
        except ValueError as exc:
            raise ValueError(f"invalid scenario {value!r}") from exc
        scenario = (distribution, dependence)
        if scenario not in SCENARIOS:
            raise ValueError(f"unsupported scenario {value!r}")
        selected.append(scenario)
    return tuple(selected)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def raw_path(output_dir: Path, n: int, p: int, distribution: str, dependence: str) -> Path:
    return output_dir / "raw" / f"model_{distribution}_{dependence}_n{n}_p{p}.csv"


def run_scenario(
    args: argparse.Namespace,
    n: int,
    p: int,
    distribution: str,
    dependence: str,
    seed: int,
) -> Path:
    output = raw_path(args.output_dir, n, p, distribution, dependence)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.force and output.exists():
        output.unlink()
    environment = os.environ.copy()
    environment["R_LIBS"] = str(args.r_lib)
    worker_threads = max(1, (os.cpu_count() or 1) // args.jobs)
    environment["OMP_NUM_THREADS"] = str(worker_threads)
    environment["OPENBLAS_NUM_THREADS"] = str(worker_threads)
    environment["VECLIB_MAXIMUM_THREADS"] = str(worker_threads)
    command = [
        args.rscript,
        str(R_SCRIPT),
        str(n),
        str(p),
        str(args.replications),
        distribution,
        dependence,
        str(seed),
        str(args.checkpoint_every),
        str(output),
        args.statistic,
        str(args.signal_scale),
    ]
    completed = subprocess.run(command, env=environment, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"R simulation failed for Model {distribution}, {dependence}, n={n}"
        )
    return output


def summarize(
    paths: list[tuple[int, int, float, str, str, Path]],
    alpha: float,
    statistics: tuple[str, ...],
) -> list[dict[str, object]]:
    critical_value = NormalDist().inv_cdf(1.0 - alpha)
    summary = []
    for n, p, q_n, distribution, dependence, path in paths:
        rows = read_csv(path)
        for statistic in statistics:
            statistic_rows = [row for row in rows if row["statistic"] == statistic]
            for power in POWERS:
                parameters = limit_parameters(statistic, power, n, p, q_n)
                rejection_count = 0
                standardized = []
                for row in statistic_rows:
                    value = float(row[f"q{power}"])
                    z_value = parameters.standardize(value)
                    standardized.append(z_value)
                    rejection_count += z_value > critical_value
                replications = len(standardized)
                rate = rejection_count / replications
                summary.append(
                    {
                        "distribution": distribution,
                        "dependence": dependence,
                        "n": n,
                        "p": p,
                        "q_n": q_n,
                        "statistic": DISPLAY_NAMES[statistic],
                        "power": power,
                        "replications": replications,
                        "rejections": rejection_count,
                        "rejection_rate": rate,
                        "mc_standard_error": math.sqrt(rate * (1.0 - rate) / replications),
                        "critical_value": critical_value,
                    }
                )
    return summary


def save_wide_tables(output_dir: Path, rows: list[dict[str, object]]) -> None:
    for dependence in ("null", "toeplitz", "nonlinear"):
        subset = [row for row in rows if row["dependence"] == dependence]
        if not subset:
            continue
        keys = sorted(
            {(str(row["distribution"]), int(row["n"]), int(row["p"]), float(row["q_n"])) for row in subset},
            key=lambda item: (item[3], item[0], item[1]),
        )
        wide = []
        for distribution, n, p, q_n in keys:
            record: dict[str, object] = {
                "distribution": distribution,
                "n": n,
                "p": p,
                "q_n": q_n,
            }
            matching = [
                row
                for row in subset
                if row["distribution"] == distribution
                and row["n"] == n
                and row["p"] == p
            ]
            for row in matching:
                short = {
                    "Hoeffding's D": "D",
                    "BKR R": "R",
                    "BDY tau*": "tau_star",
                }[str(row["statistic"])]
                record[f"{short}_{row['power']}"] = row["rejection_rate"]
            wide.append(record)
        write_csv(output_dir / f"{dependence}_rejection_rates.csv", wide)


def main() -> None:
    args = parse_args()
    if (
        any(n < 6 for n in args.n)
        or any(q <= 0 for q in args.q)
        or args.replications < 1
        or args.signal_scale <= 0
    ):
        raise ValueError("n>=6, q>0, and replications>=1 are required")
    if not 0 < args.alpha < 1:
        raise ValueError("alpha must lie in (0,1)")

    scenarios = selected_scenarios(args.scenario)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    tasks = []
    for q_index, target_q in enumerate(args.q):
        for n_index, n in enumerate(args.n):
            d_n = n * (n - 3) // 2
            p = int(round(target_q * d_n))
            if p < 2:
                raise ValueError(f"q={target_q} gives p={p} at n={n}")
            q_n = p / d_n
            for distribution, dependence in scenarios:
                scenario_index = SCENARIOS.index((distribution, dependence))
                seed = (
                    args.seed
                    + 10000000 * q_index
                    + 1000000 * n_index
                    + 100000 * scenario_index
                )
                tasks.append((n, p, q_n, distribution, dependence, seed))

    paths = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                run_scenario, args, n, p, distribution, dependence, seed
            ): (n, p, q_n, distribution, dependence)
            for n, p, q_n, distribution, dependence, seed in tasks
        }
        for future in as_completed(futures):
            n, p, q_n, distribution, dependence = futures[future]
            path = future.result()
            paths.append((n, p, q_n, distribution, dependence, path))

    statistics = STATISTICS if args.statistic == "all" else (args.statistic,)
    rows = summarize(paths, args.alpha, statistics)
    write_csv(args.output_dir / "rejection_rates_long.csv", rows)
    save_wide_tables(args.output_dir, rows)
    metadata = {
        "date": date.today().isoformat(),
        "n": args.n,
        "target_q": args.q,
        "dimension_rule": "p = round(q * (binom(n,2)-n))",
        "replications": args.replications,
        "nominal_level": args.alpha,
        "master_seed": args.seed,
        "scenario_seed_rule": (
            "master_seed + 10000000*q_index + 1000000*n_index "
            "+ 100000*scenario_index"
        ),
        "replication_seed_rule": "scenario_seed + replication - 1",
        "parallel_jobs": args.jobs,
        "scenarios": [f"{distribution}:{dependence}" for distribution, dependence in scenarios],
        "matrices": "complete symmetrized U-statistic rank correlation matrices",
        "critical_values": "Corollary 2 finite center and limiting mean/variance evaluated at q_n",
        "alternative_bandwidth": "max(1, floor(p/100))",
        "statistic": args.statistic,
        "signal_scale": args.signal_scale,
        "signal_scaling_rule": "multiply Toeplitz rho and nonlinear r1,r2,r3; retain nonlinear noise coefficient r4",
        "elapsed_seconds": time.perf_counter() - started,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "r_library": str(args.r_lib),
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
