from __future__ import annotations

from functools import lru_cache
from itertools import combinations, permutations

import numpy as np


ORDERS = {
    "hoeffding_d": 5,
    "bkr_r": 6,
    "bdy_tau_star": 4,
}

KERNEL_DIVISORS = {
    "hoeffding_d": 16.0,
    "bkr_r": 32.0,
    "bdy_tau_star": 16.0,
}


@lru_cache(maxsize=None)
def _permutations(order: int) -> tuple[tuple[int, ...], ...]:
    return tuple(permutations(range(order)))


@lru_cache(maxsize=None)
def _combinations(n: int, order: int) -> tuple[tuple[int, ...], ...]:
    return tuple(combinations(range(n), order))


def _threshold_product(values: np.ndarray, perm: tuple[int, ...], threshold: int) -> np.ndarray:
    first = (values[perm[0]] <= values[perm[threshold]]).astype(float)
    second = (values[perm[1]] <= values[perm[threshold]]).astype(float)
    third = (values[perm[2]] <= values[perm[threshold]]).astype(float)
    fourth = (values[perm[3]] <= values[perm[threshold]]).astype(float)
    return (first - second) * (third - fourth)


def _tau_star_score(values: np.ndarray, perm: tuple[int, ...]) -> np.ndarray:
    def separated(a: int, b: int, c: int, d: int) -> np.ndarray:
        return (
            np.maximum(values[perm[a]], values[perm[b]])
            < np.minimum(values[perm[c]], values[perm[d]])
        ).astype(float)

    return (
        separated(0, 2, 1, 3)
        + separated(1, 3, 0, 2)
        - separated(0, 3, 1, 2)
        - separated(1, 2, 0, 3)
    )


def full_rank_matrix(uniforms: np.ndarray, statistic: str) -> np.ndarray:
    """Exact small-n U-statistic matrix using the paper's symmetrized kernel."""
    if statistic not in ORDERS:
        raise ValueError(f"unknown statistic: {statistic}")
    if uniforms.ndim != 2:
        raise ValueError("uniforms must be an n by p matrix")
    n, p = uniforms.shape
    order = ORDERS[statistic]
    if n < order:
        raise ValueError(f"{statistic} requires n >= {order}")

    result = np.zeros((p, p), dtype=float)
    subsets = _combinations(n, order)
    perms = _permutations(order)
    divisor = KERNEL_DIVISORS[statistic] * len(subsets)

    for subset in subsets:
        values = uniforms[np.asarray(subset), :]
        for perm in perms:
            if statistic == "hoeffding_d":
                score = _threshold_product(values, perm, 4)
                result += np.outer(score, score)
            elif statistic == "bkr_r":
                score_x = _threshold_product(values, perm, 4)
                score_y = _threshold_product(values, perm, 5)
                result += np.outer(score_x, score_y)
            else:
                score = _tau_star_score(values, perm)
                result += np.outer(score, score)

    result /= divisor
    result = 0.5 * (result + result.T)
    np.fill_diagonal(result, 1.0)
    return result
