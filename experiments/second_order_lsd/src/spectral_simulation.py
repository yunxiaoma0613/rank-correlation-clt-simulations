from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


STATISTICS = {
    "hoeffding_d": 1.0 / 3.0,
    "bkr_r": 1.0,
    "bdy_tau_star": 3.0 / 5.0,
}


PROPORTIONAL_SEMICIRCLE_RADIUS_SQUARED = {
    "hoeffding_d": 8.0 / 9.0,
    "bkr_r": 8.0,
    "bdy_tau_star": 72.0 / 25.0,
}


def hoeffding_d_factor(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """One-coordinate factor g_D from Dong--Han--Yao Proposition 2.1."""
    return (np.sqrt(3.0) / 6.0) * (
        3.0 * x * x + 3.0 * y * y - 6.0 * np.maximum(x, y) + 2.0
    )


def normalized_edge_features(
    n: int, p: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the M by p normalized canonical edge-feature matrix."""
    if n < 2:
        raise ValueError("n must be at least 2")
    if p < 1:
        raise ValueError("p must be positive")

    uniforms = rng.random((n, p))
    return edge_features_from_uniforms(uniforms), uniforms


def edge_features_from_uniforms(uniforms: np.ndarray) -> np.ndarray:
    """Build normalized canonical edge features from a fixed uniform array."""
    if uniforms.ndim != 2:
        raise ValueError("uniforms must be an n by p matrix")
    n, p = uniforms.shape
    if n < 2 or p < 1:
        raise ValueError("uniforms must have n>=2 and p>=1")
    left, right = np.triu_indices(n, k=1)
    g = hoeffding_d_factor(uniforms[left, :], uniforms[right, :])
    sigma_d = np.sqrt(1.0 / 30.0)
    return g / sigma_d


@dataclass(frozen=True)
class GramSpectrum:
    n: int
    p: int
    m_edges: int
    rho_actual: float
    eigen_s: np.ndarray
    eigen_d0_s: np.ndarray
    diagonal_s: np.ndarray


def compute_gram_spectrum(features: np.ndarray, n: int) -> GramSpectrum:
    """Compute spectra of S=A^T A/M and its diagonal-removed version."""
    if features.ndim != 2:
        raise ValueError("features must be a matrix")
    m_edges, p = features.shape
    expected_m = n * (n - 1) // 2
    if m_edges != expected_m:
        raise ValueError(f"expected {expected_m} edge rows, found {m_edges}")

    gram = (features.T @ features) / float(m_edges)
    diagonal = np.diag(gram).copy()
    d0_gram = gram.copy()
    np.fill_diagonal(d0_gram, 0.0)
    return GramSpectrum(
        n=n,
        p=p,
        m_edges=m_edges,
        rho_actual=p / float(m_edges),
        eigen_s=np.linalg.eigvalsh(gram),
        eigen_d0_s=np.linalg.eigvalsh(d0_gram),
        diagonal_s=diagonal,
    )


@dataclass(frozen=True)
class MPGrid:
    rho: float
    atom_mass: float
    support_low: float
    support_high: float
    x: np.ndarray
    cdf: np.ndarray


def mp_grid(rho: float, grid_size: int = 40_001) -> MPGrid:
    """Numerical CDF grid for MP_rho, including its zero atom."""
    if rho <= 0:
        raise ValueError("rho must be positive")
    if grid_size < 101:
        raise ValueError("grid_size must be at least 101")

    theta = np.linspace(0.0, np.pi, grid_size)
    sqrt_rho = np.sqrt(rho)
    x = 1.0 + rho - 2.0 * sqrt_rho * np.cos(theta)
    denominator = np.pi * x
    theta_density = np.zeros_like(theta)
    np.divide(
        2.0 * np.sin(theta) ** 2,
        denominator,
        out=theta_density,
        where=denominator > 1e-15,
    )
    if abs(rho - 1.0) < 1e-12:
        theta_density[0] = 2.0 / np.pi

    increments = 0.5 * (theta_density[:-1] + theta_density[1:]) * np.diff(theta)
    continuous_cdf = np.concatenate(([0.0], np.cumsum(increments)))
    continuous_mass = min(1.0, 1.0 / rho)
    continuous_cdf *= continuous_mass / continuous_cdf[-1]
    atom_mass = max(1.0 - 1.0 / rho, 0.0)
    return MPGrid(
        rho=rho,
        atom_mass=atom_mass,
        support_low=(1.0 - sqrt_rho) ** 2,
        support_high=(1.0 + sqrt_rho) ** 2,
        x=x,
        cdf=atom_mass + continuous_cdf,
    )


def mp_cdf(values: np.ndarray, grid: MPGrid) -> np.ndarray:
    """Evaluate the MP CDF, including the jump at zero."""
    values = np.asarray(values, dtype=float)
    result = np.zeros_like(values)
    nonnegative = values >= 0.0
    result[nonnegative] = grid.atom_mass
    inside = (values >= grid.support_low) & (values < grid.support_high)
    if np.any(inside):
        result[inside] = np.interp(values[inside], grid.x, grid.cdf)
    result[values >= grid.support_high] = 1.0
    return result


def mp_quantiles(size: int, grid: MPGrid) -> np.ndarray:
    """Deterministic midpoint quantiles of the MP law."""
    if size < 1:
        raise ValueError("size must be positive")
    probabilities = (np.arange(size, dtype=float) + 0.5) / size
    quantiles = np.zeros(size, dtype=float)
    continuous = probabilities > grid.atom_mass
    if np.any(continuous):
        quantiles[continuous] = np.interp(
            probabilities[continuous], grid.cdf, grid.x
        )
    return quantiles


def affine_cdf(values: np.ndarray, tau: float, grid: MPGrid) -> np.ndarray:
    transformed = (np.asarray(values) - (1.0 - tau)) / tau
    return mp_cdf(transformed, grid)


def affine_quantiles(size: int, tau: float, grid: MPGrid) -> np.ndarray:
    return 1.0 - tau + tau * mp_quantiles(size, grid)


def affine_density(values: np.ndarray, tau: float, rho: float) -> np.ndarray:
    """Continuous density of 1-tau+tau*MP_rho; excludes the atom."""
    values = np.asarray(values, dtype=float)
    x = (values - (1.0 - tau)) / tau
    low = (1.0 - np.sqrt(rho)) ** 2
    high = (1.0 + np.sqrt(rho)) ** 2
    density = np.zeros_like(values)
    inside = (x > low) & (x < high) & (x > 0.0)
    if np.any(inside):
        density[inside] = (
            np.sqrt((high - x[inside]) * (x[inside] - low))
            / (2.0 * np.pi * rho * x[inside] * tau)
        )
    return density


def proportional_semicircle_radius(statistic: str, gamma: float) -> float:
    """Radius in DHY's law for sqrt(n) times the centered rank matrix."""
    if statistic not in PROPORTIONAL_SEMICIRCLE_RADIUS_SQUARED:
        raise ValueError(f"unknown statistic: {statistic}")
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    return float(
        np.sqrt(PROPORTIONAL_SEMICIRCLE_RADIUS_SQUARED[statistic] * gamma)
    )


def proportional_semicircle_density(
    values: np.ndarray, n: int, statistic: str, gamma: float
) -> np.ndarray:
    """DHY proportional-regime semicircle density on the raw R-matrix scale."""
    if n < 1:
        raise ValueError("n must be positive")
    radius = proportional_semicircle_radius(statistic, gamma)
    centered = np.sqrt(float(n)) * (np.asarray(values, dtype=float) - 1.0)
    density = np.zeros_like(centered)
    inside = np.abs(centered) < radius
    density[inside] = (
        np.sqrt(float(n))
        * 2.0
        * np.sqrt(radius * radius - centered[inside] * centered[inside])
        / (np.pi * radius * radius)
    )
    return density


def proportional_semicircle_quantiles(
    size: int,
    n: int,
    statistic: str,
    gamma: float,
    grid_size: int = 80_001,
) -> np.ndarray:
    """Midpoint quantiles of DHY's semicircle law on the raw R scale."""
    if size < 1:
        raise ValueError("size must be positive")
    if n < 1:
        raise ValueError("n must be positive")
    if grid_size < 1_001:
        raise ValueError("grid_size must be at least 1001")

    radius = proportional_semicircle_radius(statistic, gamma)
    scaled = np.linspace(-radius, radius, grid_size)
    density = 2.0 * np.sqrt(np.maximum(radius * radius - scaled * scaled, 0.0)) / (
        np.pi * radius * radius
    )
    increments = 0.5 * (density[:-1] + density[1:]) * np.diff(scaled)
    cdf = np.concatenate(([0.0], np.cumsum(increments)))
    cdf /= cdf[-1]
    probabilities = (np.arange(size, dtype=float) + 0.5) / size
    scaled_quantiles = np.interp(probabilities, cdf, scaled)
    return 1.0 + scaled_quantiles / np.sqrt(float(n))


def kolmogorov_distance(eigenvalues: np.ndarray, cdf: Callable[[np.ndarray], np.ndarray]) -> float:
    ordered = np.sort(np.asarray(eigenvalues, dtype=float))
    n = ordered.size
    theoretical = cdf(ordered)
    upper = np.arange(1, n + 1, dtype=float) / n
    lower = np.arange(n, dtype=float) / n
    return float(max(np.max(upper - theoretical), np.max(theoretical - lower)))


def wasserstein_quantile_distance(eigenvalues: np.ndarray, quantiles: np.ndarray) -> float:
    ordered = np.sort(np.asarray(eigenvalues, dtype=float))
    if ordered.shape != quantiles.shape:
        raise ValueError("eigenvalues and quantiles must have the same size")
    return float(np.mean(np.abs(ordered - quantiles)))


def statistic_eigenvalues(spectrum: GramSpectrum, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Return eigenvalues of R^(1) and the affine Gram baseline B."""
    leading = 1.0 + tau * spectrum.eigen_d0_s
    affine_gram = 1.0 - tau + tau * spectrum.eigen_s
    return leading, affine_gram


def metric_row(
    eigenvalues: np.ndarray,
    baseline_eigenvalues: np.ndarray,
    spectrum: GramSpectrum,
    tau: float,
    rho_target: float,
    matrix_name: str,
) -> dict[str, float | int | str]:
    grid = mp_grid(spectrum.rho_actual)
    quantiles = affine_quantiles(spectrum.p, tau, grid)
    cdf = lambda values: affine_cdf(values, tau, grid)
    shift = 1.0 - tau
    support_low = shift + tau * grid.support_low
    support_high = shift + tau * grid.support_high
    atom_window = max(0.02, 0.05 * tau)

    allowed_by_bulk = (eigenvalues >= support_low - atom_window) & (
        eigenvalues <= support_high + atom_window
    )
    allowed_by_atom = np.abs(eigenvalues - shift) <= atom_window
    empirical_variance = float(np.var(eigenvalues))
    theoretical_variance = tau * tau * spectrum.rho_actual
    return {
        "n": spectrum.n,
        "M": spectrum.m_edges,
        "p": spectrum.p,
        "rho_target": rho_target,
        "rho_actual": spectrum.rho_actual,
        "tau": tau,
        "matrix": matrix_name,
        "wasserstein_1": wasserstein_quantile_distance(eigenvalues, quantiles),
        "kolmogorov": kolmogorov_distance(eigenvalues, cdf),
        "mean": float(np.mean(eigenvalues)),
        "mean_abs_error": abs(float(np.mean(eigenvalues)) - 1.0),
        "variance": empirical_variance,
        "variance_abs_error": abs(empirical_variance - theoretical_variance),
        "diagonal_frobenius": tau
        * tau
        * float(np.mean((spectrum.diagonal_s - 1.0) ** 2)),
        "leading_vs_affine_w1": float(
            np.mean(np.abs(np.sort(eigenvalues) - np.sort(baseline_eigenvalues)))
        ),
        "support_leakage": float(np.mean(~(allowed_by_bulk | allowed_by_atom))),
        "atom_mass_empirical": float(np.mean(np.abs(eigenvalues - shift) <= atom_window)),
        "atom_mass_theory": grid.atom_mass,
        "minimum_eigenvalue": float(np.min(eigenvalues)),
        "maximum_eigenvalue": float(np.max(eigenvalues)),
        "support_low": support_low,
        "support_high": support_high,
    }
