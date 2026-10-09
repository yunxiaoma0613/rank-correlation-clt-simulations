from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))

from src.spectral_simulation import (  # noqa: E402
    STATISTICS,
    affine_quantiles,
    compute_gram_spectrum,
    hoeffding_d_factor,
    mp_grid,
    normalized_edge_features,
    proportional_semicircle_density,
    proportional_semicircle_radius,
    statistic_eigenvalues,
)
from src.full_rank_statistics import full_rank_matrix  # noqa: E402
from run_full_density_grid import (  # noqa: E402
    DEFAULT_CONFIGURATIONS,
    actual_ratio,
)
from run_replicated_full_density_grid import replication_seed  # noqa: E402


class SpectralSimulationTests(unittest.TestCase):
    def test_kernel_is_symmetric_and_canonical_numerically(self) -> None:
        grid = (np.arange(20_000, dtype=float) + 0.5) / 20_000
        for x in (0.1, 0.35, 0.8):
            forward = hoeffding_d_factor(np.full_like(grid, x), grid)
            backward = hoeffding_d_factor(grid, np.full_like(grid, x))
            self.assertLess(abs(float(np.mean(forward))), 2e-5)
            np.testing.assert_allclose(forward, backward, atol=1e-14)

    def test_kernel_variance(self) -> None:
        rng = np.random.default_rng(710)
        x = rng.random(400_000)
        y = rng.random(400_000)
        variance = float(np.mean(hoeffding_d_factor(x, y) ** 2))
        self.assertLess(abs(variance - 1.0 / 30.0), 3e-4)

    def test_feature_dimensions_and_leading_diagonal(self) -> None:
        rng = np.random.default_rng(91)
        features, _ = normalized_edge_features(8, 30, rng)
        self.assertEqual(features.shape, (28, 30))
        spectrum = compute_gram_spectrum(features, 8)
        self.assertEqual(spectrum.eigen_s.shape, (30,))
        leading, _ = statistic_eigenvalues(spectrum, STATISTICS["hoeffding_d"])
        self.assertEqual(leading.shape, (30,))
        gram = features.T @ features / 28.0
        r1 = np.eye(30) + STATISTICS["hoeffding_d"] * (
            gram - np.diag(np.diag(gram))
        )
        np.testing.assert_allclose(np.diag(r1), 1.0, atol=1e-14)
        np.testing.assert_allclose(np.linalg.eigvalsh(r1), leading, atol=1e-10)

    def test_mp_quantiles_encode_zero_atom(self) -> None:
        grid = mp_grid(2.0)
        quantiles = affine_quantiles(2_000, 1.0, grid)
        zero_fraction = float(np.mean(quantiles == 0.0))
        self.assertLess(abs(zero_fraction - 0.5), 0.002)

    def test_named_tau_constants(self) -> None:
        self.assertAlmostEqual(STATISTICS["hoeffding_d"], 1.0 / 3.0)
        self.assertAlmostEqual(STATISTICS["bkr_r"], 1.0)
        self.assertAlmostEqual(STATISTICS["bdy_tau_star"], 3.0 / 5.0)

    def test_selected_dimension_grid_has_the_requested_ratios(self) -> None:
        ratios = [actual_ratio(n, p) for n, p in DEFAULT_CONFIGURATIONS]
        expected = [600.0 / 1225.0, 1000.0 / 1770.0, 2000.0 / 2415.0]
        np.testing.assert_allclose(ratios, expected, atol=1e-14)

    def test_replication_seeds_are_distinct(self) -> None:
        seeds = {replication_seed(20260718, 90, 1000, r) for r in range(30)}
        self.assertEqual(len(seeds), 30)

    def test_proportional_semicircle_constants_and_density(self) -> None:
        gamma = 1.7
        self.assertAlmostEqual(
            proportional_semicircle_radius("hoeffding_d", gamma) ** 2,
            8.0 * gamma / 9.0,
        )
        self.assertAlmostEqual(
            proportional_semicircle_radius("bkr_r", gamma) ** 2,
            8.0 * gamma,
        )
        self.assertAlmostEqual(
            proportional_semicircle_radius("bdy_tau_star", gamma) ** 2,
            72.0 * gamma / 25.0,
        )
        n = 90
        radius = proportional_semicircle_radius("hoeffding_d", gamma)
        half_width = radius / np.sqrt(n)
        x = np.linspace(1.0 - half_width, 1.0 + half_width, 100_001)
        density = proportional_semicircle_density(x, n, "hoeffding_d", gamma)
        self.assertLess(abs(float(np.trapz(density, x)) - 1.0), 2e-6)

    def test_full_small_n_matrices_are_real_symmetric_with_unit_diagonal(self) -> None:
        rng = np.random.default_rng(2026)
        uniforms = rng.random((7, 8))
        for statistic in STATISTICS:
            matrix = full_rank_matrix(uniforms, statistic)
            np.testing.assert_allclose(matrix, matrix.T, atol=1e-14)
            np.testing.assert_allclose(np.diag(matrix), 1.0, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
