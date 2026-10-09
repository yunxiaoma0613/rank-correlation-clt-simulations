from __future__ import annotations

import math
import unittest

from theory import POWERS, STATISTICS, limit_parameters


class TheoryFormulaTests(unittest.TestCase):
    def test_standardization_inverts_the_theoretical_location_scale(self) -> None:
        for statistic in STATISTICS:
            for power in POWERS:
                parameters = limit_parameters(
                    statistic, power, n=50, p=294, q_n=294 / 1175
                )
                for z in (-2.0, 0.0, 1.5):
                    value = (
                        parameters.center
                        + parameters.mean
                        + math.sqrt(parameters.variance) * z
                    )
                    self.assertAlmostEqual(parameters.standardize(value), z)

    def test_limiting_variances_are_positive(self) -> None:
        for statistic in STATISTICS:
            for power in POWERS:
                parameters = limit_parameters(
                    statistic, power, n=50, p=294, q_n=294 / 1175
                )
                self.assertGreater(parameters.variance, 0.0)


if __name__ == "__main__":
    unittest.main()
