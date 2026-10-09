from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "polynomial_lss_clt_validation"))

from experiments.polynomial_lss_independence_test.run_experiment import (
    SCENARIOS,
    save_wide_tables,
    selected_scenarios,
)
from theory import limit_parameters


class ExperimentConfigurationTest(unittest.TestCase):
    def test_li_scenarios(self) -> None:
        self.assertEqual(len(SCENARIOS), 8)
        self.assertIn(("I", "null"), SCENARIOS)
        self.assertIn(("III", "toeplitz"), SCENARIOS)
        self.assertNotIn(("I", "nonlinear"), SCENARIOS)

    def test_scenario_filter(self) -> None:
        self.assertEqual(selected_scenarios(["II:nonlinear"]), (("II", "nonlinear"),))

    def test_upper_tail_threshold_standardizes_to_zero_at_mean(self) -> None:
        parameters = limit_parameters("hoeffding_d", 2, 50, 29, 29 / 1175)
        value = parameters.center + parameters.mean
        self.assertAlmostEqual(parameters.standardize(value), 0.0)

    def test_wide_table_keeps_distinct_aspect_ratios(self) -> None:
        rows = []
        for p, q_n in ((352, 0.3), (822, 0.7)):
            rows.append(
                {
                    "distribution": "I",
                    "dependence": "null",
                    "n": 50,
                    "p": p,
                    "q_n": q_n,
                    "statistic": "Hoeffding's D",
                    "power": 2,
                    "rejection_rate": q_n,
                }
            )
        with TemporaryDirectory() as directory:
            output = Path(directory)
            save_wide_tables(output, rows)
            table = (output / "null_rejection_rates.csv").read_text()
        self.assertIn("352,0.3", table)
        self.assertIn("822,0.7", table)


if __name__ == "__main__":
    unittest.main()
