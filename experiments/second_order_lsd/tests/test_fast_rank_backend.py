from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_full_fast_rank import validate_backend  # noqa: E402


class FastRankBackendIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("KENDALL_FAST_R_LIB"),
        "set KENDALL_FAST_R_LIB to test the optional R backend",
    )
    def test_fast_matrices_match_enumerated_kernels(self) -> None:
        errors = validate_backend(
            "Rscript", Path(os.environ["KENDALL_FAST_R_LIB"])
        )
        for statistic, error in errors.items():
            with self.subTest(statistic=statistic):
                self.assertLessEqual(error, 1e-12)


if __name__ == "__main__":
    unittest.main()

