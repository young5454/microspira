import unittest

import numpy as np

from microspira.config import SupportConfig
from microspira.support import ADEQUATE, NOT_TESTABLE, assess_support


class SupportTests(unittest.TestCase):
    def test_concentrated_counts_are_not_testable(self):
        result = assess_support(np.array([100, 0, 0, 0]))
        self.assertEqual(result.support_tier, NOT_TESTABLE)
        self.assertFalse(result.testable)

    def test_distributed_counts_are_testable(self):
        config = SupportConfig(min_total_reads=10, min_occupied_units=3)
        result = assess_support(np.repeat(8, 10), config)
        self.assertEqual(result.support_tier, ADEQUATE)
        self.assertGreater(result.effective_units, 9.9)


if __name__ == "__main__":
    unittest.main()
