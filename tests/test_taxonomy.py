import unittest

import numpy as np
import pandas as pd
from scipy import sparse

from microspira.taxonomy import aggregate_counts


class TaxonomyTests(unittest.TestCase):
    def test_sparse_genus_aggregation(self):
        X = sparse.csr_matrix([[1, 2, 3], [4, 5, 6]])
        var = pd.DataFrame({"genus": ["A", "A", "B"]})
        result = aggregate_counts(X, var, rank="genus", feature_names=["s1", "s2", "s3"])
        self.assertEqual(result.names.tolist(), ["A", "B"])
        np.testing.assert_array_equal(result.matrix.toarray(), [[3, 3], [9, 6]])
        self.assertEqual(result.members["A"], ("s1", "s2"))


if __name__ == "__main__":
    unittest.main()

