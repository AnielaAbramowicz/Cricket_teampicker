import unittest
import pandas as pd
import numpy as np
from data_helper import SimDataHelper

class TestCalculateSemiCompressedTaus(unittest.TestCase):

    def setUp(self):
        self.helper = SimDataHelper()

    def test_all_ones_semi(self):
        wicket_transition_factors = np.ones((20, 10, 8))
        overs_transition_factors = np.ones((20, 10, 8))
        helper = SimDataHelper()
        compressed_taus = helper._calculate_taus_semi_compressed(overs_transition_factors, wicket_transition_factors)
        self.assertTrue(np.allclose(compressed_taus, np.ones((20, 10, 8))))

    def test_all_zeros_semi(self):
        wicket_transition_factors = np.zeros((20, 10, 8))
        overs_transition_factors = np.zeros((20, 10, 8))
        helper = SimDataHelper()
        compressed_taus = helper._calculate_taus_semi_compressed(overs_transition_factors, wicket_transition_factors)
        expected = np.zeros((20, 10, 8))
        expected[6, 0, :] = 1
        self.assertTrue(np.allclose(compressed_taus, expected))

    def test_ones_and_twos(self):
        over_transition_factors = np.ones((20, 10, 8))
        wicket_transition_factors = np.ones((20, 10, 8))
        helper = SimDataHelper()
        for i in range(20):
            for j in range(10):
                for k in range(8):
                    over_transition_factors[i][j][k] = 2
        taus = helper._calculate_taus_semi_compressed(wicket_transition_factors, over_transition_factors)
        self.assertEqual(taus[8, 3, 0], 8.0)

    def test_random_data_semi(self):
        np.random.seed(42)
        wicket_transition_factors = np.random.rand(20, 10, 8)
        overs_transition_factors = np.random.rand(20, 10, 8)
        compressed_taus = self.helper._calculate_taus_semi_compressed(overs_transition_factors, wicket_transition_factors)
        self.assertEqual(compressed_taus.shape, (20, 10, 8))
        self.assertFalse(np.isnan(compressed_taus).any())

    def test_matrix_of_ones_with_one_value_changed_semi(self):
        over_transition_factors = np.ones((20, 10, 8))
        wicket_transition_factors = np.ones((20, 10, 8))
        over_transition_factors[2, 2, 0] = 2
        wicket_transition_factors[2, 2, 0] = 2
        result = self.helper._calculate_taus_semi_compressed(over_transition_factors, wicket_transition_factors)
        expected_0_3 = 0.875
        actual = result[0][3][0]
        self.assertAlmostEqual(actual, expected_0_3, places=5)

    def test_one_row_changed_alfa_semi(self):
        over_transition_factors = np.ones((20, 10, 8))
        wicket_transition_factors = np.ones((20, 10, 8))
        for i in range(20):
            over_transition_factors[i][0][0] = 0
        result = self.helper._calculate_taus_semi_compressed(over_transition_factors, wicket_transition_factors)
        self.assertEqual(result[7][0][0], 0.0)
        self.assertEqual(result[7][1][0], 0.0)
        self.assertEqual(result[7][5][0], 0.8)
        self.assertEqual(result[6][0][0], 1.0)
        self.assertEqual(result[6][2][0], 1.0)
        self.assertEqual(result[6][6][0], 1.0)
        self.assertAlmostEqual(result[8][0][0], 0.0, places = 5)
        self.assertAlmostEqual(result[8][3][0], 0.44444444, places = 5)
        self.assertAlmostEqual(result[8][7][0], 0.734693, places = 5)

    def test_one_row_changed_both_semi(self):
        over_transition_factors = np.ones((20, 10, 8))
        wicket_transition_factors = np.ones((20, 10, 8))
        for i in range(20):
            over_transition_factors[i][0][0] = 0
            wicket_transition_factors[i][0][0] = 0
        result = self.helper._calculate_taus_semi_compressed(over_transition_factors, wicket_transition_factors)
        for i in range(20):
            for j in range(10):
                if i!=6 and j!=0:
                    self.assertEqual(result[i][j][0], 0.0)

if __name__ == '__main__':
    unittest.main()
