#this file contains the test cases for the data_helper.py file

import unittest
import pandas as pd
import numpy as np
from data_helper import SimDataHelper

class TestCalculateCompressedTaus(unittest.TestCase):

    def setUp(self):
        self.helper = SimDataHelper()

    def test_all_ones(self):
        wicket_transition_factors = np.ones((20, 10, 8))
        overs_transition_factors = np.ones((20, 10, 8))
        helper = SimDataHelper()
        compressed_taus = helper._calculate_taus_compressed(overs_transition_factors, wicket_transition_factors)
        self.assertTrue(np.allclose(compressed_taus, np.ones((20, 10, 8))))

    def test_zeros_and_ones(self):
        wicket_transition_factors = np.zeros((20, 10, 8))
        overs_transition_factors = np.ones((20, 10, 8))
        helper = SimDataHelper()
        compressed_taus = helper._calculate_taus_compressed(overs_transition_factors, wicket_transition_factors)
        self.assertTrue(np.allclose(compressed_taus, np.zeros((20, 10, 8))))

    def test_random_data(self):
        np.random.seed(42)
        wicket_transition_factors = np.random.rand(20, 10, 8)
        overs_transition_factors = np.random.rand(20, 10, 8)
        compressed_taus = self.helper._calculate_taus_compressed(overs_transition_factors, wicket_transition_factors)
        self.assertEqual(compressed_taus.shape, (20, 10, 8))
        self.assertFalse(np.isnan(compressed_taus).any())

    def test_with_nans(self):
        wicket_transition_factors = np.full((20, 10, 8), np.nan)
        overs_transition_factors = np.full((20, 10, 8), np.nan)
        compressed_taus = self.helper._calculate_taus_compressed(overs_transition_factors, wicket_transition_factors)
        self.assertTrue(np.all(compressed_taus == 0))

    def test_matrix_of_ones_with_one_value_changed(self):
        over_transition_factors = np.ones((20, 10, 8))
        wicket_transition_factors = np.ones((20, 10, 8))
        over_transition_factors[2, 2, 0] = 2
        wicket_transition_factors[2, 2, 0] = 2
        result = self.helper._calculate_taus_compressed(over_transition_factors, wicket_transition_factors)
        expected_2_2 = 1.05/1.1
        actual = result[2][2][0]
        self.assertAlmostEqual(actual, expected_2_2, places=5)

    def test_one_row_changed_alfa(self):
        over_transition_factors = np.ones((20, 10, 8))
        wicket_transition_factors = np.ones((20, 10, 8))
        for i in range(20):
            over_transition_factors[i][0][0] = 0
        result = self.helper._calculate_taus_compressed(over_transition_factors, wicket_transition_factors)
        self.assertEqual(result[7][0][0], 0.81)
        self.assertEqual(result[7][1][0], 0.81)
        self.assertEqual(result[7][5][0], 0.81)
        self.assertEqual(result[6][0][0], 0.9)
        self.assertEqual(result[6][2][0], 0.9)
        self.assertEqual(result[6][6][0], 0.9)
        self.assertAlmostEqual(result[8][0][0], 0.729, places = 5)
        self.assertAlmostEqual(result[8][3][0], 0.729, places = 5)
        self.assertAlmostEqual(result[8][7][0], 0.729, places = 5)

    def test_one_row_changed_both(self):
        over_transition_factors = np.ones((20, 10, 8))
        wicket_transition_factors = np.ones((20, 10, 8))
        for i in range(20):
            over_transition_factors[i][0][0] = 0
            wicket_transition_factors[i][0][0] = 0
        result = self.helper._calculate_taus_compressed(over_transition_factors, wicket_transition_factors)
        for i in range(20):
            for j in range(10):
                self.assertEqual(result[i][j][0], 0)

if __name__ == '__main__':
    unittest.main()

    