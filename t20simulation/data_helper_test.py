#this file contains the test cases for the data_helper.py file

import unittest
import pandas as pd
import numpy as np
from data_helper import SimDataHelper

class TestCalculateCompressedTaus(unittest.TestCase):

    def test_all_ones(self):
        wicket_transition_factors = np.ones((20, 10, 8))
        overs_transition_factors = np.ones((20, 10, 8))
        helper = SimDataHelper()
        compressed_taus = helper.__calculate_taus_compressed(overs_transition_factors, wicket_transition_factors)
        self.assertTrue(np.allclose(compressed_taus, np.ones((20, 10, 8))))

    def test_zeros_and_ones(self):
        wicket_transition_factors = np.zeros((20, 10, 8))
        overs_transition_factors = np.ones((20, 10, 8))
        helper = SimDataHelper()
        compressed_taus = helper.__calculate_taus_compressed(overs_transition_factors, wicket_transition_factors)
        self.assertTrue(np.allclose(compressed_taus, np.zeros((20, 10, 8))))

    if __name__ == '__main__':
        unittest.main()