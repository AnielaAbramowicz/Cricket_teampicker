# This file contains the class SimDataHelper which is used to get statistics from the ball by ball data
# Import this class for use in simulation related stuff

import pandas as pd
import numpy as np
import os
import pickle

path = os.path.dirname(os.path.abspath(__file__))

class SimDataHelper:
    """
    Helper class to get statistics from the ball by ball data
    """

    cached_batter_outcomes = {}

    def __init__(self):
        self.batter_file = pd.read_csv(os.path.join(path, 'batter_runs.csv'))

    def create_batter_outcome_matrix(self, normalize=False, ignore_cache=False) -> np.ndarray:
        """
        Create a matrix containing the outcome frequencies of each batter at each game stage

        Parameters:
        normalize (bool): Whether to normalize the outcome frequencies so that they can be used as probabilities

        Returns:
        np.ndarray: A matrix containing the outcome frequencies of each batter at each game stage
        """

        # Check if the matrix is already pickled
        if not ignore_cache and os.path.exists(os.path.join(path, 'pickle_jar/batter_outcome_matrix.pkl')):
            with open(os.path.join(path, 'pickle_jar/batter_outcome_matrix.pkl'), 'rb') as f:
                return pickle.load(f)

        # Create an empty matrix to store the outcome frequencies
        batter_outcome_matrix = np.zeros((self.batter_file['batter id'].max(), 20, 10, 8))

        # Iterate through each batter
        for batter in range(1, self.batter_file['batter id'].max()+1):
            print(f'Processing batter {batter} out of {self.batter_file["batter id"].max()}...')
            # Iterate through each game stage
            for overs in range(1, 21):
                for wickets in range(0, 10):
                    # Get the outcome frequencies of the batter at the game stage
                    outcomes = self.get_batter_outcomes_at_gamestage(batter, overs, wickets, normalize)

                    # Store the outcome frequencies in the matrix
                    batter_outcome_matrix[batter - 1, overs - 1, wickets, :] = outcomes

        # pickle the matrix
        with open(os.path.join(path, 'pickle_jar/batter_outcome_matrix.pkl'), 'wb') as f:
            pickle.dump(batter_outcome_matrix, f)

        return batter_outcome_matrix

    def get_batter_outcomes_at_gamestage(self, batter : int, overs : int, wickets : int, normalize=False) -> np.ndarray[int]:
        """
        Get the outcomes of a batter at a given game stage

        Parameters:
        batter (int): ID of the batter
        overs (float): The number of the over (1-20)
        wickets (int): Number of wickets fallen
        normalize (bool): Whether to normalize the outcome frequencies so that they can be used as probabilities

        Returns:
        np.ndarray: A list containing the outcome frequencies of the batter at the given game stage, where the first seven indices correspond to the number of runs scored and the last index corresponds to the number of dismissals
        """

        # Check if the data is already cached
        if (batter, overs, wickets) in self.cached_batter_outcomes:
            return self.cached_batter_outcomes[(batter, overs, wickets)]

        # Filter by batter
        outcomes = self.batter_file[self.batter_file['batter id'] == batter]

        # Only consider first innings
        outcomes = outcomes[outcomes['innings'] == 1]

        # Filter by game stage
        outcomes = outcomes[(outcomes['over'] == overs) & (outcomes['wickets'] == wickets)]

        outcomes = outcomes.drop(columns=['match id', 'batter', 'date', 'extra runs', 'innings'], errors='ignore')

        # Check if outcomes is empty
        # This means the batter has not faced a ball at this game stage
        if len(outcomes) == 0:
            return [0] * 8

        # Summing the columns 'wickets_in_over', '1', '2', '3', '4', '5', '6', 'extras' to "squash" the data
        outcomes = outcomes.groupby(['over', 'batter id', 'wickets'], as_index=False).sum()

        # Get the count of 1s, 2s, etc.
        # I want to sum vertically 
        # i.e. sum of 1s, sum of 2s, etc.

        # Convert to numpy array
        results = outcomes[['0', '1', '2', '3', '4', '5', '6', 'wicket']].to_numpy(dtype=int)[0]

        # Normalize the results
        if normalize:
            results = results / np.sum(results)

        # Cache the result
        self.cached_batter_outcomes[(batter, overs, wickets)] = results

        return results

    def get_wicket_transition_factors(self):
        wicket_transition_factors = np.ndarray((20, 10,  8))

        # We need to calculate the transition factors for each batter
        outcome_matrix = self.create_batter_outcome_matrix()

        variances = np.zeros((self.batter_file['batter id'].max(), 19, 9, 8))

        numerator = np.zeros((19, 9, 8))
        denominator = np.zeros((19, 9, 8))

        # Calculate the variances for each batter
        for batter in self.batter_file['batter id'].unique():
            for wicket in range(0, 9):
                # we first need the emperical probabilities of batting outcomes at this wicket and the next, for the current batter

                # these are 2d arrays of shape (10, 8)
                p_now = outcome_matrix[batter - 1, :-1, wicket, :] 
                p_next = outcome_matrix[batter - 1, :-1, wicket+1, :]

                # calculating the total deliveries faced by the batter at this wicket and the next
                # I am going to smooth the probabilities (laplace)
                n_now = np.sum(p_now, axis=1) + 8
                n_now_T = n_now[:, np.newaxis]
                n_next = np.sum(p_next, axis=1) + 8
                n_next_T = n_next[:, np.newaxis]
                
                p_now = p_now + 1
                p_next = p_next + 1

                p_now = np.divide(p_now.astype(float), n_now_T)
                p_next = np.divide(p_next.astype(float), n_next_T)

                factors = np.divide(p_next, p_now, out=np.zeros_like(p_next), where=p_now != 0)

                # Calculate variance
                r1 = np.divide(np.ones_like(p_next) - p_next, n_next_T * p_next, out=np.zeros_like(p_next), where=(p_next != 0) & (n_next_T != 0))
                r2 = np.divide(np.ones_like(p_now) - p_now, n_now_T * p_now, out=np.zeros_like(p_now), where=(p_now != 0) & (n_now_T != 0))
                v = np.square(factors) * (r1 + r2)

                variances[batter - 1, :, wicket, :] = v

                numerator[:, wicket, :] += np.divide(factors, np.sqrt(v), out=np.zeros_like(factors), where=v != 0)
                denominator[:, wicket, :] += np.divide(np.ones_like(v), np.sqrt(v), out=np.zeros_like(v), where=v != 0)

        wicket_transition_factors = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)

        return wicket_transition_factors

    def get_over_transition_factors(self):

        # There are not much comments here because idk whats going on tbh
        # I am just implementing the estimation of the multiplicative parameters according
        # to the appending of the paper

        over_transition_factors = np.ndarray((20, 10,  8))

        # We need to calculate the transition factors for each batter

        outcome_matrix = self.create_batter_outcome_matrix()

        #outcome_matrix = smooth_matrix(outcome_matrix)

        variances = np.zeros((self.batter_file['batter id'].max(), 19, 9, 8))

        numerator = np.zeros((19, 9, 8))
        denominator = np.zeros((19, 9, 8))

        # Calculate the variances for each batter
        for batter in self.batter_file['batter id'].unique():
            for over in range(1, 20):
                # we first need the emperical probabilities of batting outcomes at this over and the next, for the current batter

                # these are 2d arrays of shape (10, 8)
                p_now = outcome_matrix[batter - 1, over - 1, :-1, :] 
                p_next = outcome_matrix[batter - 1, over, :-1, :]

                # calculating the total deliveries faced by the batter at this over and the next
                # I am going to smooth the probabilities (laplace)
                n_now = np.sum(p_now, axis=1) + 8
                n_now_T = n_now[:, np.newaxis]
                n_next = np.sum(p_next, axis=1) + 8
                n_next_T = n_next[:, np.newaxis]
                
                p_now = p_now + 1
                p_next = p_next + 1

                p_now = np.divide(p_now.astype(float), n_now_T)
                p_next = np.divide(p_next.astype(float), n_next_T)

                factors = np.divide(p_next, p_now, out=np.zeros_like(p_next), where=p_now != 0)

                # Calculate variance
                r1 = np.divide(np.ones_like(p_next) - p_next, n_next_T * p_next, out=np.zeros_like(p_next), where=(p_next != 0) & (n_next_T != 0))
                r2 = np.divide(np.ones_like(p_now) - p_now, n_now_T * p_now, out=np.zeros_like(p_now), where=(p_now != 0) & (n_now_T != 0))
                v = np.square(factors) * (r1 + r2)

                variances[batter - 1, over - 1, :, :] = v

                numerator[over - 1, :, :] += np.divide(factors, np.sqrt(v), out=np.zeros_like(factors), where=v != 0)
                denominator[over - 1, :, :] += np.divide(np.ones_like(v), np.sqrt(v), out=np.zeros_like(v), where=v != 0)

        over_transition_factors = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)

        return over_transition_factors



def main():
    # Just for testing
    simdatahelper = SimDataHelper()
    simdatahelper.get_wicket_transition_factors()

if __name__ == '__main__':
    main()






