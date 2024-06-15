# This file contains the class SimDataHelper which is used to get statistics from the ball by ball data
# Import this class for use in simulation related stuff

import pandas as pd
import numpy as np
import os
import pickle

path = os.path.dirname(os.path.abspath(__file__))

num_batters = 0

class SimDataHelper:
    """
    Helper class to get statistics from the ball by ball data
    """

    prior_outcomes = np.array([0.194901, 0.185065, 0.031819, 0.001530, 0.055905, 0.000104, 0.024381, 0.506294])

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
        #if (batter, overs, wickets) in self.cached_batter_outcomes:
            #return self.cached_batter_outcomes[(batter, overs, wickets)]

        # Filter by batter
        outcomes = self.batter_file[self.batter_file['batter id'] == batter]

        # Only consider first innings
        outcomes = outcomes[outcomes['innings'] == 1]

        # Filter by game stage
        outcomes = outcomes[(outcomes['over'] == overs) & (outcomes['wickets'] == wickets)]

        outcomes = outcomes.drop(columns=['match id', 'batter', 'date', 'extra runs', 'innings', 'batter id'], errors='ignore')

        # Check if outcomes is empty
        # This means the batter has not faced a ball at this game stage
        if len(outcomes) == 0:
            return [0] * 8

        # Summing the columns 'wickets_in_over', '1', '2', '3', '4', '5', '6', 'extras' to "squash" the data
        outcomes = outcomes.groupby(['over', 'wickets'], as_index=False).sum()

        # Get the count of 1s, 2s, etc.
        # I want to sum vertically 
        # i.e. sum of 1s, sum of 2s, etc.

        # Convert to numpy array
        results = outcomes[['0', '1', '2', '3', '4', '5', '6', 'wicket']].to_numpy(dtype=int)[0]

        # Normalize the results
        if normalize:
            results = results / np.sum(results)

        # Cache the result
        #self.cached_batter_outcomes[(batter, overs, wickets)] = results

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

                p_now, n_now = self.smooth_outcomes(p_now)
                p_next, n_next = self.smooth_outcomes(p_next)

                n_now_T = n_now[:, np.newaxis]
                n_next_T = n_next[:, np.newaxis]

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

    def smooth_outcomes(self, outcomes):
        prior_weight = 1
        n = np.sum(outcomes, axis=1) + (prior_weight * self.prior_outcomes.sum())
        return outcomes + (prior_weight * self.prior_outcomes), n

    def get_over_transition_factors(self):

        # There are not much comments here because idk whats going on tbh
        # I am just implementing the estimation of the multiplicative parameters according
        # to the appending of the paper

        #first demensin is the overs
        #second dimension is the wickets
        #third dimension is the outcomes
        over_transition_factors = np.ndarray((20, 10,  8))

        # We need to calculate the transition factors for each batter

        outcome_matrix = self.create_batter_outcome_matrix()

        #outcome_matrix = smooth_matrix(outcome_matrix)

        variances = np.zeros((self.batter_file['batter id'].max(), 19, 9, 8))

        numerator = np.zeros((19, 9, 8))
        denominator = np.zeros((19, 9, 8))

        batter_ids = self.batter_file['batter id'].unique()
        batter_ids.sort()

        # Calculate the variances for each batter
        for batter in batter_ids:
            for over in range(1, 20):
                # we first need the emperical probabilities of batting outcomes at this over and the next, for the current batter

                # these are 2d arrays of shape (10, 8)
                p_now = outcome_matrix[batter - 1, over - 1, :-1, :] 
                p_next = outcome_matrix[batter - 1, over, :-1, :]

                # calculating the total deliveries faced by the batter at this over and the next
                # I am going to smooth the probabilities (laplace)
                #n_now = np.sum(p_now, axis=1)
                #n_now_T = n_now[:, np.newaxis]
                #n_next = np.sum(p_next, axis=1) 
                #n_next_T = n_next[:, np.newaxis]
                
                p_now, n_now = self.smooth_outcomes(p_now)
                p_next, n_next = self.smooth_outcomes(p_next)

                n_now_T = n_now[:, np.newaxis]
                n_next_T = n_next[:, np.newaxis]

                p_now = np.divide(p_now, n_now_T)
                p_next = np.divide(p_next, n_next_T)

                factors = np.divide(p_next, p_now, out=np.zeros_like(p_next), where=p_now != 0)

                # Calculate variance
                r1 = np.divide(np.ones_like(p_next) - p_next, n_next_T * p_next, out=np.full_like(p_next, np.inf), where=p_next != 0)
                r2 = np.divide(np.ones_like(p_now) - p_now, n_now_T * p_now, out=np.full_like(p_now, np.inf), where=p_now != 0)
                v = np.square(factors) * (r1 + r2)

                numerator[over - 1, :, :] += np.divide(factors, np.sqrt(v), out=np.zeros_like(factors), where=v != 0)
                denominator[over - 1, :, :] += np.divide(np.ones_like(v), np.sqrt(v), out=np.zeros_like(v), where=v != 0)

        over_transition_factors = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)

        return over_transition_factors
    

    def calculate_taus_compressed(self, over_transition_factors : np.ndarray, wicket_transition_factors : np.ndarray) -> np.ndarray:
        #initiate the matrix and set the baseline element to the all 1s vector
        taus = np.zeros(over_transition_factors.shape)
        taus[7, 0, :] = np.ones(8)
        #compress the over transition factors and wicket transition factors
        over_transition_factors_compressed = np.mean(over_transition_factors, axis=1) #average along the wickets
        wicket_transition_factors_compressed = np.mean(wicket_transition_factors, axis=0) #average along the overs

        #for all the taus they are calculated by going from the baseline element to the current element
        for over in range(8, 20):
            #multitply each over compressed transition factor with the compressed previous transition factor
            over_transition_factors_compressed[over, :] = np.multiply(over_transition_factors_compressed[over, :], over_transition_factors_compressed[over - 1, :])
        for over in range(7, 0):
            #devide each over compressed transition factor with the compressed next transition factor
            over_transition_factors_compressed[over, :] = np.divide(over_transition_factors_compressed[over, :], over_transition_factors_compressed[over + 1, :])
        #now we have a matrix which stores the effects of transition in the over direction
        #we do the same for the wicket transition factors
        for wicket in range(1, 10):
            wicket_transition_factors_compressed[wicket, :] = np.multiply(wicket_transition_factors_compressed[wicket, :], wicket_transition_factors_compressed[wicket - 1, :])
        
        #now calculate the taus by multiplying the over and wicket transition factors
        for over in range(0, 20):
            for wicket in range(0, 10):
                taus[over, wicket, :] = np.multiply(over_transition_factors_compressed[over, :], wicket_transition_factors_compressed[wicket, :])
        
        return taus
    
    def calculate_taus(self) -> np.ndarray:
        taus = np.zeros(20,10,8)
        taus[7, 0, :] = np.ones(8)
        #populate every element in taus by calling calculate one tau with the correct arguments
        
    
    #this is most likely super inneficient but it works for now and we only calcuate the tau matrix once
    def calculate_one_tau(self, wicket : int, over : int, over_transition_factors : np.ndarray, wicket_transition_factors : np.ndarray) -> float:
        over_factor = np.ones(8)
        if over > 7:
            for i in range(7, over):
                over_factor = np.multiply(over_factor, over_transition_factors[i, wicket, :])
        else:
            for i in range(7,0):
                over_factor = np.divide(over_factor, over_transition_factors[i, wicket, :])
        wicket_factor = np.ones(8)
        for i in range(0,wicket):
            wicket_factor = np.multiply(wicket_factor, wicket_transition_factors[over, i, :])
        return np.multiply(over_factor, wicket_factor)

def main():
    # Just for testing
    simdatahelper = SimDataHelper()
    #simdatahelper.get_wicket_transition_factors()
    over_transition_factors = simdatahelper.get_over_transition_factors()
    print(over_transition_factors[:, 5, 4])

if __name__ == '__main__':
    main()






