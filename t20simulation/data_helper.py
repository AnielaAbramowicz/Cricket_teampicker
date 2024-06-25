# This file contains the class SimDataHelper which is used to get statistics from the ball by ball data
# Import this class for use in simulation related stuff

import pandas as pd
import numpy as np
import os
import pickle
import sys

from scipy.ndimage import gaussian_filter
from parameter_estimation import ParameterSampler

path = os.path.dirname(os.path.abspath(__file__))

num_batters = 0

class SimDataHelper:
    """
    Helper class to get statistics from the ball by ball data
    """

    prior_outcomes = np.array([0.400795, 0.340703, 0.065071, 0.003958, 0.097477, 0.000188, 0.041134, 0.050673])
    prior_outcomes_baseline = np.array([0.353, 0.448615, 0.069696, 0.003433, 0.074731, 0, 0.034676, 0.01])
    prior_outcomes_paper = np.array([0.3421, 0.3700, 0.0856, 0.01, 0.1212, 0, 0.0341, 0.0340])

    emperical_outcomes_by_over = np.zeros((20, 8))

    batter_outcomes_matrix = None
    batting_outcome_totals = None

    taus = None

    outcomes_is_initialized = False
    taus_is_initialized = False
    baselines_is_initialized = False

    def __init__(self):
        self.batter_file = pd.read_csv(os.path.join(path, 'batter_runs.csv'))
        self.bowler_file = pd.read_csv(os.path.join(path, 'bowler_runs.csv'))
        self.num_batters = len(self.batter_file['batter id'].unique())
        self.num_bowlers = len(self.bowler_file['bowler id'].unique())

    def initialize(self):
        """
        Loads and precalculates data into memory for quick access.
        Call this before using any other functions.
        """

        # Calculate batter outcome matrix
        self.batter_outcomes_matrix = self.__create_batter_outcome_matrix()

        # Calculate bowler outcome matrix
        self.bowler_outcome_matrix = self.__create_bowler_outcome_matrix()
        self.outcomes_is_initialized = True

        # The total number of outcomes for each batter
        self.batting_outcome_totals = np.ndarray((self.num_batters, 20, 10))
        for batter in range(1, self.num_batters+1):
            self.batting_outcome_totals[batter-1, :, :] = self.get_outcomes_for_batter(batter).sum(axis=2)

        # The taus
        self.taus = self.__calculate_taus_semi_compressed()
        self.taus_is_initialized = True

        self.batter_sampler = ParameterSampler(60, 500, 100, self.batter_outcomes_matrix, self.taus, for_batters=True)
        #print("Sampling parameters...")
        self.batter_sampler.initialize()

        self.bowler_sampler = ParameterSampler(60, 500, 100, self.bowler_outcome_matrix, self.taus, for_batters=False)
        self.bowler_sampler.initialize()

        self.baselines_is_initialized = True

        self.is_initialized = True

    def load_batter_baselines(self, batters):
        """
        Load the baselines for the batters

        Args:
            batters (list): A list of batter IDs
        """

        self.batter_sampler.load_batter_baselines(batters)

    def get_batting_probabilities_against_bowler(self, batter, bowler, over, wicket):
        """
        This function returns the probabilities associated with batting outcomes for a certain batter against a certain bowler at a game stage,
        which is the current over of play and the number of wickets that have fallen.
        
        Args:
            batter (int): The batter index.
            bowler (int): The bowler index.
            over (int): The over index.
            wickets (int): The wickets index.
            
        Returns:
            float: The probability of the outcome.
        """

        assert self.baselines_is_initialized, 'Parameter sampler has not been initialized.'

        return self.batter_sampler.get_probability(batter, over, wicket) + self.bowler_sampler.get_probability(bowler, over, wicket) - self.batter_sampler.get_probability(-1, over, wicket)

    def get_batting_probabilities(self, batter, over, wicket):
        """
        This function returns the probabilities associated with batting outcomes for a certain batter at a game stage,
        which is the current over of play and the number of wickets that have fallen.

        Args:
            batter (int): The batter index.
            over (int): The over index.
            wickets (int): The wickets index.
        Returns:
            float: The probability of the outcome.
        """

        assert self.baselines_is_initialized, 'Parameter sampler has not been initialized.'

        return self.batter_sampler.get_probability(batter, over, wicket)
        
    def get_bowling_probabilities(self, bowler, over, wicket):
        assert self.baselines_is_initialized, 'Parameter sampler has not been initialized.'

        return self.bowler_sampler.get_probability(bowler, over, wicket)


    def get_outcomes_for_batter(self, batter : int) -> np.ndarray:
        """
        Get the outcomes of a batter

        Parameters:
        batter (int): ID of the batter (>= 1)

        Returns:
        pd.DataFrame: A DataFrame containing the outcomes of the batter
        """

        assert self.outcomes_is_initialized, 'Batting outcome matrix has not been initialized.'

        return self.batter_outcomes_matrix[batter - 1]

    def get_outcomes_for_bowler(self, bowler : int) -> np.ndarray:
        """
        Get the outcomes of a bowler

        Parameters:
        bowler (int): ID of the bowler (>= 1)

        Returns:
        pd.DataFrame: A DataFrame containing the outcomes of the bowler
        """

        assert self.outcomes_is_initialized, 'Batting outcome matrix has not been initialized.'

        return self.bowler_outcome_matrix[bowler - 1]

    def get_taus(self) -> np.ndarray:
        """
        Get the taus, multiplicative parameters used to scale outcome probabilities to different game stages.

        Returns:
        np.ndarray: A matrix (20x10) containing the taus for each over,wicket game stage.
        """

        assert self.taus_is_initialized, 'Taus have not been initialized.'

        return self.taus
    
    def get_total_outcomes(self, batter : int, over : int, wicket : int) -> np.ndarray:
        """
        Returns the total number of batting outcomes that have occured for a batter in a specific game stage.
        These are the m_iow values.

        Parameters:
        batter (int): ID of the batter (>= 1)
        overs (int): The number of the over (1-20)
        wickets (int): Number of wickets fallen
        Returns:
        the number of balls batter i has faced at game stage (over, wickets)
        """

        assert self.is_initialized, 'Data has not been initialized.'

        return self.batting_outcome_totals[batter, over, wicket]

    def __create_bowler_outcome_matrix(self, normalize=False, ignore_cache=False) -> np.ndarray:
        """
        Create a matrix containing the outcome frequencies of each bowler at each game stage
        
        Parameters:
        normalize (bool): Whether to normalize the outcome frequencies so that they can be used as probabilities
        
        Returns:
        np.ndarray: A matrix containing the outcome frequencies of each bowler at each game stage
        """

        if not ignore_cache and os.path.exists(os.path.join(path, 'pickle_jar/bowler_outcome_matrix.pkl')):
            with open(os.path.join(path, 'pickle_jar/bowler_outcome_matrix.pkl'), 'rb') as f:
                return pickle.load(f)
            
        bowler_outcome_matrix = np.zeros((self.bowler_file['bowler id'].max(), 20, 10, 8))

        outcomes = self.bowler_file[self.bowler_file['innings']==1]
        outcomes = outcomes.drop(columns=['match id', 'bowler', 'date', 'extra runs', 'innings', 'ipl-it20'], errors='ignore')
        outcomes = outcomes.groupby(['bowler id', 'over', 'wickets'], as_index=False).sum()

        # Iterate through each batter
        for bowler in range(1, self.bowler_file['bowler id'].max()+1):
            print(f'Processing batter {bowler} out of {self.bowler_file["bowler id"].max()}...')
            # Iterate through each game stage
            for overs in range(1, 21):
                for wickets in range(0, 10):
                    # Get the outcome frequencies of the batter at the game stage
                    bowler_outcomes = outcomes[(outcomes['bowler id'] == bowler) & (outcomes['over'] == overs) & (outcomes['wickets'] == wickets)]

                    if len(bowler_outcomes) == 0:
                        bowler_outcome_matrix[bowler - 1, overs - 1, wickets, :] = [0] * 8
                        continue

                    # Convert to numpy array
                    results = bowler_outcomes[['0', '1', '2', '3', '4', '5', '6', 'wicket']].to_numpy(dtype=int)[0]

                    # Store the outcome frequencies in the matrix
                    bowler_outcome_matrix[bowler - 1, overs - 1, wickets, :] = results

        # pickle the matrix
        with open(os.path.join(path, 'pickle_jar/bowler_outcome_matrix.pkl'), 'wb') as f:
            pickle.dump(bowler_outcome_matrix, f)

        return bowler_outcome_matrix

    def __create_batter_outcome_matrix(self, normalize=False, ignore_cache=False) -> np.ndarray:
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

        # Prepare the data
        print("Preparing data...")
        outcomes = self.batter_file[self.batter_file['innings']==1]
        outcomes = outcomes.drop(columns=['match id', 'batter', 'date', 'extra runs', 'innings', 'ipl-it20'], errors='ignore')
        outcomes = outcomes.groupby(['batter id', 'over', 'wickets'], as_index=False).sum()
        
        # Iterate through each batter
        for batter in range(1, self.batter_file['batter id'].max()+1):
            print(f'Processing batter {batter} out of {self.batter_file["batter id"].max()}...')
            # Iterate through each game stage
            for overs in range(1, 21):
                for wickets in range(0, 10):
                    # Get the outcome frequencies of the batter at the game stage
                    batter_outcomes = outcomes[(outcomes['batter id'] == batter) & (outcomes['over'] == overs) & (outcomes['wickets'] == wickets)]

                    if len(batter_outcomes) == 0:
                        batter_outcome_matrix[batter - 1, overs - 1, wickets, :] = [0] * 8
                        continue

                    # Convert to numpy array
                    results = batter_outcomes[['0', '1', '2', '3', '4', '5', '6', 'wicket']].to_numpy(dtype=int)[0]

                    # Store the outcome frequencies in the matrix
                    batter_outcome_matrix[batter - 1, overs - 1, wickets, :] = results

        # pickle the matrix
        with open(os.path.join(path, 'pickle_jar/batter_outcome_matrix.pkl'), 'wb') as f:
            pickle.dump(batter_outcome_matrix, f)

        return batter_outcome_matrix

    def __get_batter_outcomes_at_gamestage(self, batter : int, overs : int, wickets : int, normalize=False) -> np.ndarray[int]:
        """
        Get the outcomes of a batter at a given game stage

        Parameters:
        batter (int): ID of the batter
        overs (int): The number of the over (1-20)
        wickets (int): Number of wickets fallen
        normalize (bool): Whether to normalize the outcome frequencies so that they can be used as probabilities

        Returns:
        np.ndarray: A list containing the outcome frequencies of the batter at the given game stage, where the first seven indices correspond to the number of runs scored and the last index corresponds to the number of dismissals
        """

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
        #outcomes = outcomes.groupby(['over', 'wickets'], as_index=False).sum()
        outcomes = outcomes.drop(columns=['over', 'wickets']).sum(axis=0)

        # Get the count of 1s, 2s, etc.
        # I want to sum vertically 
        # i.e. sum of 1s, sum of 2s, etc.

        # Convert to numpy array
        results = outcomes[['0', '1', '2', '3', '4', '5', '6', 'wicket']].to_numpy(dtype=int)[0]

        # Normalize the results
        if normalize:
            results = results / np.sum(results)

        return results

    def __smooth_outcomes(self, outcomes):
        prior_weight = 0
        n = np.sum(outcomes, axis=1) + (prior_weight * self.prior_outcomes_baseline.sum())
        return outcomes + (prior_weight * self.prior_outcomes_baseline), n

    def __smooth_transition_factors(self, factors, type='guassian'):
        smoothed = np.zeros_like(factors)

        if type == 'guassian':

            sigma=1

            # Create two arrays, one with nans replaced with zeros
            # and one with 1s everywhere besides nans which are 0
            # taken from https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python

            V = factors.copy()
            V[np.isnan(factors)] = 0
            W = 0 * factors.copy() + 1
            W[np.isnan(factors)] = 0

            for outcome in range(0, 8):
                V_slice = gaussian_filter(V[:, :, outcome], sigma=sigma)
                W_slice = gaussian_filter(W[:, :, outcome], sigma=sigma)
                smoothed_outcome_matrix = V_slice / W_slice
                smoothed[:, :, outcome] = smoothed_outcome_matrix

        return smoothed

    def __get_wicket_transition_factors(self, recalculate=False):

        # Check if pickled
        if not recalculate and os.path.exists(os.path.join(path, 'pickle_jar/wicket_transition_factors.pkl')):
            with open(os.path.join(path, 'pickle_jar/wicket_transition_factors.pkl'), 'rb') as f:
                return pickle.load(f)

        wicket_transition_factors = np.zeros((20, 10,  8))

        outcome_matrix = self.__create_batter_outcome_matrix(ignore_cache=False)

        # Sum the outcomes across all batters
        outcome_matrix = np.sum(outcome_matrix, axis=0)

        # Smooth the outcome matrix
        for wicket in range(0, 20):
            outcome_matrix[wicket, :, :] = self.__smooth_outcomes(outcome_matrix[wicket, :, :])[0]

        # Normalize the outcome matrix
        sums = np.sum(outcome_matrix, axis=2)
        outcome_matrix = np.divide(outcome_matrix, sums[:, :, np.newaxis], out=np.zeros_like(outcome_matrix), where=sums[:, :, np.newaxis] != 0)

        # Calculate the transition factors for each wicket
        for wicket in range(0, 9):
            factors = np.divide(outcome_matrix[:, wicket+1, :], outcome_matrix[:, wicket, :], out=np.full_like(outcome_matrix[:, wicket+1, :], np.nan), where=outcome_matrix[:, wicket, :] != 0)

            wicket_transition_factors[:, wicket, :] = factors

        # Smooth the transition factors
        #wicket_transition_factors = self.__smooth_transition_factors(wicket_transition_factors)

        # Replace all nans with 1
        wicket_transition_factors[np.isnan(wicket_transition_factors)] = 1

        # Cache the transition factors
        with open(os.path.join(path, 'pickle_jar/wicket_transition_factors.pkl'), 'wb') as f:
            f.write(pickle.dumps(wicket_transition_factors))

        return wicket_transition_factors


    def __get_over_transition_factors(self, recalculate=False):

        # Check if pickled
        if not recalculate and os.path.exists(os.path.join(path, 'pickle_jar/over_transition_factors.pkl')):
            with open(os.path.join(path, 'pickle_jar/over_transition_factors.pkl'), 'rb') as f:
                return pickle.load(f)

        over_transition_factors = np.zeros((20, 10,  8))

        outcome_matrix = self.__create_batter_outcome_matrix(ignore_cache=False)

        # Sum the outcomes across all batters for each over,wicket
        outcome_matrix = np.sum(outcome_matrix, axis=0)

        # Smooth the outcome matrix
        #for wicket in range(0, 20):
            #outcome_matrix[wicket] = self.__smooth_outcomes(outcome_matrix[wicket, :, :])[0]

        # Normalize the outcome matrix
        sums = np.sum(outcome_matrix, axis=2)
        # Divide the elements in the second axis by the sums,
        # so that we have proportions of outcomes instead of total occurances
        for i in range(0, 20):
            for j in range(0, 10):
                outcome_matrix[i, j, :] = np.divide(outcome_matrix[i, j, :], sums[i, j], out=np.zeros_like(outcome_matrix[i, j, :]), where=sums[i, j] != 0)

        # Calculate the transition factors for each over
        for over in range(1, 20):
            factors = np.divide(outcome_matrix[over, :, :], outcome_matrix[over-1, :, :], out=np.full_like(outcome_matrix[over, :, :], np.nan), where=outcome_matrix[over-1, :, :] != 0)

            over_transition_factors[over-1, :, :] = factors

        # Since some game stages have never been reached in the data, we get nans when
        # we try to calculate the transition factors, which are replaced by zeros a few lines above.
        # I am just going to replace these zeros with ones, so that when calculating the taus,
        # the taus in these positions should get their values from neighbouring game stages.
        over_transition_factors[over_transition_factors==0] = 1

        # Smooth the transition factors
        #over_transition_factors = self.__smooth_transition_factors(over_transition_factors)

        # Replace all nans with 1
        over_transition_factors[np.isnan(over_transition_factors)] = 1

        # Cache the transition factors
        with open(os.path.join(path, 'pickle_jar/over_transition_factors.pkl'), 'wb') as f:
            f.write(pickle.dumps(over_transition_factors))

        return over_transition_factors

    def __calculate_taus_compressed(self, over_transition_factors : np.ndarray = None, wicket_transition_factors : np.ndarray = None) -> np.ndarray:
        #default values for the matrices are the ones calculated by the helper,
        #for testing perposes we can pass in custom matrices
        if over_transition_factors is None:
            over_transition_factors = self.__get_over_transition_factors()
        if wicket_transition_factors is None:
            wicket_transition_factors = self.__get_wicket_transition_factors()

        #initiate the matrix and set the baseline element to the all 1s vector
        taus = np.zeros_like(over_transition_factors)
        taus[6, 0, :] = np.ones(8)
        #compress the over transition factors and wicket transition factors
        over_transition_factors_compressed = np.mean(over_transition_factors, axis=1) #average along the wickets
        wicket_transition_factors_compressed = np.mean(wicket_transition_factors, axis=0) #average along the overs

        #for all the taus they are calculated by going from the baseline element to the current element
        for over in range(7, 20):
            #multitply each over compressed transition factor with the compressed previous transition factor
            over_transition_factors_compressed[over, :] = np.multiply(over_transition_factors_compressed[over, :], over_transition_factors_compressed[over - 1, :])
        for over in range(5, -1,-1):
            #devide each over compressed transition factor with the compressed next transition factor
            over_transition_factors_compressed[over, :] = np.divide(over_transition_factors_compressed[over + 1, :], over_transition_factors_compressed[over, :])
        #now we have a matrix which stores the effects of transition in the over direction
        #we do the same for the wicket transition factors
        for wicket in range(0, 10):
            wicket_transition_factors_compressed[wicket, :] = np.multiply(wicket_transition_factors_compressed[wicket, :], wicket_transition_factors_compressed[wicket - 1, :])
        
        #now calculate the taus by multiplying the over and wicket transition factors
        for over in range(0, 20):
            for wicket in range(0, 10):
                taus[over, wicket, :] = np.multiply(over_transition_factors_compressed[over, :], wicket_transition_factors_compressed[wicket, :])

        taus[np.isnan(taus)] = 0
        taus[np.isinf(taus)] = 0
        
        return taus
    
    def __calculate_taus(self, over_transition_factors : np.ndarray = None, wicket_transition_factors : np.ndarray = None) -> np.ndarray:
        #default values for the matrices are the ones calculated by the helper,
        #for testing perposes we can pass in custom matrices
        if over_transition_factors is None:
            over_transition_factors = self.__get_over_transition_factors()
        if wicket_transition_factors is None:
            wicket_transition_factors = self.__get_wicket_transition_factors()

        taus = np.zeros((over_transition_factors.shape))
        taus[6, 0, :] = np.ones(8)
        #populate every element in taus by calling calculate one tau with the correct arguments
        for over in range(0, 20):
            for wicket in range(0, 10):
                taus[over, wicket, :] = self.__calculate_one_tau(wicket, over, over_transition_factors, wicket_transition_factors)

        taus[np.isnan(taus)] = 0 

        return taus
    
    #this is most likely super inneficient but it works for now and we only calcuate the tau matrix once
    def __calculate_one_tau(self, wicket : int, over : int, over_factors : np.ndarray, wicket_factors : np.ndarray, decay=0) -> float:
        if over == 6 and wicket  == 0:
            return np.ones(8)
        over_factor = np.ones(8)
        wicket_factor = np.ones(8)
        if over > 6:
            w = np.exp(-decay * (over - np.arange(7, over+1)))
            #w /= np.sum(w)
            temp = np.ones(8)
            for i in range(6, over):
                temp = np.multiply(temp, over_factors[i, wicket, :])
                over_factor = np.multiply(over_factor, over_factors[i, wicket, :]) ** w[i-6]
        else:
            w = np.exp(-decay * (np.arange(over, 6)))
            #w /= np.sum(w)
            for i in range(5, over-1,-1):
                over_factor = np.divide(over_factor, over_factors[i, wicket, :]) ** w[5-i]
        for i in range(0,wicket):
            w = np.exp(-decay * (wicket - i))
            #w /= np.sum(w)
            wicket_factor = np.multiply(wicket_factor, wicket_factors[over, i, :]) ** w
        return np.multiply(over_factor, wicket_factor, out=np.zeros(8), where=(over_factor != np.inf) & (wicket_factor != np.inf) & (over_factor != 0) & (wicket_factor != 0))
    
    def __calculate_taus_semi_compressed(self, over_transition_factors : np.ndarray = None, wicket_transition_factors : np.ndarray = None) -> np.ndarray:
        #default values for the matrices are the ones calculated by the helper,
        #for testing purposes we can pass in custom matrices
        if over_transition_factors is None:
            over_transition_factors = self.__get_over_transition_factors()
        if wicket_transition_factors is None:
            wicket_transition_factors = self.__get_wicket_transition_factors()

        #initiate the matrix and set the baseline element to the all 1s vector
        taus = np.zeros(over_transition_factors.shape)
        taus[6, 0] = np.ones(8)
        #this is a version where we compress only the necesarry part of the matrix instead of the whole thing
        for over in range(0, 20):
            for wicket in range(0, 10):
                taus[over, wicket, :] = self.__calculate_one_tau_compressed(wicket, over, over_transition_factors, wicket_transition_factors)

        taus[np.isnan(taus)] = 0

        return taus


    def __calculate_one_tau_compressed(self, wicket: int, over: int, over_factors: np.ndarray, wicket_factors: np.ndarray) -> np.ndarray:
        # First compress the necessary parts of the matrices
        if over == 6 and wicket == 0:
            return np.ones((1, 8))
        
        if over > 6:
            if wicket == 0:
                over_factors_new = over_factors[6 : over, :1, :]
                wicket_factors_new = wicket_factors[6 : over, :1, :]
            else:
                over_factors_new = over_factors[6 : over, 0 : wicket, :]
                wicket_factors_new = wicket_factors[6 : over, 0 : wicket, :]
        elif over < 6:
            if wicket == 0:
                over_factors_new = over_factors[over : 6, :1, :]
                wicket_factors_new = wicket_factors[over : 6, :1, :]
            else:
                over_factors_new = over_factors[over : 6, 0 : wicket, :]
                wicket_factors_new = wicket_factors[over : 6 , 0 : wicket, :]
        else:
            if wicket == 0:
                wicket_factors_new = wicket_factors[over:7, :1, :]
            else:
                wicket_factors_new = wicket_factors[over:7, 0 : wicket, :]
                
            compressed_over_factors = np.ones((1, 8))

        if over != 6:
            compressed_over_factors = np.mean(over_factors_new, axis=1)

        
        compressed_wicket_factors = np.mean(wicket_factors_new, axis=0)

        # Calculate the taus in a similar fashion to compressed taus
        over_factor = np.ones(8)
        wicket_factor = np.ones(8)
        if over > 6:
            for i in range(over - 6):
                over_factor = np.multiply(over_factor, compressed_over_factors[i])
        else:
            for i in range(6 - over):
                over_factor = np.divide(over_factor, compressed_over_factors[i])
        for i in range(wicket):
            wicket_factor = np.multiply(wicket_factor, compressed_wicket_factors[i])


        tau = np.multiply(over_factor, wicket_factor, out=np.ones_like(over_factor), where=(over_factor != np.inf) & (wicket_factor != np.inf) & (over_factor != 0) & (wicket_factor != 0))



        return tau  
    

def main():
    pass

if __name__ == '__main__':
    main()






