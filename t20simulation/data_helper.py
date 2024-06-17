# This file contains the class SimDataHelper which is used to get statistics from the ball by ball data
# Import this class for use in simulation related stuff

import pandas as pd
import numpy as np
import os
import pickle
from scipy.ndimage import gaussian_filter

path = os.path.dirname(os.path.abspath(__file__))

num_batters = 0

class SimDataHelper:
    """
    Helper class to get statistics from the ball by ball data
    """

    prior_outcomes = np.array([0.400795, 0.340703, 0.065071, 0.003958, 0.097477, 0.000188, 0.041134, 0.050673])

    emperical_outcomes_by_over = np.zeros((20, 8))

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

        # Cache the result
        #self.cached_batter_outcomes[(batter, overs, wickets)] = results

        return results

    def get_wicket_transition_factors(self, recalculate=False):

        # Check if pickled
        if not recalculate and os.path.exists(os.path.join(path, 'pickle_jar/wicket_transition_factors.pkl')):
            with open(os.path.join(path, 'pickle_jar/wicket_transition_factors.pkl'), 'rb') as f:
                return pickle.load(f)

        wicket_transition_factors = np.zeros((20, 10,  8))

        # We need to calculate the transition factors for each batter
        outcome_matrix = self.create_batter_outcome_matrix()

        numerator = np.zeros((20, 9, 8))
        denominator = np.zeros((20, 9, 8))

        # Calculate the variances for each batter
        for batter in self.batter_file['batter id'].unique():
            for wicket in range(0, 9):
                # we first need the emperical probabilities of batting outcomes at this wicket and the next, for the current batter

                # these are 2d arrays of shape (10, 8)
                p_now = outcome_matrix[batter - 1, :, wicket, :] 
                p_next = outcome_matrix[batter - 1, :, wicket+1, :]

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

                numerator[:, wicket, :] += np.divide(factors, np.sqrt(v), out=np.zeros_like(factors), where=v != 0)
                denominator[:, wicket, :] += np.divide(np.ones_like(v), np.sqrt(v), out=np.zeros_like(v), where=v != 0)

        wicket_transition_factors = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)

        # Pickle
        with open(os.path.join(path, 'pickle_jar/wicket_transition_factors.pkl'), 'wb') as f:
            f.write(pickle.dumps(wicket_transition_factors))

        return wicket_transition_factors

    def smooth_outcomes(self, outcomes):
        prior_weight = 0
        n = np.sum(outcomes, axis=1) + (prior_weight * self.prior_outcomes.sum())
        return outcomes + (prior_weight * self.prior_outcomes), n

    def get_wicket_transition_factors_simple(self, recalculate=False):

        # Check if pickled
        if not recalculate and os.path.exists(os.path.join(path, 'pickle_jar/wicket_transition_factors_simple.pkl')):
            with open(os.path.join(path, 'pickle_jar/wicket_transition_factors_simple.pkl'), 'rb') as f:
                return pickle.load(f)

        wicket_transition_factors = np.zeros((20, 10,  8))

        outcome_matrix = self.create_batter_outcome_matrix(ignore_cache=False)

        # Sum the outcomes across all batters
        outcome_matrix = np.sum(outcome_matrix, axis=0)

        # Normalize the outcome matrix
        sums = np.sum(outcome_matrix, axis=2)
        outcome_matrix = np.divide(outcome_matrix, sums[:, :, np.newaxis], out=np.zeros_like(outcome_matrix), where=sums[:, :, np.newaxis] != 0)

        # Calculate the transition factors for each wicket
        for wicket in range(0, 9):
            factors = np.divide(outcome_matrix[:, wicket+1, :], outcome_matrix[:, wicket, :], out=np.full_like(outcome_matrix[:, wicket+1, :], np.nan), where=outcome_matrix[:, wicket, :] != 0)

            wicket_transition_factors[:, wicket, :] = factors

        return wicket_transition_factors

    def smooth_transition_factors(self, factors):
        smoothed = np.zeros_like(factors)

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

    def get_over_transition_factors_simple(self, recalculate=False):

        # Check if pickled
        if not recalculate and os.path.exists(os.path.join(path, 'pickle_jar/over_transition_factors_simple.pkl')):
            with open(os.path.join(path, 'pickle_jar/over_transition_factors_simple.pkl'), 'rb') as f:
                return pickle.load(f)

        over_transition_factors = np.zeros((20, 10,  8))

        outcome_matrix = self.create_batter_outcome_matrix(ignore_cache=False)

        # Sum the outcomes across all batters
        outcome_matrix = np.sum(outcome_matrix, axis=0)

        # Normalize the outcome matrix
        sums = np.sum(outcome_matrix, axis=2)
        outcome_matrix = np.divide(outcome_matrix, sums[:, :, np.newaxis], out=np.zeros_like(outcome_matrix), where=sums[:, :, np.newaxis] != 0)

        # Calculate the transition factors for each over
        for over in range(1, 20):
            factors = np.divide(outcome_matrix[over, :, :], outcome_matrix[over-1, :, :], out=np.full_like(outcome_matrix[over, :, :], np.nan), where=outcome_matrix[over-1, :, :] != 0)

            over_transition_factors[over-1, :, :] = factors

        # Smooth the transition factors
        over_transition_factors = self.smooth_transition_factors(over_transition_factors)

        return over_transition_factors

    def get_over_transition_factors(self, recalculate=False):

        # Check if pickled
        if not recalculate and os.path.exists(os.path.join(path, 'pickle_jar/over_transition_factors.pkl')):
            with open(os.path.join(path, 'pickle_jar/over_transition_factors.pkl'), 'rb') as f:
                return pickle.load(f)

        # There are not much comments here because idk whats going on tbh
        # I am just implementing the estimation of the multiplicative parameters according
        # to the appending of the paper

        over_transition_factors = np.zeros((20, 10,  8))

        # We need to calculate the transition factors for each batter

        # Small constant for regularization
        epsilon = 1e-6

        outcome_matrix = self.create_batter_outcome_matrix(ignore_cache=False)

        numerator = np.zeros((19, 10, 8))
        denominator = np.zeros((19, 10, 8))

        #average_factors = np.zeros((19, 10, 8))

        batter_ids = self.batter_file['batter id'].unique()
        batter_ids.sort()

        all_factors = np.zeros((len(batter_ids), 19, 10, 8))
        all_variances = np.zeros((len(batter_ids), 19, 10, 8))

        for batter in batter_ids:
            for over in range(1, 20):
                # we first need the emperical probabilities of batting outcomes at this over and the next, for the current batter

                # these are 2d arrays of shape (10, 8)
                p_now = outcome_matrix[batter - 1, over - 1, :, :] 
                p_next = outcome_matrix[batter - 1, over, :, :]

                p_now, n_now = self.smooth_outcomes(p_now)
                p_next, n_next = self.smooth_outcomes(p_next)

                n_now_T = n_now[:, np.newaxis]
                n_next_T = n_next[:, np.newaxis]

                p_now = np.divide(p_now, n_now_T, out=np.zeros_like(p_now), where=n_now_T != 0)
                p_next = np.divide(p_next, n_next_T, out=np.zeros_like(p_next), where=n_next_T != 0)

                factors = np.divide(p_next, p_now, out=np.zeros_like(p_next), where=p_now != 0)
                all_factors[batter-1, over-1, :, :] = factors
                #average_factors[over - 1, :, :] += factors

                # Calculate variance
                r1 = np.divide(np.ones_like(p_next) - p_next, n_next_T * p_next, out=np.full_like(p_next, np.inf), where=p_next != 0)
                r2 = np.divide(np.ones_like(p_now) - p_now, n_now_T * p_now, out=np.full_like(p_now, np.inf), where=p_now != 0)
                v = np.multiply(np.square(factors), (r1 + r2), out=np.full_like(factors, np.inf), where=(r1 != np.inf) & (r2 != np.inf))

                all_variances[batter-1, over-1, :, :] = v

                if 0 in v:
                    print(f"Batter: {batter}")
                    print(f"Over: {over}")
                    print(f"Variance: {v}")

                numerator[over - 1, :, :] += np.divide(factors, np.sqrt(v), out=np.zeros_like(factors), where=~np.isinf(v))
                
                denominator[over - 1, :, :] += np.divide(np.ones_like(v), np.sqrt(v), out=np.zeros_like(v), where=~np.isinf(v))

                pass



        # all variances is a 4d array of shape (n_batters, 19, 10, 8)
        smallest_variances = np.min(all_variances)
        # Find factor with smallest variance
        #for batter in range(len(batter_ids)):
        #    for over in range(1, 20):
        #        for wicket in range(0, 10):
        #            for run in range(0, 8):
        #                if all_variances[batter, over-1, wicket, run] == smallest_variances:
        #                    print(f"Batter: {batter_ids[batter]}")
        #                    print(f"Over: {over}")
        #                    print(f"Wicket: {wicket}")
        #                    print(f"Outcome: {run}")
        #                    print(f"Factor with smallest variance: {all_factors[batter, over-1, wicket, run]}")

        over_transition_factors = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator != 0)

        # Pickle the matrix
        with open(os.path.join(path, 'pickle_jar/over_transition_factors.pkl'), 'wb') as f:
            f.write(pickle.dumps(over_transition_factors))

        return over_transition_factors

    def scale_over_transition_factors(self, transition_factors):
        # I want to scale it so that the product of the factors on the path from 1,0 to 7,0 is 1
        # This is because the factors are multiplicative

        # Calculate the product of the factors on the path from 1,0 to 7,0
        product = np.prod(transition_factors[0:6, 0, :], axis=0)

        # Scale the factors
        for over in range(1, 20):
            for wicket in range(0, 10):
                transition_factors[over-1, wicket, :] /= product

        return transition_factors

    def get_taus(self):    
        over_transition_factors = self.get_over_transition_factors_simple(recalculate=True)
        #over_transition_factors = self.scale_over_transition_factors(over_transition_factors)
        wicket_transition_factors = self.get_wicket_transition_factors_simple()

        # Average the transition factors
        over_transition_factors = np.mean(over_transition_factors, axis=1)
        wicket_transition_factors = np.mean(wicket_transition_factors, axis=0)

        taus = np.ones((20, 10, 8))

        # Calculate the taus relative to the seventh over and zero wickets
        taus[6, 0, :] = np.ones(8)

        for over in range(0, 7):
            for wicket in range(0, 9):
                tau = np.ones(8)
                for i in range(5, over-1, -1):
                    tau /= over_transition_factors[i, :]
                for i in range(0, wicket):
                    tau *= wicket_transition_factors[i, :]
                taus[over, wicket, :] = tau


        for over in range(7, 20):
            for wicket in range(0, 9):
                tau = np.ones(8)
                for i in range(6, over):
                    tau *= over_transition_factors[i, :]
                for i in range(0, wicket):
                    tau *= wicket_transition_factors[i, :]
                taus[over, wicket, :] = tau

        #for over in range(0, 20):
        #    for wicket in range(0, 9):
        #        if over == 0 and wicket == 0:
        #            continue
        #        elif over == 0:
        #            taus[over, wicket, :] = wicket_transition_factors[over, wicket-1, :]
        #        elif wicket == 0:
        #            taus[over, wicket, :] = over_transition_factors[over-1, wicket, :]
        #        else:
        #            taus[over, wicket, :] = over_transition_factors[over-1, wicket, :] * wicket_transition_factors[over, wicket-1, :]


        #for over in range(1, 20):
        #    for wicket in range(0, 9):
        #        cur_over = 1
        #        cur_wicket = 0
        #        while(cur_over < over or cur_wicket < wicket):
        #            if cur_over < over and cur_wicket < wicket:
        #                taus[over-1,wicket,:] *= over_transition_factors[cur_over-1, cur_wicket, :]
        #                taus[over-1,wicket,:] *= wicket_transition_factors[cur_over-1, cur_wicket, :]
        #                cur_over += 1
        #                cur_wicket += 1 
        #            elif cur_over < over:
        #                taus[over-1,wicket,:] *= over_transition_factors[cur_over-1, cur_wicket, :]
        #                cur_over += 1
        #            elif cur_wicket < wicket:
        #                taus[over-1,wicket,:] *= wicket_transition_factors[cur_over-1, cur_wicket, :]
        #                cur_wicket += 1

        return taus

    def calculate_taus_compressed(self) -> np.ndarray:
        #initiate the matrix and set the baseline element to the all 1s vector
        taus = np.zeros(self.get_over_transition_factors().shape)
        taus[6, 0, :] = np.ones(8)
        #compress the over transition factors and wicket transition factors
        over_transition_factors_compressed = np.mean(self.get_over_transition_factors(), axis=1) #average along the wickets
        wicket_transition_factors_compressed = np.mean(self.get_wicket_transition_factors(), axis=0) #average along the overs

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
        
        return taus
    
    def calculate_taus(self) -> np.ndarray:
        taus = np.zeros(20,10,8)
        taus[6, 0, :] = np.ones(8)
        #populate every element in taus by calling calculate one tau with the correct arguments
        for over in range(0, 20):
            for wicket in range(0, 10):
                taus[over, wicket, :] = self.calculate_one_tau(wicket, over)
        return taus
    
        
    
    #this is most likely super inneficient but it works for now and we only calcuate the tau matrix once
    def calculate_one_tau(self, wicket : int, over : int) -> float:
        if over == 6 and wicket  == 0:
            return np.ones(8)
        over_factor = np.ones(8)
        wicket_factor = np.ones(8)
        if over > 6:
            for i in range(6, over):
                over_factor = np.multiply(over_factor, self.get_over_transition_factors()[i, wicket, :])
        else:
            for i in range(5, over-1,-1):
                over_factor = np.divide(over_factor, self.get_over_transition_factors()[i, wicket, :])
        for i in range(0,wicket):
            wicket_factor = np.multiply(wicket_factor, self.get_wicket_transition_factors()[over, i, :])
        return np.multiply(over_factor, wicket_factor)

def main():
    # Just for testing
    simdatahelper = SimDataHelper()
    #simdatahelper.get_wicket_transition_factors()
    #over_transition_factors = simdatahelper.get_over_transition_factors()
    taus = simdatahelper.get_taus()

    baselines = simdatahelper.prior_outcomes


    n_games = 20
    total_runs = 0
    outcomes = np.zeros((20, 8))

    for _ in range(n_games):

        # Simulate a match
        ball_number = 0
        wickets = 0
        score = 0

        while wickets < 10 and ball_number < 120:
            over = ball_number // 6 + 1

            # Get the probabilities of each outcome
            p = (baselines * taus[over-1, wickets, :]) / np.sum(baselines * taus[over-1, wickets, :])

            # Get the outcome
            outcome = np.random.choice(8, p=p)

            # Check if the outcome is a wicket
            if outcome == 7:
                wickets += 1
            else:
                score += outcome

            outcomes[over-1, outcome] += 1

            ball_number += 1

        total_runs += score

    print(outcomes)

    print(f"Final score: {score}/{wickets} in {ball_number//6}.{ball_number%6} overs")



if __name__ == '__main__':
    main()






