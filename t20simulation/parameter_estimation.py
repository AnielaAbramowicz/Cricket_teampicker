import pandas as pd
import numpy as np
import os
import pickle

class ParameterSampler():

    def __init__(self, c : int, num_iterations : int, burn_in : int, outcomes : np.ndarray, taus : np.ndarray):
        self.taus = taus
        self.outcomes = outcomes
        self.c = c
        self.a_j = None
        self.num_iterations = num_iterations
        self.burn_in = burn_in
        self.p_i70j = None

    def initialize(self):
        self.a_j = self.calculate_a_j()
        self.p_i70j = self.sample_parameters()

    def get_probability(self, batter : int, over : int, wickets : int) -> float:
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

        if self.p_i70j == None:
            self.p_i70j = self.sample_parameters()

        return self.taus[over, wickets, :] * self.p_i70j[batter, :] / np.sum(self.taus[over, wickets] * self.p_i70j[batter])

    def sample_parameters(self) -> np.ndarray:
        """
        This function samples the parameters by calling metropolis_within_gibbs for each batter.
        Returns:
            np.ndarray: the p_i70j matrix containing the probabilities of each outcome for each batter at baseline
        """

        result = np.zeros((self.outcomes.shape[0], self.outcomes.shape[3]))

        for batter in range(self.outcomes.shape[0]):
            result[batter, :] =  self.metropolis_within_gibbs(batter)

        return result

    def metropolis_within_gibbs(self, batter : int) -> float:
        """
        This function samples the parameters using the Metropolis within Gibbs algorithm for one batter.
        
        Args:
            batter (int): The batter index.
        Returns:
            np.ndarray: The mean of all sampled p's at every iteration.
        """
        samples = []
        p_initial = self.a_j / np.sum(self.a_j)
        p_current = p_initial.copy()
        for i in range(self.num_iterations):
            for j in range(self.outcomes.shape[3]):
                #get the current value of p_j that we are sampling with gibbs
                #in this loop we sample the current outcome while we fix the other values of p

                #calculate the proposed value of p_j
                p_j_proposed = np.random.dirichlet(self.calc_exponents(batter))[j]

                #calculate the probability of the current and proposed values of p_j given the other values of p
                p_current_j_replaced = p_current.copy() 
                p_current_j_replaced[j] = p_j_proposed #replace the value of p_j with the proposed value
                current_prob = self.joint_probability_one_batter(p_current, batter) #the probablity of the p values being p current
                proposal_prob = self.joint_probability_one_batter(p_current_j_replaced, batter) #the probability of the p values being p current but the value for p_j being replaced by p_j_proposed

                #calculate the acceptance probability
                alpha = min(1, np.divide(proposal_prob, current_prob, out=np.zeros_like(proposal_prob), where=current_prob!=0))

                if alpha != 0:
                    print(alpha)

                # Accept or reject the proposal
                if np.random.uniform(0, 1) < alpha:
                    #we accept
                    p_current = p_current_j_replaced

            if i > self.burn_in:
                samples.append(p_current)
        
        return np.mean(samples, axis=0)

    def calculate_a_j(self) -> np.ndarray:
        """
        Calculates the value of a_j using the outcomes and taus arrays.

        Returns:
            np.ndarray: The calculated value of a_j.
        """
        if self.a_j != None:
            return self.a_j

        numerator = np.zeros(self.outcomes.shape[3])
        denominator = 0.0

        for batter in range(self.outcomes.shape[0]):
            for over in range(self.outcomes.shape[1]):
                for wickets in range(self.outcomes.shape[2]):
                    for outcome in range(self.outcomes.shape[3]):
                        denominator += self.outcomes[batter, over, wickets, outcome] / self.taus[over, wickets, outcome]
                    numerator += self.outcomes[batter, over, wickets] / self.taus[over, wickets]
        a_j = self.c * numerator / denominator
        self.a_j = a_j
        return a_j

    def joint_probability_one_batter(self, p : np.ndarray, batter : int) -> float:
        """
        Calculates the joint probability of the probility of outcomes for a single batter.

        Args:
            p (np.ndarray): The probability of each outcome for this batter i.
            batter (int): The batter index.
        Returns:
            float: The calculated joint probability.
        """

        numerator = np.prod((p) ** self.calc_exponents(batter))
        denominator = np.prod([np.sum(self.taus[o, w,  :] * p) ** np.sum(self.outcomes[batter, o, w, :]) for o in range(self.outcomes.shape[1]) for w in range(self.outcomes.shape[2])])

        if numerator == 0:
            print("Numerator is 0")

        return numerator / denominator
    
    def joint_probability_one_batter_old(self, p : np.ndarray, batter : int) -> float:
        """
        Calculates the joint probability of the probility of outcomes for a single batter.

        Args:
            p (np.ndarray): The probability of each outcome for this batter i.
            batter (int): The batter index.
        Returns:
            float: The calculated joint probability.
        """
        upper = 1.0
        for j in range(p.shape[0]):
            upper *= p[j] ** (self.calc_exponent(batter, j))

        lower = 1.0
        # a big problem here is we have to know the sum over j of p_i_7_o_j even though we only know p_i_7_0_j
        # this is equal to the probability of the batter i getting result j on the 7th over for o wickets being down
        # we can try to obtain this directly from the data, but it is not clear how to do this
        # i continue as if it is a typo and should be p_i_7_0_j
        summed_taus = np.sum(self.taus, axis=2)
        p_factor = np.sum(p)
        summed_taus *= p_factor
        for over in range(self.outcomes.shape[1]):
            for wicket in range(self.outcomes.shape[2]):
                lower *= summed_taus[over, wicket]**self.outcomes[batter,over, wicket]
        return upper / lower
    

    def calc_exponent(self, batter : int, outcome : int) -> float:
        """
        Helper function that calculates the exponent of probablities in the joint probability calculation.
        It is separated out to make the code more readable and because it is used also in the proposal distribution.

        Args:
            outcome (int): the oucome
            batter (int): The batter index.
        Returns:
            float: The exponent used in the joint probability calculation for the specified batting outcome.
        """
        exp = 0.0
        for over in range(self.outcomes.shape[1]):
            for wicket in range(self.outcomes.shape[2]):
                    exp += self.outcomes[batter, over, wicket, outcome]
        exp += (self.a_j[outcome] - 1)
        return exp

    def calc_exponents(self, batter : int, with_alpha=True) -> np.array:
        """
        Helper function that calculates the exponent of probablities in the joint probability calculation.

        Args:
            batter (int): The batter index.
        Returns:
            np.array[float]: The exponents used in the joint probability calculation for each batting outcome.
        """
        exp = np.zeros(self.outcomes.shape[3])
        for over in range(self.outcomes.shape[1]):
            for wicket in range(self.outcomes.shape[2]):
                exp += self.outcomes[batter, over, wicket, :]

        if with_alpha:
            exp += (self.a_j - 1)

        if np.any(exp < 0):
            print("Negative exponent")

        return exp
    





    
    