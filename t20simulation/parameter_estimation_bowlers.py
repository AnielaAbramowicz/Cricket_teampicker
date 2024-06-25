import math
import pandas as pd
import numpy as np
import os
import pickle

class ParameterSampler():

    def __init__(self, c : int, num_iterations : int, burn_in : int, outcomes : np.ndarray, taus : np.ndarray):
        self.taus = taus
        self.outcomes = outcomes # here outcome is the bowling outcomes
        self.c = c
        self.a_j = None
        self.num_iterations = num_iterations
        self.burn_in = burn_in
        self.q_i70j = None

    def initialize(self):
        self.a_j = self.calculate_a_j()
        self.q_i70j = self.sample_parameters()

    def get_probability(self, bowler : int, over : int, wickets : int) -> float:
        """
        This function returns the probabilities associated with batting outcomes for a certain bowler at a game stage,
        which is the current over of play and the number of wickets that have fallen.

        Args:
            bowler (int): The bowler index.
            over (int): The over index.
            wickets (int): The wickets index.
        Returns:
            float: The probability of the outcome.
        """

        try:
            if self.q_i70j == None:
                self.q_i70j = self.sample_parameters()
        except ValueError:
            pass

        return self.taus[over, wickets, :] * self.q_i70j[bowler, :] / np.sum(self.taus[over, wickets] * self.q_i70j[bowler])

    def sample_parameters(self) -> np.ndarray:
        """
        This function samples the parameters by calling metropolis_within_gibbs for each bowler.
        Returns:
            np.ndarray: the q_i70j matrix containing the probabilities of each outcome for each bowler at baseline
        """

        result = np.zeros((self.outcomes.shape[0], self.outcomes.shape[3]))

        for bowler in range(self.outcomes.shape[0]):
            #result[bowler, :] =  self.metropolis_within_gibbs(bowler)
            result[bowler, :] = self.simple_baselines(bowler)

        return result

    def simple_baselines(self, bowler : int) -> np.ndarray:
        return self.calc_exponents(bowler) / np.sum(self.calc_exponents(bowler))

    def metropolis_within_gibbs(self, bowler : int) -> float:
        """
        This function samples the parameters using the Metropolis within Gibbs algorithm for one bowler.
        
        Args:
            bowler (int): The bowler index.
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
                p_j_proposed = np.random.dirichlet(self.calc_exponents(bowler))[j]

                #calculate the probability of the current and proposed values of p_j given the other values of p
                p_current_j_replaced = p_current.copy() 
                p_current_j_replaced[j] = p_j_proposed #replace the value of p_j with the proposed value

                #the code section below was numerically impossible to compute so we use a different approach
                #current_prob = self.joint_probability_one_bowler(p_current, bowler) #the probablity of the p values being p current
                #proposal_prob = self.joint_probability_one_bowler(p_current_j_replaced, bowler) #the probability of the p values being p current but the value for p_j being replaced by p_j_proposed

                exponent = self.calc_exponent(bowler, j)
                denominator_new = self.calc_denominator_joint_probability(p_current_j_replaced, bowler)
                denominator_old = self.calc_denominator_joint_probability(p_current, bowler)
            
                needed_exponent = self.calculate_x(p_current[j], p_j_proposed, denominator_old, denominator_new)
                if needed_exponent < 0:
                    #reject the proposal
                    continue

                #calculate the acceptance probability
                #alpha = min(1, np.divide(proposal_prob, current_prob, out=np.zeros_like(proposal_prob), where=current_prob!=0))

                #we ac
                if exponent > needed_exponent and p_current[j] > p_j_proposed:
                    alpha = 1
                elif exponent < needed_exponent and p_current[j] < p_j_proposed:
                    alpha = 1
                else:
                    alpha = ((p_j_proposed**exponent)/denominator_new)/((p_current[j]**exponent)/denominator_old)


                # Accept or reject the proposal
                if np.random.uniform(0, 1) < alpha:
                    #we accept
                    p_current = p_current_j_replaced

            if i > self.burn_in:
                samples.append(p_current)
        
        return np.mean(samples, axis=0)

    def calculate_x(self, p1, p2, a, b):
        """
        Calculate the value of x for the equation (p1^x) / a = (p2^x) / b.

        Parameters:
        p1 (float): Base of the exponent on the left side.
        p2 (float): Base of the exponent on the right side.
        a (float): Coefficient on the left side.
        b (float): Coefficient on the right side.

        Returns:
        float: The value of x.
        """
        # Calculate the natural logarithms of p1 and p2
        log_p1 = math.log(p1)
        log_p2 = math.log(p2)
        
        # Calculate the ratio b/a
        ratio = b / a
        
        # Calculate the natural logarithm of the ratio
        log_ratio = math.log(ratio)
        
        # Calculate x using the formula derived
        if log_p1 == log_p2:
            raise ValueError("Logarithms of p1 and p2 are equal, which would result in division by zero.")
        else:
            x = log_ratio / (log_p2 - log_p1)
        
        return x

    def joint_probability_one_bowler(self, p : np.ndarray, bowler : int) -> float:
        """
        Calculates the joint probability of the probility of outcomes for a single bowler.

        Args:
            p (np.ndarray): The probability of each outcome for this bowler i.
            bowler (int): The bowler index.
        Returns:
            float: The calculated joint probability.
        """

        numerator = np.prod((p) ** self.calc_exponents(bowler))
        denominator = np.prod([np.sum(self.taus[o, w,  :] * p) ** np.sum(self.outcomes[bowler, o, w, :]) for o in range(self.outcomes.shape[1]) for w in range(self.outcomes.shape[2])])

        if numerator == 0:
            print("Numerator is 0")

        return numerator / denominator
    
    def calc_denominator_joint_probability(self, p : np.ndarray, bowler : int) -> float:
        """
        Calculates the denominator of the joint probability of the probility of outcomes for a single bowler.

        Args:
            p (np.ndarray): The probability of each outcome for this bowler i.
            bowler (int): The bowler index.
        Returns:
            float: The calculated denominator of the joint probability.
        """
        denominator = np.prod([np.sum(self.taus[o, w,  :] * p) ** np.sum(self.outcomes[bowler, o, w, :]) for o in range(self.outcomes.shape[1]) for w in range(self.outcomes.shape[2])])
        return denominator

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

        for bowler in range(self.outcomes.shape[0]):
            for over in range(self.outcomes.shape[1]):
                for wickets in range(self.outcomes.shape[2]):
                    for outcome in range(self.outcomes.shape[3]):
                        denominator += self.outcomes[bowler, over, wickets, outcome] / self.taus[over, wickets, outcome]
                    numerator += self.outcomes[bowler, over, wickets] / self.taus[over, wickets]
        a_j = self.c * numerator / denominator
        self.a_j = a_j
        return a_j

    
    def joint_probability_one_bowler_old(self, p : np.ndarray, bowler : int) -> float:
        """
        Calculates the joint probability of the probility of outcomes for a single bowler.

        Args:
            p (np.ndarray): The probability of each outcome for this bowler i.
            bowler (int): The bowler index.
        Returns:
            float: The calculated joint probability.
        """
        upper = 1.0
        for j in range(p.shape[0]):
            upper *= p[j] ** (self.calc_exponent(bowler, j))

        lower = 1.0
        # a big problem here is we have to know the sum over j of p_i_7_o_j even though we only know p_i_7_0_j
        # this is equal to the probability of the bowler i getting result j on the 7th over for o wickets being down
        # we can try to obtain this directly from the data, but it is not clear how to do this
        # i continue as if it is a typo and should be p_i_7_0_j
        summed_taus = np.sum(self.taus, axis=2)
        p_factor = np.sum(p)
        summed_taus *= p_factor
        for over in range(self.outcomes.shape[1]):
            for wicket in range(self.outcomes.shape[2]):
                lower *= summed_taus[over, wicket]**self.outcomes[bowler,over, wicket]
        return upper / lower
    

    def calc_exponent(self, bowler : int, outcome : int) -> float:
        """
        Helper function that calculates the exponent of probablities in the joint probability calculation.
        It is separated out to make the code more readable and because it is used also in the proposal distribution.

        Args:
            outcome (int): the oucome
            bowler (int): The bowler index.
        Returns:
            float: The exponent used in the joint probability calculation for the specified batting outcome.
        """
        exp = 0.0
        for over in range(self.outcomes.shape[1]):
            for wicket in range(self.outcomes.shape[2]):
                    exp += self.outcomes[bowler, over, wicket, outcome]
        exp += (self.a_j[outcome] - 1)
        return exp

    def calc_exponents(self, bowler : int, with_alpha=True) -> np.array:
        """
        Helper function that calculates the exponent of probablities in the joint probability calculation.

        Args:
            bowler (int): The bowler index.
        Returns:
            np.array[float]: The exponents used in the joint probability calculation for each batting outcome.
        """
        exp = np.zeros(self.outcomes.shape[3])
        for over in range(self.outcomes.shape[1]):
            for wicket in range(self.outcomes.shape[2]):
                exp += self.outcomes[bowler, over, wicket, :]

        if with_alpha:
            #exp += (self.a_j - 1)
            exp += (self.a_j)

        if np.any(exp < 0):
            print("Negative exponent")

        return exp
    