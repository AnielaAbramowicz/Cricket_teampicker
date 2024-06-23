import numpy as np

from parameter_estimation import ParameterSampler
from data_helper import SimDataHelper

class MatchSimulator:

    extra_prob = 0.033

    # Random
    rng : np.random.Generator = np.random.default_rng()

    def __init__(self, team1 : np.ndarray[int], team2: np.ndarray[int]):
        """
        This function initializes the MatchSimulator object.

        param team1: The first team playing the match, represented as an array of batting ids
        param team2: The second team playing the match, reprersented as an array of batting ids

        Both team arrays need to be of length 11.
        """

        assert len(team1) == 11, f'Team 1 has to contain exactly 11 players, but it contains {len(team1)}'
        assert len(team2) == 11, f'Team 2 has to contain exactly 11 players, but it contains {len(team2)}'

        self.team1 = team1
        self.team2 = team2

        self.team1_runs = 0
        self.team1_wickets = 0

        self.team2_runs = 0
        self.team2_wickets = 0

        self.helper = SimDataHelper
        print('Initializing...')
        self.helper.initialize()
        print('Done.')

    def simulate_inning_1(self, team1_is_batting):
        wickets = 0
        ball_number = 1
        runs = 0

        if team1_is_batting:
            batting_order = self.team1
        else:
            batting_order = self.team2

        on_strike = batting_order[0]
        off_strike = batting_order[1]

        while wickets < 10 and ball_number < 120:
            over = (ball_number - 1) // 6

            # Extra
            if self.rng.random() < self.extra_prob:
                runs += 1
                continue

            # Get the probabilities of each outcome
            p = self.helper.get_batting_probabilities(on_strike, over, wickets)

            # Get the outcome
            outcome = self.rng.choice(8, p=p)

            # Check if the outcome is a wicket
            if outcome == 7:
                wickets += 1
                on_strike = batting_order[wickets+1] # New batter on strike
            else:
                score += outcome

            # If it is the last ball of the over, change who is on strike
            if ball_number % 6  == 0:
                on_strike, off_strike = off_strike, on_strike

        # Store the number of runs,wickets the batting team ended up with
        if team1_is_batting:
            self.team1_runs = runs
            self.team1_wickets = wickets
        else:
            self.team2_runs = runs
            self.team2_wickets = wickets

    def calculate_over_needed_aggresivness(self):
        #we calculate the over with needed aggressiveness for the team batting second for the current state of the game
        #assumes that the team batting second is the team with the lower score
        #returns the over number needed for the team batting second to win the match
        ratio = (self.team1_runs - self.team2_runs + 1) / self.get_duckworth_lewis_table_value(self.current_over, self.team2_wickets)
        for over in range(self.current_over, 20):
            if ratio < self.__expected_runs(over) / self.__expected_resource_loss(over):
                return over
        return 20

    def simulate_inning_2():

        pass

    def get_duckworth_lewis_table_value(self, overs : int, wickets : int) -> float:

        pass

    def __expected_runs(self, over : int) -> float:
        pass

    def __expected_resource_loss(self, over : int) -> float:
        pass