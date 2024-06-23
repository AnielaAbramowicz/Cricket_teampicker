import numpy as np

from parameter_estimation import parameterSampler

class MatchSimulator:

    def __init__(self, team1 : np.ndarray[int], team2: np.ndarray[int]):
        """
        This function initializes the MatchSimulator object.
        param team1: The first team playing the match. 
        param team2: The second team playing the match.
        both teams are stored as arrays of player indices indicating the batting index of the player.

        """
        self.team1 = team1
        self.team2 = team2
        self.team1_runs = 0
        self.team1_wickets = 0
        self.team2_runs = 0
        self.team2_wickets = 0
        self.current_over = 0
        self.current_batter = 0





    def simulate_inning_1():

        pass

    def calculate_over_needed_aggresivness():
        #we calculate the over with needed aggressiveness for the team batting second for the current state of the game
        #assumes that the team batting second is the team with the lower score
        #returns the over number needed for the team batting second to win the match
        ratio = (self.team1_runs - self.team2_runs + 1) / get_duckworth_lewis_table_value(self.current_over, self.team2_wickets)
        for over in range(self.current_over, 20):
            if ratio < __expected_runs(over) / __expected_resource_loss(over):
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