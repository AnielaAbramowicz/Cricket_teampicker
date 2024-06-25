import numpy as np
import pandas as pd
import os

from parameter_estimation import ParameterSampler
from data_helper import SimDataHelper

class MatchSimulator:

    # Path of this file
    path = os.path.dirname(os.path.realpath(__file__))

    extra_prob = 0.033

    # Random
    rng : np.random.Generator = np.random.default_rng()

    def __init__(self, team1 : np.ndarray[int], team2: np.ndarray[int], team1_bowling_lineup : np.ndarray[int] = np.array([-1,-1,-1,-1,-1]), team2_bowling_lineup : np.ndarray[int] = np.array([-1,-1,-1,-1,-1])):
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

        self.current_batter = None # i think we need to have this aswell

        self.team1_bowling_lineup = team1_bowling_lineup
        self.team2_bowling_lineup = team2_bowling_lineup

        self.helper = SimDataHelper()
        self.duckwort_lewis_table = pd.read_csv(os.path.join(self.path, 'duckworth_lewis.csv')) #replace with actual path 
        print('Initializing...')
        self.helper.initialize()
        print('Done.')

    def reset_with_new_teams(self, team1 : np.ndarray[int], team2: np.ndarray[int], team1_bowling_lineup : np.ndarray[int] = np.array([-1,-1,-1,-1,-1]), team2_bowling_lineup : np.ndarray[int] = np.array([-1,-1,-1,-1,-1])):
        """
        This function resets the match simulator with new teams.

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

        self.current_batter = None

        self.team1_bowling_lineup = team1_bowling_lineup
        self.team2_bowling_lineup = team2_bowling_lineup

    def simulate_inning_1(self, team1_is_batting : bool):
        wickets = 0
        ball_number = 1
        runs = 0

        run_history = np.array([0]*120) # These keep track of the runs and wickets at each ball
        wicket_history = np.array([0]*120)

        bowler_counter = np.array([0]*len(self.team1_bowling_lineup))
        bowlers = self.team1_bowling_lineup.copy()

        if team1_is_batting:
            batting_order = self.team1
        else:
            batting_order = self.team2

        on_strike = batting_order[0]
        off_strike = batting_order[1]
        cur_bowler_index = self.rng.integers(0, len(bowlers))
        current_bowler = self.team1_bowling_lineup[cur_bowler_index]

        while wickets < 10 and ball_number < 120:
            over = (ball_number - 1) // 6

            # Extra
            if self.rng.random() < self.extra_prob:
                runs += 1
                run_history[ball_number-1] += 1
                continue

            # Get the probabilities of each outcome
            p = self.helper.get_batting_probabilities_against_bowler(on_strike, current_bowler, over, wickets)

            # Get the outcome
            outcome = self.rng.choice(8, p=p)

            # Check if the outcome is a wicket
            if outcome == 7:
                wickets += 1
                wicket_history[ball_number-1] = 1
                if wickets == 10:
                    break
                on_strike = batting_order[wickets+1] # New batter on strike
            else:
                runs += outcome
                run_history[ball_number-1] += outcome

            # If it is the last ball of the over, change who is on strike and change the bowler
            if ball_number % 6  == 0:
                on_strike, off_strike = off_strike, on_strike
                bowler_counter[cur_bowler_index] += 1
                if bowler_counter[cur_bowler_index] == 4:
                    bowlers = np.delete(bowlers, cur_bowler_index)
                    bowler_counter = np.delete(bowler_counter, cur_bowler_index)
                cur_bowler_index = self.rng.integers(0, len(bowlers))
                current_bowler = self.team1_bowling_lineup[cur_bowler_index]

            ball_number += 1

        # Store the number of runs,wickets the batting team ended up with
        if team1_is_batting:
            self.team1_runs = runs
            self.team1_wickets = wickets
        else:
            self.team2_runs = runs
            self.team2_wickets = wickets

        return runs,wickets,ball_number,run_history,wicket_history

    def simulate_inning_2(self, team2_is_batting, target_score):
        wickets = 0
        ball_number = 1
        runs = 0

        run_history = np.array([0]*120) # These keep track of the runs and wickets at each ball
        wicket_history = np.array([0]*120)

        bowler_counter = np.array([0]*len(self.team2_bowling_lineup))
        bowlers = self.team2_bowling_lineup.copy()

        if team2_is_batting:
            batting_order = self.team2
        else:
            batting_order = self.team1

        on_strike = batting_order[0]
        off_strike = batting_order[1]
        cur_bowler_index = self.rng.integers(0, len(bowlers))
        current_bowler = self.team1_bowling_lineup[cur_bowler_index]

        while wickets < 10 and ball_number < 120 and runs < target_score:
            over = (ball_number - 1) // 6

            # Extra
            if self.rng.random() < self.extra_prob:
                runs += 1
                run_history[ball_number-1] += 1
                continue

            # Get the probabilities of each outcome
            if over == 20:
                pass
            characteristic_over = self.calculate_over_needed_aggresivness(target_score, runs, over, wickets, on_strike)
            p = self.helper.get_batting_probabilities_against_bowler(on_strike, current_bowler, characteristic_over, wickets)

            # Get the outcome
            outcome = self.rng.choice(8, p=p)

            # Check if the outcome is a wicket
            if outcome == 7:
                wickets += 1
                wicket_history[ball_number-1] = 1
                if wickets == 10:
                    break
                on_strike = batting_order[wickets+1] # New batter on strike
            else:
                runs += outcome
                run_history[ball_number-1] += outcome

            # If it is the last ball of the over, change who is on strike
            if ball_number % 6  == 0:
                on_strike, off_strike = off_strike, on_strike
                bowler_counter[cur_bowler_index] += 1
                if bowler_counter[cur_bowler_index] == 4:
                    bowlers = np.delete(bowlers, cur_bowler_index)
                    bowler_counter = np.delete(bowler_counter, cur_bowler_index)
                cur_bowler_index = self.rng.integers(0, len(bowlers))
                current_bowler = self.team1_bowling_lineup[cur_bowler_index]

            ball_number += 1

        # Store the number of runs,wickets the batting team ended up with
        if team2_is_batting:
            self.team2_runs = runs
            self.team2_wickets = wickets
        else:
            self.team1_runs = runs
            self.team1_wickets = wickets

        return runs,wickets,ball_number,run_history,wicket_history

    def calculate_over_needed_aggresivness(self, target_runs : int, current_runs : int, current_over : int, current_wickets : int, current_batter : int):
        #we calculate the over with needed aggressiveness for the team batting second for the current state of the game
        #assumes that the team batting second is the team with the lower score
        #returns the over number needed for the team batting second to win the match
        ratio = 0.6 * (target_runs - current_runs + 1) / self.get_duckworth_lewis_table_value(current_over, current_wickets)
        for over in range(current_over, 19):
            if ratio < self.__expected_runs(over, current_batter, current_wickets) / self.__expected_resource_loss(over, current_batter, current_wickets):
                return over
        return 19

    def get_duckworth_lewis_table_value(self, overs : int, wickets : int) -> float:
        return self.duckwort_lewis_table.iloc[overs, wickets]

    def __expected_runs(self, over: int, current_batter, wickets) -> float:
        batting_probabilities = self.helper.get_batting_probabilities(current_batter, over, wickets)
        outcomes = np.arange(1, 7)
        expected_runs = np.sum(outcomes * batting_probabilities[1:7])
        return expected_runs


    def __expected_resource_loss(self, over : int, current_batter, wickets) -> float:
        batting_probabilities = self.helper.get_batting_probabilities(current_batter, over, wickets)
        x = self.get_duckworth_lewis_table_value(over, wickets) - self.get_duckworth_lewis_table_value(over + 1, wickets)
        y = self.get_duckworth_lewis_table_value(over, wickets) - self.get_duckworth_lewis_table_value(over, wickets + 1)
        expected_loss = x*np.sum(batting_probabilities[0:7]) + y*batting_probabilities[7]
        return expected_loss
        

def get_run_distribution(team1, team2, num_iterations):
    """
    This function simulates a match between team1 and team2 for num_iterations and returns the total number of runs scored in the first innings of all the games.
    """

    runs = []

    for i in range(num_iterations):
        sim = MatchSimulator(team1, team2)

        target = sim.simulate_inning_1(True)[0]

        runs.append(target)

    return np.array(runs)

def run_standard_match_experiment():
    team1 = np.full(11, -1)
    team2 = np.full(11, -1)

    n = 1
    total_run_history_team1 = np.zeros(120)
    total_wicket_history_team1 = np.zeros(120)
    total_run_history_team2 = np.zeros(120)
    total_wicket_history_team2 = np.zeros(120)

    for i in range(n):
        sim = MatchSimulator(team1, team2)
        target,team1_wickets,team1_deliveries,run_history_team1,wicket_history_team1 = sim.simulate_inning_1(True)
        chase,team2_wickets,team2_deliveries,run_history_team2,wicket_history_team2 = sim.simulate_inning_2(True, target)

        total_run_history_team1 += run_history_team1
        total_wicket_history_team1 += wicket_history_team1
        total_run_history_team2 += run_history_team2
        total_wicket_history_team2 += wicket_history_team2

    run_history_team1 = total_run_history_team1 / n
    wicket_history_team1 = total_wicket_history_team1 / n
    run_history_team2 = total_run_history_team2 / n
    wicket_history_team2 = total_wicket_history_team2 / n

    # Save to csv
    results = pd.DataFrame(columns=['ball', 'team1_runs', 'team1_wickets', 'team2_runs', 'team2_wickets'])
    results['ball'] = np.arange(1, 121)
    results['team1_runs'] = run_history_team1
    results['team1_wickets'] = wicket_history_team1
    results['team2_runs'] = run_history_team2
    results['team2_wickets'] = wicket_history_team2
    results.to_csv(os.path.join(MatchSimulator.path, 'avg_match_progression.csv'), index=False)

def main():
    team1 = np.full(11, -1)
    team2 = np.full(11, -1)
    run_distribution = get_run_distribution(team1, team2, 100)
    # Save to file
    pd.DataFrame(run_distribution).to_csv(os.path.join(MatchSimulator.path, 'first_innings_run_distribution.csv'), index=False)


if __name__ == '__main__':
    main()