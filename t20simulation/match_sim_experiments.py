import pandas as pd
import numpy as np
import os
from match_simulator import MatchSimulator


#all_player.csv should have the following columns: player_id, player_type, player_value

class MatchSimulatorExperiment:

    def __init__(self, num_iterations : int) -> None:
        #self.player_data = pd.read_csv('ipl-teampicker-main/2024_auction_pool.csv') #TODO: replace with actual path
        self.rng = np.random.default_rng()
        self.num_iterations = num_iterations

    def run_experiments(self, opponent_teams : np.ndarray[np.ndarray[np.ndarray[int]]], our_team : np.ndarray[np.ndarray[int]]) -> dict[np.ndarray[int], dict[np.ndarray[int], int]]:
        """
        This function runs the experiments for the match simulator
        param opponent_teams: np.ndarray[np.ndarray[int]]: The teams that we are playing against
        param our_team: np.ndarray[int]: The team that we are playing with
        """
        results = []
        for opponent_team in opponent_teams:
            result = [0, 0] #stores the number of wins for each team

            match = MatchSimulator(np.ones(11), np.ones(11))

            # Simulate matches when our team bats first
            for _ in range(self.num_iterations):

                selected_team_batting = self.select_players(our_team[0])
                selected_opponents_batting =  self.select_players(opponent_team[0])

                selected_team_bowling = self.select_players(our_team[1])
                selected_opponents_bowling =  self.select_players(opponent_team[1])

                match.reset_with_new_teams(selected_team_batting, selected_opponents_batting, selected_team_bowling, selected_opponents_bowling)

                target = match.simulate_inning_1(True)[0]
                chase = match.simulate_inning_2(True, target)[1]

                if chase < target:
                    result[0] += 1
                elif chase > target:
                    result[1] += 1
                else :
                    result[0] += 0.5
                    result[1] += 0.5

            # Simulate matches when our team bats second
            for _ in range(self.num_iterations):
                selected_team_batting = self.select_players(our_team[0])
                selected_opponents_batting =  self.select_players(opponent_team[0])

                selected_team_bowling = self.select_players(our_team[1])
                selected_opponents_bowling =  self.select_players(opponent_team[1])

                match.reset_with_new_teams(selected_team_batting, selected_opponents_batting, selected_team_bowling, selected_opponents_bowling)

                target = match.simulate_inning_1(False)[0]
                chase = match.simulate_inning_2(False, target)[0]

                if chase < target:
                    result[0] += 1
                elif chase > target:
                    result[1] += 1
                else :
                    result[0] += 0.5
                    result[1] += 0.5

            results.append(result)

        return results

    def select_players(self, players : np.ndarray[int]) -> np.ndarray[int]:
        """
        This function selects the playing 11 for the match from the 25 players available in a team.
        It does so randomly.
        """

        # Randomly select 11 players
        return self.rng.choice(players, size=11, replace=False)

    def select_players_old(self, players : np.ndarray[int]) -> np.ndarray[int]:
        """
        This function selects the playing 11 for the match from the 25 players available in a team
        by drawing from the distribution of evaluations of the players.
        Returns:
            np.ndarray[int]: The player ids of the playing 11
        """
        # Filter bowlers and other players
        player_types = self.player_data.set_index('player_id')['player_type']
        bowlers_mask = player_types.loc[players].isin(['Bowler', 'All-Rounder'])
        bowlers = players[bowlers_mask]
        other_players = players[~bowlers_mask]

        # Get performance values for all players
        performance_values = self.player_data.set_index('player_id')['Performance'].loc[players]
        probs = performance_values / performance_values.sum()

        # Sample 5 bowlers
        sampled_bowlers = self.rng.choice(bowlers, size=5, replace=False, p=probs.loc[bowlers].values)

        # Remove sampled bowlers from other players and adjust probabilities
        remaining_players = np.setdiff1d(other_players, sampled_bowlers, assume_unique=True)
        remaining_probs = probs.loc[remaining_players]

        # Sample 6 other players
        sampled_players = self.rng.choice(remaining_players, size=6, replace=False, p=remaining_probs.values)

        return np.concatenate((sampled_bowlers, sampled_players))

def main():
    our_team_batting_ids = np.array([-1, 399, 548, 483, 590, 97, 476, 3046, 423, 389, 1890, 368, -1, 346, -1, 340, -1, -1, -1, 544, -1, -1, -1, -1])
    out_team_bowling_ids = np.array([-1, 301, -1, -1, 456, -1, -1, -1, 299, 291, 427, 174, -1, -1, -1, -1, -1, -1, -1, 235, -1, -1, -1, -1])
    our_team = np.array([our_team_batting_ids, out_team_bowling_ids])

    # Read in the opponent teams
    path = os.path.dirname(os.path.realpath(__file__))
    squads = pd.read_csv(os.path.join(path, 'IPL_2024_Player_Squads.csv'))
    teams = squads['TEAM'].unique()
    opponent_teams = []
    for i, t in enumerate(teams):
        batting_ids = squads[squads['TEAM'] == t]['batting id'].values
        bowling_ids = squads[squads['TEAM'] == t]['bowling id'].values
        opponent_teams.append(np.array([batting_ids, bowling_ids]))

    match_simulator = MatchSimulatorExperiment(100)
    results = match_simulator.run_experiments(opponent_teams, our_team)
    print(results)

if __name__ == '__main__':
    main()
    




        

    



    