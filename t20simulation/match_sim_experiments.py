from match_simulator import matchSimulator
import pandas as pd
import numpy as np


#all_player.csv should have the following columns: player_id, player_type, player_value

class MatchSimulatorExperiment:

    def __init__(self, num_iterations : int) -> None:
        self.player_data = pd.read_csv('ipl-teampicker-main/2024_auction_pool.csv') #TODO: replace with actual path
        self.rng = np.random.default_rng()
        self.num_iterations = num_iterations

    def run_experiments(self, opponent_teams : np.ndarray[np.ndarrays[int]], our_team : np.ndarray_int) -> dict[np.ndarray[int], dict[np.ndarray[int], int]]:
        """
        This function runs the experiments for the match simulator
        param opponent_teams: np.ndarray[np.ndarray[int]]: The teams that we are playing against
        param our_team: np.ndarray[int]: The team that we are playing with
        """
        results = {}
        for opponent_team in opponent_teams:
            result = {our_team: 0, opponent_team: 0} #stores the number of wins for each team
            for iteration in range(self.num_iterations):

                selected_players = self.select_players(our_team)
                selected_opponents =  self.select_players(opponent_team)
                our_bowlers = selected_players[selected_players['player_type'] == 'Bowler' | selected_players['player_type'] == 'All-Rounder']
                opponent_bowlers = selected_opponents[selected_opponents['player_type'] == 'Bowler' | selected_opponents['player_type'] == 'All-Rounder']
                match = matchSimulator(selected_players, selected_opponents, our_bowlers, opponent_bowlers)
                target, our_team_wickets, our_team_deliveries = match.simulate_inning_1(True)
                chase, opponent_wickets,opponent_team_deliveries = match.simulate_inning_2(True, target)
                if chase < target:
                    result[our_team] += 2
                elif chase > target:
                    result[opponent_team] += 2
                else :
                    result[our_team] += 1
                    result[opponent_team] += 1
            for iteration in range(self.num_iterations):
                selected_players = self.select_players(our_team)
                selected_opponents =  self.select_players(opponent_team)
                our_bowlers = selected_players[selected_players['player_type'] == 'Bowler' | selected_players['player_type'] == 'All-Rounder']
                opponent_bowlers = selected_opponents[selected_opponents['player_type'] == 'Bowler' | selected_opponents['player_type'] == 'All-Rounder']
                match = matchSimulator(selected_players, selected_opponents, our_bowlers, opponent_bowlers)
                target, opponent_team_wickets, opponent_team_deliveries = match.simulate_inning_1(False)
                chase, our_team_wickets, our_team_deliveries = match.simulate_inning_2(False, target)
                if chase < target:
                    result[our_team] += 2
                elif chase > target:
                    result[opponent_team] += 2
                else :
                    result[our_team] += 1
                    result[opponent_team] += 1
            results[opponent_team] = result

    def select_players(self, players : np.ndarary[int]) -> np.ndarray[int]:
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

    




        

    
    