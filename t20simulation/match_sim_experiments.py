from match_simulator import matchSimulator
import pandas as pd
import numpy as np


#all_player.csv should have the following columns: player_id, player_type, player_value

class MatchSimulatorExperiment:

    def __init__(self, num_iterations : int) -> None:
        self.player_data = pd.read_csv('all_players.csv') #TODO: replace with actual path NATE
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
            match = matchSimulator(our_team, opponent_team)
            for iteration in range(self.num_iterations):
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
        It simply selects the eleven players by drawing from the distribution of evaluations of the players
            returns : np.ndarray[int]: The player ids of the playing 11
        """
        #sample without replacement from the distrubution of player values
        player_values = {player: value for player, value in zip(players, self.player_data[self.player_data['player_id'].isin(players)]['player_value'].values)}
        players = np.array(list(player_values.keys()))
        probs = np.array(list(player_values.values()))
        # Normalize the probabilities
        probs = probs / np.sum(probs)
        # Sample 11 players without replacement
        sampled_players = self.rng.choice(players, size=11, replace=False, p=probs)
        return sampled_players
    




        

    
    