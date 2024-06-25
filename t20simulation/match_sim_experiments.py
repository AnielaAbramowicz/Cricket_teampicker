from match_simulator import matchSimulator
import pandas as pd
import numpy as np


#all_player.csv should have the following columns: player_id, player_type, player_value

class MatchSimulatorExperiment:

    def __init__(self) -> None:
        self.player_data = pd.read_csv('all_players.csv') #TODO: replace with actual path NATE
        self.rng = np.random.default_rng()

    def initialize(self, opponent_teams : np.ndarray[np.ndarrays[int]], our_team : np.ndarray_int) -> None:
        print('Initializing...')

        print('Done.')

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
    




        

    
    