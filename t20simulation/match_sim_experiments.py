from match_simulator import matchSimulator
import pandas as pd
import numpy as np


#all_player.csv should have the following columns: player_id, player_type, player_value

class MatchSimulatorExperiment:

    def __init__(self) -> None:
        self.player_data = pd.read_csv('all_players.csv')

    def select_players(players : np.ndarary[int]) -> np.ndarray[int]:
        """
        This function selects the playing 11 for the match from the 25 players available in a team
        It simply selects the eleven players by drawing from the distribution of evaluations of the players
            returns : np.ndarray[int]: The player ids of the playing 11
        """
        #sample without replacement from the distrubution of player values
        value_array = np.array()



        

    
    