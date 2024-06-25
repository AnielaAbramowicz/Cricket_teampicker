import pandas as pd
import numpy as np
import os
from match_simulator import MatchSimulator

class ERD:

    avg_runs = 134.3843

    def __init__(self):
        pass

    def calculate_average_runs_scored(self, iterations : int):
        team1 = np.full(11, -1) # -1 means we do not have the player id, so that match simulator treats it as an avg player
        team2 = np.full(11, -1)
        self.match_simulator = MatchSimulator(team1, team2)

        team1_runs = 0

        for i in range(iterations):
            team1_runs += self.match_simulator.simulate_inning_1(True)[0]

        team1_runs /= iterations

        self.avg_runs = team1_runs

        return team1_runs

    def calculate_batting_contribution(self, player : int, iterations : int):
        """
        This function calculates the batting contribution of a player to the team.
        
        param
            player: The batting id of the player whose batting contribution is to be calculated
        """

        team1 = np.full(11, -1)
        team2 = np.full(11, -1)

        self.match_simulator = MatchSimulator(team1, team2)

        team1_runs = 0

        for _ in range(iterations):
            team1 = np.full(11, -1)
            team1[np.random.randint(11)] = player
            team1_runs += self.match_simulator.simulate_inning_1(True)[0]

        team1_runs /= iterations

        # Difference between average runs and runs with player
        return team1_runs - self.avg_runs

    def calculate_bowling_contribution(self, player : int, iterations : int):
        """
        This function calculates the bowling contribution of a player to the team.
        
        param 
            player: The bowling id of the player whose bowling contribution is to be calculated
        """

        team1 = np.full(11, -1)
        team2 = np.full(11, -1)

        self.match_simulator = MatchSimulator(team1, team2, team1_bowling_lineup=[player, -1, -1, -1, -1], team2_bowling_lineup=[-1, -1, -1, -1, -1])

        team2_runs = 0

        for _ in range(iterations):
            team2 = np.full(11, -1)
            team2[np.random.randint(11)] = player
            team2_runs += self.match_simulator.simulate_inning_1(False)[0]

        team2_runs /= iterations

        # Difference between average runs and runs with player
        return team2_runs - self.avg_runs

    def calc_erd(self, batting_id : int, bowling_id : int, can_bowl : bool, iterations : int):
        batting_contribution = self.calculate_batting_contribution(batting_id, iterations)
        bowling_contribution = 0
        if can_bowl:
            bowling_contribution = self.calculate_bowling_contribution(bowling_id, iterations)
        
        return batting_contribution - bowling_contribution, batting_contribution, bowling_contribution

    def calculate_runs_with(self, player : int, iterations : int):
        team1 = np.full(11, -1) # -1 means we do not have the player id, so that match simulator treats it as an avg player
        team2 = np.full(11, -1)

        self.match_simulator = MatchSimulator(team1, team2)

        team1_runs = 0

        for i in range(iterations):
            team1 = np.full(11, -1) # -1 means we do not have the player id, so that match simulator treats it as an avg player
            team1[np.random.randint(11)] = player
            team1_runs += self.match_simulator.simulate_inning_1(True)[0]

        team1_runs /= iterations

        self.avg_runs = team1_runs

        return team1_runs

def main():
    erd_calc = ERD()
    # Calculate erd for all players in the auction and 

    # Path of current file
    path = os.path.dirname(os.path.realpath(__file__))

    squad_players = pd.read_csv(os.path.join(path, 'IPL_2024_Player_Squads.csv'))
    auction_players = pd.read_csv(os.path.join(path,'2024_auction_pool.csv'))

    batter_ids = squad_players['batting id'].values
    bowler_ids = squad_players['bowling id'].values

    # Add auction player ids
    batter_ids = np.append(batter_ids, auction_players['batting id'].values)
    bowler_ids = np.append(bowler_ids, auction_players['bowling id'].values)

    # Remove duplicates
    batter_ids = np.unique(batter_ids)
    bowler_ids = np.unique(bowler_ids)

    # For each batter id, calculate the batting contribution

    # For each bowler id, calculate the bowling contribution
    bowling_contributions = pd.DataFrame(columns=['bowler_id', 'bowling_contribution']).set_index('bowler_id')
    for i, bowler_id in enumerate(bowler_ids):
        bowling_contributions.loc[bowler_id] = erd_calc.calculate_bowling_contribution(bowler_id-1, 2000)
        print('Bowler', i, 'of', len(bowler_ids), 'has contribution of ', bowling_contributions.loc[bowler_id])
        # save to file
        bowling_contributions.to_csv('bowling_contributions_new.csv', index=True)

if __name__ == '__main__':
    main()