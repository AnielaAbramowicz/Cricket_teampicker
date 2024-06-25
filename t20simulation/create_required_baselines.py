"""
This file was used to calculate parameters for only the players we needed to calculate ERD for.
"""

from data_helper import SimDataHelper
import numpy as np
import pandas as pd
import os

def init_batting_baselines(batters : np.array):
    print('Initializing batting baselines for batters', batters)
    helper = SimDataHelper()
    helper.initialize()
    for i, batter in enumerate(batters):
        print('Calculating baselines for batter', batter)
        print('Batter', i, 'of', len(batters))
        p = helper.get_batting_probabilities(batter, 0, 0)
        print(p)

def init_bowling_baselines(bowlers : np.array):
    print('Initializing bowling baselines for bowlers', bowlers)
    helper = SimDataHelper()
    helper.initialize()
    for i, bowler in enumerate(bowlers):
        print('Calculating baselines for bowler', bowler)
        print('Bowler', i, 'of', len(bowlers))
        q = helper.get_bowling_probabilities(bowler, 0, 0)
        print(q)

def main():
    # Path of current file
    path = os.path.dirname(os.path.realpath(__file__))

    helper = SimDataHelper()

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

    #init_batting_baselines(batter_ids)
    init_bowling_baselines(bowler_ids)

if __name__ == '__main__':
    main()