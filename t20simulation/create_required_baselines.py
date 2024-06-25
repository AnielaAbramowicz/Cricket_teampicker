from data_helper import SimDataHelper
import numpy as np

def init_batting_baselines(batters : np.array):
    helper = SimDataHelper()
    helper.initialize()
    for batter in batters:
        print('Calculating baselines for batter', batter)
        helper.get_batting_probabilities(batter, 0, 0)

def init_bowling_baselines(bowlers : np.array):
    helper = SimDataHelper()
    helper.initialize()
    for bowler in bowlers:
        print('Calculating baselines for bowler', bowler)
        helper.get_bowling_probabilities(bowler, 0, 0)

def main():
    helper = SimDataHelper()

    ids = [1,2,3,4,5,6,7,8,9,10,11]
    init_batting_baselines(ids)

if __name__ == '__main__':
    main()