import pandas as pd
import numpy as np
from match_simulator import MatchSimulator

class ERD:
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

    def calculate_runs_with(self, player : int, iterations : int):
        team1 = np.full(11, -1) # -1 means we do not have the player id, so that match simulator treats it as an avg player
        #team2 = np.full(11, -1)
        team2 = np.array([1,2,3,4,5,6,7,8,9,10,11])


        self.match_simulator = MatchSimulator(team1, team2)

        team1_runs = 0

        for i in range(iterations):
            #team1 = np.full(11, -1) # -1 means we do not have the player id, so that match simulator treats it as an avg player
            team1 = np.array([1,2,3,4,5,6,7,8,9,10,11])
            team1[np.random.randint(11)] = player
            team1 = np.array([1,2,3,4,5,6,7,8,9,10,11])
            team1_runs += self.match_simulator.simulate_inning_1(True)[0]

        team1_runs /= iterations

        self.avg_runs = team1_runs

        return team1_runs



def main():
    erd_calc = ERD()
    avg_runs = erd_calc.calculate_average_runs_scored(10000)
    runs_with = erd_calc.calculate_runs_with(0, 10000)
    print(avg_runs)
    print(runs_with)

if __name__ == '__main__':
    main()