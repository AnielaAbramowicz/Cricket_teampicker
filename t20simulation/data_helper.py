# This file contains the class SimDataHelper which is used to get statistics from the ball by ball data
# Import this class for use in simulation related stuff

import pandas as pd
import os

path = os.path.dirname(os.path.abspath(__file__))

class SimDataHelper:
    """
    Helper class to get statistics from the ball by ball data
    """

    def __init__(self):
        self.batter_file = pd.read_csv(os.path.join(path, 'batter_runs.csv'))

    def get_batter_outcomes_at_gamestage(self, batter : str, overs : int, wickets : int) -> list[int]:
        """
        Get the outcomes of a batter at a given game stage

        Parameters:
        batter (str): Name of the batter
        overs (float): The number of the over (1-20)
        wickets (int): Number of wickets fallen

        Returns:
        list: A list containing the outcome frequencies of the batter at the given game stage, where the first seven indices correspond to the number of runs scored and the last index corresponds to the number of dismissals
        """

        # Filter by batter
        outcomes = self.batter_file[self.batter_file['batter'] == batter]

        # Only consider first innings
        outcomes = outcomes[outcomes['innings'] == 1]

        # Filter by game stage
        outcomes = outcomes[(outcomes['over'] == overs) & (outcomes['wickets'] == wickets)]

        outcomes = outcomes.drop(columns=['match id', 'date', 'extra runs', 'innings'], errors='ignore')

        # Check if outcomes is empty
        # This means the batter has not faced a ball at this game stage
        if len(outcomes) == 0:
            return None

        # Summing the columns 'wickets_in_over', '1', '2', '3', '4', '5', '6', 'extras' to "squash" the data
        outcomes = outcomes.groupby(['over', 'batter', 'wickets'], as_index=False).sum()

        # Get the count of 1s, 2s, etc.
        # I want to sum vertically 
        # i.e. sum of 1s, sum of 2s, etc.

        return outcomes[['0', '1', '2', '3', '4', '5', '6', 'wicket']].values.tolist()[0]


def main():
    # Just for testing
    simdatahelper = SimDataHelper()
    outcomes = simdatahelper.get_batter_outcomes_at_gamestage('RD Gaikwad', 1, 1)
    print(outcomes)

if __name__ == '__main__':
    main()






