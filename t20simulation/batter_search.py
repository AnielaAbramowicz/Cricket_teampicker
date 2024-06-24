import pandas as pd
import os
from difflib import SequenceMatcher

def get_batter_id(batter_name, batter_file):
    df = pd.read_csv('batter_file')
    # Find similar names
    similar_names = []
    for name in df['Name']:
        if SequenceMatcher(None, batter_name, name).ratio() > 0.8:
            similar_names.append(name)

    # Return list of ids for similar names
    return df[df['Name'].isin(similar_names)]['ID'].values.tolist()

def main():
    # Path of this file
    path = os.path.dirname(os.path.realpath(__file__))
    get_batter_id('Virat Kohli', 'batter_file.csv')

if __name__ == '__main__':
    main()
