import pandas as pd
import os

def same_person(name1, name2, exact=False):
    if exact:
        return name1 == name2

    same_surname = name1.split(' ')[-1] == name2.split(' ')[-1]
    same_firstname = name1[0] == name2[0]
    
    name1_is_initial = name1.split(' ')[0].isupper()
    name2_is_initial = name2.split(' ')[0].isupper()

    # If neither names are initials, check if the first names are the same
    if not (name1_is_initial or name2_is_initial):
        same_firstname = name1.split(' ')[0].lower() == name2.split(' ')[0].lower()

    return same_surname and same_firstname

def get_batter_ids(batter_names, batter_file):
    batter_ids = pd.read_csv(batter_file)[['batter', 'batter id']]
    batter_ids = batter_ids.drop_duplicates()
    batter_ids.set_index('batter', inplace=True)

    ids = []

    for b1 in batter_names:
        found_id = -1
        for b2 in batter_ids.index:
            if b1 == b2:
                found_id = batter_ids.loc[b2]['batter id']
                print("Exact match found for", b1, "with id", found_id)
                break
            if same_person(b1, b2):
                if found_id != -1:
                    print("Multiple batters found for", b1)
                found_id = batter_ids.loc[b2]['batter id']
        if found_id == -1:
            print('No batter found for', b1)
        ids.append(found_id)

    return ids

def return_same_names(name, batter_file):
    batter_ids = pd.read_csv(batter_file)[['batter', 'batter id']]
    batter_ids = batter_ids.drop_duplicates()
    batter_ids.set_index('batter', inplace=True)
    same_names = []
    for b in batter_ids.index:
        if same_person(name, b):
            same_names.append(b)
    return same_names

def main():
    # Path of this file
    path = os.path.dirname(os.path.realpath(__file__))
    auction_24 = pd.read_csv(r'C:\\Users\\natem\\University\\Project2-2\\ipl-teampicker\\ipl-teampicker-main\\2024_auction_pool.csv')
    batters = auction_24['PLAYER']
    ids = get_batter_ids(batters, os.path.join(path,'batter_runs.csv'))
    #same_names = return_same_names('Ramandeep Singh', os.path.join(path,'batter_runs.csv'))
    print(ids)

if __name__ == '__main__':
    main()
