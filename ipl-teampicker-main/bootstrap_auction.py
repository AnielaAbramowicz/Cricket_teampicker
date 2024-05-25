import numpy as np
import pandas as pd
import argparse
import sys
from functools import partial

def resample_auction(auction : pd.DataFrame, reshuffle : bool) -> pd.DataFrame:
    """
    This function takes the dataframe for an auction and resamples it. Resulting in a new auction.
    It samples 3 groups seperately as they are distinct in price distribution.
    - capped indian players
    - uncapped indian players
    - international
    Args:
        auction (pd.DataFrame): The original auction that we resample.
        reshuffle (bool): True if we want to change the order of players being sold for the new auction
    Returns:
        pd.DataFrame : the new auction
    """

    indian_capped_price = auction[(auction['OverseasIndian'] == 'Indian') & (auction['Played For Country'] == 1)]['Selling Price'].tolist()
    indian_uncapped_price = auction[(auction['OverseasIndian'] == 'Indian') & (auction['Played For Country'] == 0)]['Selling Price'].tolist()
    international_price = auction[auction['OverseasIndian'] == 'Overseas']['Selling Price'].tolist()

    def new_price(row, indian_capped_price, indian_uncapped_price, international_price):
        if row['OverseasIndian'] == 'Overseas':
            return np.random.choice(international_price)
        elif row['Played For Country'] == 0:
            return np.random.choice(indian_uncapped_price)
        else:
            return np.random.choice(indian_capped_price)

    new_price_func = partial(new_price, indian_capped_price = indian_capped_price, indian_uncapped_price = indian_uncapped_price, international_price = international_price)

    new_auction = auction.copy()

    new_auction['Selling Price'] = new_auction.apply(new_price_func, axis=1)

    if reshuffle:
        new_auction = new_auction.sample(frac=1).reset_index(drop=True)

    return new_auction

def main():
    """
    Main function to run the bootstrapping of the auction.
    Accepts arguments from the command line, example:

    python bootstrap_auction.py --file auction.csv -n 100

    In the above example, a new file will be created, called 'bootstrapped_auction_100.csv'
    which will contain 100 bootstrapped auctions, which were created by sampling with replacement from the auction.csv file.
    The resulting file will be in the same format as the original auction.csv file, but with new selling prices, and an
    addition column called 'Auction Number' which will indicate which bootstrapped auction a row belongs to.
    So, to extract only the first bootstrapped auction, you can drop the rows where 'Auction Number' is not 1.
    """

    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help='The file to load the auction from.', required=True)
    parser.add_argument('-n', type=int, default=100, help='The number of auctions to generate.')
    args = parser.parse_args()

    num_auctions = args.n
    auction = pd.read_csv(args.file)

    # Check if the number of auctions is valid
    if num_auctions < 1:
        sys.exit('Number of auctions must be greater than 0.')

    bootstrapped_auctions = resample_auction(auction, False) # Bootstrap the first auction
    bootstrapped_auctions['Auction Number'] = 1 # Add the auction number to the first auction

    # Boostrap the rest of the auctions
    for i in range(2, num_auctions):
        new_auction = resample_auction(auction, False)
        new_auction['Auction Number'] = i # Add the auction number to the new auction
        bootstrapped_auctions = pd.concat([bootstrapped_auctions, new_auction], ignore_index=True) # Append the new auction to the previous auctions

    bootstrapped_auctions.to_csv(f'bootstrapped_auctions_{num_auctions}.csv', index=False)

if __name__ == '__main__':
    main()