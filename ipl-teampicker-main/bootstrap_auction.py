import numpy as np
import pandas as pd

def resample_acution(auction : pd.DataFrame, reshuffle : bool) -> pd.DataFrame:
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

    indian_capped_price = auction[(auction['IndianOverseas'] == 'Indian') & (auction['Played For Country'] == 1)]['Selling Price'].tolist()
    indian_uncapped_price = auction[(auction['IndianOverseas'] == 'Indian') & (auction['Played For Country'] == 0)]['Selling Price'].tolist()
    international_price = auction[auction['IndianOverseas'] == 'Overseas']['Selling Price'].tolist()


    new_auction = auction.copy()

    new_auction['Selling Price'] = new_auction.apply(lambda row:new_price(row['IndianOverseas'], row['Played For Country']), axis=1)

    return new_auction

def new_price(nationality : str , capped : int) -> float:
    global indian_capped_price, indian_uncapped_price, international_price

    if nationality == 'Overseas':
        return np.random.choice(international_price)
    elif capped == 0:
        return np.random.choice(indian_uncapped_price)
    else:
        return np.random.choice(indian_capped_price)
