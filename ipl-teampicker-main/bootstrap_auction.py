import numpy as np
import pandas as pd
#this file containts one main function which you can call and it returns a list with resampled auction prices as a csv

def resample_acution(auction : pd.DataFrame, resample : float) -> pd.DataFrame:
    """
    This function takes the dataframe for an auction and resamples it. Resulting in a new auction.
    Args:
        auction (pd.DataFrame): The original auction that we resample.
        resample (float): The percent of the bids we resample.

    Returns:
        pd.DataFrame
    """
    prices = auction['Price'].to_list() #list of the prices

    players = np.arange(len(auction))

    n = np.floor(len(auction) * resample) # how many players price we resample

    resample_prices = (np.random.choice(players, size = n, replace = False)) # index for the players whose price we reample

    filtered_prices = [price for idx, price in enumerate(prices) if idx not in resample_prices]

    sampled_prices = np.random.choice(filtered_prices, size = n, replace = True)

    for i in range(n):
        prices[resample_prices[i]] = sampled_prices[i]
    
    auction['Sample-Price'] = prices
    return auction