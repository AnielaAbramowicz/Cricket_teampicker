import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=3)

def generate_performances(data : pd.DataFrame):
    # Generate normal performances
    return rng.normal(loc=50, scale=20, size=len(data))

def generate_prices(data : pd.DataFrame):
    # gonna generate log normal prices for international players
    international = rng.lognormal(mean=17, sigma=1.12, size=len(data['OverseasIndian']=='Overseas')).tolist() # mean and std are emperically determined from the data

    uncapped_indian = []
    num_uncapped_indian = ((data['OverseasIndian'] == 'Indian') & (data['Played For Country'] == 0)).value_counts()[True]
    num_indian = len(data['OverseasIndian'] == 'Indian')
    for _ in range(num_uncapped_indian):
        rnum = rng.uniform(0, 1)
        if rnum <= 0.7: # 70% of the time, uncapped indian players are sold for 2e6 rupees
            uncapped_indian.append(2e6)
        else:
            uncapped_indian.append(rng.uniform(2e6, 9e7))

    capped_indian = rng.lognormal(mean=17, sigma=1.22, size=num_indian-num_uncapped_indian).tolist() # mean and std are emperically determined from the data
    
    prices = []

    for i in range(len(data)):
        if data['OverseasIndian'][i] == 'Overseas':
            prices.append(international.pop())
        elif (data['OverseasIndian'][i] == 'Indian') and (not data['Played For Country'][i]):
            prices.append(uncapped_indian.pop())
        else:
            prices.append(capped_indian.pop())


    return prices