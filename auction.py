import pandas as pd
import numpy as np

class AuctionEvent:

    def __init__(self, player : str, is_bid : bool, team : int = None, price_sold : int = None):
        self.player = player # The name of the player
        self.team = team # 1 if is is our team, 0 otherwise.
        self.is_bid = is_bid # If true, the event is a bid. If false, the event is a purchase.
        self.price_sold = price_sold # The price the player was sold for, if the event is a purchase.

class Auction:

    event_log = []
    current_player = None

    def __init__(self, player_data : pd.DataFrame):
        self.player_data = player_data.copy()

        self.pool_data = {
            'batters' : self.player_data['TYPE'].value_counts()['Batter'],
            'bowlers' : self.player_data['TYPE'].value_counts()['Bowler'],
            'wicketkeepers' : self.player_data['TYPE'].value_counts()['Wicket-Keeper'],
            'allrounders' : self.player_data['TYPE'].value_counts()['All-Rounder'],
            'foreign' : self.player_data['OverseasIndian'].value_counts()['Foreign'],
            'indian' : self.player_data['OverseasIndian'].value_counts()['Indian'],
        }

        self.rng = np.random.default_rng(seed=420)

        # Set the current player
        self.current_player = self.player_data.index[self.rng.integers(0, len(self.player_data))]

    def current_player(self):
        """
        Get the player that is currently being bid on.

        Returns:
            str: The name of the player.
        """

        return self.current_player

    def new_bid(self, price: int):
        """
        Reflects a bid being placed on the current player being bid on.
        Sets the price of the player to the new bid price.

        Args:
            player (str): The name of the player.
            price (int): The bid price.

        Returns:
            None
        """

        self.event_log.append(AuctionEvent(self.current_player, False))
        self.player_data = self.player_data.at[player, 'price'] = price

    def new_purchase(self, team : int):
        """
        Update the auction to reflect the purchase of the player currently being bid on.
        Sets the player as sold and removes them from the pool.
        Picks a new player to bid on.

        Args:
            player (str): The name of the player.
            team (int): 1 if is is our team, 0 otherwise.

        Returns:
            None
        """

        self.event_log.append(AuctionEvent(self.current_player, team, True))

        # Update the player data
        self.player_data = self.player_data.drop(self.current_player, axis = 0) # Remove the player from the pool

        # Update the role counts
        role = self.player_data.loc[self.current_player]['TYPE']
        self.pool_data[role] -= 1

        # Update the foreign/indian counts
        origin = self.player_data.loc[self.current_player]['OverseasIndian']
        if origin == 'Overseas':
            self.pool_data['foreign'] -= 1
        else:
            self.pool_data['indian'] -= 1

        # Pick a new player
        self.current_player = self.player_data.index[self.rng.integers(0, len(self.player_data))]

        # Here we should set the price of the player to asking price (we don't have that data in the csv yet)