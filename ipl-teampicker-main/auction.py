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
    current_player : str = None

    def __init__(self, player_data : pd.DataFrame):
        self.player_data = player_data.copy()

        # Add an "available" column
        self.player_data['Available'] = True

        self.pool_data = {
            'batters' : self.player_data['TYPE'].value_counts()['Batter'],
            'bowlers' : self.player_data['TYPE'].value_counts()['Bowler'],
            'wicketkeepers' : self.player_data['TYPE'].value_counts()['Wicket-Keeper'],
            'allrounders' : self.player_data['TYPE'].value_counts()['All-Rounder'],
            'foreign' : self.player_data['OverseasIndian'].value_counts()['Overseas'],
            'indian' : self.player_data['OverseasIndian'].value_counts()['Indian'],
            'total' : len(self.player_data)
        }

        self.role_mapping = {
            'Batter' : 'batters',
            'Bowler' : 'bowlers',
            'Wicket-Keeper' : 'wicketkeepers',
            'All-Rounder' : 'allrounders'
        }

        self.rng = np.random.default_rng(seed=1)

        # Set the current player
        self.current_player = self.player_data.index[self.rng.integers(0, len(self.player_data))]

    def current_player(self):
        """
        Get the player that is currently being bid on.

        Returns:
            str: The name of the player.
        """

        return self.current_player

    def new_bid(self, price: int, team : int):
        """
        Reflects a bid being placed on the current player being bid on.
        Sets the price of the player to the new bid price.

        Args:
            player (str): The name of the player.
            price (int): The bid price.

        Returns:
            None
        """

        self.event_log.append(AuctionEvent(self.current_player, True, team))
        self.player_data.at[self.current_player, 'Selling Price'] = price

    def new_purchase(self, team : int):
        """
        Update the auction to reflect the purchase of the player currently being bid on.
        Sets the player as sold and removes them from the pool.
        Picks a new player to bid on.

        Args:
            player (str): The name of the player.
            team (int): 1 if is is our team, 0 otherwise.

        Returns:
            str: The name of the player that was purchased.
        """

        self.event_log.append(AuctionEvent(self.current_player, False, team=team, price_sold=self.player_data.at[self.current_player, 'Selling Price']))

        # Update the role counts
        role = self.player_data.at[self.current_player, 'TYPE']
        self.pool_data[self.role_mapping[role]] -= 1
        self.pool_data['total'] -= 1

        # Update the foreign/indian counts
        origin = self.player_data.loc[self.current_player]['OverseasIndian']
        if origin == 'Overseas':
            self.pool_data['foreign'] -= 1
        else:
            self.pool_data['indian'] -= 1

        # Set the player as unavailable
        self.player_data.at[self.current_player, 'Available'] = False

        # Pick a new player
        purchased_player = self.current_player
        if self.pool_data['total'] == 0:
            self.current_player = None
        else:
            self.current_player = self.player_data[self.player_data['Available']].index[self.rng.integers(0, self.pool_data['total'])]

        # Here we should set the price of the player to asking price (we don't have that data in the csv yet)

        return purchased_player