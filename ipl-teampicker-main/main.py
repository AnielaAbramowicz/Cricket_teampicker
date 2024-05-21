import pandas as pd
import numpy as np
import os

from auction import Auction, AuctionEvent
from project_mock_lp import optimize_team
from os import path
from colorama import init as colorama_init, Fore, Style


# Get the current working directory
dir_path = os.path.dirname(os.path.realpath(__file__))

TOTAL_PLAYERS = 332

def update_constraints(constraint_data: dict, event : AuctionEvent, player_data : pd.DataFrame) -> dict:
    """
    Creates a new set of LP constraints based on the type of auction event and the old constraints.
    Does not modify the original constraint data.

    Parameters:
    constraint_data (dict): A dictionary containing the constraint data.
    team_data (dict): A dictionary containing the team data.
    event (int): The type of auction event.

    Returns:
    dict: The new constraints.
    """
    new_constraints = constraint_data.copy()

    # No constraints change unless we buy a player
    if event.is_bid or event.team != 1:
        return new_constraints

    new_constraints['min_players'] -= 1 # we have added a player to our team

    # Deal with the player role:

    if player_data.loc[event.player]['TYPE'] == 'Batter':
        new_constraints['min_batters'] -= 1
    
    elif player_data.loc[event.player]['TYPE'] == 'Bowler':
        new_constraints['min_bowlers'] -= 1

    elif player_data.loc[event.player]['TYPE'] == 'Wicket-Keeper':
        new_constraints['min_wicketkeepers'] -= 1
        new_constraints['min_batters'] -= 1 # wicketkeepers are also batters

    elif player_data.loc[event.player]['TYPE'] == 'All-Rounder':
        new_constraints['min_allrounders'] -= 1
        new_constraints['min_batters'] -= 1 # allrounders are also batters
        new_constraints['min_bowlers'] -= 1 # allrounders are also bowlers

    if player_data.loc[event.player]['OverseasIndian'] != 'Indian':
        new_constraints['max_foreign'] -= 1

    new_constraints['budget'] -= event.price_sold # adjust the budget

    return new_constraints

def update_team_data(team_data : dict, event : AuctionEvent, player_data : pd.DataFrame) -> dict:
    """
    Updates the team data based on the event that has occured.

    Parameters:
    team_data (dict): A dictionary containing the team data.
    event (AuctionEvent): The event that has occured.

    Returns:
    dict: The updated team data.
    """
    new_team_data = team_data.copy()

    if event.is_bid or event.team != 1:
        return new_team_data
    
    new_team_data['total'] += 1

    # Deal with the player role:
    if player_data.loc[event.player]['TYPE'] == 'Batter':
        new_team_data['batters'] += 1
    
    elif player_data.loc[event.player]['TYPE'] == 'Bowler':
        new_team_data['bowlers'] += 1

    elif player_data.loc[event.player]['TYPE'] == 'Wicket-Keeper':
        new_team_data['wicketkeepers'] += 1

    elif player_data.loc[event.player]['TYPE'] == 'All-Rounder':
        new_team_data['allrounders'] += 1

    if player_data.loc[event.player]['OverseasIndian'] == 'Indian':
        new_team_data['indian'] += 1
    else:
        new_team_data['foreign'] += 1

    return new_team_data 

def calc_evaluations(player_data : pd.DataFrame, pool_data : dict, constraint_data : dict, team_data : dict) -> list:
    """
    #this is a test comment to see if im gonna punch a wall because of github
    Update the evaluations of players based on the given player data, player weights and team data.

    Parameters:
    player_data (pd.DataFrame): The data frame containing player information.
    pool_data (dict): the dictionary containing how many players are left in the pool at each position
    constraint_data (dict): the dictionary containing how many players we need at each position
    team_data (dict): The dictionary containing current team composition data.

    Returns:
    list: A list of player evaluations.
    """
    #the weight of the positional constraint is as follow:
    # w = exp((how many players we still need)/(how many players are left - how many players we need))
    # we only need to calculate the weights for batters, bowlers and wicketkeepers as all rounders count as both batters and bowlers
    # wicketkeepers also count as batters

    #stores the weights for every position
    weight_dict = {
        'Batter' : 1,
        'Bowler' : 1,
        'Wicket-Keeper' : 1,
        'All-Rounder' : 1
    }

    players_left = pool_data['total']

    # We can tweak this hyperparameter
    alpha = 1

    # for batters
    if constraint_data['min_batters'] > 0:
        #checks if we neeed more batters
        r_batters = (constraint_data['min_batters'])/(pool_data['batters'] + pool_data['wicketkeepers'] + pool_data['allrounders'] - constraint_data['min_batters'])
        # pool_data['batters'] + pool_data['wicketkeepers'] + pool_data['allrounders'] = how many batters are left
        weight_dict['Batter'] = np.exp(alpha * r_batters)

    # for bowlers
    if constraint_data['min_bowlers'] > 0:
        #checks if we neeed more bowlers
        r_bowler = (constraint_data['min_bowlers'])/(pool_data['bowlers'] + pool_data['allrounders']  - constraint_data['min_bowlers'])
        weight_dict['Bowler'] = np.exp(alpha * r_bowler)

    # for wicketkeepers
    if constraint_data['min_wicketkeepers'] > 0:
        #checks if we neeed more wicketkeepers
        r_wicketkeeper = (constraint_data['min_wicketkeepers'])/(pool_data['wicketkeepers'] -  constraint_data['min_wicketkeepers'])
        weight_dict['Wicket-Keeper'] = np.exp(alpha * r_wicketkeeper)

    weight_dict['All-Rounder'] = max(weight_dict['Batter'],weight_dict['Bowler'])

    return player_data.apply(lambda row: row['Performance'] * weight_dict[row['TYPE']], axis=1).tolist()

def inflate_current_player_eval(evaluation, players_left):
    # I thought that maybe it would be beneficial to inflate the evaluation of the player currently being bid on
    # to encourage the algorithm to buy players earlier on, in order to avoid the risk that comes with waiting until
    # the end of the auction to buy players
    a = 1
    return evaluation * (1 + (players_left / TOTAL_PLAYERS)) * a
    

def bid_margin(price : int) -> int:
    """
    This function returns the margin that should be added to the highest bid to make a new bid.
    There are IPL rules that specify the minimum margin that should be added to the highest bid.
    The margin is dependent on the bid price.
    """

    # For bids up to Rs. 1 Crore, the increment is Rs. 5 Lakh
    # For bids from Rs. 1 Crore to Rs. 2 Crore, the increments is Rs. 10 Lakh
    # For bids from Rs. 2 Crore to Rs. 3 Crore, the increment is Rs. 20 Lakh
    # The increments for each Crore above 3 is up to the auctioner's dicretion... 
    # they "usually go up to Rs. 25 lakh after the 10-crore mark" and "cannot be below 20 Lakh"

    # Rs. 1 Lakh = Rs. 100 000
    # Rs. 1 Crore = Rs. 100 Lakh = Rs. 10 000 000

    if price < 10000000: # If price < 1 Crore
        return 500000 # 5 Lakh
    elif price < 20000000: # If price < 2 Crore
        return 1000000 # 10 Lakh
    elif price < 30000000: # If price < 3 Crore
        return 2000000 # 20 Lakh
    else: # If price is above 3 Crore
        return 2500000 # 25 Lakh

def main():
    colorama_init() # initialize colorama

    # load the dataset
    # NB:
    # We should access the player data through the auction object that we create later,
    # so that we are using the up to date player data.
    pd_og = pd.read_csv(path.join(dir_path, "2024_auction_pool_mock.csv")) #here choose the actual path

    # There are two people named Shashank Singh, so we need to change one of their names
    i = pd_og[pd_og['PLAYER'] == 'Shashank Singh'].index[0]
    pd_og.at[i, 'PLAYER'] = 'Shashank Singh (2)'

    pd_og.set_index('PLAYER', inplace=True) # set the player names as the index

    # an array storing the current info on our team, below what each index means
    team_data = {
        'batters' : 0,
        'bowlers' : 0,
        'allrounders' : 0,
        'wicketkeepers' : 0,
        'foreign' : 0,
        'indian' : 0,
        'total' : 0
    }

    # list of all the constraints, in order of index
    constraint_data = {
        'min_batters' : 8,
        'min_bowlers' : 8,
        'min_wicketkeepers' : 3,
        'min_allrounders' : 5,
        'max_foreign' : 4,
        'min_players' : 25,
        'budget' : 100 * 1000000 # 100 Crore
    }

    selected_team = []

    print(f"{Fore.GREEN}Starting Auction.")
    print(f"Budget: {Style.RESET_ALL}{constraint_data['budget']}")
    print()

    # Create the auction
    auction = Auction(pd_og)

    # Auction loop
    auction_over = False
    while not auction_over:
        print(f"Currently bidding on {Fore.RED}{auction.current_player}{Style.RESET_ALL}\t\t\t", end='\r')

        # Here we do some stuff to handle what happens in the auction.
        # This is where the auction simulation stuff kinda comes into play.
        # For now we can just generate a random bid from a distribution that somewhat approximates a real bid for this type of player.

        # Calculate player evaluations
        player_evaluations = calc_evaluations(auction.player_data, auction.pool_data, constraint_data, team_data)

        highest_opponent_bid = auction.player_data.at[auction.current_player, 'Price'] # since this is a mock csv with prices generated form a log normal distribution, i'm just using that as the highest bid for now

        auction.new_bid(highest_opponent_bid, team=0) 

        # Now we have to answer: do we bid on this player?
        lp_data = auction.player_data
        lp_data['Evaluation'] = player_evaluations
        # Filter out unavailable players
        lp_data = lp_data[lp_data['Available']]
        # Add the bid margin to each player's price
        lp_data.loc[:, 'Price'] = lp_data['Price'].apply(lambda x: x + bid_margin(x))
        # Inflate the current player's evaluation
        #lp_data.loc[auction.current_player, 'Evaluation'] = inflate_current_player_eval(lp_data.loc[auction.current_player, 'Evaluation'], auction.pool_data['total'])
        # WE CALL THE LP HERE WITH THE DATA ABOVE AND THE CURRENT CONSTRAINTS
        # the price of the player should be the highest_opponent_bid + margin (there are specific, ipl specified, minimum margins depending on the bid price)
        # THE LP RETURNS A TEAM
        dream_team = optimize_team(lp_data, constraint_data)
        if auction.current_player in dream_team: # If the player is in the LP's optimal team, we bid
            auction.new_bid(highest_opponent_bid + bid_margin(highest_opponent_bid), team=1)

            # WAS OUR BID SUCCESFUl?
            # since we are not simulating auction dynamics yet, we say that it was
            team = 1 # 1 is us
            purchased_player = auction.new_purchase(1) # this purchases the current player

            print(end='\x1b[2K') # Clear the line
            print(f"Purchased player: {Fore.GREEN}{purchased_player}{Style.RESET_ALL}")

            # Add the player to our team
            selected_team.append(purchased_player)
        else:
            # In this case, the opponent "won" the bid (we don't need this player)
            auction.new_purchase(0)

        # Now we have to update the constraints, team data and player value weights
        constraint_data = update_constraints(constraint_data, auction.event_log[-1], auction.player_data)
        team_data = update_team_data(team_data, auction.event_log[-1], auction.player_data)

        # check if the auction is over
        if constraint_data['min_players'] == 0 or constraint_data['budget'] == 0 or auction.player_data['Available'].value_counts()[True] == 0:
            auction_over = True
            
    print()
    print(f"{Fore.BLUE}Auction Over.")
    print(f"Team statistics: {Style.RESET_ALL}")
    print()
    print(f"Batters: {team_data['batters']}")
    print(f"Bowlers: {team_data['bowlers']}")
    print(f"Allrounders: {team_data['allrounders']}")
    print(f"Wicketkeepers: {team_data['wicketkeepers']}")
    print(f"Foreign Players: {team_data['foreign']}")
    print(f"Total Players: {team_data['total']}")
    print(f"Budget Remaining: {constraint_data['budget']}")

if __name__ == '__main__':
    main()