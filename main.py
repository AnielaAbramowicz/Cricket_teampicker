import pandas as pd
from auction import Auction, AuctionEvent

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

    if player_data.loc[event.player]['Role'] == 'Batter':
        new_constraints['min_batters'] -= 1
    
    elif player_data.loc[event.player]['Role'] == 'Bowler':
        new_constraints['min_bowlers'] -= 1

    elif player_data.loc[event.player]['Role'] == 'Wicketkeeper':
        new_constraints['min_wicketkeepers'] -= 1
        new_constraints['min_batters'] -= 1 # wicketkeepers are also batters

    elif player_data.loc[event.player]['Role'] == 'Allrounder':
        new_constraints['min_allrounders'] -= 1
        new_constraints['min_batters'] -= 1 # allrounders are also batters
        new_constraints['min_bowlers'] -= 1 # allrounders are also bowlers

    if player_data.loc[event.player]['OverseasIndian'] != 'India':
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
    if player_data.loc[event.player]['Role'] == 'Batter':
        new_team_data['batters'] += 1
    
    elif player_data.loc[event.player]['Role'] == 'Bowler':
        new_team_data['bowlers'] += 1

    elif player_data.loc[event.player]['Role'] == 'Wicketkeeper':
        new_team_data['wicketkeepers'] += 1

    elif player_data.loc[event.player]['Role'] == 'Allrounder':
        new_team_data['allrounders'] += 1

    if player_data.loc[event.player]['OverseasIndian'] != 'India':
        new_team_data['foreign'] += 1

    return new_team_data 

def calc_evaluations(player_data : pd.DataFrame, player_weights : list, team_data : dict) -> pd.DataFrame:
    """
    Update the evaluations of players based on the given player data, player weights and team data.

    Parameters:
    player_data (pd.DataFrame): The data frame containing player information.
    player_weights (list): The list of weights for each player.
    team_data (dict): The dictionary containing current team composition data.

    Returns:
    list: A list of player evaluations.
    """
    pass

def bid_margin(price : int) -> int:
    """
    This function returns the margin that should be added to the highest bid to make a new bid.
    There are IPL rules that specify the minimum margin that should be added to the highest bid.
    The margin is dependent on the bid price.
    """

    return price + 0.1*price # IDK what the actual margin is rn so i made this up

def main():
    # load the dataset
    player_data = pd.DataFrame("2024_auction_pool_mock.csv") #here choose the actual path
    player_data.set_index('player', inplace=True) # set the player names as the index

    # We should access the player data through the auction object that we create later,
    # so that we are using the up to date player data.

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
        'min_batters' : 0,
        'min_bowlers' : 0,
        'min_wicketkeepers' : 0,
        'min_allrounders' : 0,
        'max_foreign' : 0,
        'min_players' : 0,
        'budget' : 0
    }

    player_weights = [0 for k in range(len(player_data))]
    player_evaluations = player_data['Performance'].to_list()

    # Create the auction
    auction = Auction(player_data)

    # Auction loop
    auction_over = False
    while not auction_over:
        print(f"Currently bidding on {auction.current_player}: ")

        # Here we do some stuff to handle what happens in the auction.
        # This is where the auction simulation stuff kinda comes into play.
        # For now we can just generate a random bid from a distribution that somewhat approximates a real bid for this type of player.

        highest_opponent_bid = auction.player_data.loc[auction.current_player] # since this is a mock csv with prices generated form a log normal distribution, i'm just using that as the highest bid for now

        auction.new_bid(highest_opponent_bid) 

        # Now we have to answer: do we bid on this player?
        lp_data = auction.player_data.drop('Performance')
        lp_data['Evaluation'] = player_evaluations
        # WE CALL THE LP HERE WITH THE DATA ABOVE AND THE CURRENT CONSTRAINTS
        # the price of the player should be the highest_opponent_bid + margin (there are specific, ipl specified, minimum margins depending on the bid price)
        # THE LP RETURNS A TEAM
        team = []
        if auction.current_player in team: # If the player is in the LP's optimal team, we bid
            auction.new_bid(highest_opponent_bid + bid_margin(highest_opponent_bid))

        # WAS OUR BID SUCCESFUl?
        # since we are not simulating auction dynamics yet, we say that it was
        team = 1 # 1 is us
        auction.new_purchase(team) # this purchases the current player

        # Now we have to update the constraints, team data and player value weights
        constraint_data = update_constraints(constraint_data, auction.event_log[-1])
        team_data = update_team_data(team_data, auction.event_log[-1])
        player_evaluations = calc_evaluations(player_data, player_weights, team_data, auction.event_log[-1])

        # check if the auction is over
        if constraint_data['min_players'] == 0 or constraint_data['budget'] == 0 or len(auction.player_data) == 0:
            auction_over = True

if __name__ == '__main__':
    main()