import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value
from pulp.apis import PULP_CBC_CMD

def optimize_team(data : pd.DataFrame, constraints : dict):
    # Unpack the constraints:
    number_players = constraints['min_players']
    min_batsmen = constraints['min_batters']
    min_bowlers = constraints['min_bowlers']
    max_foreign_players = constraints['max_foreign']
    min_allrounders = constraints['min_allrounders']
    min_wicketkeepers = constraints['min_wicketkeepers']
    price_cap = constraints['budget']
    player_evaluations = data['Evaluation'].tolist()
    
    # Create model
    model : LpProblem = LpProblem(name="Fantasy_Team_Optimization", sense=LpMaximize)
    
    # Create binary variables
    n = len(data)
    players = []
    
    # Create n binary variables and add them to the list
    for i in range(n):
        var_name = f"x_{i}"  # Variable names like x_0, x_1, ..., x_n-1
        binary_var = LpVariable(var_name, lowBound=0, upBound=1, cat= "Binary")
        players.append(binary_var)
    
    # Create constraints
    # Total player constraint
    model += lpSum(players) == number_players
    
    # Total price constraint
    total_price_constraint = lpSum([players[i] * data['Price'].iloc[i] for i in range(n)]) <= price_cap
    model += total_price_constraint
    
    # Position constraints
    batsmen_vars = [players[i] for i in range(n) if data['TYPE'].iloc[i] == 'Batter']
    bowlers_vars = [players[i] for i in range(n) if data['TYPE'].iloc[i] == 'Bowler']
    allrounders_vars = [players[i] for i in range(n) if data['TYPE'].iloc[i] == 'All-Rounder']
    wicketkeepers_vars = [players[i] for i in range(n) if data['TYPE'].iloc[i] == 'Wicket-Keeper']
    
    model += lpSum(batsmen_vars) >= min_batsmen
    model += lpSum(bowlers_vars) >= min_bowlers
    model += lpSum(allrounders_vars) >= min_allrounders
    model += lpSum(wicketkeepers_vars) >= min_wicketkeepers
    
    # Nationality constraints (at most max_foreign_players non-Indian players)
    foreign_vars = [players[i] for i in range(n) if data['OverseasIndian'].iloc[i] != 'Indian']
    model += lpSum(foreign_vars) <= max_foreign_players
    
    # Objective function
    total_performance = lpSum(players[i] * player_evaluations[i] for i in range(n))
    model += total_performance
    
    # Optimize the model
    model.solve(PULP_CBC_CMD(msg=0)) # PUlP_CBC_CMD is the default solver, msg=0 suppresses the text output
    
    # Get the selected team
    selected_team = [data.index[i] for i in range(len(players)) if value(players[i]) == 1]

    return selected_team
