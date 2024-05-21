import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value

def optimize_team(data, price_cap, number_players, min_batsmen, min_bowlers, max_foreign_players, player_evaluations):
    # Create model
    model = LpProblem(name="Fantasy_Team_Optimization", sense=LpMaximize)
    
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
    total_price_constraint = lpSum(players * data['Price']) <= price_cap
    model += total_price_constraint
    
    # Position constraints (at least two batsmen, at least two bowlers)
    batsmen_vars = [players[i] for i in range(n) if data['TYPE'][i] == 'Batsman']
    bowlers_vars = [players[i] for i in range(n) if data['TYPE'][i] == 'Bowler']
    
    model += lpSum(batsmen_vars) >= min_batsmen
    model += lpSum(bowlers_vars) >= min_bowlers
    
    # Nationality constraints (at most max_foreign_players non-Indian players)
    foreign_vars = [players[i] for i in range(n) if data['OverseasIndian'][i] != 'Indian']
    model += lpSum(foreign_vars) <= max_foreign_players
    
    # Objective function
    total_performance = lpSum(players[i] * player_evaluations[i] for i in range(len(player_evaluations)))
    model += total_performance
    
    # Optimize the model
    model.solve()
    
    # Get the selected team
    selected_team = [data['PLAYER'][i] for i in range(len(players)) if value(players[i]) == 1]
    
    return selected_team
