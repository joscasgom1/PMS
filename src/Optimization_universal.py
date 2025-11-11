import gurobipy as gp
from gurobipy import GRB

def _BMOptimization_universal(error_universal, costs_v, costs_f, budget, cat, cat_combination,n):
    """
    Function to solve the selection of modalities. 
    Parameters:
        - error_universal: matrix with estimation for prediction errors
        - budget : Maximum budget allowed
        - costs_v : variable cost for each Supplementeary Modality
        - cost_f : fixed cost for each Supplementary Modality
        - mod: set of Supplementary Modalities
        - mod_combination: combination of Supplementary Modalities to be considered
        - n: size of prescriptive set
    
    Returns:
        - Selection of Modalities for each individual
        - Total budget spent
        - Total real error
    """
    m = gp.Model('gol')
    nc = len(error_universal)
    ncat = len(cat)
    pi = {}
    pi= m.addVars(range(nc),vtype=GRB.INTEGER, lb=0, ub=n, name='pi')
    
    
    Gamma = m.addVars(range(ncat), vtype=GRB.BINARY, name='Gamma')
    
    # n variables seleccionadas
    m.addConstr(gp.quicksum(pi[j] for j in range(nc)) == n, name="sum_pi_col")
        
    m.addConstr(gp.quicksum(costs_v[j] * pi[j] for j in range(nc)) + gp.quicksum(Gamma[j] * costs_f[j] for j in range(ncat)) <= budget, name="sum_Budget_total")        
    
    for k in range(ncat):
        for j in range(nc):
            if cat[k] in cat_combination[j]:
                m.addConstr(pi[j] <= n*Gamma[k], name="Fixed_costs")
    
    m.setObjective(gp.quicksum(pi[j] * error_universal[j] for j in range(nc)), sense=GRB.MINIMIZE)
    
    m.setParam('OutputFlag', 0)
    m.optimize()
    
    if m.status == GRB.OPTIMAL:
        feature_selection = []        
        for j in range(nc):
            if pi[j].x > 0.9:
                for _ in range(int(pi[j].x)):
                    feature_selection.append(j)

        total_cost = (
        sum(costs_v[j] * pi[j].X for j in range(nc))+
        sum(Gamma[j].X * costs_f[j] for j in range(ncat))        
    )
        total_error = m.ObjVal

        return feature_selection, total_cost, total_error
    else:
        print('Optimal solution not found')
    
    m.dispose()