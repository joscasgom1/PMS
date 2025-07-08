import gurobipy as gp
from gurobipy import GRB

def _BMOptimization(error_matrix, costs_v, costs_f, budget, cat, cat_combination):
    
    m = gp.Model('gol')
    nr, nc = error_matrix.shape
    ncat = len(cat)
    pi = {}
    pi= m.addVars(nr, nc, vtype=GRB.BINARY, name='pi')
    
    
    Gamma = m.addVars(range(ncat), vtype=GRB.BINARY, name='Gamma')
    
    for i in range(nr):  
        m.addConstr(gp.quicksum(pi[i, j] for j in range(nc)) == 1, name=f"sum_pi_col_{i}")
        
    m.addConstr(gp.quicksum(costs_v[j] * pi[i, j] for i in range(nr) for j in range(nc)) + gp.quicksum(Gamma[j] * costs_f[j] for j in range(ncat)) <= budget, name="sum_Budget_total")        
    
    for k in range(ncat):
        for j in range(nc):
            if cat[k] in cat_combination[j]:
                for i in range(nr):
                    m.addConstr(pi[i,j] <= Gamma[k], name="Fixed_costs")
    
    m.setObjective(gp.quicksum(pi[i, j] * error_matrix[i, j] for i in range(nr) for j in range(nc)), sense=GRB.MINIMIZE)
    
    m.setParam('OutputFlag', 0)
    m.optimize()
    
    if m.status == GRB.OPTIMAL:
        feature_selection = []
        for i in range(nr):
            for j in range(nc):
                if pi[i,j].x > 0.9:
                    feature_selection.append(j)

        total_cost = (
        sum(costs_v[j] * pi[i, j].X for i in range(nr) for j in range(nc))+
        sum(Gamma[j].X * costs_f[j] for j in range(ncat))        
    )
        total_error = m.ObjVal

        return feature_selection, total_cost, total_error
    else:
        print('Optimal solution not found')
    
    m.dispose()