import gurobipy as gp
from gurobipy import GRB
import time

def _BMOptimization(error_matrix, costs_v, costs_f, budgets, mod, mod_combination):

    feature_selection_list = []
    total_cost_list = []
    total_error_list = []
    time_list = []

    nr, nc = error_matrix.shape
    nmod = len(mod)

    # Precompute binary mask
    mod_in_comb = {
        (k, j): int(mod[k] in mod_combination[j]) for k in range(nmod) for j in range(nc)
    }

    t1 = time.time()

    # Build model ONCE
    m = gp.Model('gol')
    m.setParam('OutputFlag', 0)

    pi = m.addVars(nr, nc, vtype=GRB.BINARY, name='pi')
    Gamma = m.addVars(nmod, vtype=GRB.BINARY, name='Gamma')

    # Each row selects exactly one combination
    for i in range(nr):
        m.addConstr(gp.quicksum(pi[i, j] for j in range(nc)) == 1)

    # Linking constraints
    for k in range(nmod):
        for j in range(nc):
            if mod_in_comb[(k, j)]:
                for i in range(nr):
                    m.addConstr(pi[i, j] <= Gamma[k])

    # Objective
    m.setObjective(
        gp.quicksum(pi[i, j] * error_matrix[i, j] for i in range(nr) for j in range(nc)),
        GRB.MINIMIZE
    )

    # Budget constraint with dummy RHS
    budget_expr = (
        gp.quicksum(costs_v[j] * pi[i, j] for i in range(nr) for j in range(nc))
        + gp.quicksum(costs_f[k] * Gamma[k] for k in range(nmod))
    )
    budget_constr = m.addConstr(budget_expr <= 0.0, name="Budget_dynamic")

    t2 = time.time()

    
    for budget in budgets:
       
        budget_constr.RHS = budget


        t_start = time.time()
        m.reset()
        m.optimize()
        t_opt = time.time()

        if m.status == GRB.OPTIMAL:
            feature_selection = [
                j
                for i in range(nr)
                for j in range(nc)
                if pi[i, j].X > 0.9
            ]

            total_cost = (
                sum(costs_v[j] * pi[i, j].X for i in range(nr) for j in range(nc))
                + sum(Gamma[k].X * costs_f[k] for k in range(nmod))
            )

            feature_selection_list.append(feature_selection)
            total_cost_list.append(total_cost)
            total_error_list.append(m.ObjVal)
            time_list.append((t_opt - t_start, t2 - t1))
        else:
            print(f'Optimal solution not found for budget {budget}.')
            feature_selection_list.append(None)
            total_cost_list.append(None)
            total_error_list.append(None)
            time_list.append(None)

    m.dispose()   

    return (
        feature_selection_list,
        total_cost_list,
        total_error_list,
        time_list
    )
