import gurobipy as gp
from gurobipy import GRB
import time

def _BMOptimization_universal(error_universal, costs_v, costs_f, budgets, cat, cat_combination, n):
    """
    Universal prescriptive selection with multiple budgets.
    """

    feature_selection_list = []
    total_cost_list = []
    total_error_list = []
    time_list = []

    nc = len(error_universal)
    ncat = len(cat)

    # --- Build model ONCE ---
    t1 = time.time()
    m = gp.Model('gol')
    m.setParam('OutputFlag', 0)

    # Variables
    pi = m.addVars(range(nc), vtype=GRB.INTEGER, lb=0, ub=n, name='pi')
    Gamma = m.addVars(range(ncat), vtype=GRB.BINARY, name='Gamma')

    # Exactly n items selected (sum of counts = n)
    m.addConstr(gp.quicksum(pi[j] for j in range(nc)) == n)

    # Linking constraints
    for k in range(ncat):
        for j in range(nc):
            if cat[k] in cat_combination[j]:
                m.addConstr(pi[j] <= n * Gamma[k])

    # Create budget constraint with dummy RHS (0)
    budget_expr = (
        gp.quicksum(costs_v[j] * pi[j] for j in range(nc))
        + gp.quicksum(costs_f[k] * Gamma[k] for k in range(ncat))
    )
    budget_constr = m.addConstr(budget_expr <= 0.0, name="Budget_dynamic")

    # Objective
    m.setObjective(
        gp.quicksum(pi[j] * error_universal[j] for j in range(nc)),
        GRB.MINIMIZE
    )

    t2 = time.time()

    # --- Solve for each budget ---
    for budget in budgets:

        # Update RHS
        budget_constr.RHS = budget


        m.reset()

        t_start = time.time()
        m.optimize()
        t_opt = time.time()

        if m.status == GRB.OPTIMAL:

            # Expand pi[j] selections
            feature_selection = []
            for j in range(nc):
                cnt = int(round(pi[j].X))
                for _ in range(cnt):
                    feature_selection.append(j)

            total_cost = (
                sum(costs_v[j] * pi[j].X for j in range(nc)) +
                sum(costs_f[k] * Gamma[k].X for k in range(ncat))
            )

            feature_selection_list.append(feature_selection)
            total_cost_list.append(total_cost)
            total_error_list.append(m.ObjVal)
            time_list.append((t_opt - t_start, t2 - t1))

        else:
            print(f"Optimal solution not found for budget {budget}.")
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
