# Auxiliary Functions for Plotting



### Classic and Modality Combination Importance
import numpy as np
import matplotlib.pyplot as plt

def plot_feature_importances(importances, feature_names=None, title=""):
    """
    Draw a horizontal bar chart sorted by importance.
    
    Parameters:
    - importances: array-like with the variable importances (e.g., from model.feature_importances_)
    - feature_names: optional list with the variable names. If not provided, Var0, Var1, ... will be used
    - title: title of the chart (optional)
    """
    importances = np.array(importances)
    if feature_names is None:
        feature_names = [f"{i}" for i in range(len(importances))]
    
    
    indices = np.argsort(importances)[::-1]
    sorted_importances = importances[indices]
    sorted_names = [feature_names[i] for i in indices]
    
    # Gráfica
    plt.figure(figsize=(8, 6))
    plt.barh(sorted_names, sorted_importances,color='grey')
    plt.xlabel("Importance")
    plt.title(title)
    plt.gca().invert_yaxis()  
    plt.tight_layout()
    plt.show()



def plot_frozenset_bars(data, title=""):
    """
    Draw a horizontal bar chart using the elements of the frozenset as labels,
    and the second value of each tuple as the bar height.
    
    - data: list of tuples (frozenset, value, ...)
    - title: optional title for the chart
    """
    labels = [str(set(t[0])) for t in data]
    values = [t[1] for t in data]

    # Ordenar por valor descendente
    sorted_indices = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]

    # Dibujar gráfica
    plt.figure(figsize=(8, 6))
    plt.barh(sorted_labels, sorted_values, color='grey')
    plt.xlabel("Modality Cobination Importance")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    



import pandas as pd
import seaborn as sns

def heat_map_graph(X,X2,budget_list, rf, modalities, fdic, k,universal = False, title='', selected_indices=None, annotate=False, regularize = False, p = None):
    """
    Generates a heatmap showing the relative importance of modalities
for different budget values (valores_B).

Args:
    - X: train set
    - X2 : prescriptive set
    - valores_B (list): List of budget values to evaluate.
    - rf: Instance of the customized Random Forest model.
    - modalities (list/dict): Set or list of considered modalities.
    - fdic (func): Function that receives (rf, B, modalities, selected_indices)
                               and returns a dictionary {modality: normalized value}.
    - k : number of neighbours used to estimate the prediction error
    - title (str, optional): Title for the plot.
    - selected_indices (list, optional): List of indices to filter selected features.
    - annotate: Boolean to display values in each cell (default False).
    - universal: Boolean that indicates is the prescription is done with the 
                Universal prescription. False by default.
    - regularize: Boolean that indicates is the heatmap in the universal prescription needs to bea averaged
                  across different possible selections
    - p: parameter indicating how many selections are chosen to average the results
Returns:
    None: Displays the heatmap using seaborn.
    """
    rf._error_estimation_RFbased(X, X2, k)
    data = {}
            
    for B in budget_list:
        diccionario, rc = fdic(rf, k, B, modalities, selected_indices, universal, n = X2.shape[0], p = p, regularize= regularize)
        data[rc] = diccionario

    df = pd.DataFrame(data).sort_index()
    df.index = [f"M{i}" for i in df.index]

    plt.figure(figsize=(10.5, 6.8))
    sns.heatmap(df, annot=annotate, fmt=".3f", cmap="Greys", vmin=0, vmax=1)
    plt.title(title)
    plt.xlabel('Budget spent')
    plt.tight_layout()
    plt.show()


def heat_map_values(rf,k, B, modalities, selected_indices=None, universal=None, n = None,regularize = False, p = None):
    """
   Calculates the normalized proportion of selected modalities for a given budget B.

Args:
    - rf: Instance of the customized Random Forest model.
    - B (int/float): Budget for the optimization.
    - modalities (list/dict): Considered modalities.
    - k : number of neighbours used to estimate the prediction error
    - selected_indices (list, optional): Feature indices to filter.
    - universal: Boolean that indicates is the prescription is done with the 
                Universal prescription. False by default.
    - n: individuals in the prescriptive set

Returns:
    dict: Dictionary with the proportion of presence for each modality.
    """
    
    if universal:
        rf.BMOptimization_universal(B, n)
    else:
        rf.BMOptimization(B)

    features = rf.feature_selection

    if universal and regularize:
       
        if p is None:
            raise ValueError("Parameter 'p' must be provided when 'regularize' is True.")

        all_counts = []

        for _ in range(p):
            # Permute the feature list
            permuted = random.sample(features, len(features))

            # Apply selection AFTER permutation
            if selected_indices is not None:
                permuted = [permuted[i] for i in selected_indices]

            count = {mod: 0 for mod in modalities}
            for f in permuted:
                for mod in rf.cat_combination[f]:
                    count[mod] += 1

            # Normalize for this permutation
            n_feats = len(permuted)
            norm_count = {k: v / n_feats for k, v in count.items()}
            all_counts.append(norm_count)

        # Average over all normalized counts
        averaged_count = {
            mod: np.mean([d[mod] for d in all_counts]) for mod in modalities
        }
        return averaged_count, rf.total_cost

    else:
        
        # No permutation, use features directly
        if selected_indices is not None:
            features = [features[i] for i in selected_indices]

        n_feats = len(features)
        count = {mod: 0 for mod in modalities}
        for f in features:
            for mod in rf.cat_combination[f]:
                count[mod] += 1

        count = {k: v / n_feats for k, v in count.items()}
        return count, rf.total_cost
    
    

def indices_repetidas_con_condicion_total(arr, cond, precision=6):
    from collections import defaultdict

    grupos = defaultdict(list)
    for idx, fila in enumerate(arr):
        fila_str = ",".join(f"{x:.{precision}f}" for x in fila)
        grupos[fila_str].append(idx)

    resultado = []
    for indices in grupos.values():
        conds = {cond[i] for i in indices}
        if len(conds) > 1:
            resultado.append(indices)
    return resultado


import random

def permutar_aleatorio_por_grupos(base_list, grupos, n):
    """
    Generates n random permutations of base_list respecting the exchange groups.
    
    Parameters:
        base_list (list): Base list to permute.
        groups (list[list[int]]): Indices that can be swapped among themselves.
        n (int): Number of random permutations to generate.
    
    Returns:
        list[list]: List of valid random permutations.
    """
    permutaciones = set()
    intentos = 0
    max_intentos = n * 10  # para evitar bucles infinitos si hay pocas permutaciones posibles

    while len(permutaciones) < n and intentos < max_intentos:
        copia = base_list[:]
        for grupo in grupos:
            valores = [base_list[i] for i in grupo]
            random.shuffle(valores)
            for idx, i in enumerate(grupo):
                copia[i] = valores[idx]
        permutaciones.add(tuple(copia))
        intentos += 1

    return [list(p) for p in permutaciones]


def plot_frozenset_bars_labeled(data, title=""):
    """
    Draws a horizontal bar chart using the elements of the frozenset as labels,
    and the second value of each tuple as the bar height.
    
    - data: list of tuples (frozenset, value, ...)
    - title: optional title for the chart
    
    The elements of the frozenset are labeled as M1, M2, etc.
    """
    labels = [ "{" + ",".join(f"M{i}" for i in sorted(t[0])) + "}" for t in data ]
    values = [t[1] for t in data]

    sorted_indices = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]

    
    plt.figure(figsize=(8, 6))
    plt.barh(sorted_labels, sorted_values, color='grey')
    plt.xlabel("Modality Combination Importance")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


# Estimated and Real error functions 

def evaluate_prescription_strategy(f, X, y, X2, y2,k,ncomb=1, cost=range(0, 600, 50), xlim=None, ylim=None):
    """
    Runs the full prescription evaluation pipeline:
    - Precomputes OOB info
    - Estimates error
    - Performs modality selection
    - Compares estimated vs. real error
    - Plots results
    
    Args:
        f: Customized Random Forest
        X, y: training data
        X2, y2: Prescriptive data
        k: number of neighbors used to estimate the prediction error
        costs: Allowed budget values for which the prescription problem is solved
        xlim: tuple (xmin, xmax) for x-axis limits
        ylim: tuple (ymin, ymax) for y-axis limits
        ncomb: fraction indicating the threshold for importance in the modality combinations considered during prescription.
    """

    f._precompute_oob_mapping()
    f.error_estimation_oob(X, y, ncomb)
    f._error_estimation_RFbased(X, X2, k)
    f.cat_comb_importance()

    real_cost = []
    error = []
    real_error = []
    

    for c in cost:
        f.BMOptimization(c)

        Xc = X2.copy()
        for i in range(len(f.feature_selection)):
            Xc[i, f.null_feat[f.feature_selection[i]]] = np.nan

        
        real_cost.append(f.total_cost)
        error.append(f.total_error)
        real_error.append(f.actualError(X2, y2))

    ref_error = np.sum(np.abs(y2 - f.predictRF(X2)))

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.scatter(real_cost, error, color='blue', label='Estimated error')
    plt.plot(real_cost, error, linestyle='--', color='gray', alpha=0.5)
    plt.scatter(real_cost, real_error, color='red', label='Real error')
    plt.plot(real_cost, real_error, linestyle='--', color='red', alpha=0.5)
    plt.axhline(y=ref_error, linestyle='--', color='green', alpha=0.7)

    plt.xlabel('Budget spent')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)

    if xlim:
        plt.xlim(xlim)
        ticks = plt.xticks()[0]
        plt.xticks([tick for tick in ticks if tick >= 0])
    if ylim:
        plt.ylim(ylim)

    plt.show()

    return {
        'cost': real_cost,
        'estimated_error': error,
        'real_error': real_error
    }

# Estimated and real errors for the Universal Prescription, averaged over 5 different selections
def evaluate_prescription_strategy_universal(f, X, y, X2, y2, cost = range(0,600,50), xlim=None, ylim=None):
    """
    Adapted evaluation using BMOptimization_universal and averaging over 5 permutations.

    Args:
        f: Customized Random Forest
        X, y: training data
        X2, y2: prescriptive data
        cost: list or array of budget values to test
        xlim: tuple for x-axis limits
        ylim: tuple for y-axis limits
    """

    np.random.seed(42)

    real_cost = []
    error = []
    real_error = []

    for c in cost:
        f.BMOptimization_universal(c, X2.shape[0])

        r = []
        for _ in range(5):
            p = np.random.permutation(X2.shape[0])
            Xc = X2[p].copy()
            yc = y2[p].copy()
            r.append(f.actualError(Xc, yc))

        real_cost.append(f.total_cost)
        error.append(f.total_error)
        real_error.append(np.mean(r))

    ref_error = np.sum(np.abs(y2 - f.predictRF(X2)))

    plt.figure(figsize=(8, 5))
    plt.scatter(real_cost, error, color='blue', label='Estimated error')
    plt.plot(real_cost, error, linestyle='--', color='gray', alpha=0.5)
    plt.scatter(real_cost, real_error, color='red', label='Real error')
    plt.plot(real_cost, real_error, linestyle='--', color='red', alpha=0.5)
    plt.axhline(y=ref_error, linestyle='--', color='green', alpha=0.7)

    plt.xlabel('Budget spent')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)

    if xlim is not None:
        plt.xlim(xlim)
        ticks = plt.xticks()[0]
        plt.xticks([tick for tick in ticks if tick >= 0])
    if ylim is not None:
        plt.ylim(ylim)

    plt.show()

    return {
        'cost': real_cost,
        'estimated_error': error,
        'real_error': real_error
    }

# Comparison of results for different values of k
def comparison_kvalues(k, cost, colors, X1, X2, y1, y2, f, ylim=None):
    """
    Compare the Universal Prescription and different k-NN based strategies over a range of budgets.
    
    Args:
        k (list or array): values of k for k-NN error estimation
        cost (list or array): budget values to evaluate
        colors (list): list of colors for plotting each curve
        X1, y1: training data
        X2, y2: prescriptive data
        f: Customized Random Forest instance with required methods
        
    Returns:
        None, but plots the comparison graph.
    """
    n, m = len(k) + 1, len(cost)
    real_cost = np.zeros((n, m))
    error = np.zeros((n, m))
    real_error = np.zeros((n, m))

    # For each k value, estimate errors and optimize
    for j in range(n - 1):
        f._error_estimation_RFbased(X1, X2, k[j])
        for i in range(m):
            f.BMOptimization(cost[i])
            real_cost[j, i] = f.total_cost
            error[j, i] = f.total_error
            real_error[j, i] = f.actualError(X2, y2)

    # Universal Prescription (last row)
    np.random.seed(42)
    for i in range(m):
        f.BMOptimization_universal(cost[i], X2.shape[0])
        r = np.zeros(5)
        for j in range(5):
            p = np.random.permutation(X2.shape[0])
            Xc = X2[p].copy()
            yc = y2[p].copy()
            r[j] = f.actualError(Xc, yc)
        real_cost[n - 1, i] = f.total_cost
        error[n - 1, i] = f.total_error
        real_error[n - 1, i] = np.mean(r)

    ref_error = np.sum(np.abs(y2 - f.predictRF(X2)))

    plt.figure(figsize=(8, 5))

    for i in range(n):
        label = "Universal prescription" if i == n - 1 else f"k = {k[i]}"
        print(label)
        plt.scatter(real_cost[i, :], real_error[i, :], color=colors[i], label=label)
        plt.plot(real_cost[i, :], real_error[i, :], linestyle='--', color=colors[i], alpha=0.5)

    plt.axhline(y=ref_error, linestyle='--', color='green', alpha=0.7)

    plt.xlabel('Budget spent')
    plt.ylabel('Error')
    plt.legend()
    if ylim is not None:
        plt.ylim(ylim)
    plt.grid(True)
    plt.show()
    
# Estimated and Real error functions 
def evaluate_prescription_strategy_classification_hom(relevant_features,mean, covariance, size, model,y_threshold, X,y, k,costs, use_regularization=False, n_sets=5, ncomb = 1):
    """
    Evaluates a model over different cost budgets and visualizes estimated error and real accuracy.

    Args:
        - relevant_features: features used to generate the variable response
        - mean: mean of the distribution to generate the prescriptive set
        - covariance: covariance matrix to generate the prescriptive set
        - size = size of the prescriptive set
        - y_threshold: threshold for the response variable
        - model: Instance of the customized Random Forest model
        - X: Training data
        - y: Variable response for training set
        - k: Number of neighbors used in error estimation
        - costs: list of costs
        - use_regularization: Whether to average over multiple randomized test sets
        - n_sets: Number of randomized sets to average over
        - ncomb: fraction indicating the threshold for importance in the modality combinations considered during prescription.

    Returns:
        None. Displays a dual-axis plot of cost vs. estimated error and real accuracy.
    """

    model._precompute_oob_mapping()
    model.error_estimation_oob(X, y, ncomb)


    m = len(costs)
    n = 1
    if use_regularization:
        seeds = list(np.random.randint(0, 10000, size=n_sets))
        n = len(seeds)

    real_cost_m = np.zeros((n, m))
    error_m = np.zeros((n, m))
    real_error_m = np.zeros((n, m))
    accuracies_m = np.zeros((n, m))
    ref_accuracies_m = np.zeros(n)


    for s in range(n):
        if use_regularization:
            seed = int(seeds[s])
            np.random.seed(seed)
            random.seed(seed)

        X2 = np.random.multivariate_normal(mean, covariance, size=size)
        noise = np.random.randn(size)
        y2 = np.sum(X2[:, relevant_features], axis=1) + noise
        y2 = (y2 > y_threshold).astype(int)

        model._error_estimation_RFbased(X, X2, k)
        ref_accuracies_m[s] = np.mean(model.predictRF(X2) == y2)

        for j, c in enumerate(costs):
            model.BMOptimization(c)
            Xc = X2.copy()
            for i in range(len(model.feature_selection)):
                Xc[i, model.null_feat[model.feature_selection[i]]] = np.nan
            yc = model.predictRF(Xc)
            acc = np.mean(yc == y2)

            accuracies_m[s, j] = acc
            real_cost_m[s, j] = model.total_cost
            error_m[s, j] = model.total_error
            real_error_m[s, j] = model.actualError(X2, y2)

    ref_accuracy = np.mean(ref_accuracies_m)
    real_accuracy = np.mean(accuracies_m, axis=0)
    real_cost = np.mean(real_cost_m, axis=0)
    error = np.mean(error_m, axis=0)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.scatter(real_cost, error, color='blue', label='Estimated error')
    ax1.plot(real_cost, error, linestyle='--', color='gray', alpha=0.5)
    ax1.set_xlabel('Budget spent')
    ax1.set_ylabel('Error')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.scatter(real_cost, real_accuracy, color='red', label='Real accuracy')
    ax2.plot(real_cost, real_accuracy, linestyle='--', color='red', alpha=0.5)
    ax2.axhline(y=ref_accuracy, linestyle='--', color='green', alpha=0.7)
    ax2.set_ylabel('Real accuracy')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)

    plt.tight_layout()
    plt.show()


def evaluate_prescription_strategy_classification_het(relevant_features, mean, covariance, size, model, y_threshold, X, y, k, costs, use_regularization=False, n_sets=5, ncomb = 1):
    """
    Evaluates a model over different cost budgets and visualizes estimated error and real accuracy.

    Args:
        - relevant_features: lists of features used to generate the variable response
        - mean: lists of mean of the distribution to generate the prescriptive set
        - covariance: lists of covariance matrix to generate the prescriptive set
        - size: lists of sizes of the prescriptive set
        - y_threshold: lists of threshold for the response variable
        - model: Instance of the customized Random Forest model
        - X: Training data
        - y: Variable response for training set
        - k: Number of neighbors used in error estimation
        - costs: list of costs
        - use_regularization: Whether to average over multiple randomized test sets
        - n_sets: Number of randomized sets to average over
        - ncomb: fraction indicating the threshold for importance in the modality combinations considered during prescription.

    Returns:
        None. Displays a dual-axis plot of cost vs. estimated error and real accuracy.
    """

    model._precompute_oob_mapping()
    model.error_estimation_oob(X, y,ncomb)

    m = len(costs)
    n = 1
    if use_regularization:
        seeds = list(np.random.randint(0, 10000, size=n_sets))
        n = len(seeds)

    real_cost_m = np.zeros((n, m))
    error_m = np.zeros((n, m))
    real_error_m = np.zeros((n, m))
    accuracies_m = np.zeros((n, m))
    ref_accuracies_m = np.zeros(n)

    for s in range(n):
        if use_regularization:
            seed = int(seeds[s])
            np.random.seed(seed)
            random.seed(seed)

        # Listas para acumular datos de subpoblaciones
        X2_list = []
        y2_list = []

        for sp in range(len(size)):
            X2_sp = np.random.multivariate_normal(mean[sp], covariance[sp], size=size[sp])
            noise = np.random.randn(size[sp])
            y2_sp = X2_sp@relevant_features[sp] + noise
            y2_sp = (y2_sp > y_threshold[sp]).astype(int)

            X2_list.append(X2_sp)
            y2_list.append(y2_sp)

        # Apilar verticalmente todas las subpoblaciones
        X2 = np.vstack(X2_list)
        y2 = np.concatenate(y2_list)

        # Evaluación con el modelo
        model._error_estimation_RFbased(X, X2, k)
        ref_accuracies_m[s] = np.mean(model.predictRF(X2) == y2)

        for j, c in enumerate(costs):
            model.BMOptimization(c)
            Xc = X2.copy()
            for i in range(len(model.feature_selection)):
                Xc[i, model.null_feat[model.feature_selection[i]]] = np.nan
            yc = model.predictRF(Xc)
            acc = np.mean(yc == y2)

            accuracies_m[s, j] = acc
            real_cost_m[s, j] = model.total_cost
            error_m[s, j] = model.total_error
            real_error_m[s, j] = model.actualError(X2, y2)

    ref_accuracy = np.mean(ref_accuracies_m)
    real_accuracy = np.mean(accuracies_m, axis=0)
    real_cost = np.mean(real_cost_m, axis=0)
    error = np.mean(error_m, axis=0)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.scatter(real_cost, error, color='blue', label='Estimated error')
    ax1.plot(real_cost, error, linestyle='--', color='gray', alpha=0.5)
    ax1.set_xlabel('Budget spent')
    ax1.set_ylabel('Error')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.scatter(real_cost, real_accuracy, color='red', label='Real accuracy')
    ax2.plot(real_cost, real_accuracy, linestyle='--', color='red', alpha=0.5)
    ax2.axhline(y=ref_accuracy, linestyle='--', color='green', alpha=0.7)
    ax2.set_ylabel('Real accuracy')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)

    plt.tight_layout()
    plt.show()
