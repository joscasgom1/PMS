# Auxiliar functions to representate prescriptions
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import seaborn as sns

def global_prescription(f, X, y, X2, y2, cost, seed = 42):
    """
    Performs the prescription for the global prescription.

    Args:
        f: Customized Random Forest
        X, y: training data
        X2, y2: prescriptive data
        cost: list or array of budget values to test
        xlim: tuple for x-axis limits
        ylim: tuple for y-axis limits
        seed: random seed to reproduce the experiments
    """

    np.random.seed(seed)

    real_cost = []
    error = []
    real_error = []

    for c in cost:
        f.BMOptimization_universal(c, X2.shape[0])

        """
        The result of the global prescription dependes on the selection made.
        In order to mitigate this effect, the result is averaged across 5 posible
        selections.
        """
        r = []
        for _ in range(5):
            p = np.random.permutation(X2.shape[0])
            Xc = X2[p].copy()
            yc = y2[p].copy()
            r.append(f.actualError(Xc, yc))

        real_cost.append(f.total_cost)
        error.append(f.total_error)
        real_error.append(np.mean(r))

    return real_error, error, cost
    
def k_prescription(f, X, y, X2, y2, cost):
    """
    Runs the full prescription, performing the modality selection and calculating real a estimated prediction errors.
    
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
    
    return real_error, error, cost

def prescription_graph_reg(f, X, y, X2, y2, cost, k = 1, xlim=None, ylim=None, ncomb=1, seed = 42):
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
    
    """
    Local prescription evaluation
    """
    k_re, k_e, k_cost = k_prescription(f, X, y, X2, y2, cost)
    
    
    """
    Global prescription evaluation
    """
    g_re, g_e, g_cost = global_prescription(f, X, y, X2, y2, cost, seed = seed)
    
    
    """
    Reference error
    """
    ref_error = np.sum(np.abs(y2 - f.predictRF(X2)))
    
    """
    Graphical representation
    """
    
    plt.figure(figsize=(8, 5))
    
    # K- precription. Marker -> Real error: , Estimated error: 
    plt.plot(k_cost, k_e, marker='s', markersize = 7,linewidth=0.8, linestyle='--', color='black')
    plt.plot(k_cost, k_re, marker='o',markersize = 7,linewidth=0.8, linestyle='--', color='black')
    
    # Global prescription. Marker ->
    plt.plot(g_cost, g_re, marker='^',markersize = 7,linewidth=0.8, linestyle='--', color='black')
  
    # Reference error
    plt.axhline(y=ref_error, color='black', linestyle=':')

    plt.xlabel('Budget')
    plt.ylabel('Error')
    plt.grid(True)

    if xlim:
        plt.xlim(xlim)

    if ylim:
        plt.ylim(ylim)

    


def prescription_graph_cls(f, X, y, X2, y2, cost, k = 1, xlim=None, ylim=None,accuracy_lim =None, ncomb=1, seed = 42):
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
    
    """
    Local prescription evaluation
    """
    k_re, k_e, k_cost = k_prescription(f, X, y, X2, y2, cost)
    
    
    """
    Global prescription evaluation
    """
    g_re, g_e, g_cost = global_prescription(f, X, y, X2, y2, cost, seed = seed)
    
    
    """
    Reference error
    """
    ref_error =  np.mean(f.predictRF(X2) == y2)
    
    """
    Graphical representation
    """
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    # K- precription. Marker -> Real error: , Estimated error: 
    ax1.plot(k_cost, k_e, marker='s', markersize = 7,linewidth=0.8, linestyle='--', color='black')
    ax1.set_xlabel('Budget')
    ax1.set_ylabel('Error')
    ax1.grid(True)
    if ylim:
        ax1.set_ylim(ylim)
    
    ax2 = ax1.twinx()
    ax2.plot(k_cost, k_re, marker='o',markersize = 7,linewidth=0.8, linestyle='--', color='black')
    
    # Global prescription. Marker ->
    ax2.plot(g_cost, g_re, marker='^',markersize = 7,linewidth=0.8, linestyle='--', color='black')
  
    # Reference error
    ax2.axhline(y=ref_error, color='black', linestyle=':')

    ax2.set_ylabel('Accuracy')
    if accuracy_lim:
        ax2.set_ylim(accuracy_lim)

    
    
def heat_map(rf, B, selected_indices=None, universal=False, n = None,regularize = False, p = None):
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
        
        all_counts = []

        for _ in range(5):
            # Permute the feature list
            permuted = random.sample(rf.feature_selection, len(rf.feature_selection))

            # Apply selection AFTER permutation
            if selected_indices is not None:
                permuted = [permuted[i] for i in selected_indices]

            count = {mod.id: 0 for mod in rf.modalities}
            for f in permuted:
                for mod in rf.cat_combination[f]:
                    count[mod] += 1

            # Normalize for this permutation
            n_feats = len(permuted)
            norm_count = {k: v / n_feats for k, v in count.items()}
            all_counts.append(norm_count)
            
        # Average over all normalized counts
        averaged_count = {
            mod.id: np.mean([d[mod.id] for d in all_counts]) for mod in rf.modalities
        }
        return averaged_count

    else:
        rf.BMOptimization(B)
        # No permutation, use features directly
        features = rf.feature_selection
        if selected_indices is not None:
            features = [features[i] for i in selected_indices]

        n_feats = len(features)
        count = {mod.id: 0 for mod in rf.modalities}
        for f in features:
            for mod in rf.cat_combination[f]:
                count[mod] += 1

        count = {k: v / n_feats for k, v in count.items()}
        return count
    
def heat_map_graph(X,X2,cost, rf, fdic, k,universal = False, title='', selected_indices=None, annotate=False, ax = None):
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

    for B in cost:
        diccionario = fdic(rf, B, selected_indices, universal, n=X2.shape[0])
        data[B] = diccionario

    df = pd.DataFrame(data).sort_index()
    df.index = [f"M{i}" for i in df.index]

    # It creates a subplot is none is defined
    if ax is None:
        fig, ax = plt.subplots(figsize=(10.5, 3.5))

    sns.heatmap(df, annot=annotate, fmt=".3f", cmap="Greys", vmin=0, vmax=1, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Budget")



def modality_importance(rf,filename="modality_importance.pdf"):
    """
    Draws a horizontal bar chart using the elements of the frozenset as labels,
    and the second value of each tuple as the bar height.
    
    - data: list of tuples (frozenset, value, ...)
    - title: optional title for the chart
    
    The elements of the frozenset are labeled as M1, M2, etc.
    """
    list_imp = rf.cat_importance_g()
    data = [a for a in list_imp]
    labels = [ "{" + ",".join(f"M{i}" for i in sorted(t[0])) + "}" for t in data ]
    values = [t[1] for t in data]

    sorted_indices = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]

    
    plt.figure(figsize=(8, 6))
    plt.barh(sorted_labels, sorted_values, color='grey')
    plt.xlabel("Modality Combination Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
   
    plt.savefig(filename, format='pdf')
    plt.close()

def importance(rf):
    """
    Draw a horizontal bar chart sorted by importance.
    
    Parameters:
    - importances: array-like with the variable importances (e.g., from model.feature_importances_)
    - feature_names: optional list with the variable names. If not provided, Var0, Var1, ... will be used
    - title: title of the chart (optional)
    """
    
    importances = rf.var_importance()
    feature_names = [f"{i}" for i in range(len(importances))]
    
    
    indices = np.argsort(importances)[::-1]
    sorted_importances = importances[indices]
    sorted_names = [feature_names[i] for i in indices]
    
    # Gráfica
    plt.figure(figsize=(8, 6))
    plt.barh(sorted_names, sorted_importances,color='grey')
    plt.xlabel("Importance")
    plt.gca().invert_yaxis()  
    plt.tight_layout()
    plt.show()
