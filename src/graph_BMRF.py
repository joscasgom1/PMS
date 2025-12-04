# Auxiliar functions to representate prescriptions
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import seaborn as sns
import time

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
    f.BMOptimization_universal(cost, X2.shape[0])
    
    for c in range(len(cost)):
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
            for i in range(len(f.feature_selection[c])):
                Xc[i, f.null_feat[f.feature_selection[c][i]]] = np.nan
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
    """

    real_cost = []
    error = []
    real_error = []
    times = []
    
    f.BMOptimization(cost)

    for c in range(len(cost)):
        Xc = X2.copy()
        for i in range(len(f.feature_selection[c])):
            Xc[i, f.null_feat[f.feature_selection[c][i]]] = np.nan
        
        real_cost.append(f.total_cost[c])
        error.append(f.total_error[c])
        real_error.append(f.actualError(Xc, y2))
        
    times.append(f.times)
    return real_error, error, cost, times

def prescription_graph_reg(f, X, y, X2, y2, cost, k = 1, xlim=None, ylim=None, selected_indices=None, ncomb=1, seed = 42):
    print('llamada')
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
    t_1_estimate = time.time()
    f.error_estimation_oob(X, y, ncomb)
    t_2_estimate = time.time()
    print('Error train estimate:',t_2_estimate-t_1_estimate)
    f._error_estimation_RFbased(X, X2, k)    
    
    
    """
    Local prescription evaluation
    """

    k_re, k_e, k_cost, k_t = k_prescription(f, X, y, X2, y2, cost)
    print('prescription times:',k_t)
    d1 = heatmap_calculate_local(f, cost)
    if selected_indices is not None:
        l_selection = []
        for ind in selected_indices:
            l_selection.append(heatmap_calculate_local(f, cost,selected_indices=ind))
    """
    Global prescription evaluation
    """
    
    g_re, g_e, g_cost = global_prescription(f, X, y, X2, y2, cost, seed = seed)
    d2 = heatmap_calculate_global(f, cost)
    if selected_indices is not None:
        g_selection = []
        for ind in selected_indices:
            g_selection.append(heatmap_calculate_global(f, cost,selected_indices=ind))
   
    plot_two_heatmaps(d1, d2, cost, title1="Local prescription", title2="Global prescription", annotate=True)
    
    if selected_indices is not None:
        for i in range(len(selected_indices)):
            plot_two_heatmaps(l_selection[i], g_selection[i], cost, title1="Local prescription", title2="Global prescription", annotate=True, save_index=i)
    
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

    


def prescription_graph_cls(f, X, y, X2, y2, cost, k = 1, xlim=None, ylim=None,accuracy_lim =None,selected_indices=None, ncomb=1, seed = 42):
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
    t_1_estimate = time.time()
    f.error_estimation_oob(X, y, ncomb)
    t_2_estimate = time.time()
    print('Error train estimate:',t_2_estimate-t_1_estimate)
    f._error_estimation_RFbased(X, X2, k)
    f.cat_comb_importance()
    
    """
    Local prescription evaluation
    """
    k_re, k_e, k_cost, k_t = k_prescription(f, X, y, X2, y2, cost)
    print('prescription times:',k_t)
    d1 = heatmap_calculate_local(f, cost)
    if selected_indices is not None:
        l_selection = []
        for ind in selected_indices:
            l_selection.append(heatmap_calculate_local(f, cost,selected_indices=ind))
    """
    Global prescription evaluation
    """
    
    g_re, g_e, g_cost = global_prescription(f, X, y, X2, y2, cost, seed = seed)
    d2 = heatmap_calculate_global(f, cost)
    if selected_indices is not None:
        g_selection = []
        for ind in selected_indices:
            g_selection.append(heatmap_calculate_global(f, cost,selected_indices=ind))
   
    plot_two_heatmaps(d1, d2, cost, title1="Local prescription", title2="Global prescription", annotate=True)
    
    if selected_indices is not None:
        for i in range(len(selected_indices)):
            plot_two_heatmaps(l_selection[i], g_selection[i], cost, title1="Local prescription", title2="Global prescription", annotate=True, save_index=i)
    
    
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
    
def heatmap_calculate_local(f,costs,selected_indices = None):
    data_local = {}
    for c in range(len(costs)):
        features = f.feature_selection[c]
        if selected_indices is not None:
            features = [features[i] for i in selected_indices]
        n_feats = len(features)
        count = {mod.id: 0 for mod in f.modalities}
        for feat in features:
            for mod in f.cat_combination[feat]:
                count[mod] += 1
        count = {k: v / n_feats for k, v in count.items()}
        data_local[c] = count
    return data_local    


def heatmap_calculate_global(f,costs,selected_indices = None):
    data_global = {}
    
    for c in range(len(costs)):
        all_counts = []
        for _ in range(5):
            # Permute the feature list
            permuted = random.sample(f.feature_selection[c], len(f.feature_selection[c]))    
            # Apply selection AFTER permutation
            if selected_indices is not None:
                permuted = [permuted[i] for i in selected_indices]    
            count = {mod.id: 0 for mod in f.modalities}
            for perm in permuted:
                for mod in f.cat_combination[perm]:
                    count[mod] += 1
    

            n_feats = len(permuted)
            norm_count = {k: v / n_feats for k, v in count.items()}
            all_counts.append(norm_count)
            
        # Average over all normalized counts
        averaged_count = {
            mod.id: np.mean([d[mod.id] for d in all_counts]) for mod in f.modalities
        }
        data_global[c] = averaged_count
    return data_global

def plot_two_heatmaps(data1, data2, costs, title1="Local prescription", title2="Global prescription", annotate=True, save_index=None):
    

    df1 = pd.DataFrame(data1).sort_index()
    df2 = pd.DataFrame(data2).sort_index()
    
    
    df1.index = [f"M{i}" for i in df1.index]
    df2.index = [f"M{i}" for i in df2.index]

    df1.columns = costs
    df2.columns = costs
    # Create figure with two subplots 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 7), constrained_layout=True)

    # Upper heatmap
    sns.heatmap(df1, annot=annotate, fmt=".3f", cmap="Greys",
                vmin=0, vmax=1, ax=ax1)
    ax1.set_title(title1)
    ax1.set_xlabel("Budget")

    # Lower heatmap
    sns.heatmap(df2, annot=annotate, fmt=".3f", cmap="Greys",
                vmin=0, vmax=1, ax=ax2)
    ax2.set_title(title2)
    ax2.set_xlabel("Budget")
    if save_index is None:
        save_path = "hm_comparison.pdf"
    else:
        save_path = f"hm_comparison_{save_index}.pdf"
    # Save and close exactly como antes
    plt.savefig(save_path, dpi=600)
    plt.close()