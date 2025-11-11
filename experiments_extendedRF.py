"""
Main script to perform the prescriptive modality selection using an extended Random Forest (RF) 
as predictive model. The script performs both the training phase and the prescriptive analysis.
"""

from src.extendedRF import extendedRF                # Extended Random Forest implementation
import src.graph_crf as graph                        # Graphing and visualization utilities
import time                                          # For measuring computation times
from dataset_generation import get_dataset, presets  # Dataset generation utilities
import matplotlib.pyplot as plt                      # Plotting library




# ============================================================
#                   PARAMETER CONFIGURATION
# ============================================================

# Dataset and scenario selection
dataset = "synthetic_heterogeneous"  # Dataset identifier (see presets for available options)
scenario = "4"                       # Scenario ID within the dataset
setting = "reg"                      # Task type: "reg" = regression, "cls" = classification

# Prescription parameters
k = 1                                 # Number of neighbors used to estimate the individualized prediction error

# ------------------------------------------------------------
# Load configuration preset from the predefined settings.
# Each preset includes impurity criterion, tree depth, cost, limits, etc.
# ------------------------------------------------------------
cfg = presets[dataset]["scenarios"][scenario][setting]
print("impurity criterion:", cfg["impurity"], "\n",
      "tree depth:", cfg["tree_depth"])



# ============================================================
#                   DATASET PREPARATION
# ============================================================

# The get_dataset() function returns:
#   X_train, y_train : training features and labels
#   X_pres, y_pres   : prescription evaluation set
#   modalities       : information about the modalities (for prescriptive problem structure)
X_train, y_train, X_pres, y_pres, modalities = get_dataset(dataset, scenario, setting=setting)


# ============================================================
#                   MODEL INITIALIZATION
# ============================================================

# BMRandomForest arguments:
#   n_trees          : number of Decision Trees in each RF
#   n_samples        : number of training samples used for each DT in each RF
#   modalities       : set of Supplementary Modalities considered. 
#                      Features not included in any Supplementary Modality 
#                      are considered part of the Baseline Modality.
#   setting          : "reg" or "cls"
#   impurity         : criterion for tree splitting (from cfg)
#   tree_depth       : maximum depth of each DT in each RF
#   random_state     : for reproducibility


f = extendedRF(100, X_train.shape[0], modalities, setting, cfg["impurity"], cfg["tree_depth"], random_state=42)


# ============================================================
#                   MODEL TRAINING PHASE
# ============================================================

t1 = time.time()
f.fit_RF(X_train, y_train)  # Fit the extended Random Forest
t2 = time.time()
T_total = (t2 - t1) / 60
print('Total time to fit the extended RF:', T_total, 'minutes')


# ============================================================
#                   PRESCRIPTIVE ANALYSIS
# ============================================================

# The prescription phase evaluates the local and global prescription
# under both regression and classification contexts.
# It produces the main prescription plot and saves it as 'extended_prescription.pdf'.
t1_k1 = time.time()
if setting == "reg":
    graph.prescription_graph_reg(f, X_train, y_train, X_pres, y_pres, cfg["cost"], k , ylim = cfg["ylim"])
    plt.savefig("extended_prescription.pdf", dpi=600)
    plt.close()
else:
    graph.prescription_graph_cls(f, X_train, y_train, X_pres, y_pres, cfg["cost"], k , ylim = cfg["ylim"], accuracy_lim = cfg["accuracy_lim"] )
    plt.savefig("extended_prescription.pdf", dpi=600)
    plt.close()
t2_k1 = time.time()
T_total_k1 = (t2_k1-t1_k1)/60; print('Total time to evaluate prescription for k = 1:', T_total_k1)


# ============================================================
#                   HEATMAP COMPARISON VISUALIZATION
# ============================================================

# Two heatmaps are generated:
#   - Local prescription
#   - Global prescription
# The output is saved as 'extended_hm_comparison.pdf'.
#
# The function 'heat_map_graph' includes the argument 'selected_indices', 
# which can receive a set of indices to represent heatmaps 
# for the corresponding individuals in the prescriptive set.

fig, axes = plt.subplots(2, 1, figsize=(10.5, 7), constrained_layout=True)
graph.heat_map_graph(X_train, X_pres,cfg["cost"], f, graph.heat_map, k, universal = False, title='Local prescription', annotate=True, ax=axes[0])
graph.heat_map_graph(X_train, X_pres,cfg["cost"], f, graph.heat_map, k, universal = True, title='Global prescription', annotate=True, ax=axes[1])
plt.savefig("extended_hm_comparison.pdf", dpi=600)
plt.close()