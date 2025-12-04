"""
Main script to perform the prescriptive modality selection using a Random Forest (RF) 
with customized Decision Tree Structures (DTs) as predictive model. 
The script performs both the training phase and the prescriptive analysis.
"""

from src.compactRF import compactRF         # Compact Random Forest implementation
import src.graph_BMRF as graph               # Graphing and visualization utilities
import time                                  # For measuring computation times
from dataset_generation import get_dataset, presets  # Dataset generation utilities
import matplotlib.pyplot as plt              # Plotting library


# ============================================================
#                   PARAMETER CONFIGURATION
# ============================================================

# Dataset and scenario selection
dataset = "wine"  # Dataset identifier (see presets for available options)
scenario = "6"                       # Scenario ID within the dataset
setting = "cls"                      # Task type: "reg" = regression, "cls" = classification

# Prescription parameters
k = 1                             # Number of neighbors used to estimate the individualized prediction error
nc = 1.1                            # Fraction of total modality-combinations considered in the prescription resolution

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
#   n_trees          : number of Decision Trees in the forest (100 here)
#   n_samples        : number of training samples used for each DT
#   modalities       : set of Supplementary Modalities considered. 
#                      Features not included in any Supplementary Modality 
#                      are considered part of the Baseline Modality.
#   setting          : "reg" or "cls"
#   impurity         : criterion for tree splitting (from cfg)
#   tree_depth       : maximum depth of each tree
#   random_state     : for reproducibility
f = compactRF(
     100, X_train.shape[0], modalities, setting,
     cfg["impurity"], cfg["tree_depth"], random_state=42 )


# ============================================================
#                   MODEL TRAINING PHASE
# ============================================================

t1 = time.time()
f.fit_RF(X_train, y_train)  # Fit the compact Random Forest
t2 = time.time()
T_total = (t2 - t1) / 60
print('Total time to fit the compact RF:', T_total, 'minutes')


# ============================================================
#                   PRESCRIPTIVE ANALYSIS
# ============================================================
# Generate two heatmaps for the selection:
#   - Local prescription
#   - Global prescription
# The heatmaps are saved as 'hm_comparison.pdf'.
# Optionally, the selection can be segmented for subpopulations.
# The prediction phase evaluates both local and global prescriptions
# for regression and classification tasks.
# The prediction plot is generated and saved as 'prescription.pdf'.

if setting == "reg":
    # Regression prescription: continuous outcomes
    graph.prescription_graph_reg(
        f, X_train, y_train, X_pres, y_pres,
        cfg["cost"], k, ylim=cfg["ylim"], ncomb=nc
    )
else:
    # Classification prescription: categorical outcomes
    graph.prescription_graph_cls(
        f, X_train, y_train, X_pres, y_pres,
        cfg["cost"], k, ylim=cfg["ylim"],
        accuracy_lim=cfg["accuracy_lim"], ncomb=nc
    )

plt.savefig("prescription.pdf", dpi=600)
plt.close()


#=================================================================
#                 Modality Importance Calculation
#=================================================================
# Computes the modality importance according to the criterion defined
#
# To generate the plot, run:
#     f.modality_importance()
#
# The resulting figure will be saved as 'modality_importance.pdf'.


