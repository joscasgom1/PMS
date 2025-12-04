"""
Main script to compare the time required to fit the extended Random Forest
using the scikit-learn library on the Wine dataset.
"""

from src.extended_scikit import extendedRFscikit             # Extended Random Forest implementation                    # Graphing and visualization utilities
import time                                          # For measuring computation times
from dataset_generation import get_dataset, presets  # Dataset generation utilities
                   




# ============================================================
#                   PARAMETER CONFIGURATION
# ============================================================

# Dataset and scenario selection
dataset = "wine"  
scenario = "6"
setting = "cls"                      # Task type: "reg" = regression, "cls" = classification


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


# ============================================================
#                   MODEL TRAINING PHASE
# ============================================================

f = extendedRFscikit(100, X_train.shape[0], modalities, setting, cfg["impurity"], cfg["tree_depth"], random_state=42)
t1 = time.time()
f.fit_RF(X_train, y_train)  # Fit the extended Random Forest
t2 = time.time()
T_total = (t2 - t1) / 60
print('Total time to fit the extended RF:', T_total, 'minutes')




