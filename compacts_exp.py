"""
Main script to perform the prescriptive modality selection using a Random Forest (RF) 
with customized Decision Tree Structures (DTs) as predictive model. 
The script performs both the training phase and the prescriptive analysis.

Results are averaged over N_SEEDS independent train/prescriptive splits
to provide more robust estimates of ELPE, RLPE and RGPE.
"""

from src.compactRF import compactRF
import src.graph_BMRF as graph
import time
import numpy as np
import matplotlib.pyplot as plt
from dataset_generation import get_dataset, presets


# ============================================================
#                   PARAMETER CONFIGURATION
# ============================================================

dataset  = "nhanes_cardio"
scenario = "8"
setting  = "cls"

k      = 30  # Number of neighbors for individualized error estimation
nc     = 1.1  # Fraction of modality combinations considered in prescription

N_SEEDS = 10                        # Number of independent splits to average over
SEEDS   = [44, 123, 256, 512, 999, 42, 67, 9, 13, 7] # Fixed seeds for reproducibility

cfg  = presets[dataset]["scenarios"][scenario][setting]
cost = list(cfg["cost"])

print(f"Dataset: {dataset} | Scenario: {scenario} | Setting: {setting}")
print(f"Impurity: {cfg['impurity']} | Tree depth: {cfg['tree_depth']}")
print(f"Averaging over {N_SEEDS} seeds: {SEEDS}\n")


# ============================================================
#             MULTI-SEED EXPERIMENT LOOP
# ============================================================

all_k_re       = []   # RLPE  (local,  real error)
all_k_e        = []   # ELPE  (local,  estimated error)
all_g_re       = []   # RGPE  (global, real error)
all_ref_error  = []   # Reference accuracy/error
all_hm_local   = []   # heatmap data local
all_hm_global  = []   # heatmap data global
all_hm_local_under_40 = []
all_hm_local_40_60 = []
all_hm_local_over_60 = []
all_hm_global_under_40 = []
all_hm_global_40_60 = []
all_hm_global_over_60 = []

for i, seed in enumerate(SEEDS):
    print(f"── Seed {seed} ({i+1}/{N_SEEDS}) ──")

    # Different train/pres split each seed
    X_train, y_train, X_pres, y_pres, modalities = get_dataset(
        dataset, scenario, setting=setting, seed=seed
    )
    
    # Age groups indices (assuming age is column 0)
    idx_under_40 = np.where(X_pres[:, 0] < 40)[0]
    idx_40_60 = np.where((X_pres[:, 0] >= 40) & (X_pres[:, 0] <= 60))[0]
    idx_over_60 = np.where(X_pres[:, 0] > 60)[0]

    f = compactRF(
        100, X_train.shape[0], modalities, setting,
        cfg["impurity"], cfg["tree_depth"], random_state=42
    )

    t1 = time.time()
    f.fit_RF(X_train, y_train)
    print(f"   Fit: {(time.time()-t1)/60:.2f} min")

    f._precompute_oob_mapping()
    f.error_estimation_oob(X_train, y_train, nc)
    f._error_estimation_RFbased(X_train, X_pres, k)
    if setting == "cls":
        f.cat_comb_importance()

    k_re, k_e, _, k_t = graph.k_prescription(f, X_train, y_train, X_pres, y_pres, cost)
    print(f"   Prescription times: {k_t}")
    # d1 must be calculated HERE — k_prescription sets f.feature_selection to LOCAL.
    # global_prescription will overwrite f.feature_selection with GLOBAL.
    d1 = graph.heatmap_calculate_local(f, cost)
    d1_under_40 = graph.heatmap_calculate_local(f, cost, selected_indices=idx_under_40)
    d1_40_60 = graph.heatmap_calculate_local(f, cost, selected_indices=idx_40_60)
    d1_over_60 = graph.heatmap_calculate_local(f, cost, selected_indices=idx_over_60)

    g_re, _, _ = graph.global_prescription(f, X_train, y_train, X_pres, y_pres, cost, seed=seed)
    # d2 after global_prescription, which has set f.feature_selection to GLOBAL.
    d2 = graph.heatmap_calculate_global(f, cost)
    d2_under_40 = graph.heatmap_calculate_global(f, cost, selected_indices=idx_under_40)
    d2_40_60 = graph.heatmap_calculate_global(f, cost, selected_indices=idx_40_60)
    d2_over_60 = graph.heatmap_calculate_global(f, cost, selected_indices=idx_over_60)

    all_k_re.append(k_re)
    all_k_e.append(k_e)
    all_g_re.append(g_re)
    
    if setting == "cls":
        all_ref_error.append(np.mean(f.predictRF(X_pres) == y_pres))
    else:
        all_ref_error.append(np.sum(np.abs(y_pres - f.predictRF(X_pres))))

    all_hm_local.append(d1)
    all_hm_global.append(d2)
    all_hm_local_under_40.append(d1_under_40)
    all_hm_local_40_60.append(d1_40_60)
    all_hm_local_over_60.append(d1_over_60)
    all_hm_global_under_40.append(d2_under_40)
    all_hm_global_40_60.append(d2_40_60)
    all_hm_global_over_60.append(d2_over_60)

    print(f"   Done.\n")


# ============================================================
#             AVERAGE RESULTS ACROSS SEEDS
# ============================================================

mean_k_re = np.mean(all_k_re, axis=0)
mean_k_e  = np.mean(all_k_e,  axis=0)
mean_g_re = np.mean(all_g_re, axis=0)

# Average heatmaps across splits
mod_ids = [mod.id for mod in modalities]

def average_heatmaps(all_hm):
    avg = {}
    for c in range(len(cost)):
        avg[c] = {
            mid: np.mean([all_hm[s][c][mid] for s in range(N_SEEDS)])
            for mid in mod_ids
        }
    return avg

avg_hm_local  = average_heatmaps(all_hm_local)
avg_hm_global = average_heatmaps(all_hm_global)

avg_hm_local_under_40 = average_heatmaps(all_hm_local_under_40)
avg_hm_local_40_60 = average_heatmaps(all_hm_local_40_60)
avg_hm_local_over_60 = average_heatmaps(all_hm_local_over_60)

avg_hm_global_under_40 = average_heatmaps(all_hm_global_under_40)
avg_hm_global_40_60 = average_heatmaps(all_hm_global_40_60)
avg_hm_global_over_60 = average_heatmaps(all_hm_global_over_60)

# ============================================================
#             HEATMAP  (averaged across splits)
# ============================================================

graph.plot_two_heatmaps(
    avg_hm_local, avg_hm_global, cost,
    title1="Local prescription",
    title2="Global prescription",
    annotate=True
)


# ============================================================
#             PRESCRIPTION PLOT  (averaged, original format)
# ============================================================

ref_error = np.mean(all_ref_error)

if setting == "cls":
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(cost, mean_k_e, marker='s', markersize=7,
             linewidth=0.8, linestyle='--', color='black')
    ax1.set_xlabel('Budget')
    ax1.set_ylabel('Error')
    ax1.grid(True)
    if cfg.get("ylim"):
        ax1.set_ylim(cfg["ylim"])

    ax2 = ax1.twinx()
    ax2.plot(cost, mean_k_re, marker='o', markersize=7,
             linewidth=0.8, linestyle='--', color='black')
    ax2.plot(cost, mean_g_re, marker='^', markersize=7,
             linewidth=0.8, linestyle='--', color='black')
    ax2.axhline(y=ref_error, color='black', linestyle=':')
    ax2.set_ylabel('Accuracy')
    if cfg.get("accuracy_lim"):
        ax2.set_ylim(cfg["accuracy_lim"])

else:  # reg
    plt.figure(figsize=(8, 5))
    plt.plot(cost, mean_k_e,  marker='s', markersize=7,
             linewidth=0.8, linestyle='--', color='black')
    plt.plot(cost, mean_k_re, marker='o', markersize=7,
             linewidth=0.8, linestyle='--', color='black')
    plt.plot(cost, mean_g_re, marker='^', markersize=7,
             linewidth=0.8, linestyle='--', color='black')
    plt.axhline(y=ref_error, color='black', linestyle=':')
    plt.xlabel('Budget')
    plt.ylabel('Error')
    plt.grid(True)
    if cfg.get("ylim"):
        plt.ylim(cfg["ylim"])

plt.savefig("prescription.pdf", dpi=600)
plt.close()

print("Saved: prescription.pdf, hm_comparison.pdf, and hm_comparison_<age_group>.pdf")
print(f"\nSummary (mean across {N_SEEDS} seeds):")
print(f"  Max budget RLPE (local):  {mean_k_re[-1]:.3f}")
print(f"  Max budget RGPE (global): {mean_g_re[-1]:.3f}")

