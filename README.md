# Prescriptive Modality Selection for Classification and Regression

Repository containing the implementation of the methodology **Prescriptive Modality Selection (PMS)** for both classification and regression tasks, as well as the experiments conducted to evaluate its performance. 

The methodology provides **prescriptions** under a budget based on predictive models, including **Compact Random Forest (RF)** and **Extended Random Forest (Extended RF)** variants.

---

## Computational Environment

- **Python version:** 3.12.3  
- **Main libraries:** numpy, random, collections.namedtuple, matplotlib.pyplot, seaborn  
- **Optimization solver:** Gurobi  

---

## Predictive Models

The repository includes two main predictive models:

- **Compact RF:** A Random Forest with customized Decision Trees (DTs).  
- **Extended RF:** A Random Forest with CARTs for each possible combination of modalities.

These models serve as the predictive base for generating prescriptions.

---

## Repository Structure

- **`src/`**: Source code for the methodology, including:
  - `compactRF.py`: Implementation of the Compact RF.  
  - `graph_BMRF.py`: Visualization functions for Compact RF, including prescription plots and heatmaps.  
  - `extendedRF.py`: Implementation of the Extended RF.  
  - `graph_crf.py`: Visualization functions for Extended RF, including prescription plots and heatmaps.
- **`dataset_generation.py`**: Scripts to generate synthetic datasets and manage configuration presets.  
- **Root directory (`/`)**: Experiment scripts:
  - `experiments_compactRF.py`
  - `experiments_extendedRF.py`

---

## Getting Started

1. Clone the repository:
    ```bash
    git clone https://github.com/username/repo-name.git
    cd repo-name
    ```

2. Create and activate the conda environment:
    ```bash
    conda create -n pms_env python=3.12
    conda activate pms_env
    ```

3. Ensure that **Gurobi** is installed and the license is active.

4. Select the desired datasets and scenarios inside each script  
   (`experiments_compactRF.py` and `experiments_extendedRF.py`).

5. Run the experiment scripts:
    ```bash
    python experiments_compactRF.py
    python experiments_extendedRF.py
    ```
---

## Presets

- **Presets** are defined in `dataset_generation.py` under the `presets` dictionary.  
- They include parameters such as:
  - Tree depth  
  - Impurity criterion  
  - Cost structure  
  - Plot limits (`ylim`, `accuracy_lim`)

---

## Output

The experiments generate:

- **Prescription results**: Heatmaps comparing local vs. global prescriptions  
  (file: 'hm_comparison.pdf').
- **Prediction results**: Summary of prediction metrics  
  (file: 'prescription.pdf').

---


