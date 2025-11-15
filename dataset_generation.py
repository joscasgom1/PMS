import matplotlib.pyplot as plt
from collections import namedtuple

Modality = namedtuple('Modality', ['features', 'CV','CF','id'])

def get_dataset(dataset_name, scenario= None, setting ="reg", seed=42):
    
    """
    Return train and prescriptive dataset

     Parameters:
    dataset_name : str
        Name of the dataset. E.g.: 'synthetic_homogeneous', 'synthetic_heterogeneous',
        'wine', 'bikesharing'.
    scenario : str or int
        Specific scenario within the dataset, if applicable.
    setting : str
       'reg' for regression, 'cls' for classification.
    seed : int
        Seed for reproducibility.

    Return:
        X_train : np.array
        X_pres : np.array
        y_train : np.array
        y_pres : np.array
    """
    import numpy as np
    np.random.seed(42)

    # ==== Synthetic Homogeneous ====
    if dataset_name == "synthetic_homogeneous":
        scenarios = {
            "1": [1, 3, 11, 13],
            "2": [1, 3, 6, 9, 11, 13],
            "3": [1, 3, 6, 9, 11, 13, 16, 19]
        }
        if scenario is None:
            scenario = "1"
        index = scenarios[scenario]

        n_blocks = 4  
        n_features_per_block = 5  
        total_features = n_blocks * n_features_per_block  
        n_samples = 2000

        correlation_matrix = np.eye(total_features)



        intra_block_corr = (0.4, 0.6)  
        inter_block_corr = (0, 0.2)  


        for i in range(n_blocks):
            for j in range(n_features_per_block):
                for k in range(n_features_per_block):
                    if j != k:
                        correlation_matrix[i*n_features_per_block + j, i*n_features_per_block + k] = np.random.uniform(*intra_block_corr)


        for i in range(n_blocks):
            for j in range(i+1, n_blocks):
                for k in range(n_features_per_block):
                    for l in range(n_features_per_block):
                        correlation_matrix[i*n_features_per_block + k, j*n_features_per_block + l] = np.random.uniform(*inter_block_corr)
                        correlation_matrix[j*n_features_per_block + l, i*n_features_per_block + k] = correlation_matrix[i*n_features_per_block + k, j*n_features_per_block + l]


        covariance_matrix = correlation_matrix  

        mean = np.zeros(total_features)
        X_train = np.random.multivariate_normal(mean, covariance_matrix, size=n_samples)

        noise = np.random.randn(n_samples)

        # Response variable
        y_train = np.zeros(n_samples)
        y_train[:] = X_train[:, index].sum(axis=1) + noise

        """
        Prescriptive dataset
        """

        X_pres = np.random.multivariate_normal(mean, covariance_matrix, size=200)
        y_pres = np.zeros(200)
        noise2 = np.random.randn(200)
        y_pres[:] = X_pres[:, index].sum(axis=1) + noise2
        
        # Convert to classification if needed
        if setting == 'cls':
            y_threshold = np.median(y_train)
            y_train = (y_train > y_threshold).astype(int)
            y_pres = (y_pres > y_threshold).astype(int)
            
        mod1 = Modality(features=[5,6,7,8,9], CV=1, CF=0, id=1)
        mod2 = Modality(features=[10,11,12,13,14], CV=1, CF=0, id=2)
        mod3 = Modality(features=[15,16,17,18,19], CV=1, CF=0,  id=3)
        modalities = [mod1,mod2,mod3]
        return X_train,y_train,X_pres,y_pres, modalities

    # ==== Synthetic heterogeneous ====
    
    elif  dataset_name == "synthetic_heterogeneous":
        n_blocks = 4
        n_features_per_block = 5
        total_features = n_blocks * n_features_per_block
        n_samples_per_pop = 1000  
        n_samples = 2 * n_samples_per_pop  
    
    
        correlation_matrix = np.eye(total_features)
        intra_block_corr = (0.4, 0.6)
        intra_block_corr_B = (0.6,0.8)
        inter_block_corr = (0.0, 0.2)
    
        SEED = 42
        np.random.seed(SEED)
    
    
        for i in range(n_blocks):
            for j in range(n_features_per_block):
                for k in range(n_features_per_block):
                    if j != k:
                        correlation_matrix[i * n_features_per_block + j, i * n_features_per_block + k] = np.random.uniform(*intra_block_corr)
    
        for i in range(n_blocks):
            for j in range(i + 1, n_blocks):
                for k in range(n_features_per_block):
                    for l in range(n_features_per_block):
                        val = np.random.uniform(*inter_block_corr)
                        correlation_matrix[i * n_features_per_block + k, j * n_features_per_block + l] = val
                        correlation_matrix[j * n_features_per_block + l, i * n_features_per_block + k] = val
    
        # We define different means and covariances for each population
        mean_A = np.zeros(total_features)   # Population A with mean 0
        mean_B = np.zeros(total_features)
        mean_B = mean_B + 2  # Population B with higher mean
    
    
        # Correlation matrix for subpopulation B
        correlation_matrix_B = np.copy(correlation_matrix)
        for i in range(n_blocks):
            for j in range(n_features_per_block):
                for k in range(n_features_per_block):
                    if j != k:
                        correlation_matrix_B[i * n_features_per_block + j, i * n_features_per_block + k] = np.random.uniform(*intra_block_corr_B)
    
        # Features within each subpopulation
        X_A = np.random.multivariate_normal(mean_A, correlation_matrix, size=n_samples_per_pop)
        X_B = np.random.multivariate_normal(mean_B, correlation_matrix_B, size=n_samples_per_pop)
    
        # Complete training set
        X = np.vstack([X_A, X_B])
    
        # Noise generation
        ruido = np.random.randn(n_samples)
    
        # betas for each subpopulation
        weights_A = np.zeros(total_features)  
        weights_A[:n_features_per_block * 2] = np.random.uniform(0.5, 1.5, size=n_features_per_block * 2)
    
        weights_B = np.zeros(total_features)  
        weights_B[:n_features_per_block] = np.random.uniform(0.5, 1.5, size=n_features_per_block)
        weights_B[n_features_per_block * 2:n_features_per_block * 3] = np.random.uniform(0.5, 1.5, size=n_features_per_block)
    
        # Response value for each subpopulation
        y = np.zeros(n_samples)
        y[:n_samples_per_pop] = X_A @ weights_A + ruido[:n_samples_per_pop]  
        y[n_samples_per_pop:] = X_B @ weights_B + ruido[n_samples_per_pop:]  
    
        median_A = np.median(y[:n_samples_per_pop])
        median_B = np.median(y[n_samples_per_pop:])    
    
        if setting == 'cls':

            y[:n_samples_per_pop] = (y[:n_samples_per_pop] > median_A).astype(int)
            y[n_samples_per_pop:] = (y[n_samples_per_pop:] > median_B).astype(int)
            y = y.astype(int)
    
        # Graph for the features and variable response
        plt.figure(figsize=(12, 6))
        for i in range(4):  # 4 first features are plotted
            plt.subplot(2, 2, i + 1)
            plt.hist(X_A[:, i], bins=30, alpha=0.6, label="Population A")
            plt.hist(X_B[:, i], bins=30, alpha=0.6, label="Population B")
            #plt.title(f"Feature {i+1}")
            plt.xlabel(fr'$X_{{{i+1}}}^{{(0)}}$', fontsize=14)
            plt.xlim(-5,5)
            plt.ylim(0,110)
            plt.ylabel("Frequency")
            plt.legend(loc='upper left')
    
        plt.tight_layout()
        plt.show()
    
        # Response variable distribution
        plt.figure(figsize=(8, 5))
        plt.hist(y[:n_samples_per_pop], bins=30, alpha=0.6, label="Population A")
        plt.hist(y[n_samples_per_pop:], bins=30, alpha=0.6, label="Population B")
        # plt.title("Response value distribution")
        plt.xlabel(r"$Y$", fontsize = 14)
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()
    
        is_classification = (setting == 'cls')

        def gen_dataset_pres(mean, cov, weights, size, threshold=None):
            X = np.random.multivariate_normal(mean, cov, size=size)
            noise = np.random.randn(size)
            y = X @ weights + noise
            if threshold is not None:
                y = (y > threshold).astype(int)
            return X, y
        
        threshold_A = median_A if is_classification else None
        threshold_B = median_B if is_classification else None
        scenarios = {
            "4": [
                {"mean": mean_A, "cov": correlation_matrix, "weights": weights_A, "size": 200, "threshold": threshold_A}
            ],
            "5": [
                {"mean": mean_A, "cov": correlation_matrix, "weights": weights_A, "size": 100, "threshold": threshold_A},
                {"mean": mean_B, "cov": correlation_matrix_B, "weights": weights_B, "size": 100, "threshold": threshold_B}
            ]
        }
        
        datasets = [gen_dataset_pres(**cfg) for cfg in scenarios[scenario]]
        X2 = np.vstack([d[0] for d in datasets])
        y2 = np.concatenate([d[1] for d in datasets])

        mod1 = Modality(features=[5,6,7,8,9], CV=1, CF=0, id=1)
        mod2 = Modality(features=[10,11,12,13,14], CV=1, CF=0, id=2)
        mod3 = Modality(features=[15,16,17,18,19], CV=1, CF=0,  id=3)
        modalities = [mod1,mod2,mod3]

        return X,y,X2,y2, modalities

    # ==== Wine dataset ====
    elif dataset_name == "wine":
        from ucimlrepo import fetch_ucirepo
        from sklearn.model_selection import train_test_split

        # Load Wine dataset
        wine_quality = fetch_ucirepo(id=186)
        X = wine_quality.data.features.values
        y = wine_quality.data.targets.values.ravel()  # Aplanar target
        
        # Convertir a clasificación si es necesario
        if setting == 'cls':
            y = (y > 5).astype(int)
        
        # Dividir en entrenamiento y prescriptivo
        X_train, X_pres, y_train, y_pres = train_test_split(
            X, y, test_size=0.1, random_state=42
        )
        
        mod1 =Modality(features=[1], CV=1, CF=0,  id=1)
        mod2 =Modality(features=[2], CV=1, CF=0,  id=2)
        mod3 =Modality(features=[3], CV=1, CF=0,  id=3)
        mod4 =Modality(features=[4], CV=1, CF=0,  id=4)
        mod5 =Modality(features=[5], CV=1, CF=0,  id=5)
        mod6 =Modality(features=[6], CV=1, CF=0,  id=6)
        mod7 =Modality(features=[7], CV=1, CF=0,  id=7)
        mod8 =Modality(features=[8], CV=1, CF=0,  id=8)
        mod9 =Modality(features=[9], CV=1, CF=0,  id=9)
        mod10 =Modality(features=[10], CV=1, CF=0,  id=10)   
        modalities = [mod1,mod2,mod3,mod4,mod5,mod6,mod7,mod8,mod9,mod10]
        return X_train,y_train, X_pres,y_pres, modalities

    # ==== Bikesharing (ejemplo) ====
    elif dataset_name == "bikesharing":
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from ucimlrepo import fetch_ucirepo
        
        # Load BikeSharing dataset
        bike_data = fetch_ucirepo(id=275)
        
        X_raw = bike_data.data.features
        y = bike_data.data.targets.squeeze()  # Flatten to 1D array
        
        # ------------------- Convert to classification if needed -------------------
        # Apply threshold only if setting == 'clf'
        if setting == 'cls':
            y = (y > np.median(y)).astype(int)
        
        # ------------------- Drop unnecessary columns -------------------
        # Columns not relevant for learning:
        # - 'instant': row index
        # - 'dteday': date
        # - 'casual', 'registered': components of the target variable
        cols_to_drop = ['instant', 'dteday', 'casual', 'registered']
        X_raw = X_raw.drop(columns=[col for col in cols_to_drop if col in X_raw.columns])
        
        # ------------------- Define categorical variables -------------------
        categorical_to_ohe = ['season', 'mnth', 'weekday']
        
        # ------------------- Split into training and prescriptive sets -------------------
        X_train_raw, X_pres_raw, y_train, y_pres = train_test_split(
            X_raw, y, test_size=0.2, random_state=42
        )
        
        # ------------------- One-Hot Encoding -------------------
        X_train_ohe = pd.get_dummies(X_train_raw, columns=categorical_to_ohe, drop_first=False)
        X_pres_ohe = pd.get_dummies(X_pres_raw, columns=categorical_to_ohe, drop_first=False)
        
        # Ensure both train and prescriptive sets have the same columns
        X_pres_ohe = X_pres_ohe.reindex(columns=X_train_ohe.columns, fill_value=0)
        
        # ------------------- Convert to NumPy arrays -------------------
        X_train = X_train_ohe.to_numpy()
        X_pres = X_pres_ohe.to_numpy()
        y_train = y_train.to_numpy()
        y_pres = y_pres.to_numpy()

        mod1 = Modality(features=[4, 5], CV=1, CF=0, id=1)
        mod2 = Modality(features= [6, 7, 8], CV=1, CF=0, id=2)
        mod3 = Modality(features=[9, 10, 11, 12], CV=1, CF=0, id=3)
        modalities = [mod1,mod2,mod3]

        return X_train,y_train, X_pres,y_pres, modalities
    
    else:
        raise ValueError(f"Dataset {dataset_name} no reconocido")

"""   PRESETS """

presets = {
    "synthetic_homogeneous": {
        "scenarios": {
            "1": {
                "reg":  {"impurity":"mse","tree_depth": 10, "cost": range(0, 601, 50), "ylim": (80,350)},
                "cls":  {"impurity":"gini","tree_depth": 10, "cost": range(0, 601, 50), "ylim": (20,70), "accuracy_lim": (0.7,0.85)}
            },
            "2": {
                "reg":  {"impurity":"mse","tree_depth": 10, "cost": range(0, 601, 50), "ylim": (80,480)},
                "cls":  {"impurity":"gini","tree_depth": 10, "cost": range(0, 601, 50), "ylim": (20,100), "accuracy_lim": (0.65,0.85)}
            },
            "3": {
                "reg":  {"impurity":"mse","tree_depth": 10, "cost": range(0, 601, 50), "ylim": (80,550)},
                "cls":  {"impurity":"gini","tree_depth": 10, "cost": range(0, 601, 50), "ylim": (10,80), "accuracy_lim": (0.65,0.9)}
            }
        }
    },
    "synthetic_heterogeneous": {
        "scenarios": {
            "4": {
                "reg":  {"impurity":"mse","tree_depth": 10, "cost": range(0, 601, 50), "ylim": (150,900)},
                "cls":  {"impurity":"gini","tree_depth": 10, "cost": range(0, 601, 50), "ylim": (20,90), "accuracy_lim": (0.65,0.9)}
            },
            "5": {
                "reg":  {"impurity":"mse","tree_depth": 10, "cost": range(0, 601, 50), "ylim": (150,920)},
                "cls":  {"impurity":"gini","tree_depth": 10, "cost": range(0, 601, 50), "ylim": (20,90), "accuracy_lim": (0.65,0.9)}
            }
        }
    },
    "wine": {
        "scenarios": {
            "6": {
                "reg": {"impurity":"mse","tree_depth": 6, "cost": range(0, 4001, 250), "ylim": (150,470)},
                "cls": {"impurity":"gini","tree_depth": 6, "cost": range(0, 4001, 250), "ylim": (0,220), "accuracy_lim": (0.64,0.77)}
            }
        }
    },
    "bikesharing": {
        "scenarios": {
            "7": {
                "reg": {"impurity":"mse","tree_depth": 10, "cost": range(0, 10451, 750), "ylim": (60000,180000)},
                "cls": {"impurity":"gini","tree_depth": 10, "cost": range(0, 10451, 750), "ylim": (130,450), "accuracy_lim": (0.9,0.94)}
            }
        }
    }
}

