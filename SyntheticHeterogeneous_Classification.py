import numpy as np
from collections import namedtuple
from src.BMRF import BMRandomForest
import src.graph_aux as graph

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
y[:n_samples_per_pop] = X_A @ weights_A + ruido[:n_samples_per_pop]  # Población A
y[n_samples_per_pop:] = X_B @ weights_B + ruido[n_samples_per_pop:]  # Población B

# Thresholding of the response variable within each subpopulation
y_threshold_A = np.median(y[:n_samples_per_pop])
y_threshold_B = np.median(y[n_samples_per_pop:])
y[:n_samples_per_pop] = (y[:n_samples_per_pop] > y_threshold_A).astype(int)
y[n_samples_per_pop:] = (y[n_samples_per_pop:] > y_threshold_B).astype(int)
y = y.astype(int)


###### Prescriptive Modality Selection

n_samples = X.shape[0]  

Modality = namedtuple('Modality', ['features', 'CV','CF','id'])

mod1 = Modality(features=[5,6,7,8,9], CV=1, CF=0, id=1)
mod2 = Modality(features=[10,11,12,13,14], CV=1, CF=0, id=2)
mod3 = Modality(features=[15,16,17,18,19], CV=1, CF=0,  id=3)

f = BMRandomForest(100, X.shape[0], [mod1,mod2,mod3], 'cls', 'gini', 10)
f.fit_RF(X,y)

# Prescriptive set

# Prescriptive set generation (depends on the selected scenario)
# We encompass all possible cases, taking into account the potential regularization by averaging over different prescriptive sets, in a single function. In this case, we only study scenario 2 (prescriptive set from both subpopulations)
graph.evaluate_prescription_strategy_classification_het([weights_A,weights_B],[mean_A, mean_B], [correlation_matrix, correlation_matrix_B], [100,100], f,[y_threshold_A,y_threshold_B], X,y, 1,list(range(0,600,50)), use_regularization=True, n_sets=5)