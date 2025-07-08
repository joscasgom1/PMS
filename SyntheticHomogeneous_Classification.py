import numpy as np
import random
from collections import namedtuple
from src.BMRF import BMRandomForest
import src.graph_aux as graph

"""

Firstly, the dataset is generated.


"""

n_blocks = 4  
n_features_per_block = 5  
total_features = n_blocks * n_features_per_block  
n_samples = 2000

correlation_matrix = np.eye(total_features)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

intra_block_corr = (0.4, 0.6)  
inter_block_corr = (0.0, 0.2)  


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
X = np.random.multivariate_normal(mean, covariance_matrix, size=n_samples)

ruido = np.random.randn(n_samples)

# Response variable
y = np.zeros(n_samples)


# We solve the third scenario as an example. 
# The others can be solved by uncommenting the corresponding response definitions provided below.

# y[:] = X[:, 1]+X[:, 3]+X[:, 11]+X[:, 13]+ruido
# y[:] = X[:, 1]+X[:, 3]+X[:,6]+X[:,9]+X[:, 11]+X[:, 13]+ruido
y[:] = X[:, 1]+X[:, 3]+X[:,6]+X[:,9]+X[:, 11]+X[:, 13]+X[:,16]+X[:,19]+ruido

# Response variable thresholding to turn it into a classification setting

y_threshold = np.median(y)
y = (y > y_threshold).astype(int)


n_samples = X.shape[0]  

Modality = namedtuple('Modality', ['features', 'CV','CF','id'])
mod1 = Modality(features=[5,6,7,8,9], CV=1, CF=0, id=1)
mod2 = Modality(features=[10,11,12,13,14], CV=1, CF=0, id=2)
mod3 = Modality(features=[15,16,17,18,19], CV=1, CF=0,  id=3)


f = BMRandomForest(100, X.shape[0], [mod1, mod2, mod3], 'cls', 'gini',  10)
f.fit_RF(X,y)
# 1,3,6,9,11,13,16,19

# Prescriptive set generation (depends on the selected scenario)
# We encompass all possible cases, taking into account the potential regularization by averaging over different prescriptive sets, in a single function. In this case, we only study k = 1.
graph.evaluate_prescription_strategy_classification_hom([1,3,6,9,11,13,16,19],mean, covariance_matrix, 200, f,y_threshold, X,y, 1,list(range(0,600,50)), use_regularization=True, n_sets=5)