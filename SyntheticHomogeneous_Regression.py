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


# We solve the first scenario as an example. 
# The others can be solved by uncommenting the corresponding response definitions provided below.

y[:] = X[:, 1]+X[:, 3]+X[:, 11]+X[:, 13]+ruido
# y[:] = X[:, 1]+X[:, 3]+X[:,6]+X[:,9]+X[:, 11]+X[:, 13]+ruido
# y[:] = X[:, 1]+X[:, 3]+X[:,6]+X[:,9]+X[:, 11]+X[:, 13]+X[:,16]+X[:,19]+ruido




n_samples = X.shape[0]  

Modality = namedtuple('Modality', ['features', 'CV','CF','id'])

mod1 = Modality(features=[5,6,7,8,9], CV=1, CF=0, id=1)
mod2 = Modality(features=[10,11,12,13,14], CV=1, CF=0, id=2)
mod3 = Modality(features=[15,16,17,18,19], CV=1, CF=0,  id=3)


f = BMRandomForest(100, X.shape[0], [mod1, mod2, mod3], 'reg', 'mse',  10)
f.fit_RF(X,y)


# Prescriptive set generation (depends on the selected scenario)
X2 = np.random.multivariate_normal(mean, covariance_matrix, size=200)
y2 = np.zeros(200)
ruido2 = np.random.randn(200)
y2[:] = X2[:, 1]+X2[:, 3]+X2[:, 11]+X2[:, 13]+ruido2


# Prescription resolution
graph.evaluate_prescription_strategy(f, X, y, X2, y2,1, ylim = (120,380), xlim = (-20,350))
graph.evaluate_prescription_strategy(f, X, y, X2, y2,5 ,ylim = (120,380), xlim = (-20,350))
graph.evaluate_prescription_strategy(f, X, y, X2, y2,10, ylim = (120,380), xlim = (-20,350))
graph.evaluate_prescription_strategy_universal(f, X, y, X2, y2, cost = range(0,600,50), ylim = (120,380), xlim = (-20,350))

# Comparison for different k-values and Universal Prescription:
graph.comparison_kvalues([1,5,10],list(range(0, 600, 50)), ['#800080', '#FF8C00', '#4B4B4B', '#00CED1'],X,X2,y,y2,f)


# Heatmaps for k = 1 and the Universal Prescription
graph.heat_map_graph(X,X2,list(range(0, 600, 50)), f, [1,2,3], graph.heat_map_values,1, annotate= True, universal = False)
graph.heat_map_graph(X,X2,list(range(0, 600, 50)), f, [1,2,3], graph.heat_map_values,1, annotate= True, universal = True)

