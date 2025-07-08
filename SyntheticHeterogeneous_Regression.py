import numpy as np
import matplotlib.pyplot as plt
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


###### Prescriptive Modality Selection

n_samples = X.shape[0]  

Modality = namedtuple('Modality', ['features', 'CV','CF','id'])

mod1 = Modality(features=[5,6,7,8,9], CV=1, CF=0, id=1)
mod2 = Modality(features=[10,11,12,13,14], CV=1, CF=0, id=2)
mod3 = Modality(features=[15,16,17,18,19], CV=1, CF=0,  id=3)

f = BMRandomForest(100, X.shape[0], [mod1,mod2,mod3], 'reg', 'mse', 10)
f.fit_RF(X,y)

# Prescriptive set

# Scenario 1:
# X2 = np.random.multivariate_normal(mean_A, correlation_matrix, size=200)
# ruido2 = np.random.randn(200)
# y2 = X2 @ weights_A + ruido2


# Scenario 2:
X2_A = np.random.multivariate_normal(mean_A, correlation_matrix, size=100)
X2_B = np.random.multivariate_normal(mean_B, correlation_matrix_B, size=100)
ruido2A = np.random.randn(100)
y2_A = X2_A @ weights_A + ruido2A
ruido2B = np.random.randn(100)
y2_B = X2_B @ weights_B + ruido2B

X2 = np.vstack([X2_A, X2_B])
y2 = np.concatenate([y2_A, y2_B])

# Prescription resolution
graph.evaluate_prescription_strategy(f, X, y, X2, y2,1, ylim = (150,900), xlim = (-20,420))
graph.evaluate_prescription_strategy(f, X, y, X2, y2,5, ylim = (150,900), xlim = (-20,420))
graph.evaluate_prescription_strategy(f, X, y, X2, y2,10, ylim = (150,900), xlim = (-20,420))

# Universal Prescription
graph.evaluate_prescription_strategy_universal(f, X, y, X2, y2, cost = range(0,600,50), ylim = (150,900), xlim = (-20,420))

# Comparison for different k-values and Universal Prescription:
graph.comparison_kvalues([1,5,10],list(range(0, 600, 50)), ['#800080', '#FF8C00', '#4B4B4B', '#00CED1'],X,X2,y,y2,f)

# Heatmaps for k = 1 and the Universal Prescription
graph.heat_map_graph(X,X2,list(range(0, 600, 50)), f, [1,2,3], graph.heat_map_values,1, annotate= True, universal = False)
graph.heat_map_graph(X,X2,list(range(0, 600, 50)), f, [1,2,3], graph.heat_map_values,1, annotate= True, universal = True)

# Heatmaps differentiating between subpopulations
# For k = 1
# Population A
graph.heat_map_graph(X,X2,list(range(0, 600, 50)), f, [1,2,3], graph.heat_map_values,1, annotate= True, universal = False, selected_indices=list(range(100)))
# Population B
graph.heat_map_graph(X,X2,list(range(0, 600, 50)), f, [1,2,3], graph.heat_map_values,1, annotate= True, universal = False, selected_indices=list(range(100,200)))

# Regularization for Universal Prescription across different selections:
# Population A:
graph.heat_map_graph(X,X2,list(range(0, 600, 50)), f, [1,2,3], graph.heat_map_values,1, annotate= True, universal = True, selected_indices=list(range(100)), regularize = True, p = 5)
# Population B:
graph.heat_map_graph(X,X2,list(range(0, 600, 50)), f, [1,2,3], graph.heat_map_values,1, annotate= True, universal = True, selected_indices=list(range(100,200)), regularize = True, p = 5)