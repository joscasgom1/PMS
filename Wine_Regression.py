from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from collections import namedtuple
from src.BMRF import BMRandomForest
import src.graph_aux as graph


# Load the Wine Quality dataset from the UCI repository 
wine_quality = fetch_ucirepo(id=186)

# Extract features and targets from the dataset
X = wine_quality.data.features.values
y = wine_quality.data.targets.values.ravel()  # Flatten target array if needed

#  Split data into training and test sets 
X_train, X_pres, y_train, y_pres = train_test_split(
    X, y, test_size=0.1, random_state=42
)


Modality = namedtuple('Modality', ['features', 'CV','CF','id'])


####### Supplementary Modalities #######
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

######## Prescriptive Modality Selection
 
f = BMRandomForest(100, X_train.shape[0], [mod1,mod2,mod3,mod4,mod5,mod6,mod7,mod8,mod9,mod10], 'reg', 'mse', 6, random_state=42)
f.fit_RF(X_train,y_train)


### Importance of Combination of Modalities
### Considering only the combination of the Baseline Modality with a Supplementary Modality
list_imp = f.cat_importance_g()
l2 = [a for a in list_imp if 0 in set(a[0]) and len(set(a[0])) == 2]
graph.plot_frozenset_bars_labeled(l2)    

# Prescription resolution for k = 1 
# graph.evaluate_prescription_strategy(f, X_train, y_train, X_pres, y_pres,1,cost = list(range(0, 3000, 250)))

# Heatmap for k = 1 
# graph.heat_map_graph(X_train,X_pres,list(range(0, 12501, 250)), f, [1,2,3,4,5,6,7,8,9,10], graph.heat_map_values,1, annotate= True, universal = False)

# Prescription resolution for k = 1, retaining only those modality combinations 
# whose cumulative importance accounts for at least 75% of the total
graph.evaluate_prescription_strategy(f, X_train, y_train, X_pres, y_pres,1,ncomb = 0.75, cost = list(range(0, 3000, 250)),xlim=(-20,2700), ylim=(150,470))

# Different values for k and the case of the universal prescription
graph.evaluate_prescription_strategy(f, X_train, y_train, X_pres, y_pres,5,ncomb = 0.75, cost = list(range(0, 3000, 250)),xlim=(-20,2700), ylim=(150,470))
graph.evaluate_prescription_strategy(f, X_train, y_train, X_pres, y_pres,10,ncomb = 0.75, cost = list(range(0, 3000, 250)),xlim=(-20,2700), ylim=(150,470))
graph.evaluate_prescription_strategy_universal(f, X_train, y_train, X_pres, y_pres, cost = list(range(0, 3000, 250)), xlim=(-20,2700), ylim=(150,470))
