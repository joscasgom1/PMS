
import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo  # Fetch datasets from UCI Machine Learning Repository
from collections import namedtuple
from src.BMRF import BMRandomForest
import src.graph_aux as graph

# ------------------- Load the Dataset and Preprocess -------------------
# Fetch the Bike Sharing Dataset (UCI ID: 275)
bike_data = fetch_ucirepo(id=275)

# Separate features and target variable
X_raw = bike_data.data.features
y = bike_data.data.targets.squeeze()  # Convert to 1D array if needed

# ------------------- Drop Unnecessary Columns -------------------
# These columns should not be used as features:
# - 'instant': Row index (irrelevant for learning)
# - 'dteday': Date (already encoded in other columns)
# - 'casual' and 'registered': Components of the target variable
cols_to_drop = ['instant', 'dteday', 'casual', 'registered']
X_raw = X_raw.drop(columns=[col for col in cols_to_drop if col in X_raw.columns])

# ------------------- Define Categorical Variables -------------------
# These will be one-hot encoded later
categorical_to_ohe = ['season', 'mnth', 'weekday']

# ------------------- Split into Train and Test Sets -------------------
# Split the dataset before encoding to avoid data leakage
X_train_raw, X_pres_raw, y_train, y_pres = train_test_split(
    X_raw, y, test_size=0.2, random_state=42
)

# ------------------- One-Hot Encoding -------------------
# Apply one-hot encoding to categorical variables (without dropping first category)
X_train_ohe = pd.get_dummies(X_train_raw, columns=categorical_to_ohe, drop_first=False)
X_pres_ohe = pd.get_dummies(X_pres_raw, columns=categorical_to_ohe, drop_first=False)

# Ensure both train and test sets have the same columns
X_pres_ohe = X_pres_ohe.reindex(columns=X_pres_ohe.columns, fill_value=0)

# ------------------- Convert to NumPy Arrays -------------------
# Prepare data for models that expect NumPy inputs
X_train = X_train_ohe.to_numpy()
X_pres = X_pres_ohe.to_numpy()
y_train = y_train.to_numpy()
y_pres = y_pres.to_numpy()



### Prescriptive Modality Selection

Modality = namedtuple('Modality', ['features', 'CV','CF','id'])


mod1 = Modality(features=[4, 5], CV=1, CF=0, id=1)
mod2 = Modality(features= [6, 7, 8], CV=1, CF=0, id=2)
mod3 = Modality(features=[9, 10, 11, 12], CV=1, CF=0, id=3)


f = BMRandomForest(100, X_train.shape[0], [mod1,mod2,mod3], 'reg', 'mse', 10, random_state=42)
f.fit_RF(X_train,y_train)

# Prescription resolution for different k values and Universal Prescription

graph.evaluate_prescription_strategy(f, X_train, y_train, X_pres, y_pres,1,cost = list(range(0, 12501, 250)),xlim=(-50,7400), ylim=(60000,180000))
graph.evaluate_prescription_strategy(f, X_train, y_train, X_pres, y_pres,5,cost = list(range(0, 12501, 250)),xlim=(-50,7400), ylim=(60000,180000))
graph.evaluate_prescription_strategy(f, X_train, y_train, X_pres, y_pres,10,cost = list(range(0, 12501, 250)),xlim=(-50,7400), ylim=(60000,180000))
graph.evaluate_prescription_strategy_universal(f, X_train, y_train, X_pres, y_pres, cost = list(range(0, 12501, 250)),xlim=(-50,7400), ylim=(60000,180000))

# Comparison for different k-values and Universal Prescription:
graph.comparison_kvalues([1,5,10,50],list(range(0, 12501, 250)),  ['#800080', '#FF8C00', '#4B4B4B', '#00CED1','#EB00E7'],X_train,X_pres,y_train,y_pres,f)

# Heatmap for k = 1 
graph.heat_map_graph(X_train,X_pres,list(range(0, 12501, 250)), f, [1,2,3], graph.heat_map_values,1, annotate= True, universal = False)


# Prescription resolution for k = 50
graph.evaluate_prescription_strategy(f, X_train, y_train, X_pres, y_pres,50,cost = list(range(0, 12501, 250)))
