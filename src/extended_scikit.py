"""
Extended Random Forest implementation using scikit-learn.
Intended solely for time-to-fit comparison purposes.
"""

import numpy as np
from itertools import chain, combinations
from collections import namedtuple
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import time

Modality = namedtuple('Modality', ['features','CV','CF', 'id'])

class extendedRFscikit:
    def __init__(self, n_trees, n_bs, modalities,tree = 'cls',criterion='gini', max_depth=5, replacement = True, random_state = None):
        """
        Initialization of a Classic Random Forest instance:
        - tree: type of tree, either classification ('cls') or regression ('reg')
        - n_trees: number of trees in the forest
        - n_bs: number of individuals in each bootstrap training sample
        - max_depth: maximum depth of each tree
        - replacement: boolean indicating whether the bootstrap samples are drawn with replacement
        - modalities: Supplementary Modalities considered. These are provided as tuples called 'Modality', each consisting of:
                      the features that define the modality, its variable and fixed costs, and an ID identifying the 
                      corresponding Supplementary Modality.
         For example, a value of 0.75 means that, after ranking combinations by importance, only the most relevant
         combinations accounting for 75% of the total relative importance will be retained.
        - random_state: seed value to ensure reproducibility of the experiments
        
        Note: Features not included in any Supplementary Modality are considered part of the Baseline Modality.
        """

        self.tree = tree
        self.n_trees = n_trees
        self.n_bs = n_bs
        self.modalities = modalities
        self.criterion = criterion
        self.max_depth = max_depth
        self.total_exp_features = {f for c in self.modalities for f in c.features}
        self.replacement = replacement
        self.random_state = random_state
        
    def _bagging(self,X,y,n):
        """
        Random selection of a bootstrap sample to train each tree
        """
        ind_bootstrap = np.random.choice(n, size=n, replace=self.replacement)
        Xbs = X[ind_bootstrap, :].astype(np.float32, copy=True)
              
        ybs = y[ind_bootstrap]
        ind_oob = list(set(range(n)) - set(ind_bootstrap))
        return ind_oob,Xbs,ybs
    
            
    def fit_RF(self,X,y):
        """
        Fitting of a Customized Random Forest model
        """
          
        self.ch_feat = [i for i in range(X.shape[1]) if i not in self.total_exp_features]
        
        # Calculation of all possible modalities combination
        self.modality_comb(self.modalities)
        
        # Corresponding features to the previous calculated combinations
        self.cat_comb(self.ch_feat,self.m_comb)
        
        # A possible random forest for each possible combination
        if self.tree == 'cls':
            t1 = time.time()
            classic_forests = []
            for comb in self.f_comb:
                X_comb = X[:,comb].copy()
                rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, max_features = "sqrt")
                rf.fit(X_comb,y)
                classic_forests.append(rf)
            t2 = time.time()
            print('time to fit:', t2-t1)
            self.classic_rf = classic_forests
        else: 
            t1 = time.time()
            classic_forests = []
            for comb in self.f_comb:
                X_comb = X[:,comb].copy()
                rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, max_features = 0.3)
                rf.fit(X_comb,y)
                classic_forests.append(rf)
            t2 = time.time()
            print('time to fit:', t2-t1)
            self.classic_rf = classic_forests

        
    
    def _precompute_oob_mapping(self):
        """
        Calculates for which trees is OOB each individual
        """
        n_samples = self.n_bs
        
        # Inicializa directamente la estructura anidada
        t_oob_mapping = {
            i: {j: [] for j in range(n_samples)}
            for i in range(len(self.f_comb))
        }
        
        for i in range(len(self.f_comb)):
            for tree, oob_indices in zip(self.classic_rf[i], self.t_oob[i]):
                oob_set = set(oob_indices)
                for k in oob_set:
                    t_oob_mapping[i][k].append(tree)
        
        self.t_oob_mapping = t_oob_mapping

        
    def modality_comb(self,lst):
        """
        Returns the power set of the input list `lst`.
        """
        l = [i.id for i in self.modalities]
        self.m_comb = [list(subset) for subset in chain.from_iterable(combinations(l, r) for r in range(len(l) + 1))]
        
        
    def cat_comb(self,bsl_feat,supp_feat):
        """
        Returns the combination of features that can be used in each modalities combination
        """
        op = []
        for i in supp_feat:
            f_comb = [f for c in self.modalities for f in c.features if c.id in i]
            op.append(bsl_feat+f_comb)
        self.f_comb = op
        
    # def _predictRF(self,x,i_comb):
        
    #     """
    #     Prediction for an individual sample `x` using the i-th forest in `self.classic_rf`
    #     (i_comb indicates which forest to use).
    #     Aggregates predictions from the trees in that forest.
    #     """
    #     xt = x[self.f_comb[i_comb]].copy()
    #     preds = np.array([tree.root._predict(xt) for tree in self.classic_rf[i_comb]])
    #     return stats.mode(preds)[0] if self.tree == 'cls' else np.mean(preds)

    # def predictRF(self,X,I_comb):
    #     """
    #     Prediction for an individual
    #     """
    #     return np.array([self._predictRF(x,i_comb) for x,i_comb in zip(X,I_comb)])    
    
    # def reference_error(self,X,y):
    #     """
    #     Calculation of the reference error for a prescriptive set
    #     """
    #     yp = np.array([self._predictRF(x,-1) for x in X])
    #     return np.mean(yp==y) if self.tree == 'cls' else np.sum(np.abs(yp-y))
    
    # def _predict_oob(self,x,i_comb,i):
    #     trees = self.t_oob_mapping[i_comb].get(i, [])
    #     xt = x[self.f_comb[i_comb]].copy()
    #     preds = [tree.root._predict(xt) for tree in trees]
    #     if not preds:
    #         return None

    #     if self.tree == 'cls':
    #         counts = np.bincount(preds)
    #         return np.argmax(counts)
    #     else:
    #         return np.mean(preds)

    
    # def predict_OOB(self,X,i_comb):
    #     """
    #     Out-Of-Bag prediction for each individual in the training set
    #     """
    #     nr = X.shape[0]
    #     return np.array([self._predict_oob(X[i],i_comb,i) for i in range(nr)])

    # def prob_class_OOB(self,X,i_comb,y):
    #         nr = X.shape[0]
    #         X_t = X[:,self.f_comb[i_comb]].copy()
    #         return np.array([np.mean(np.array([tree.root._predict(X_t[i]) for tree in  self.t_oob_mapping[i_comb].get(i, [])])==y[i]) for i in range(nr)])
        

    # def error_estimation_oob(self, X, y):
    #     """
    #     OOB prediction error for each individual in the training set
    #     Parameters:
    #         - X: features of training individuals
    #         - y: response value of training individuals           
    #     """
    #     l = len(self.f_comb)
    #     prediction_comb,error_comb = np.zeros((X.shape[0],l)),np.zeros((X.shape[0],l))
    #     for i in range(l):
    #         if self.tree=='cls':
    #             X_t = X.copy()
    #             prediction_comb[:,i] = self.predict_OOB(X_t,i)
    #             error_comb[:,i] =  1-self.prob_class_OOB(X_t,i,y)
    #         else:
    #             X_t = X.copy()
    #             prediction_comb[:,i] = self.predict_OOB(X_t,i)
    #             error_comb[:,i] = np.abs(prediction_comb[:,i]-y)
    #     self.BM_train_estimation = prediction_comb
    #     self.BM_train_error = error_comb      
        
    # def _terminal_leaf_OOB(self, X):
    #     """
    #     Lists of terminal nodes across the trees taking into account only the RF grown using only the Baseline Modality and OOB trees
    #     """
    #     Xt = X[:,self.ch_feat].copy()  # We use only features from the BaseLine Modality
        
    #     hojas_OOB = {}  
    #     for i, (tree, oob) in enumerate(zip(self.classic_rf[0], self.t_oob[0])):  
    #         for idx in oob:  
    #             x = Xt[idx]  
    #             hoja = tree.root._terminal_leaf(x, 1)  
    #             if idx not in hojas_OOB:
    #                 hojas_OOB[idx] = {}  
    #             hojas_OOB[idx][i] = hoja      
    #     return hojas_OOB
    
    
    # def _terminal_leaf_test(self, X):
    #     """
    #     Lists of terminal nodes across the trees taking into account only the Baseline Modality for individuals in Prescriptive Set
    #     """
    #     Xt = X[:,self.ch_feat].copy() 
    #     hojas_test = {}  
    #     for i, tree in enumerate(self.classic_rf[0]):  
    #         for idx, x in enumerate(Xt):  
    #             hoja = tree.root._terminal_leaf(x, 1)  
    #             if idx not in hojas_test:
    #                 hojas_test[idx] = {}  
    #             hojas_test[idx][i] = hoja  
    #     return hojas_test
    
    # def comparar_hojas(self, hojas_OOB, hojas_test, k):
    #     """
    #     Comparison of terminal nodes for individuals in the training set and in the prescriptive set,
    #     in order to measure the proximity between them
    #     """
    #     coincidencias = {}    
    #     hojas_OOB_sets = {
    #         idx_muestra: set((idx_arbol, hoja_terminal) for idx_arbol, hoja_terminal in hojas.items())
    #         for idx_muestra, hojas in hojas_OOB.items()
    #     }    
    #     hojas_test_sets = {
    #         idx_muestra: set((idx_arbol, hoja_terminal) for idx_arbol, hoja_terminal in hojas.items())
    #         for idx_muestra, hojas in hojas_test.items()
    #     }
    
    #     for idx_test, hojas_t_set in hojas_test_sets.items():
    #         mejores_matches = heapq.nlargest(
    #             k,
    #             (    
    #                 (idx_train, len(hojas_t_set & hojas_o_set)) 
    #                 for idx_train, hojas_o_set in hojas_OOB_sets.items()
    #             ),
    #             key=lambda x: x[1]  
    #         )
    #         coincidencias[idx_test] = mejores_matches
    
    #     return coincidencias


    

    # def _error_estimation_RFbased(self, X_train, X_test, k):
    #     """
    #     Estimation of prediction errors for individuals in the prescriptive set according to the
    #     proximity measure provided by the Random Forest
    #     """
    #     matches = self.comparar_hojas(self._terminal_leaf_OOB(X_train), self._terminal_leaf_test(X_test), k)
    
    #     error_estimation = np.zeros((len(X_test), self.BM_train_error.shape[1]))  
    #     error_estimation[:] = np.nan  # Inizalization  with NaN
    
    #     for i, t in matches.items(): 
    #         ind = [x[0] for x in t if x[0] < len(X_train)]  
            
    #         if len(ind) > 0:
    #             error_estimation[i] = np.nanmean(self.BM_train_error[ind], axis=0)  
    #         else:
    #             error_estimation[i] = np.nan  
        
    #     self.error_estimation_test = error_estimation
        
    # def prob_class_predict(self,X,y):
    #     """
    #     Estimates the probability that an individual belongs to a given class in a classification setting
    #     """
    #     Xc = X.copy()
    #     for i, feat_idx in enumerate(self.feature_selection):
    #         Xc[i, self.null_feat[feat_idx]] = np.nan
    #     nr = Xc.shape[0]
    #     return np.sum(np.array([np.mean(np.array([tree.root._predict(Xc[i]) for tree, oob_indices in zip(self.rf, self.OOB)])==y[i]) for i in range(nr)]))

    # def BMOptimization(self,Budget):
    #     """
    #     Resolution of the selection of modalities.
        
    #     Parameters:
    #         - Budget: Maximum budget available
    #     """
    #     CV = [sum(c.CV for c in self.modalities if c.id in i) if len(i) > 0 else 0 for i in self.m_comb]
    #     self.Cost_V = CV
    #     self.distinct_mod = sorted({x for sublista in self.m_comb for x in sublista})
    #     self.CF_unique = [sum(c.CF for c in self.modalities if c.id == i) for i in self.distinct_mod]
    #     self.feature_selection, self.total_cost, self.total_error, self.times = _BMOptimization(self.error_estimation_test, CV, self.CF_unique, Budget, self.distinct_mod,self.m_comb)

    # def actualError(self,X,y):
    #     """
    #     Calculation of the real prediction error
    #     """
    #     yp = self.predictRF(X,self.feature_selection);
    #     return np.mean(yp==y) if self.tree == 'cls' else np.sum(np.abs(yp-y))
    
    # def actualError_c(self,X,y,c):
    #     """
    #     Calculation of the real prediction error
    #     """
    #     yp = self.predictRF(X,self.feature_selection[c]);
    #     return np.mean(yp==y) if self.tree == 'cls' else np.sum(np.abs(yp-y))
    
    # def BMOptimization_universal(self,Budget,n):
    #     """
    #     Resolution of the selection of modalities for the Universal Prescription
        
    #     Parameters:
    #         - Budget: List of maximum budgets available
    #     """
    #     CV = [sum(c.CV for c in self.modalities if c.id in i) if len(i) > 0 else 0 for i in self.m_comb]
    #     CF = [sum(c.CF for c in self.modalities if c.id in i) if len(i) > 0 else 0 for i in self.m_comb]
    #     self.Cost_V, self.Cost_F = CV,CF
    #     self.distinct_cat = sorted({x for sublista in self.m_comb for x in sublista})
    #     self.CF_unique = [sum(c.CF for c in self.modalities if c.id == i) for i in self.distinct_cat]
    #     universal_error = np.mean(self.BM_train_error, axis = 0)
    #     self.feature_selection, self.total_cost, self.total_error, self.times = _BMOptimization_universal(universal_error, CV, self.CF_unique, Budget, self.distinct_cat,self.m_comb,n)