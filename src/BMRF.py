import numpy as np
from scipy import stats
from itertools import chain, combinations
from collections import namedtuple, Counter
from .BMTree import BM_Tree 
from .Optimization import _BMOptimization
from .Optimization_universal import _BMOptimization_universal
import heapq


Modality = namedtuple('Modality', ['features','CV','CF', 'id'])

class BMRandomForest:
    def __init__(self, n_trees, n_bs, modalities,tree = 'cls',criterion='gini', max_depth=5, replacement = True, random_state = None):
        """
        Initialization of a customized Random Forest instance:
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
        
    def _bagging(self,X,y,n,cat):

        ind_bootstrap = np.random.choice(n, size=n, replace=self.replacement)
        Xbs = X[ind_bootstrap, :].astype(np.float32, copy=True)
              
        ybs = y[ind_bootstrap]
        ind_oob = list(set(range(n)) - set(ind_bootstrap))
        return ind_oob,Xbs,ybs
    
            
    def fit_RF(self,X,y):
        """
        Fitting of a Customized Random Forest model
        """
        if self.random_state:
            np.random.seed(self.random_state)
            
        trees, OOB = [], []
        self.ch_feat = [i for i in range(X.shape[1]) if i not in self.total_exp_features]

        for _ in range(self.n_trees):
            indices, Xb, yb = self._bagging(X, y, self.n_bs, self.ch_feat)
            tree = BM_Tree(tree=self.tree, criterion=self.criterion,  max_depth=self.max_depth, 
                            modalities=self.modalities)
            tree.fit(Xb, yb)
            trees.append(tree)
            OOB.append(indices)  
        self.rf,self.OOB= trees, OOB
        return trees, OOB
    
    def _precompute_oob_mapping(self):
        """
        Calculates for which trees is OOB each individual
        """
        all_indices = set()
        for oob_indices in self.OOB:
            all_indices.update(oob_indices)
        n_samples = max(all_indices) + 1 if all_indices else 0

        oob_mapping = {i: [] for i in range(n_samples)}
        for tree, oob_indices in zip(self.rf, self.OOB):
            
            if not isinstance(oob_indices, set):
                oob_set = set(oob_indices)
            else:
                oob_set = oob_indices
            for i in oob_set:
                oob_mapping[i].append(tree)
        self.oob_mapping = oob_mapping

    def _predictRF(self,x):

        preds = np.array([tree.root._predict(x) for tree in self.rf])
        return stats.mode(preds)[0] if self.tree == 'cls' else np.mean(preds)

    def predictRF(self,X):
        return np.array([self._predictRF(x) for x in X])

    def _predict_oob(self,x,i):
        trees = self.oob_mapping.get(i, [])
        preds = [tree.root._predict(x) for tree in trees]
        if not preds:
            return None

        if self.tree == 'cls':
            counts = np.bincount(preds)
            return np.argmax(counts)
        else:
            return np.mean(preds)
        
    
    def predict_OOB(self,X):
        nr = X.shape[0]
        return np.array([self._predict_oob(X[i],i) for i in range(nr)])

    def prob_class_OOB(self,X,y):
            nr = X.shape[0]
            return np.array([np.mean(np.array([tree.root._predict(X[i]) for tree, oob_indices in zip(self.rf, self.OOB) if i in oob_indices])==y[i]) for i in range(nr)])
            
    def _partitions(self,lst):
        return [list(subset) for r in range(len(lst) + 1) for subset in combinations(lst, r)]

            
    def _features_comb(self):
        id_list = [c.id for c in self.modalities]
        part = self._partitions(id_list)
        feature_combinations = [
            sorted(chain.from_iterable(c.features for c in self.modalities if c.id in subset))
            for subset in part
        ]
        return part, feature_combinations
    
    
    def error_estimation_oob(self, X, y, ncomb = 1):
        self.cat_combination, self.feat_comb = self.cat_selection(ncomb)
        tf = set(chain.from_iterable([t.features for t in self.modalities]))
        self.null_feat = [list(tf - set(subset)) for subset in self.feat_comb]
        l = len(self.feat_comb)
        prediction_comb,error_comb = np.zeros((X.shape[0],l)),np.zeros((X.shape[0],l))
        for i in range(l):
            if self.tree=='cls':
                X_t = X.copy(); X_t[:,self.null_feat[i]] = np.nan
                prediction_comb[:,i] = self.predict_OOB(X_t)
                error_comb[:,i] =  1-self.prob_class_OOB(X_t,y)
            else:
                X_t = X.copy(); X_t[:,self.null_feat[i]] = np.nan
                prediction_comb[:,i] = self.predict_OOB(X_t)
                error_comb[:,i] = np.abs(prediction_comb[:,i]-y)
        self.BM_train_estimation = prediction_comb
        self.BM_train_error = error_comb
        
    def _error_estimation_test(self,X_train,X_test):
       dist_matrix = np.linalg.norm(X_train[None, :, self.ch_feat] - X_test[:, None, self.ch_feat], axis=2)  
       self.error_estimation_test = self.BM_train_error[np.argmin(dist_matrix, axis=1)]
       
    def BMOptimization(self,Budget):
        """
        After fitting a customized Random Forest and with the errors estimation, providing a budget value
        the selection of modalities is solved by means of an IP
        """
        CV = [sum(c.CV for c in self.modalities if c.id in i) if len(i) > 0 else 0 for i in self.cat_combination]
        self.Cost_V = CV
        self.distinct_cat = sorted({x for sublista in self.cat_combination for x in sublista})
        self.CF_unique = [sum(c.CF for c in self.modalities if c.id == i) for i in self.distinct_cat]
        self.feature_selection, self.total_cost, self.total_error = _BMOptimization(self.error_estimation_test, CV, self.CF_unique, Budget, self.distinct_cat,self.cat_combination)
   
    def BMOptimization_universal(self,Budget,n):
        """
        After fitting a customized Random Forest and with the errors estimation, providing a budget value
        the selection of modalities is solved by means of an IP for the Universal Prescription
        """
        CV = [sum(c.CV for c in self.modalities if c.id in i) if len(i) > 0 else 0 for i in self.cat_combination]
        CF = [sum(c.CF for c in self.modalities if c.id in i) if len(i) > 0 else 0 for i in self.cat_combination]
        self.Cost_V, self.Cost_F = CV,CF
        self.distinct_cat = sorted({x for sublista in self.cat_combination for x in sublista})
        self.CF_unique = [sum(c.CF for c in self.modalities if c.id == i) for i in self.distinct_cat]
        universal_error = np.mean(self.BM_train_error, axis = 0)
        self.feature_selection, self.total_cost, self.total_error = _BMOptimization_universal(universal_error, CV, self.CF_unique, Budget, self.distinct_cat,self.cat_combination,n)
    
    
    def FeatureSelecion(self,X_train,X_test,y_train,B):
        self.fit_RF(X_train,y_train)
        self.error_estimation_oob(X_train,y_train)
        self._error_estimation_RFbased(X_train,X_test,1)
        self.BMOptimization(B)
        
    def actualError(self,X,y):
        Xc = X.copy()
        for i, feat_idx in enumerate(self.feature_selection):
            Xc[i, self.null_feat[feat_idx]] = np.nan
        yp = self.predictRF(Xc);
        return np.mean(yp==y) if self.tree == 'cls' else np.sum(np.abs(yp-y))
    
    def _terminal_leaf_OOB(self, X):
        X = X.copy()  
        X[:, list(self.total_exp_features)] = np.nan  
    
        hojas_OOB = {}  
        for i, (tree, oob) in enumerate(zip(self.rf, self.OOB)):  
            for idx in oob:  
                x = X[idx]  
                hoja = tree.root._terminal_leaf(x, 1)  
                if idx not in hojas_OOB:
                    hojas_OOB[idx] = {}  
                hojas_OOB[idx][i] = hoja      
        return hojas_OOB
    
    
    def _terminal_leaf_test(self, X):
        X = X.copy()  
        X[:, list(self.total_exp_features)] = np.nan  
        hojas_test = {}  
        for i, tree in enumerate(self.rf):  
            for idx, x in enumerate(X):  
                hoja = tree.root._terminal_leaf(x, 1)  
                if idx not in hojas_test:
                    hojas_test[idx] = {}  
                hojas_test[idx][i] = hoja  
        return hojas_test
    
    def comparar_hojas(self, hojas_OOB, hojas_test, k):
        coincidencias = {}    
        hojas_OOB_sets = {
            idx_muestra: set((idx_arbol, hoja_terminal) for idx_arbol, hoja_terminal in hojas.items())
            for idx_muestra, hojas in hojas_OOB.items()
        }    
        hojas_test_sets = {
            idx_muestra: set((idx_arbol, hoja_terminal) for idx_arbol, hoja_terminal in hojas.items())
            for idx_muestra, hojas in hojas_test.items()
        }
    
        for idx_test, hojas_t_set in hojas_test_sets.items():
            mejores_matches = heapq.nlargest(
                k,
                (    
                    (idx_train, len(hojas_t_set & hojas_o_set)) 
                    for idx_train, hojas_o_set in hojas_OOB_sets.items()
                ),
                key=lambda x: x[1]  
            )
            coincidencias[idx_test] = mejores_matches
    
        return coincidencias


    

    def _error_estimation_RFbased(self, X_train, X_test, k):
        matches = self.comparar_hojas(self._terminal_leaf_OOB(X_train), self._terminal_leaf_test(X_test), k)
    
        error_estimation = np.zeros((len(X_test), self.BM_train_error.shape[1]))  
        error_estimation[:] = np.nan  # Inizalization  with NaN
    
        for i, t in matches.items(): 
            ind = [x[0] for x in t if x[0] < len(X_train)]  
            
            if len(ind) > 0:
                error_estimation[i] = np.nanmean(self.BM_train_error[ind], axis=0)  
            else:
                error_estimation[i] = np.nan  
        
        self.error_estimation_test = error_estimation
        
    def prob_class_predict(self,X,y):
        Xc = X.copy()
        for i, feat_idx in enumerate(self.feature_selection):
            Xc[i, self.null_feat[feat_idx]] = np.nan
        nr = Xc.shape[0]
        return np.sum(np.array([np.mean(np.array([tree.root._predict(Xc[i]) for tree, oob_indices in zip(self.rf, self.OOB)])==y[i]) for i in range(nr)]))

    
    def var_importance(self):
        importancia = {}
        
        def recorrer_nodos(nodo):
            if nodo is None:
                return
            
            variable = nodo.feature
            ganancia = nodo.gain if not nodo.sn_split else 0
            samples = nodo.n_samples
    
            if variable is not None:
                if variable not in importancia:
                    importancia[variable] = 0
                importancia[variable] += ganancia * samples
    
            recorrer_nodos(nodo.left)
            recorrer_nodos(nodo.right)
    
        for arbol in self.rf:
            recorrer_nodos(arbol.root)
    
        
        num_features = max(importancia.keys(), default=-1) + 1  
        importancia_array = np.zeros(num_features)  
    
        for var, imp in importancia.items():
            importancia_array[var] = imp  
    
        # Normalization
        total_importancia = np.sum(importancia_array)
        if total_importancia > 0:
            importancia_array /= total_importancia
    
        return importancia_array
    
    def cat_comb_importance(self):
        """
        Calculation of the relative importance of each possible combination of modalities
        within the Random Forest
        """
        var_imp = self.var_importance()
        # ch_i = np.sum(var_imp[self.ch_feat])
        cat_comb_importance = np.zeros(len(self.feat_comb))
        for i in range(len(cat_comb_importance)):
            cat_comb_importance[i] = np.sum(var_imp[self.feat_comb[i]])
        cat_comb_importance[0] = 1
        self.cat_comb_imp = cat_comb_importance
    
    def cat_count(self):
        cat = [f.id for f in self.modalities]
        d = {i:0 for i in cat}
        total_count = Counter(self.feature_selection)
        for i in cat:
            count = 0
            for j in range(len(self.cat_combination)):
                if i in self.cat_combination[j]:
                    count += total_count[j]
            d[i] = count
        return d
    
    def cat_importance(self, binary = False):
        l = []
        for t in self.rf:
            if binary:
                s = [frozenset(c) for c in t.root.get_mod_comb()]
                l += list(set(s))               
            else:    
                l += t.root.get_mod_comb()
        c = Counter(frozenset(s) for s in l)
        total = 1 if binary else sum(c.values()) 
        nc = {k: v / total for k, v in c.items()}
        return dict(sorted(nc.items(), key=lambda item: item[1], reverse=True))
    
    def cat_importance_g(self):
        
        def unificar_mod_comb_g_normalizado(listas):
            suma = {}
            for lista in listas:
                for s, valor in lista:
                    clave = frozenset(s)
                    suma[clave] = suma.get(clave, 0) + valor
            total = sum(suma.values())
            norm = {clave: valor / total for clave, valor in suma.items()}
            sorted_norm = sorted(norm.items(), key=lambda item: item[1], reverse=True)
    
            cumulative = 0
            sorted_with_cumulative = []
            for clave, value in sorted_norm:
                cumulative += value
                sorted_with_cumulative.append((clave, value, cumulative))
            
            return sorted_with_cumulative

        return unificar_mod_comb_g_normalizado([rf.root.get_mod_comb_g() for rf in self.rf])
         
             
    def cat_selection(self, k):
        """
        Selection of those modalities combinations whose cummulative relative importance add up to the fraction k
        """
        d = self.cat_importance_g()
        temp = [tuple(sorted(item[0] - {0})) for item in d if item[2] < k]
        unique = set(temp)
        
        unique.add(())
        result = sorted(unique, key=lambda t: len(t))
        part =  [list(t) for t in result]
        feature_combinations = [
            sorted(chain.from_iterable(c.features for c in self.modalities if c.id in subset))
            for subset in part
        ]
        return part, feature_combinations
