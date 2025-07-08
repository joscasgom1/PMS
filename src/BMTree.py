from collections import namedtuple, Counter
import numpy as np
import math
import random


Modality = namedtuple('Modality', ['features','CV','CF', 'id'])

class BM_Tree:
    def __init__(self, tree='cls', criterion='gini', max_depth=4, modalities=[]):
        """
        Initialization of a tree instance considering the Preliminary splits:
        - tree: type of tree, classification ('cls') or regression ('reg')
        - criterion: impurity reduction criterion; 'gini' and 'entropy' for classification, 'mse' for regression
        - max_depth: maximum depth of the tree
        - modalities: Supplementary Modalities considered. These are included as a tuple called 'Modality', composed of:
                      the features that define them, their variable and fixed costs, and an ID indicating which 
                      Supplementary Modality it is.
        Note: Features that are not included in any Supplementary Modality form the Baseline Modality.
        """
        self.feature = None
        self.label = None
        self.n_samples = None
        self.gain = None
        self.left = None
        self.right = None
        self.threshold = None
        self.depth = 0
        self.sn_split = False
        self.cat_used = set()
        
        self.tree = tree
        self.criterion = criterion
        self.max_depth = max_depth
        self.modalities = modalities
        self.fs = None
        self.root = None
        
        
    def fit(self, features, target):
        feature_counts = Counter(f for cat in self.modalities for f in cat.features)
        if any(count > 1 for count in feature_counts.values()):
            raise ValueError("Not excluding modalities")
        
        feat_unique = np.array(list(feature_counts.keys()))
        if np.any((feat_unique < 0) | (feat_unique >= features.shape[1])):
            raise ValueError("Out of range modalities")
        
        if 0 in np.array([c.id for c in self.modalities]):
            raise ValueError("Id 0 keyword for Baseline Modality")
        
        self.root = BM_Tree(tree=self.tree, criterion=self.criterion, max_depth=self.max_depth, modalities=self.modalities)
        self.root.fs = self._feature_selection(features.shape[1])
        self.root._grow_tree(features, target, self.criterion)
        
    def _feature_selection(self,f):
        m = {}
        for t in self.modalities:
            for n in t.features:
                m[n] = (t.id)
        return [(n, m.get(n, 0)) for n in range(f)]
    
    
    def _grow_tree(self, features, target, criterion):
        
        self.n_samples = len(target)
        if len(np.unique(target)) == 1:
            self.label = target[0]
            return
        
        self.label = max([(c, np.sum(target == c)) for c in np.unique(target)], key=lambda x: x[1])[0] if criterion in {'gini', 'entropy'} else np.mean(target)
       
        if criterion in {'gini', 'entropy'}:            
            self.label = int(self.label) 
                
        if self.depth >= self.max_depth:
            return
        
        impurity_node = self._calc_impurity(criterion, target)
        
        self.gain, self.feature, self.threshold, self.best_id = self._cat_split(features, target, criterion, impurity_node)
               
        if self.threshold is None:
            self.impurity = True
            return
        
        if self.best_id > 0:
            self.sn_split = True
            self._split_tree(features, target, criterion, True, self.best_id)
            return        
        self.cat_used.add(self.best_id)
        self._split_tree(features, target, criterion)
        
    def _calc_impurity(self, criterion, target):
        if len(target) == 0:
            return 0
        
        if criterion == 'gini':
            counts = np.bincount(target, minlength=np.max(target) + 1)
            probs = counts / target.shape[0]
            return 1.0 - np.sum(probs ** 2)
        
        elif criterion == 'mse':
            return np.var(target)
        
        else:  
            counts = np.bincount(target, minlength=np.max(target) + 1)
            probs = counts / target.shape[0]
            return -np.sum(probs * np.log2(probs, where=probs > 0))
    
    def _shuffle_variable(self, n):

        return random.sample(self.fs, n)
             
    def _cat_split(self, features, target, criterion, impurity_node):
        
        if not self.fs or len(np.unique(target)) <= 1:
            return 0, None, None, None, None
    
        best_gain, best_feature, best_threshold, best_id = 0.0, None, None, None

        n_block = len(target)
        
        n_var = max(1,len(self.fs) // 3) if self.tree == 'reg' else math.isqrt(len(self.fs))
        
        self.selected_var = self._shuffle_variable(n_var)
    
        for col, node_id in self.fs:
            sorted_idx = np.argsort(features[:, col], kind='stable')
            sorted_features = features[sorted_idx, col]
            sorted_target = target[sorted_idx]
    
            unique_vals, idx, counts = np.unique(sorted_features, return_index=True, return_counts=True)
            if len(unique_vals) < 2:
                continue  
    
            thresholds = (unique_vals[:-1] + unique_vals[1:]) * 0.5
            
    
            count_l = np.cumsum(counts)[:-1]  
            count_r = n_block - count_l  
    
            valid_splits = (count_l > 0) & (count_r > 0)
            if not np.any(valid_splits):
                continue  
    
            if criterion == "gini":
                unique_classes, encoded_target = np.unique(sorted_target, return_inverse=True)  
                num_classes = len(unique_classes)  
                
                hist_cumsum = np.cumsum(np.eye(num_classes)[encoded_target], axis=0)   
                            
                total_count = hist_cumsum[-1]
                count_l = hist_cumsum[idx[1:] - 1] 
                count_r = total_count - count_l  
            
                prob_l = count_l / np.maximum(count_l.sum(axis=1, keepdims=True), 1)
                prob_r = count_r / np.maximum(count_r.sum(axis=1, keepdims=True), 1)
            
                impurity_l = 1.0 - np.sum(prob_l**2, axis=1)
                impurity_r = 1.0 - np.sum(prob_r**2, axis=1)
                
                mean_l = np.sum(count_l, axis=1) / n_block
                mean_r = np.sum(count_r, axis=1) / n_block

            
            elif criterion == "entropy":
                unique_classes, encoded_target = np.unique(sorted_target, return_inverse=True)  
                num_classes = len(unique_classes)  
                
                hist_cumsum = np.cumsum(np.eye(num_classes)[encoded_target], axis=0)   
                            
                total_count = hist_cumsum[-1]
                count_l = hist_cumsum[idx[1:] - 1] 
                count_r = total_count - count_l  
            
                prob_l = count_l / np.maximum(count_l.sum(axis=1, keepdims=True), 1)
                prob_r = count_r / np.maximum(count_r.sum(axis=1, keepdims=True), 1)
        
                impurity_l = -np.sum(prob_l * np.log2(prob_l + 1e-9), axis=1)
                impurity_r = -np.sum(prob_r * np.log2(prob_r + 1e-9), axis=1)
                
                mean_l = np.sum(count_l, axis=1) / n_block
                mean_r = np.sum(count_r, axis=1) / n_block
    
            elif criterion == "mse":
                sum_cumsum = np.cumsum(sorted_target)
                sum_sq_cumsum = np.cumsum(sorted_target ** 2)
    
                total_sum = sum_cumsum[-1]
                total_sum_sq = sum_sq_cumsum[-1]
    
                sum_l = sum_cumsum[idx[1:] - 1]
                sum_r = total_sum - sum_l
    
                sum_sq_l = sum_sq_cumsum[idx[1:] - 1]
                sum_sq_r = total_sum_sq - sum_sq_l
    
                count_l = np.maximum(count_l, 1)
                count_r = np.maximum(count_r, 1)
    
                impurity_l = (sum_sq_l / count_l) - (sum_l / count_l) ** 2
                impurity_r = (sum_sq_r / count_r) - (sum_r / count_r) ** 2
    
                mean_l = count_l / n_block
                mean_r = count_r / n_block
                
            gains = impurity_node - mean_l * impurity_l - mean_r * impurity_r #* factor  
    
            max_idx = np.argmax(gains)
       
            max_idx = np.argmax(gains)
            
            if gains[max_idx] > best_gain:
                best_gain = gains[max_idx]
                best_feature = col
                best_threshold = thresholds[max_idx]
                best_id = node_id
    
        return best_gain, best_feature, best_threshold, best_id

    def _split_tree(self, features, target, criterion, preliminary=False, id_remove=None):
        mask = features[:, self.feature] <= self.threshold
                
        features_left = features[mask]
        target_left = target[mask]
        features_right = features[~mask]
        target_right = target[~mask]
        
        if preliminary:
            fs_l = [(n, 0) if feat_id == id_remove else (n,  feat_id) for n, feat_id in self.fs]
            fs_r = [(n, feat_id) for n,  feat_id in self.fs if feat_id != id_remove]
            self.right = BM_Tree(tree=self.tree, criterion=self.criterion, max_depth=self.max_depth)
            self.right.depth = self.depth 
            self.right.fs = fs_r
            self.right.cat_used = self.cat_used.copy()

            self.right._grow_tree(features, target, criterion)
            
            self.left = BM_Tree(tree=self.tree, criterion=self.criterion, max_depth=self.max_depth)
            self.left.depth = self.depth + 1
            self.left.fs = fs_l
            self.left.n_samples = self.n_samples
            self.left.feature = self.feature
            self.left.threshold = self.threshold
            self.left.gain = self.gain
            self.left.split_cat = True
            self.left.label = self.label
            self.left.cat_used = self.cat_used.copy()
            self.left.cat_used.add(id_remove)
            for side, data in zip(["left", "right"], [(features_left, target_left), (features_right, target_right)]):
                subtree = BM_Tree(tree=self.tree, criterion=self.criterion, max_depth=self.max_depth)
                subtree.depth = self.depth + 1
                subtree.fs = fs_l
                subtree.cat_used = self.left.cat_used
                subtree._grow_tree(*data, criterion)  
                setattr(self.left, side, subtree) 
        else:        
            self.left = self._create_subtree(features_left, target_left, criterion)
            self.right = self._create_subtree(features_right, target_right, criterion)
        
        return

    def _create_subtree(self, features, target, criterion):    
        side = BM_Tree(tree=self.tree, criterion=self.criterion, max_depth=self.max_depth)
        side.depth = self.depth + 1
        side.fs = self.fs
        side.cat_used = self.cat_used.copy()
           
        side._grow_tree(features, target, criterion)
        
        return side
    
    def print_tree(self):
        self.root._show_tree(0, ' ')
    
    def _show_tree(self, depth, cond):
        prefix = '    ' * depth + cond
        if self.feature is not None:
            print(f"{prefix}if X[{self.feature}] {'not null' if self.sn_split else '<= ' + str(self.threshold)}")
            self.left._show_tree(depth + 1, 'then ')
            self.right._show_tree(depth + 1, 'else ')
        else:
            print(f"{prefix}{{value: {self.label}, samples: {self.n_samples}}}")

    def get_mod_comb(self, terminal_features=None):
        if terminal_features is None:
            terminal_features = [] 
        
        if self is not None:
            if self.threshold is None:  
                terminal_features.append(self.cat_used) 
            else: 
                if self.left:
                    self.left.get_mod_comb(terminal_features)  
                if self.right:
                    self.right.get_mod_comb(terminal_features)  
    
        return terminal_features 
    
    def get_mod_comb_g(self, terminal_features=None,gain = 0):
        if terminal_features is None:
            terminal_features = [] 
        
        if self is not None:
            if self.threshold is None:  
                terminal_features.append((self.cat_used,gain)) 
            else: 
                gain_nodo = self.gain if not self.sn_split else 0
                samples = self.n_samples
                gain += gain_nodo*samples
                if self.left.threshold is None and self.right.threshold is None:
                    terminal_features.append((self.cat_used,gain))
                else:
                    self.left.get_mod_comb_g(terminal_features,gain)  
                    self.right.get_mod_comb_g(terminal_features,gain)  
    
        return terminal_features 
   
    def get_mod_comb_g_2(self, feat_baseline, terminal_features=None,gain = 0):
        if terminal_features is None:
            terminal_features = [] 
        
        if self is not None:
            if self.threshold is None:  
                terminal_features.append((self.cat_used,gain)) 
            else: 
                gain_nodo = self.gain if not self.sn_split else 0
                gain_nodo = gain_nodo if self.feature not in feat_baseline else 0
                samples = self.n_samples
                gain += gain_nodo*samples
                if self.left.threshold is None and self.right.threshold is None:
                    terminal_features.append((self.cat_used,gain))
                else:
                    self.left.get_mod_comb_g_2(feat_baseline,terminal_features,gain)  
                    self.right.get_mod_comb_g_2(feat_baseline,terminal_features,gain)  
    
        return terminal_features 
    
    def predict(self, features):
        return np.array([self.root._predict(f) for f in features])
    
    def _predict(self, d):
        if self.feature != None:
           if self.sn_split == True:
               if not np.isnan(d[self.feature]):
                   return self.left._predict(d)
               else:
                   return self.right._predict(d)
           else:
               if d[self.feature] <= self.threshold:
                   return self.left._predict(d)
               else:
                   return self.right._predict(d)
        else: 
            return self.label
        
    def  _terminal_leaf(self, d, n):
        if self.feature != None:
           if self.sn_split == True:
               if not np.isnan(d[self.feature]):
                   return self.left._terminal_leaf(d,2*n)
               else:
                   return self.right._terminal_leaf(d,2*n+1)
           else:
               if d[self.feature] <= self.threshold:
                   return self.left._terminal_leaf(d,2*n)
               else:
                   return self.right._terminal_leaf(d,2*n+1)
        else: 
            return n

