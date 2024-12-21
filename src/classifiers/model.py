from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np

class RFConfig:
    def __init__(
        self,
        n_estimators = 100, *,
        criterion = 'gini', max_depth = None, min_samples_split = 2,
        min_samples_leaf = 1, min_weight_fraction_leaf=0.0, max_features='sqrt', 
        max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, 
        oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, 
        class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None
    ):
        temp = locals()
        self.__dict__ = {k: v for k, v in temp.items() if k not in ['self', '__class__']}

class BaseClassifier(object):
    def __init__(self):
        pass
    def train(self):
        pass
    def save(self):
        pass
    def predict(self):
        pass

class RandomForestClassifier(BaseClassifier):
    def __init__(self, model_config: RFConfig):
        self.model = RandomForestClassifier(**model_config.__dict__)
        
    def train(self, train_data: np.ndarray, train_config: dict):
        self.model.fit

class XGBoostClassifier(BaseClassifier):
    def __init__(self):
        pass

class XGBChunkClassifier(BaseClassifier):
    def __init__(self):
        pass

        