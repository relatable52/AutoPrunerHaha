import pickle as pkl
import os
from typing import Callable, List, Tuple
import tempfile

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, DataIter, DMatrix
import xgboost as xgb
import numpy as np
from tqdm.auto import tqdm

class BaseClassifier(object):
    def __init__(self):
        pass
    def train(self):
        pass
    def save(self):
        pass
    def predict(self):
        pass
    def load(self):
        pass

class RFClassifier(BaseClassifier):
    def __init__(self, model_config: dict, **kwargs):
        self.model = RandomForestClassifier(**model_config)
        
    def train(self, train_data: np.ndarray, train_labels: np.ndarray, **kwargs):
        self.model.fit(train_data, train_labels)

    def predict(self, test_data: np.ndarray, **kwargs):
        return self.model.predict_proba(test_data)
    
    def save(self, path: str):
        with open(path, 'wb') as f:
            pkl.dump(self.model, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            self.model = pkl.load(f)

class XGBoostClassifier(BaseClassifier):
    def __init__(self, model_config: dict, **kwargs):
        self.model = XGBClassifier(**model_config)
    
    def train(self, train_data: np.ndarray, train_labels: np.ndarray, **kwargs):
        self.model.fit(train_data, train_labels)
    
    def predict(self, test_data: np.ndarray, **kwargs):
        return self.model.predict_proba(test_data, **kwargs)
    
    def save(self, path: str):
        self.model.save_model(path)

    def load(self, path: str):
        self.model.load_model(path)

class XGBChunkClassifier(BaseClassifier):
    def __init__(self, model_config: dict, **kwargs):
        self.params = model_config

    def make_batches(self, X: np.ndarray, y: np.ndarray, batch_size: int, tmpdir: str):
        n_samples = X.shape[0]

        files: List[Tuple[str, str]] = []

        for i in tqdm(range(int(np.ceil(n_samples/batch_size)))):
            end = (i+1)*batch_size
            end = n_samples if end>n_samples else end

            data = X[i*batch_size:end]
            labels = y[i*batch_size:end]

            X_path = os.path.join(tmpdir, "X-" + str(i) + ".npy")
            y_path = os.path.join(tmpdir, "y-" + str(i) + ".npy")
            
            np.save(X_path, data)
            np.save(y_path, labels)
            
            files.append((X_path, y_path))
        return files
    
    def train(self, train_data: np.ndarray, train_labels: np.ndarray, batch_size: int, train_config: dict, **kwargs):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_files = self.make_batches(train_data, train_labels, batch_size, tmpdir)
            it_train = XGBIterator(train_files)
            
            missing = np.nan
            Xy_train = DMatrix(it_train, missing=missing, enable_categorical=False)

            self.booster = xgb.train(self.params, Xy_train, **train_config)


    def predict(self, test_data: np.ndarray, batch_size: int, **kwargs):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_files = self.make_batches(test_data, np.zeros(len(test_data)), batch_size, tmpdir)
            it_test = XGBIterator(test_files)

            missing = np.nan
            Xy_test = DMatrix(it_test, missing=missing, enable_categorical=False)

            preds = self.booster.predict(Xy_test)
        
        return preds

    def save(self, path: str):
        self.booster.save_model(path)
        
    def load(self, path: str):
        self.booster = xgb.Booster()
        self.booster.load_model(path)

class XGBIterator(DataIter):
    """A custom iterator for loading files in batches."""

    def __init__(self, file_paths: List[Tuple[str, str]]) -> None:
        self._file_paths = file_paths
        self._it = 0
        # XGBoost will generate some cache files under current directory with the prefix
        # "cache"
        super().__init__(cache_prefix=os.path.join(".", "cache"))

    def load_file(self) -> Tuple[np.ndarray, np.ndarray]:
        X_path, y_path = self._file_paths[self._it]
        X = np.load(X_path)
        y = np.load(y_path)
        assert X.shape[0] == y.shape[0]
        return X, y

    def next(self, input_data: Callable) -> int:
        """Advance the iterator by 1 step and pass the data to XGBoost.  This function is
        called by XGBoost during the construction of ``DMatrix``

        """
        if self._it == len(self._file_paths):
            # return 0 to let XGBoost know this is the end of iteration
            return 0

        # input_data is a function passed in by XGBoost who has the similar signature to
        # the ``DMatrix`` constructor.
        X, y = self.load_file()
        input_data(data=X, label=y)
        self._it += 1
        return 1

    def reset(self) -> None:
        """Reset the iterator to its beginning"""
        self._it = 0

        