import xgboost as xgb
import pandas as pd
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import tempfile
from typing import Callable, List, Tuple
from tqdm import tqdm
from src.utils.utils import read_config_file
from sklearn.metrics import (
    fbeta_score,
    classification_report, 
    roc_curve, 
    precision_recall_curve,
    auc
)

import datetime

class Logger:
    def __init__(self, name):
        self.name = name

    def log(self, text):
        filepath = f'log/{self.name}.txt'
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        with open(filepath, 'a') as f:
            f.write(datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S') + text + '\n')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default="../temporary")
    parser.add_argument("--feature", type=int, default=2)
    parser.add_argument("--batchsize", type=int, default=20000)
    parser.add_argument("--boost_rounds", type=int, default=5)
    return parser.parse_args()

class Iterator(xgb.DataIter):
    def __init__(self, tmpdir, data_folder: str,  mode: str = "train", feature: int = 2, batch_size: int = 1):
        assert batch_size > 0
        
        # feature = 0: structural features only
        # feature = 1: semantic feature only
        # feature = 2: all features
        if (mode not in ("train", "test")) or (feature not in (0, 1, 2)):
            raise NotImplementedError

        self.mode = mode
        self.feature = feature
        self.batch_size = batch_size
        self.data_folder = data_folder
        self.tmpdir = tmpdir
        self._file_paths = self.__load_dataset()
        self._it = 0
        # XGBoost will generate some cache files under current directory with the prefix
        # "cache"
        super().__init__(cache_prefix=os.path.join(".", "cache"))

    def __load_dataset(self):
        with open(os.path.join(self.data_folder, f"ft_{self.mode}.pkl"), "rb") as f:
            data = pkl.load(f)
        
        n_samples = len(data["target"])

        files: List[Tuple[str, str]] = []
        for i in tqdm(range(int(np.ceil(n_samples/self.batch_size)))):
            start = i*self.batch_size
            end = (i+1)*self.batch_size
            end = n_samples if end>n_samples else end

            struct = np.array(data["struct"][start:end])
            code = np.array(data["code"][start:end]) 

            X = np.concatenate((struct, code), axis=1)
            y = np.array(data["target"][start:end])

            X_path = os.path.join(self.tmpdir, "X-" + str(i) + ".npy")
            y_path = os.path.join(self.tmpdir, "y-" + str(i) + ".npy")

            np.save(X_path, X)
            np.save(y_path, y)

            files.append((X_path, y_path))
        return files

    def load_file(self) -> Tuple[np.ndarray, np.ndarray]:
        X_path, y_path = self._file_paths[self._it]

        if self.feature == 0:
            X = np.load(X_path)[:,:22]
        elif self.feature == 1:
            X = np.load(X_path)[:,22:]
        elif self.feature == 2:
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

def main(tmpdir: str):
    args = get_args()
    data_folder = args.data_folder
    feature = args.feature
    batchsize = args.batchsize
    boost_rounds = args.boost_rounds

    log = Logger(f"xgbpruner-{feature}-{batchsize}-{boost_rounds}")

    train_iterator = Iterator(tmpdir, data_folder, "train", feature, batchsize)
    test_iterator = Iterator(tmpdir, data_folder, "test", feature, batchsize)

    missing = np.nan
    Xy_train = xgb.DMatrix(train_iterator, missing=missing, enable_categorical=False)
    Xy_test = xgb.DMatrix(test_iterator, missing=missing, enable_categorical=False)

    params = {
        "colsample_bynode": 0.8,
        "learning_rate": 1,
        "max_depth": 5,
        "num_parallel_tree": 100,
        "objective": "binary:logistic",
        "eval_metric": "error",
        "subsample": 0.8,
        "tree_method": "hist"
    }

    booster = xgb.train(
        params,
        Xy_train,
        evals=[(Xy_train, "eval")],
        verbose_eval=True,
        num_boost_round=boost_rounds
    )
        
    preds = booster.predict(Xy_test)

    with open(os.path.join(data_folder, "ft_test.pkl"), "rb") as f:
        y_test = pd.Series(pkl.load(f)["target"])
    
    y_preds = pd.Series(np.where(preds>0.45, 1, 0))
    y_preds_proba = pd.Series(preds)

    print("Classification report:")
    print(classification_report(y_test, y_preds))
    log.log(classification_report)

    fpr, tpr, thresh_1 = roc_curve(y_test, y_preds_proba)
    pre, rec, thresh_2 = precision_recall_curve(y_test, y_preds_proba)
    auroc = auc(fpr, tpr)
    auprc = auc(rec, pre)
    f1_score = fbeta_score(y_test, y_preds, beta=1)
    f2_score = fbeta_score(y_test, y_preds, beta=2)
    res = f"AUROC score: {auroc}"+"\n"+f"AUPRC score: {auprc}"+"n"+f"f1 score: {f1_score}"+"\n"+f"f1 score: {f2_score}"
    print(res)
    log.log(res)

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    # Plot ROC Curve
    ax[0, 0].plot(fpr, tpr, color='blue')
    ax[0, 0].set_title('ROC Curve')
    ax[0, 0].set_xlabel('False Positive Rate')
    ax[0, 0].set_ylabel('True Positive Rate')

    # Plot Precision-Recall Curve
    ax[0, 1].plot(rec, pre, color='green')
    ax[0, 1].set_title('Precision-Recall Curve')
    ax[0, 1].set_xlabel('Recall')
    ax[0, 1].set_ylabel('Precision')

    # Plot Precision by Threshold
    ax[1, 0].plot(thresh_2, pre[:-1], color='red')
    ax[1, 0].set_title('Precision by Threshold')
    ax[1, 0].set_xlabel('Threshold')
    ax[1, 0].set_ylabel('Precision')

    # Plot Recall by Threshold
    ax[1, 1].plot(thresh_2, rec[:-1], color='orange')
    ax[1, 1].set_title('Recall by Threshold')
    ax[1, 1].set_xlabel('Threshold')
    ax[1, 1].set_ylabel('Recall')

    # Adjust layout
    plt.tight_layout()
    plt.savefig("xgb_result.png")

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        main(tmpdir)