from argparse import ArgumentParser
import pickle as pkl
import os
import json

from src.classifiers.model import RFClassifier, XGBoostClassifier, XGBChunkClassifier
from src.utils.utils import read_config_file

import pandas as pd
import numpy as np
import statistics
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score, recall_score, fbeta_score, 
    classification_report, roc_curve, auc, precision_recall_curve
)

def get_args():
    parser = ArgumentParser(description="Train a classifier")
    parser.add_argument("--config_path", required=True, type=str, default="config/wala.config", help="Path to the config file")
    parser.add_argument("--classifier_name", required=True, type=str, default="rf", help="Name of the classifier to train")
    parser.add_argument("--mode", type=str, default="both", help="Mode: train, test or both")
    parser.add_argument("--data_features", type=str, default="all", help="Features to use: semantic, structure or all")
    parser.add_argument("--run_config", required=True, type=str, default="config/run_config.json", help="Path to the model config file")
    parser.add_argument("--output_prefix", type=str, default="njr1", help="Prefix for output files")
    return parser.parse_args()

def get_data(pkl_file: str, mode: str):
    with open(pkl_file, 'rb') as f:
        data = pkl.load(f)
    
    structure_data = np.vstack(data["struct"])
    semantic_data = np.vstack(data["code"]) if len(data["code"]) > 0 else np.empty((structure_data.shape[0], 0))
    all_data = np.hstack([structure_data, semantic_data])
    labels = np.stack(data['target'])
    ids = np.stack(data['program_ids'])
    static_ids = np.stack(data['static_ids'])
    if mode == "all":
        return all_data, labels, static_ids, ids
    elif mode == "semantic":
        return semantic_data, labels, static_ids, ids
    elif mode == "structure":
        return structure_data, labels, static_ids, ids
    else:
        raise NotImplemented

def get_data_by_id(pkl_file: str, mode: str):
    features, labels, static_ids, ids = get_data(pkl_file, mode)

    program_ids = np.unique(ids)
    test_split = {pid: {"features": [], "labels": [], "static_ids": []} for pid in program_ids}

    for pid, feature, label, static_id in zip(ids, features, labels, static_ids):
        test_split[pid]["features"].append(feature)
        test_split[pid]["labels"].append(label)
        test_split[pid]["static_ids"].append(static_id)
    return test_split

model_dict = {
    "rf": RFClassifier,
    "xgb": XGBoostClassifier,
    "xgb_chunk": XGBChunkClassifier
}

extension_dict = {
    "rf": "pkl",
    "xgb": "json",
    "xgb_chunk": "json"
}

def tune_model_with_optuna(classifier_name: str, X, y, run_config: dict):
    def rf_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 5, 50),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False])
        }
        model = RandomForestClassifier(**params)
        return cross_val_score(model, X, y, cv=3, scoring='accuracy').mean()

    def xgb_objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 1)
        }
        model = XGBClassifier(**params)
        return cross_val_score(model, X, y, cv=3, scoring='accuracy').mean()

    study = optuna.create_study(direction="maximize")
    if classifier_name == "rf":
        study.optimize(rf_objective, n_trials=run_config.get("tune_trials", 20))
    elif classifier_name == "xgb":
        study.optimize(xgb_objective, n_trials=run_config.get("tune_trials", 20))
    else:
        raise ValueError(f"Tuning not supported for {classifier_name}")

    return study.best_trial.params

def train_classifier(
        classifier_name: str, train_data: np.ndarray, train_labels: np.ndarray, 
        save_dir: str, run_config: dict = {}, mode: str = "all"
    ):
    assert classifier_name in model_dict, f"Classifier {classifier_name} not found"

     # Check if tuning is requested
    if run_config.get("tune", False):
        best_params = tune_model_with_optuna(classifier_name, train_data, train_labels, run_config)
        model_config = best_params
    else:
        model_config = run_config.get("model_config", {})
    
    train_config = run_config.get("train_config", {})
    
    model = model_dict[classifier_name](model_config)
    model.train(train_data, train_labels, **train_config)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, f"{classifier_name}_{mode}.{extension_dict[classifier_name]}")
    model.save(save_path)
    return model

def evaluate(model, test_data: np.ndarray, test_labels: np.ndarray, predict_config: dict = {}):
    rf = model

    preds = rf.predict(test_data, **predict_config)

    if len(preds.shape) == 2:
        preds = preds[:, 1]

    y_test = pd.Series(test_labels)
    y_preds = pd.Series(np.where(preds>0.45, 1, 0))
    y_preds_proba = pd.Series(preds)
    print(classification_report(y_test, y_preds))

    fpr, tpr, thresh = roc_curve(y_test, y_preds_proba)
    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_preds_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)  # Avoid division by zero
    auroc = auc(fpr, tpr)
    auprc = auc(recall, precision)
    f1_score = fbeta_score(y_test, y_preds, beta=1)
    f2_score = fbeta_score(y_test, y_preds, beta=2)
    print("AUROC score: ", auroc)
    print("AUPRC score: ", auprc)
    print("f1 score: ", f1_score)
    print("f2 score: ", f2_score)

def evaluateByProgram(model, test_split, threshold=0.5, predict_config: dict = {}, return_predictions=False):
    precision_avg = []
    recall_avg = []
    f1_avg = []
    
    predictions_by_program = {}  # To store predictions and labels

    for pid in test_split:
        # Prepare data
        program_features = np.array(test_split[pid]["features"])
        program_labels = np.array(test_split[pid]["labels"])
        program_static_ids = np.array(test_split[pid]["static_ids"])
    
        # Predict probabilities
        output = model.predict(program_features, **predict_config)
        if len(output.shape) == 2:
            output = output[:, 1]
        
        # Save outputs for later use
        predictions_by_program[pid] = {
            "output": output,
            "labels": program_labels,
            "static_mask": program_static_ids
        }
    
        # Compute binary prediction using threshold
        pred = np.where(output >= threshold, 1, 0)
    
        # Metrics
        precision = precision_score(program_labels, pred, zero_division=0)
        recall = recall_score(program_labels, pred, zero_division=0)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    
        precision_avg.append(precision)
        recall_avg.append(recall)
        f1_avg.append(f1)
    
    # Average + stddev
    precision_mean = round(statistics.mean(precision_avg), 2)
    precision_std = round(statistics.stdev(precision_avg), 2) if len(precision_avg) > 1 else 0
    recall_mean = round(statistics.mean(recall_avg), 2)
    recall_std = round(statistics.stdev(recall_avg), 2) if len(recall_avg) > 1 else 0
    f1_mean = round(statistics.mean(f1_avg), 2)
    f1_std = round(statistics.stdev(f1_avg), 2) if len(f1_avg) > 1 else 0
    
    print(f"[EVAL-AVG] Threshold {threshold} | "
          f"Precision {precision_mean} ({precision_std}), "
          f"Recall {recall_mean} ({recall_std}), "
          f"F1 {f1_mean} ({f1_std})")

    if return_predictions:
        return predictions_by_program


def save_predictions(
        classifier_name: str, test_data: np.ndarray,
        model_path: str, save_dir: str, run_config: dict = {}, name: str = "all"
    ):
    assert classifier_name in model_dict, f"Classifier {classifier_name} not found"
    
    model_config = run_config.get("model_config", {})
    predict_config = run_config.get("predict_config", {})
    
    model = model_dict[classifier_name](model_config)
    model.load(model_path)
    predictions = model.predict(test_data, **predict_config)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(os.path.join(save_dir, f"{classifier_name}_{name}_predictions.npy"), predictions)

    return predictions

def main():
    args = get_args()
    config = read_config_file(args.config_path)
    classifier_name = args.classifier_name
    mode = args.mode
    data_features = args.data_features
    run_config = args.run_config
    model_name = config.get("MODEL", None)
    output_prefix = args.output_prefix
    
    with open(run_config, 'r') as f:
        run_config = json.load(f)

    predict_config = run_config.get("predict_config", {})

    data_dir = config["CACHE_DIR"]
    if model_name is not None:
        data_dir = os.path.join(data_dir, model_name)
    train_data_path = os.path.join(data_dir, "ft_train.pkl")
    test_data_path = os.path.join(data_dir, "ft_test.pkl")
    save_dir = config["CLASSIFIER_MODEL_DIR"]  

    if mode == "train" or mode == "both":
        train_data, train_labels, _ = get_data(train_data_path, data_features)
        model = train_classifier(classifier_name, train_data, train_labels, save_dir, run_config, data_features)
    if mode == "test" or mode == "both":
        if mode == "test":
            model = model_dict[classifier_name](run_config.get("model_config", {}))

        model_path = os.path.join(save_dir, f"{classifier_name}_{data_features}.{extension_dict[classifier_name]}")
        model.load(model_path)
        
        test_data, test_labels, _, _ = get_data(test_data_path, data_features)
        test_split = get_data_by_id(test_data_path, data_features)
        evaluate(model, test_data, test_labels, predict_config=predict_config)
        predictions = evaluateByProgram(model, test_split, predict_config=predict_config, return_predictions=True)
        with open(os.path.join(save_dir, f"{output_prefix}_{classifier_name}_{data_features}_predictions.pkl"), 'wb') as f:
            pkl.dump(predictions, f)
        # save_predictions(classifier_name, test_data, 
        #                  os.path.join(save_dir, f"{classifier_name}_{data_features}.{extension_dict[classifier_name]}"), 
        #                  save_dir, run_config, data_features)

if __name__ == "__main__":
    main()