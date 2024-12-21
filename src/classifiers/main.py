from argparse import ArgumentParser
import pickle as pkl
import os
import json

from src.classifiers.model import RFClassifier, XGBClassifier, XGBChunkClassifier
from src.utils.utils import read_config_file

import pandas as pd
import numpy as np
import statistics
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
    return parser.parse_args()

def get_data(pkl_file: str, mode: str):
    with open(pkl_file, 'rb') as f:
        data = pkl.load(f)
    
    semantic_data = np.vstack(data["code"])
    structure_data = np.vstack(data["struct"])
    all_data = np.hstack([structure_data, semantic_data])
    labels = np.vstack(data['target'])
    ids = np.vstack(data['program_ids'])
    if mode == "all":
        return all_data, labels, ids
    elif mode == "semantic":
        return semantic_data, labels, ids
    elif mode == "structure":
        return structure_data, labels, ids
    else:
        raise NotImplemented

def get_data_by_id(pkl_file: str, mode: str):
    features, labels, ids = get_data(pkl_file, mode)

    program_ids = np.unique(ids)
    test_split = {pid: {"features": [], "labels": []} for pid in program_ids}

    for pid, feature, label in zip(ids, features, labels):
        test_split[pid]["features"].append(feature)
        test_split[pid]["labels"].append(label)
    return test_split

model_dict = {
    "rf": RFClassifier,
    "xgb": XGBClassifier,
    "xgb_chunk": XGBChunkClassifier
}

extention_dict = {
    "rf": "pkl",
    "xgb": "json",
    "xgb_chunk": "json"
}

def train_classifier(
        classifier_name: str, train_data: np.ndarray, train_labels: np.ndarray, 
        save_dir: str, run_config: dict = {}, mode: str = "all"
    ):
    assert classifier_name in model_dict, f"Classifier {classifier_name} not found"
    
    model_config = run_config.get("model_config", {})
    train_config = run_config.get("train_config", {})
    
    model = model_dict[classifier_name](model_config)
    model.train(train_data, train_labels, **train_config)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, f"{classifier_name}_{mode}.{extention_dict[classifier_name]}")
    model.save(save_path)
    return model

def evaluate(model, test_data: np.ndarray, test_labels: np.ndarray):
    rf = model

    preds = rf.predict(test_data)

    if preds.shape[1] == 2:
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

def evaluateByProgram(model, test_split, threshold=0.5):
    precision_avg = []
    recall_avg = []
    f1_avg = []
    
    for pid in test_split:
        # Prepare data for this program
        program_features = np.array(test_split[pid]["features"])
        program_labels = np.array(test_split[pid]["labels"])
    
        # Predict probabilities for the positive class
        output = model.predict(program_features)
        if output.shape[1] == 2:
            output = output[:, 1]
    
        # Convert probabilities to binary predictions using threshold 0.5
        pred = np.where(output >= threshold, 1, 0)
    
        # Calculate precision and recall
        precision = precision_score(program_labels, pred, zero_division=0)
        recall = recall_score(program_labels, pred, zero_division=0)
    
        # Avoid division by zero in F1 calculation
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
    
        # Append metrics for this program
        precision_avg.append(precision)
        recall_avg.append(recall)
        f1_avg.append(f1)
    
    # Calculate mean and standard deviation of metrics across programs
    precision_mean = round(statistics.mean(precision_avg), 2)
    precision_std = round(statistics.stdev(precision_avg), 2) if len(precision_avg) > 1 else 0
    recall_mean = round(statistics.mean(recall_avg), 2)
    recall_std = round(statistics.stdev(recall_avg), 2) if len(recall_avg) > 1 else 0
    f1_mean = round(statistics.mean(f1_avg), 2)
    f1_std = round(statistics.stdev(f1_avg), 2) if len(f1_avg) > 1 else 0
    
    # Log results
    print(f"[EVAL-AVG] Precision {precision_mean} ({precision_std}), "
          f"Recall {recall_mean} ({recall_std}), "
          f"F1 {f1_mean} ({f1_std})")

def save_predictions(
        classifier_name: str, test_data: np.ndarray, test_labels: np.ndarray, 
        model_path: str, save_dir: str, run_config: dict = {}, mode: str = "all"
    ):
    assert classifier_name in model_dict, f"Classifier {classifier_name} not found"
    
    model_config = run_config.get("model_config", {})
    predict_config = run_config.get("predict_config", {})
    
    model = model_dict[classifier_name](model_config)
    model.load(model_path)
    predictions = model.predict(test_data, predict_config)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(os.path.join(save_dir, f"{classifier_name}_{mode}_predictions.npy"), predictions)

    return predictions

def main():
    args = get_args()
    config = read_config_file(args.config_path)
    classifier_name = args.classifier_name
    mode = args.mode
    data_features = args.data_features
    run_config = args.run_config
    
    with open(run_config, 'r') as f:
        run_config = json.load(f)

    data_dir = config["CACHE_DIR"]
    train_data_path = os.path.join(data_dir, "ft_train.pkl")
    test_data_path = os.path.join(data_dir, "ft_test.pkl")
    save_dir = config["CLASSIFIER_MODEL_DIR"]  

    if mode == "train" or mode == "both":
        train_data, train_labels, _ = get_data(train_data_path, data_features)
        model = train_classifier(classifier_name, train_data, train_labels, save_dir, run_config, data_features)
    if mode == "test" or mode == "both":
        test_data, test_labels, _ = get_data(test_data_path, data_features)
        test_split = get_data_by_id(test_data_path, data_features)
        evaluate(model, test_data, test_labels)
        evaluateByProgram(model, test_split)

if __name__ == "__main__":
    main()