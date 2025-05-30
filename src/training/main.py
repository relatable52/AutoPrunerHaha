from tqdm import tqdm
import numpy as np
from src.training.dataset import FinetunedDataset
from torch.utils.data import DataLoader
from src.training.model import NNClassifier_Combine, NNClassifier_Structure, NNClassifier_Semantic
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
import torch
import argparse
from src.utils.utils import Logger, AverageMeter, evaluation_metrics, read_config_file
import os
import warnings
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_curve, auc, precision_recall_curve
import torch
import math
# Import statistics Library
import statistics
import pickle

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_PARAMS = {'batch_size': 100, 'shuffle': True, 'num_workers': 8}
TEST_PARAMS = {'batch_size': 100, 'shuffle': False, 'num_workers': 8}

logger = Logger()

def train(dataloader, model, mean_loss, loss_fn, optimizer, cfx_matrix):
    model.train()
    loop=tqdm(enumerate(dataloader), leave=False, total=len(dataloader))
    for idx, batch in loop:
        code=batch['code'].to(device)
        struct= batch['struct'].to(device)
        label = batch['label'].to(device)
        output = model(
                code=code,
                struct=struct)

        loss = loss_fn(output, label)
        # print(output)
        # print(label)
        
        num_samples = output.shape[0]
        mean_loss.update(loss.item(), n=num_samples)
        
        output = F.softmax(output)
        output = output.detach().cpu().numpy()[:, 1]
        pred = np.where(output >= 0.5, 1, 0)
        label = label.detach().cpu().numpy()
        
        cfx_matrix, precision, recall, f1 = evaluation_metrics(label, pred, cfx_matrix)

        # logger.log("Iter {}: Loss {}, Precision {}, Recall {}, F1 {}".format(idx, loss.item(), precision, recall, f1))
        loop.set_postfix(loss=mean_loss.item(), pre=precision, rec=recall, f1 = f1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return model, cfx_matrix

def do_test(dataloader, model, is_write=False, save_path="program_predictions.pkl"):
    model.eval()
    cfx_matrix = np.array([[0, 0], [0, 0]])
    result_per_programs = {}

    all_outputs = []
    all_labels = []

    # For saving predictions in pickle format
    saved_data = {}

    loop = tqdm(enumerate(dataloader), leave=False, total=len(dataloader))
    for idx, batch in loop:
        code = batch['code'].to(device)
        struct = batch['struct'].to(device)
        label = batch['label'].to(device)
        sanity_check = batch['static'].numpy()
        program_ids = batch['program_ids'].numpy()

        output = model(code=code, struct=struct)
        output = F.softmax(output, dim=1).detach().cpu().numpy()[:, 1]  # get positive class proba
        output = output * sanity_check
        pred = np.where(output >= 0.5, 1, 0)
        label = label.detach().cpu().numpy()

        for i in range(len(label)):
            prog_idx = program_ids[i]
            out = output[i]
            lb = label[i]
            mask = sanity_check[i]

            if prog_idx not in result_per_programs:
                result_per_programs[prog_idx] = {'lb': [], 'output': []}
                saved_data[prog_idx] = {'labels': [], 'output': [], 'static_mask': []}

            result_per_programs[prog_idx]['lb'].append(lb)
            result_per_programs[prog_idx]['output'].append(out)

            saved_data[prog_idx]['labels'].append(lb)
            saved_data[prog_idx]['output'].append(out)
            saved_data[prog_idx]['static_mask'].append(mask)

            all_outputs.append(out)
            all_labels.append(lb)

        cfx_matrix, precision, recall, f1 = evaluation_metrics(label, pred, cfx_matrix)
        loop.set_postfix(pre=precision, rec=recall, f1=f1)

    # Optionally save all predictions
    if is_write:
        np.save("prediction.npy", np.array(all_outputs))
        with open(save_path, "wb") as f:
            pickle.dump(saved_data, f)
        print(f"[SAVED] Predictions saved to {save_path}")

    (tn, fp), (fn, tp) = cfx_matrix
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    logger.log("[EVAL] Iter {}, Precision {}, Recall {}, F1 {}".format(idx, precision, recall, f1))

    # Average metrics by program
    precision_avg, recall_avg, f1_avg = [], [], []
    for i in result_per_programs:
        lb = np.array(result_per_programs[i]['lb'])
        output = np.array(result_per_programs[i]['output'])
        pred = np.where(output >= 0.5, 1, 0)
        p = precision_score(lb, pred, zero_division=0)
        r = recall_score(lb, pred, zero_division=0)
        precision_avg.append(p)
        recall_avg.append(r)
        f1_avg.append(2 * p * r / (p + r) if (p + r) > 0 else 0)

    logger.log("[EVAL-AVG] Iter {}, Precision {} ({}), Recall {} ({}), F1 {} ({})".format(
        idx,
        round(statistics.mean(precision_avg), 2), round(statistics.stdev(precision_avg), 2),
        round(statistics.mean(recall_avg), 2), round(statistics.stdev(recall_avg), 2),
        round(statistics.mean(f1_avg), 2), round(statistics.stdev(f1_avg), 2)
    ))


def do_train(epochs, train_loader, test_loader, model, loss_fn, optimizer, learned_model_dir):
    cfx_matrix = np.array([[0, 0],
                           [0, 0]])
    mean_loss = AverageMeter()
    for epoch in range(epochs):
        logger.log("Start training at epoch {} ...".format(epoch))
        model, cfx_matrix = train(train_loader, model, mean_loss, loss_fn, optimizer, cfx_matrix)
        
        logger.log("Evaluating ...")
        do_test(test_loader, model, False)
    logger.log("Saving model ...")
    if(not os.path.exists(learned_model_dir)):
        os.makedirs(learned_model_dir)
    torch.save(model.state_dict(), os.path.join(learned_model_dir, "model.pth"))
    logger.log("Done !!!")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config/wala.config") 
    parser.add_argument("--mode", type=str, default="test") 
    parser.add_argument("--model_path", type=str, default="../replication_package/model/rq1/autopruner/wala.pth", help="Path to checkpoint (for test only)") 
    parser.add_argument("--feature", type=int, default=2, help="0: structure, 1: semantic, 2:combine")     
    parser.add_argument("--model_name", type=str, default="codebert")
    parser.add_argument("--output_prefix", type=str, default="njr1")
    return parser.parse_args()


def main():
    args = get_args()
    config = read_config_file(args.config_path)
    print("Running on config {}".format(args.config_path))
    print("Mode: {}".format(args.mode))
    
    mode = args.mode
    model_name = args.model_name
    learned_model_dir = os.path.join(config["CLASSIFIER_MODEL_DIR"], f"{model_name}/")


    if mode == "train":
        train_dataset= FinetunedDataset(config, "train", model_name)
    test_dataset= FinetunedDataset(config, "test", model_name)

    # print("Dataset have {} train samples and {} test samples".format(len(train_dataset), len(test_dataset)))

    if mode == "train" and not train_dataset.has_cache():
        train_loader = DataLoader(train_dataset, **TRAIN_PARAMS)
    test_loader = DataLoader(test_dataset, **TEST_PARAMS)

    if args.feature == 2:
        model=NNClassifier_Combine(32)
    elif args.feature == 1:
        model=NNClassifier_Semantic(32)
    elif args.feature == 0:
        model=NNClassifier_Structure(32)
    else:
        raise NotImplemented

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
    
    model.apply(init_weights)
    

    loss_fn = nn.CrossEntropyLoss()

    optimizer= optim.Adam(model.parameters(),lr= 5e-6)

    if mode == "train":
        do_train(8, train_loader, test_loader, model, loss_fn, optimizer, learned_model_dir)
    elif mode == "test":
        model.load_state_dict(torch.load(args.model_path))
        do_test(test_loader, model, True, save_path=os.path.join(learned_model_dir, f"{args.output_prefix}_{args.feature}_program_predictions.pkl"))
    else:
        raise NotImplemented
    


if __name__ == '__main__':
    main()

    # a = [0, 1, 0, 0, 1, 0]
    # b = [0, 1, 0, 1, 0, 1]
    # fpr, tpr, thresholds = roc_curve(a, b)
    # print(roc_curve(a, b, pos_label=2))