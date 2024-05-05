from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from src.finetune.model import BERT, CodeT5Enc, CodeT5pEmb, CodeSageBase, CodeT5pEnc
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
import torch
import argparse
from src.utils.utils import Logger, AverageMeter, evaluation_metrics, read_config_file
import os
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_PARAMS = {'batch_size': 10, 'shuffle': True, 'num_workers': 8}
TEST_PARAMS = {'batch_size': 10, 'shuffle': False, 'num_workers': 8}

logger = Logger()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config/wala.config") 
    parser.add_argument("--model_name", type=str, default="codet5p-110m-embedding")
    parser.add_argument("--model_path", type=str, default="../replication_package/model/finetuned_model/model.pth", help="Path to checkpoint (for test only)") 
    parser.add_argument("--mode", type=str, default="train") 
    
    return parser.parse_args()


def get_model(model_name):
    models_dict = {
        "codebert": BERT,
        "codet5-base": CodeT5Enc,
        "codet5p-770m": CodeT5pEnc,
        "codet5p-110m-embedding": CodeT5pEmb,
        "codesage": CodeSageBase
    }
    if model_name in models_dict:
        model = models_dict[model_name]()
    else:
        return NotImplemented
    return model


def main():
    args = get_args()
    config = read_config_file(args.config_path)
    print("Running on config {}".format(args.config_path))
    print("CLM: {}".format(args.model_name))
    print("Mode: {}".format(args.mode))
    
    mode = args.mode
    model_name = args.model_name
    learned_model_dir = config["LEARNED_MODEL_DIR"]

    if args.config_path == "config/kaggle_finetune_wala.config":
        from src.finetune.kaggle_dataset import KaggleCallGraphDataset
        train_dataset= KaggleCallGraphDataset(config, "train", model_name)
        test_dataset= KaggleCallGraphDataset(config, "test", model_name)
    else:
        from src.finetune.dataset import CallGraphDataset
        train_dataset= CallGraphDataset(config, "train", model_name)
        test_dataset= CallGraphDataset(config, "test", model_name)

    print("Dataset have {} train samples and {} test samples".format(len(train_dataset), len(test_dataset)))

    train_loader = DataLoader(train_dataset, **TRAIN_PARAMS)
    test_loader = DataLoader(test_dataset, **TEST_PARAMS)

    model = get_model(model_name)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer= optim.Adam(model.parameters(),lr= 0.00001)

    batch = next(iter(train_loader))
    ids = batch['ids']
    mask = batch['mask']
    label = batch['label']
    print(batch)
    print(ids)
    print(mask)
    print(label)
    output = model(ids=ids, mask=mask)
    print(output)

if __name__ == "__main__":
    main()
