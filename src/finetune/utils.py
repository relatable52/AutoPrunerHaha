import torch
import os
from tqdm import tqdm
from src.finetune.dataset import CallGraphDataset
from torch.utils.data import DataLoader
from src.finetune.model import BERT
import numpy as np
from torch import nn
import argparse
from src.utils.utils import Logger, AverageMeter, evaluation_metrics, read_config_file
from src.finetune.main import get_model

PARAMS = {'batch_size': 10, 'shuffle': False, 'num_workers': 8} 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_finetune(config, mode, model_name):
    dataset= CallGraphDataset(config, mode)
    dataloader = DataLoader(dataset, **PARAMS)
    model_path = os.path.join(config["LEARNED_MODEL_DIR"], f"{model_name}/", "model.pth")
    save_dir = os.path.join(config["CACHE_DIR"], f"{model_name}/", "{}_finetuned".format(mode))
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model = get_model(model_name=model_name)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    loop=tqdm(enumerate(dataloader),leave=False,total=len(dataloader))
    for idx, batch in loop:
        ids=batch['ids'].to(device)
        mask= batch['mask'].to(device)
        _, emb =model(
                ids=ids,
                mask=mask)
        emb = emb.detach().cpu().numpy()
        save_path = os.path.join(save_dir, "{}.npy".format(idx))
        np.save(save_path, emb)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config/wala.config") 
    parser.add_argument("--model_name", type=str, default="codet5p-110m-embedding")
    parser.add_argument("--mode", type=str, default="test")
    
    return parser.parse_args()

def main():
    args = get_args()
    config = read_config_file(args.config_path)
    mode = args.mode
    model_name = args.model_name
    save_finetune(config, mode, model_name)
    


