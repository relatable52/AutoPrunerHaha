import os
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch import nn
from src.finetune.dataset import CallGraphDataset
from src.utils.utils import read_config_file
from src.finetune.model import get_model

PARAMS = {"batch_size": 10, "shuffle": False, "num_workers": 8}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_finetune(config, mode, model_name, loss_fn):
    dataset = CallGraphDataset(config, mode, model_name)
    dataloader = DataLoader(dataset, **PARAMS)
    model_path = os.path.join(
        config["LEARNED_MODEL_DIR"], model_name, loss_fn, "model.pth"
    )
    save_dir = os.path.join(
        config["CACHE_DIR"], model_name, loss_fn, "{}_finetuned".format(mode)
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if len(os.listdir(save_dir)) > 0:
        print("Directory {} already exists".format(save_dir))
        return

    model = get_model(model_name=model_name)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["state_dict"])
    loop = tqdm(enumerate(dataloader), leave=False, total=len(dataloader))
    for idx, batch in loop:
        ids = batch["ids"].to(device)
        mask = batch["mask"].to(device)
        _, emb = model(ids=ids, mask=mask)
        emb = emb.detach().cpu().numpy()
        save_path = os.path.join(save_dir, "{}.npy".format(idx))
        np.save(save_path, emb)

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config/wala.config", required=True)
    parser.add_argument("--model_name", type=str, default="codebert", required=True)
    parser.add_argument("--loss_fn", type=str, default="cross_entropy", required=True)
    return parser.parse_args()

def main():
    args = get_args()
    config = read_config_file(args.config_path)
    save_finetune(config, "train", args.model_name, args.loss_fn)
    save_finetune(config, "test", args.model_name, args.loss_fn)
    
if __name__ == "__main__":
    main()