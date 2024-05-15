from src.utils.utils import get_input_and_mask
import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
from src.utils.utils import read_config_file, load_code, get_input_and_mask
from src.utils.converter import convert
from torch.utils.data import DataLoader
import numpy as np
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PARAMS = {'batch_size': 10, 'shuffle': False, 'num_workers': 8} 
models_dict = {
    "codebert":"microsoft/codebert-base",
    "codet5-base":"Salesforce/codet5-base",
    "codet5p-770m":"Salesforce/codet5p-770m",
    "codet5p-110m-embedding":"Salesforce/codet5p-110m-embedding",
    "codesage":"codesage/codesage-small"
}

class KaggleCallGraphDataset(Dataset):
    def __init__(self, config, mode, model_name):
        self.mode = mode
        self.train_mode = mode
        self.config = config
        self.raw_data_path = self.config["BENCHMARK_CALLGRAPHS"]
        self.processed_path = self.config["PROCESSED_DATA"]
        # self.size_mode = size_mode
        self.model_name = model_name
        self.save_dir = os.path.join(self.config["CACHE_DIR"], f"{self.model_name}/")
        self.save_path = os.path.join(self.save_dir, f"{self.mode}.pkl")
        self.cg_file = self.config["FULL_FILE"]

        self.max_length = 512

        if (self.mode == "train"):
            self.program_lists = os.path.join(self.config["TRAINING_PROGRAMS_LIST"])
        elif self.mode == "test":
            self.program_lists = os.path.join(self.config["TEST_PROGRAMS_LIST"])
        else:
            return NotImplemented

        print(self.has_cache())
        if self.has_cache():
            self.load()
        else:
            return NotImplemented
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        ids = self.data[index]
        mask = self.mask[index]
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'label': torch.tensor(self.labels[index], dtype=torch.long),
            'static': torch.tensor(self.static_ids[index], dtype=torch.long),
            }
      
    def load(self):
        print("Loading data ...")
        info_dict = pd.read_pickle(self.save_path)
        self.labels = info_dict['label']
        self.data = info_dict['data']
        self.mask = info_dict['mask']
        self.static_ids = info_dict['static_ids']

    def has_cache(self):
        if os.path.exists(self.save_path):
            print("Data exists")
            return True
        return False

if __name__ == '__main__':
    config = read_config_file("config/kaggle_wala.config")
    data = KaggleCallGraphDataset(config, "test", "codet5p-110m-embedding")
    data = KaggleCallGraphDataset(config, "train", "codet5p-110m-embedding")