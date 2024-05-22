from src.utils.utils import get_input_and_mask, load_code
from src.utils.converter import convert
from src.finetune.model import models
import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from dgl.data.utils import save_info, load_info
from tqdm import tqdm

class CallGraphDataset(Dataset):
    def __init__(self, config, mode, model, logger):
        self.mode = mode
        model_name, model_size = model.split("-")
        self.model = model
        self.config = config
        self.logger = logger
        self.raw_data_path = self.config["BENCHMARK_CALLGRAPHS"]
        self.processed_path = self.config["PROCESSED_DATA"]
        self.save_dir = os.path.join(self.config["CACHE_DIR"], model)
        self.save_path = os.path.join(self.save_dir, f"{self.mode}.pkl")
        self.cg_file = self.config["FULL_FILE"]

        self.max_length = models[model_name]["max_length"]

        if self.mode == "train":
            self.program_lists = os.path.join(self.config["TRAINING_PROGRAMS_LIST"])
        elif self.mode == "test":
            self.program_lists = os.path.join(self.config["TEST_PROGRAMS_LIST"])
        else:
            return NotImplemented

        if self.has_cache():
            self.load()
        elif model_name in models:
            self.tokenizer = AutoTokenizer.from_pretrained(models[model_name]["pretrained_name"][model_size])
            self.process()
            self.save()
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

    def process(self):
        self.logger.info(f"Processing data using {self.model} model ...")
        self.data = []
        self.mask = []
        self.static_ids = []
        self.labels = []
        cnt = 0
        with open(self.program_lists, "r") as f:
            for line in f:
                self.logger.info(cnt)
                cnt += 1
                filename = line.strip()
                self.logger.info(f"[{cnt}/{len(f)}] Processing {filename} ...")
                file_path = os.path.join(self.raw_data_path, filename, self.cg_file)
                df = pd.read_csv(file_path)
                for i in tqdm(range(len(df['wiretap']))):
                    src, dst, lb, sanity_check = df['method'][i], df['target'][i], df['wiretap'][i], df[self.config["SA_LABEL"]][i]
                    if self.mode != "train" or sanity_check == 1:

                        descriptor2code = load_code(os.path.join(self.processed_path, filename, 'code.csv'))

                        if src != '<boot>':
                            if src in descriptor2code:
                                src = descriptor2code[src]
                            else:
                                src = convert(src).__tocode__()

                        dst_descriptor = convert(dst)

                        if dst in descriptor2code:
                            dst = descriptor2code[dst]
                        else:
                            dst = dst_descriptor.__tocode__()

                        token_ids, mask = get_input_and_mask(src, dst, self.max_length, self.tokenizer)
                        self.data.append(token_ids)
                        self.mask.append(mask)
                        self.labels.append(lb)
                        self.static_ids.append(sanity_check)

    def save(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        save_info(self.save_path, {'label': self.labels,
                                    'data': self.data,
                                    'mask': self.mask,
                                    'static_ids': self.static_ids,
                                   }
                  )

    def load(self):
        self.logger.info(f"Loading data from {self.save_path} ...")
        info_dict = load_info(self.save_path)
        self.labels = info_dict['label']
        self.data = info_dict['data']
        self.mask = info_dict['mask']
        self.static_ids = info_dict['static_ids']

    def has_cache(self):
        if os.path.exists(self.save_path):
            return True
        return False

# if __name__ == '__main__':
#     config = read_config_file("config/wala.config")
#     data = CallGraphDataset(config, "test")
#     data = CallGraphDataset(config, "train")
