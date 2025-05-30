from src.utils.utils import get_input_and_mask
import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import pickle as pkl
import argparse
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from src.utils.utils import read_config_file, load_code, get_input_and_mask
from src.utils.converter import convert
from src.finetune.model import BERT
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

header_names = [
"direct#depth_from_main",
"direct#src_node_in_deg",
"direct#dest_node_out_deg",
"direct#dest_node_in_deg",
"direct#src_node_out_deg",
"direct#repeated_edges",
"direct#fanout",
"direct#graph_node_count", 
"direct#graph_edge_count",
"direct#graph_avg_deg",
"direct#graph_avg_edge_fanout",
"trans#depth_from_main",
"trans#src_node_in_deg",
"trans#dest_node_out_deg",
"trans#dest_node_in_deg",
"trans#src_node_out_deg",
"trans#repeated_edges",
"trans#fanout",
"trans#graph_node_count",
"trans#graph_edge_count",
"trans#graph_avg_deg",
"trans#graph_avg_edge_fanout"
]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config/wala.config") 
    parser.add_argument("--model_name", type=str, default="codebert", choices=list(models_dict.keys()), help="Name of the model to use")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--skip_embedding", type=bool, default=False, help="Whether to skip embedding extraction")
    return parser.parse_args()

class CallGraphDataset(Dataset):
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

        if self.mode == "train":
            self.program_lists = os.path.join(self.config["TRAINING_PROGRAMS_LIST"])
        elif self.mode == "test":
            self.program_lists = os.path.join(self.config["TEST_PROGRAMS_LIST"])
        else:
            raise NotImplementedError(f"Mode {self.mode} is not implemented")

        print(self.has_cache())
        if self.has_cache():
            self.load()
        elif self.model_name in models_dict:
            self.tokenizer = AutoTokenizer.from_pretrained(models_dict[self.model_name])
            self.process()
            self.save()
        else:
            raise NotImplementedError(f"Model {self.model_name} is not implemented")
    
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
        self.data = []
        self.mask = []
        self.static_ids = []
        self.labels = []
        cnt = 0
        with open(self.program_lists, "r") as f:
            for line in f:
                print(cnt)
                cnt += 1
                filename = line.strip()
                file_path = os.path.join(self.raw_data_path, filename, self.cg_file)
                df = pd.read_csv(file_path)
                descriptor2code = load_code(os.path.join(self.processed_path, filename, 'code.csv'))

                for i in tqdm(range(len(df['dynamic']))):
                    src, dst, lb, sanity_check = df['method'][i], df['target'][i], df['dynamic'][i], df[self.config["SA_LABEL"]][i]
                    if self.mode != "train" or sanity_check == 1:                        
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

        with open(self.save_path, 'wb') as f:
            pkl.dump({'label': self.labels,
                      'data': self.data,
                      'mask': self.mask,
                      'static_ids': self.static_ids}, f)

    def load(self):
        print("Loading data ...")
        with open(self.save_path, 'rb') as f:
            print("Loading from cache")
            # Load the data from the pickle file   
            info_dict = pkl.load(f)
        self.labels = info_dict['label']
        self.data = info_dict['data']
        self.mask = info_dict['mask']
        self.static_ids = info_dict['static_ids']

    def has_cache(self):
        if os.path.exists(self.save_path):
            print("Data exists")
            return True
        return False


class ClassifierDataset(Dataset):
    def __init__(self, config, mode, model_name, skip_embedding=False):
        self.skip_embedding = skip_embedding
        self.mode = mode
        self.config = config
        self.raw_data_path = self.config["BENCHMARK_CALLGRAPHS"]
        self.processed_path = self.config["PROCESSED_DATA"]
        self.model_name = model_name
        self.save_dir = os.path.join(self.config["CACHE_DIR"], f"{self.model_name}/")
        self.save_path = os.path.join(self.save_dir, f"ft_{self.mode}.pkl")
        self.cg_file = self.config["FULL_FILE"]
        self.emb_file = os.path.join(self.save_dir, f"{self.mode}_embeddings.npy")

        if self.mode == "train":
            self.program_lists = os.path.join(self.config["TRAINING_PROGRAMS_LIST"])
        elif self.mode == "test":
            self.program_lists = os.path.join(self.config["TEST_PROGRAMS_LIST"])
        else:
            raise NotImplementedError(f"Mode {self.mode} is not implemented")

        print(self.has_cache())
        print(self.skip_embedding)
        if self.has_cache():
            self.load()
        elif self.model_name in models_dict:
            self.header_names = header_names
            self.process()
            self.save()
        else:
            raise NotImplementedError(f"Model {self.model_name} is not implemented")
    
    def __len__(self):
        return len(self.code_feats)
    
    def __getitem__(self, index):
        struct_feats = np.where(self.struct_feats[index] == 1000000000, 100000, self.struct_feats[index])
        return {
            'code': torch.tensor(self.code_feats[index], dtype=torch.float),
            'struct': torch.tensor(struct_feats, dtype=torch.float),
            'label': torch.tensor(self.labels[index], dtype=torch.long),
            'static': torch.tensor(self.static_ids[index], dtype=torch.float),
            'program_ids': self.program_ids[index]
        }
    
    def process(self):
        self.code_feats = []
        self.struct_feats = []
        self.labels = []
        self.static_ids = []
        self.program_ids = []
        program_idx = 0
        with open(self.program_lists, "r") as f:
            for line in f:
                filename = line.strip()
                file_path = os.path.join(self.raw_data_path, filename, self.cg_file)
                df = pd.read_csv(file_path)
                features = df[self.header_names].to_numpy()
                if not self.skip_embedding:
                    emb = np.load(self.emb_file)
                for i in tqdm(range(len(df['dynamic']))):
                    lb, sanity_check = df['dynamic'][i], df[self.config["SA_LABEL"]][i]
                    if self.mode != "train" or sanity_check == 1:
                        if not self.skip_embedding:
                            self.code_feats.append(emb[i])
                        self.struct_feats.append(features[i])
                        self.labels.append(lb)
                        self.static_ids.append(sanity_check)
                        self.program_ids.append(program_idx)
                program_idx += 1
    
    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)

        with open(self.save_path, 'wb') as f:
            pkl.dump({
                'code': self.code_feats,
                'struct': self.struct_feats,
                'target': self.labels,
                'static_ids': self.static_ids,
                'program_ids': self.program_ids
            }, f)

    def load(self):
        print("Loading data ...")
        with open(self.save_path, 'rb') as f:
            print("Loading from cache")
            # Load the data from the pickle file   
            info_dict = pkl.load(f)
        self.code_feats = info_dict['code_feats']
        self.struct_feats = info_dict['struct_feats']
        self.labels = info_dict['labels']
        self.static_ids = info_dict['static_ids']
        self.program_ids = info_dict['program_ids']

    def has_cache(self):
        if os.path.exists(self.save_path):
            print("Data exists")
            return True
        return False
                        
                        
if __name__ == '__main__':
    args = get_args()
    config = read_config_file(args.config_path)

    skip_embedding = args.skip_embedding
    model_name = args.model_name
    if not skip_embedding:
        print("Loading dataset...")
        dataset = CallGraphDataset(config, "test", model_name)
        print(f"Dataset size: {len(dataset)}")

        save_dir = os.path.join(config["CACHE_DIR"], f"{model_name}/")
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, "test_embeddings.npy")

        if not os.path.exists(save_path):
            dataloader = DataLoader(dataset, **PARAMS)

            print("Loading model...")
            model = BERT()
            checkpoint = torch.load(args.model_path, map_location=device)

            # Robust handling of possible DataParallel or plain checkpoints
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Remove 'module.' prefix if present
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

            # Optionally filter out keys not in model
            model_keys = set(model.state_dict().keys())
            filtered_state_dict = {k: v for k, v in new_state_dict.items() if k in model_keys}

            # Load with strict=False to handle minor differences like position_ids
            model.load_state_dict(filtered_state_dict, strict=False)

            # Wrap with DataParallel if needed
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)

            model.to(device)
            model.eval()

            all_embeddings = []

            print("Extracting embeddings...")
            with torch.no_grad():
                for batch in tqdm(dataloader):
                    ids = batch['ids'].to(device)
                    mask = batch['mask'].to(device)

                    _, emb = model(ids=ids, mask=mask)
                    all_embeddings.append(emb.detach().cpu().numpy())

            print("Concatenating embeddings...")
            all_embeddings = np.concatenate(all_embeddings, axis=0)

            print(f"Saving all embeddings to {save_path}")
            np.save(save_path, all_embeddings)
        else:
            print(f"Embeddings already exist at {save_path}, skipping extraction.")

        print("All embeddings saved successfully.")
    
    classifier_dataset = ClassifierDataset(config, "test", model_name, skip_embedding=skip_embedding)
    print(f"Classifier dataset size: {len(classifier_dataset)}")

    print("Done!")
    