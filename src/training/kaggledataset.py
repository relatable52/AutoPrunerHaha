import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
from src.utils.utils import read_config_file
from src.utils.converter import convert
import numpy as np

header_names = [
    "-direct#depth_from_main",
    "-direct#src_node_in_deg",
    "-direct#dest_node_out_deg",
    "-direct#dest_node_in_deg",
    "-direct#src_node_out_deg",
    "-direct#repeated_edges",
    "-direct#fanout",
    "-direct#graph_node_count",
    "-direct#graph_edge_count",
    "-direct#graph_avg_deg",
    "-direct#graph_avg_edge_fanout",
    "-trans#depth_from_main",
    "-trans#src_node_in_deg",
    "-trans#dest_node_out_deg",
    "-trans#dest_node_in_deg",
    "-trans#src_node_out_deg",
    "-trans#repeated_edges",
    "-trans#fanout",
    "-trans#graph_node_count",
    "-trans#graph_edge_count",
    "-trans#graph_avg_deg",
    "-trans#graph_avg_edge_fanout",
]


def compute_header(header_names, header):
    return [header + header_names[i] for i in range(len(header_names))]


class FinetunedDataset(Dataset):
    def __init__(self, config, mode, model, loss_fn, logger):
        self.mode = mode
        self.config = config
        self.logger = logger
        self.raw_data_path = self.config["BENCHMARK_CALLGRAPHS"]
        self.processed_path = self.config["PROCESSED_DATA"]
        self.save_dir = os.path.join(
            self.config["CACHE_DIR"], model, loss_fn
        )
        self.save_path = os.path.join(self.save_dir, f"ft_{self.mode}.pkl")
        self.cg_file = self.config["FULL_FILE"]

        if self.mode == "train":
            self.program_lists = os.path.join(self.config["TRAINING_PROGRAMS_LIST"])
        elif self.mode == "test":
            self.program_lists = os.path.join(self.config["TEST_PROGRAMS_LIST"])
        else:
            return NotImplemented

        self.header_names = compute_header(header_names, self.config["HEADERS"])

        if self.has_cache():
            self.load()
        else:
            self.process()
            self.save()

    def __len__(self):
        return len(self.code_feats)

    def __getitem__(self, index):
        struct_feats = np.where(
            self.struct_feats[index] == 1000000000, 100000, self.struct_feats[index]
        )
        code = torch.tensor(self.code_feats[index], dtype=torch.float)
        struct = torch.tensor(struct_feats, dtype=torch.float)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        data = torch.cat([code, struct], axis=1)
        return data, label

    def process(self):
        pass
    
    def save(self):
        pass

    def load(self):
        self.logger.info(f"Loading data from {self.save_path} ...")
        info_dict = pd.read_pickle(self.save_path)
        self.code_feats = info_dict["code"]
        self.struct_feats = info_dict["struct"]
        self.labels = info_dict["target"]
        self.static_ids = info_dict["static_ids"]
        self.program_ids = info_dict["program_ids"]

    def has_cache(self):
        if os.path.exists(self.save_path):
            return True
        return False


if __name__ == "__main__":
    config = read_config_file("config/wala.config")
    data = FinetunedDataset(config, "train")
    data = FinetunedDataset(config, "test")
