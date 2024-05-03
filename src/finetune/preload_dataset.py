from src.finetune.dataset import CallGraphDataset
from src.utils.utils import read_config_file
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config/wala.config") 
    parser.add_argument("--model_name", type=str, default="codet5p-110m-embedding")
    
    return parser.parse_args()

def main():
    args = get_args()
    config = read_config_file(args.config_path)
    model_name = args.model_name
    data = CallGraphDataset(config, "test", model_name)
    data = CallGraphDataset(config, "train", model_name)



