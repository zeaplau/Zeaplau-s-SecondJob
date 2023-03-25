import os
import torch
import argparse
import torch.nn as nn

from model.lstm_ref import RefModel
from tools.create_config import ModelConfig
from tools.tokenization import BasicTokenizer

import pdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="", help="Model to convert.")
    parser.add_argument("--config", type=str, default="", help="Config of model.")
    parser.add_argument("--vocab", type=str, default="", help="The path of vocab.")
    args = parser.parse_args()

    config = ModelConfig(args.config)
    tokenizer = BasicTokenizer(args.vocab)
    assert os.paths.exists(args.path)

    convert_model = torch.load(args.path, map_location="cpu")

    pdb.set_trace()

    refmodel = RefModel(config=args.config, tokenizer=tokenizer, device="cpu")
    
    ...