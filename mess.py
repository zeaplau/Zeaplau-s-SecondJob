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
    assert os.path.exists(args.path)

    convert_model = torch.load(args.path, map_location="cpu")
    sim_keys = filter(lambda x: "sim_model" in x, convert_model.keys())
    convert_layer = {}
    for k in sim_keys:
        new_k = k
        if "g_encoder" in k:
            new_k = new_k.replace("g_encoder", "cp_encoder")
        convert_layer[".".join(new_k.split(".")[1:])] = convert_model[k]

    refmodel = RefModel(config=config, tokenizer=tokenizer, device="cpu")
    refmodel_dict = refmodel.state_dict()

    for k in convert_layer.keys():
        refmodel_dict[k] = convert_layer[k]
    refmodel.load_state_dict(refmodel_dict)
    torch.save(refmodel.state_dict(), "./ckpt/base.pth")

    print("--- Check reload ---")
    check_model = RefModel(config=config, tokenizer=tokenizer, device="cpu")
    if os.path.exists("./ckpt/base.pth"):
        check_model_dict = torch.load("./ckpt/base.pth")

    check_model.load_state_dict(check_model_dict)