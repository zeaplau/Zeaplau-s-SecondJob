import json
import os
import argparse

from pathlib import Path

import pdb

ROOT_PATH = Path(os.getcwd())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_type", type=str, help="The type of dataset.")
    args = parser.parse_args()
    
    with open(f"{ROOT_PATH}/ConvRef/data/ConvRef_{args.set_type}.json", "r", encoding="utf-8") as f:
        convs = json.load(f)
    
    with open(f"{ROOT_PATH}/ConvRef/data/ner_{args.set_type}.json", "r", encoding="utf-8") as f:
        ners = json.load(f)
    
    new_convs = []
    for c_idx, conv in enumerate(convs):
        new_qas = []
        new_conv = conv
        for q_idx, qa in enumerate(conv['questions']):
            new_qa = qa
            new_qa['NER'] = ners[c_idx][q_idx]
            new_qas.append(new_qa)
        new_conv['questions'] = new_qas
        new_convs.append(new_conv)
    
    with open(f"{ROOT_PATH}/ConvRef/data/ConvRef_processed_{args.set_type}.json", "w", encoding="utf-8") as f:
        json.dump(new_convs, f)