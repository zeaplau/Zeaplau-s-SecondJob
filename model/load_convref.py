import os
import re
import sys
import json
import torch
import logging
import torch.nn as nn

from typing import List
from transformers import BertTokenizer
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

import pdb

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO)
logger = logging.getLogger(__name__)

ROOT_PATH = Path(os.getcwd())

def join_ans_triple(paths: List, ans_text: str) -> List:
    return ["".join(p + [ans_text] for p in paths)]

# Trick here
# process dataset
def gold_dataset(set_type: str):
    assert set_type in ['trainset', 'devset', 'testset', 'debug']
    if set_type == 'debug':
            filename = f"{ROOT_PATH}/ConvRef/debug/debug.json"
    else:
        filename = f"{ROOT_PATH}/ConvRef/data/ConvRef_processed_{set_type}.json"
    
    # raw conversations
    with open(filename, "r", encoding="utf-8") as f:
        conversations = json.load(f)
    
    # qa with gold path
    with open(f"{ROOT_PATH}/ConvRef/{set_type}_gold.json", "r", encoding="utf-8") as f:
        gold_p = json.load(f)
    
    gold_ans = []
    for qas in gold_p:
        gold_paths = list(map(lambda x: x["correct_paths"], qas))
        gold_ans.append(gold_paths)

    # remote qa without gold ans
    if set_type in ['trainset', 'devset', 'debug']:
        new_convs = []
        for c_idx, conv in enumerate(conversations):
            new_qas = []
            new_conv = conv
            for q_idx, qa in enumerate(conv["questions"]):
                new_ref = list(map(lambda x: x["reformulation"], qa["reformulations"]))
                new_qa = qa
                new_qa["gold_paths"] = gold_ans[c_idx][q_idx]
                new_qa["reformulations"] = new_ref
                new_qas.append(new_qa)
            new_conv["questions"] = new_qas
            new_conv["seed_entity"] = re.search("^Q\d*", conv["seed_entity"]).group()
            new_convs.append(new_conv)
        with open(f"{ROOT_PATH}/ConvRef/data/ConvRef_gold_{set_type}.json", "w", encoding="utf-8") as f:
            json.dump(new_convs, f)

# Dataset for the processed data
class ConvRefDataset(Dataset):
    def __init__(self, vocab_file: str, set_type: str="", set_path: str = "") -> None:
        if set_type is None or set_type not in ['trainset', 'devset', 'testset', "debug"]:
            raise Exception(f"Unknown set type {set_type}")
        else:
            self.set_type = set_type

        assert os.path.exists(set_path), f"{set_type}: {set_path} is not exists"

        logger.info(f"Loading dataset {set_type} from {set_path}")
        with open(set_path, "r", encoding="utf-8") as f:
            self.converstaions = json.load(f)

        # tokenizer
        # self.tokenizer = BasicTokenizer(vocab_file=vocab_file)

        # Encode the questions and answers
        self.unique_convs = []
        for conv in self.conversations:
            unique_conv = conv
            unique_conv["seed_entity"] = re.search("Q\d+", conv["seed_entity"])
            unique_qas = []
            for qa in conv["questions"]:
                unique_qa = qa
                qs_id = torch.nn.utils.rnn.pad_sequence(list(map(
                    lambda q: torch.tensor(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(q)), dtype=torch.LongTensor), 
                    qa["question"] + qa["reformulations"]
                )), batch_first=True)
                ans_path = [" ".join(p, qa["gold_answer_text"]) for p in qa["gold_paths"]]
                ans_id = torch.nn.utils.rnn.pad_sequence(list(map(
                    lambda a: torch.tensor(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(a)), dtype=torch.LongTensor), 
                    ans_path
                )), batch_first=True)
                unique_qa["qs_id"] = qs_id
                unique_qa["ans_id"] = ans_id
                unique_qa["gold_answer_entity"] = re.search("^Q\d+", qa["gold_answer_entity"]).group()
                unique_qas.append(unique_qa)
            unique_conv["questions"] = unique_qas
            self.unique_convs.append(unique_conv)

    def __len__(self):
        return len(self.unique_convs)

    def __getitem__(self, index):
        return self.unique_convs[index]