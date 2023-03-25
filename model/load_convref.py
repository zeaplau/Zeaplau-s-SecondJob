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

from tools.tokenization import BasicTokenizer

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
    with open(f"{ROOT_PATH}/ConvRef/data/{set_type}_gold.json", "r", encoding="utf-8") as f:
        gold_p = json.load(f)
    
    gold_ans = []
    for qas in gold_p:
        gold_paths = list(map(lambda x: x["correct_paths"], qas))
        gold_ans.append(gold_paths)

    # remote qa without gold ans
    if set_type in ['trainset', 'devset', 'testset', 'debug']:
        new_convs = []
        for c_idx, conv in enumerate(conversations):
            new_qas = []
            new_conv = conv
            for q_idx, qa in enumerate(conv["questions"]):
                new_ref = list(map(lambda x: x["reformulation"], qa["reformulations"]))
                new_qa = qa
                new_qa["gold_paths"] = gold_ans[c_idx][q_idx]
                new_qa["reformulations"] = new_ref
                try:
                    new_qa["gold_answer"] = re.search("Q\d+", qa["gold_answer"]).group() if re.search("Q\d+", qa["gold_answer"]) else qa["gold_answer"]
                except:
                    pdb.set_trace()
                new_qas.append(new_qa)
            new_conv["questions"] = new_qas
            new_conv["seed_entity"] = re.search("Q\d*", conv["seed_entity"]).group()
            new_convs.append(new_conv)
        with open(f"{ROOT_PATH}/ConvRef/data/ConvRef_gold_{set_type}.json", "w", encoding="utf-8") as f:
            json.dump(new_convs, f)

class ConvRefInstance:
    def __init__(self, conv) -> None:
        self.seed_entity = conv["seed_entity"]
        self.seed_entity_text = conv["seed_entity_text"]
        self.conv_id = conv["conv_id"]
        self.domain = conv["domain"]
        self.questions = conv["questions"]
        self.current_paths = []
        self.orig_candidate_paths = []
        self.candidate_paths = []
        self.current_F1s = []
        self.orig_F1s = []
        self.F1s = []
        self.historical_frontier = []
        self.current_frontier = []
        self.path2ans = {}
        self.statements = []
    
    def __str__(self):
        # question, gold_answer_text, gold_answer, reformulation number
        instance_format = "q:{} a: {} a_e: {} r_num: {}\n"
        s = ""
        for qa in self.questions:
            s += instance_format.format(qa["question"], qa["gold_answer_text"], qa["gold_answer"], len(qa["reformulations"]))
        return s

    def __repr__(self) -> str:
        return self.__str__()
    
    def reset(self):
        self.historical_frontier = []
        self.historical_frontier_text = []
        self.current_frontier = []
        self.path2ans = {}

# Dataset for the processed data
class ConvRefDataset:
    def __init__(self, vocab_file: str, set_type: str="", set_path: str = "") -> None:
        if set_type is None or set_type not in ['trainset', 'devset', 'testset', "debug"]:
            raise Exception(f"Unknown set type {set_type}")
        else:
            self.set_type = set_type

        assert os.path.exists(set_path), f"{set_type}: {set_path} is not exists"

        logger.info(f"Loading dataset {set_type} from {set_path}")
        with open(set_path, "r", encoding="utf-8") as f:
            self.conversations = json.load(f)

        self.tokenizer = BasicTokenizer(vocab_file=vocab_file)

        # Reformulate the conversation and encode the questions and answers
        self.unique_convs = []
        for conv in self.conversations:
            unique_conv = conv
            unique_conv["seed_entity"] = conv["seed_entity"]
            unique_qas = []
            for qa in conv["questions"]:
                unique_qa = qa

                qs_ = [qa["question"]] + qa["reformulations"]
                qs_token = map(lambda q: self.tokenizer.tokenize(q), qs_)
                qs_id = list(map(lambda q: torch.tensor(self.tokenizer.convert_tokens_to_ids(q), dtype=torch.long), qs_token))
                qs_tensor = torch.nn.utils.rnn.pad_sequence(qs_id, batch_first=True)

                if len(qa["gold_paths"]) != 0:
                    ans_path = [" ".join(p + [qa["gold_answer_text"]]) for p in qa["gold_paths"]]
                else:
                    ans_path = [qa["gold_answer_text"]]
                ans_token = map(lambda a: self.tokenizer.tokenize(a), ans_path)
                ans_id = list(map(lambda a: torch.tensor(self.tokenizer.convert_tokens_to_ids(a), dtype=torch.long), ans_token))
                ans_tensor = []
                ans_tensor = torch.nn.utils.rnn.pad_sequence(ans_id, batch_first=True)

                unique_qa["qs_id"] = qs_tensor
                unique_qa["ans_id"] = ans_tensor
                unique_qa["gold_answer"] = qa["gold_answer"]
                unique_qa["relation"] = ""
                unique_qas.append(unique_qa)
            unique_conv["questions"] = unique_qas
            self.unique_convs.append(unique_conv)

    def __len__(self):
        return len(self.unique_convs)

    def __getitem__(self, index):
        return self.unique_convs[index]
