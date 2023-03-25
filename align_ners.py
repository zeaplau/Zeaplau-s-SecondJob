import json
import os
import argparse
import re

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

            if isinstance(qa['gold_answer'], list):
                new_qa['gold_answer'] = qa['gold_answer'][0]
            else:
                try:
                    new_qa['gold_answer'] = re.search("Q\d+", qa['gold_answer']).group() if re.search("Q\d+", qa['gold_answer']) else qa['gold_answer']
                except:
                    pdb.set_trace()
            new_qas.append(new_qa)
        new_conv['questions'] = new_qas
        new_convs.append(new_conv)
    
    with open(f"{ROOT_PATH}/ConvRef/data/ConvRef_processed_{args.set_type}.json", "w", encoding="utf-8") as f:
        json.dump(new_convs, f)