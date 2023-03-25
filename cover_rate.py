import pdb
import numpy as np
import json

if __name__ == "__main__":
    with open("./ConvRef/data/trainset_gold.json", mode="r", encoding="utf-8") as f:
        instances = json.load(f)
    
    print(f"---- instances amount {len(instances)}, questions amount {len(instances) * 5} ----")

    yn = ['yes', 'no']
    covered = 0
    multi_ans = 0
    for ins in instances:
        for qa in ins:
            if len(qa['correct_answer']) != 0 or len(set([qa['answer_text'].lower()]) & set(yn)) != 0:
                covered += 1
                if (np.array(qa['correct_answer']) < 1.0).any():
                    multi_ans += 1
    
    print(f"---- cover rate {covered / (len(instances) * 5)} ----")
    print(f"---- multi answer rate {multi_ans / (len(instances) * 5)} ----")