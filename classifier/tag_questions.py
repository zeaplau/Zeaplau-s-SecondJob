import json
import re

from typing import Tuple, List

import pdb

# interrogative sentence prefix
const_person_dic = '(who|which person|whom)'
const_time_dic = '(when|what time|which year|which day|what timepoint)'
const_location_dic = '(where|which country|which city)'
const_amount_dic = '(how many|how much)'
const_verification_dic = '(do|is|does|are|did|was|were)'
const_uncertain_dic = '(how do|how did|how does|how long|what)'

question_types = {
    const_person_dic: "person", 
    const_location_dic: "location", 
    const_time_dic: "time", 
    const_amount_dic: "amount",
    const_uncertain_dic: "uncertain",
    const_verification_dic: "verfication",
}

def tag_questions(ins_idx, convs) -> List[dict]:
    tags = []
    questions = map(
        lambda x: x['question'], convs['questions']
    )
    turns = []
    for q_idx, q in enumerate(questions):
        q_tag = "uncertain"
        for t in question_types:
            if re.search("%s" % t, q.lower()):
                q_tag = question_types[t]
                break
        turn = {'question': q, 'tag': q_tag, 'time': q_idx}
        turns.append(turn)
    return turns

if __name__ == "__main__":
    dataset = ['debug', 'trainset', 'devset', 'testset']
    for t in dataset:
        with open(f"../ConvRef/data/ConvRef_gold_{t}.json", "r", encoding="utf-8") as f:
            instances = json.load(f)

        questions_with_tag = {}
        for ins_idx, ins in enumerate(instances):
            res = tag_questions(ins_idx, ins)
            questions_with_tag[ins['conv_id']] = res

        with open(f"../ConvRef/tags/ConvRef_{t}_question_tags.json", "w", encoding="utf-8") as f:
            json.dump(questions_with_tag, f)