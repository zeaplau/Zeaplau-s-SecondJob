import torch
from model.lstm_ref import LSTMRef
from tools.tokenization import BasicTokenizer
from tools.create_config import ModelConfig

import random
import numpy as np

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

config = ModelConfig("./config/config_SimpleRanker.json")
tokenizer = BasicTokenizer("./config/vocab.txt")

device = "cpu"

model = LSTMRef(config=config, tokenizer=tokenizer, device=device)

questions = [
    "When did The Carpenters sign with A&M Records?",
    "At which date did The Carpenters sign with A&M Records?",
    "Date when A&M Records sign The Carpenters?",
    "When did A&M Records sign with The Carpenters?",
    "Day when The Carpenters sign with the A&M Records?"
]

cps = [
    ["The Carpenters", "work period (start)"], ["The Carpenters", "inception"], ['facet of', 'The Simpsons'], ['followed by', 'The Simpsons'], 
    ['performer', 'The Simpsons'], ['series spin-off', 'The Simpsons']
]

ga_cps = [["The Carpenters", "work period (start)"], ["The Carpenters", "inception"]]

cps = [" ".join(path) for path in cps]
ga_cps = [" ".join(path) for path in ga_cps]

qs_ids = model.tokenize_sentence(questions)
cps_ids = model.tokenize_sentence(cps)
ga_ids = model.tokenize_sentence(ga_cps)

model.debug(qs_ids=qs_ids, cps=cps_ids, ga_ids=ga_ids)