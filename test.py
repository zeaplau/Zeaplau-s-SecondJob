import torch

from model.load_convref import ConvRefDataset, ConvRefInstance
from model.lstm_ref import RefModel
from tools.tokenization import BasicTokenizer
from tools.create_config import ModelConfig

import random
import numpy as np

import pdb

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

config = ModelConfig("./config/config_SimpleRanker.json")
tokenizer = BasicTokenizer("./config/vocab.txt")

device = "cpu"

model = RefModel(config=config, tokenizer=tokenizer, device=device)

train_set = ConvRefDataset("./config/vocab.txt", "trainset", "./ConvRef/data/ConvRef_gold_trainset.json").unique_convs
train_set = [ConvRefInstance(conv) for conv in train_set]

questions = ["When did Karen die?", "Karen's day of death?", "Karen's year of death?", "What day did Karen die?", "What month did Karen die?"]

questions_ids = list(
    map(lambda x: torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x)), dtype=torch.long), questions)
)

questions_ids = torch.nn.utils.rnn.pad_sequence(questions_ids)
questionts_pad_idx = torch.sum(1 - torch.eq(questions_ids, 0).type(torch.LongTensor), 1).squeeze(0) - 1

questions_embs = model.embedder(questions_ids)
questions_encs = model.q_encoder(questions_embs)
questions_vecs = model.pooler(questions_encs, questionts_pad_idx)

# 计算他们之间的距离