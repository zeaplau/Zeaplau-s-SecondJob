import os
import torch
import torch.nn as nn
import pytorch_lightning as L
import torch.nn.functional as F
import numpy as np

from transformers import BartTokenizer, BertForMultipleChoice, BartForConditionalGeneration, BertTokenizer
from typing import List, Tuple
from torch.nn import KLDivLoss, MSELoss
from model.advanced_layer import SimpleEmbLayer, SimplePooler
from tools.create_config import ModelConfig
from tools.tokenization import BasicTokenizer

import pdb

class LSTMRef(nn.Module):
    def __init__(self, config, tokenizer: BasicTokenizer, device=None):
        super(LSTMRef, self).__init__()
        self.device = device
        self.tokenizer = tokenizer
        self.hidden_size = config.hidden_size
        self.embedder = SimpleEmbLayer(config, device=self.device)
        self.pooler = SimplePooler(config)
        self.heads = 5
        self.emb = self.hidden_size // self.heads

        # Conversation progress, consider to use this ?
        self.initial_hidden = (torch.randn(1, self.hidden_size).to(self.device), torch.randn(1, self.hidden_size).to(self.device))

        # Encode question and candidate answer paths
        self.q_encoder = nn.LSTM(input_size=self.hidden_size, hidden_size=int(self.hidden_size / 2), bidirectional=True, batch_first=True)
        self.cp_encoder = nn.LSTM(input_size=self.hidden_size, hidden_size=int(self.hidden_size / 2), bidirectional=True, batch_first=True)

        # Regard the selecting topic entity problem as multiple choice question
        self.choice_template = "Which answer is the subject of the question '{}'"
        self.choice_tokenizer: BertTokenizer = BertTokenizer.from_pretrained("bert-base-uncased") if not os.path.exists(config.Choice) else BertTokenizer.from_pretrained(config.Choice)
        self.choice_model: BertForMultipleChoice = BertForMultipleChoice.from_pretrained("bert-base-uncased") if not os.path.exists(config.Choice) else BertForMultipleChoice.from_pretrained(config.Choice)

        # Rewrite the question into complete one
        self.rewrite_tokenizer: BartTokenizer = BartTokenizer.from_pretrained("eugenesiow/bart-paraphrase") if not os.path.exists(config.PLM) else BartModel.from_pretrained(config.PLM)
        self.rewrite_model: BartForConditionalGeneration = BartForConditionalGeneration.from_pretrained("eugenesiow/bart-paraphrase") if not os.path.exists(config.PLM) else BartForConditionalGeneration.from_pretrained(config.PLM)


    def init_hidden(self) -> List[torch.Tensor, torch.Tensor]:
        self.hidden = [self.initial_hidden]


    def choose_topic_entity(self, topic_entities: List[str], question: str) -> List[str, int]:
        topic_choices = list(map(lambda x: self.choice_template.format(x), topic_entities))
        topic_choices = self.choice_tokenizer([self.choice_template] * len(topic_choices), topic_choices, return_tensors='pt', padding=True)

        labels = torch.tensor(0).unsqueeze(0)
        outputs = self.choice_model(
            **{k: v.unsqueeze(0) for k, v in topic_choices.items()}, labels=labels
        )
        logits, loss = outputs.loss, outputs.logits
        topic_idx = torch.argmax(logits).detach().cpu().item()
        return topic_entities[topic_idx], topic_idx


    def rewrite(self, topic_entity: str, question: str) -> str:
        input_seq = topic_entity + ", " + question
        
        # Rephrase the question without explicity mentioned topic entity
        seq_ = self.rewrite_tokenizer(input_seq, return_tensors='pt')
        generate_ids = self.rewrite_model.generate(seq_['input_ids'])
        generate_sentences = self.rewrite_tokenizer.batch_decode(generate_ids, skip_special_tokens=True)

        return generate_sentences


    def tokenize_sentence(self, sentences: List[str]) -> torch.Tensor:
        sentences_ids = [torch.tensor(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(s)), dtype=float) for s in sentences]
        return torch.nn.utils.rnn.pad_sequence(sentences_ids, batch_first=True).type(torch.LongTensor).unsqueeze(0).to(self.device)


    def attention_fuse(self, q_tensors: torch.Tensor) -> torch.Tensor:
        b, t, e = q_tensors.size()
        h = self.heads
        assert e == self.hidden_size, f"Input dim not match, expect {self.hidden_size}, get {e}."

        s = e // h
        
        keys: torch.Tensor = q_tensors.view(b, t, h, s)
        queries: torch.Tensor = q_tensors.view(b, t, h, s)
        values: torch.Tensor = q_tensors.view(b, t, h, s)

        # Fold head into batch
        keys: torch.Tensor = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        queries: torch.Tensor = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values: torch.Tensor = values.transpose(1, 2).contiguous().view(b * h, t, s)

        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))
        
        # Get dot product
        dot = torch.bmm(queries, keys.transpose(1, 2))
        assert dot.size() == (b*h, t, t)

        dot = F.softmax(dot, dim=2)
        
        out = torch.bmm(dot, values).view(b, h, t, s)
        out = out.transpose(1, 2).contiguous().view(b, t, s * h)
        return out


    def get_loss(self, raw_logits: torch.Tensor, gold_score: np.ndarray, q_vec: torch.Tensor, ga_vecs: torch.Tensor, alpha: int = 0) -> torch.Tensor:
        logits = F.softmax(raw_logits, dim=2)

        # FIXME use for debug
        if torch.isnan(logits).any():
            print(logits[:10])
            exit()
        k = np.min([1, logits.size()[1]])

        # KL_loss
        kl_loss = nn.KLDivLoss(reduction='sum')(logits.log(), gold_score)
        # MSE_loss
        mse_loss = nn.MSELoss(reduction='mean')(q_vec, ga_vecs)

        return alpha * kl_loss + (1 - alpha) * mse_loss


    def debug(self, qs_ids, cps, ga_ids):
        # get idx
        q_idx = torch.sum(1 - torch.eq(qs_ids, 0).type(torch.LongTensor), 2).squeeze(0) - 1
        ga_idx = torch.sum(1 - torch.eq(ga_ids, 0).type(torch.LongTensor), 2).squeeze(0) - 1
        ans_idx = torch.sum(1 - torch.eq(cps, 0).type(torch.LongTensor), 2).squeeze(0) - 1

        # embedding
        q_embeddings = self.embedder(qs_ids)
        ans_embeddings = self.embedder(cps)
        ga_embeddings = self.embedder(ga_ids)

        # encoding
        q_encs, _ = self.q_encoder(q_embeddings)
        ans_encs, _ = self.cp_encoder(ans_embeddings)
        ga_encs, _ = self.cp_encoder(ga_embeddings)

        # pool
        q_vecs = self.pooler(q_encs, q_idx).unsqueeze(0)
        ans_vecs = self.pooler(ans_encs, ans_idx).unsqueeze(0)
        ga_vecs = self.pooler(ga_encs, ga_idx).unsqueeze(0)
        ...
        if q_vecs.size()[1] != 1:
            q_vecs = self.attention_fuse(q_vecs)

        q_vecs = torch.mean(q_vecs, dim=1).unsqueeze(0)

        dot_sim = torch.bmm(q_vecs, ans_vecs.transpose(1, 2))
        dot_sim = F.softmax(dot_sim, dim=2)

        return dot_sim


    def forward(self, instance, cps_ids, time) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the raw_logits, encoded questions and encoded gold answer

            :param batch: the instances of conversation
            :param cps: the retrieved candidate answer paths
            :param time: the turn of conversation
        """

        qs_ids = instance['questions'][time]['qs_ids']
        ga_ids = instance['questions'][time]['ans_ids']
        
        # get idx
        q_idx = torch.sum(1 - torch.eq(qs_ids, 0).type(torch.LongTensor), 2).squeeze(0) - 1
        ga_idx = torch.sum(1 - torch.eq(ga_ids, 0).type(torch.LongTensor), 2).squeeze(0) - 1
        ans_idx = torch.sum(1 - torch.eq(cps_ids, 0).type(torch.LongTensor), 2).squeeze(0) - 1

        # embedding
        q_embeddings = self.embedder(qs_ids)
        ans_embeddings = self.embedder(cps_ids)
        ga_embeddings = self.embedder(ga_ids)

        # encoding
        q_encs, _ = self.q_encoder(q_embeddings)
        ans_encs, _ = self.cp_encoder(ans_embeddings)
        ga_encs, _ = self.cp_encoder(ga_embeddings)

        # pool
        q_vecs = self.pooler(q_encs, q_idx)
        ans_vecs = self.pooler(ans_encs, ans_idx)
        ga_vecs = self.pooler(ga_encs, ga_idx)
        ...
        if q_vecs.size()[1] != 1:
            q_vecs = self.attention_fuse(q_vecs)

        # mean
        q_vecs = torch.mean(q_vecs, dim=1)
        dot_sim = torch.bmm(q_vecs, ans_vecs.transpose(1, 2))
        dot_sim = F.softmax(dot_sim, dim=2)
        return dot_sim, q_vecs, ans_vecs, ga_vecs

