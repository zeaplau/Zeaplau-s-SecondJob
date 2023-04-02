import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from copy import deepcopy
from transformers import BartTokenizer, BertForMultipleChoice, BartForConditionalGeneration, BertModel, BertTokenizer, AlbertModel, AlbertTokenizer
from typing import List, Tuple
from torch.nn import KLDivLoss, MSELoss
from model.advanced_layer import SimpleEmbLayer, SimplePooler
from model.ntxent_loss import QANTXent
from tools.create_config import ModelConfig
from tools.tokenization import BasicTokenizer

import pdb


class RefModel(nn.Module):
    def __init__(self, config, tokenizer: BasicTokenizer, device=None):
        super(RefModel, self).__init__()
        self.config = config
        self.device = device
        self.hidden_size = config.hidden_size
        self.embedder = SimpleEmbLayer(config, device=self.device)
        self.pooler = SimplePooler(config)
        self.dropout = nn.Dropout(0.3)
        self.heads = 5
        self.emb = self.hidden_size // self.heads
        
        if config.use_bert:
            self.classifer = nn.Linear(768, 1)
            self.bert_model: BertModel = BertModel.from_pretrained(config.BERT) if os.path.exists(config.BERT) else BertModel.from_pretrained("bert-base-cased")
            self.bert_tokenier: BertTokenizer = BertTokenizer.from_pretrained(config.BERT) if os.path.exists(config.BERT) else BertTokenizer.from_pretrained("bert-base-cased")
            self.cosine = nn.CosineSimilarity(dim=-1, eps=1e-6)
            # freeze front layer
            unfreeze_layers = ['layer.10', 'layer.11', 'bert.pooler', 'out.']
            for name, param in self.bert_model.named_parameters():
                param.requires_grad = False
                for ele in unfreeze_layers:
                    if ele in name:
                        param.requires_grad = True
                        break
        else:
            # LSTM tokenizer
            self.tokenizer = tokenizer
            # Conversation progress, consider to use this ?
            self.initial_hidden = (torch.randn(1, self.hidden_size).to(self.device), torch.randn(1, self.hidden_size).to(self.device))

            # Encode question and candidate answer paths
            self.q_encoder = nn.LSTM(input_size=self.hidden_size, hidden_size=int(self.hidden_size / 2), bidirectional=True, batch_first=True)
            self.cp_encoder = nn.LSTM(input_size=self.hidden_size, hidden_size=int(self.hidden_size / 2), bidirectional=True, batch_first=True)
            self.apply(self.init_glove_weight)

        self.kl_loss = nn.KLDivLoss(reduction="sum")
        self.ntxent = QANTXent(temperature=0.07, device=self.device)

        # Regard the selecting topic entity problem as multiple choice question
        self.choice_template = "Which answer is the subject of the question '{}'"
        self.choice_tokenizer: BertTokenizer = BertTokenizer.from_pretrained("bert-base-uncased") if not os.path.exists(config.BERT) else BertTokenizer.from_pretrained(config.BERT)
        self.choice_model: BertForMultipleChoice = BertForMultipleChoice.from_pretrained("bert-base-uncased") if not os.path.exists(config.BERT) else BertForMultipleChoice.from_pretrained(config.BERT)

        # Rewrite the question into complete one
        self.rewrite_tokenizer: BartTokenizer = BartTokenizer.from_pretrained("eugenesiow/bart-paraphrase") if not os.path.exists(config.BART) else BartTokenizer.from_pretrained(config.BART)
        self.rewrite_model: BartForConditionalGeneration = BartForConditionalGeneration.from_pretrained("eugenesiow/bart-paraphrase") if not os.path.exists(config.BART) else BartForConditionalGeneration.from_pretrained(config.BART)
        
        # Use choice logits
        # dot_sim, choice_logits, question_type, anaswer_entity desc
        if self.config.mix:
            self.mix_layer = nn.Linear(4, 1)


    def init_glove_weight(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, (nn.Embedding)) and self.config.hidden_size == 300 and module.weight.data.size(0) > 1000:
            if os.path.exists(self.config.Word2vec_path):
                embedding = np.load(self.config.Word2vec_path)
                module.weight.data = torch.tensor(embedding, dtype=torch.float)
                print('Pretrained GloVe embeddings init')
            else:
                assert f"{self.config.Word2vec_path} is not found."
        if isinstance(module, (nn.Linear)) and module.bias is not None:
            module.bias.data.zero_()


    def init_hidden(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self.hidden = [self.initial_hidden]
    

    def encode_answer_desc(self, answer_entities, cache):
        answer_desc = map(lambda x: cache[x] if x in cache.keys() else ["UNK"], answer_entities)
        if self.config.use_bert:
            ...
        else:
            desc_ids = list(map(lambda x: torch.tensor(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(x)), dtype=torch.long), answer_desc))
            pad_desc = torch.nn.utils.rnn.pad_sequence(desc_ids, batch_first=True)
            pool_idx = torch.sum(1 - torch.eq(pad_desc, 0).type(torch.LongTensor), 1).squeeze(0) - 1

            # encode
            desc_embedding = self.embedder(pad_desc)
            desc_encoded = self.q_encoder(desc_embedding)
            desc_vecs = self.dropout(desc_encoded)
            desc_vecs = self.pooler(desc_vecs)
        return desc_vecs


    def choose_topic_entity(self, topic_entities: List[str], question: str) -> Tuple[str, int]:
        topic_choices = list(map(lambda x: self.choice_template.format(x), topic_entities))
        topic_choices = self.choice_tokenizer([self.choice_template] * len(topic_choices), topic_choices, return_tensors='pt', padding=True).to(self.device)

        labels = torch.tensor(0).unsqueeze(0).to(self.device)
        outputs = self.choice_model(
            **{k: v.unsqueeze(0) for k, v in topic_choices.items()}, labels=labels
        )
        logits, loss = outputs.logits, outputs.loss
        topic_idx = torch.argmax(logits).detach().cpu().item()
        return topic_entities[topic_idx], logits


    def rewrite(self, topic_entity: str, questions: str) -> str:
        input_seqs = list(map(lambda x: topic_entity + ", " + x, questions))
        
        # Rephrase the question without explicity mentioned topic entity
        seqs_ = self.rewrite_tokenizer(input_seqs, padding=True, truncation=True, return_tensors='pt').to(self.device)
        generate_ids = self.rewrite_model.generate(seqs_['input_ids'], max_length=30)
        generate_sentences = self.rewrite_tokenizer.batch_decode(generate_ids, skip_special_tokens=True)

        return generate_sentences


    def tokenize_sentence(self, sentences: List[str]):
        if self.config.use_bert:
            return self.albert_tokenier.encode_plus(sentences, return_tensors="pt").to(self.device)
        else:
            sentences_ids = [torch.tensor(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(s)), dtype=torch.long) for s in sentences]
            return torch.nn.utils.rnn.pad_sequence(sentences_ids, batch_first=True).type(torch.LongTensor).to(self.device)


    def attention_fuse(self, q_tensors: torch.Tensor) -> torch.Tensor:
        b = 1
        t, e = q_tensors.size()
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
        return out, dot


    def get_loss(self, is_const, raw_logits: torch.Tensor, gold_score: np.ndarray, q_vecs: torch.Tensor, ga_vecs: torch.Tensor, nega_vecs: torch.Tensor, alpha: int = 0) -> torch.Tensor:
        if is_const is not None:
            return 0.
        logits = F.softmax(raw_logits, dim=1)

        # use for debug
        if torch.isnan(logits).any():
            print(logits[:10])
            exit()

        if isinstance(gold_score, np.ndarray):
            gold_score = torch.tensor(gold_score, dtype=torch.float).repeat(logits.size()[0], 1).to(self.device)

        # KL_loss
        kl_loss = self.kl_loss.forward(logits.log(), gold_score)

        # NTXent Loss
        ntx_loss = self.ntxent.forward(question_vec=q_vecs, positive_sample=ga_vecs, negative_sample=nega_vecs)

        # if kl_loss < 0 or ntx_loss < 0:
        #     print(f"---- kl_loss {kl_loss} ntx_loss {ntx_loss} ----")

        # the kl_loss need to recover from mean for num of questions and ref
        return alpha * kl_loss + (1 - alpha) * ntx_loss


    def forward(self, instance, cps, ref_qs, time) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the raw_logits, encoded questions and encoded gold answer

            :param batch: the instances of conversation
            :param cps: the retrieved candidate answer paths
            :param time: the turn of conversation
        """
        # gold_paths = list(map(lambda x: x + [instance.questions[time]['gold_answer_text']], instance.questions[time]['gold_paths'])) if len(instance.questions[time]['gold_paths']) != 0 else ["UNK"]
        gold_paths = list(map(lambda x: x, instance.questions[time]['gold_paths'])) if len(instance.questions[time]['gold_paths']) != 0 else instance.questions[time]['gold_answer_text']
        negative_sample = deepcopy(instance.candidate_paths)

        for p in gold_paths:
            if p in negative_sample:
                negative_sample.remove(p)

        if self.config.use_bert:
            labels = torch.tensor(0).unsqueeze(0)
            # tokenize
            gold_paths = [" ".join(p) for p in gold_paths]
            negative_paths = [" ".join(p) for p in negative_sample]
            encoding = self.bert_tokenier(cps, return_tensors="pt", padding=True, truncation=True).to(self.device)
            ga_encoding = self.bert_tokenier(gold_paths, return_tensors="pt", padding=True, truncation=True).to(self.device)
            ng_encoding = self.bert_tokenier(negative_paths, return_tensors="pt", padding=True, truncation=True).to(self.device)
            q_encoding = self.bert_tokenier(ref_qs, return_tensors="pt", padding=True, truncation=True).to(self.device)

            # get hidden state
            outputs = self.bert_model(**encoding)
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            q_vecs = self.bert_model(**q_encoding)[1]
            q_vecs = self.dropout(q_vecs)
            ga_vecs = self.bert_model(**ga_encoding)[1]
            ga_vecs = self.dropout(ga_vecs)
            nega_vecs = self.bert_model(**ng_encoding)[1]
            nega_vecs = self.dropout(nega_vecs)
            
            # get logits
            logits = self.cosine(q_vecs, pooled_output)
            logits = logits.view(-1, len(cps))
            return logits, q_vecs, pooled_output, ga_vecs, nega_vecs
        else:
            cps_ids = self.tokenize_sentence(cps)
            negative_sample = [" ".join(x) for x in negative_sample]
            negative_ids = self.tokenize_sentence(negative_sample)
            # negative_ids = torch.nn.utils.rnn.pad_sequence(
            #     list(
            #         map(lambda x: torch.tensor(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(" ".join(x)))), negative_sample)), 
            #         batch_first=True)
            qs_ids = self.tokenize_sentence(ref_qs)

            # qs_ids = torch.nn.utils.rnn.pad_sequence(
            #     list(
            #         map(lambda x: torch.tensor(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(x))), ref_qs)), 
            #         batch_first=True)

            # qs_ids = instance.questions[time]['qs_id'] # not rewrite question
            ga_ids = instance.questions[time]['ans_id']

            # get idx
            q_idx = torch.sum(1 - torch.eq(qs_ids, 0).type(torch.LongTensor), 1).squeeze(0) - 1
            ga_idx = torch.sum(1 - torch.eq(ga_ids, 0).type(torch.LongTensor), 1).squeeze(0) - 1
            ans_idx = torch.sum(1 - torch.eq(cps_ids, 0).type(torch.LongTensor), 1).squeeze(0) - 1
            nega_idx = torch.sum(1 - torch.eq(negative_ids, 0).type(torch.LongTensor), 1).squeeze(0) - 1

            # embedding
            q_embeddings = self.embedder(qs_ids.to(self.device)) # [q_nums, q_len, hidden_size]
            ans_embeddings = self.embedder(cps_ids.to(self.device)) # [cps_nums, cp_len, hidden_size]
            ga_embeddings = self.embedder(ga_ids.to(self.device)) # [gold_ans_nums, ga_len, hidden_size]
            nega_embeddings = self.embedder(negative_ids.to(self.device))

            # encoding
            q_encs, _ = self.q_encoder(q_embeddings) # [q_nums, q_len, hidden_size]
            ans_encs, _ = self.cp_encoder(ans_embeddings) # [cps_nums, cp_len, hidden_size]
            ga_encs, _ = self.cp_encoder(ga_embeddings) # [gold_ans_nums, ga_len, hidden_size]
            nega_encs, _ = self.cp_encoder(nega_embeddings)

            # dropout
            q_encs = self.dropout(q_encs)
            ans_encs = self.dropout(ans_encs)
            ga_encs = self.dropout(ga_encs)
            nega_encs = self.dropout(nega_encs)

            # pool
            q_vecs = self.pooler(q_encs, q_idx)
            ans_vecs = self.pooler(ans_encs, ans_idx)
            ga_vecs = self.pooler(ga_encs, ga_idx)
            nega_vecs = self.pooler(nega_encs, nega_idx)

            if q_vecs.size()[1] != 1:
                q_vecs, dot = self.attention_fuse(q_vecs)

            # mean
            # q_vecs = torch.mean(q_vecs, dim=0)
            dot_sim = torch.mm(q_vecs.squeeze(0), ans_vecs.transpose(0, 1))
            # dot_sim = torch.cosine_similarity(q_vecs.squeeze(0), ans_vecs).unsqueeze(0)
            return dot_sim, q_vecs, ans_vecs, ga_vecs, nega_vecs
