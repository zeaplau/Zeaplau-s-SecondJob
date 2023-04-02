"""
    TODO: Debug this module
"""
import torch
import numpy as np
from torch import nn

import pdb

class QANTXent(nn.Module):
    """QANTXent Loss for our model, we use dot product to replace the cosine similarity 
    """
    def __init__(self, temperature: int, device) -> None:
        super(QANTXent, self).__init__()
        self.temperature = temperature
        self.cos_func = nn.CosineSimilarity(dim=1, eps=1e-6)
        # self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        self.device = device


    def cal_sim(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        # sim = self.cos_func(v1.squeeze(0), v2)
        sim = torch.mm(v1.squeeze(0), v2.transpose(0, 1))
        return sim


    def forward(self, question_vec: torch.Tensor, positive_sample: torch.Tensor, negative_sample: torch.Tensor, type: str="mean") -> torch.Tensor:
        """Get mean contrastive loss 
        """
        positive_sim = self.cal_sim(question_vec, positive_sample)
        # negative_sim = self.cal_sim(question_vec, negative_sample)
        all_sim = self.cal_sim(question_vec, torch.cat([positive_sample, negative_sample]))

        positive_logits = torch.exp(positive_sim / self.temperature)
        # negative_logits = torch.exp(negative_sim / self.temperature)
        all_logits = torch.exp(all_sim / self.temperature)
        # negative_sum = torch.sum(negative_logits)
        all_sum = torch.sum(all_logits)
        
        n = positive_sample.size()[0]

        # loss = torch.sum((positive_logits / negative_sum).log())
        loss = torch.sum((positive_logits / all_sum + 1e-10).log())
        return -loss / n