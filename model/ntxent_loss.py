"""
    TODO: Debug this module
"""
import torch
from torch import nn

import pdb

class QANTXent(nn.Module):
    """QANTXent Loss for our model, we use dot product to replace the cosine similarity 
    """
    def __init__(self, temperature: int) -> None:
        super(QANTXent, self).__init__()
        self.temperature = temperature
        self.cos_func = nn.CosineSimilarity(dim=1, eps=1e-6)


    def cal_sim(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        # reutrn nn.CosineSimilarity(dim=1, eps=1e-6)
        sim = self.cos_func(v1.squeeze(0), v2)
        return sim


    def forward(self, question_vec: torch.Tensor, positive_sample: torch.Tensor, negative_sample: torch.Tensor, type: str="mean") -> torch.Tensor:
        """Get mean contrastive loss 
        """
        positive_sim = self.cal_sim(question_vec, positive_sample) / self.temperature
        negative_sim = self.cal_sim(question_vec, negative_sample) / self.temperature

        exp_positive = torch.exp(positive_sim)
        exp_negative = torch.exp(negative_sim)
        negative_sum = torch.sum(exp_negative, dim=-1) 

        n = exp_positive.size()[-1]
        # check here
        loss = -torch.sum((exp_positive / (negative_sum.unsqueeze(-1) + 1e-6)), dim=-1).log() / (n + 1e-6)
        return torch.mean(loss) if type == "mean" else torch.sum(loss, dim=2)