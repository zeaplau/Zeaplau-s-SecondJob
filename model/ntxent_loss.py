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


    def dot_product(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        return v1 @ v2.transpose(0, 1)


    def forward(self, question_vec: torch.Tensor, positive_sample: torch.Tensor, negative_sample: torch.Tensor, type: str="mean") -> torch.Tensor:
        """Get mean contrastive loss 
        """
        positive_sim = self.dot_product(question_vec, positive_sample) / self.temperature
        negative_sim = self.dot_product(question_vec, negative_sample) / self.temperature

        exp_positive = torch.exp(positive_sim)
        exp_negative = torch.exp(negative_sim)
        negative_sum = torch.sum(exp_negative, dim=2) 
        
        n = exp_positive.size()[-1]
        loss = -torch.sum((exp_positive / negative_sum.unsqueeze(-1)), dim=2).log() / n
        return torch.mean(loss) if type == "mean" else torch.sum(loss, dim=2)