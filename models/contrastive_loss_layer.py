import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, input1, input2, target):
        d = F.pairwise_distance(input1, input2, p=2)
        return torch.mean((target) * torch.pow(d, 2) +
                          (1-target) * torch.pow(torch.clamp(self.margin - d, min=0.0), 2))
