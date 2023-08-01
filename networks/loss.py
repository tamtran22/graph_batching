import torch
from torch.nn.modules.loss import _Loss, _WeightedLoss
from typing import Optional
from torch import Tensor

def mean_squared_error(input, target):
    return torch.mean(torch.square(input - target))

def weighted_mean_squared_error(input, target, weight):
    squared_error = torch.square(input - target)
    # weighted_square_error = torch.einsum('ijk,i->ijk', squared_error, weight)
    weighted_square_error = torch.einsum('ij,i->ij', squared_error, weight)
    return torch.mean(weighted_square_error)

class WeightedMSELoss(_Loss):
    def __init__(self, size_average=None, 
                reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
    def forward(self, input : Tensor, target : Tensor, weight : Tensor):
        return weighted_mean_squared_error(input, target, weight)