import torch
from torch.nn.modules.loss import _Loss, _WeightedLoss
from typing import Optional
from torch import Tensor
import numpy as np

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

# All input tensor are transformed node to edge
def energy_balanced_error(P, Q, L, d, delta_t):
    # params
    rho = 1.12
    pi = 3.1415926
    mu = 1.64e-5
    K = 1.
    # term
    Uns = (4*rho*L) / (pi * torch.square(d))
    Kin = (16*K*rho) / (pi**2 * torch.square(torch.square(d)))
    Vis = (128*mu*L) / (pi * torch.square(torch.square(d)))
    # loss
    P_curr = P[:,1:]
    Q_curr = Q[:,1:]
    Q_prev = Q[:,:-2]
    loss = P_curr + (Uns / delta_t + Vis) * Q_curr + Kin * Q_prev + (Uns / delta_t) * Q_prev
    return torch.mean(loss)
