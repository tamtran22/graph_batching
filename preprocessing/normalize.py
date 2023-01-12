import numpy as np
# from data.data import TorchGraphData
from torch import Tensor

import networkx as nx

import nxmetis

def min_max_scaler(x : Tensor, axis=0) -> Tensor:
    return (x - x.min(axis).values) / (x.max(axis).values - x.min(axis).values)

def standard_scaler(x : Tensor, axis=0, eps=1e-10) -> Tensor:
    return (x - x.mean(axis)) / (x.std(axis) + eps)

def robust_scaler(x : Tensor, axis=0) -> Tensor:
    return (x - x.median(axis)) / 10

# def normalize(data : TorchGraphData):
    # pass

if __name__ == '__main__':
    x = Tensor([[1,2], [4,13], [4, 3]])
    y = Tensor([1,2,4,2,1])
    print(x.quantile(0.75, dim=0))