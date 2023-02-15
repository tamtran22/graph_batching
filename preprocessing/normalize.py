import numpy as np
from torch import Tensor
from typing import Union
import networkx as nx

import nxmetis

def min_max_scaler(x : Tensor, min : Union[float, Tensor],
                    max : Union[float, Tensor]) -> Tensor:
    return (x - min) / (max - min)

def standard_scaler(x : Tensor, mean : Union[float, Tensor], 
                    std: Union[float, Tensor], eps : float=1e-10) -> Tensor:
    return (x - mean) / (std + eps)

# def robust_scaler(x : Tensor, axis=0) -> Tensor:
#     return (x - x.median(axis)) / 10

if __name__ == '__main__':
    x = Tensor([[1,2], [4,13], [4, 3]])
    y = Tensor([1,2,4,2,1])
    print(x.quantile(0.75, dim=0))