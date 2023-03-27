import numpy as np
import torch
from torch import Tensor
from typing import Union
# import networkx as nx
from data.data import TorchGraphData


class objectview(object):
    def __init__(self, d) -> None:
        self.__dict__ = d


def min_max_scaler(x : Tensor, min : Union[float, Tensor],
                    max : Union[float, Tensor]) -> Tensor:
    return (x - min) / (max - min)

def standard_scaler(x : Tensor, mean : Union[float, Tensor], 
                    std: Union[float, Tensor], eps : float=1e-10) -> Tensor:
    return (x - mean) / (std + eps)

# def robust_scaler(x : Tensor, axis=0) -> Tensor:
#     return (x - x.median(axis)) / 10

def flowrate_bc_sin(flowrate, n_edge : int, 
                    n_time : int)->torch.Tensor:
    _flowrate = []
    T = 4.8 # seconds
    for i in range(n_time):
        t = i * T / (n_time - 1)
        value = flowrate(t)
        _flowrate.append(torch.full(size=(n_edge, 1), fill_value=value))
    return torch.cat(_flowrate, dim=1)

def normalize_data_1d(data, **kwargs):
    kwargs = objectview(kwargs)
    
    # x
    x = data.x
    if hasattr(kwargs, 'x_min') and hasattr(kwargs, 'x_max'):
        x = min_max_scaler(x, min=kwargs.x_min, max=kwargs.x_max)

    # edge_index
    edge_index = data.edge_index

    # edge_attr
    edge_attr = data.edge_attr
    if hasattr(kwargs, 'ea_min') and hasattr(kwargs, 'ea_max'):
        edge_attr = min_max_scaler(edge_attr, min=kwargs.ea_min,
                            max=kwargs.ea_max)
    
    # pressure
    pressure = data.pressure
    if hasattr(kwargs, 'p_min') and hasattr(kwargs, 'p_max'):
        pressure = min_max_scaler(pressure, min=kwargs.p_min,
                            max=kwargs.p_max)

    # velocity
    velocity = data.velocity
    if hasattr(kwargs, 'u_min') and hasattr(kwargs, 'u_max'):
        velocity = min_max_scaler(velocity, min=kwargs.u_min,
                            max=kwargs.u_max)

    # flowrate bc
    flowrate_bc = data.flowrate[0,:] # Fixed flowrate bc at entrance
    # flowrate_bc = min_max_scaler(flowrate_bc, min=flowrate_bc.min(),
    #                              max=flowrate_bc.max())
    n_edge = data.flowrate.size(0)
    flowrate_bc = [flowrate_bc.unsqueeze(0)] * n_edge
    flowrate_bc = torch.cat(flowrate_bc, dim=0)

    # loss weight by diameter
    weight = cal_weight(x = edge_attr[:,0], bins=100)

    return TorchGraphData(x=x,edge_index=edge_index,edge_attr=edge_attr,
                        pressure=pressure, velocity=velocity,
                        flowrate_bc = flowrate_bc, weight=weight)

def cal_weight(x : np.array, bins=1000) -> np.array:
    (count, bin) = np.histogram(x, bins=bins)
    N = x.shape[0]
    def _weight(value : float):
        _bin_id = np.where(bin >= value)[0][0] - 1
        _weight = 1. / max(count[_bin_id], 1.)
        return _weight
    v_weight = np.vectorize(_weight)
    return v_weight(x)

if __name__ == '__main__':
    # x = Tensor([[1,2], [4,13], [4, 3]])
    # y = Tensor([1,2,4,2,1])
    # print(x.quantile(0.75, dim=0))
    normalize_data_1d(x = 'sfdfdf', y='fdsfd')