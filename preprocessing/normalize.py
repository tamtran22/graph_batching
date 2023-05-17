import numpy as np
import torch
from torch import Tensor
from typing import Union, Tuple, List, Callable
# import networkx as nx
from data.data import TorchGraphData





class objectview(object):
    def __init__(self, d) -> None:
        self.__dict__ = d

def min_max_scaler(x : Tensor, min : Union[float, Tensor],
                    max : Union[float, Tensor]) -> Tensor:
    return -1+2*(x - min) / (max - min)

def standard_scaler(x : Tensor, mean : Union[float, Tensor], 
                    std: Union[float, Tensor], eps : float=1e-10) -> Tensor:
    return (x - mean) / (std + eps)

def logarithmic_scaler_v1(x : Tensor,
                min : Union[float, Tensor],
                max : Union[float, Tensor], pivot : str='mean') -> Tensor:
    if pivot == 'mean':
        mean = x.mean()
    elif pivot == 'median':
        mean = x.median()
    # C_min = int(torch.log((10^-1-1)/(min-mean)))
    # C_max = int(torch.log((10^1-1)/(max-mean)))
    C = np.min([-int(torch.log(torch.abs(min-mean))),-int(torch.log(torch.abs(max-mean)))])
    print(C)
    log_scale = 10.**C
    # log_scale = 1e12
    return torch.log((x - mean) * log_scale + 1)

def logarithmic_scaler_v2(x : Tensor,
                logscale : Union[float, Tensor],
                pivot : str='mean') -> Tensor:
    if pivot == 'mean':
        mean = x.mean()
    elif pivot == 'median':
        mean = x.median()
    return torch.log((x - mean) * logscale + 1)

def logarithmic_scaler_v3(x : Tensor,
                min : Union[float, Tensor],
                max : Union[float, Tensor],
                logscale : float = 1e12):
    F = lambda x : torch.sign(x) * torch.log10(torch.abs(x)*logscale + 1)
    x = F(x)
    min = F(min)
    max = F(max)
    return 2 * (x - min) / (max - min) - 1



def normalize_graph(data, **kwargs) -> TorchGraphData:
    '''
    Min-max scaling for graph data with given min-max params.
    '''
    data_dict = {}
    for key in data._store:
        data_dict[key] = data._store[key]
        if (f'{key}_min' in kwargs.keys()) and (f'{key}_max' in kwargs.keys()):
            data_dict[key] = min_max_scaler(
                x=data._store[key],
                min=kwargs[f'{key}_min'],
                max=kwargs[f'{key}_max']
            )
        
        if (f'{key}_min' in kwargs.keys()) and (f'{key}_max' in kwargs.keys()) and (f'{key}_logscale' in kwargs.keys()):
            data_dict[key] = logarithmic_scaler_v3(
                x=data._store[key],
                min=kwargs[f'{key}_min'],
                max=kwargs[f'{key}_max'],
                logscale=kwargs[f'{key}_logscale']
            )

        if (f'{key}_pipeline' in kwargs.keys()):
            data_dict[key] = data._store[key]
            for tup in kwargs[f'{key}_pipeline']:
                columns = tup[0]
                scaler = tup[1]
                scaler_kwargs = tup[2]

                if scaler == 'minmax':
                    for column in columns:
                        data_dict[key][:,column] = min_max_scaler(
                            x=data_dict[key][:,column],
                            min=scaler_kwargs.min[column],
                            max=scaler_kwargs.max[column]
                        )
                
                if scaler == 'logarithmic':
                    for column in columns:
                        data_dict[key][:,column] = logarithmic_scaler_v3(
                            x=data_dict[key][:,column],
                            min=scaler_kwargs.min[column],
                            max=scaler_kwargs.max[column],
                            logscale=scaler_kwargs.logscale[column]
                        )
            
    normalized_data = TorchGraphData()
    for key in data_dict:
        setattr(normalized_data, key, data_dict[key])
    return normalized_data





def calculate_weight(x : np.array, bins=1000) -> np.array:
    '''
    Calculate weight based on bin's value count (for imbalance data).
    '''
    (count, bin) = np.histogram(x, bins=bins)
    N = x.shape[0]
    def _weight(value : float):
        _bin_id = np.where(bin >= value)[0][0] - 1
        _weight = 1. / max(count[_bin_id], 1.)
        return _weight
    v_weight = np.vectorize(_weight)
    return v_weight(x)



def calculate_derivative(data : TorchGraphData, var_name=None, axis=None, delta_t=None) -> torch.Tensor:
    if var_name==None:
        return 0
    else:
        F = data._store[var_name]
        F = F.transpose(0, axis)
        deriv_F = []
        for i in range(0, F.size(0)):
            i_prev = max(i-1, 0)
            i_next = min(i+1, F.size(0)-1)
            deriv_F_i = (F[i_next] - F[i_prev]) / ((i_next-i_prev)*delta_t)
            deriv_F.append(deriv_F_i.unsqueeze(axis))
        return torch.cat(deriv_F, dim=axis)

    



if __name__ == '__main__':
    # x = Tensor([[1,2], [4,13], [4, 3]])
    # y = Tensor([1,2,4,2,1])
    # print(x.quantile(0.75, dim=0))
    n = 10
    for i in range(0, n):
        i_prev = max(i, 0)
        i_next = min(i, n-1)
        print(i_prev, i, i_next)