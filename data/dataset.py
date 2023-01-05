import os
import numpy as np
import torch
from torch_geometric.data import Dataset
from data.data import TorchGraphData
from typing import Optional, Callable, Union, List, Tuple


class BaseGraphDataset(Dataset):
    r"""Dataset base class for 
    
    """
    def __init__(self, 
        root_dir: Optional[str] = None, 
        transform: Optional[Callable] = None, 
        pre_transform: Optional[Callable] = None, 
        pre_filter: Optional[Callable] = None,
        loader: Optional[bool] = False
    ):
        self.raw_dir = raw_dir
        self.loader = loader
        super().__init__(root_dir, transform, pre_transform, pre_filter)
    
    @property
    def data_names(self) -> Union[List[str], Tuple]:
        raise NotImplemented

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return super().processed_file_names