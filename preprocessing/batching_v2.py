import numpy as np
from data.data import TorchGraphData
import torch
import nxmetis

#####################################################################
def get_batch_x(data : TorchGraphData, partition : np.array):
    return data.x[partition]

def get_batch_edge_index(data : TorchGraphData, partition : np.array):
    _batch_edge_index = torch.isin(data.edge_index, torch.tensor(partition))
    print(torch.argwhere(_batch_edge_index))

def get_batch_edge_attr(data : TorchGraphData, partition : np.array):
    pass
def get_batch_graphs(data : TorchGraphData, batch_size : int, recursive : bool):
    n_batchs = int(data.num_nodes / batch_size)

    (_, partitions) = nxmetis.partition(
        G=data.graph,
        nparts=n_batchs
    )

    partition = partitions[0]
    get_batch_edge_index(data, partition)

    