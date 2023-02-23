import numpy as np
from data.data import TorchGraphData

import nxmetis

#####################################################################
def get_batch_x(data : TorchGraphData, partition : np.array):
    return data.x[partition]

def get_batch_edge_index(data : TorchGraphData, partition : np.array):
    

def get_batch_edge_attr(data : TorchGraphData, partition : np.array):
    pass
def get_batch_graphs(data : TorchGraphData, batch_size : int, recursive : bool):
    n_batchs = int(data.num_nodes / batch_size)

    (_, partitions) = nxmetis.partition(
        G=data.graph,
        nparts=n_batchs
    )

    