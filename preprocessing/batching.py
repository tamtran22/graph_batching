import numpy as np
from data.data import TorchGraphData

import networkx as nx

import nxmetis


def metis_batching_torch(data : TorchGraphData, relative_batch_size : int = 100, recursive : bool = False):
    num_batch = int(data.num_nodes / relative_batch_size)
    
    (_, parts) = nxmetis.partition(
        G=data.graph,
        nparts=num_batch
    )

    batchs = []
    for i in range(num_batch):
        # ID of nodes in batch.
        batch_nodes = np.array(parts[i])
        # ID of edges in batch.
        batch_edges = np.isin(data.edge_index.numpy(), batch_nodes)
        if recursive:
            batch_edges = np.logical_or(batch_edges[0], batch_edges[1])
        else:
            batch_edges = np.logical_and(batch_edges[0], batch_edges[1])
        batch_edges = np.where(batch_edges == True)[0]
        # Edge index includes all edge in batch.
        batch_edge_index = data.edge_index[:, batch_edges]
        batch_nodes = np.where(np.isin(list(range(data.num_nodes)), batch_edge_index) == True)[0].tolist()
        for j in range(batch_edges.shape[0]):
            batch_edge_index[0,j] = batch_nodes.index(batch_edge_index[0,j])
            batch_edge_index[1,j] = batch_nodes.index(batch_edge_index[1,j])
        # Node features of nodes in batch.
        batch_x = data.x[batch_nodes]
        # Edge featrues of edges in batch.
        if data.edge_attr is not None:
            batch_edge_attr = data.edge_attr[batch_edges]
        else:
            batch_edge_attr = None
        # Batching remain data properties.
        batch_kwargs = {}
        for var in data._store:
            if var in ['x', 'edge_index', 'edge_attr']:
                continue
            if data._store[var].shape[0] == data.num_nodes:
                batch_kwargs[var] = data._store[var][batch_nodes]
            if data._store[var].shape[0] == data.num_edges:
                batch_kwargs[var] = data._store[var][batch_edges]
        # Batch graph data.
        batch_data = TorchGraphData(
            x=batch_x,
            edge_index=batch_edge_index,
            edge_attr=batch_edge_attr
        )
        for var in batch_kwargs:
            setattr(batch_data, var, batch_kwargs[var])
        batchs.append(batch_data)
    return batchs
