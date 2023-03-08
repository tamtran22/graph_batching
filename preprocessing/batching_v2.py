import numpy as np
from data.data import TorchGraphData
import torch
import nxmetis
import networkx as nx

#####################################################################

def _get_graph_partition(data : TorchGraphData, partition : np.array, recursive : bool):
    edge_index = data.edge_index.numpy()
    partition_edge_mark = np.isin(edge_index, partition)
    if recursive:
        partition_edge_mark = np.logical_or(partition_edge_mark[0], partition_edge_mark[1])
    else:
        partition_edge_mark = np.logical_and(partition_edge_mark[0], partition_edge_mark[1])
    partition_edge_id = np.argwhere(partition_edge_mark == True).squeeze(1)
    partition_edge_index = edge_index[:,partition_edge_id]

    partition = np.unique(np.concatenate(partition_edge_index))
    index = lambda n : list(partition).index(n)
    v_index = np.vectorize(index)
    
    partition_edge_index = torch.tensor(v_index(partition_edge_index))
    
    partition_x = data.x[partition]

    partition_edge_attr = data.edge_attr[partition_edge_id]

    partition_pressure = data.pressure[partition]

    partition_velocity = data.velocity[partition_edge_id]

    if 'flowrate' in data._store:
        partition_flowrate = data.flowrate[partition_edge_id]
        return TorchGraphData(
            x = partition_x,
            edge_index = partition_edge_index,
            edge_attr = partition_edge_attr,
            pressure = partition_pressure,
            flowrate = partition_flowrate,
            velocity = partition_velocity
        )
    
    if 'flowrate_bc' in data._store:
        partition_flowrate_bc = data.flowrate_bc[partition_edge_id]
        return TorchGraphData(
            x = partition_x,
            edge_index = partition_edge_index,
            edge_attr = partition_edge_attr,
            pressure = partition_pressure,
            flowrate_bc = partition_flowrate_bc,
            velocity = partition_velocity
        )
    
def _get_time_partition(data : TorchGraphData, time_id : np.array):
    if 'flowrate' in data._store:
        return TorchGraphData(
            x = data.x,
            edge_index = data.edge_index,
            edge_attr = data.edge_attr,
            pressure = data.pressure[:,time_id],
            flowrate = data.flowrate[:,time_id],
            velocity = data.velocity[:,time_id]
        )
    if 'flowrate_bc' in data._store:
        return TorchGraphData(
            x = data.x,
            edge_index = data.edge_index,
            edge_attr = data.edge_attr,
            pressure = data.pressure[:,time_id],
            flowrate_bc = data.flowrate_bc[:,time_id],
            velocity = data.velocity[:,time_id]
        )

def get_batch_graphs(data : TorchGraphData, batch_size : int, batch_n_times : int, 
                    recursive : bool):
    # Spatial/graph partitioning.
    n_batchs = int(data.number_of_nodes / batch_size)

    (_, partitions) = nxmetis.partition(
        G=data.graph,
        nparts=n_batchs
    )

    _batch_graphs = []
    for partition in partitions:
        _batch_graphs.append(_get_graph_partition(data, partition, recursive))
    
    # Temporal/time partitioning.
    time_ids = []
    i = 0
    while i < data.number_of_timesteps - 1:
        i_end = i + batch_n_times + recursive
        time_ids.append(np.arange(i, min(i_end, data.number_of_timesteps), 1, dtype=int))
        i = i_end - recursive
    
    batch_graphs = []
    for _batch_graph in _batch_graphs:
        for time_id in time_ids:
            batch_graphs.append(_get_time_partition(_batch_graph, time_id))

    return batch_graphs

def _test_graph():
    G = nx.Graph()
    for i in range(23):
        G.add_node(i)
    edge_index = [(0,1),(1,2),(1,3),(2,4),(2,5),(3,6),(3,7),(4,8),\
        (4,9),(5,10),(6,11),(7,12),(8,13),(8,14),(10,15),(10,16),(11,17),(11,18),\
        (12,19),(12,20),(19,21),(19,22)]
    G.add_edges_from(edge_index)
    return G, np.transpose(np.array(edge_index))


#####################################################################
