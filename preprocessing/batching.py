import numpy as np
from data.data import TorchGraphData
import torch
import nxmetis
import networkx as nx
from typing import List, Union, Tuple

#####################################################################

def _get_graph_partition_v1(data : TorchGraphData, partition : np.array, recursive : bool) -> TorchGraphData:
    '''
    Get sub-graph data of a data with given partition's node index.
    '''
    edge_index = data.edge_index.numpy()
    
    # Marking edges containing nodes in partition's node list
    edge_mark = np.isin(edge_index, partition)
    if recursive:
        _edge_mark = np.logical_and(edge_mark[0], edge_mark[1])
        edge_mark = np.logical_or(edge_mark[0], edge_mark[1])
    else:
        edge_mark = np.logical_and(edge_mark[0], edge_mark[1])
        _edge_mark = edge_mark
    partition_edge_id = np.argwhere(edge_mark == True).squeeze(1)
    # Gather edge index containing nodes in partition's node list
    partition_edge_index = edge_index[:,partition_edge_id]
    
    # Gather nodes which are contained in partition's edges
    partition_node_id = np.unique(np.concatenate(list(partition_edge_index) + [partition]))
    # print(partition_edge_index)
    # Lambda function to convert global node index to partition's node index
    index = lambda n : list(partition_node_id).index(n)
    v_index = np.vectorize(index)
    
    # Tranform edge_index value into partition nodes index
    if partition_edge_index.shape[1] > 0:
        partition_edge_index = torch.tensor(v_index(partition_edge_index))
    
    # Gather partition's nodes x
    partition_x = None
    if data.x is not None:
        partition_x = data.x[partition_node_id]

    # Gather partition's edge attributes
    partition_edge_attr = None
    if data.edge_attr is not None:
        partition_edge_attr = data.edge_attr[partition_edge_id]
    
    # Gather partition's node attributes
    partition_node_attr = None
    if data.node_attr is not None:
        partition_node_attr = data.node_attr[partition_node_id]

    # Gather partition's auxiliary attributes
    partition_pressure = None
    if data.pressure is not None:
        partition_pressure = data.pressure[partition_node_id]

    partition_flowrate = None
    if data.flowrate is not None:
        partition_flowrate = data.flowrate[partition_node_id]

    partition_velocity = None
    # if data.velocity is not None:
    #     partition_velocity = data.velocity[partition_node_id]

    partition_pressure_dot = None
    if data.pressure_dot is not None:
        partition_pressure_dot = data.pressure_dot[partition_node_id]

    partition_flowrate_dot = None
    if data.flowrate_dot is not None:
        partition_flowrate_dot = data.flowrate_dot[partition_node_id]

    partition_velocity_dot = None
    # if data.velocity is not None:
    #     partition_velocity_dot = data.velocity_dot[partition_node_id]
    
    partition_node_weight = None
    if data.node_weight is not None:
        partition_node_weight = data.node_weight[partition_node_id]
        partition_node_mark = np.isin(partition_node_id, partition)
        partition_node_mark = torch.tensor(partition_node_mark, dtype=torch.int)
        partition_node_weight *= partition_node_mark

    partition_edge_weight = None
    if data.edge_weight is not None:
        partition_edge_weight = data.edge_weight[partition_edge_id]
        # Mark for main and auxiliary edges
        _partition_edge_id = np.argwhere(_edge_mark == True).squeeze(1)
        partition_edge_mark = np.isin(partition_edge_id, _partition_edge_id)
        partition_edge_mark = torch.tensor(partition_edge_mark, dtype=torch.int)
        partition_edge_weight *= partition_edge_mark

    return TorchGraphData(
        x = partition_x,
        edge_index = partition_edge_index,
        edge_attr = partition_edge_attr,
        node_attr = partition_node_attr,
        pressure = partition_pressure,
        flowrate = partition_flowrate,
        velocity = partition_velocity, 
        pressure_dot = partition_pressure_dot,
        flowrate_dot = partition_flowrate_dot,
        velocity_dot = partition_velocity_dot, 
        node_weight=partition_node_weight,
        edge_weight=partition_edge_weight
    )


def _get_graph_partition_v2(data : TorchGraphData, partition : np.array, recursive : bool) -> TorchGraphData:
    '''
    Get sub-graph data of a data with given partition's node index.
    '''
    edge_index = data.edge_index.numpy()
    
    # Marking edges containing nodes in partition's node list
    edge_mark = np.isin(edge_index, partition)
    if recursive:
        # _edge_mark = np.logical_and(edge_mark[0], edge_mark[1])
        edge_mark = np.logical_or(edge_mark[0], edge_mark[1])
    else:
        edge_mark = np.logical_and(edge_mark[0], edge_mark[1])
        # _edge_mark = edge_mark
    partition_edge_id = np.argwhere(edge_mark == True).squeeze(1)
    # Gather edge index containing nodes in partition's node list
    partition_edge_index = edge_index[:,partition_edge_id]
    
    # Gather nodes which are contained in partition's edges
    partition_node_id = np.unique(np.concatenate(list(partition_edge_index) + [partition]))

    # Lambda function to convert global node index to partition's node index
    index = lambda n : list(partition_node_id).index(n)
    v_index = np.vectorize(index)
    
    # Tranform edge_index value into partition nodes index
    if partition_edge_index.shape[1] > 0:
        partition_edge_index = torch.tensor(v_index(partition_edge_index))
    
    # Gather partition's nodes x
    partition_data_dict = {}
    for key in data._store:
        if key == 'edge_index':
            partition_data_dict[key] = partition_edge_index
        elif key in ['node_attr', 'pressure', 'flowrate', 'flowrate_bc']:
            partition_data_dict[key] = data._store[key][partition_node_id]
        elif key in ['edge_attr']:
            partition_data_dict[key] = data._store[key][partition_edge_id]

    partition_node_weight = None
    # if data.node_weight is not None:
    #     partition_node_weight = data.node_weight[partition_node_id]
    #     partition_node_mark = np.isin(partition_node_id, partition)
    #     partition_node_mark = torch.tensor(partition_node_mark, dtype=torch.int)
    #     partition_node_weight *= partition_node_mark

    # partition_edge_weight = None
    # if data.edge_weight is not None:
    #     partition_edge_weight = data.edge_weight[partition_edge_id]
    #     # Mark for main and auxiliary edges
    #     _partition_edge_id = np.argwhere(_edge_mark == True).squeeze(1)
    #     partition_edge_mark = np.isin(partition_edge_id, _partition_edge_id)
    #     partition_edge_mark = torch.tensor(partition_edge_mark, dtype=torch.int)
    #     partition_edge_weight *= partition_edge_mark

    partition_data = TorchGraphData()
    for key in partition_data_dict:
        setattr(partition_data, key, partition_data_dict[key])
    if partition_node_weight is not None:
        setattr(partition_data, 'node_weight', partition_node_weight)
    return partition_data




def _get_time_partition_v1(data : TorchGraphData, time_id : np.array) -> TorchGraphData:
    '''
    Get slice of graph data with given time step slicing.
    '''
    return TorchGraphData(
        x = data.x if hasattr(data, 'x') else None,
        edge_index = data.edge_index if hasattr(data, 'edge_index') else None,
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None,
        node_attr = data.node_attr if hasattr(data, 'node_attr') else None,
        pressure = data.pressure[:,time_id] if hasattr(data, 'pressure') else None,
        flowrate = data.flowrate[:,time_id] if hasattr(data, 'flowrate') else None,
        velocity = data.velocity[:,time_id] if hasattr(data, 'velocity') else None,
        pressure_dot = data.pressure_dot[:,time_id] if hasattr(data, 'pressure_dot') else None,
        flowrate_dot = data.flowrate_dot[:,time_id] if hasattr(data, 'flowrate_dot') else None,
        velocity_dot = data.velocity_dot[:,time_id] if hasattr(data, 'velocity_dot') else None,
        node_weight = data.node_weight if hasattr(data, 'node_weight') else None,
        edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None
    )

def _get_time_partition_v2(data : TorchGraphData, time_id : np.array) -> TorchGraphData:
    '''
    Get slice of graph data with given time step slicing.
    '''
    partition_data = TorchGraphData()
    for key in data._store:
        if key in ['pressure', 'flowrate', 'velocity', 'flowrate_bc', 'pressure_dot', 'flowrate_dot']:
            setattr(partition_data, key, data._store[key][:,time_id])
        else:
            setattr(partition_data, key, data._store[key])
    return partition_data




def get_batch_graphs(data : TorchGraphData, batch_size : int = None, batch_n_times : int = None, 
                    recursive : bool = False, step : int = 1) -> List[TorchGraphData]:
    '''
    Get list of sub-graphs and slices of data with given batch size and batch time step count.
    '''
    # Spatial/graph partitioning.
    # # nx metis partitioning
    # n_batchs = int(data.number_of_nodes / batch_size)
    # (_, partitions) = nxmetis.partition(
    #     G=data.graph,
    #     nparts=n_batchs
    # )

    # BFS partitioning
    _batch_graphs = []
    if batch_size is not None:
        if batch_size <= 100:
            partitions = BFS_partition(
                edge_index=data.edge_index.numpy(),
                partition_size=batch_size,
                threshold=0.3
            )
        else:
            (_, partitions) = nxmetis.partition(
                G=data.graph,
                nparts=int(data.number_of_nodes / batch_size)
            )

        # for part in partitions:
        #     print(len(part), part)

        for partition in partitions:
            _batch_graphs.append(_get_graph_partition_v2(data, partition, recursive))
    else:
        _batch_graphs.append(data)
    
    # Temporal/time partitioning.
    batch_graphs = []
    if batch_n_times is not None:
        time_ids = []
        i = 0
        while i < data.number_of_timesteps - 1:
            i_end = i + batch_n_times + recursive
            time_ids.append(
                np.arange(start=i, 
                          stop=min(i_end, data.number_of_timesteps), 
                          step=step, 
                          dtype=int)
            )
            i = i_end - recursive
        
        for _batch_graph in _batch_graphs:
            for time_id in time_ids:
                batch_graphs.append(_get_time_partition_v2(_batch_graph, time_id))
    else:
        batch_graphs = _batch_graphs

    return batch_graphs





def BFS_partition(edge_index, partition_size=None, n_partitions=None, threshold=0.3) -> List[List[int]]:
    '''
    Perform breath first search to partition a graph(given by edge index).
    '''
    def BFS(edge_index, root, visited, part_size=100):
        queue = [root]
        visited = []
        part = []
        while queue:
            current = queue.pop(0)
            while current in visited:
                current = queue.pop(0)
            visited.append(current)
            part.append(current)
            # find child nodes
            child_edges = np.where(edge_index[0] == current)[0]
            child_nodes = list(edge_index[1][child_edges])
            # add child nodes to queue
            queue += child_nodes
            # break
            if len(part) >= part_size:
                break
        return part, queue, visited
    
    if partition_size is None:
        partition_size = int((edge_index.shape[1] + 1)/n_partitions)
    root_queue = [0]
    visited = []
    partitions = []
    root_parrent = [None]
    
    while root_queue:
        root = root_queue.pop(0)
        parrent = root_parrent.pop(0)
        partition, queue, visited = BFS(edge_index, root, visited, partition_size)
        root_queue += queue
        root_parrent += [len(partitions)] * len(queue)
        if len(partition) >= threshold*partition_size:
            partitions.append(partition)
        else:
            partitions[parrent] += partition
    return partitions





def merge_graphs(datas : List[TorchGraphData]) -> TorchGraphData:
    '''
    Merge a list of separated graph datas into single graph data.
    '''
    keys = list(datas[0]._store.keys())
    data_dict = {}
    for key in keys:
        data_dict[key] = []
    node_count = 0
    for data in datas:
        data_dict['edge_index'].append(data.edge_index + node_count)
        for key in keys:
            if key == 'edge_index':
                continue
            data_dict[key].append(data._store[key])
        node_count += data.node_attr.size(0)
    merged_data = TorchGraphData()
    for key in data_dict:
        if key == 'edge_index':
            setattr(merged_data, key, torch.cat(data_dict[key], dim=1))
        else:
            setattr(merged_data, key, torch.cat(data_dict[key], dim=0))
    return merged_data





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
