import numpy as np
import random
from typing import Optional
import time

def get_random_batch_id(data_size : int, batch_size : int):
    if data_size <= batch_size:
        data_id = list(range(data_size))
        random.shuffle(data_id)
        batch_id = [data_id]
    else:
        data_id = list(range(data_size))
        random.shuffle(data_id)
        batch_id = []
        batch_start = 0
        while batch_start + batch_size < data_size:
            batch_end = batch_start + batch_size
            batch_id.append(data_id[batch_start:batch_end])
            batch_start = batch_end
        batch_id.append(data_id[batch_start:data_size])
    return batch_id

def batching(x, edge_index, edge_attr : Optional[np.ndarray] = None, batch_size = 100000):
    num_node = np.shape(x)[0]
    node_dim = np.shape(x)[1]
    num_edge = np.shape(edge_index)[1]
    if edge_attr is not None:
        edge_dim = np.shape(edge_attr)[1]
    batch_id_list = get_random_batch_id(num_node, batch_size)

    batch_x_list = []
    batch_edge_index_list = []
    if edge_attr is not None:
        batch_edge_attr_list = []
    for batch_id in batch_id_list:
        batch_node_id = np.array(batch_id, dtype=np.int32)

        batch_edge_id = np.isin(edge_index, batch_node_id)
        batch_edge_id = np.logical_or(batch_edge_id[0], batch_edge_id[1])
        batch_edge_id = np.where(batch_edge_id)[0]

        extend_batch_node_id = edge_index[:,batch_edge_id]
        extend_batch_node_id = np.concatenate([extend_batch_node_id[0], extend_batch_node_id[1]])
        extend_batch_node_id = np.unique(extend_batch_node_id)

        batch_x = x[extend_batch_node_id,:]
        
        get_batch_node_id = lambda node_id : np.where(extend_batch_node_id == node_id)[0][0]
        v_get_batch_node_id = np.vectorize(get_batch_node_id)
        batch_edge_index = edge_index[:,batch_edge_id]
        batch_edge_index = v_get_batch_node_id(batch_edge_index)
        
        batch_x_list.append(batch_x)
        batch_edge_index_list.append(batch_edge_index)
        if edge_attr is not None:
            batch_edge_attr = edge_attr[batch_edge_id,:]
            batch_edge_attr_list.append(batch_edge_attr)
    if edge_attr is None:
        return batch_x_list, batch_edge_index_list
    else:
        return batch_x_list, batch_edge_index_list, batch_edge_attr_list

if __name__ == '__main__':
    num_node = 567891
    node_dim = 4
    num_edge = 3002315
    edge_dim = 4

    x = np.random.random(size=(num_node, node_dim))
    edge_index = np.random.randint(low=0, high=num_node, size=(2,num_edge))
    edge_attr = np.random.random(size=(num_edge, edge_dim))

    c_time = time.time()
    batching(x, edge_index, batch_size=100)
    print(f'Done in {time.time() - c_time} seconds.')