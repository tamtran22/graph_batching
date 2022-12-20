import numpy as np
from data import GraphData
from batching import bfs_batching


if __name__ == '__main__':
    n_node = 56891
    node_dim = 4
    n_edge = 302315
    edge_dim = 4
    x = np.random.random(size=(n_node, node_dim))
    edge_index = np.random.randint(low=0, high=n_node, size=(2,n_edge))
    edge_attr = np.random.random(size=(n_edge, edge_dim))
    
    data = GraphData(x, edge_index, edge_attr)
    bfs_batching(data, start_root=0, batch_size=10)