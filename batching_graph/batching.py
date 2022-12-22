import numpy as np
from data import GraphData
from scipy import sparse
from sklearn.cluster import SpectralClustering
from metis import part_graph
import networkx as nx
import time

def spectral_clustering(data : GraphData, n_cluster : int = 10):
    # Adjacency matrix.
    adj_mat = sparse.dok_matrix((data.n_node, data.n_node))
    for i in range(data.n_edge):
        u = data.edge_index[0][i]
        v = data.edge_index[1][i]
        adj_mat[u,v] = 1
        adj_mat[v,u] = 1
    # for i in range(data.n_node):
    #     adj_mat[i,i] = 1
    # adj_mat = adj_mat.todense().as
    # Clustering
    # cluster_handler = SpectralClustering(
    #     n_clusters=n_cluster,
    #     affinity='precomputed',
    #     n_init=100
    # )
    graph = nx.from_scipy_sparse_matrix(adj_mat)
    curr_time = time.time()
    # cluster_handler.fit(adj_mat)
    (edge_cuts, parts) = part_graph(graph=graph, nparts=n_cluster)
    print(f'Clustering in {time.time() - curr_time} seconds.')
    batch_node = np.where(np.array(parts) == 0)[0]
    print(batch_node)
    
    edge_batch = np.isin(data.edge_index, batch_node)
    edge_batch = np.logical_and(edge_batch[0], edge_batch[1])
    edge_batch = np.where(edge_batch == True)[0]
    for i in range(edge_batch.shape[0]):
        print(data.edge_index[0, i], data.edge_index[1, i])
