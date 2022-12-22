import numpy as np
from data import GraphData
from scipy import sparse
from sklearn.cluster import SpectralClustering
import time

def spectral_clustering(data : GraphData, n_cluster : int = 10):
    # Adjacency matrix.
    adj_mat = sparse.dok_matrix((data.n_node, data.n_node))
    for i in range(data.n_edge):
        u = data.edge_index[0][i]
        v = data.edge_index[1][i]
        adj_mat[u,v] = 1
        adj_mat[v,u] = 1
    for i in range(data.n_node):
        adj_mat[i,i] = 1
    # adj_mat = adj_mat.todense().as
    # Clustering
    cluster_handler = SpectralClustering(
        n_clusters=n_cluster,
        affinity='precomputed',
        n_init=100
    )
    curr_time = time.time()
    cluster_handler.fit(adj_mat)
    print(f'Clustering in {time.time() - curr_time} seconds.')
    print(cluster_handler.labels_)
