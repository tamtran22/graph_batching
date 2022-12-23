import numpy as np
from data import GraphData
from scipy import sparse
# from sklearn.cluster import SpectralClustering
from metis import part_graph
import networkx as nx
import time

def clustering(data : GraphData, n_cluster : int = 10):
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
    (edge_cuts, parts) = part_graph(graph=graph, nparts=n_cluster, recursive=True)
    print(f'Clustering in {time.time() - curr_time} seconds.')

    a = np.partition(parts, kth=100)

    print(a)

    # batch_node = np.where(np.array(parts) == 0)[0]
    # batch_edge = np.isin(data.edge_index, batch_node)
    # batch_edge = np.logical_and(batch_edge[0], batch_edge[1])
    # batch_edge = np.where(batch_edge == True)[0]
    # batch_edge = data.edge_index[:,batch_edge]
    # batch_adj = sparse.dok_matrix((batch_node.shape[0], batch_node.shape[0]))
    # for i in range(batch_edge.shape[1]):
    #     u = list(batch_node).index(batch_edge[0][i])
    #     v = list(batch_node).index(batch_edge[1][i])
    #     batch_adj[u,v] = 1
    #     batch_adj[v,u] = 1
    # batch_graph = nx.from_scipy_sparse_matrix(batch_adj)
    # print(batch_adj)
    # nx.draw_networkx(batch_graph)
    
