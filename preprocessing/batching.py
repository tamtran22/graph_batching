import numpy as np
from data import GraphData
from scipy import sparse
# from sklearn.cluster import SpectralClustering
import metis
import networkx as nx


from sknetwork.clustering import Louvain
from sklearn.metrics.cluster import normalized_mutual_info_score

def metis_batching(data : GraphData, relative_batch_size : int = 100, recursive : bool = False):
    n_batch = int(data.n_node / relative_batch_size)
    nodelist = list(range(data.n_node))
    edgelist = list(map(tuple,data.edge_index.transpose()))
    graph = nx.from_edgelist(edgelist=edgelist)
    
    (_, parts) = metis.part_graph(
        graph=graph,
        nparts=n_batch
    )

    batchs = []
    for i in range(n_batch):
        batch_node = np.where(np.array(parts) == i)[0]
        batch_edge = np.isin(data.edge_index, batch_node)
        if recursive:
            batch_edge = np.logical_or(batch_edge[0], batch_edge[1])
        else:
            batch_edge = np.logical_and(batch_edge[0], batch_edge[1])
        batch_edge = np.where(batch_edge == True)[0]
        batch_edge_index = data.edge_index[:, batch_edge]
        batch_node = np.where(np.isin(nodelist, batch_edge_index) == True)[0]
        for j in range(batch_edge_index.shape[1]):
            batch_edge_index[0,j] = list(batch_node).index(batch_edge_index[0,j])
            batch_edge_index[1,j] = list(batch_node).index(batch_edge_index[1,j])
        batch_x = data.x[batch_node]
        batch_edge_attr = data.edge_attr[batch_edge]
        batchs.append(GraphData(batch_x, batch_edge_index, batch_edge_attr))
    return batchs

def louvain_batching(data : GraphData, relative_batch_size : int = 100, recursive : bool = False):
    n_batch = int(data.n_node / relative_batch_size)
    nodelist = list(range(data.n_node))
    edgelist = list(map(tuple,data.edge_index.transpose()))
    graph = nx.from_edgelist(edgelist=edgelist)
    
    adj = nx.adjacency_matrix(graph)
    louvain = Louvain()
    clusters = louvain.fit_transform(adj)

    print(clusters)