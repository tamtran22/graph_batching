import numpy as np
from data import GraphData
from scipy import sparse
# from sklearn.cluster import SpectralClustering
import metis
import networkx as nx
import time

def clustering(data : GraphData, n_cluster : int = 10):
    G = metis.example_networkx()
    (edgecuts, parts) = metis.part_graph(G, 4)
    colors = ['red','blue','green','yellow']
    for i, p in enumerate(parts):
        G.node[i]['color'] = colors[p]
    # nx.write_dot(G, 'example.dot')
