import numpy as np
from data import GraphData
from scipy import sparse
from scipy.sparse.linalg import eigs

def spectral_clustering(data : GraphData, n_cluster : int = 10):
    # Adjacency matrix.
    A = sparse.dok_matrix((data.n_node, data.n_node))

    for i in range(data.n_edge):
        u = data.edge_index[0][i]
        v = data.edge_index[1][i]
        A[u,v] = -1
    # Degree of each nodes.
    D = -1 * A.sum(axis = 1)
    # Laplacian matrix
    for i in range(data.n_node):
        A[i,i] += D[i]
    A = A.asfptype()
    # Eigenvalues of Laplacian matrix
    vals, vecs = eigs(A = A, k=10000)
    print(vals)