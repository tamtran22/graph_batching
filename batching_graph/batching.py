import numpy as np
from data import GraphData
from scipy import sparse
from scipy.sparse.linalg import eigs

# def bfs_batching(data : GraphData, start_root : int, batch_size : int):
#     '''
#     Find all sub graph of the size less or equal to batch_size.
#     '''
#     # Mark for all visited/explored nodes.
#     mark = np.zeros((data.n_node,), dtype = int)
#     # Root nodes to start finding sub graph by BFS
#     root_queue = [start_root]

#     def bfs_sub_graph(root : int):
#         '''
#         Find sub graph of the size less or equal to batch_size.
#         '''
#         # Unexplored queue to search on graph
#         queue = [root]
#         upper = []
#         ending = []
#         while queue:
#             # Explore next node stored in queue.
#             curr_node = queue.pop(0)
#             # Find all neighbor nodes from current node.
#             child_nodes = np.unique(data.edge_index[1][np.where(data.edge_index[0] == curr_node)[0]])
#             # Filter explored nodes.
#             child_nodes = list(filter(lambda node : mark[node] == 0, child_nodes))
#             # Add child-less nodes to ending
#             if len(child_nodes) == 0:
#                 ending.append(curr_node)
#             else:
#                 upper.append(curr_node)
#                 if curr_node in ending:
#                     ending.remove(curr_node)
#                 mark[curr_node] = 1
#                 for node in child_nodes:
#                     queue.append(node)
#                     ending.append(node)
#                     if len(upper) + len(ending) >= batch_size:
#                         print('stop')
#                         return upper, ending
#         return upper, ending

#     while root_queue:
#         # Explore next root node stored in root queue.
#         root = root_queue.pop(0)
#         # Find sub graph started from root and Divide sub graph by upper and ending part.
#         batch_upper, batch_ending = bfs_sub_graph(root)
#         # Mark upper sub graph as explored.
#         mark[batch_upper] = 1
#         # Append lower sub graph into root queue.
#         root_queue += batch_ending

#         # Testing
#         print(batch_upper, batch_ending)

def spectral_clustering(data : GraphData, n_cluster : int = 10):
    
    A = sparse.dok_matrix((data.n_node, data.n_node), dtype=int)

    for i in range(data.n_edge):
        u = data.edge_index[0][i]
        v = data.edge_index[1][i]
        A[u,v] = -1
    
    D = A.sum(axis = 1)
    
    for i in range(data.n_node):
        A[i,i] += D[i]
    
    A = A.asfptype()

    vals, vecs = eigs(A = A, k=10, )
    print(vals)