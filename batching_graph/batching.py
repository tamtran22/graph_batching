import numpy as np
from data import GraphData
# from typing import Tuple, Dict, List

def bfs_batching(data : GraphData, start_root : int, batch_size : int = 100):
    '''
    Find all sub graph of the size less or equal to batch_size.
    '''
    # Mark for all visited/explored nodes.
    mark = np.zeros((data.n_node,), dtype = int)
    # Root nodes to start finding sub graph by BFS
    root_queue = [start_root]

    def bfs_sub_graph(data : GraphData, root : int, batch_size : int):
        '''
        Find sub graph of the size less or equal to batch_size.
        '''
        # Unexplored queue to search on graph
        queue = [root]
        # Upper part of sub graph.
        sub_graph_upper = []
        while queue:
            # Explore next node stored in queue.
            curr_node = queue.pop(0)
            # Append a parrent node to upper part.
            sub_graph_upper.append(curr_node)
            # Find all neighbor nodes from current node.
            neighbor_nodes = np.unique(data.edge_index[1][np.where(data.edge_index[0] == curr_node)[0]]).tolist()
            for node in neighbor_nodes:
                if mark[node] > 0:
                    continue
                mark[node] = 1
                queue.append(node)
                if len(sub_graph_upper) + len(queue) >= batch_size:
                    return sub_graph_upper, queue
        return sub_graph_upper, queue

    while root_queue:
        # Explore next root node stored in root queue.
        root = root_queue.pop(0)
        # Find sub graph started from root and Divide sub graph by upper and ending part.
        batch_upper, batch_ending = bfs_sub_graph(data, root, batch_size)
        print(batch_upper)
        # Mark upper sub graph as explored.
        mark[batch_upper] = 1
        # Append lower sub graph into root queue.
        root_queue += batch_ending