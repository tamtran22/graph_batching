import numpy as np
from data import GraphData
from typing import Tuple, Dict, List

def bfs_batching(data : GraphData, root_id : int, sub_graph_size = 100) -> List[Dict]:
    
    visited = np.zeros(shape=(data.n_node,), dtype=int)
    root_queue = [root_id]
    
    # func: find nodes in subtree
    def bfs_sub_graph(root_id : int, sub_graph_size : int):
        
        queue = [root_id]
        
        sub_graph = []
        while queue:
            if len(sub_graph) > sub_graph_size:
                break
            current_node_id = queue.pop(0)
            neighbor_node_id = data.edge_index[1][np.where(data.edge_index[0] == current_node_id)[0]]
            visited[current_node_id] = 1
            for node_id in neighbor_node_id:
                if visited[node_id] == 0:
                    queue.append(node_id)
                    visited[node_id] = 2
        
        return queue, queue

    while root_queue:
        # find root of next subtree in front of queue
        current_root_id = root_queue.pop(0)
        # find node in subtree according to current root id
        node_sub_graph_visited, node_sub_graph_ending  = bfs_sub_graph(current_root_id, sub_graph_size)
        # mark visited for nodes in subtree
        visited[node_sub_graph_visited] = 1
        root_queue += node_sub_graph_ending

if __name__ == '__main__':
    n_node = 100
    node_dim = 4
    n_edge = 500
    edge_dim = 4
    x = np.random.random(size=(n_node, node_dim))
    edge_index = np.random.randint(low=0, high=n_node, size=(2,n_edge))
    edge_attr = np.random.random(size=(n_edge, edge_dim))
    
    data = GraphData(x, edge_index, edge_attr)
    
    
    bfs_batching(data, root_id = 0, sub_graph_size=20)