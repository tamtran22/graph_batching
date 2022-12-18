import numpy as np
from data import GraphData
from typing import Tuple, Dict, List

def bfs_batching(data : GraphData, root_id : int) -> List[Dict]:
    
    global_visited = np.zeros((data.n_node,), dtype = np.int32)
    global_queue = np.zeros((data.n_node,), dtype = np.int32)
    current_root_id = root_id

    pop_id = 0
    append_id = 0
    
    # func: find nodes in subtree
    def bfs_subtree(root_id : int):
        return 0
    
    while 1:
        # find root of next subtree in front of queue
        current_root_id = global_queue[pop_id]
        pop_id += 1

        # find node in subtree according to current root id
        sub_tree = bfs_subtree(current_root_id)

        # mark visited for nodes in subtree
        global_visited[sub_tree] = 1