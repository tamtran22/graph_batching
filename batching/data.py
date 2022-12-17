import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class GraphData:
    x: np.ndarray
    edge_index: np.ndarray
    edge_attr: Optional[np.ndarray]

    @property
    def n_node(self):
        return self.x.shape[0]

    @property
    def n_edge(self):
        return self.edge_index.shape[1]

    @property
    def is_connected(self) -> bool:
        # x_index = np.arange(start=0, stop=self.n_node)
        # all_nodes_are_in_edges = np.all(np.isin(x_index, self.edge_index))
        return True

    @property
    def is_tree(self) -> bool:
        return (self.n_edge == self.n_node-1)

    def adjacency(self, u, v) -> bool:
        u_neighbor = np.where(self.edge_index[0] == u)[0]
        for _u in u_neighbor:
            if _u == v:
                return 1
        return 0

    def breadth_first_search(self, node_id, n_node_max=100):
        visited = np.zeros((self.n_node,), dtype = np.int32)
        queue = []
        current_node_id = node_id
        queue.append(node_id)
        visited[node_id] = 1

        queue_id = 0
        while len(queue) < n_node_max:
            current_node_id = queue[queue_id]
            queue_id += 1
            neighbor_node_ids = np.unique(np.concatenate([\
                self.edge_index[1][np.where(self.edge_index[0]==current_node_id)[0]],\
                self.edge_index[0][np.where(self.edge_index[1]==current_node_id)[0]]
            ]))
            for neighbor_node_id in neighbor_node_ids:
                if visited[neighbor_node_id] == 0:
                    queue.append(neighbor_node_id)
                    visited[neighbor_node_id] = 1
                    if len(queue) > n_node_max:
                        break
        return queue
        

if __name__ == '__main__':
    n_node = 56891
    node_dim = 4
    n_edge = 302315
    edge_dim = 4
    x = np.random.random(size=(n_node, node_dim))
    edge_index = np.random.randint(low=0, high=n_node, size=(2,n_edge))
    edge_attr = np.random.random(size=(n_edge, edge_dim))
    
    data = GraphData(x, edge_index, edge_attr)
    queue = data.breadth_first_search(100)
    print(np.shape(queue))
    edge_batch = np.isin(data.edge_index, queue)
    edge_batch = np.logical_and(edge_batch[0], edge_batch[1])
    edge_batch = np.where(edge_batch == True)[0]
    print(edge_batch.shape)
    t_edge_index = np.transpose(data.edge_index)
    for i in edge_batch:
        print(t_edge_index[i])