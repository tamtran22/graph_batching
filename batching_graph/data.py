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

    # def bfs_node(self, node_id, n_node_max=100):
    #     visited = np.zeros((self.n_node,), dtype = np.int32)
    #     queue = np.zeros(shape=(n_node_max,), dtype=np.int32)
    #     current_node_id = node_id
    #     pop_id = 0
    #     append_id = 0

    #     queue[append_id] = current_node_id
    #     visited[current_node_id] = 1
    #     append_id += 1

    #     while append_id < n_node_max:
    #         if pop_id >= append_id:
    #             break

    #         current_node_id = queue[pop_id]
    #         pop_id += 1

    #         # neighbor_node_ids = np.unique(np.concatenate([\
    #         #     self.edge_index[1][np.where(self.edge_index[0]==current_node_id)[0]],\
    #         #     self.edge_index[0][np.where(self.edge_index[1]==current_node_id)[0]]
    #         # ]))

    #         neighbor_node_ids = np.unique(
    #             self.edge_index[1][np.where(self.edge_index[0]==current_node_id)[0]]
    #         )

    #         for neighbor_node_id in neighbor_node_ids:
    #             if visited[neighbor_node_id] == 0:

    #                 queue[append_id] = neighbor_node_id
    #                 visited[neighbor_node_id] = 1
    #                 append_id += 1

    #                 if append_id >= n_node_max:
    #                     break
    #     return queue

    # def get_batch_edge_index(self, batch_node):
    #     _batch_edge_index = np.isin(self.edge_index, batch_node)
    #     _batch_edge_index = np.logical_and(_batch_edge_index[0], _batch_edge_index[1])
    #     _batch_edge_index = np.where(_batch_edge_index == True)[0]

    #     batch_edge_index = np.transpose(np.transpose(self.edge_index)[_batch_edge_index])

    #     return batch_edge_index
        

if __name__ == '__main__':
    n_node = 56891
    node_dim = 4
    n_edge = 302315
    edge_dim = 4
    x = np.random.random(size=(n_node, node_dim))
    edge_index = np.random.randint(low=0, high=n_node, size=(2,n_edge))
    edge_attr = np.random.random(size=(n_edge, edge_dim))
    
    data = GraphData(x, edge_index, edge_attr)
    
    
    bfs_batching(data, root_id = 0)