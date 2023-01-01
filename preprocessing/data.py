import numpy as np
from dataclasses import dataclass
from typing import Optional
import networkx as nx

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
        raise NotImplementedError

    @property
    def is_tree(self) -> bool:
        return (self.n_edge == self.n_node-1)

    def adjacency(self, u, v) -> bool:
        u_neighbor = np.where(self.edge_index[0] == u)[0]
        for _u in u_neighbor:
            if _u == v:
                return 1
        return 0
    
    def render(self):
        edgelist = list(map(tuple,self.edge_index.transpose()))
        graph = nx.from_edgelist(edgelist)
        nx.draw(graph)

    @property
    def number_connected_components(self):
        edgelist = list(map(tuple,self.edge_index.transpose()))
        graph = nx.from_edgelist(edgelist)
        return nx.number_connected_components(graph)





#######################################################################
from torch_geometric.data import Data

class TorchGraphData(Data):

    @property
    def graph(self):
        edgelist = list(map(tuple, self.edge_index.numpy().transpose()))
        return nx.from_edgelist(edgelist)
    
    def render(self):
        nx.draw(self.graph)

if __name__=='__main__':
    data = TorchGraphData(
        x=np.ones(shape=(10,3)),
        edge_index=np.ones(shape=(2,15), dtype=int),
        edge_attr=np.ones(shape=(15,2))
    )
