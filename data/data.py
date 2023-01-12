import networkx as nx
from torch_geometric.data import Data
from typing import Optional
from torch import Tensor

class TorchGraphData(Data):
    r"""
    Graph data class expanded from torch_geometric.data.Data()
    """
    def __init__(self, 
        x: Optional[Tensor] = None, 
        edge_index: Optional[Tensor] = None, 
        edge_attr: Optional[Tensor] = None, 
        y: Optional[Tensor] = None, 
        pos: Optional[Tensor] = None, 
        **kwargs
    ):
        super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)
        # for key in self.graph.__dict__:
        #     self.__dict__[key] = self.graph[key]
    
    @property
    def graph(self):
        edgelist = list(map(tuple, self.edge_index.numpy().transpose()))
        return nx.from_edgelist(edgelist=edgelist)

    def number_of_nodes(self):
        return self.graph.number_of_nodes()

    def number_of_edges(self):
        return self.graph.number_of_edges()

    def plot(self, **kwargs):
        nx.draw(self.graph, **kwargs)