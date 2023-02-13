import torch
import torch_scatter
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU, LayerNorm

from torch_geometric.nn import MessagePassing
from torch_geometric.nn import remove_self_loop, add_self_loop




#####################################################################
class objectview(object):
    def __init__(self, d) -> None:
        self.__dict__ = d


#####################################################################
class ProcessorLayer(MessagePassing):
    '''
    Graph processor takes node wise and edge wise input and return
    node wise and edge wise output with given shape
    '''
    def __init__(self, in_channels, out_channels, use_residual = True, 
                aggregate_type = 'sum', **kwargs) -> None:
        super(ProcessorLayer, self).__init__( **kwargs )
        self.use_residual = use_residual
        self.aggregate_type = aggregate_type
        self.node_mlp = Sequential(
            Linear( 2*in_channels , out_channels),
            ReLU(),
            Linear( out_channels, out_channels),
            LayerNorm( out_channels)
        )
        self.edge_mlp = Sequential(
            Linear( 3*in_channels, out_channels),
            ReLU(),
            Linear( out_channels, out_channels),
            LayerNorm( out_channels)
        )
        self.reset_parameters()
    
    def reset_parameters(self):
        '''
        reset parameters for stacked MLP layers
        '''
        self.node_mlp[0].reset_parameters()
        self.node_mlp[2].reset_parameters()

        self.edge_mlp[0].reset_parameters()
        self.edge_mlp[2].reset_parameters()
    
    def forward(self, x, edge_index, edge_attr, size = None):
        '''
        pre and post-process node features->embeddings
        message passing (propagate)
        
        x: [node_num, in_channels]
        edge_index: [2, edge_num]
        edge_attr: [edge_num, in_channels]
        '''

        out, updated_edges = self.propagate(edge_index, x = x, edge_attr = edge_attr, size = size)

        updated_nodes = torch.cat([x, out], dim=1)

        if self.use_residual:
            updated_nodes = x + self.node_mlp(updated_nodes)
        else:
            updated_nodes = self.node_mlp(updated_nodes)

        return updated_nodes, updated_edges
    
    def message(self, x_i, x_j, edge_attr):
        updated_edges = torch.cat([x_i, x_j, edge_attr], dim=1)
        if self.use_residual:
            updated_edges = edge_attr + self.edge_mlp(updated_edges)
        else:
            updated_edges = self.edge_mlp(updated_edges)
        
        return updated_edges
    
    def aggregate(self, updated_edges, edge_index, dim_size = None):
        out = torch_scatter.scatter(updated_edges, edge_index[0,:], dim=0, reduce = self.aggregate_type)

        return out, updated_edges


#####################################################################
class RecurrentProcessorCell(nn.Module):
    def __init__(self, in_channels, out_channels, num_processors, use_residual = True, 
                aggregate_type = 'sum', **kwargs) -> None:
        super(RecurrentProcessorCell, self).__init__()
        self.num_processors = num_processors

        self.processor = nn.ModuleList()
        assert (self.num_processors >= 1), 'Number of message passing layer is not >= 1'

        for _ in range(self.num_processors):
            self.processor.append(ProcessorLayer(in_channels, out_channels,
                                                use_residual, aggregate_type))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for i in range(self.num_processors):
            self.processor[i].reset_parameters()
    
    def forward(self, x, edge_index, edge_attr):
        for i in range(self.num_processors):
            x, edge_attr = self.processor[i](x, edge_index, edge_attr)
        return x, edge_attr



#####################################################################
class RecurrentMeshGraphNet(nn.Module):
    def __init__(self, input_dim_node, input_dim_edge, output_dim_node, output_dim_edge,
                hidden_dim, num_processors, use_processor_residual=True, 
                aggregate_type='sum', emb=False):

        self.node_encoder = Sequential(
            Linear(input_dim_node, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            LayerNorm(hidden_dim)
        )

        self.edge_encoder = Sequential(
             Linear(input_dim_edge, hidden_dim),
             ReLU(),
             Linear(hidden_dim, hidden_dim),
             LayerNorm(hidden_dim)
        )

        self.processor = RecurrentProcessorCell(hidden_dim, hidden_dim, num_processors,
                                                use_processor_residual, aggregate_type)
        
        self.node_decoder = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, output_dim_node)
        )

        self.edge_decoder = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, output_dim_edge)
        )

    def forward(self, x, edge_index, edge_attr):
        hidden_x = self.node_encoder(x)
        hidden_edge_attr = self.edge_encoder(edge_attr)
        hidden_x, hidden_edge_attr = self.processor(hidden_x, edge_index, hidden_edge_attr)
        return self.node_decoder(hidden_x), self.edge_decoder(hidden_edge_attr), \
                hidden_edge_attr