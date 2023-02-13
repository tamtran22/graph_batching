import torch
import torch_scatter
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU, LayerNorm

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops









###################################################################################
###################################################################################
# Objective view class
###################################################################################
class objectview(object):
    def __init__(self, d):
        self.__dict__ = d










###################################################################################
###################################################################################
# Processor class
###################################################################################

class ProcessorLayer(MessagePassing):
    def __init__(self, in_channels, out_channels,  **kwargs):
        super(ProcessorLayer, self).__init__(  **kwargs )
        """
        in_channels: dim of node embeddings [128], out_channels: dim of edge embeddings [128]

        """

        # Note that the node and edge encoders both have the same hidden dimension
        # size. This means that the input of the edge processor will always be
        # three times the specified hidden dimension
        # (input: adjacent node embeddings and self embeddings)
        self.edge_mlp = Sequential(Linear( 3* in_channels , out_channels),
                                   ReLU(),
                                   Linear( out_channels, out_channels),
                                   LayerNorm(out_channels))

        self.node_mlp = Sequential(Linear( 2* in_channels , out_channels),
                                   ReLU(),
                                   Linear( out_channels, out_channels),
                                   LayerNorm(out_channels))


        self.reset_parameters()

    def reset_parameters(self):
        """
        reset parameters for stacked MLP layers
        """
        self.edge_mlp[0].reset_parameters()
        self.edge_mlp[2].reset_parameters()

        self.node_mlp[0].reset_parameters()
        self.node_mlp[2].reset_parameters()

    def forward(self, x, edge_index, edge_attr, size = None):
        """
        Handle the pre and post-processing of node features/embeddings,
        as well as initiates message passing by calling the propagate function.

        Note that message passing and aggregation are handled by the propagate
        function, and the update

        x has shpae [node_num , in_channels] (node embeddings)
        edge_index: [2, edge_num]
        edge_attr: [E, in_channels]

        """

        out, updated_edges = self.propagate(edge_index, x = x, edge_attr = edge_attr, size = size) # out has the shape of [E, out_channels]

        updated_nodes = torch.cat([x,out],dim=1)        # Complete the aggregation through self-aggregation

        updated_nodes = x + self.node_mlp(updated_nodes) # residual connection
        # updated_nodes = self.node_mlp(updated_nodes) # non residual connection

        return updated_nodes, updated_edges

    def message(self, x_i, x_j, edge_attr):
        """
        source_node: x_i has the shape of [E, in_channels]
        target_node: x_j has the shape of [E, in_channels]
        target_edge: edge_attr has the shape of [E, out_channels]

        The messages that are passed are the raw embeddings. These are not processed.
        """

        updated_edges=torch.cat([x_i, x_j, edge_attr], dim = 1) # tmp_emb has the shape of [E, 3 * in_channels]
        updated_edges=self.edge_mlp(updated_edges)+edge_attr

        return updated_edges

    def aggregate(self, updated_edges, edge_index, dim_size = None):
        """
        First we aggregate from neighbors (i.e., adjacent nodes) through concatenation,
        then we aggregate self message (from the edge itself). This is streamlined
        into one operation here.
        """

        # The axis along which to index number of nodes.
        node_dim = 0

        out = torch_scatter.scatter(updated_edges, edge_index[0, :], dim=node_dim, reduce = 'sum')

        return out, updated_edges










###################################################################################
###################################################################################
# MeshGraphNet class
###################################################################################

class MeshGraphNet(torch.nn.Module):
    def __init__(self, input_dim_node, input_dim_edge, hidden_dim, output_dim, \
                args, emb=False):
        super(MeshGraphNet, self).__init__()
        """
        MeshGraphNet model. This model is built upon Deepmind's 2021 paper.
        This model consists of three parts: (1) Preprocessing: encoder (2) Processor
        (3) postproccessing: decoder. Encoder has an edge and node decoders respectively.
        Processor has two processors for edge and node respectively. Note that edge attributes have to be
        updated first. Decoder is only for nodes.

        Input_dim: dynamic variables + node_type + node_position
        Hidden_dim: 128 in deepmind's paper
        Output_dim: dynamic variables: velocity changes (1)

        """

        self.num_layers = args.num_layers

        # encoder convert raw inputs into latent embeddings
        self.node_encoder = Sequential(Linear(input_dim_node , hidden_dim),
                              ReLU(),
                              Linear( hidden_dim, hidden_dim),
                              LayerNorm(hidden_dim))

        self.edge_encoder = Sequential(Linear( input_dim_edge , hidden_dim),
                              ReLU(),
                              Linear( hidden_dim, hidden_dim),
                              LayerNorm(hidden_dim)
                              )

        self.processor = nn.ModuleList()
        assert (self.num_layers >= 1), 'Number of message passing layers is not >=1'

        processor_layer=self.build_processor_model()
        for _ in range(self.num_layers):
            self.processor.append(processor_layer(hidden_dim,hidden_dim))


        # decoder: only for node embeddings
        self.node_decoder = Sequential(Linear( hidden_dim , hidden_dim),
                              ReLU(),
                              Linear( hidden_dim, output_dim)
                              )
        self.edge_decoder = Sequential(Linear( hidden_dim , hidden_dim),
                              ReLU(),
                              Linear( hidden_dim, output_dim)
                              )


    def build_processor_model(self):
        return ProcessorLayer


    # def forward(self,data,mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge):
    #     """
    #     Encoder encodes graph (node/edge features) into latent vectors (node/edge embeddings)
    #     The return of processor is fed into the processor for generating new feature vectors
    #     """
    #     x, edge_index, edge_attr, pressure = data.x, data.edge_index, data.edge_attr, data.p

    #     x = normalize(x,mean_vec_x,std_vec_x)
    #     edge_attr=normalize(edge_attr,mean_vec_edge,std_vec_edge)

    #     # Step 1: encode node/edge features into latent node/edge embeddings
    #     x = self.node_encoder(x) # output shape is the specified hidden dimension

    #     edge_attr = self.edge_encoder(edge_attr) # output shape is the specified hidden dimension

    #     # step 2: perform message passing with latent node/edge embeddings
    #     for i in range(self.num_layers):
    #         x,edge_attr = self.processor[i](x,edge_index,edge_attr)

    #     # step 3: decode latent node embeddings into physical quantities of interest

    #     return self.decoder(x)

    def forward(self, x, edge_index, edge_attr):
        """
        Encoder encodes graph (node/edge features) into latent vectors (node/edge embeddings)
        The return of processor is fed into the processor for generating new feature vectors
        """
        # Step 1: encode node/edge features into latent node/edge embeddings
        x = self.node_encoder(x) # output shape is the specified hidden dimension
        # x_prev = []
        # x_prev.append(x)

        edge_attr = self.edge_encoder(edge_attr) # output shape is the specified hidden dimension

        # step 2: perform message passing with latent node/edge embeddings
        for i in range(self.num_layers):
            x,edge_attr = self.processor[i](x,edge_index,edge_attr)
            # using dense net
            # for j in range(len(x_prev)):
            #     x = x + x_prev[j]
            # x_prev.append(x)

        # step 3: decode latent node embeddings into physical quantities of interest

        return self.node_decoder(x), self.edge_decoder(edge_attr)

    # def loss(self, pred, inputs,mean_vec_y,std_vec_y):
    #     #Define the node types that we calculate loss for
    #     normal=torch.tensor(0)
    #     outflow=torch.tensor(5)

    #     #Get the loss mask for the nodes of the types we calculate loss for
    #     loss_mask=torch.logical_or((torch.argmax(inputs.x[:,2:],dim=1)==torch.tensor(0)),
    #                                (torch.argmax(inputs.x[:,2:],dim=1)==torch.tensor(5)))

    #     #Normalize labels with dataset statistics
    #     labels = normalize(inputs.y,mean_vec_y,std_vec_y)

    #     #Find sum of square errors
    #     error=torch.sum((labels-pred)**2,axis=1)

    #     #Root and mean the errors for the nodes we calculate loss for
    #     loss=torch.sqrt(torch.mean(error[loss_mask]))
        
    #     return loss










###################################################################################
###################################################################################
# Main class
###################################################################################
        
class EmbeddedNet(torch.nn.Module):
    def __init__(self, input_dim_node, input_dim_edge, hidden_dim, output_dim, 
        num_processor_layers = 5, emb=False, add_self_loops=True):
        super(EmbeddedNet, self).__init__()
        args = objectview({'num_layers' : num_processor_layers})
        self.mesh_graph_net = MeshGraphNet(
            input_dim_node = output_dim,
            input_dim_edge = input_dim_edge + output_dim,
            hidden_dim = hidden_dim,
            output_dim = output_dim,
            args = args,
            emb = emb
        )
        self.add_self_loops = add_self_loops
    def forward(self, x, f, p, edge_index, edge_attr):
        edge_attr = torch.cat([edge_attr, f], dim=-1)
        x = p
        # x = torch.cat([p, f], dim=-1)
        if self.add_self_loops:
            edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
            edge_index, edge_attr = add_self_loops(edge_index, edge_attr)
        node_out, edge_out = self.mesh_graph_net(
            edge_index = edge_index,
            x = x,
            edge_attr = edge_attr
        )
        if self.add_self_loops:
            edge_index, edge_out = remove_self_loops(edge_index, edge_out)
        return node_out, edge_out

    # def forward(self, data):
        










###################################################################################
###################################################################################
# Main class
###################################################################################

if __name__ == '__main__':
    print('Testing model.')

    args = objectview({'num_layers' : 4})

    model = MeshGraphNet(
        input_dim_node = 10, 
        input_dim_edge = 12, 
        hidden_dim = 128, 
        output_dim = 2, 
        args = args, 
        emb=False
    )

    print(model.parameters)