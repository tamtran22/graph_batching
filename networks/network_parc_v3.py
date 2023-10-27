import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as gnn
from typing import Union, Callable
from torch_geometric.typing import OptTensor, Tensor
import torch_scatter


###############################################################################
# Graph processor layer
class ProcessorLayer(gnn.MessagePassing):
    def __init__(self,
        n_channels : int,
        use_edge_attr : bool = False,
        aggregation : str = 'sum',
        **kwargs
    ) -> None:
        super(ProcessorLayer, self).__init__( **kwargs )
        self.aggregation = aggregation
        
        self.edge_mlp = nn.Sequential(
            gnn.Linear(2*n_channels+use_edge_attr*n_channels, n_channels),
            nn.ReLU(),
            gnn.Linear(n_channels, n_channels),
            nn.LayerNorm(n_channels)
        )

        self.node_mlp = nn.Sequential(
            gnn.Linear(2*n_channels, n_channels),
            nn.ReLU(),
            gnn.Linear(n_channels, n_channels),
            nn.LayerNorm(n_channels)
        )
        self.reset_parameters()
    
    def reset_parameters(self):
        self.node_mlp[0].reset_parameters()
        self.node_mlp[2].reset_parameters()

        self.edge_mlp[0].reset_parameters()
        self.edge_mlp[2].reset_parameters()
    
    def forward(self, x, edge_index, edge_attr : OptTensor = None, size = None):
        out, updated_edges = self.propagate(
            edge_index, 
            x=x,
            edge_attr=edge_attr,
            size=size
        )
        updated_nodes = torch.cat([x, out], dim=1)

        updated_nodes = self.node_mlp(updated_nodes)
        return updated_nodes, updated_edges
    
    def message(self, x_i, x_j, edge_attr : OptTensor = None):
        updated_edges = torch.cat([x_i, x_j], dim=1)
        if edge_attr is not None:
            updated_edges = torch.cat([updated_edges, edge_attr], dim=1)
            updated_edges = edge_attr + self.edge_mlp(updated_edges)
        else:
            updated_edges = self.edge_mlp(updated_edges)
        return updated_edges
    
    def aggregate(self, updated_edges, edge_index, dim_size = None):
        node_dim=0
        out = torch_scatter.scatter(updated_edges, edge_index[0,:], dim=node_dim, 
                                    reduce = self.aggregation)
        return out, updated_edges
    

class ProcessorLayerV3(gnn.MessagePassing):
    def __init__(self,
        n_channels : int,
        use_edge_attr : bool = True,
        aggregation : str = 'sum',
        **kwargs
    ) -> None:
        super(ProcessorLayerV3, self).__init__( **kwargs )
        self.aggregation = aggregation
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(2*n_channels+use_edge_attr*n_channels, n_channels),
            nn.ReLU(),
            nn.LayerNorm(n_channels),
            nn.Linear(n_channels, n_channels),
            nn.ReLU(),
            nn.Linear(n_channels, n_channels),
            nn.ReLU(),
            nn.Linear(n_channels, n_channels),
            nn.ReLU(),
            nn.Linear(n_channels, n_channels),
            nn.ReLU(),
            nn.Linear(n_channels, n_channels),
            nn.ReLU(),
            nn.Linear(n_channels, n_channels),
            nn.ReLU(),
            nn.Linear(n_channels, n_channels),
            nn.ReLU(),
            nn.Linear(n_channels, n_channels),
            nn.ReLU(),
            nn.Linear(n_channels, n_channels)
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(2*n_channels, n_channels),
            nn.ReLU(),
            nn.LayerNorm(n_channels),
            nn.Linear(n_channels, n_channels),
            nn.ReLU(),
            nn.Linear(n_channels, n_channels),
            nn.ReLU(),
            nn.Linear(n_channels, n_channels),
            nn.ReLU(),
            nn.Linear(n_channels, n_channels),
            nn.ReLU(),
            nn.Linear(n_channels, n_channels),
            nn.ReLU(),
            nn.Linear(n_channels, n_channels),
            nn.ReLU(),
            nn.Linear(n_channels, n_channels)
        )
        self.reset_parameters()
    
    def reset_parameters(self):
        self.node_mlp[0].reset_parameters()
        self.node_mlp[3].reset_parameters()
        self.node_mlp[5].reset_parameters()
        self.node_mlp[7].reset_parameters()
        self.node_mlp[9].reset_parameters()
        self.node_mlp[11].reset_parameters()
        self.node_mlp[13].reset_parameters()
        self.node_mlp[15].reset_parameters()

        self.edge_mlp[0].reset_parameters()
        self.edge_mlp[3].reset_parameters()
        self.edge_mlp[5].reset_parameters()
        self.edge_mlp[7].reset_parameters()
        self.edge_mlp[9].reset_parameters()
        self.edge_mlp[11].reset_parameters()
        self.edge_mlp[13].reset_parameters()
        self.edge_mlp[15].reset_parameters()
        self.edge_mlp[17].reset_parameters()
        self.edge_mlp[19].reset_parameters()
    
    def forward(self, x, edge_index, edge_attr : OptTensor = None, size = None):
        out, updated_edges = self.propagate(
            edge_index, 
            x=x,
            edge_attr=edge_attr,
            size=size
        )
        updated_nodes = torch.cat([x, out], dim=1)

        updated_nodes = self.node_mlp(updated_nodes)
        return updated_nodes, updated_edges
    
    def message(self, x_i, x_j, edge_attr : OptTensor = None):
        updated_edges = torch.cat([x_i, x_j], dim=1)
        if edge_attr is not None:
            updated_edges = torch.cat([updated_edges, edge_attr], dim=1)
            updated_edges = updated_edges + self.edge_mlp(updated_edges)
        else:
            updated_edges = self.edge_mlp(updated_edges)
        return updated_edges
    
    def aggregate(self, updated_edges, edge_index, dim_size = None):
        node_dim=0
        out = torch_scatter.scatter(updated_edges, edge_index[0,:], dim=node_dim, 
                                    reduce = self.aggregation)
        return out, updated_edges
    

############################################################################
# Mesh graph net
class MeshGraphNet(nn.Module):
    def __init__(self,
        node_in_channels : int,
        node_out_channels : int, 
        edge_in_channels : int = 0,
        edge_out_channels : int = 0,
        hidden_channels : int = 128,
        n_layers : int = 10
    ) -> None:
        super().__init__()

        # Node encoder #######################################
        if node_in_channels <= 0:
            self.node_encoder = None
        else:
            self.node_encoder = nn.Sequential(
                gnn.Linear(node_in_channels, hidden_channels),
                nn.ReLU(),
                gnn.Linear(hidden_channels, hidden_channels),
                nn.LayerNorm(hidden_channels)
            )

        # Edge encoder #######################################
        if edge_in_channels <= 0:
            self.edge_encoder = None
        else:
            self.edge_encoder = nn.Sequential(
                gnn.Linear(edge_in_channels, hidden_channels),
                nn.ReLU(),
                gnn.Linear(hidden_channels, hidden_channels),
                nn.LayerNorm(hidden_channels)
            )

        # Node decoder #######################################
        if node_out_channels <= 0:
            self.node_decoder = None
        else:
            self.node_decoder = nn.Sequential(
                gnn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                gnn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                gnn.Linear(hidden_channels, node_out_channels)
            )
        
        # Edge decoder #######################################
        if edge_out_channels <= 0:
            self.edge_decoder = None
        else:
            self.edge_decoder = nn.Sequential(
                gnn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                gnn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                gnn.Linear(hidden_channels, edge_out_channels)
            )

        # Processor layers ###################################
        self.processor = nn.ModuleList()
        assert (n_layers >= 1)
        # First layer may input None type edge attr
        use_edge_attr = edge_in_channels > 0
        self.processor.append(ProcessorLayer(hidden_channels, use_edge_attr))
        for _ in range(n_layers - 1):
            # Second layers onward input edge attr from the previous layer.
            self.processor.append(ProcessorLayer(hidden_channels, True))
        
        self.reset_parameters()

    def reset_parameters(self):
        if self.node_encoder is not None:
            self.node_encoder[0].reset_parameters()
            self.node_encoder[2].reset_parameters()
        
        if self.edge_encoder is not None:
            self.edge_encoder[0].reset_parameters()
            self.edge_encoder[2].reset_parameters()

        if self.node_decoder is not None:
            self.node_decoder[0].reset_parameters()
            self.node_decoder[3].reset_parameters()
            self.node_decoder[6].reset_parameters()
        
        if self.edge_decoder is not None:
            self.edge_decoder[0].reset_parameters()
            self.edge_decoder[3].reset_parameters()
            self.edge_decoder[6].reset_parameters()
        
        for i in range(len(self.processor)):
            self.processor[i].reset_parameters()

    def forward(self, x, edge_index, edge_attr : OptTensor = None):
        x = self.node_encoder(x)
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
        for i in range(len(self.processor)):
            x, edge_attr = self.processor[i](x, edge_index, edge_attr)
        
        if self.edge_decoder is None:
            return self.node_decoder(x)
        else:
            return self.node_decoder(x) , self.edge_decoder(edge_attr)
        


class MeshGraphNetV3(nn.Module):
    def __init__(self,
        node_in_channels : int,
        node_out_channels : int, 
        edge_in_channels : int = 0,
        edge_out_channels : int = 0,
        hidden_channels : int = 128
    ) -> None:
        super().__init__()

        # Node encoder #######################################
        if node_in_channels <= 0:
            self.node_encoder = None
        else:
            self.node_encoder = nn.Sequential(
                gnn.Linear(node_in_channels, hidden_channels),
                nn.ReLU(),
                gnn.Linear(hidden_channels, hidden_channels),
                nn.LayerNorm(hidden_channels)
            )

        # Edge encoder #######################################
        if edge_in_channels <= 0:
            self.edge_encoder = None
        else:
            self.edge_encoder = nn.Sequential(
                gnn.Linear(edge_in_channels, hidden_channels),
                nn.ReLU(),
                gnn.Linear(hidden_channels, hidden_channels),
                nn.LayerNorm(hidden_channels)
            )

        # Node decoder #######################################
        if node_out_channels <= 0:
            self.node_decoder = None
        else:
            self.node_decoder = nn.Sequential(
                gnn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                gnn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                gnn.Linear(hidden_channels, node_out_channels)
            )
        
        # Edge decoder #######################################
        if edge_out_channels <= 0:
            self.edge_decoder = None
        else:
            self.edge_decoder = nn.Sequential(
                gnn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                gnn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                gnn.Linear(hidden_channels, edge_out_channels)
            )

        # Processor layers ###################################
        self.processor = ProcessorLayerV3(hidden_channels, use_edge_attr=True)
        
        self.reset_parameters()

    def reset_parameters(self):
        if self.node_encoder is not None:
            self.node_encoder[0].reset_parameters()
            self.node_encoder[2].reset_parameters()
        
        if self.edge_encoder is not None:
            self.edge_encoder[0].reset_parameters()
            self.edge_encoder[2].reset_parameters()

        if self.node_decoder is not None:
            self.node_decoder[0].reset_parameters()
            self.node_decoder[3].reset_parameters()
            self.node_decoder[6].reset_parameters()
        
        if self.edge_decoder is not None:
            self.edge_decoder[0].reset_parameters()
            self.edge_decoder[3].reset_parameters()
            self.edge_decoder[6].reset_parameters()
        
        self.processor.reset_parameters()

    def forward(self, x, edge_index, edge_attr : OptTensor = None):
        x = self.node_encoder(x)
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)

        x, edge_attr = self.processor(x, edge_index, edge_attr)
        
        if self.edge_decoder is None:
            return self.node_decoder(x)
        else:
            return self.node_decoder(x) , self.edge_decoder(edge_attr)



############################################################################
# Parc graph
class PARC(nn.Module):
    def __init__(self,
        n_fields,
        n_timesteps,
        n_hiddenfields,
        n_meshfields,
        n_bcfields,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.n_fields = n_fields
        self.n_timesteps = n_timesteps
        self.n_hiddenfields = n_hiddenfields
        self.n_meshfields = n_meshfields
        self.n_bcfields = n_bcfields
        
        self.derivative_solver = MeshGraphNet(
            node_in_channels=self.n_fields + self.n_hiddenfields + self.n_bcfields,
            node_out_channels=self.n_fields,
            hidden_channels=self.n_hiddenfields,
            n_layers=8
        )

        self.integral_solver = MeshGraphNet(
            node_in_channels=self.n_fields,
            node_out_channels=self.n_fields,
            hidden_channels=self.n_hiddenfields,
            n_layers=8
        )

        self.shape_descriptor = MeshGraphNet(
            node_in_channels=n_meshfields,
            node_out_channels=n_hiddenfields,
            hidden_channels=n_hiddenfields,
            n_layers=8
        )
        # self.shape_descriptor = nn.Identity()

        self.reset_parameters()
    
    def reset_parameters(self):
        self.derivative_solver.reset_parameters()
        self.integral_solver.reset_parameters()
        self.shape_descriptor.reset_parameters()

    def forward(self, 
        F_initial : Tensor, 
        mesh_features : Tensor, 
        edge_index : Tensor, 
        F_bc : OptTensor = None
    ):
        feature_map = self.shape_descriptor(mesh_features, edge_index)
        
        F_previous = F_initial

        F_dots, Fs = [], []
        for timestep in range(self.n_timesteps):
            F_temp = torch.cat([feature_map, F_previous], dim=-1)

            if self.n_bcfields > 0:
                F_temp = torch.cat([F_temp, F_bc[:,timestep + 1].unsqueeze(1)], dim=1)

            F_dot = self.derivative_solver(F_temp, edge_index)

            F_current = self.integral_solver(F_dot.detach(), edge_index)

            F_previous = F_current.detach() # detach is important

            F_dots.append(F_dot.unsqueeze(1))
            Fs.append(F_current.unsqueeze(1))
        
        F_dots = torch.cat(F_dots, dim=1)
        Fs = torch.cat(Fs, dim=1)

        return Fs, F_dots
    


##########################
class PARC_reduced(nn.Module):
    def __init__(self,
        n_fields,
        n_timesteps,
        n_hiddenfields,
        n_meshfields,
        n_bcfields,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.n_fields = n_fields
        self.n_timesteps = n_timesteps
        self.n_hiddenfields = n_hiddenfields
        self.n_meshfields = n_meshfields
        self.n_bcfields = n_bcfields
        
        self.derivative_solver = MeshGraphNetV3(
            node_in_channels=self.n_fields + self.n_hiddenfields + self.n_bcfields,
            node_out_channels=self.n_fields,
            hidden_channels=self.n_hiddenfields,
            n_layers=10
        )

        self.shape_descriptor = MeshGraphNetV3(
            node_in_channels=n_meshfields,
            node_out_channels=n_hiddenfields,
            hidden_channels=n_hiddenfields,
            n_layers=8
        )

        self.reset_parameters()
    
    def reset_parameters(self):
        self.derivative_solver.reset_parameters()
        self.shape_descriptor.reset_parameters()

    def forward(self, 
        F_initial : Tensor, 
        mesh_features : Tensor, 
        edge_index : Tensor, 
        F_bc : OptTensor = None,
        timesteps : float = None,
        rho : float = 1.12,

    ):
        feature_map = self.shape_descriptor(mesh_features, edge_index)
        
        F_previous = F_initial
        F_dot_previous = torch.zeros_like(F_initial)

        F_dots, Fs, PINN_errors = [], [], []

        # Physical components
        rho = 1.12
        pi = 3.1415926
        mu = 1.64e-5
        K = 1.
        L = mesh_features[:, 3]
        d = mesh_features[:, 4]
        Uns = (4 * rho * L) / (pi * d * d)
        Kin = (16 * K * rho) / (pi * pi * d * d * d * d)
        Vis = (128 * mu * L) / (pi * d * d * d * d)

        for timestep in range(self.n_timesteps):
            Q_previous = F_previous[:, 1]

            F_temp = torch.cat([feature_map, F_previous], dim=-1)

            if self.n_bcfields > 0:
                F_temp = torch.cat([F_temp, F_bc[:,timestep + 1].unsqueeze(1)], dim=1)

            F_dot = self.derivative_solver(F_temp, edge_index)

            # Crank Nicolson
            F_current = F_previous + timesteps * 0.5* (F_dot + F_dot_previous)

            F_previous = F_current.detach() # detach is important???
            F_dot_previous = F_dot

            F_dots.append(F_dot.unsqueeze(1))
            Fs.append(F_current.unsqueeze(1))

            P_current = F_current[:, 0]
            Q_current = F_current[:, 1]
            PINN_error = P_current + (Uns / timesteps + Vis) * Q_current + Kin * Q_previous \
                    + (Uns / timesteps) * Q_previous
            PINN_errors.append(PINN_error.unsqueeze(1))
            
        
        F_dots = torch.cat(F_dots, dim=1)
        Fs = torch.cat(Fs, dim=1)
        PINN_errors = torch.cat(PINN_errors, dim=1)

        return Fs, F_dots, PINN_errors
    

class PARC_reducedV3(nn.Module):
    def __init__(self,
        n_fields,
        n_timesteps,
        n_hiddenfields,
        n_meshfields,
        n_bcfields
    ) -> None:
        super().__init__()
        self.n_fields = n_fields
        self.n_timesteps = n_timesteps
        self.n_hiddenfields = n_hiddenfields
        self.n_meshfields = n_meshfields
        self.n_bcfields = n_bcfields
        
        self.derivative_solver = MeshGraphNet(
            node_in_channels=self.n_fields + self.n_hiddenfields + self.n_bcfields,
            node_out_channels=self.n_fields,
            hidden_channels=self.n_hiddenfields
        )

        self.shape_descriptor = MeshGraphNet(
            node_in_channels=n_meshfields,
            node_out_channels=n_hiddenfields,
            hidden_channels=n_hiddenfields
        )

        self.reset_parameters()
    
    def reset_parameters(self):
        self.derivative_solver.reset_parameters()
        self.shape_descriptor.reset_parameters()

    def forward(self, 
        F_initial : Tensor, 
        mesh_features : Tensor, 
        edge_index : Tensor, 
        F_bc : OptTensor = None,
        timesteps : float = None

    ):
        feature_map = self.shape_descriptor(mesh_features, edge_index)
        
        F_previous = F_initial
        F_dot_previous = torch.zeros_like(F_initial)

        F_dots, Fs, PINN_errors = [], [], []

        # Physical components
        L = mesh_features[:, 3]
        d = mesh_features[:, 4]
        Uns = (L) / (d * d)
        Kin = (1.) / (d * d * d * d)
        Vis = (L) / (d * d * d * d)

        for timestep in range(self.n_timesteps):
            Q_previous = F_previous[:, 1]

            F_temp = torch.cat([feature_map, F_previous], dim=-1)

            if self.n_bcfields > 0:
                F_temp = torch.cat([F_temp, F_bc[:,timestep + 1].unsqueeze(1)], dim=1)

            F_dot = self.derivative_solver(F_temp, edge_index)

            # Crank Nicolson
            F_current = F_previous + timesteps * 0.5* (F_dot + F_dot_previous)

            F_previous = F_current.detach() # detach is important???
            F_dot_previous = F_dot

            F_dots.append(F_dot.unsqueeze(1))
            Fs.append(F_current.unsqueeze(1))

            P_current = F_current[:, 0]
            Q_current = F_current[:, 1]
            PINN_error = P_current - (Uns / timesteps + Vis) * Q_current + Kin * Q_previous \
                    + (Uns/timesteps) * Q_previous
            PINN_errors.append(PINN_error.unsqueeze(1))
            
        
        F_dots = torch.cat(F_dots, dim=1)
        Fs = torch.cat(Fs, dim=1)
        PINN_errors = torch.cat(PINN_errors, dim=1)

        return Fs, F_dots, PINN_errors