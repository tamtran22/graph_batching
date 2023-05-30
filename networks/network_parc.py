import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.nn import GCNConv, GraphUNet, LayerNorm, Linear
from torch_geometric.nn.resolver import activation_resolver
from typing import Union, Callable
from torch_geometric.typing import OptTensor



class GCNBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels : int,
        out_channels : int,
        n_layers : int,
        sum_res : bool = True,
        act: Union[str, Callable] = 'relu'
    ) -> None:
        super().__init__()
        assert n_layers > 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.sum_res = sum_res
        self.act = act

        self.layers = ModuleList()
        self.layers.append(
            GCNConv(self.in_channels, self.out_channels, dtype=torch.float32)
        )
        for _ in range(self.n_layers - 1):
            self.layers.append(
                GCNConv(self.out_channels, self.out_channels, dtype=torch.float32)
            )
    
    def reset_parameters(self):
        for i in range(self.n_layers):
            self.layers[i].reset_parameters()
        
    def forward(self, x, edge_index):
        x_temp = self.layers[0](x, edge_index)
        x = x_temp
        for i in range(1, self.n_layers):
            x = self.layers[i](x, edge_index)
            x = self.act(x)
        if self.sum_res:
            x = x + x_temp
        return x



class DerivativeSolver(torch.nn.Module):
    def __init__(
        self,
        in_channels : int,
        hidden_channels : int,
        out_channels : int,
        act : Union[str, Callable] = 'relu'
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.act = act

        self.gcn_block1 = GCNBlock(
            in_channels = self.in_channels,
            out_channels = self.hidden_channels,
            n_layers = 3,
            sum_res = True,
            act = self.act
        )

        self.gcn_block2 = GCNBlock(
            in_channels = self.hidden_channels,
            out_channels = self.hidden_channels,
            n_layers = 3,
            sum_res = True,
            act = self.act
        )

        self.gcn_block3 = GCNBlock(
            in_channels = self.hidden_channels,
            out_channels = self.hidden_channels,
            n_layers = 3,
            sum_res = False,
            act = self.act
        )

        self.F_dot = GCNConv(self.hidden_channels, out_channels = self.out_channels,
                            dtype = torch.float32)
    
    def reset_parameters(self):
        super().reset_parameters()
        self.gcn_block1.reset_parameters()
        self.gcn_block2.reset_parameters()
        self.gcn_block3.reset_parameters()
        self.F_dot.reset_parameters()

    def forward(self, x, edge_index):
        x = self.gcn_block1(x, edge_index)
        x = self.act(x)
        x = self.gcn_block2(x, edge_index)
        x = self.act(x)
        x = self.gcn_block3(x, edge_index)
        x = self.act(x)
        # x = dropout(x,0.2)
        x = self.F_dot(x, edge_index)
        # x = F.tanh(x)
        return x



class IntegralSolver(torch.nn.Module):
    def __init__(
        self,
        in_channels : int,
        hidden_channels : int,
        out_channels : int,
        act : Union[str, Callable] = 'relu'
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.act = act

        self.gcn_block1 = GCNBlock(
            in_channels = self.in_channels,
            out_channels = self.hidden_channels,
            n_layers = 3,
            sum_res = True,
            act = self.act
        )

        self.gcn_block2 = GCNBlock(
            in_channels = self.hidden_channels,
            out_channels = self.hidden_channels,
            n_layers = 3,
            sum_res = True,
            act = self.act
        )

        self.gcn_block3 = GCNBlock(
            in_channels = self.hidden_channels,
            out_channels = self.hidden_channels,
            n_layers = 3,
            sum_res = False,
            act = self.act
        )

        self.F_int = GCNConv(self.hidden_channels, out_channels = self.out_channels,
                            dtype = torch.float32)
    
    def reset_parameters(self):
        super().reset_parameters()
        self.gcn_block1.reset_parameters()
        self.gcn_block2.reset_parameters()
        self.gcn_block3.reset_parameters()
        self.F_dot.reset_parameters()

    def forward(self, x, edge_index):
        x = self.gcn_block1(x, edge_index)
        x = self.act(x)
        x = self.gcn_block2(x, edge_index)
        x = self.act(x)
        # x = dropout(x, 0.2)
        x = self.gcn_block3(x, edge_index)
        x = self.act(x)
        # x = dropout(x,0.2)
        x = self.F_int(x, edge_index)
        # x = F.tanh(x)
        return x
    


class PARC(torch.nn.Module):
    def __init__(
        self,
        n_fields,
        n_timesteps,
        n_hiddenfields,
        n_meshfields,
        n_bcfields,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.n_fields = n_fields
        self.n_hiddenfields = n_hiddenfields
        self.n_timesteps = n_timesteps
        self.n_meshfields = n_meshfields
        self.n_bcfields = n_bcfields

        # self.input_fields = GCNConv(self.n_fields+self.n_bcfields, 
                                        # self.n_hiddenfields, dtype=torch.float32)
        self.input_fields = torch.nn.Sequential(
            Linear(self.n_fields + self.n_bcfields, self.n_hiddenfields),
            LayerNorm(self.n_hiddenfields),
            torch.nn.ReLU(),
            Linear(self.n_hiddenfields, self.n_hiddenfields),
            LayerNorm(self.n_hiddenfields),
            torch.nn.ReLU()
        )

        self.derivative_solver = DerivativeSolver(
            in_channels = self.n_hiddenfields + self.n_hiddenfields,
            hidden_channels = self.n_hiddenfields,
            out_channels = self.n_fields,
            act = F.relu
        )

        self.integral_solver = IntegralSolver(
            in_channels = self.n_fields,
            hidden_channels = self.n_hiddenfields,
            out_channels = self.n_fields,
            act = F.relu
        )

        self.shape_descriptor = GraphUNet(
            in_channels = self.n_meshfields,
            hidden_channels = self.n_hiddenfields,
            out_channels = n_hiddenfields,
            depth = 3,
            pool_ratios = 0.2,
            sum_res = True,
            act = F.relu
        )
    
    def reset_parameters(self):
        self.input_fields.reset_parameters()
        self.derivative_solver.reset_parameters()
        self.integral_solver.reset_parameters()
        self.shape_descriptor.reset_parameters()

    def forward(self, F_initial, mesh_features, edge_index, F_bc : OptTensor = None):
        feature_map = self.shape_descriptor(mesh_features, edge_index)

        F_dots, Fs = [], []
        F_current = F_initial
        for timestep in range(self.n_timesteps):
            if self.n_bcfields > 0:
                F_temp = torch.cat([F_current, F_bc[:,timestep + 1].unsqueeze(1)], dim=1)
            else:
                F_temp = F_current
            F_temp = self.input_fields(F_temp)
            F_temp = torch.cat([feature_map, F_temp], dim=-1)
            F_dot = self.derivative_solver(F_temp, edge_index)
            # F_dot = F.relu(F_dot)
            F_int = self.integral_solver(F_dot, edge_index)
            # F_int = F.relu(F_dot)
            F_current = F_current + F_int

            F_dots.append(F_dot.unsqueeze(1))
            Fs.append(F_current.unsqueeze(1))
        
        F_dots = torch.cat(F_dots, dim=1)
        Fs = torch.cat(Fs, dim=1)

        return Fs, F_dots


if __name__=='__main__':
    F_initial = torch.ones(size=(7,2)).float()
    mesh_features = torch.ones(size=(7,3)).float()
    edge_index = torch.tensor([[0,1,1,2,2,3],[1,2,3,4,5,6]], dtype=int)

    model = PARC(2, 21, 128, 3)

    out = model.forward(F_initial, mesh_features, edge_index)

    print(out[0].size(), out[1].size())
