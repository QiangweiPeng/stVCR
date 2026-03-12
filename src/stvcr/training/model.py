import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class stVCR_DynamicModel(nn.Module):
    '''The dynamic model of neural ordinary differential equations (ODEs) in stVCR, including 
    spatial migration velocity, RNA velocity and growth rate.
    '''
    def __init__(self, in_out_gene_dim, spatial_dim, hidden_dim, n_hiddens, activation, use_gene=True, use_spatial=True):
        super().__init__()
        self.in_out_gene_dim = in_out_gene_dim
        self.spatial_dim = spatial_dim
        self.hidden_dim = hidden_dim
        self.use_gene = use_gene
        self.use_spatial = use_spatial
        if self.use_gene and self.use_spatial:
            self.spatial_velocity_net = HyperNetwork1(spatial_dim+in_out_gene_dim, spatial_dim, hidden_dim, n_hiddens, activation)  # v = dx/dt (spatial velocity)
            self.gene_velocity_net = HyperNetwork1(spatial_dim+in_out_gene_dim, in_out_gene_dim, hidden_dim, n_hiddens, activation)  # p = dq/dt (RNA velocity)
            self.growth_rate_net = HyperNetwork2(spatial_dim+in_out_gene_dim, hidden_dim, activation)  # g = dlog_w/dt (log weight velocity)
        elif self.use_gene and not self.use_spatial:
            self.gene_velocity_net = HyperNetwork1(in_out_gene_dim, in_out_gene_dim, hidden_dim, n_hiddens, activation)
            self.growth_rate_net = HyperNetwork2(in_out_gene_dim, hidden_dim, activation)
        elif not self.use_gene and self.use_spatial:
            self.spatial_velocity_net = HyperNetwork1(spatial_dim, spatial_dim, hidden_dim, n_hiddens, activation)
            self.growth_rate_net = HyperNetwork2(spatial_dim, hidden_dim, activation)

    def forward(self, t, states):
        z = states[0]
        # log_w = states[1]

        batchsize = z.shape[0]

        with torch.set_grad_enabled(True):
            z.requires_grad_(True)
            if self.use_gene and self.use_spatial:
                dx_dt = self.spatial_velocity_net(t, z)
                dq_dt = self.gene_velocity_net(t, z)
                dz_dt = torch.cat((dx_dt, dq_dt), dim=1)
            elif self.use_gene and not self.use_spatial:
                dq_dt = self.gene_velocity_net(t, z)
                dz_dt = dq_dt
            elif not self.use_gene and self.use_spatial:
                dx_dt = self.spatial_velocity_net(t, z)
                dz_dt = dx_dt

            g = self.growth_rate_net(t, z)
            dlog_w_dt = g.view(batchsize, 1)

        return (dz_dt, dlog_w_dt)
    

class HyperNetwork1(nn.Module):
    # input x, t to get v= dx/dt
    def __init__(self, in_dim, out_dim, hidden_dim, n_hiddens, activation='Tanh'):
        super().__init__()
        Layers = [in_dim + 1]
        for i in range(n_hiddens):
            Layers.append(hidden_dim)
        Layers.append(out_dim)

        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()

        self.net = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(Layers[i], Layers[i + 1]),
                self.activation,
            )
                for i in range(len(Layers) - 2)
            ]
        )
        self.out = nn.Linear(Layers[-2], Layers[-1])

    def forward(self, t, x):
        # x is N*2
        batchsize = x.shape[0]
        t = torch.tensor(t).repeat(batchsize).reshape(batchsize, 1)
        t.requires_grad = True
        state = torch.cat((t, x), dim=1)

        ii = 0
        for layer in self.net:
            if ii == 0:
                x = layer(state)
            else:
                x = layer(x)
            ii = ii + 1
        x = self.out(x)
        return x


class HyperNetwork2(nn.Module):
    # input x, t to get g
    def __init__(self, in_out_dim, hidden_dim, activation='Tanh'):
        super().__init__()
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()

        self.net = nn.Sequential(
            nn.Linear(in_out_dim + 1, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, 1))

    def forward(self, t, x):
        # x is N*2
        batchsize = x.shape[0]
        t = torch.tensor(t).repeat(batchsize).reshape(batchsize, 1)
        t.requires_grad = True
        state = torch.cat((t, x), dim=1)
        return self.net(state)
    

class RigidTransformation_2D(nn.Module):
    '''The parameterized rigid transformation in 2D space, including translation and rotation (parameterized using rotation angle).'''
    def __init__(self, num_time_point):
        super().__init__()
        translation = torch.zeros((num_time_point - 1, 2), requires_grad=True, dtype=torch.float32)
        self.translation = nn.Parameter(translation)
        rotation_angle = torch.zeros(num_time_point - 1, requires_grad=True, dtype=torch.float32)
        self.rotation_angle = nn.Parameter(rotation_angle)

    def forward(self, z, cur_time_id):
        z = z - self.translation[cur_time_id - 1, :]
        rot_matrix = self.get_rot_matrix(cur_time_id - 1)
        z = torch.matmul(rot_matrix, z.T).T
        return z

    def get_rot_matrix(self, cur_time_id):
        theta = self.rotation_angle[cur_time_id]
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        return torch.stack([cos_theta, -sin_theta, sin_theta, cos_theta]).view(2, 2)



class RigidTransformation_3D(nn.Module):
    '''The rigid transformation in 3D space, including translation and rotation (parameterized using Euler angle).'''
    def __init__(self, num_time_point):
        super().__init__()
        translation = torch.zeros((num_time_point - 1, 3), requires_grad=True, dtype=torch.float32)
        self.translation = nn.Parameter(translation)
        rotation_angle = torch.zeros((num_time_point - 1, 3), requires_grad=True, dtype=torch.float32)
        self.rotation_angle = nn.Parameter(rotation_angle)

    def forward(self, z, cur_time_id):
        z = z - self.translation[cur_time_id - 1, :]
        rot_matrix = self.get_rot_matrix(cur_time_id - 1)
        z = torch.matmul(rot_matrix, z.T).T
        return z

    def get_rot_matrix(self, cur_time_id):
        rotation_angle = self.rotation_angle[cur_time_id]
        alpha, beta, gamma = rotation_angle[0], rotation_angle[1], rotation_angle[2]
        cos_alpha = torch.cos(alpha)
        sin_alpha = torch.sin(alpha)
        cos_beta = torch.cos(beta)
        sin_beta = torch.sin(beta)
        cos_gamma = torch.cos(gamma)
        sin_gamma = torch.sin(gamma)
        # tensor_one = torch.ones(1)[0].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        # tensor_zero = torch.zeros(1)[0].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        tensor_one = torch.ones(1)[0].to(alpha.device)
        tensor_zero = torch.zeros(1)[0].to(alpha.device)
        matrix_alpha = torch.stack([tensor_one, tensor_zero, tensor_zero, tensor_zero, cos_alpha, -sin_alpha, tensor_zero, sin_alpha, cos_alpha]).view(3, 3)
        matrix_beta = torch.stack([cos_beta, tensor_zero, -sin_beta, tensor_zero, tensor_one, tensor_zero, sin_beta, tensor_zero, cos_beta]).view(3, 3)
        matrix_gamma = torch.stack([cos_gamma, -sin_gamma, tensor_zero, sin_gamma, cos_gamma, tensor_zero, tensor_zero, tensor_zero, tensor_one]).view(3, 3)
        rot_matrix = torch.matmul(torch.matmul(matrix_alpha, matrix_beta), matrix_gamma)
        return rot_matrix