import time
import numpy as np
import torch
import ot as pot
import random
import torch.nn as nn

from TorchDiffEqPack import odesolve
from torchdiffeq import odeint
from functools import partial
from tqdm import tqdm

from .data import load_data, Sampling_without_noise, Sampling_with_group, Sampling_with_group_and_neighbor
from .model import RigidTransformation_2D, RigidTransformation_3D, stVCR_DynamicModel

default_config = {'learning_rate' : 1e-3,
                  'learning_rate_rigid' : 1e-4,
                  'n_epochs' : 2000,
                  'num_samples' : 500,
                  'lambda_match' : 1,
                  'lambda_SSP' : 1e9,
                  'alpha_exp' : 0.02,
                  'alpha_gro' : 0.0002,
                  'kappa_exp' : 0.02,
                  'kappa_gro' : 0.1,
                  'spa_neigbbor': 30,
                  'exp_neigbbor': 10,}


def train_stvcr(adata, 
                model_path,
                rigid_transformation_path,
                model = None,
                cell_type_prior=None, 
                SSP_prior=None,
                cell_type_key = None,
                cell_number=None,
                config=default_config,
                gene_input_key = 'X_gene_input',
                spatial_input_key = 'X_spatial_input',
                time_input_key = 'time_input',
                use_gene = True,
                use_spatial = True,
                use_growth = True,
                use_alignment = True,
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    '''The training process with priori information
    Args:
        adata: AnnData object
        model: nn.Module, dynamic model including spatial migration rate, rna velocity and growth rate
        model_path: str, path to save the trained model 
        rigid_transformation_path: str, path to save the trained rigid transformation
        cell_type_prior: dict, priori information of cell type
        SSP_prior: dict, priori information of spatial structure
        cell_number: list of int, number of cells at each time point
        config: dict, configuration
    '''
    data_train, integral_time = load_data(adata, gene_input_key, spatial_input_key, time_input_key, device)
    gene_dim = adata.obsm[gene_input_key].shape[1]
    spatial_dim = adata.obsm[spatial_input_key].shape[1]
    time_pts = range(len(integral_time))

    model = stVCR_DynamicModel(in_out_gene_dim=gene_dim, spatial_dim=spatial_dim, hidden_dim=128, n_hiddens=6, activation='relu',
                            use_gene=use_gene, use_spatial=use_spatial).to(device)

    config.update({'model_path': model_path})
    config.update({'rigid_transformation_path': rigid_transformation_path})
    if type(config['num_samples']) is int:
        num_samples = config['num_samples']
        config['num_samples'] = [num_samples] * len(integral_time)

    if cell_number is None:
        cell_number = [cur_data.shape[0] for cur_data in data_train]

    if cell_type_prior is None and SSP_prior is None:
        if use_gene and use_spatial:
            _, rigid_transformation = train_base(data_train, integral_time, model, spatial_dim, config, cell_number,
                                             use_growth=use_growth, use_alignment=use_alignment, device=device)
        elif use_gene and not use_spatial:
            _, rigid_transformation = train_base_without_spatial(data_train, integral_time, model, spatial_dim, config, cell_number,
                                             use_growth=use_growth, use_alignment=use_alignment, device=device)
        elif not use_gene and use_spatial:
            _, rigid_transformation = train_base_without_gene(data_train, integral_time, model, spatial_dim, config, cell_number,
                                             use_growth=use_growth, use_alignment=use_alignment, device=device)                   
    elif cell_type_prior is not None and SSP_prior is None:
        cell_type = []
        for i in time_pts:
            cell_type.append(np.array(adata[adata.obs[time_input_key] == integral_time[i], :].obs[cell_type_key]))
        _, rigid_transformation = train_with_cell_type_prior(data_train, integral_time, model, spatial_dim, config, 
                                                            cell_type_prior, cell_type, cell_number,
                                                            use_gene=use_gene, use_spatial=use_spatial, use_growth=use_growth, 
                                                            use_alignment=use_alignment, device=device)
    elif cell_type_prior is not None and SSP_prior is not None:
        cell_type = []
        for i in time_pts:
            cell_type.append(np.array(adata[adata.obs[time_input_key] == integral_time[i], :].obs[cell_type_key]))
        _, rigid_transformation = train_with_cell_type_prior_and_SSP_prior(data_train, integral_time, 
                            model, spatial_dim, config, cell_type_prior, cell_type, SSP_prior, cell_number, 
                            use_gene=use_gene, use_spatial=use_spatial, use_growth=use_growth, 
                            use_alignment=use_alignment, device=device)

    adata.obsm['X_spatial_aligned'] = np.zeros_like(adata.obsm[spatial_input_key])
    for i in time_pts:
        spatial_i = torch.tensor(adata[adata.obs[time_input_key] == integral_time[i], :].obsm[spatial_input_key].copy(), 
                                    dtype=torch.float32, device=device)
        if i > 0:
            spatial_i = rigid_transformation(spatial_i, i)
        adata.obsm['X_spatial_aligned'][adata.obs[time_input_key] == integral_time[i], :] = spatial_i.detach().cpu().numpy()
    

def train_base(data_train, 
               integral_time, 
               model, 
               spatial_dim,
               config,
               cell_number = None,
               use_growth = True,
               use_alignment = True,
               device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    ''''The training process without priori information
    Args:
        observed_data: list of torch.Tensor, all observed data
        train_time: list of int, index of time points for training
        integral_time: list of float, all time points
        model: nn.Module, dynamic model including spatial migration rate, rna velocity and growth rate
        spatial_dim: int, spatial dimension, default 2
        device: torch.device, default 'cuda' if available else 'cpu'
        config: dict, configuration
    '''
    def with_wfr_loss_model(t, y, func, device, alpha_exp, alpha_gro):
        outputs = func.forward(t, y[0:2])
        dz_dt = outputs[0]
        v = dz_dt[:, 0:spatial_dim]
        p = dz_dt[:, spatial_dim:]
        g = outputs[1]
        # wfr_loss = (torch.norm(v, dim=1) ** 2 + torch.norm(p, dim=1) ** 2 * alpha_exp 
        #             + torch.norm(g, dim=1) ** 2 * alpha_gro).unsqueeze(1) * torch.exp(y[1])
        if use_growth == False:
            wfr_loss = (torch.norm(v, dim=1) ** 2 + torch.norm(p, dim=1) ** 2 * alpha_exp).unsqueeze(1) * torch.ones_like(y[1])
        else:
            wfr_loss = (torch.norm(v, dim=1) ** 2 + torch.norm(p, dim=1) ** 2 * alpha_exp 
                        + torch.norm(g, dim=1) ** 2 * alpha_gro).unsqueeze(1) * torch.exp(y[1])
        return dz_dt, g, wfr_loss
    
    num_time_point = len(integral_time)
    if spatial_dim == 2:
        rigid_transformation = RigidTransformation_2D(num_time_point).to(device)
    elif spatial_dim == 3:
        rigid_transformation = RigidTransformation_3D(num_time_point).to(device)

    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': config['learning_rate']},
        {'params': rigid_transformation.parameters(), 'lr': config['learning_rate_rigid']}
    ], weight_decay=1e-5)

    if cell_number is None:
        cell_number = [cur_data.shape[0] for cur_data in data_train]
    unit_mass = 1.0

    n_epochs = config['n_epochs']
    num_samples = config['num_samples']

    model_path = config['model_path']
    rigid_transformation_path = config['rigid_transformation_path']

    alpha_exp = config['alpha_exp']
    alpha_gro = config['alpha_gro']
    lambda_match = config['lambda_match']
    kappa_exp = config['kappa_exp']
    kappa_gro = config['kappa_gro']

    if use_growth == False:
        alpha_gro = 1
        kappa_gro = 0


    # configure training options
    options = {}
    options.update({'method': 'Dopri5'})
    options.update({'h': None})
    options.update({'rtol': 1e-3})
    options.update({'atol': 1e-5})
    options.update({'print_neval': False})
    options.update({'neval_max': 1000000})
    options.update({'safety': None})
    odeint_setp = 0.1

    for iter in tqdm(range(n_epochs)):
        optimizer.zero_grad()
        data_train_align = []
        for i in range(num_time_point):
            data_train_align.append(torch.tensor(data_train[i], device=device))

        if use_alignment:
            for i in range(1, num_time_point):
                coordinate = torch.tensor(data_train_align[i][:, 0:spatial_dim], device=device)
                coordinate = rigid_transformation(coordinate, i)
                data_train_align[i][:, 0:spatial_dim] = coordinate

        loss = torch.zeros(1).to(device)
        L2_value1 = torch.zeros(1, num_time_point).type(torch.float32).to(device)
        L2_value2 = torch.zeros(1, num_time_point).type(torch.float32).to(device)

        model_with_wfr = partial(with_wfr_loss_model, func=model, device=device, alpha_exp=alpha_exp, alpha_gro=alpha_gro)

        z0 = Sampling_without_noise(num_samples[0], 0, data_train_align, None, device)
        log_w_t0 = torch.log(torch.ones(z0.shape[0], 1).type(torch.float32) * torch.tensor(unit_mass)).to(
            device)
        wfr_loss_t0 = torch.zeros_like(log_w_t0, device=device)
        z_time, log_w_time, wfr_loss_time = \
            odeint(model_with_wfr, y0=(z0, log_w_t0, wfr_loss_t0),
                t=torch.tensor(integral_time).type(torch.float32).to(device),
                atol=1e-5, rtol=1e-5, method='dopri5', options={'step_size': odeint_setp})  # method='midpoint'

        integral_time_back = integral_time[-1::-1]
        z_end = Sampling_without_noise(num_samples[-1], num_time_point - 1, data_train_align, None, device)
        log_w_t_end = torch.log(torch.ones(z_end.shape[0], 1).type(torch.float32) * torch.tensor(unit_mass)).to(
            device)
        wfr_loss_t_end = torch.zeros_like(log_w_t_end, device=device)
        z_time_back, log_w_time_back, wfr_loss_time_back = \
            odeint(model_with_wfr, y0=(z_end, log_w_t_end, wfr_loss_t_end),
                t=torch.tensor(integral_time_back).type(torch.float32).to(device),
                atol=1e-5, rtol=1e-5, method='dopri5', options={'step_size': odeint_setp})  # method='midpoint'
        
        for i in range(num_time_point):
            z_ti = Sampling_without_noise(num_samples[i], i, data_train_align, None, device)
            w_ti = torch.ones(z_ti.shape[0]).type(torch.float32).to(device)

            if i == 0:
                z_ti_back, log_w_ti_back = z_time_back[-(i + 1)], log_w_time_back[-(i + 1)]
                w_ti_back = torch.exp(log_w_ti_back).view(-1)
                M_spa = pot.dist(z_ti_back[:, 0:spatial_dim], z_ti[:, 0:spatial_dim])
                M_exp = pot.dist(z_ti_back[:, spatial_dim:], z_ti[:, spatial_dim:])
                M = M_exp * kappa_exp + M_spa * (1 - kappa_exp)
                if use_growth:
                    ot_dist_i = pot.emd2(w_ti_back / torch.sum(w_ti_back), w_ti / torch.sum(w_ti), M)
                else:
                    ot_dist_i = pot.emd2([], [], M)
                ot_dist_i = ot_dist_i + torch.abs(torch.mean(w_ti_back)-cell_number[i]/cell_number[-1]) / (cell_number[i]/cell_number[-1]) * kappa_gro
                L2_value2[0][i] = ot_dist_i
                loss = loss + L2_value2[0][i] * lambda_match

                # print('back')
                # print('ot_dist_i:', ot_dist_i)
                # print('abs(sum(a)-sum(b))/sum(a):', torch.abs(torch.mean(w_ti_back)-cell_number[i]/cell_number[-1]) / (cell_number[i]/cell_number[-1]))
                # if iter % 10 == 0:
                #     temp_pi = pot.emd(w_ti_back / torch.sum(w_ti_back), w_ti / torch.sum(w_ti), M)
                #     print('dis_spa:', torch.sum(temp_pi * M_spa))
                #     print('dis_exp:', torch.sum(temp_pi * M_exp))

            elif i == (num_time_point - 1):
                z_ti_forward, log_w_ti_forward = z_time[i], log_w_time[i]
                w_ti_forward = torch.exp(log_w_ti_forward).view(-1)
                M_spa = pot.dist(z_ti_forward[:, 0:spatial_dim], z_ti[:, 0:spatial_dim])
                M_exp = pot.dist(z_ti_forward[:, spatial_dim:], z_ti[:, spatial_dim:])
                M = M_exp * kappa_exp + M_spa * (1 - kappa_exp)
                if use_growth:
                    ot_dist_i = pot.emd2(w_ti_forward / torch.sum(w_ti_forward), w_ti / torch.sum(w_ti), M)
                else:
                    ot_dist_i = pot.emd2([], [], M)
                ot_dist_i = ot_dist_i + torch.abs(torch.mean(w_ti_forward)-cell_number[i]/cell_number[0]) / (cell_number[i]/cell_number[0]) * kappa_gro
                L2_value1[0][i] = ot_dist_i
                loss = loss + L2_value1[0][i] * lambda_match

                # print('forward')
                # print('ot_dist_i:', ot_dist_i)
                # print('abs(sum(a)-sum(b))/sum(a):', torch.abs(torch.mean(w_ti_forward)-cell_number[i]/cell_number[0]) / (cell_number[i]/cell_number[0]))
                # if iter % 10 == 0:
                #     temp_pi = pot.emd(w_ti_forward / torch.sum(w_ti_forward), w_ti / torch.sum(w_ti), M)
                #     print('dis_spa:', torch.sum(temp_pi * M_spa))
                #     print('dis_exp:', torch.sum(temp_pi * M_exp))

            else:
                z_ti_forward, log_w_ti_forward = z_time[i], log_w_time[i]
                w_ti_forward = torch.exp(log_w_ti_forward).view(-1)
                M_spa = pot.dist(z_ti_forward[:, 0:spatial_dim], z_ti[:, 0:spatial_dim])
                M_exp = pot.dist(z_ti_forward[:, spatial_dim:], z_ti[:, spatial_dim:])
                M = M_exp * kappa_exp + M_spa * (1 - kappa_exp)
                if use_growth:
                    ot_dist_i = pot.emd2(w_ti_forward / torch.sum(w_ti_forward), w_ti / torch.sum(w_ti), M)
                else:
                    ot_dist_i = pot.emd2([], [], M)
                ot_dist_i = ot_dist_i + torch.abs(torch.mean(w_ti_forward)-cell_number[i]/cell_number[0]) / (cell_number[i]/cell_number[0]) * kappa_gro
                L2_value1[0][i] = ot_dist_i
                loss = loss + L2_value1[0][i] * lambda_match
                
                # print('forward')
                # print('ot_dist_i:', ot_dist_i)
                # print('abs(sum(a)-sum(b))/sum(a):', torch.abs(torch.mean(w_ti_forward)-cell_number[i]/cell_number[0]) / (cell_number[i]/cell_number[0]))

                z_ti_back, log_w_ti_back = z_time_back[-(i + 1)], log_w_time_back[-(i + 1)]
                w_ti_back = torch.exp(log_w_ti_back).view(-1)
                M_spa = pot.dist(z_ti_back[:, 0:spatial_dim], z_ti[:, 0:spatial_dim])
                M_exp = pot.dist(z_ti_back[:, spatial_dim:], z_ti[:, spatial_dim:])
                M = M_exp * kappa_exp + M_spa * (1 - kappa_exp)
                
                if use_growth:
                    ot_dist_i = pot.emd2(w_ti_back / torch.sum(w_ti_back), w_ti / torch.sum(w_ti), M)
                else:
                    ot_dist_i = pot.emd2([], [], M)
                ot_dist_i = ot_dist_i + torch.abs(torch.mean(w_ti_back)-cell_number[i]/cell_number[-1]) / (cell_number[i]/cell_number[-1]) * kappa_gro
                L2_value2[0][i] = ot_dist_i
                loss = loss + L2_value2[0][i] * lambda_match

                # print('back')
                # print('ot_dist_i:', ot_dist_i)
                # print('abs(sum(a)-sum(b))/sum(a):', torch.abs(torch.mean(w_ti_back)-cell_number[i]/cell_number[-1]) / (cell_number[i]/cell_number[-1]))
        
        loss = loss + (wfr_loss_time[-1].mean(0) - wfr_loss_time_back[-1].mean(0))
        loss.backward()
        optimizer.step()

        if iter % 200 == 0:
            print('----------')
            print('iter:', iter)
            print('loss:', loss)
            print('loss_wfr:', wfr_loss_time[-1].mean(0) - wfr_loss_time_back[-1].mean(0))
            # print('rigid rotation_angle', rigid_transformation.rotation_angle)
            # print('rigid translation', rigid_transformation.translation)

        if iter > 0 and iter % 100 == 0:
            for i in range(len(num_samples)):
                num_samples[i] = num_samples[i] + 20

        if iter % 100 == 0 and iter > 1:
            torch.save(model, model_path)
            torch.save(rigid_transformation, rigid_transformation_path)
    torch.save(model, model_path)
    torch.save(rigid_transformation, rigid_transformation_path)

    return model, rigid_transformation


def train_base_without_gene(data_train, 
               integral_time, 
               model, 
               spatial_dim,
               config,
               cell_number = None,
               use_growth = True,
               use_alignment = True,
               device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    ''''The training process without priori information and gene information
    Args:
        observed_data: list of torch.Tensor, all observed data
        train_time: list of int, index of time points for training
        integral_time: list of float, all time points
        model: nn.Module, dynamic model including spatial migration rate, rna velocity and growth rate
        spatial_dim: int, spatial dimension, default 2
        device: torch.device, default 'cuda' if available else 'cpu'
        config: dict, configuration
    '''
    def with_wfr_loss_model(t, y, func, device, alpha_exp, alpha_gro):
        outputs = func.forward(t, y[0:2])
        dz_dt = outputs[0]
        v = dz_dt
        g = outputs[1]
        if use_growth == False:
            wfr_loss = (torch.norm(v, dim=1) ** 2).unsqueeze(1) * torch.ones_like(y[1])
        else:
            wfr_loss = (torch.norm(v, dim=1) ** 2
                        + torch.norm(g, dim=1) ** 2 * alpha_gro).unsqueeze(1) * torch.exp(y[1])
        return dz_dt, g, wfr_loss
    
    num_time_point = len(integral_time)
    if spatial_dim == 2:
        rigid_transformation = RigidTransformation_2D(num_time_point).to(device)
    elif spatial_dim == 3:
        rigid_transformation = RigidTransformation_3D(num_time_point).to(device)

    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': config['learning_rate']},
        {'params': rigid_transformation.parameters(), 'lr': config['learning_rate_rigid']}
    ], weight_decay=1e-5)

    if cell_number is None:
        cell_number = [cur_data.shape[0] for cur_data in data_train]
    unit_mass = 1.0

    n_epochs = config['n_epochs']
    num_samples = config['num_samples']

    model_path = config['model_path']
    rigid_transformation_path = config['rigid_transformation_path']

    alpha_exp = config['alpha_exp']
    alpha_gro = config['alpha_gro']
    lambda_match = config['lambda_match']
    kappa_exp = config['kappa_exp']
    kappa_gro = config['kappa_gro']

    if use_growth == False:
        alpha_gro = 1
        kappa_gro = 0


    # configure training options
    options = {}
    options.update({'method': 'Dopri5'})
    options.update({'h': None})
    options.update({'rtol': 1e-3})
    options.update({'atol': 1e-5})
    options.update({'print_neval': False})
    options.update({'neval_max': 1000000})
    options.update({'safety': None})
    odeint_setp = 0.1

    for iter in tqdm(range(n_epochs)):
        optimizer.zero_grad()
        data_train_align = []
        for i in range(num_time_point):
            data_train_align.append(torch.tensor(data_train[i], device=device))

        if use_alignment:
            for i in range(1, num_time_point):
                coordinate = torch.tensor(data_train_align[i][:, 0:spatial_dim], device=device)
                coordinate = rigid_transformation(coordinate, i)
                data_train_align[i][:, 0:spatial_dim] = coordinate

        loss = torch.zeros(1).to(device)
        L2_value1 = torch.zeros(1, num_time_point).type(torch.float32).to(device)
        L2_value2 = torch.zeros(1, num_time_point).type(torch.float32).to(device)

        model_with_wfr = partial(with_wfr_loss_model, func=model, device=device, alpha_exp=alpha_exp, alpha_gro=alpha_gro)

        z0 = Sampling_without_noise(num_samples[0], 0, data_train_align, None, device)

        z0 = z0[:, :spatial_dim]

        log_w_t0 = torch.log(torch.ones(z0.shape[0], 1).type(torch.float32) * torch.tensor(unit_mass)).to(
            device)
        wfr_loss_t0 = torch.zeros_like(log_w_t0, device=device)
        z_time, log_w_time, wfr_loss_time = \
            odeint(model_with_wfr, y0=(z0, log_w_t0, wfr_loss_t0),
                t=torch.tensor(integral_time).type(torch.float32).to(device),
                atol=1e-5, rtol=1e-5, method='dopri5', options={'step_size': odeint_setp})  # method='midpoint'

        integral_time_back = integral_time[-1::-1]
        z_end = Sampling_without_noise(num_samples[-1], num_time_point - 1, data_train_align, None, device)

        z_end = z_end[:, :spatial_dim]

        log_w_t_end = torch.log(torch.ones(z_end.shape[0], 1).type(torch.float32) * torch.tensor(unit_mass)).to(
            device)
        wfr_loss_t_end = torch.zeros_like(log_w_t_end, device=device)
        z_time_back, log_w_time_back, wfr_loss_time_back = \
            odeint(model_with_wfr, y0=(z_end, log_w_t_end, wfr_loss_t_end),
                t=torch.tensor(integral_time_back).type(torch.float32).to(device),
                atol=1e-5, rtol=1e-5, method='dopri5', options={'step_size': odeint_setp})  # method='midpoint'
        
        for i in range(num_time_point):
            z_ti = Sampling_without_noise(num_samples[i], i, data_train_align, None, device)

            z_ti = z_ti[:, :spatial_dim]

            w_ti = torch.ones(z_ti.shape[0]).type(torch.float32).to(device)

            if i == 0:
                z_ti_back, log_w_ti_back = z_time_back[-(i + 1)], log_w_time_back[-(i + 1)]
                w_ti_back = torch.exp(log_w_ti_back).view(-1)
                M = pot.dist(z_ti_back, z_ti)
                if use_growth:
                    ot_dist_i = pot.emd2(w_ti_back / torch.sum(w_ti_back), w_ti / torch.sum(w_ti), M)
                else:
                    ot_dist_i = pot.emd2([], [], M)
                ot_dist_i = ot_dist_i + torch.abs(torch.mean(w_ti_back)-cell_number[i]/cell_number[-1]) / (cell_number[i]/cell_number[-1]) * kappa_gro
                L2_value2[0][i] = ot_dist_i
                loss = loss + L2_value2[0][i] * lambda_match

                # print('back')
                # print('ot_dist_i:', ot_dist_i)
                # print('abs(sum(a)-sum(b))/sum(a):', torch.abs(torch.mean(w_ti_back)-cell_number[i]/cell_number[-1]) / (cell_number[i]/cell_number[-1]))
                
                # temp_pi = pot.emd(w_ti_back / torch.sum(w_ti_back), w_ti / torch.sum(w_ti), M)
                # print('dis_spa:', torch.sum(temp_pi * M_spa))
                # print('dis_exp:', torch.sum(temp_pi * M_exp))

            elif i == (num_time_point - 1):
                z_ti_forward, log_w_ti_forward = z_time[i], log_w_time[i]
                w_ti_forward = torch.exp(log_w_ti_forward).view(-1)
                M = pot.dist(z_ti_forward, z_ti)
                if use_growth:
                    ot_dist_i = pot.emd2(w_ti_forward / torch.sum(w_ti_forward), w_ti / torch.sum(w_ti), M)
                else:
                    ot_dist_i = pot.emd2([], [], M)
                ot_dist_i = ot_dist_i + torch.abs(torch.mean(w_ti_forward)-cell_number[i]/cell_number[0]) / (cell_number[i]/cell_number[0]) * kappa_gro
                L2_value1[0][i] = ot_dist_i
                loss = loss + L2_value1[0][i] * lambda_match

                # print('forward')
                # print('ot_dist_i:', ot_dist_i)
                # print('abs(sum(a)-sum(b))/sum(a):', torch.abs(torch.mean(w_ti_forward)-cell_number[i]/cell_number[0]) / (cell_number[i]/cell_number[0]))
                
                # temp_pi = pot.emd(w_ti_forward / torch.sum(w_ti_forward), w_ti / torch.sum(w_ti), M)
                # print('dis_spa:', torch.sum(temp_pi * M_spa))
                # print('dis_exp:', torch.sum(temp_pi * M_exp))

            else:
                z_ti_forward, log_w_ti_forward = z_time[i], log_w_time[i]
                w_ti_forward = torch.exp(log_w_ti_forward).view(-1)
                M = pot.dist(z_ti_forward, z_ti)
                if use_growth:
                    ot_dist_i = pot.emd2(w_ti_forward / torch.sum(w_ti_forward), w_ti / torch.sum(w_ti), M)
                else:
                    ot_dist_i = pot.emd2([], [], M)
                ot_dist_i = ot_dist_i + torch.abs(torch.mean(w_ti_forward)-cell_number[i]/cell_number[0]) / (cell_number[i]/cell_number[0]) * kappa_gro
                L2_value1[0][i] = ot_dist_i
                loss = loss + L2_value1[0][i] * lambda_match
                
                # print('forward')
                # print('ot_dist_i:', ot_dist_i)
                # print('abs(sum(a)-sum(b))/sum(a):', torch.abs(torch.mean(w_ti_forward)-cell_number[i]/cell_number[0]) / (cell_number[i]/cell_number[0]))

                z_ti_back, log_w_ti_back = z_time_back[-(i + 1)], log_w_time_back[-(i + 1)]
                w_ti_back = torch.exp(log_w_ti_back).view(-1)
                M = pot.dist(z_ti_back, z_ti)
                if use_growth:
                    ot_dist_i = pot.emd2(w_ti_back / torch.sum(w_ti_back), w_ti / torch.sum(w_ti), M)
                else:
                    ot_dist_i = pot.emd2([], [], M)
                ot_dist_i = ot_dist_i + torch.abs(torch.mean(w_ti_back)-cell_number[i]/cell_number[-1]) / (cell_number[i]/cell_number[-1]) * kappa_gro
                L2_value2[0][i] = ot_dist_i
                loss = loss + L2_value2[0][i] * lambda_match

                # print('back')
                # print('ot_dist_i:', ot_dist_i)
                # print('abs(sum(a)-sum(b))/sum(a):', torch.abs(torch.mean(w_ti_back)-cell_number[i]/cell_number[-1]) / (cell_number[i]/cell_number[-1]))
        
        loss = loss + (wfr_loss_time[-1].mean(0) - wfr_loss_time_back[-1].mean(0))
        loss.backward()
        optimizer.step()

        if iter % 200 == 0:
            print('----------')
            print('iter', iter)
            print('loss:', loss)
            print('loss_wfr:', wfr_loss_time[-1].mean(0) - wfr_loss_time_back[-1].mean(0))
            # print('rigid rotation_angle', rigid_transformation.rotation_angle)
            # print('rigid translation', rigid_transformation.translation)

        if iter > 0 and iter % 100 == 0:
            for i in range(len(num_samples)):
                num_samples[i] = num_samples[i] + 20

        if iter % 100 == 0 and iter > 1:
            torch.save(model, model_path)
            torch.save(rigid_transformation, rigid_transformation_path)
    torch.save(model, model_path)
    torch.save(rigid_transformation, rigid_transformation_path)

    return model, rigid_transformation


def train_base_without_spatial(data_train, 
               integral_time, 
               model, 
               spatial_dim,
               config,
               cell_number = None,
               use_growth = True,
               use_alignment = True,
               device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    ''''The training process without priori information
    Args:
        observed_data: list of torch.Tensor, all observed data
        train_time: list of int, index of time points for training
        integral_time: list of float, all time points
        model: nn.Module, dynamic model including spatial migration rate, rna velocity and growth rate
        spatial_dim: int, spatial dimension, default 2
        device: torch.device, default 'cuda' if available else 'cpu'
        config: dict, configuration
    '''
    def with_wfr_loss_model(t, y, func, device, alpha_exp, alpha_gro):
        outputs = func.forward(t, y[0:2])
        dz_dt = outputs[0]
        # v = dz_dt[:, 0:spatial_dim]
        p = dz_dt
        g = outputs[1]
        # wfr_loss = (torch.norm(v, dim=1) ** 2 + torch.norm(p, dim=1) ** 2 * alpha_exp 
        #             + torch.norm(g, dim=1) ** 2 * alpha_gro).unsqueeze(1) * torch.exp(y[1])
        if use_growth == False:
            wfr_loss = (torch.norm(p, dim=1) ** 2 * alpha_exp).unsqueeze(1) * torch.ones_like(y[1])
        else:
            wfr_loss = (torch.norm(p, dim=1) ** 2 * alpha_exp 
                        + torch.norm(g, dim=1) ** 2 * alpha_gro).unsqueeze(1) * torch.exp(y[1])
        return dz_dt, g, wfr_loss
    
    num_time_point = len(integral_time)
    if spatial_dim == 2:
        rigid_transformation = RigidTransformation_2D(num_time_point).to(device)
    elif spatial_dim == 3:
        rigid_transformation = RigidTransformation_3D(num_time_point).to(device)

    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': config['learning_rate']},
        {'params': rigid_transformation.parameters(), 'lr': config['learning_rate_rigid']}
    ], weight_decay=1e-5)

    if cell_number is None:
        cell_number = [cur_data.shape[0] for cur_data in data_train]
    unit_mass = 1.0

    n_epochs = config['n_epochs']
    num_samples = config['num_samples']

    model_path = config['model_path']
    rigid_transformation_path = config['rigid_transformation_path']

    alpha_exp = config['alpha_exp']
    alpha_gro = config['alpha_gro']
    lambda_match = config['lambda_match']
    kappa_exp = config['kappa_exp']
    kappa_gro = config['kappa_gro']

    use_alignment = False
    if use_growth == False:
        alpha_gro = 1
        kappa_gro = 0


    # configure training options
    options = {}
    options.update({'method': 'Dopri5'})
    options.update({'h': None})
    options.update({'rtol': 1e-3})
    options.update({'atol': 1e-5})
    options.update({'print_neval': False})
    options.update({'neval_max': 1000000})
    options.update({'safety': None})
    odeint_setp = 0.1

    for iter in tqdm(range(n_epochs)):
        optimizer.zero_grad()
        data_train_align = []
        for i in range(num_time_point):
            data_train_align.append(torch.tensor(data_train[i], device=device))

        if use_alignment:
            for i in range(1, num_time_point):
                coordinate = torch.tensor(data_train_align[i][:, 0:spatial_dim], device=device)
                coordinate = rigid_transformation(coordinate, i)
                data_train_align[i][:, 0:spatial_dim] = coordinate

        loss = torch.zeros(1).to(device)
        L2_value1 = torch.zeros(1, num_time_point).type(torch.float32).to(device)
        L2_value2 = torch.zeros(1, num_time_point).type(torch.float32).to(device)

        model_with_wfr = partial(with_wfr_loss_model, func=model, device=device, alpha_exp=alpha_exp, alpha_gro=alpha_gro)

        z0 = Sampling_without_noise(num_samples[0], 0, data_train_align, None, device)

        z0 = z0[:, spatial_dim:]

        log_w_t0 = torch.log(torch.ones(z0.shape[0], 1).type(torch.float32) * torch.tensor(unit_mass)).to(
            device)
        wfr_loss_t0 = torch.zeros_like(log_w_t0, device=device)
        z_time, log_w_time, wfr_loss_time = \
            odeint(model_with_wfr, y0=(z0, log_w_t0, wfr_loss_t0),
                t=torch.tensor(integral_time).type(torch.float32).to(device),
                atol=1e-5, rtol=1e-5, method='dopri5', options={'step_size': odeint_setp})  # method='midpoint'

        integral_time_back = integral_time[-1::-1]
        z_end = Sampling_without_noise(num_samples[-1], num_time_point - 1, data_train_align, None, device)

        z_end = z_end[:, spatial_dim:]

        log_w_t_end = torch.log(torch.ones(z_end.shape[0], 1).type(torch.float32) * torch.tensor(unit_mass)).to(
            device)
        wfr_loss_t_end = torch.zeros_like(log_w_t_end, device=device)
        z_time_back, log_w_time_back, wfr_loss_time_back = \
            odeint(model_with_wfr, y0=(z_end, log_w_t_end, wfr_loss_t_end),
                t=torch.tensor(integral_time_back).type(torch.float32).to(device),
                atol=1e-5, rtol=1e-5, method='dopri5', options={'step_size': odeint_setp})  # method='midpoint'
        
        for i in range(num_time_point):
            z_ti = Sampling_without_noise(num_samples[i], i, data_train_align, None, device)

            z_ti = z_ti[:, spatial_dim:]

            w_ti = torch.ones(z_ti.shape[0]).type(torch.float32).to(device)

            if i == 0:
                z_ti_back, log_w_ti_back = z_time_back[-(i + 1)], log_w_time_back[-(i + 1)]
                w_ti_back = torch.exp(log_w_ti_back).view(-1)
                M = pot.dist(z_ti_back, z_ti)
                if use_growth:
                    ot_dist_i = pot.emd2(w_ti_back / torch.sum(w_ti_back), w_ti / torch.sum(w_ti), M)
                else:
                    ot_dist_i = pot.emd2([], [], M)
                ot_dist_i = ot_dist_i + torch.abs(torch.mean(w_ti_back)-cell_number[i]/cell_number[-1]) / (cell_number[i]/cell_number[-1]) * kappa_gro
                L2_value2[0][i] = ot_dist_i
                loss = loss + L2_value2[0][i] * lambda_match

                # print('back')
                # print('ot_dist_i:', ot_dist_i)
                # print('abs(sum(a)-sum(b))/sum(a):', torch.abs(torch.mean(w_ti_back)-cell_number[i]/cell_number[-1]) / (cell_number[i]/cell_number[-1]))
                
                # temp_pi = pot.emd(w_ti_back / torch.sum(w_ti_back), w_ti / torch.sum(w_ti), M)
                # print('dis_spa:', torch.sum(temp_pi * M_spa))
                # print('dis_exp:', torch.sum(temp_pi * M_exp))

            elif i == (num_time_point - 1):
                z_ti_forward, log_w_ti_forward = z_time[i], log_w_time[i]
                w_ti_forward = torch.exp(log_w_ti_forward).view(-1)
                M = pot.dist(z_ti_forward, z_ti)
                if use_growth:
                    ot_dist_i = pot.emd2(w_ti_forward / torch.sum(w_ti_forward), w_ti / torch.sum(w_ti), M)
                else:
                    ot_dist_i = pot.emd2([], [], M)
                ot_dist_i = ot_dist_i + torch.abs(torch.mean(w_ti_forward)-cell_number[i]/cell_number[0]) / (cell_number[i]/cell_number[0]) * kappa_gro
                L2_value1[0][i] = ot_dist_i
                loss = loss + L2_value1[0][i] * lambda_match

                # print('forward')
                # print('ot_dist_i:', ot_dist_i)
                # print('abs(sum(a)-sum(b))/sum(a):', torch.abs(torch.mean(w_ti_forward)-cell_number[i]/cell_number[0]) / (cell_number[i]/cell_number[0]))
                
                # temp_pi = pot.emd(w_ti_forward / torch.sum(w_ti_forward), w_ti / torch.sum(w_ti), M)
                # print('dis_spa:', torch.sum(temp_pi * M_spa))
                # print('dis_exp:', torch.sum(temp_pi * M_exp))

            else:
                z_ti_forward, log_w_ti_forward = z_time[i], log_w_time[i]
                w_ti_forward = torch.exp(log_w_ti_forward).view(-1)
                M = pot.dist(z_ti_forward, z_ti)
                if use_growth:
                    ot_dist_i = pot.emd2(w_ti_forward / torch.sum(w_ti_forward), w_ti / torch.sum(w_ti), M)
                else:
                    ot_dist_i = pot.emd2([], [], M)
                ot_dist_i = ot_dist_i + torch.abs(torch.mean(w_ti_forward)-cell_number[i]/cell_number[0]) / (cell_number[i]/cell_number[0]) * kappa_gro
                L2_value1[0][i] = ot_dist_i
                loss = loss + L2_value1[0][i] * lambda_match
                
                # print('forward')
                # print('ot_dist_i:', ot_dist_i)
                # print('abs(sum(a)-sum(b))/sum(a):', torch.abs(torch.mean(w_ti_forward)-cell_number[i]/cell_number[0]) / (cell_number[i]/cell_number[0]))

                z_ti_back, log_w_ti_back = z_time_back[-(i + 1)], log_w_time_back[-(i + 1)]
                w_ti_back = torch.exp(log_w_ti_back).view(-1)
                M = pot.dist(z_ti_back, z_ti)
                if use_growth:
                    ot_dist_i = pot.emd2(w_ti_back / torch.sum(w_ti_back), w_ti / torch.sum(w_ti), M)
                else:
                    ot_dist_i = pot.emd2([], [], M)
                ot_dist_i = ot_dist_i + torch.abs(torch.mean(w_ti_back)-cell_number[i]/cell_number[-1]) / (cell_number[i]/cell_number[-1]) * kappa_gro
                L2_value2[0][i] = ot_dist_i
                loss = loss + L2_value2[0][i] * lambda_match

                # print('back')
                # print('ot_dist_i:', ot_dist_i)
                # print('abs(sum(a)-sum(b))/sum(a):', torch.abs(torch.mean(w_ti_back)-cell_number[i]/cell_number[-1]) / (cell_number[i]/cell_number[-1]))
        
        loss = loss + (wfr_loss_time[-1].mean(0) - wfr_loss_time_back[-1].mean(0))
        loss.backward()
        optimizer.step()

        if iter % 200 == 0:
            print('----------')
            print('iter', iter)
            print('loss:', loss)
            print('loss_wfr:', wfr_loss_time[-1].mean(0) - wfr_loss_time_back[-1].mean(0))
            # print('rigid rotation_angle', rigid_transformation.rotation_angle)
            # print('rigid translation', rigid_transformation.translation)

        if iter > 0 and iter % 100 == 0:
            for i in range(len(num_samples)):
                num_samples[i] = num_samples[i] + 20

        if iter % 100 == 0 and iter > 1:
            torch.save(model, model_path)
            torch.save(rigid_transformation, rigid_transformation_path)
    torch.save(model, model_path)
    torch.save(rigid_transformation, rigid_transformation_path)

    return model, rigid_transformation


def train_with_cell_type_prior(data_train, 
                            integral_time, 
                            model, 
                            spatial_dim,
                            config,
                            cell_type_prior,
                            cell_type,
                            cell_number = None,
                            use_gene = True,
                            use_spatial = True,
                            use_growth = True,
                            use_alignment = True,
                            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    ''''The training process with priori information about known cell type transition
    Args:
        observed_data: list of torch.Tensor, all observed data
        train_time: list of int, index of time points for training
        integral_time: list of float, all time points
        model: nn.Module, dynamic model including spatial migration rate, rna velocity and growth rate
        spatial_dim: int, spatial dimension, default 2
        device: torch.device, default 'cuda' if available else 'cpu'
        config: dict, configuration
    '''
    def with_wfr_loss_model(t, y, func, device, alpha_exp, alpha_gro):
        outputs = func.forward(t, y[0:2])
        dz_dt = outputs[0]
        v = dz_dt[:, 0:spatial_dim]
        p = dz_dt[:, spatial_dim:]
        g = outputs[1]
        wfr_loss = (torch.norm(v, dim=1) ** 2 + torch.norm(p, dim=1) ** 2 * alpha_exp 
                    + torch.norm(g, dim=1) ** 2 * alpha_gro).unsqueeze(1) * torch.exp(y[1])
        return dz_dt, g, wfr_loss
    
    num_time_point = len(integral_time)

    if cell_number is None:
        cell_number = [cur_data.shape[0] for cur_data in data_train]
    
    cell_group = [np.zeros_like(cur_cell_type) for cur_cell_type in cell_type]
    unique_group = []
    cell_group_number = []
    for i in range(len(cell_type_prior)):
        cur_bio_guid = cell_type_prior[i]
        cur_group_number = []
        for j in range(num_time_point):
            indices = [index for index, value in enumerate(cell_type[j]) if value in cur_bio_guid[j]]
            cell_group[j][indices] = 'group' + str(i)
            # cur_group_number.append(len(indices))
            cur_group_number.append(len(indices)*cell_number[j]/len(cell_type[j]))
        cell_group_number.append(cur_group_number)
        unique_group.append('group' + str(i))

        # 如果 cell_group 中不再包含任何 0，说明全部分配完毕
        is_all_assigned = all(np.all(cg != 0) for cg in cell_group)

        if not is_all_assigned:
            cur_group_number = []
            for j in range(num_time_point):
                indices = [index for index, value in enumerate(cell_group[j]) if value is 0]
                cell_group[j][indices] = 'other'
                # cur_group_number.append(len(indices))
                cur_group_number.append(len(indices)*cell_number[j]/len(cell_type[j]))
            cell_group_number.append(cur_group_number)
            unique_group.append('other')

    if spatial_dim == 2:
        rigid_transformation = RigidTransformation_2D(num_time_point).to(device)
    elif spatial_dim == 3:
        rigid_transformation = RigidTransformation_3D(num_time_point).to(device)

    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': config['learning_rate']},
        {'params': rigid_transformation.parameters(), 'lr': config['learning_rate_rigid']}
    ], weight_decay=1e-5)

    unit_mass = 1.0

    n_epochs = config['n_epochs']
    num_samples = config['num_samples']

    model_path = config['model_path']
    rigid_transformation_path = config['rigid_transformation_path']

    alpha_exp = config['alpha_exp']
    alpha_gro = config['alpha_gro']
    lambda_match = config['lambda_match']
    kappa_exp = config['kappa_exp']
    kappa_gro = config['kappa_gro']

    if use_gene == False:
        alpha_exp = 1
        kappa_exp = 0
    if use_spatial == False:
        kappa_exp = 1
        use_alignment = False
    if use_growth == False:
        alpha_gro = 1
        kappa_gro = 0

    # configure training options
    options = {}
    options.update({'method': 'Dopri5'})
    options.update({'h': None})
    options.update({'rtol': 1e-3})
    options.update({'atol': 1e-5})
    options.update({'print_neval': False})
    options.update({'neval_max': 1000000})
    options.update({'safety': None})
    odeint_setp = 0.1

    for iter in tqdm(range(n_epochs)):
        optimizer.zero_grad()
        data_train_align = []
        for i in range(num_time_point):
            data_train_align.append(torch.tensor(data_train[i], device=device))

        if use_alignment:
            for i in range(1, num_time_point):
                coordinate = torch.tensor(data_train_align[i][:, 0:spatial_dim], device=device)
                coordinate = rigid_transformation(coordinate, i)
                data_train_align[i][:, 0:spatial_dim] = coordinate

        loss = torch.zeros(1).to(device)
        L2_value1 = torch.zeros(1, num_time_point, len(unique_group)).type(torch.float32).to(device)
        L2_value2 = torch.zeros(1, num_time_point, len(unique_group)).type(torch.float32).to(device)

        model_with_wfr = partial(with_wfr_loss_model, func=model, device=device, alpha_exp=alpha_exp, alpha_gro=alpha_gro)

        z0, t0_group = Sampling_with_group(num_samples[0], 0, data_train_align, cell_group, None, device)
        log_w_t0 = torch.log(torch.ones(z0.shape[0], 1).type(torch.float32) * torch.tensor(unit_mass)).to(
            device)
        wfr_loss_t0 = torch.zeros_like(log_w_t0, device=device)
        z_time, log_w_time, wfr_loss_time = \
            odeint(model_with_wfr, y0=(z0, log_w_t0, wfr_loss_t0),
                t=torch.tensor(integral_time).type(torch.float32).to(device),
                atol=1e-5, rtol=1e-5, method='dopri5', options={'step_size': odeint_setp})  # method='midpoint'

        integral_time_back = integral_time[-1::-1]
        z_end, t_end_group = Sampling_with_group(num_samples[-1], num_time_point - 1, 
                                                 data_train_align, cell_group, None, device)
        log_w_t_end = torch.log(torch.ones(z_end.shape[0], 1).type(torch.float32) * torch.tensor(unit_mass)).to(device)
        wfr_loss_t_end = torch.zeros_like(log_w_t_end, device=device)
        z_time_back, log_w_time_back, wfr_loss_time_back = \
            odeint(model_with_wfr, y0=(z_end, log_w_t_end, wfr_loss_t_end),
                t=torch.tensor(integral_time_back).type(torch.float32).to(device),
                atol=1e-5, rtol=1e-5, method='dopri5', options={'step_size': odeint_setp})  # method='midpoint'
        
        for i in range(num_time_point):
            
            z_ti, ti_group = Sampling_with_group(num_samples[i], i, data_train_align, cell_group, None, device)
            w_ti = torch.ones(z_ti.shape[0]).type(torch.float32).to(device)

            if i == 0:
                z_ti_back, log_w_ti_back = z_time_back[-(i + 1)], log_w_time_back[-(i + 1)]
                w_ti_back = torch.exp(log_w_ti_back).view(-1)

                for k in range(len(unique_group)):
                    index_back = np.where(np.array(t_end_group) == unique_group[k])[0]
                    index_ti = np.where(np.array(ti_group) == unique_group[k])[0]

                    M_spa = pot.dist(z_ti_back[index_back, 0:spatial_dim], z_ti[index_ti, 0:spatial_dim])
                    M_exp = pot.dist(z_ti_back[index_back, spatial_dim:], z_ti[index_ti, spatial_dim:])
                    M = M_exp * kappa_exp + M_spa * (1 - kappa_exp)

                    if use_growth:
                        ot_dist_ik = pot.emd2(w_ti_back[index_back] / torch.sum(w_ti_back[index_back]),
                                            w_ti[index_ti] / torch.sum(w_ti[index_ti]), M)
                        # TODO The weights may differ between groups
                        # ot_dist_ik = (cell_group_number[k][i]/cell_number[i])*pot.emd2(w_ti_back[index_back] / torch.sum(w_ti_back[index_back]),
                        #                     w_ti[index_ti] / torch.sum(w_ti[index_ti]), M)
                    else:
                        ot_dist_ik = pot.emd2([], [], M)

                    ot_dist_ik = ot_dist_ik + \
                        torch.abs(torch.mean(w_ti_back[index_back]) - cell_group_number[k][i] / cell_group_number[k][-1]) / (cell_group_number[k][i] / cell_group_number[k][-1]) * kappa_gro
                    
                    L2_value2[0][i][k] = ot_dist_ik
                    loss = loss + L2_value2[0][i][k] * lambda_match

                    # print('back')
                    # print('ot_dist_ik:', ot_dist_ik)
                    # print('abs(sum(a)-sum(b))/sum(a):', torch.abs(torch.mean(w_ti_back[index_back]) - cell_group_number[k][i] / cell_group_number[k][-1]) / (cell_group_number[k][i] / cell_group_number[k][-1]))
                    # temp_pi = pot.emd(w_ti_back[index_back] / torch.sum(w_ti_back[index_back]),
                    #                         w_ti[index_ti] / torch.sum(w_ti[index_ti]), M)
                    # print('dis_spa:', torch.sum(temp_pi * M_spa))
                    # print('dis_exp:', torch.sum(temp_pi * M_exp))

            elif i == (num_time_point - 1):
                z_ti_forward, log_w_ti_forward = z_time[i], log_w_time[i]
                w_ti_forward = torch.exp(log_w_ti_forward).view(-1)

                for k in range(len(unique_group)):
                    index_forward = np.where(np.array(t0_group) == unique_group[k])[0]
                    index_ti = np.where(np.array(ti_group) == unique_group[k])[0]

                    M_spa = pot.dist(z_ti_forward[index_forward, 0:spatial_dim], z_ti[index_ti, 0:spatial_dim])
                    M_exp = pot.dist(z_ti_forward[index_forward, spatial_dim:], z_ti[index_ti, spatial_dim:])
                    M = M_exp * kappa_exp + M_spa * (1 - kappa_exp)

                    if use_growth:
                        ot_dist_ik = pot.emd2(w_ti_forward[index_forward]/ torch.sum(w_ti_forward[index_forward]),
                                            w_ti[index_ti] / torch.sum(w_ti[index_ti]), M)
                        # ot_dist_ik = (cell_group_number[k][i]/cell_number[i]) * pot.emd2(w_ti_forward[index_forward]/ torch.sum(w_ti_forward[index_forward]),
                        #                     w_ti[index_ti] / torch.sum(w_ti[index_ti]), M)
                    else:
                        ot_dist_ik = pot.emd2([], [], M)
                    
                    ot_dist_ik = ot_dist_ik + \
                        torch.abs(torch.mean(w_ti_forward[index_forward]) - cell_group_number[k][i] / cell_group_number[k][0]) / (cell_group_number[k][i] / cell_group_number[k][0]) * kappa_gro

                    L2_value1[0][i][k] = ot_dist_ik
                    loss = loss + L2_value1[0][i][k] * lambda_match

                    # print('forward')
                    # print('ot_dist_ik:', ot_dist_ik)
                    # print('abs(sum(a)-sum(b))/sum(a):', torch.abs(torch.mean(w_ti_forward[index_forward]) - cell_group_number[k][i] / cell_group_number[k][0]) / (cell_group_number[k][i] / cell_group_number[k][0]))
                    # temp_pi = pot.emd(w_ti_forward[index_forward]/ torch.sum(w_ti_forward[index_forward]),
                    #                         w_ti[index_ti] / torch.sum(w_ti[index_ti]), M)
                    # print('dis_spa:', torch.sum(temp_pi * M_spa))
                    # print('dis_exp:', torch.sum(temp_pi * M_exp))

            else:
                z_ti_forward, log_w_ti_forward = z_time[i], log_w_time[i]
                w_ti_forward = torch.exp(log_w_ti_forward).view(-1)

                for k in range(len(unique_group)):
                    index_forward = np.where(np.array(t0_group) == unique_group[k])[0]
                    index_ti = np.where(np.array(ti_group) == unique_group[k])[0]

                    M_spa = pot.dist(z_ti_forward[index_forward, 0:spatial_dim], z_ti[index_ti, 0:spatial_dim])
                    M_exp = pot.dist(z_ti_forward[index_forward, spatial_dim:], z_ti[index_ti, spatial_dim:])
                    M = M_exp * kappa_exp + M_spa * (1 - kappa_exp)

                    if use_growth:
                        ot_dist_ik = pot.emd2(w_ti_forward[index_forward]/ torch.sum(w_ti_forward[index_forward]),
                                            w_ti[index_ti] / torch.sum(w_ti[index_ti]), M)
                        # ot_dist_ik = (cell_group_number[k][i]/cell_number[i]) * pot.emd2(w_ti_forward[index_forward]/ torch.sum(w_ti_forward[index_forward]),
                        #                     w_ti[index_ti] / torch.sum(w_ti[index_ti]), M)
                    else:
                        ot_dist_ik = pot.emd2([], [], M)
                    
                    ot_dist_ik = ot_dist_ik + \
                        torch.abs(torch.mean(w_ti_forward[index_forward]) - cell_group_number[k][i] / cell_group_number[k][0]) / (cell_group_number[k][i] / cell_group_number[k][0]) * kappa_gro

                    L2_value1[0][i][k] = ot_dist_ik
                    loss = loss + L2_value1[0][i][k] * lambda_match

                    # print('forward')
                    # print('ot_dist_ik:', ot_dist_ik)
                    # print('abs(sum(a)-sum(b))/sum(a):', torch.abs(torch.mean(w_ti_forward[index_forward]) - cell_group_number[k][i] / cell_group_number[k][0]) / (cell_group_number[k][i] / cell_group_number[k][0]))
                
                z_ti_back, log_w_ti_back = z_time_back[-(i + 1)], log_w_time_back[-(i + 1)]
                w_ti_back = torch.exp(log_w_ti_back).view(-1)

                for k in range(len(unique_group)):
                    index_back = np.where(np.array(t_end_group) == unique_group[k])[0]
                    index_ti = np.where(np.array(ti_group) == unique_group[k])[0]

                    M_spa = pot.dist(z_ti_back[index_back, 0:spatial_dim], z_ti[index_ti, 0:spatial_dim])
                    M_exp = pot.dist(z_ti_back[index_back, spatial_dim:], z_ti[index_ti, spatial_dim:])
                    M = M_exp * kappa_exp + M_spa * (1 - kappa_exp)

                    if use_growth:
                        ot_dist_ik = pot.emd2(w_ti_back[index_back] / torch.sum(w_ti_back[index_back]),
                                            w_ti[index_ti] / torch.sum(w_ti[index_ti]), M)
                        # ot_dist_ik = (cell_group_number[k][i]/cell_number[i]) * pot.emd2(w_ti_back[index_back] / torch.sum(w_ti_back[index_back]),
                        #                     w_ti[index_ti] / torch.sum(w_ti[index_ti]), M)
                    else:
                        ot_dist_ik = pot.emd2([], [], M)

                    ot_dist_ik = ot_dist_ik + \
                        torch.abs(torch.mean(w_ti_back[index_back]) - cell_group_number[k][i] / cell_group_number[k][-1]) / (cell_group_number[k][i] / cell_group_number[k][-1]) * kappa_gro
                    
                    L2_value2[0][i][k] = ot_dist_ik
                    loss = loss + L2_value2[0][i][k] * lambda_match

                    # print('back')
                    # print('ot_dist_ik:', ot_dist_ik)
                    # print('abs(sum(a)-sum(b))/sum(a):', torch.abs(torch.mean(w_ti_back[index_back]) - cell_group_number[k][i] / cell_group_number[k][-1]) / (cell_group_number[k][i] / cell_group_number[k][-1]))
        
        loss = loss + (wfr_loss_time[-1].mean(0) - wfr_loss_time_back[-1].mean(0))
        loss.backward()
        optimizer.step()

        if iter % 200 == 0:
            print('----------')
            print('iter', iter)
            print('loss:', loss)
            print('loss_wfr:', wfr_loss_time[-1].mean(0) - wfr_loss_time_back[-1].mean(0))
            # print('rigid rotation_angle', rigid_transformation.rotation_angle)
            # print('rigid translation', rigid_transformation.translation)

        if iter > 0 and iter % 100 == 0:
            for i in range(len(num_samples)):
                num_samples[i] = num_samples[i] + 20

        if iter % 100 == 0 and iter > 1:
            torch.save(model, model_path)
            torch.save(rigid_transformation, rigid_transformation_path)

    torch.save(model, model_path)
    torch.save(rigid_transformation, rigid_transformation_path)

    return model, rigid_transformation


def train_with_cell_type_prior_and_SSP_prior(data_train, 
                            integral_time, 
                            model, 
                            spatial_dim,
                            config,
                            cell_type_prior,
                            cell_type,
                            SSP_prior,
                            cell_number = None,
                            use_gene = True,
                            use_spatial = True,
                            use_growth = True,
                            use_alignment = True,
                            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    ''''The training process with priori information about known cell type transition and spatial structure preservation
    Args:
        observed_data: list of torch.Tensor, all observed data
        train_time: list of int, index of time points for training
        integral_time: list of float, all time points
        model: nn.Module, dynamic model including spatial migration rate, rna velocity and growth rate
        spatial_dim: int, spatial dimension, default 2
        device: torch.device, default 'cuda' if available else 'cpu'
        config: dict, configuration
    '''
    
    def with_wfr_and_ssp_loss_model(t, y, func, neighbor_index_new, device, alpha_exp, alpha_gro):
        outputs = func.forward(t, y[0:2])
        dz_dt = outputs[0]
        v = dz_dt[:, 0:spatial_dim]
        p = dz_dt[:, spatial_dim:]
        g = outputs[1]
        wfr_loss = (torch.norm(v, dim=1) ** 2 + torch.norm(p, dim=1) ** 2 * alpha_exp 
                    + torch.norm(g, dim=1) ** 2 * alpha_gro).unsqueeze(1) * torch.exp(y[1])

        ssp_loss = torch.zeros(dz_dt.shape[0], 1, device=device)
        for i in range(dz_dt.shape[0]):
            if neighbor_index_new[i].shape[0] == 0:
                continue
            else:
                ssp_loss[i] = torch.mean(torch.norm(v[i, :].unsqueeze(0) - v[neighbor_index_new[i], :], dim=1) ** 2 * torch.exp(y[1][i]))

        # ssp_loss = torch.zeros(dz_dt.shape[0], 1, device=device)
        # x = y[0][:, 0:spatial_dim]
        # for i in range(dz_dt.shape[0]):
        #     if neighbor_index_new[i].shape[0] == 0:
        #         continue
        #     else:
        #         ssp_loss[i] = torch.mean((
        #             (v[i].unsqueeze(0) - v[neighbor_index_new[i]]) *
        #             ((x[i].unsqueeze(0) - x[neighbor_index_new[i]]) /
        #             (torch.norm(x[i].unsqueeze(0) - x[neighbor_index_new[i]], dim=1, keepdim=True) + 1e-8))
        #         ).sum(dim=1) ** 2 * torch.exp(y[1][i]))
        return dz_dt, g, wfr_loss, ssp_loss

    
    num_time_point = len(integral_time)

    if cell_number is None:
        cell_number = [cur_data.shape[0] for cur_data in data_train]
    
    cell_group = [np.zeros_like(cur_cell_type) for cur_cell_type in cell_type]
    unique_group = []
    cell_group_number = []
    for i in range(len(cell_type_prior)):
        cur_bio_guid = cell_type_prior[i]
        cur_group_number = []
        for j in range(num_time_point):
            indices = [index for index, value in enumerate(cell_type[j]) if value in cur_bio_guid[j]]
            cell_group[j][indices] = 'group' + str(i)
            # cur_group_number.append(len(indices))
            cur_group_number.append(len(indices)*cell_number[j]/len(cell_type[j]))
        cell_group_number.append(cur_group_number)
        unique_group.append('group' + str(i))

    cur_group_number = []
    for j in range(num_time_point):
        indices = [index for index, value in enumerate(cell_group[j]) if value is 0]
        cell_group[j][indices] = 'other'
        # cur_group_number.append(len(indices))
        cur_group_number.append(len(indices)*cell_number[j]/len(cell_type[j]))
    cell_group_number.append(cur_group_number)
    unique_group.append('other')


    spa_neighbor = config['spa_neighbor']
    exp_neighbor = config['exp_neighbor']
    adj_matrix = [-torch.ones(data_train[j].shape[0], exp_neighbor, dtype=torch.int) for j in range(num_time_point)]

    for i in range(len(cell_type_prior)):
        cur_bio_guid = cell_type_prior[i]
        if SSP_prior[i]:
            for j in range(num_time_point):
                indices = torch.tensor([index for index, value in enumerate(cell_type[j]) if value in cur_bio_guid[j]], dtype=torch.int64).to(device)

                exp_matrix = data_train[j][indices, spatial_dim:]
                spa_matrix = data_train[j][indices, 0:spatial_dim]

                _, index_spa = torch.topk(torch.cdist(spa_matrix, spa_matrix), largest=False, k=spa_neighbor)
                for k in range(len(indices)):
                    _, index_exp_k = torch.topk(torch.cdist(exp_matrix[k, :].unsqueeze(0), exp_matrix[index_spa[k], :]), largest=False, k=exp_neighbor)
                    index_k = indices[index_spa[k][index_exp_k]]
                    adj_matrix[j][indices[k]] = index_k
    

    if spatial_dim == 2:
        rigid_transformation = RigidTransformation_2D(num_time_point).to(device)
    elif spatial_dim == 3:
        rigid_transformation = RigidTransformation_3D(num_time_point).to(device)

    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': config['learning_rate']},
        {'params': rigid_transformation.parameters(), 'lr': config['learning_rate_rigid']}
    ], weight_decay=1e-5)

    unit_mass = 1.0

    n_epochs = config['n_epochs']
    num_samples = config['num_samples']

    model_path = config['model_path']
    rigid_transformation_path = config['rigid_transformation_path']

    alpha_exp = config['alpha_exp']
    alpha_gro = config['alpha_gro']
    lambda_match = config['lambda_match']
    lambda_SSP = config['lambda_SSP']
    kappa_exp = config['kappa_exp']
    kappa_gro = config['kappa_gro']

    if use_gene == False:
        alpha_exp = 1
        kappa_exp = 0
    if use_spatial == False:
        kappa_exp = 1
        use_alignment = False
    if use_growth == False:
        alpha_gro = 1
        kappa_gro = 0

    # configure training options
    options = {}
    options.update({'method': 'Dopri5'})
    options.update({'h': None})
    options.update({'rtol': 1e-3})
    options.update({'atol': 1e-5})
    options.update({'print_neval': False})
    options.update({'neval_max': 1000000})
    options.update({'safety': None})
    odeint_setp = 0.1

    for iter in tqdm(range(n_epochs)):
        optimizer.zero_grad()
        data_train_align = []
        for i in range(num_time_point):
            data_train_align.append(torch.tensor(data_train[i], device=device))

        if use_alignment:
            for i in range(1, num_time_point):
                coordinate = torch.tensor(data_train_align[i][:, 0:spatial_dim], device=device)
                coordinate = rigid_transformation(coordinate, i)
                data_train_align[i][:, 0:spatial_dim] = coordinate

        loss = torch.zeros(1).to(device)
        L2_value1 = torch.zeros(1, num_time_point, len(unique_group)).type(torch.float32).to(device)
        L2_value2 = torch.zeros(1, num_time_point, len(unique_group)).type(torch.float32).to(device)

        # model_with_wfr = partial(with_wfr_loss_model, func=model, device=device, alpha_exp=alpha_exp, alpha_gro=alpha_gro)


        z0, t0_group, t0_neighbor_index = Sampling_with_group_and_neighbor(num_samples[0], 0, data_train_align, cell_group, adj_matrix, None, device)
        t0_neighbor_index_new = []
        for i in range(z0.shape[0]):
            t0_neighbor_index_new.append(t0_neighbor_index[i, :][np.isfinite(t0_neighbor_index[i, :])])
        log_w_t0 = torch.log(torch.ones(z0.shape[0], 1).type(torch.float32) * torch.tensor(unit_mass)).to(
            device)
        wfr_loss_t0 = torch.zeros_like(log_w_t0, device=device)
        ssp_loss_t0 = torch.zeros_like(log_w_t0, device=device)
        t0_model_with_wfr_and_ssp = partial(with_wfr_and_ssp_loss_model, func=model, neighbor_index_new=t0_neighbor_index_new, device=device, alpha_exp=alpha_exp, alpha_gro=alpha_gro)
        z_time, log_w_time, wfr_loss_time, ssp_loss_time = \
            odeint(t0_model_with_wfr_and_ssp, y0=(z0, log_w_t0, wfr_loss_t0, ssp_loss_t0),
                t=torch.tensor(integral_time).type(torch.float32).to(device),
                atol=1e-5, rtol=1e-5, method='dopri5', options={'step_size': odeint_setp})  # method='midpoint'

        integral_time_back = integral_time[-1::-1]
        z_end, t_end_group, t_end_neighbor_index = Sampling_with_group_and_neighbor(num_samples[-1], num_time_point - 1, 
                                                 data_train_align, cell_group, adj_matrix, None, device)
        t_end_neighbor_index_new = []
        for i in range(z_end.shape[0]):
            t_end_neighbor_index_new.append(t_end_neighbor_index[i, :][np.isfinite(t_end_neighbor_index[i, :])])
        log_w_t_end = torch.log(torch.ones(z_end.shape[0], 1).type(torch.float32) * torch.tensor(unit_mass)).to(device)
        wfr_loss_t_end = torch.zeros_like(log_w_t_end, device=device)
        ssp_loss_t_end = torch.zeros_like(log_w_t_end, device=device)
        t_end_model_with_wfr_and_ssp = partial(with_wfr_and_ssp_loss_model, func=model, neighbor_index_new=t_end_neighbor_index_new, device=device, alpha_exp=alpha_exp, alpha_gro=alpha_gro)
        z_time_back, log_w_time_back, wfr_loss_time_back, ssp_loss_time_back = \
            odeint(t_end_model_with_wfr_and_ssp, y0=(z_end, log_w_t_end, wfr_loss_t_end, ssp_loss_t_end),
                t=torch.tensor(integral_time_back).type(torch.float32).to(device),
                atol=1e-5, rtol=1e-5, method='dopri5', options={'step_size': odeint_setp})  # method='midpoint'
        
        for i in range(num_time_point):
            
            z_ti, ti_group = Sampling_with_group(num_samples[i], i, data_train_align, cell_group, None, device)
            w_ti = torch.ones(z_ti.shape[0]).type(torch.float32).to(device)

            if i == 0:
                z_ti_back, log_w_ti_back = z_time_back[-(i + 1)], log_w_time_back[-(i + 1)]
                w_ti_back = torch.exp(log_w_ti_back).view(-1)

                for k in range(len(unique_group)):
                    index_back = np.where(np.array(t_end_group) == unique_group[k])[0]
                    index_ti = np.where(np.array(ti_group) == unique_group[k])[0]

                    M_spa = pot.dist(z_ti_back[index_back, 0:spatial_dim], z_ti[index_ti, 0:spatial_dim])
                    M_exp = pot.dist(z_ti_back[index_back, spatial_dim:], z_ti[index_ti, spatial_dim:])
                    M = M_exp * kappa_exp + M_spa * (1 - kappa_exp)

                    if use_growth:
                        ot_dist_ik = pot.emd2(w_ti_back[index_back] / torch.sum(w_ti_back[index_back]),
                                            w_ti[index_ti] / torch.sum(w_ti[index_ti]), M)
                        # TODO The weights may differ between groups
                        # ot_dist_ik = (cell_group_number[k][i]/cell_number[i]) * pot.emd2(w_ti_back[index_back] / torch.sum(w_ti_back[index_back]),
                        #                     w_ti[index_ti] / torch.sum(w_ti[index_ti]), M)
                    else:
                        ot_dist_ik = pot.emd2([], [], M)

                    ot_dist_ik = ot_dist_ik + \
                        torch.abs(torch.mean(w_ti_back[index_back]) - cell_group_number[k][i] / cell_group_number[k][-1]) / (cell_group_number[k][i] / cell_group_number[k][-1]) * kappa_gro
                    
                    L2_value2[0][i][k] = ot_dist_ik
                    loss = loss + L2_value2[0][i][k] * lambda_match

                    # print('back')
                    # print('ot_dist_ik:', ot_dist_ik)
                    # print('abs(sum(a)-sum(b))/sum(a):', torch.abs(torch.mean(w_ti_back[index_back]) - cell_group_number[k][i] / cell_group_number[k][-1]) / (cell_group_number[k][i] / cell_group_number[k][-1]))
                    # if iter % 100 == 0:
                    #     temp_pi = pot.emd(w_ti_back[index_back] / torch.sum(w_ti_back[index_back]),
                    #                             w_ti[index_ti] / torch.sum(w_ti[index_ti]), M)
                    #     print('dis_spa:', torch.sum(temp_pi * M_spa))
                    #     print('dis_exp:', torch.sum(temp_pi * M_exp))

            elif i == (num_time_point - 1):
                z_ti_forward, log_w_ti_forward = z_time[i], log_w_time[i]
                w_ti_forward = torch.exp(log_w_ti_forward).view(-1)

                for k in range(len(unique_group)):
                    index_forward = np.where(np.array(t0_group) == unique_group[k])[0]
                    index_ti = np.where(np.array(ti_group) == unique_group[k])[0]

                    M_spa = pot.dist(z_ti_forward[index_forward, 0:spatial_dim], z_ti[index_ti, 0:spatial_dim])
                    M_exp = pot.dist(z_ti_forward[index_forward, spatial_dim:], z_ti[index_ti, spatial_dim:])
                    M = M_exp * kappa_exp + M_spa * (1 - kappa_exp)

                    if use_growth:
                        ot_dist_ik = pot.emd2(w_ti_forward[index_forward]/ torch.sum(w_ti_forward[index_forward]),
                                            w_ti[index_ti] / torch.sum(w_ti[index_ti]), M)
                        # ot_dist_ik = (cell_group_number[k][i]/cell_number[i]) * pot.emd2(w_ti_forward[index_forward]/ torch.sum(w_ti_forward[index_forward]),
                        #                     w_ti[index_ti] / torch.sum(w_ti[index_ti]), M)
                    else:
                        ot_dist_ik = pot.emd2([], [], M)
                    
                    ot_dist_ik = ot_dist_ik + \
                        torch.abs(torch.mean(w_ti_forward[index_forward]) - cell_group_number[k][i] / cell_group_number[k][0]) / (cell_group_number[k][i] / cell_group_number[k][0]) * kappa_gro

                    L2_value1[0][i][k] = ot_dist_ik
                    loss = loss + L2_value1[0][i][k] * lambda_match


                    # print('forward')
                    # print('ot_dist_ik:', ot_dist_ik)
                    # print('abs(sum(a)-sum(b))/sum(a):', torch.abs(torch.mean(w_ti_forward[index_forward]) - cell_group_number[k][i] / cell_group_number[k][0]) / (cell_group_number[k][i] / cell_group_number[k][0]))
                    # if iter % 100 == 0:
                    #     temp_pi = pot.emd(w_ti_forward[index_forward]/ torch.sum(w_ti_forward[index_forward]),
                    #                             w_ti[index_ti] / torch.sum(w_ti[index_ti]), M)
                    #     print('dis_spa:', torch.sum(temp_pi * M_spa))
                    #     print('dis_exp:', torch.sum(temp_pi * M_exp))

            else:
                z_ti_forward, log_w_ti_forward = z_time[i], log_w_time[i]
                w_ti_forward = torch.exp(log_w_ti_forward).view(-1)

                for k in range(len(unique_group)):
                    index_forward = np.where(np.array(t0_group) == unique_group[k])[0]
                    index_ti = np.where(np.array(ti_group) == unique_group[k])[0]

                    M_spa = pot.dist(z_ti_forward[index_forward, 0:spatial_dim], z_ti[index_ti, 0:spatial_dim])
                    M_exp = pot.dist(z_ti_forward[index_forward, spatial_dim:], z_ti[index_ti, spatial_dim:])
                    M = M_exp * kappa_exp + M_spa * (1 - kappa_exp)

                    if use_growth:
                        ot_dist_ik = pot.emd2(w_ti_forward[index_forward]/ torch.sum(w_ti_forward[index_forward]),
                                            w_ti[index_ti] / torch.sum(w_ti[index_ti]), M)
                        # ot_dist_ik = (cell_group_number[k][i]/cell_number[i]) * pot.emd2(w_ti_forward[index_forward]/ torch.sum(w_ti_forward[index_forward]),
                        #                     w_ti[index_ti] / torch.sum(w_ti[index_ti]), M)
                    else:
                        ot_dist_ik = pot.emd2([], [], M)
                    
                    ot_dist_ik = ot_dist_ik + \
                        torch.abs(torch.mean(w_ti_forward[index_forward]) - cell_group_number[k][i] / cell_group_number[k][0]) / (cell_group_number[k][i] / cell_group_number[k][0]) * kappa_gro

                    L2_value1[0][i][k] = ot_dist_ik
                    loss = loss + L2_value1[0][i][k] * lambda_match

                    # print('forward')
                    # print('ot_dist_ik:', ot_dist_ik)
                    # print('abs(sum(a)-sum(b))/sum(a):', torch.abs(torch.mean(w_ti_forward[index_forward]) - cell_group_number[k][i] / cell_group_number[k][0]) / (cell_group_number[k][i] / cell_group_number[k][0]))
                
                z_ti_back, log_w_ti_back = z_time_back[-(i + 1)], log_w_time_back[-(i + 1)]
                w_ti_back = torch.exp(log_w_ti_back).view(-1)

                for k in range(len(unique_group)):
                    index_back = np.where(np.array(t_end_group) == unique_group[k])[0]
                    index_ti = np.where(np.array(ti_group) == unique_group[k])[0]

                    M_spa = pot.dist(z_ti_back[index_back, 0:spatial_dim], z_ti[index_ti, 0:spatial_dim])
                    M_exp = pot.dist(z_ti_back[index_back, spatial_dim:], z_ti[index_ti, spatial_dim:])
                    M = M_exp * kappa_exp + M_spa * (1 - kappa_exp)

                    if use_growth:
                        ot_dist_ik = pot.emd2(w_ti_back[index_back] / torch.sum(w_ti_back[index_back]),
                                            w_ti[index_ti] / torch.sum(w_ti[index_ti]), M)
                        # ot_dist_ik = (cell_group_number[k][i]/cell_number[i]) * pot.emd2(w_ti_back[index_back] / torch.sum(w_ti_back[index_back]),
                        #                     w_ti[index_ti] / torch.sum(w_ti[index_ti]), M)
                    else:
                        ot_dist_ik = pot.emd2([], [], M)

                    ot_dist_ik = ot_dist_ik + \
                        torch.abs(torch.mean(w_ti_back[index_back]) - cell_group_number[k][i] / cell_group_number[k][-1]) / (cell_group_number[k][i] / cell_group_number[k][-1]) * kappa_gro
                    
                    L2_value2[0][i][k] = ot_dist_ik
                    loss = loss + L2_value2[0][i][k] * lambda_match

                    # print('back')
                    # print('ot_dist_ik:', ot_dist_ik)
                    # print('abs(sum(a)-sum(b))/sum(a):', torch.abs(torch.mean(w_ti_back[index_back]) - cell_group_number[k][i] / cell_group_number[k][-1]) / (cell_group_number[k][i] / cell_group_number[k][-1]))
        
        loss = loss + (wfr_loss_time[-1].mean(0) - wfr_loss_time_back[-1].mean(0)) + \
            lambda_SSP * (ssp_loss_time[-1].mean(0) - ssp_loss_time_back[-1].mean(0))
        loss.backward()
        optimizer.step()

        if iter % 200 == 0:
            print('----------')
            print('iter', iter)
            print('loss:', loss)
            print('loss_wfr:', wfr_loss_time[-1].mean(0) - wfr_loss_time_back[-1].mean(0))
            print('loss_ssp:', ssp_loss_time[-1].mean(0) - ssp_loss_time_back[-1].mean(0))
            # print('rigid rotation_angle', rigid_transformation.rotation_angle)
            # print('rigid translation', rigid_transformation.translation)

        if iter > 0 and iter % 100 == 0:
            for i in range(len(num_samples)):
                num_samples[i] = num_samples[i] + 20

        if iter % 100 == 0 and iter > 1:
            torch.save(model, model_path)
            torch.save(rigid_transformation, rigid_transformation_path)

    torch.save(model, model_path)
    torch.save(rigid_transformation, rigid_transformation_path)

    return model, rigid_transformation


def train_base_old(data_train, 
               integral_time, 
               model, 
               spatial_dim,
               config,
               cell_number = None,
               use_gene = True,
               use_spatial = True,
               use_growth = True,
               use_alignment = True,
               device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    ''''The training process without priori information
    Args:
        observed_data: list of torch.Tensor, all observed data
        train_time: list of int, index of time points for training
        integral_time: list of float, all time points
        model: nn.Module, dynamic model including spatial migration rate, rna velocity and growth rate
        spatial_dim: int, spatial dimension, default 2
        device: torch.device, default 'cuda' if available else 'cpu'
        config: dict, configuration
    '''
    def with_wfr_loss_model(t, y, func, device, alpha_exp, alpha_gro):
        outputs = func.forward(t, y[0:2])
        dz_dt = outputs[0]
        v = dz_dt[:, 0:spatial_dim]
        p = dz_dt[:, spatial_dim:]
        g = outputs[1]
        # wfr_loss = (torch.norm(v, dim=1) ** 2 + torch.norm(p, dim=1) ** 2 * alpha_exp 
        #             + torch.norm(g, dim=1) ** 2 * alpha_gro).unsqueeze(1) * torch.exp(y[1])
        if use_gene == False:
            wfr_loss = (torch.norm(v, dim=1) ** 2
                        + torch.norm(g, dim=1) ** 2 * alpha_gro).unsqueeze(1) * torch.exp(y[1])
        elif use_spatial == False:
            wfr_loss = (torch.norm(p, dim=1) ** 2 * alpha_exp 
                        + torch.norm(g, dim=1) ** 2 * alpha_gro).unsqueeze(1) * torch.exp(y[1])
        elif use_growth == False:
            wfr_loss = (torch.norm(v, dim=1) ** 2 + torch.norm(p, dim=1) ** 2 * alpha_exp).unsqueeze(1) * torch.ones_like(y[1])
        else:
            wfr_loss = (torch.norm(v, dim=1) ** 2 + torch.norm(p, dim=1) ** 2 * alpha_exp 
                        + torch.norm(g, dim=1) ** 2 * alpha_gro).unsqueeze(1) * torch.exp(y[1])
        return dz_dt, g, wfr_loss
    
    num_time_point = len(integral_time)
    if spatial_dim == 2:
        rigid_transformation = RigidTransformation_2D(num_time_point).to(device)
    elif spatial_dim == 3:
        rigid_transformation = RigidTransformation_3D(num_time_point).to(device)

    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': config['learning_rate']},
        {'params': rigid_transformation.parameters(), 'lr': config['learning_rate_rigid']}
    ], weight_decay=1e-5)

    if cell_number is None:
        cell_number = [cur_data.shape[0] for cur_data in data_train]
    unit_mass = 1.0

    n_epochs = config['n_epochs']
    num_samples = config['num_samples']

    model_path = config['model_path']
    rigid_transformation_path = config['rigid_transformation_path']

    alpha_exp = config['alpha_exp']
    alpha_gro = config['alpha_gro']
    lambda_match = config['lambda_match']
    kappa_exp = config['kappa_exp']
    kappa_gro = config['kappa_gro']

    if use_gene == False:
        alpha_exp = 1
        kappa_exp = 0
    if use_spatial == False:
        kappa_exp = 1
        use_alignment = False
    if use_growth == False:
        alpha_gro = 1
        kappa_gro = 0


    # configure training options
    options = {}
    options.update({'method': 'Dopri5'})
    options.update({'h': None})
    options.update({'rtol': 1e-3})
    options.update({'atol': 1e-5})
    options.update({'print_neval': False})
    options.update({'neval_max': 1000000})
    options.update({'safety': None})
    odeint_setp = 0.1

    for iter in tqdm(range(n_epochs)):
        optimizer.zero_grad()
        data_train_align = []
        for i in range(num_time_point):
            data_train_align.append(torch.tensor(data_train[i], device=device))

        if use_alignment:
            for i in range(1, num_time_point):
                coordinate = torch.tensor(data_train_align[i][:, 0:spatial_dim], device=device)
                coordinate = rigid_transformation(coordinate, i)
                data_train_align[i][:, 0:spatial_dim] = coordinate

        loss = torch.zeros(1).to(device)
        L2_value1 = torch.zeros(1, num_time_point).type(torch.float32).to(device)
        L2_value2 = torch.zeros(1, num_time_point).type(torch.float32).to(device)

        model_with_wfr = partial(with_wfr_loss_model, func=model, device=device, alpha_exp=alpha_exp, alpha_gro=alpha_gro)

        z0 = Sampling_without_noise(num_samples[0], 0, data_train_align, None, device)
        log_w_t0 = torch.log(torch.ones(z0.shape[0], 1).type(torch.float32) * torch.tensor(unit_mass)).to(
            device)
        wfr_loss_t0 = torch.zeros_like(log_w_t0, device=device)
        z_time, log_w_time, wfr_loss_time = \
            odeint(model_with_wfr, y0=(z0, log_w_t0, wfr_loss_t0),
                t=torch.tensor(integral_time).type(torch.float32).to(device),
                atol=1e-5, rtol=1e-5, method='dopri5', options={'step_size': odeint_setp})  # method='midpoint'

        integral_time_back = integral_time[-1::-1]
        z_end = Sampling_without_noise(num_samples[-1], num_time_point - 1, data_train_align, None, device)
        log_w_t_end = torch.log(torch.ones(z_end.shape[0], 1).type(torch.float32) * torch.tensor(unit_mass)).to(
            device)
        wfr_loss_t_end = torch.zeros_like(log_w_t_end, device=device)
        z_time_back, log_w_time_back, wfr_loss_time_back = \
            odeint(model_with_wfr, y0=(z_end, log_w_t_end, wfr_loss_t_end),
                t=torch.tensor(integral_time_back).type(torch.float32).to(device),
                atol=1e-5, rtol=1e-5, method='dopri5', options={'step_size': odeint_setp})  # method='midpoint'
        
        for i in range(num_time_point):
            z_ti = Sampling_without_noise(num_samples[i], i, data_train_align, None, device)
            w_ti = torch.ones(z_ti.shape[0]).type(torch.float32).to(device)

            if i == 0:
                z_ti_back, log_w_ti_back = z_time_back[-(i + 1)], log_w_time_back[-(i + 1)]
                w_ti_back = torch.exp(log_w_ti_back).view(-1)
                M_spa = pot.dist(z_ti_back[:, 0:spatial_dim], z_ti[:, 0:spatial_dim])
                M_exp = pot.dist(z_ti_back[:, spatial_dim:], z_ti[:, spatial_dim:])
                if use_gene == False:
                    M = M_spa
                elif use_spatial == False:
                    M = M_exp
                else:
                    M = M_exp * kappa_exp + M_spa * (1 - kappa_exp)
                if use_growth:
                    ot_dist_i = pot.emd2(w_ti_back / torch.sum(w_ti_back), w_ti / torch.sum(w_ti), M)
                else:
                    ot_dist_i = pot.emd2([], [], M)
                ot_dist_i = ot_dist_i + torch.abs(torch.mean(w_ti_back)-cell_number[i]/cell_number[-1]) / (cell_number[i]/cell_number[-1]) * kappa_gro
                L2_value2[0][i] = ot_dist_i
                loss = loss + L2_value2[0][i] * lambda_match

                print('back')
                print('ot_dist_i:', ot_dist_i)
                print('abs(sum(a)-sum(b))/sum(a):', torch.abs(torch.mean(w_ti_back)-cell_number[i]/cell_number[-1]) / (cell_number[i]/cell_number[-1]))
                temp_pi = pot.emd(w_ti_back / torch.sum(w_ti_back), w_ti / torch.sum(w_ti), M)
                print('dis_spa:', torch.sum(temp_pi * M_spa))
                print('dis_exp:', torch.sum(temp_pi * M_exp))

            elif i == (num_time_point - 1):
                z_ti_forward, log_w_ti_forward = z_time[i], log_w_time[i]
                w_ti_forward = torch.exp(log_w_ti_forward).view(-1)
                M_spa = pot.dist(z_ti_forward[:, 0:spatial_dim], z_ti[:, 0:spatial_dim])
                M_exp = pot.dist(z_ti_forward[:, spatial_dim:], z_ti[:, spatial_dim:])
                if use_gene == False:
                    M = M_spa
                elif use_spatial == False:
                    M = M_exp
                else:
                    M = M_exp * kappa_exp + M_spa * (1 - kappa_exp)

                if use_growth:
                    ot_dist_i = pot.emd2(w_ti_forward / torch.sum(w_ti_forward), w_ti / torch.sum(w_ti), M)
                else:
                    ot_dist_i = pot.emd2([], [], M)
                ot_dist_i = ot_dist_i + torch.abs(torch.mean(w_ti_forward)-cell_number[i]/cell_number[0]) / (cell_number[i]/cell_number[0]) * kappa_gro
                L2_value1[0][i] = ot_dist_i
                loss = loss + L2_value1[0][i] * lambda_match

                print('forward')
                print('ot_dist_i:', ot_dist_i)
                print('abs(sum(a)-sum(b))/sum(a):', torch.abs(torch.mean(w_ti_forward)-cell_number[i]/cell_number[0]) / (cell_number[i]/cell_number[0]))
                temp_pi = pot.emd(w_ti_forward / torch.sum(w_ti_forward), w_ti / torch.sum(w_ti), M)
                print('dis_spa:', torch.sum(temp_pi * M_spa))
                print('dis_exp:', torch.sum(temp_pi * M_exp))

            else:
                z_ti_forward, log_w_ti_forward = z_time[i], log_w_time[i]
                w_ti_forward = torch.exp(log_w_ti_forward).view(-1)
                M_spa = pot.dist(z_ti_forward[:, 0:spatial_dim], z_ti[:, 0:spatial_dim])
                M_exp = pot.dist(z_ti_forward[:, spatial_dim:], z_ti[:, spatial_dim:])
                if use_gene == False:
                    M = M_spa
                elif use_spatial == False:
                    M = M_exp
                else:
                    M = M_exp * kappa_exp + M_spa * (1 - kappa_exp)

                if use_growth:
                    ot_dist_i = pot.emd2(w_ti_forward / torch.sum(w_ti_forward), w_ti / torch.sum(w_ti), M)
                else:
                    ot_dist_i = pot.emd2([], [], M)
                ot_dist_i = ot_dist_i + torch.abs(torch.mean(w_ti_forward)-cell_number[i]/cell_number[0]) / (cell_number[i]/cell_number[0]) * kappa_gro
                L2_value1[0][i] = ot_dist_i
                loss = loss + L2_value1[0][i] * lambda_match
                
                print('forward')
                print('ot_dist_i:', ot_dist_i)
                print('abs(sum(a)-sum(b))/sum(a):', torch.abs(torch.mean(w_ti_forward)-cell_number[i]/cell_number[0]) / (cell_number[i]/cell_number[0]))

                z_ti_back, log_w_ti_back = z_time_back[-(i + 1)], log_w_time_back[-(i + 1)]
                w_ti_back = torch.exp(log_w_ti_back).view(-1)
                M_spa = pot.dist(z_ti_back[:, 0:spatial_dim], z_ti[:, 0:spatial_dim])
                M_exp = pot.dist(z_ti_back[:, spatial_dim:], z_ti[:, spatial_dim:])
                if use_gene == False:
                    M = M_spa
                elif use_spatial == False:
                    M = M_exp
                else:
                    M = M_exp * kappa_exp + M_spa * (1 - kappa_exp)
                
                if use_growth:
                    ot_dist_i = pot.emd2(w_ti_back / torch.sum(w_ti_back), w_ti / torch.sum(w_ti), M)
                else:
                    ot_dist_i = pot.emd2([], [], M)
                ot_dist_i = ot_dist_i + torch.abs(torch.mean(w_ti_back)-cell_number[i]/cell_number[-1]) / (cell_number[i]/cell_number[-1]) * kappa_gro
                L2_value2[0][i] = ot_dist_i
                loss = loss + L2_value2[0][i] * lambda_match

                print('back')
                print('ot_dist_i:', ot_dist_i)
                print('abs(sum(a)-sum(b))/sum(a):', torch.abs(torch.mean(w_ti_back)-cell_number[i]/cell_number[-1]) / (cell_number[i]/cell_number[-1]))
        
        loss = loss + (wfr_loss_time[-1].mean(0) - wfr_loss_time_back[-1].mean(0))
        loss.backward()
        optimizer.step()

        print('----------')
        print('loss:', loss)
        print('loss_wfr:', wfr_loss_time[-1].mean(0) - wfr_loss_time_back[-1].mean(0))
        print('rigid rotation_angle', rigid_transformation.rotation_angle)
        print('rigid translation', rigid_transformation.translation)
        print(iter)

        if iter > 0 and iter % 100 == 0:
            for i in range(len(num_samples)):
                num_samples[i] = num_samples[i] + 20

        if iter % 100 == 0 and iter > 1:
            torch.save(model, model_path)
            torch.save(rigid_transformation, rigid_transformation_path)

    torch.save(model, model_path)
    torch.save(rigid_transformation, rigid_transformation_path)

    return model, rigid_transformation