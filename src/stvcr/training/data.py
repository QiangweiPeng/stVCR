import torch
import numpy as np
import random

def load_data(adata, 
              gene_input_key = 'X_gene_input', 
              spatial_input_key = 'X_spatial_input', 
              time_input_key = 'time_input',
              device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    '''Load data for training
    Args:
        adata: AnnData object
        device: torch.device, default 'cuda' if available else 'cpu'
    Returns:
        data_train: list of torch.Tensor, training data
        integral_time: list of float, all time points
    '''
    integral_time = list(np.unique(adata.obs[time_input_key]))
    time_pts = range(len(integral_time))
    data_train = []
    for i in time_pts:
        data_train.append(torch.cat((torch.from_numpy(adata[adata.obs.time_input == integral_time[i], :].obsm[spatial_input_key]),
                                     torch.from_numpy(adata[adata.obs.time_input == integral_time[i], :].obsm[gene_input_key])),
                                    dim=1).type(torch.float32).to(device))
    return data_train, integral_time


def Sampling_without_noise_old(num_samples, time_all, time_pt, data_train, sigma, device):
    '''Sample the gene expression and spatial location of the specified number of cells 
    in the observation data at the specified time point without noise'''
    mu = data_train[time_all[time_pt]]
    num_data = mu.shape[0]  # mu is number_sample * dimension

    if num_data < num_samples:
        samples = mu[random.choices(range(0, num_data), k=num_samples)]
    else:
        samples = mu[random.sample(range(0, num_data), num_samples)]
    return samples


def Sampling_without_noise(num_samples, time_pt, data_train, sigma, device):
    '''Sample the gene expression and spatial location of the specified number of cells 
    in the observation data at the specified time point without noise'''
    mu = data_train[time_pt]
    num_data = mu.shape[0]  # mu is number_sample * dimension

    if num_data > num_samples:
        samples = mu[random.sample(range(0, num_data), num_samples)]
    else:
        samples = mu

    return samples


def Sampling_with_group_old(num_samples, time_all, time_pt, data_train, cell_group, sigma, device):
    '''Sample the gene expression, spatial location and type annotation of the specified number of cells 
    in the observation data at the specified time point without noise'''
    mu = data_train[time_all[time_pt]]
    cur_group = cell_group[time_all[time_pt]]
    num_data = mu.shape[0]  # mu is number_sample * dimension

    if num_data < num_samples:
        index = random.choices(range(0, num_data), k=num_samples)
        samples = mu[index]
        sample_group = cur_group[index]
    else:
        index = random.sample(range(0, num_data), num_samples)
        samples = mu[index]
        sample_group = cur_group[index]
    return samples, sample_group


def Sampling_with_group(num_samples, time_pt, data_train, cell_group, sigma, device):
    '''Sample the gene expression, spatial location and type annotation of the specified number of cells 
    in the observation data at the specified time point without noise'''
    mu = data_train[time_pt]
    cur_group = cell_group[time_pt]
    num_data = mu.shape[0]  # mu is number_sample * dimension

    if num_data > num_samples:
        index = random.sample(range(0, num_data), num_samples)
        samples = mu[index]
        sample_group = cur_group[index]
    else:
        samples = mu
        sample_group = cur_group

    return samples, sample_group


def Sampling_with_group_and_neighbor_old(num_samples, time_all, time_pt, data_train, cell_group, adj, sigma, device):
    '''Sample the gene expression, spatial location, type annotation and neighbor index of the specified number of cells 
    in the observation data at the specified time point without noise'''
    mu = data_train[time_all[time_pt]]
    cur_group = cell_group[time_all[time_pt]]
    cur_adj = adj[time_all[time_pt]]
    num_data = mu.shape[0]  # mu is number_sample * dimension

    if num_data < num_samples:
        index = random.choices(range(0, num_data), k=num_samples)
        samples = mu[index]
        sample_group = cur_group[index]
        mapping = dict(zip(index, np.arange(0, num_samples)))
        neighbor_index = np.array([[mapping.get(int(key), float('nan')) for key in keys_list] for keys_list in cur_adj[index]], dtype=float)
    else:
        index = random.sample(range(0, num_data), num_samples)
        samples = mu[index]
        sample_group = cur_group[index]
        mapping = dict(zip(index, np.arange(0, num_samples)))
        neighbor_index = np.array([[mapping.get(int(key), float('nan')) for key in keys_list] for keys_list in cur_adj[index]], dtype=float)
    return samples, sample_group, neighbor_index

def Sampling_with_group_and_neighbor(num_samples, time_pt, data_train, cell_group, adj, sigma, device):
    '''Sample the gene expression, spatial location, type annotation and neighbor index of the specified number of cells 
    in the observation data at the specified time point without noise'''
    mu = data_train[time_pt]
    cur_group = cell_group[time_pt]
    cur_adj = adj[time_pt]
    num_data = mu.shape[0]  # mu is number_sample * dimension

    if num_data > num_samples:
        index = random.sample(range(0, num_data), num_samples)
        samples = mu[index]
        sample_group = cur_group[index]
        mapping = dict(zip(index, np.arange(0, num_samples)))
        neighbor_index = np.array([[mapping.get(int(key), float('nan')) for key in keys_list] for keys_list in cur_adj[index]], dtype=float)
    else:
        samples = mu
        sample_group = cur_group
        mapping = dict(zip(range(0, num_data), np.arange(0, num_data)))
        neighbor_index = np.array([[mapping.get(int(key), float('nan')) for key in keys_list] for keys_list in cur_adj], dtype=float)
    return samples, sample_group, neighbor_index