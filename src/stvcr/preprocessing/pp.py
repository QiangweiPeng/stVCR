import scanpy as sc
import numpy as np
import torch

from .utils import rigid_body_transformation_invariant_OT
from .autoencoder import ae_dim_reduction

def pp_with_scanpy(adata,
               normlization = False,
               log1p = False,
               n_top_genes=2000,
               batch_key="Batch",
               min_genes=100,
               min_cells=3,):
    '''Filter cells and genes using scanpy.
    Args:
        adata: AnnData object
        n_top_genes: int, number of highly variable genes to keep, default 2000
        batch_key: str, batch key in adata.obs, default 'Batch'
        min_genes: int, minimum number of genes expressed in a cell, default 100
        min_cells: int, minimum number of cells expressing a gene, default 3
    Returns:
        Filter AnnData object
    '''

    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    if normlization:
        # sc.pp.normalize_total(adata)
        normalized_X = adata.X.copy()
        for batch in adata.obs[batch_key].unique():
            batch_adata = adata[adata.obs[batch_key] == batch, :].copy()
            sc.pp.normalize_total(batch_adata)
            normalized_X[adata.obs[batch_key] == batch, :] = batch_adata.X.copy()
        adata.X = normalized_X.copy()
    if log1p:
        sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, batch_key=batch_key, subset=True)
  
    return adata


def normalization_time(adata, time_key='time'):
    '''Set the initial time to 0.
    Args:
        adata: AnnData object
        time_key: str, time key in adata.obs, default 'time'
    Returns:    
        AnnData object with 'time_input' in adata.obs.
    '''
    init_time = np.min(adata.obs[time_key])
    adata.obs['time_input'] = adata.obs[time_key] - init_time

    return adata 


def pp_init(adata, 
            spatial_key = 'spatial', 
            gene_redunction_key = 'X_ae',
            time_key = 'time', 
            use_initial_alignment = True,
            alpha = 0.002, 
            down_sampling_number = 5000,
            normlize_spatial_coordinate = True):
    '''Preprocessing related to the initial input
    Args:
        adata: AnnData object
        spatial_key: str, key of spatial coordinates in adata.obsm, default 'spatial'
        time_key: str, time key in adata.obs, default 'time'
        use_initial_alignment: bool, whether to use initial alignment, default True
        alpha: float, parameters that weigh gene expression and spatial coordinates in the OT, default 0.002
        down_sampling_number: int, number of down sampling in the OT, default 5000
        normlize_spatial_coordinate: bool, whether to normalize spatial coordinates, default True
    Returns:
        AnnData object with 'X_gene_input' and 'X_spatial_input' in adata.obsm and 'time_input' in adata.obs.
    '''

    # set the initial time to 0
    normalization_time(adata, time_key=time_key)

    # gene expression after dimensionality reduction
    adata.obsm['X_gene_input'] = np.array(adata.obsm[gene_redunction_key])

    # initialization alignment based on rigid body transform invariant OT and normalization of spatial coordinates
    integral_time = list(np.unique(adata.obs['time_input']))
    time_pts = range(len(integral_time))

    for i in time_pts:
        spatial_i = adata[adata.obs.time_input == integral_time[i], :].obsm[spatial_key].copy()
        spatial_i = spatial_i - (np.ones(spatial_i.shape[0]) / spatial_i.shape[0]).dot(spatial_i)
        adata.obsm[spatial_key][adata.obs.time_input == integral_time[i], :] = spatial_i

    adata.obsm['X_spatial_input'] = adata.obsm[spatial_key].copy()

    if use_initial_alignment:
        for i in time_pts:
            if i == 0:
                spatial_i = adata[adata.obs.time_input == integral_time[i], :].obsm['X_spatial_input'].copy()
                spatial_i = spatial_i - (np.ones(spatial_i.shape[0]) / spatial_i.shape[0]).dot(spatial_i)
                adata.obsm['X_spatial_input'][adata.obs.time_input == integral_time[i], :] = spatial_i
            elif i > 0:
                spatial_i = adata[adata.obs.time_input == integral_time[i], :].obsm['X_spatial_input'].copy()
                spatial_i = spatial_i - (np.ones(spatial_i.shape[0]) / spatial_i.shape[0]).dot(spatial_i)
                adata.obsm['X_spatial_input'][adata.obs.time_input == integral_time[i], :] = spatial_i

                _, R = rigid_body_transformation_invariant_OT(adata[adata.obs.time_input == integral_time[i - 1], :], 
                                                            adata[adata.obs.time_input == integral_time[i], :], iter_num=5, 
                                                            spatial_key='X_spatial_input', alpha=alpha, 
                                                            down_sampling_number=down_sampling_number)
                spatial_i = R.dot(spatial_i.T).T
                adata.obsm['X_spatial_input'][adata.obs.time_input == integral_time[i], :] = spatial_i

                # for j in np.arange(i+1, len(integral_time)):
                #     spatial_j = adata[adata.obs.time_input == integral_time[j], :].obsm['X_spatial_input'].copy()
                #     spatial_j = R.dot(spatial_j.T).T
                #     adata.obsm['X_spatial_input'][adata.obs.time_input == integral_time[j], :] = spatial_j
                
                # print(R)
    else:
        adata.obsm['X_spatial_input'] = adata.obsm[spatial_key].copy()

    if normlize_spatial_coordinate:
        adata.uns['spatial_scale_factor'] = np.max(np.abs(adata.obsm['X_spatial_input']))
        adata.obsm[spatial_key] = adata.obsm[spatial_key] / np.max(np.abs(adata.obsm['X_spatial_input']))
        adata.obsm['X_spatial_input'] = adata.obsm['X_spatial_input'] / np.max(np.abs(adata.obsm['X_spatial_input']))
    return adata


def pp_stvcr(adata, 
             use_pp_with_scanpy = True,
             ae_model_save_path = None,
             normlization = False,
             log1p = False,
             n_top_genes=2000,
             batch_key="Batch",
             min_genes=100,
             min_cells=3,
             gene_expression_key = None,
             z_dims = 10,
             device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
             learning_rate = 1e-3,
             n_epochs = 1000,
             batch_size = 1000,
             early_stop = 30,
             valid_ratio = 0.1,
             seed = 19491001,
             spatial_key = 'spatial',
             time_key = 'time',
             use_initial_alignment = True,
             alpha = 0.002,
             down_sampling_number = 5000):
    
    '''Preprocessing for stVCR
    Args:
        adata: AnnData object
        ae_model_save_path: str, path to save the trained autoencoder model
        normlization: bool, whether to normalize the data, default False
        log1p: bool, whether to log1p the data, default False
        n_top_genes: int, number of highly variable genes to keep, default 2000
        batch_key: str, batch key in adata.obs, default 'Batch'
        min_genes: int, minimum number of genes expressed in a cell, default 100
        min_cells: int, minimum number of cells expressing a gene, default 3
        gene_expression_key: str, key of gene expression data in adata.layers, default None
        z_dims: int, number of latent dimensions, default 10
        device: torch.device, default 'cuda' if available else 'cpu'
        learning_rate: float, default 1e-3
        n_epochs: int, default 1000
        batch_size: int, default 1000
        early_stop: int, default 30
        valid_ratio: float, default 0.1
        seed: int, random seed, default 19491001
        spatial_key: str, key of spatial coordinates in adata.obsm, default 'spatial'
        time_key: str, time key in adata.obs, default 'time'
        alpha: float, parameters that weigh gene expression and spatial coordinates in the OT, default 0.002
        down_sampling_number: int, number of down sampling in the OT, default 5000
    Returns:
        adata: AnnData object
    '''

    # filter cells and genes
    if use_pp_with_scanpy:
        pp_with_scanpy(adata, normlization=normlization, log1p=log1p, n_top_genes=n_top_genes, batch_key=batch_key, min_genes=min_genes, min_cells=min_cells)

    # dimension reduction using autoencoder
    if 'X_ae' not in adata.obsm.keys():
        ae_dim_reduction(adata, ae_model_save_path, gene_expression_key=gene_expression_key, z_dims=z_dims, device=device, learning_rate=learning_rate, n_epochs=n_epochs, batch_size=batch_size, early_stop=early_stop, valid_ratio=valid_ratio, seed=seed)

    # preprocessing related to the initial input
    pp_init(adata, spatial_key=spatial_key, time_key=time_key, alpha=alpha, down_sampling_number=down_sampling_number, use_initial_alignment=use_initial_alignment)

    return adata