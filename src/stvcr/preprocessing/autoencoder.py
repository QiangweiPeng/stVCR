import torch
from torch.utils.data import DataLoader

from .ae_utils import seed_all, AE, DatasetAE, train_valid_split, train_ae
from .ae_utils2 import AE2


def ae_dim_reduction(adata, 
                     ae_model_save_path, 
                     gene_expression_key = None,
                     z_dims = 10,
                     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                     learning_rate = 1e-3,
                     n_epochs = 1000,
                     batch_size = 1000,
                     early_stop = 30,
                     valid_ratio = 0.1,
                     seed = 19491001):
    '''Dimension reduction using autoencoder
    Args:
        adata: AnnData object
        ae_model_save_path: str, path to save the trained autoencoder model
        gene_expression_key: str, key of gene expression data in adata.layers, default None
        z_dims: int, number of latent dimensions, default 10
        device: torch.device, default 'cuda' if available else 'cpu'
        learning_rate: float, default 1e-3
        n_epochs: int, default 1000
        batch_size: int, default 1000
        early_stop: int, default 30
        valid_ratio: float, default 0.1
        seed: int, random seed, default 19491001
    
    Returns:
        The trained autoencoder is saved in path ae_model_save_path, and the result of dimension reduction is saved in adata.osbm['X_ae'].
    '''

    # parameter setting
    config = {
    'seed': seed,
    'learning_rate': learning_rate,
    'n_epochs': n_epochs,
    'batch_size': batch_size,
    'early_stop': early_stop,
    'valid_ratio': valid_ratio,
    'save_path': ae_model_save_path,}
    seed_all(config['seed'])


    # data preparation
    if gene_expression_key is None:
        exp_data = adata.X.A
    else:
        exp_data = adata.layers[gene_expression_key].values
    exp_data = torch.tensor(adata.X.A, dtype=torch.float32).to(device)
    data_set = DatasetAE(exp_data=exp_data)
    train_data, valid_data = train_valid_split(data_set, config['valid_ratio'], config['seed'])
    train_loader = DataLoader(train_data, shuffle=True, batch_size=config['batch_size'])
    valid_loader = DataLoader(valid_data, shuffle=False, batch_size=config['batch_size'])

    # model
    model = AE(n_genes=exp_data.shape[1], z_dims=z_dims).to(device)
    # model = AE2(n_input=exp_data.shape[1], n_latent=z_dims).to(device)


    # train
    print('Start training autoencoder...')
    train_ae(model, train_loader, valid_loader, config, device)
    print('Autoencoder training finished.')

    # write back to adata
    adata.obsm['X_ae'] = model.encode_mlp1(exp_data).detach().cpu().numpy()