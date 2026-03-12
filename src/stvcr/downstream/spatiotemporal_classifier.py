import torch
import numpy as np
import pandas as pd
import random
import math
from tqdm import tqdm

import torch.nn as nn
from torch.utils.data import random_split
from torch.utils.data import DataLoader

class MLPNetWork(nn.Module):
    def __init__(self, spatial_dim=2, input_gene_dim=3, output_dim=24):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=spatial_dim + input_gene_dim+ 1, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
        )
        self.out = nn.Sequential(
            # loss function includes softmax layer
            nn.Linear(in_features=128, out_features=output_dim),
        )

    def forward(self, x):
        # x is N*2
        batchsize = x.shape[0]
        out = self.mlp(x)
        out = self.out(out)
        return out
    
class Dataset(torch.utils.data.Dataset):
    def __init__(self, spatial_data, exp_data, time, label):
        self.x = torch.cat((spatial_data, exp_data, time.unsqueeze(1)), dim=1)
        self.label = label

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index, :], self.label[index]
    
def seed_all(seed=19491001):
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_valid_split(data_set, valid_ratio, seed):
    '''Split provided training data into training set and validation set'''
    valid_set_size = int(valid_ratio * len(data_set))
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size],
                                        generator=torch.Generator().manual_seed(seed))
    return train_set, valid_set


def train_st_classifier(model, train_loader, valid_loader, config, device):

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    best_loss, early_stop_count = math.inf, 0

    for cur_epoch in range(config['n_epochs']):
        model.train()
        loss_record = []
        loss_entropy_record = []
        loss_time_l1norm_record = []
        optimizer.zero_grad()
        for cur_data in train_loader:
            inputs, labels = cur_data
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs.requires_grad = True
            outputs = model(inputs)
            loss_entropy = criterion(outputs, labels)
            grad_outputs = torch.ones_like(outputs)
            loss_time_l1norm = torch.mean(torch.abs(torch.autograd.grad(outputs, inputs, grad_outputs, create_graph=True)[0][:, -1]))
            loss = loss_entropy + config['weight_time_l1_norm'] * loss_time_l1norm
            # loss = loss_entropy
            loss_record.append(loss.detach().item())
            loss_entropy_record.append(loss_entropy.detach().item())
            loss_time_l1norm_record.append(loss_time_l1norm.detach().item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # mean_train_loss = sum(loss_record) / len(loss_record)
        # mean_train_loss_entropy = sum(loss_entropy_record) / len(loss_entropy_record)
        # mean_train_loss_time_l1norm = sum(loss_time_l1norm_record) / len(loss_time_l1norm_record)

        model.eval()
        loss_record = []
        loss_entropy_record = []
        loss_time_l1norm_record = []
        for cur_data in valid_loader:
            inputs, labels = cur_data
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs.requires_grad = True
            outputs = model(inputs)
            loss_entropy = criterion(outputs, labels)
            grad_outputs = torch.ones_like(outputs)
            loss_time_l1norm = torch.mean(
                torch.abs(torch.autograd.grad(outputs, inputs, grad_outputs, create_graph=True)[0][:, -1]))
            loss = loss_entropy + 10.0 * loss_time_l1norm
            # loss = loss_entropy
            loss_record.append(loss.detach().item())
            loss_entropy_record.append(loss_entropy.detach().item())
            loss_time_l1norm_record.append(loss_time_l1norm.detach().item())
        mean_valid_loss = sum(loss_record) / len(loss_record)
        # mean_valid_loss_entropy = sum(loss_entropy_record) / len(loss_entropy_record)
        # mean_valid_loss_time_l1norm = sum(loss_time_l1norm_record) / len(loss_time_l1norm_record)

        # print(f" epoch:{cur_epoch}\n"
        #       f" train_loss:{mean_train_loss} valid_loss:{mean_valid_loss}\n"
        #       f" train_loss_mse:{mean_train_loss_entropy} valid_loss:{mean_valid_loss_entropy}\n"
        #       f" train_loss_pearson:{mean_train_loss_time_l1norm} valid_loss:{mean_valid_loss_time_l1norm}")

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            early_stop_count = 0
            torch.save(model, config['save_path'])  # Save your best model
            # print('Saving model with loss {:.3f}...'.format(best_loss))
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('Early stopping in current iteration!')
            break

def create_spatiotemporal_classifier(adata, 
                              st_classifier_save_path,
                              annotation_key='Annotation',
                              device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                              ):
    '''Train the spatiotemporal classifier using the observation data'''

    # create annotation label
    cell_types = adata.obs[annotation_key]
    unique_types = np.array(list(cell_types.cat.categories))
    cell_type_to_label_map = {cell_type: i for i, cell_type in enumerate(unique_types)}
    label_to_cell_type_map = {i: cell_type for i, cell_type in enumerate(unique_types)}
    label = torch.tensor([cell_type_to_label_map[cell_type] for cell_type in cell_types])

    config = {
        'seed': 19491001,
        'learning_rate': 1e-3,
        'n_epochs': 1000,
        'batch_size': 1000,
        'early_stop': 50,
        'valid_ratio': 0.1,
        'weight_time_l1_norm': 10.0,
        'save_path': st_classifier_save_path,
    }
    seed_all(config['seed'])

    # data preparation
    exp_data = torch.tensor(adata.obsm['X_gene_input'], dtype=torch.float32)
    spatial_data = torch.tensor(adata.obsm['X_spatial_aligned'], dtype=torch.float32)
    time = torch.tensor(np.array(adata.obs['time_input']), dtype=torch.float32)
    data_set = Dataset(spatial_data=spatial_data, exp_data=exp_data, time=time, label=label)
    train_data, valid_data = train_valid_split(data_set, config['valid_ratio'], config['seed'])
    train_loader = DataLoader(train_data, shuffle=True, batch_size=config['batch_size'])
    valid_loader = DataLoader(valid_data, shuffle=False, batch_size=config['batch_size'])

    model = MLPNetWork(spatial_dim=spatial_data.shape[1], input_gene_dim=exp_data.shape[1],
                       output_dim=unique_types.shape[0]).to(device)

    # train
    print('Start training spatiotemporal classifier...')
    train_st_classifier(model, train_loader, valid_loader, config, device)
    print('Spatiotemporal classifier training finished.')

    return label_to_cell_type_map