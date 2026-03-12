import torch
import math
import numpy as np
import random
import torch.nn as nn
from torch.utils.data import random_split

from tqdm import tqdm

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

class AE(nn.Module):
    def __init__(self, n_genes, z_dims=10):
        super().__init__()
        self.encode_mlp1 = nn.Sequential(
            nn.Linear(in_features=n_genes, out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=z_dims),
            nn.Sigmoid(),
        )
        self.decode_mlp1 = nn.Sequential(
            nn.Linear(in_features=z_dims, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=n_genes),
            nn.Softplus()
        )

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(dim=0)
        batch, n_gene = x.shape
        z = self.encode_mlp1(x)
        out = self.decode_mlp1(z)
        return out


class DatasetAE(torch.utils.data.Dataset):
    def __init__(self, exp_data):
        self.x = exp_data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index, :]
    

def train_ae(model, train_loader, valid_loader, config, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    criterion = nn.MSELoss(reduction='mean')
    best_loss, early_stop_count = math.inf, 0

    for cur_epoch in range(config['n_epochs']):
        model.train()
        loss_record = []
        optimizer.zero_grad()
        for cur_data in train_loader:
            inputs = cur_data.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss_record.append(loss.detach().item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        mean_train_loss = sum(loss_record) / len(loss_record)

        model.eval()
        loss_record = []
        for cur_data in valid_loader:
            inputs = cur_data.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss_record.append(loss.detach().item())
        mean_valid_loss = sum(loss_record) / len(loss_record)

        if cur_epoch % 100 == 0:
            print(
                f"**Epoch {cur_epoch}**  \n"
                f"Train Loss: `{mean_train_loss:.6f}`  \n"
                f"Valid Loss: `{mean_valid_loss:.6f}`"
            )

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