import torch
from torch import nn
from typing import Iterable, Optional

class AE3(nn.Module):
    """
    自编码器（带可选协变量支持）
    默认结构与你的autoencoder相同，参数与scVI一致
    """
    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        n_output: int,
        dropout_rate: float = 0.1,
        n_cat_list: Iterable[int] = None
    ):
        super().__init__()
        self.n_cat_list = list(n_cat_list) if n_cat_list else []
        
        # 计算协变量总维度（自动过滤单类别变量）
        self.n_cov = sum(n_cat for n_cat in self.n_cat_list if n_cat > 1)
        
        # 编码器（输入维度 = 原始输入 + 协变量维度）
        self.encoder = nn.Sequential(
            nn.Linear(n_input + self.n_cov, n_hidden),
            nn.BatchNorm1d(n_hidden, momentum=0.01, eps=0.001),  # scVI默认参数
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(n_hidden, n_output)
        )
        
        # 解码器（保持你的原始结构）
        self.decoder = nn.Linear(n_output, n_input, bias=False)

    def forward(self, x: torch.Tensor, *cat_list: torch.Tensor):
        # 处理分类协变量
        if self.n_cat_list:
            # 生成one-hot编码
            one_hot_list = []
            for n_cat, cat_tensor in zip(self.n_cat_list, cat_list):
                if n_cat > 1:
                    one_hot = nn.functional.one_hot(
                        cat_tensor.squeeze(-1).long(), 
                        num_classes=n_cat
                    )
                    one_hot_list.append(one_hot)
            
            # 拼接协变量到输入数据
            x = torch.cat([x, *one_hot_list], dim=-1).float()
        
        # 编码-解码
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z