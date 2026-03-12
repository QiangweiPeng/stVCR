import torch
from torch import nn
from typing import Iterable, Optional

class FCLayers(nn.Module):
    """全连接层模块，支持协变量注入"""
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list: Iterable[int] = None,
        n_cont: int = 0,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        activation_fn: nn.Module = nn.ReLU,
        bias: bool = True
    ):
        super().__init__()
        self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list] if n_cat_list else []
        self.n_cov = n_cont + sum(self.n_cat_list)
        
        layers = []
        current_dim = n_in
        for _ in range(n_layers-1):
            layers.extend([
                nn.Linear(current_dim + self.n_cov, n_hidden, bias=bias),
                nn.BatchNorm1d(n_hidden, momentum=0.01, eps=0.001) if use_batch_norm else nn.Identity(),
                activation_fn(),
                nn.Dropout(dropout_rate)
            ])
            current_dim = n_hidden
        
        # 最后一层不需要激活函数和dropout
        layers.append(nn.Linear(current_dim + (self.n_cov if n_layers == 1 else 0), n_out, bias=bias))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, *cat_list: int, cont: Optional[torch.Tensor] = None):
        # 处理分类协变量（one-hot编码）
        cat_embeds = []
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            if n_cat > 1:
                cat_embeds.append(nn.functional.one_hot(cat.squeeze(-1), n_cat))
        
        # 拼接所有特征
        features = [x]
        if cont is not None:
            features.append(cont)
        features += cat_embeds
        
        return self.net(torch.cat(features, dim=-1))

class Encoder(nn.Module):
    """确定性编码器"""
    def __init__(
        self,
        n_input: int,
        n_latent: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 2,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True
    ):
        super().__init__()
        self.net = FCLayers(
            n_in=n_input,
            n_out=n_latent,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            activation_fn=nn.ReLU
        )

    def forward(self, x: torch.Tensor, *cat_list: int):
        return self.net(x, *cat_list)

class Decoder(nn.Module):
    """确定性解码器"""
    def __init__(
        self,
        n_latent: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 2,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        final_activation: nn.Module = nn.Identity(),
        bias: bool = False
    ):
        super().__init__()
        self.net = FCLayers(
            n_in=n_latent,
            n_out=n_output,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            activation_fn=nn.ReLU,
            bias=bias
        )
        self.final_activation = final_activation

    def forward(self, z: torch.Tensor, *cat_list: int):
        return self.final_activation(self.net(z, *cat_list))

class AE2(nn.Module):
    """自编码器"""
    def __init__(
        self,
        n_input: int,
        n_latent: int,
        n_cat_list: Iterable[int] = None,
        encoder_n_layers: int = 2,
        decoder_n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        final_activation: nn.Module = nn.Identity()
    ):
        super().__init__()
        self.encode_mlp1 = Encoder(
            n_input=n_input,
            n_latent=n_latent,
            n_cat_list=n_cat_list,
            n_layers=encoder_n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm
        )
        
        self.decode_mlp1 = Decoder(
            n_latent=n_latent,
            n_output=n_input,
            n_cat_list=None,
            n_layers=decoder_n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            final_activation=final_activation,
            bias=False,
        )

    def forward(self, x: torch.Tensor, *cat_list: int):
        z = self.encode_mlp1(x, *cat_list)
        x_recon = self.decode_mlp1(z, *cat_list)
        return x_recon