"""基于先验知识的 AU 图注意力网络。"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

AU_NAMES = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17]
UPPER_AUS = [1, 2, 4, 5, 6, 7]
LOWER_AUS = [9, 10, 12, 14, 15, 17]

# 结构先验：A=1 保留该边，A=0 丢弃
A_UPPER = [
    [1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [1, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 1, 1],
]

A_LOWER = [
    [1, 1, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],
]

# 全脸图只保留自环和上下脸之间的跨区边
A_GLOBAL = [
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
    [0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    [1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0],
    [1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0],
    [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
]

# 共现先验：由 DFME 训练集统计得到
D_UPPER = [
    [0.4096, 0.3375, 0.0179, 0.2166, 0.0032, 0.0153],
    [0.3551, 0.4027, 0.0049, 0.2246, 0.0021, 0.0106],
    [0.0123, 0.0032, 0.6831, 0.0288, 0.0268, 0.2458],
    [0.2614, 0.2576, 0.0505, 0.4173, 0.0024, 0.0108],
    [0.0110, 0.0070, 0.1345, 0.0070, 0.7978, 0.0428],
    [0.0155, 0.0102, 0.3628, 0.0091, 0.0126, 0.5898],
]

D_LOWER = [
    [0.7980, 0.0887, 0.0148, 0.0640, 0.0000, 0.0345],
    [0.0311, 0.8235, 0.0675, 0.0381, 0.0121, 0.0277],
    [0.0021, 0.0270, 0.9391, 0.0097, 0.0097, 0.0125],
    [0.0151, 0.0255, 0.0162, 0.9085, 0.0081, 0.0267],
    [0.0000, 0.0282, 0.0565, 0.0282, 0.7903, 0.0968],
    [0.0122, 0.0280, 0.0315, 0.0402, 0.0420, 0.8462],
]

D_GLOBAL = [
    [0.3997, 0.3293, 0.0175, 0.2114, 0.0031, 0.0149, 0.0008, 0.0028, 0.0099, 0.0039, 0.0017, 0.0051],
    [0.3456, 0.3920, 0.0047, 0.2186, 0.0021, 0.0103, 0.0000, 0.0024, 0.0121, 0.0050, 0.0024, 0.0047],
    [0.0110, 0.0028, 0.6130, 0.0258, 0.0240, 0.2206, 0.0190, 0.0408, 0.0121, 0.0128, 0.0048, 0.0132],
    [0.2530, 0.2493, 0.0489, 0.4040, 0.0024, 0.0104, 0.0017, 0.0054, 0.0077, 0.0084, 0.0013, 0.0074],
    [0.0066, 0.0042, 0.0808, 0.0042, 0.4794, 0.0257, 0.0072, 0.0407, 0.3303, 0.0090, 0.0054, 0.0066],
    [0.0141, 0.0093, 0.3288, 0.0082, 0.0114, 0.5345, 0.0074, 0.0406, 0.0162, 0.0159, 0.0016, 0.0119],
    [0.0084, 0.0000, 0.2989, 0.0140, 0.0335, 0.0782, 0.4525, 0.0503, 0.0084, 0.0363, 0.0000, 0.0196],
    [0.0094, 0.0075, 0.2156, 0.0151, 0.0640, 0.1441, 0.0169, 0.4482, 0.0367, 0.0207, 0.0066, 0.0151],
    [0.0157, 0.0184, 0.0306, 0.0103, 0.2481, 0.0274, 0.0013, 0.0175, 0.6099, 0.0063, 0.0063, 0.0081],
    [0.0131, 0.0159, 0.0675, 0.0235, 0.0141, 0.0563, 0.0122, 0.0206, 0.0131, 0.7355, 0.0066, 0.0216],
    [0.0195, 0.0260, 0.0877, 0.0130, 0.0292, 0.0195, 0.0000, 0.0227, 0.0455, 0.0227, 0.6364, 0.0779],
    [0.0237, 0.0211, 0.0976, 0.0290, 0.0145, 0.0594, 0.0092, 0.0211, 0.0237, 0.0303, 0.0317, 0.6385],
]


def build_prior(adjacency, co_occurrence):
    """在有效边上归一化共现矩阵，A=0 处置 0。"""
    a = torch.tensor(adjacency, dtype=torch.float32)
    d = torch.tensor(co_occurrence, dtype=torch.float32)
    d = torch.where(a > 0, d, torch.zeros_like(d))
    d = torch.softmax(d, dim=-1)
    d = torch.where(a > 0, d, torch.zeros_like(d))
    return a, d / d.sum(dim=-1, keepdim=True)


class DPKGATLayer(nn.Module):
    """单层图注意力：权重 = (1-beta)*softmax(e) + beta*先验。"""

    def __init__(self, in_features, out_features, num_heads=3, dropout=0.3,
                 negative_slope=0.2, beta_init=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.proj = nn.ModuleList([nn.Linear(in_features, out_features, bias=False) for _ in range(num_heads)])
        self.att = nn.ParameterList([
            nn.Parameter(torch.empty(2 * out_features, 1)) for _ in range(num_heads)
        ])
        for att in self.att:
            nn.init.xavier_uniform_(att, gain=1.414)
        self.out_proj = nn.Linear(num_heads * out_features, out_features)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        # beta 随训练更新
        self.beta = nn.Parameter(torch.tensor(math.log(beta_init / (1.0 - beta_init))))

    def forward(self, h, adjacency, prior):
        # h: (B, N, C)
        beta = torch.sigmoid(self.beta)
        mask = adjacency > 0
        heads = []
        for proj, att in zip(self.proj, self.att):
            f = proj(h)
            half = f.size(-1)
            e = self.leaky_relu(
                torch.matmul(f, att[:half]) + torch.matmul(f, att[half:]).transpose(-1, -2)
            )
            e = torch.where(mask, e, torch.full_like(e, -9e15))
            e = F.softmax(e, dim=-1)
            alpha = (1.0 - beta) * e + beta * torch.where(mask, prior, torch.zeros_like(prior))
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
            heads.append(torch.matmul(alpha, f))
        out = self.out_proj(torch.cat(heads, dim=-1))
        return F.elu(out) + h


class DPKGATStream(nn.Module):
    """单个区域的两层图注意力。"""

    def __init__(self, num_nodes, feature_dim=64, num_heads=3, num_layers=2, dropout=0.3):
        super().__init__()
        self.layers = nn.ModuleList([
            DPKGATLayer(feature_dim, feature_dim, num_heads, dropout) for _ in range(num_layers)
        ])

    def forward(self, h, adjacency, prior):
        for layer in self.layers:
            h = layer(h, adjacency, prior)
        return h


class DPKGAT(nn.Module):
    """上脸、下脸分别建模，再在全脸图上融合。"""

    def __init__(self, feature_dim=64, num_heads=3, num_layers=2, dropout=0.3):
        super().__init__()
        self.upper_branch = DPKGATStream(len(UPPER_AUS), feature_dim, num_heads, num_layers, dropout)
        self.lower_branch = DPKGATStream(len(LOWER_AUS), feature_dim, num_heads, num_layers, dropout)
        self.global_branch = DPKGATStream(len(AU_NAMES), feature_dim, num_heads, num_layers, dropout)
        self.register_buffer("upper_adjacency", build_prior(A_UPPER, D_UPPER)[0])
        self.register_buffer("upper_prior", build_prior(A_UPPER, D_UPPER)[1])
        self.register_buffer("lower_adjacency", build_prior(A_LOWER, D_LOWER)[0])
        self.register_buffer("lower_prior", build_prior(A_LOWER, D_LOWER)[1])
        self.register_buffer("global_adjacency", build_prior(A_GLOBAL, D_GLOBAL)[0])
        self.register_buffer("global_prior", build_prior(A_GLOBAL, D_GLOBAL)[1])

    def forward(self, au_features):
        # au_features: (B, 12, 64)
        upper = au_features[:, :len(UPPER_AUS)]
        lower = au_features[:, len(UPPER_AUS):]
        h_upper = self.upper_branch(upper, self.upper_adjacency, self.upper_prior)
        h_lower = self.lower_branch(lower, self.lower_adjacency, self.lower_prior)
        h_local = torch.cat([h_upper, h_lower], dim=1)  # (B, 12, 64)
        h_global = self.global_branch(h_local, self.global_adjacency, self.global_prior)
        return h_local, h_global
