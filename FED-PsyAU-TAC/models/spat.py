"""位置感知 Transformer：把 65 个 ROI 的局部光流编码成序列特征。"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AbsolutePositionEncoding(nn.Module):
    """绝对位置编码：坐标 -> Linear + GELU + LayerNorm。"""

    def __init__(self, dim=64):
        super().__init__()
        self.proj = nn.Linear(2, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, points):
        return self.norm(F.gelu(self.proj(points)))


class RelativePositionBias(nn.Module):
    """相对位置偏置：坐标差经两层 MLP 得到每个头的偏置。"""

    def __init__(self, num_heads=3, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_heads),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, points):
        delta = points.unsqueeze(2) - points.unsqueeze(1)  # (B, N, N, 2)
        return self.mlp(delta)  # (B, N, N, H)


class RelativeBiasMultiHeadAttention(nn.Module):
    """3 个头各自保持 d_k=64，打分时加可学习的相对位置偏置，拼接后经 W_O 回到 64 维。"""

    def __init__(self, dim=64, num_heads=3, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim
        self.qkv = nn.Linear(dim, 3 * num_heads * dim)
        self.out_proj = nn.Linear(num_heads * dim, dim)
        self.rpb = RelativePositionBias(num_heads)
        self.alpha = nn.Parameter(torch.ones(num_heads))
        self.dropout = dropout

    def forward(self, x, points):
        batch, num_tokens, _ = x.shape
        q, k, v = self.qkv(x).view(
            batch, num_tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        bias = self.rpb(points).permute(0, 3, 1, 2)  # (B, H, N, N)
        attn = attn + self.alpha.view(1, self.num_heads, 1, 1) * bias
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(batch, num_tokens, -1)
        return self.out_proj(out)


class SPATBlock(nn.Module):
    """注意力 + FFN，两者都带残差和 LayerNorm。"""

    def __init__(self, dim=64, num_heads=3, ffn_dim=256, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RelativeBiasMultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
        )

    def forward(self, x, points):
        x = x + self.attn(self.norm1(x), points)
        x = x + self.ffn(self.norm2(x))
        return x


class SPAT(nn.Module):
    """65 个 3x5x5 的 ROI patch -> (B, 65, 75)。"""

    def __init__(self, num_rois=65, in_features=75, dim=64, out_features=75,
                 num_heads=3, num_blocks=2, ffn_dim=256, dropout=0.1):
        super().__init__()
        self.num_rois = num_rois
        self.input_proj = nn.Linear(in_features, dim)
        self.abs_pos = AbsolutePositionEncoding(dim)
        self.blocks = nn.ModuleList([
            SPATBlock(dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim, out_features)

    def forward(self, roi_inputs, points):
        # roi_inputs (B, 195, 5, 5)，points (B, 65, 2)
        x = roi_inputs.view(roi_inputs.size(0), self.num_rois, -1)
        x = self.input_proj(x) + self.abs_pos(points)
        for block in self.blocks:
            x = block(x, points)
        return self.output_proj(self.norm(x))  # (B, 65, 75)
