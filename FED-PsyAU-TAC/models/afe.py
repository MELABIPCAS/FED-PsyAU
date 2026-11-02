"""AU 特征提取器。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AFE(nn.Module):
    """1x1 卷积压通道 + 空间注意力池化 + MLP，输出 12 个 AU 嵌入。"""

    def __init__(self, num_rois=65, patch_channels=3, hidden=128, num_au=12, au_dim=64, dropout=0.5):
        super().__init__()
        in_channels = num_rois * patch_channels  # 195
        self.num_au = num_au
        self.au_dim = au_dim

        # 压通道
        self.conv = nn.Conv2d(in_channels, hidden, kernel_size=1)
        self.bn = nn.BatchNorm2d(hidden)

        # 空间注意力池化
        self.attn_pool = nn.Conv2d(hidden, 1, kernel_size=1)

        # 映射到 AU
        self.mlp = nn.Sequential(
            nn.Linear(hidden, num_au * au_dim),  # 768
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, z):
        # z: (B, 65, 75) -> (B, 195, 5, 5)
        x = z.view(z.size(0), -1, 5, 5)
        x = F.relu(self.bn(self.conv(x)))

        weight = F.softmax(self.attn_pool(x).view(x.size(0), -1), dim=-1)
        pooled = torch.sum(x.view(x.size(0), x.size(1), -1) * weight.unsqueeze(1), dim=-1)

        mapped = self.mlp(pooled)  # (B, 768)
        return mapped.view(-1, self.num_au, self.au_dim)  # (B, 12, 64)
