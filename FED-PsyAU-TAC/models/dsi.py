"""AU 结构流 + 全局光流流的双流 Inception 网络。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super().__init__()
        reduced = max(in_planes // ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, reduced, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(reduced, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))
        return x * attention + x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.sigmoid(self.conv(torch.cat([
            torch.mean(x, dim=1, keepdim=True),
            torch.max(x, dim=1, keepdim=True)[0],
        ], dim=1)))
        return x * attention + x


class InceptionStage(nn.Module):
    """四分支：1x1、1x1+3x3、1x1+5x5、maxpool+1x1。"""

    def __init__(self, in_channels, branch_channels):
        super().__init__()
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1), nn.BatchNorm2d(branch_channels), nn.ReLU(inplace=True))
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1), nn.BatchNorm2d(branch_channels), nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, 3, padding=1), nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True))
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1), nn.BatchNorm2d(branch_channels), nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, 5, padding=2), nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True))
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, branch_channels, 1), nn.BatchNorm2d(branch_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        return torch.cat([
            self.branch1x1(x), self.branch3x3(x), self.branch5x5(x), self.branch_pool(x)
        ], dim=1)


class DSI(nn.Module):
    """AU 嵌入与全局光流融合成 512 维共享特征。"""

    def __init__(self, num_au=12, shared_dim=512, dropout=0.5):
        super().__init__()
        # AU 结构流：12 个 AU 各自 8x8
        self.au_stage1 = InceptionStage(num_au, 16)
        self.au_channel_attention = ChannelAttention(64)
        self.au_spatial_attention = SpatialAttention(3)
        self.au_stage2 = InceptionStage(64, 16)

        # 全局光流流：28x28
        self.of_stage1 = InceptionStage(3, 6)
        self.of_channel_attention = ChannelAttention(24)
        self.of_spatial_attention = SpatialAttention(3)
        self.of_stage2 = InceptionStage(24, 16)
        self.of_fc = nn.Sequential(nn.Linear(3136, 1024), nn.ReLU(inplace=True))

        self.dropout = nn.Dropout(dropout)
        self.fusion = nn.Linear(2048, shared_dim)

    def forward(self, au_features, global_flow):
        # AU 结构流
        y = self.au_stage1(au_features.view(au_features.size(0), -1, 8, 8))
        y = F.max_pool2d(y, kernel_size=2, stride=2)
        y = self.au_stage2(self.au_spatial_attention(self.au_channel_attention(y)))
        y = self.dropout(y.view(y.size(0), -1))  # (B, 64*4*4 = 1024)

        # 全局光流流
        z = self.of_stage1(global_flow)
        z = self.of_spatial_attention(self.of_channel_attention(z))
        z = F.max_pool2d(z, kernel_size=2, stride=2)
        z = F.max_pool2d(self.of_stage2(z), kernel_size=2, stride=2)
        z = self.dropout(self.of_fc(z.view(z.size(0), -1)))  # (B, 1024)

        return self.fusion(torch.cat([y, z], dim=1))  # (B, 512)
