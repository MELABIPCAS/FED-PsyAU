
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        reduced_planes = max(in_planes // ratio, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, reduced_planes, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(reduced_planes, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = avg_out + max_out
        attention = self.sigmoid(attention)

        out = x * attention + x
        # out = x * attention
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention_x = torch.cat([avg_out, max_out], dim=1)
        attention_x = self.conv1(attention_x)
        attention = self.sigmoid(attention_x)
        out = x * attention + x
        # out = x * attention
        return out


class InceptionNet(nn.Module):
    def __init__(self, num_classes=7):
        super(InceptionNet, self).__init__()
        in_chanel = 12
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_chanel, 16, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_chanel, 16, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(in_chanel, 16, kernel_size=(1, 1), stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=(5, 5), stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))
        self.conv1_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1),
            nn.Conv2d(in_chanel, 16, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))


        self.conv2_1 = nn.Sequential(
            nn.Conv2d(64, 48, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True))
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(64, 48, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True))
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(64, 48, kernel_size=(1, 1), stride=1, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=(5, 5), stride=1, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True))
        self.conv2_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1),
            nn.Conv2d(64, 48, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True))

        in_chanel = 3
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(in_chanel, 6, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True))
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(in_chanel, 6, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, 6, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True))
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(in_chanel, 6, kernel_size=(1, 1), stride=1, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, 6, kernel_size=(5, 5), stride=1, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True))
        self.conv3_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1),
            nn.Conv2d(in_chanel, 6, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True))
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(24, 16, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))
        self.conv4_2 = nn.Sequential(
            nn.Conv2d(24, 16, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))
        self.conv4_3 = nn.Sequential(
            nn.Conv2d(24, 16, kernel_size=(1, 1), stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=(5, 5), stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))
        self.conv4_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1),
            nn.Conv2d(24, 16, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))

        # Add Attention modules here
        # self.channel_attention1 = ChannelAttention(24, 8)
        # self.spatial_attention1 = SpatialAttention()
        # # Adding attention modules for the second set of convolutions
        # self.channel_attention2 = ChannelAttention(64, 8)
        # self.spatial_attention2 = SpatialAttention()

        self.dropout = nn.Dropout(p=0.5)
        self.face_fc = nn.Sequential(
            nn.Linear(3136, 1024),
            nn.ReLU(inplace=True))
        self.fc = nn.Linear(1792, num_classes)

    def forward(self, global_feature, x):
        # AU features extraction
        x11 = self.conv1_1(x)
        x12 = self.conv1_2(x)
        x13 = self.conv1_3(x)
        x14 = self.conv1_4(x)
        x1_con1 = torch.cat([x11, x12, x13, x14], 1)

        x1_con1 = F.max_pool2d(x1_con1, kernel_size=(2, 2), stride=(2, 2))
        y11 = self.conv2_1(x1_con1)
        y12 = self.conv2_2(x1_con1)
        y13 = self.conv2_3(x1_con1)
        y14 = self.conv2_4(x1_con1)
        y1_con2 = torch.cat([y11, y12, y13, y14], 1)
        # print(y1_con2.size())
        y = y1_con2.view(y1_con2.size(0), -1)

        # Optical flow features extraction
        m11 = self.conv3_1(global_feature)
        m12 = self.conv3_2(global_feature)
        m13 = self.conv3_3(global_feature)
        m14 = self.conv3_4(global_feature)
        m1_con1 = torch.cat([m11, m12, m13, m14], 1)

        # Apply Channel and Spatial Attention
        # m1_con1 = self.channel_attention1(m1_con1)
        # m1_con1 = self.spatial_attention1(m1_con1)

        m1_con1 = F.max_pool2d(m1_con1, kernel_size=(2, 2), stride=(2, 2))
        n11 = self.conv4_1(m1_con1)
        n12 = self.conv4_2(m1_con1)
        n13 = self.conv4_3(m1_con1)
        n14 = self.conv4_4(m1_con1)
        n1_con2 = torch.cat([n11, n12, n13, n14], 1)

        # Apply Channel and Spatial Attention
        # n1_con2 = self.channel_attention2(n1_con2)
        # n1_con2 = self.spatial_attention2(n1_con2)

        n1_con2 = F.max_pool2d(n1_con2, kernel_size=(2, 2), stride=(2, 2))
        # print(n1_con2.size())
        z = n1_con2.view(n1_con2.size(0), -1)
        z = self.face_fc(z)


        z = self.dropout(z)
        y = self.dropout(y)
        y = self.fc(torch.cat([y, z], 1))  # 输出：num_classes
        return y
