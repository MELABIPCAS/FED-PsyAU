import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model.GAT import GAT_based_on_dynamic_prior_knowledge


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 1:
            pe[:, 1:d_model:2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)


        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


# class TransformerEncoderLayer(nn.Module):
#     def __init__(self, d_model, nhead, max_len=5000):
#         super(TransformerEncoderLayer, self).__init__()
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
#         self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
#         # self.positional_encoding = PositionalEncoding(d_model, max_len)
#
#     def forward(self, src):
#         # src = self.positional_encoding(src)
#         return self.transformer_encoder(src)

# class TransformerEncoderLayer(nn.Module):
#     def __init__(self, d_model, nhead, d_ff=2048, max_len=5000):
#         super(TransformerEncoderLayer, self).__init__()
#
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
#
#         self.feed_forward = nn.Sequential(
#             nn.Linear(d_model, d_ff),
#             nn.ReLU(),
#             nn.Linear(d_ff, d_model)
#         )
#
#         self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
#
#     def forward(self, src):
#         transformer_out = self.transformer_encoder(src)
#
#         ff_output = self.feed_forward(transformer_out)
#
#         return ff_output

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff=2048, max_len=5000):
        super(TransformerEncoderLayer, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

    def forward(self, src):
        transformer_out = self.transformer_encoder(src)

        ff_output = self.feed_forward(transformer_out)

        output = ff_output + src

        return output


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(3)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x


class GroupSEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16, group_size=3):
        super(GroupSEBlock, self).__init__()
        self.group_size = group_size
        self.num_groups = in_channels // group_size
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.num_groups, max(self.num_groups // reduction, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(self.num_groups // reduction, 1), self.num_groups, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, h, w = x.size()
        assert channels % self.group_size == 0

        self.num_groups = channels // self.group_size

        grouped_x = x.view(batch_size, self.num_groups, self.group_size, h,
                           w)
        y = self.global_pool(grouped_x).mean(dim=2)
        y = y.view(batch_size, self.num_groups)

        y = self.fc(y).view(batch_size, self.num_groups, 1, 1, 1)  # (batch_size, num_groups, 1, 1, 1)

        y = y.repeat(1, 1, self.group_size, 1, 1).view(batch_size, channels, 1, 1)

        return x * y


class LandmarkToAU(nn.Module):
    def __init__(self, m, n):
        super(LandmarkToAU, self).__init__()
        self.m = m
        self.n = n

        self.conv1 = nn.Conv2d(in_channels=m, out_channels=n, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(n)

        self.conv2 = nn.Conv2d(in_channels=n, out_channels=n // 2, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(n // 2)

        self.relu = nn.ReLU(inplace=True)

        self.se = GroupSEBlock(m, group_size=3)

    def forward(self, x):
        residual = x

        x = self.se(x)  # (batch_size, m, 5, 5)

        x = x + residual

        x = self.conv1(x)  # (batch_size, n, 5, 5)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)  # (batch_size, n//2, 5, 5)
        x = self.bn2(x)
        x = self.relu(x)

        return x

def construct_adjacency_matrix(labels, matrices):
    batch_size = labels.size(0)
    for i in range(batch_size):
        label = labels[i]
        for row in range(6):
            for col in range(6, 12):
                if label[col] == 1:
                    matrices[i, row, col] = 1
        for row in range(6, 12):
            for col in range(6):
                if label[col] == 1:
                    matrices[i, row, col] = 1
    return matrices



class AU_Graph_Mer_softmax(nn.Module):
    def __init__(self, device):
        super(AU_Graph_Mer_softmax, self).__init__()
        self.device = device
        self.upper_attention_matrix = torch.tensor([[0.4096, 0.3375, 0.0179, 0.2166, 0.0032, 0.0153],
                                                    [0.3551, 0.4027, 0.0049, 0.2246, 0.0021, 0.0106],
                                                    [0.0123, 0.0032, 0.6831, 0.0288, 0.0268, 0.2458],
                                                    [0.2614, 0.2576, 0.0505, 0.4173, 0.0024, 0.0108],
                                                    [0.0110, 0.0070, 0.1345, 0.0070, 0.7978, 0.0428],
                                                    [0.0155, 0.0102, 0.3628, 0.0091, 0.0126, 0.5898]]).float().to(
            device)

        self.lower_attention_matrix = torch.tensor([[0.7980, 0.0887, 0.0148, 0.0640, 0.0000, 0.0345],
                                                    [0.0311, 0.8235, 0.0675, 0.0381, 0.0121, 0.0277],
                                                    [0.0021, 0.0270, 0.9391, 0.0097, 0.0097, 0.0125],
                                                    [0.0151, 0.0255, 0.0162, 0.9085, 0.0081, 0.0267],
                                                    [0.0000, 0.0282, 0.0565, 0.0282, 0.7903, 0.0968],
                                                    [0.0122, 0.0280, 0.0315, 0.0402, 0.0420, 0.8462]]).float().to(
            device)

        self.global_attention_matrix = torch.tensor(
            [[0.3997, 0.3293, 0.0175, 0.2114, 0.0031, 0.0149, 0.0008, 0.0028, 0.0099,
              0.0039, 0.0017, 0.0051],
             [0.3456, 0.3920, 0.0047, 0.2186, 0.0021, 0.0103, 0.0000, 0.0024, 0.0121,
              0.0050, 0.0024, 0.0047],
             [0.0110, 0.0028, 0.6130, 0.0258, 0.0240, 0.2206, 0.0190, 0.0408, 0.0121,
              0.0128, 0.0048, 0.0132],
             [0.2530, 0.2493, 0.0489, 0.4040, 0.0024, 0.0104, 0.0017, 0.0054, 0.0077,
              0.0084, 0.0013, 0.0074],
             [0.0066, 0.0042, 0.0808, 0.0042, 0.4794, 0.0257, 0.0072, 0.0407, 0.3303,
              0.0090, 0.0054, 0.0066],
             [0.0141, 0.0093, 0.3288, 0.0082, 0.0114, 0.5345, 0.0074, 0.0406, 0.0162,
              0.0159, 0.0016, 0.0119],
             [0.0084, 0.0000, 0.2989, 0.0140, 0.0335, 0.0782, 0.4525, 0.0503, 0.0084,
              0.0363, 0.0000, 0.0196],
             [0.0094, 0.0075, 0.2156, 0.0151, 0.0640, 0.1441, 0.0169, 0.4482, 0.0367,
              0.0207, 0.0066, 0.0151],
             [0.0157, 0.0184, 0.0306, 0.0103, 0.2481, 0.0274, 0.0013, 0.0175, 0.6099,
              0.0063, 0.0063, 0.0081],
             [0.0131, 0.0159, 0.0675, 0.0235, 0.0141, 0.0563, 0.0122, 0.0206, 0.0131,
              0.7355, 0.0066, 0.0216],
             [0.0195, 0.0260, 0.0877, 0.0130, 0.0292, 0.0195, 0.0000, 0.0227, 0.0455,
              0.0227, 0.6364, 0.0779],
             [0.0237, 0.0211, 0.0976, 0.0290, 0.0145, 0.0594, 0.0092, 0.0211, 0.0237,
              0.0303, 0.0317, 0.6385]]).float().to(device)

        self.FeatureExtractor = FeatureExtractor()

        self.transformer = TransformerEncoderLayer(d_model=75, nhead=5)
        self.landmark_to_au1_2_4 = LandmarkToAU(m=69, n=6)
        self.landmark_to_au5_6_7 = LandmarkToAU(m=144, n=6)
        self.landmark_to_au9_10 = LandmarkToAU(m=156, n=4)
        self.landmark_to_au12_14_15_17 = LandmarkToAU(m=114, n=8)

        self.upper_gat = GAT_based_on_dynamic_prior_knowledge(nfeat=25, nhid=64, nclass=25, dropout=0.5, alpha=0.2,
                                                              beta=0.3
                                                              , nheads=3)
        # self.adj1 = torch.ones(6, 6).to(self.device)
        self.adj1 = torch.tensor([[1, 1, 1, 1, 0, 0],
                                  [1, 1, 1, 1, 0, 0],
                                  [1, 1, 1, 0, 0, 0],
                                  [1, 1, 0, 1, 0, 0],
                                  [0, 0, 0, 0, 1, 1],
                                  [0, 0, 0, 0, 1, 1]]).to(device)

        self.lower_gat = GAT_based_on_dynamic_prior_knowledge(nfeat=25, nhid=64, nclass=25, dropout=0.5, alpha=0.2,
                                                              beta=0.3, nheads=3)
        # self.adj2 = torch.ones(6, 6).to(self.device)

        self.global_gat = GAT_based_on_dynamic_prior_knowledge(nfeat=25, nhid=64, nclass=25, dropout=0.5, alpha=0.2,
                                                               beta=0.3, nheads=3)
        self.adj2 = torch.tensor([[1, 1, 0, 0, 0, 0, ],
                                  [1, 1, 0, 0, 0, 0],
                                  [0, 0, 1, 1, 0, 0],
                                  [0, 0, 1, 1, 0, 0],
                                  [0, 0, 0, 0, 1, 1],
                                  [0, 0, 0, 0, 1, 1]]).to(device)

        self.adj3 = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                                  [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                                  [0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1],
                                  [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                                  [0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0],
                                  [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                                  [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                                  [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                                  [1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0],
                                  [1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0],
                                  [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                                  [0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1]]).to(device)

        self.auxiliary_predict_fc = self._make_fc_layers(300, 12)
        self.predict_fc = self._make_fc_layers(300, 12)

        self.dropout = nn.Dropout(0.5)

    def _make_fc_layers(self, in_features, out_features):
        layers = []
        layers += [nn.Linear(in_features, 128),
                   nn.ReLU(inplace=True),
                   nn.Dropout(0.5)]
        layers += [nn.Linear(128, out_features),
                   ]
        return nn.Sequential(*layers)

    def forward(self, x, cur_partition):
        x_au1_2_4 = x[:, 0:69, :, :]
        x_au5_6_7 = x[:, 69:213, :, :]
        x_au9_10 = x[:, 213:369, :, :]
        x_au12_14_15_17 = x[:, 369:483, :, :]

        tmp_au1_2_4 = []
        for i in range(0, 69, 3):
            current_group = x_au1_2_4[:, i:i + 3, :, :]
            output = self.FeatureExtractor(current_group)
            tmp_au1_2_4.append(output)
        x_au1_2_4 = torch.cat(tmp_au1_2_4, dim=1)

        tmp_au5_6_7 = []
        for i in range(0, 144, 3):
            current_group = x_au5_6_7[:, i:i + 3, :, :]
            output = self.FeatureExtractor(current_group)
            tmp_au5_6_7.append(output)
        x_au5_6_7 = torch.cat(tmp_au5_6_7, dim=1)


        tmp_au9_10 = []
        for i in range(0, 156, 3):
            current_group = x_au9_10[:, i:i + 3, :, :]
            output = self.FeatureExtractor(current_group)
            tmp_au9_10.append(output)
        x_au9_10 = torch.cat(tmp_au9_10, dim=1)


        tmp_au12_14_15_17 = []
        for i in range(0, 114, 3):

            current_group = x_au12_14_15_17[:, i:i + 3, :, :]
            output = self.FeatureExtractor(current_group)
            tmp_au12_14_15_17.append(output)
        x_au12_14_15_17 = torch.cat(tmp_au12_14_15_17, dim=1)


        x_au1_2_4 = x_au1_2_4.view(x_au1_2_4.size(0), 23, -1)
        x_au5_6_7 = x_au5_6_7.view(x_au5_6_7.size(0), 48, -1)
        x_au9_10 = x_au9_10.view(x_au9_10.size(0), 52, -1)
        x_au12_14_15_17 = x_au12_14_15_17.view(x_au12_14_15_17.size(0), 38, -1)

        tmp_x = torch.cat((x_au1_2_4, x_au9_10[:, 23:, :], x_au12_14_15_17[:, 25:, :]), dim=1)
        tmp_x = self.transformer(tmp_x)
        x_au1_2_4 = tmp_x[:, :23, :]
        x_au5_6_7 = torch.cat((tmp_x[:, :22, :], tmp_x[:, 32:52, :], tmp_x[:, 59:, :]), dim=1)
        x_au9_10 = tmp_x[:, :52, :]
        x_au12_14_15_17 = tmp_x[:, 27:, :]

        x_au1_2_4 = x_au1_2_4.view(x_au1_2_4.size(0), 69, 5,
                                   5)
        x_au5_6_7 = x_au5_6_7.view(x_au5_6_7.size(0), 144, 5, 5)
        x_au9_10 = x_au9_10.view(x_au9_10.size(0), 156, 5, 5)
        x_au12_14_15_17 = x_au12_14_15_17.view(x_au12_14_15_17.size(0), 114, 5, 5)

        x_au1_2_4 = self.landmark_to_au1_2_4(x_au1_2_4)  # (batch_size, 3, 4, 4)
        x_au5_6_7 = self.landmark_to_au5_6_7(x_au5_6_7)  # (batch_size, 3, 4, 4)
        x_au9_10 = self.landmark_to_au9_10(x_au9_10)  # (batch_size, 2, 4, 4)
        x_au12_14_15_17 = self.landmark_to_au12_14_15_17(x_au12_14_15_17)  # (batch_size, 11, 4, 4)

        x_au1_2_4 = x_au1_2_4.view(x_au1_2_4.size(0), 3, -1)
        x_au5_6_7 = x_au5_6_7.view(x_au5_6_7.size(0), 3, -1)
        x_au9_10 = x_au9_10.view(x_au9_10.size(0), 2, -1)
        x_au12_14_15_17 = x_au12_14_15_17.view(x_au12_14_15_17.size(0), 4, -1)

        x_au1_2_4 = F.normalize(x_au1_2_4, p=2, dim=2)
        x_au5_6_7 = F.normalize(x_au5_6_7, p=2, dim=2)
        x_au9_10 = F.normalize(x_au9_10, p=2, dim=2)
        x_au12_14_15_17 = F.normalize(x_au12_14_15_17, p=2, dim=2)

        origin_global_au_feature = torch.cat((x_au1_2_4, x_au5_6_7, x_au9_10, x_au12_14_15_17), dim=1)
        auxiliary_output = self.auxiliary_predict_fc(
            origin_global_au_feature.view(origin_global_au_feature.size(0), -1))
        auxiliary_predict_output = auxiliary_output

        # upper_gat_output = self.upper_gat(torch.cat((x_au1_2_4, x_au5_6_7), dim=1), self.adj1.repeat(x.shape[0], 1, 1),
        #                                   self.upper_attention_matrix, cur_partition) + torch.cat((x_au1_2_4, x_au5_6_7), dim=1)
        # lower_gat_output = self.lower_gat(torch.cat((x_au9_10, x_au12_14_15_17), dim=1),
        #                                   self.adj2.repeat(x.shape[0], 1, 1), self.lower_attention_matrix,
        #                                   cur_partition) + torch.cat((x_au9_10, x_au12_14_15_17), dim=1)
        upper_gat_output = self.upper_gat(torch.cat((x_au1_2_4, x_au5_6_7), dim=1), self.adj1.repeat(x.shape[0], 1, 1),
                                          self.upper_attention_matrix, cur_partition) + torch.cat(
            (x_au1_2_4, x_au5_6_7), dim=1)
        lower_gat_output = self.lower_gat(torch.cat((x_au9_10, x_au12_14_15_17), dim=1),
                                          self.adj2.repeat(x.shape[0], 1, 1), self.lower_attention_matrix,
                                          cur_partition) + torch.cat((x_au9_10, x_au12_14_15_17), dim=1)
        global_au_feature = torch.cat((upper_gat_output, lower_gat_output), dim=1)


        # global_adj = construct_adjacency_matrix(auxiliary_output, self.adj3.repeat(auxiliary_output.shape[0], 1, 1))
        global_adj = self.adj3.repeat(auxiliary_output.shape[0], 1, 1)
        tmp = global_au_feature
        global_au_feature = self.global_gat(global_au_feature, global_adj.to(self.device), self.global_attention_matrix,
                                            cur_partition)
        # predict_au = self.predict_fc(torch.reshape(global_au_feature, (global_au_feature.size(0), -1)) + torch.reshape(origin_global_au_feature, (origin_global_au_feature.size(0), -1)))
        predict_au = self.predict_fc(
            torch.reshape(global_au_feature, (global_au_feature.size(0), -1)) + torch.reshape(tmp, (tmp.size(0), -1)))
        return auxiliary_predict_output.float(), predict_au.float(), torch.reshape(global_au_feature, (
        global_au_feature.size(0), 12, 5, 5)) + torch.reshape(tmp, (tmp.size(0), 12, 5, 5))


