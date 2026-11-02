"""按表情类别做注意力加权，再统一分类。"""

import torch
import torch.nn as nn


class ESAHead(nn.Module):
    """单个类别的注意力门控和二分类头。"""

    def __init__(self, feature_dim=512, dropout=0.5):
        super().__init__()
        self.attention = nn.Linear(feature_dim, feature_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

    def forward(self, features):
        weighted = features * torch.sigmoid(self.attention(features))
        return self.classifier(weighted.view(weighted.size(0), -1)), weighted


class ESA(nn.Module):
    """每个表情一个类别头，加权特征求和后过全局分类器。"""

    def __init__(self, feature_dim=512, num_classes=7, dropout=0.5):
        super().__init__()
        self.heads = nn.ModuleList([ESAHead(feature_dim, dropout) for _ in range(num_classes)])
        self.global_classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, f_shared):
        category_logits, weighted_features = [], []
        for head in self.heads:
            logits, weighted = head(f_shared)
            category_logits.append(logits)
            weighted_features.append(weighted)
        aggregated = torch.stack(weighted_features, dim=0).sum(dim=0)
        return self.global_classifier(aggregated), category_logits
