"""FED-PsyAU 主网络：SPAT -> AFE -> DPK-GAT -> DSI -> ESA。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.afe import AFE
from models.dpk_gat import AU_NAMES, DPKGAT
from models.dsi import DSI
from models.esa import ESA
from models.spat import SPAT


class FEDPsyAU(nn.Module):
    def __init__(self, num_classes=7, num_au=len(AU_NAMES), au_dim=64, shared_dim=512, dropout=0.5):
        super().__init__()
        self.num_au = num_au
        self.spat = SPAT()
        self.afe = AFE(num_au=num_au, au_dim=au_dim)
        self.dpk_gat = DPKGAT(feature_dim=au_dim)
        self.dsi = DSI(num_au=num_au, shared_dim=shared_dim, dropout=dropout)
        self.esa = ESA(feature_dim=shared_dim, num_classes=num_classes, dropout=dropout)

        # AFE 和 GAT 两路都接 AU 预测头
        self.afe_au_head = nn.Linear(au_dim, 1)
        self.gat_au_head = nn.Sequential(
            nn.Linear(num_au * au_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_au),
        )

    def forward(self, global_flow, roi_inputs, points):
        z = self.spat(roi_inputs, points)                       # (B, 65, 75)
        au_features = F.normalize(self.afe(z), p=2, dim=-1)     # (B, 12, 64)
        h_local, h_global = self.dpk_gat(au_features)           # (B, 12, 64)

        f_shared = self.dsi(h_global, global_flow)              # (B, 512)
        main_logits, category_logits = self.esa(f_shared)

        au_logits_afe = self.afe_au_head(au_features).squeeze(-1)
        au_logits_gat = self.gat_au_head(h_global.view(h_global.size(0), -1))
        return main_logits, category_logits, au_logits_afe, au_logits_gat


def mer_loss(outputs, emo_labels, au_labels, alpha1=1.0, alpha2=0.5, alpha3=0.5):
    """L_MER = alpha1*L_main + alpha2*L_A + alpha3*L_specific。"""
    main_logits, category_logits, au_logits_afe, au_logits_gat = outputs

    l_main = F.cross_entropy(main_logits, emo_labels)
    l_au = (F.binary_cross_entropy_with_logits(au_logits_afe, au_labels)
            + F.binary_cross_entropy_with_logits(au_logits_gat, au_labels))
    specific_logits = torch.cat([logit[:, 1].unsqueeze(1) for logit in category_logits], dim=1)
    l_specific = F.cross_entropy(specific_logits, emo_labels)

    total = alpha1 * l_main + alpha2 * l_au + alpha3 * l_specific
    return total, (l_main.item(), l_au.item(), l_specific.item())
