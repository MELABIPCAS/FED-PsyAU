import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from model.dfme_baseline import *
from model.InceptionNet import *
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
        return out, attention

class au_graph_mer(nn.Module):
    def __init__(self, au_model_name, device, out_channels=7):
        super(au_graph_mer, self).__init__()
        self.au_model_name = au_model_name
        self.au_model = AU_Graph_Mer_softmax(device).to(device)

        self.mer_feature_extractor = InceptionNet(num_classes=out_channels).to(device)



        # if pretrained:
        #     self.au_model.load_state_dict(torch.load(f'../experiment_result/dfme/weight/AU_MTL_Network_prior_knowledge_softmax/{fold_num}.pth', map_location=device))
            # for param in self.au_model.parameters():
            #     param.requires_grad = False

            # for name, param in self.au_model.named_parameters():
            #     # if 'predict_fc' not in name:
            #     param.requires_grad = False


    def forward(self, global_features, au_inputs, cur_partition):
        if self.au_model_name in ['au_graph_mer_softmax', 'au_graph_mer_ratio']:
            auxiliary_predicts, au_predicts, au_features = self.au_model(au_inputs, cur_partition)
        else:
            auxiliary_predicts, au_predicts, au_features = self.au_model(au_inputs)
        emo_predicts = self.mer_feature_extractor(global_features, au_features)


        return emo_predicts, au_predicts, auxiliary_predicts


