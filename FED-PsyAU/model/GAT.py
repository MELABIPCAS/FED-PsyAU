import torch
import torch.nn as nn
import torch.nn.functional as F

from model.gat_layers import GraphAttentionLayer, SpGraphAttentionLayer, GraphAttentionLayer_prior_attention, \
    GraphAttentionLayer_based_on_dynamic_prior_knowledge, \
    GraphAttentionLayer_based_on_no_softmax_dynamic_prior_knowledge


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat,
                                                 nhid,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads,
                                             nclass,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

#
# class GAT(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
#         super(GAT, self).__init__()
#         self.dropout = dropout
#         self.attentions = nn.ModuleList([GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)])
#         self.attentions_h = nn.ModuleList([GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)])
#         self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
#
#     def forward(self, x, adj):
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
#         x = torch.cat([att(x, adj) for att in self.attentions_h], dim=2)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = F.elu(self.out_att(x, adj))
#         return x

class GAT_prior_attention(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, prior_attention):
        super(GAT_prior_attention, self).__init__()
        self.dropout = dropout
        self.attentions = nn.ModuleList([GraphAttentionLayer_prior_attention(nfeat, nhid, dropout=dropout, alpha=alpha, prior_attention=prior_attention, concat=True) for _ in range(nheads)])
        self.attentions_h = nn.ModuleList([GraphAttentionLayer_prior_attention(nhid * nheads, nhid, dropout=dropout, alpha=alpha, prior_attention=prior_attention, concat=True) for _ in range(nheads)])
        self.out_att = GraphAttentionLayer_prior_attention(nhid * nheads, nclass, dropout=dropout, alpha=alpha, prior_attention=prior_attention, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = torch.cat([att(x, adj) for att in self.attentions_h], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x



class GAT_based_on_dynamic_prior_knowledge(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, beta):
        super(GAT_based_on_dynamic_prior_knowledge, self).__init__()
        self.dropout = dropout
        self.beta = beta
        self.attentions = [GraphAttentionLayer_based_on_dynamic_prior_knowledge(nfeat, nhid, dropout=dropout, alpha=alpha, beta=self.beta, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer_based_on_dynamic_prior_knowledge(nhid * nheads, nclass, dropout=dropout, alpha=alpha, beta=self.beta, concat=False)

    def forward(self, x, adj, prior_attention, cur_partition):

        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj, prior_attention, cur_partition) for att in self.attentions], dim=2)
        # x = torch.cat([att(x, adj, prior_attention, cur_partition) for att in self.attentions_h], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj, prior_attention, cur_partition))
        return x


class GAT_based_on_dynamic_prior_knowledge_no_softmax(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, beta):

        super(GAT_based_on_dynamic_prior_knowledge_no_softmax, self).__init__()
        self.dropout = dropout
        self.beta = beta
        self.attentions = [GraphAttentionLayer_based_on_no_softmax_dynamic_prior_knowledge(nfeat, nhid, dropout=dropout, alpha=alpha, beta=self.beta, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer_based_on_no_softmax_dynamic_prior_knowledge(nhid * nheads, nclass, dropout=dropout, alpha=alpha, beta=self.beta, concat=False)

    def forward(self, x, adj, prior_attention, cur_partition):

        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj, prior_attention, cur_partition) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj, prior_attention, cur_partition))
        return x

