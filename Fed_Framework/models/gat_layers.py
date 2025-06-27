import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.permute(0, 2, 1)
        return self.leakyrelu(e)


class GraphAttentionLayer_prior_attention(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, prior_attention, concat=True):
        super(GraphAttentionLayer_prior_attention, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.prior_attention = prior_attention

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = attention * self.prior_attention
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.permute(0, 2, 1)
        return self.leakyrelu(e)


class GraphAttentionLayer_based_on_dynamic_prior_knowledge(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, beta=0.3, concat=True):
        super(GraphAttentionLayer_based_on_dynamic_prior_knowledge, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.beta = beta

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))

        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj, prior_attention, cur_partition):

        W_repeated = self.W.repeat(h.shape[0], 1, 1)
        prior_attention_repeated = prior_attention.repeat(h.shape[0], 1, 1).float()
        Wh = torch.bmm(h, W_repeated)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = torch.zeros_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        prior_attention_repeated = torch.where(adj > 0, prior_attention_repeated, zero_vec)
        prior_attention_repeated = F.softmax(prior_attention_repeated, dim=2)
        prior_attention_repeated = torch.where(adj > 0, prior_attention_repeated, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = attention * (1 - self.beta * cur_partition) + prior_attention_repeated * self.beta * cur_partition
        attention = F.normalize(attention, p=1, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):

        a_repeated = self.a.repeat(Wh.shape[0], 1, 1)
        Wh1 = torch.matmul(Wh, a_repeated[:, 0:self.out_features, :])
        Wh2 = torch.matmul(Wh, a_repeated[:, self.out_features:, :])
        e = Wh1 + Wh2.permute(0, 2, 1)

        return self.leakyrelu(e)

    def __repr__(self):

        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphAttentionLayer_based_on_no_softmax_dynamic_prior_knowledge(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, beta=0.3, concat=True):
        super(GraphAttentionLayer_based_on_no_softmax_dynamic_prior_knowledge, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.beta = beta

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))

        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj, prior_attention, cur_partition):

        W_repeated = self.W.repeat(h.shape[0], 1, 1)
        prior_attention_repeated = prior_attention.repeat(h.shape[0], 1, 1).float()
        Wh = torch.bmm(h, W_repeated)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = torch.zeros_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        prior_attention_repeated = torch.where(adj > 0, prior_attention_repeated, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = attention * (1 - self.beta * cur_partition) + prior_attention_repeated * self.beta * cur_partition
        attention = F.normalize(attention, p=1, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):

        a_repeated = self.a.repeat(Wh.shape[0], 1, 1)
        Wh1 = torch.matmul(Wh, a_repeated[:, 0:self.out_features, :])
        Wh2 = torch.matmul(Wh, a_repeated[:, self.out_features:, :])
        e = Wh1 + Wh2.permute(0, 2, 1)
        return self.leakyrelu(e)

    def __repr__(self):

        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'






