import numpy as np
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
    """
    一个简单的图注意力层，类似于https://arxiv.org/abs/1710.10903中的描述。

    参数:
    - in_features: 输入特征的数量。
    - out_features: 输出特征的数量。
    - dropout: Dropout比例，用于防止过拟合。
    - alpha: LeakyReLU激活函数的斜率。
    - concat: 是否在最后连接特征，True表示连接，False表示不连接。
    """

    def __init__(self, in_features, out_features, dropout, alpha, beta=0.3, concat=True):
        super(GraphAttentionLayer_based_on_dynamic_prior_knowledge, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.beta = beta

        # 初始化权重参数
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        """
        用于对权重张量进行 Xavier 初始化，
        这是一种在神经网络中初始化权重的常见方法。
        它的主要作用是帮助网络在训练时获得更好的收敛性能，
        通过控制权重的初始分布来平衡梯度爆炸和梯度消失的问题。
        """
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj, prior_attention, cur_partition):
        """
        前向传播过程。

        参数:
        - h: 输入节点的特征矩阵。
        - adj: 邻接矩阵，表示图的结构。

        返回:
        - 如果concat为True，返回经过ELU激活函数处理的连接后的特征。
        - 如果concat为False，返回变换后的特征。
        """
        W_repeated = self.W.repeat(h.shape[0], 1, 1)
        prior_attention_repeated = prior_attention.repeat(h.shape[0], 1, 1).float()
        Wh = torch.bmm(h, W_repeated)  # 计算变换后的特征
        e = self._prepare_attentional_mechanism_input(Wh)  # 准备注意力机制的输入

        # 通过邻接矩阵调整注意力分数
        zero_vec = torch.zeros_like(e)
        attention = torch.where(adj > 0, e, zero_vec)#对于邻接矩阵大于0的位置，用注意力矩阵中对应位置的注意力分数替换 否则替换0
        prior_attention_repeated = torch.where(adj > 0, prior_attention_repeated, zero_vec)
        prior_attention_repeated = F.softmax(prior_attention_repeated, dim=2)
        prior_attention_repeated = torch.where(adj > 0, prior_attention_repeated, zero_vec)
        attention = F.softmax(attention, dim=2)  # 应用softmax激活函数，每一行的非零（-9e15）元素对应节点到当前节点的权重
        attention = attention * (1 - self.beta * cur_partition) + prior_attention_repeated * self.beta * cur_partition
        attention = F.normalize(attention, p=1, dim=2) #归一化
        attention = F.dropout(attention, self.dropout, training=self.training)  # 应用dropout 随机将一些权重置为0
        h_prime = torch.matmul(attention, Wh)  # 根据注意力权重计算加权特征

        if self.concat:
            return F.elu(h_prime)  # 如果concat为True，应用ELU激活函数
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        """
        准备注意力机制的输入。

        参数:
        - Wh: 变换后的特征矩阵。


        返回:
        - 处理后的输入，用于计算注意力分数。
        """
        a_repeated = self.a.repeat(Wh.shape[0], 1, 1)
        # Wh1 为 Wh 矩阵与 self.a 的前 self.out_features 行相乘的结果
        Wh1 = torch.matmul(Wh, a_repeated[:, 0:self.out_features, :])
        # Wh2 为 Wh 矩阵与 self.a 的后 self.out_features 行相乘的结果
        Wh2 = torch.matmul(Wh, a_repeated[:, self.out_features:, :])
        e = Wh1 + Wh2.permute(0, 2, 1)
        # 计算注意力分数 得到一个批量大小*批量大小的重要性矩阵 eij代表第j个节点对于第i个节点的重要性
        return self.leakyrelu(e)  # 应用LeakyReLU激活函数

    def __repr__(self):
        """
        返回模块的字符串表示。
        """
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphAttentionLayer_based_on_no_softmax_dynamic_prior_knowledge(nn.Module):
    """
    一个简单的图注意力层，类似于https://arxiv.org/abs/1710.10903中的描述。

    参数:
    - in_features: 输入特征的数量。
    - out_features: 输出特征的数量。
    - dropout: Dropout比例，用于防止过拟合。
    - alpha: LeakyReLU激活函数的斜率。
    - concat: 是否在最后连接特征，True表示连接，False表示不连接。
    """

    def __init__(self, in_features, out_features, dropout, alpha, beta=0.3, concat=True):
        super(GraphAttentionLayer_based_on_no_softmax_dynamic_prior_knowledge, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.beta = beta

        # 初始化权重参数
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        """
        用于对权重张量进行 Xavier 初始化，
        这是一种在神经网络中初始化权重的常见方法。
        它的主要作用是帮助网络在训练时获得更好的收敛性能，
        通过控制权重的初始分布来平衡梯度爆炸和梯度消失的问题。
        """
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj, prior_attention, cur_partition):
        """
        前向传播过程。

        参数:
        - h: 输入节点的特征矩阵。
        - adj: 邻接矩阵，表示图的结构。

        返回:
        - 如果concat为True，返回经过ELU激活函数处理的连接后的特征。
        - 如果concat为False，返回变换后的特征。
        """
        W_repeated = self.W.repeat(h.shape[0], 1, 1)
        prior_attention_repeated = prior_attention.repeat(h.shape[0], 1, 1).float()
        Wh = torch.bmm(h, W_repeated)  # 计算变换后的特征
        e = self._prepare_attentional_mechanism_input(Wh)  # 准备注意力机制的输入

        # 通过邻接矩阵调整注意力分数
        zero_vec = torch.zeros_like(e)
        attention = torch.where(adj > 0, e, zero_vec)#对于邻接矩阵大于0的位置，用注意力矩阵中对应位置的注意力分数替换 否则替换0
        prior_attention_repeated = torch.where(adj > 0, prior_attention_repeated, zero_vec)
        attention = F.softmax(attention, dim=2)  # 应用softmax激活函数，每一行的非零（-9e15）元素对应节点到当前节点的权重
        attention = attention * (1 - self.beta * cur_partition) + prior_attention_repeated * self.beta * cur_partition
        attention = F.normalize(attention, p=1, dim=2) #归一化
        attention = F.dropout(attention, self.dropout, training=self.training)  # 应用dropout 随机将一些权重置为0
        h_prime = torch.matmul(attention, Wh)  # 根据注意力权重计算加权特征

        if self.concat:
            return F.elu(h_prime)  # 如果concat为True，应用ELU激活函数
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        """
        准备注意力机制的输入。

        参数:
        - Wh: 变换后的特征矩阵。


        返回:
        - 处理后的输入，用于计算注意力分数。
        """
        a_repeated = self.a.repeat(Wh.shape[0], 1, 1)
        # Wh1 为 Wh 矩阵与 self.a 的前 self.out_features 行相乘的结果
        Wh1 = torch.matmul(Wh, a_repeated[:, 0:self.out_features, :])
        # Wh2 为 Wh 矩阵与 self.a 的后 self.out_features 行相乘的结果
        Wh2 = torch.matmul(Wh, a_repeated[:, self.out_features:, :])
        e = Wh1 + Wh2.permute(0, 2, 1)
        # 计算注意力分数 得到一个批量大小*批量大小的重要性矩阵 eij代表第j个节点对于第i个节点的重要性
        return self.leakyrelu(e)  # 应用LeakyReLU激活函数

    def __repr__(self):
        """
        返回模块的字符串表示。
        """
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """
    一个专门用于稀疏区域反向传播层的特殊函数。

    参数:
    - ctx: 上下文对象，用于存储用于后续计算的信息。
    - indices: 稀疏矩阵的索引，一个2维tensor。
    - values: 稀疏矩阵的值，一个1维tensor。
    - shape: 稀疏矩阵的形状，一个1维tensor。
    - b: 一个 dense matrix。

    返回值:
    - torch.matmul(a, b) 的结果，其中 a 是根据 indices 和 values 构建的稀疏矩阵。
    """

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        # 确保 indices 不需要梯度
        assert indices.requires_grad == False
        # 构建稀疏矩阵 a
        a = torch.sparse_coo_tensor(indices, values, shape)
        # 保存 a 和 b 以备后用     ctx.save_for_backward(a, b)
        ctx.save_for_backward(a, b)
        # 存储矩阵 a 的行数
        ctx.N = shape[0]
        # 返回 a 和 b 的乘积
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        # 从上下文对象中恢复 a 和 b
        a, b = ctx.saved_tensors
        # 初始化梯度
        grad_values = grad_b = None
        # 如果需要 gradients for values
        if ctx.needs_input_grad[1]:
            # 计算 grad_a_dense
            grad_a_dense = grad_output.matmul(b.t())
            # 根据稀疏矩阵 a 的索引计算对应的梯度
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        # 如果需要 gradients for b
        if ctx.needs_input_grad[3]:
            # 计算 grad_b
            grad_b = a.t().matmul(grad_output)
        # 返回梯度，对于不需要梯度的输入返回 None
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    #self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
      def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

    
class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'
        """
        假设  input = tensor([[0.1, 0.2, 0.3, 0.4, 0.5],
                  [0.6, 0.7, 0.8, 0.9, 1.0],
                  [1.1, 1.2, 1.3, 1.4, 1.5]])
             adj = tensor([[0.3, 0.5, 0.2],  # 节点0与自身、节点1和节点2分别有连接
              [0.0, 0.6, 0.4],  # 节点1与自身和节点2有连接
              [0.1, 0.3, 0.6]]) # 节点2与自身、节点0和节点1分别有连接
             则 N = 3 
               edge = tensor([[0, 0, 0, 1, 1, 2, 2, 2],
                            [0, 1, 2, 1, 2, 0, 1, 2]])

        """
        # 获取输入矩阵的行数，即图中节点的数量
        N = input.size()[0]

        # 找到邻接矩阵中非零元素的位置，并转换为行-列的形式
        edge = adj.nonzero().t()

        # 将输入节点特征与权重矩阵相乘，进行特征变换
        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # 把存在边的节点拿出来，拼接对应的激活特征向量h（上下拼接），得到2*D x E
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E
        # 计算得到每条边的注意力分数 tensor([1.8233, 2.2841, 2.8614, 1.5830, 1.9831, 0.6426, 1.0971, 1.3744])
        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E
        # 计算每个节点所连接的边的重要性之和
        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E
        # 对激活特征做基于连接边重要性的融合
        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
