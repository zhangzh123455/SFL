import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT_SimGCL_Server(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, n_heads=4, dropout=0.6):
        super(GAT_SimGCL_Server, self).__init__()
        
        # GAT 编码器
        # 第一层: 多头注意力
        self.gat1 = GATConv(input_dim, hidden_dim, heads=n_heads, dropout=dropout)
        # 第二层: 聚合头，输出最终嵌入
        self.gat2 = GATConv(hidden_dim * n_heads, out_dim, heads=1, concat=False, dropout=dropout)
        
        self.dropout = dropout
        self.act = nn.ELU() # GAT 标配 ELU

    def forward(self, x, edge_index):
        # 1. 第一层 GAT
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = self.act(x)
        
        # 2. 第二层 GAT
        x = F.dropout(x, p=self.dropout, training=self.training)
        z = self.gat2(x, edge_index)
        
        return z

    def forward_cl(self, x, edge_index, lambda1, lambda2):
        """
        SimGCL 专用前向传播: 
        计算一次干净的 Z，然后加两次噪声得到 Z1, Z2
        """
        z = self.forward(x, edge_index)
        
        # [SimGCL 核心]: 在 Embedding 上加均匀分布噪声
        sigma = z.std()

        z1 = z + lambda1 * sigma * torch.randn_like(z)
        z2 = z + lambda2 * sigma * torch.randn_like(z)
        # noise1 = (torch.rand_like(z) * 2 - 1) * 0.1
        # noise2 = (torch.rand_like(z) * 2 - 1) * 0.1
        
        # z1 = z + noise1
        # z2 = z + noise2
        
        # 归一化用于计算余弦相似度
        return F.normalize(z1, dim=1), F.normalize(z2, dim=1)