import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, out_dim=64):
        super(ContrastiveEncoder, self).__init__()
        
        # 1. Encoder (Backbone): 负责进一步提取特征
        # 使用 BatchNorm 增加训练稳定性，防止坍塌
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
        # 2. Projector (Projection Head): 映射到对比空间
        # 按照 SimCLR 的经验，Projector 应该是非线性的
        self.projector = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Kaiming 初始化适合 ReLU 网络
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # 返回用于聚类的特征 h
        h = self.encoder(x)
        return h

    def forward_cl(self, x):
        # 返回用于计算对比损失的 z
        h = self.encoder(x)
        z = self.projector(h)
        # 归一化是对比学习的关键，把特征映射到单位超球面上
        return F.normalize(z, dim=1)