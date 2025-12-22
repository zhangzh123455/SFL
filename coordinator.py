import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.cluster import KMeans
from model import ContrastiveEncoder
from utils import clustering_metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

class Server:
    def __init__(self, args):
        self.args = args
        self.device = args.device

        # [修改] 使用 server_input_dim 作为输入维度
        # 如果 config 里没定义这个变量，就回退逻辑判断
        if getattr(args, 'use_pca', False) is False:
            real_input_dim = args.input_dim # Cora 原始维度
        else:
            real_input_dim = args.hidden_dim
        
        print(f"[Server] Model Input Dim: {real_input_dim}")
        # 初始化对比学习模型
        self.model = ContrastiveEncoder(
            input_dim=real_input_dim, # 输入
            hidden_dim=256,            # 中间层升维提取特征
            out_dim=64                 # 输出特征维度
        ).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=1e-3, 
            weight_decay=1e-4
        )
        
        print(f"[Server] Contrastive Encoder initialized.")

    def contrastive_loss(self, z1, z2, temperature=0.5):
        """
        InfoNCE Loss (SimCLR style)
        z1, z2: 两个视图的特征，形状 [N, D]
        """
        # 计算相似度矩阵: N x N
        # sim[i, j] = z1[i] * z2[j]
        sim_matrix = torch.mm(z1, z2.t()) / temperature
        
        # 正样本: 对角线元素 (z1[i] 和 z2[i])
        # labels: [0, 1, 2, ..., N-1]
        labels = torch.arange(z1.size(0)).to(self.device)
        
        # CrossEntropyLoss 会自动做 Softmax
        # 即使只用一半的 Loss (z1->z2) 效果也很好，为了对称通常算两边
        loss = nn.CrossEntropyLoss()(sim_matrix, labels)
        # 计算 z1 -> z2 的损失
        loss1 = nn.CrossEntropyLoss()(sim_matrix, labels)

        # 计算 z2 -> z1 的损失 (转置相似度矩阵)
        # sim_matrix.t() 就是 z2 * z1.t()
        loss2 = nn.CrossEntropyLoss()(sim_matrix.t(), labels)

        # 总损失取平均
        loss = (loss1 + loss2) / 2
        return loss

    def train_epoch(self, x_prop, tau, lambda1, lambda2):
        self.model.train()
        x_prop = x_prop.to(self.device)
        
        # ===============================
        # 1. 数据增强 (Feature Noise)
        # ===============================
        # 由于没有图结构，我们在特征上加高斯噪声生成两个视图
        # 噪声强度 0.1 是个经验值
        # noise1 = torch.randn_like(x_prop) * 0.1
        # noise2 = torch.randn_like(x_prop) * 0.1
        
        # view1 = x_prop + noise1
        # view2 = x_prop + noise2
        sigma = x_prop.std()

        view1 = x_prop + lambda1 * sigma * torch.randn_like(x_prop)
        view2 = x_prop + lambda2  * sigma * torch.randn_like(x_prop)

        
        # ===============================
        # 2. 前向传播
        # ===============================
        z1 = self.model.forward_cl(view1)
        z2 = self.model.forward_cl(view2)
        
        # ===============================
        # 3. 计算 InfoNCE Loss
        # ===============================
        loss = self.contrastive_loss(z1, z2, temperature=tau)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def evaluate(self, x_prop, y_true):
        self.model.eval()
        x_prop = x_prop.to(self.device)
            
        with torch.no_grad():
            # 获取 Backbone 提取的特征 h
            h = self.model(x_prop)
            
        h_np = h.cpu().numpy()
        
        # 1. K-Means 聚类 (计算 NMI, ARI)
        kmeans = KMeans(n_clusters=self.args.n_clusters, n_init=20, random_state=self.args.seed)
        y_pred_cluster = kmeans.fit_predict(h_np)
        nmi, ari = clustering_metrics(y_true, y_pred_cluster)
        
        # 2. 逻辑回归分类 (计算 ACC, F1)
        X_train, X_test, y_train, y_test = train_test_split(
            h_np, y_true, test_size=0.2, random_state=self.args.seed
        )
        
        clf = LogisticRegression(solver='lbfgs', max_iter=2000) # max_iter 调大点防止不收敛
        clf.fit(X_train, y_train)
        
        # 预测标签
        y_pred_class = clf.predict(X_test)
        
        # 计算 ACC
        acc = clf.score(X_test, y_test)
        
        # [新增] 计算 Macro-F1
        f1 = f1_score(y_test, y_pred_class, average='macro')

        # 返回 4 个指标
        return nmi, ari, acc, f1
