# GAT版本
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from gat import GAT_SimGCL_Server
from utils import clustering_metrics, construct_graph # [新增]
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
        
        self.model = GAT_SimGCL_Server(
            input_dim=real_input_dim,     # 这里如果是 1433，GAT 第一层就会处理它
            hidden_dim=256,          # GAT 内部隐层保持 256
            out_dim=64,
            n_heads=4,
            dropout=0.5
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=1e-3, # GAT 可能需要小一点的学习率，或者 5e-3
            weight_decay=5e-4
        )
        
        # 存储重构的邻接矩阵 A' (edge_index)
        self.reconstructed_adj = None 

    def contrastive_loss(self, z1, z2, temperature=0.5):
        # InfoNCE Loss
        sim_matrix = torch.mm(z1, z2.t()) / temperature
        labels = torch.arange(z1.size(0)).to(self.device)
        loss = nn.CrossEntropyLoss()(sim_matrix, labels)
        return loss

    def reconstruct_structure(self, x_prop):
        """
        阶段 2.1: 在服务器端基于特征重构图结构
        """
        print("[Server] Reconstructing Graph Structure (k-NN)...")
        # 这里的 k 是关键超参，通常 10-20
        # 如果是 Cora，k=10 比较稳；如果是 PubMed，可能需要大一点
        k = 5 
        edge_index = construct_graph(x_prop, k=k, metric='cosine')
        self.reconstructed_adj = edge_index.to(self.device)
        print(f"         -> Graph Reconstructed! Edges: {edge_index.shape[1]}")

    def train_epoch(self, x_prop, tau, lambda1, lambda2):
        self.model.train()
        x_prop = x_prop.to(self.device)
        
        # 确保图已经重构
        if self.reconstructed_adj is None:
            self.reconstruct_structure(x_prop)
            
        # SimGCL 前向: 传入重构的 A'
        # eps 是噪声强度，SimGCL 推荐 0.1
        z1, z2 = self.model.forward_cl(x_prop, self.reconstructed_adj, lambda1, lambda2)
        
        loss = self.contrastive_loss(z1, z2, tau)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def evaluate(self, x_prop, y_true):
        self.model.eval()
        x_prop = x_prop.to(self.device)
        
        # 确保图已经重构
        if self.reconstructed_adj is None:
            self.reconstruct_structure(x_prop)
            
        with torch.no_grad():
            h = self.model(x_prop, self.reconstructed_adj)
            
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