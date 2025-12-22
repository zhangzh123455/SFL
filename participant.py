import torch
from utils import get_normalized_adj
from sklearn.decomposition import PCA

class Client:
    """
    分布式系统中的参与方 (Participant/Client)
    职责:
    1. 持有本地数据 (原始特征 X, 拓扑结构 A)
    2. 执行随机投影 (降维 & 隐私保护)
    3. 执行 GCNII 线性传播 (预计算)
    """
    def __init__(self, data, args):
        self.args = args
        self.device = args.device
        self.x = data.x.cpu() 
        self.edge_index = data.edge_index.cpu()
        self.num_nodes = data.num_nodes
        
        # [修改] 只有在 use_pca=True 时才做降维
        if getattr(args, 'use_pca', True):
            print(f"[Client] Fitting PCA ({data.num_features} -> {args.hidden_dim})...")
            pca = PCA(n_components=args.hidden_dim)
            self.x_reduced = torch.from_numpy(pca.fit_transform(self.x.numpy())).float().to(self.device)
        else:
            print(f"[Client] PCA Disabled. Using raw features ({data.num_features})...")
            self.x_reduced = self.x.float().to(self.device) # 直接用原始特征
            
        # 归一化邻接矩阵
        self.adj = get_normalized_adj(self.edge_index, self.num_nodes).to(self.device)
        
        
    def prepare_data(self):
        """
        执行阶段一: 预计算与传播
        返回:
            x_prop: 聚合了结构信息和初始残差的特征矩阵 [N, hidden_dim]
                    这将作为一次性传输的数据发送给服务器
        """
        print(f"[Client] Starting Random Projection & Propagation ({self.args.k_hops} hops)...")
        
        # -------------------------------------------------------
        # Step 1: 随机投影 (降维)
        # -------------------------------------------------------
        # X (N, D) @ W (D, 64) -> H (N, 64)
        # 这一步去除了原始特征的显式语义，起到隐私保护作用
        # h = self.x @ self.projection_matrix
        
        # 这个投影后的特征即为 GCNII 公式中的 H^(0) (初始残差源)
        h = self.x_reduced
        h_0 = h.clone()
        
        # -------------------------------------------------------
        # Step 2: 迭代结构传播 (GCNII Linear Propagation)
        # -------------------------------------------------------
        # 核心公式: H^(k+1) = (1-alpha) * P * H^(k) + alpha * H^(0)
        #
        
        for k in range(self.args.k_hops):
            # 稀疏矩阵乘法: (N, N) @ (N, 64) -> (N, 64)
            # torch.sparse.mm 在 PyTorch 1.9+ 中对稀疏 @ 稠密非常高效
            propagated = torch.sparse.mm(self.adj, h)
            
            # 融合结构信息 (propagated) 和 初始信息 (h_0)
            h = (1 - self.args.alpha) * propagated + self.args.alpha * h_0
            
        print(f"[Client] Propagation finished. Data shape: {h.shape}")
        
        # 返回最终结果，准备发送给服务器
        return h