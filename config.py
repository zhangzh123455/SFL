import torch

class Config:
    def __init__(self, dataset_name='PubMed'):
        self.dataset = dataset_name
        
        # --- 基础路径配置 ---
        self.data_root = './data'  # 数据存放根目录
        self.device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
        self.seed = 42

        # --- 默认超参数 (General Defaults) ---
        self.hidden_dim = 256    # 投影维度 / 嵌入维度
        self.lr = 1e-3          
        self.weight_decay = 1e-5 # 稍微加一点正则
        self.epochs = 1000       # 训练轮数
        
        # APPNP 参数 (客户端传播)
        self.alpha = 0.1         # 初始残差强度
        self.k_hops = 10         # 默认跳数设为 10 (适用于大多数稠密图)
        
        # 对比学习参数
        self.tau = 0.4           # 默认温度系数

        # --- 数据集特定覆盖 (Dataset Specific Overrides) ---
        # 注意: 节点数和特征维数仅供参考，实际运行时建议从 data 对象动态获取
        # 如果设为 False，则直接使用 input_dim 进行传播
        self.use_pca = False
        # 1. Planetoid Datasets
        if self.dataset == 'PubMed':
            self.n_clusters = 3
            self.n_nodes = 19717
            self.input_dim = 500
            self.k_hops = 16        # 论文推荐 16
            self.hidden_dim = 256
            self.tau = 0.5          # PubMed 需要稍高的宽容度
            self.lambda1 = 0.05
            self.lambda2 = 0.2

        elif self.dataset == 'Cora':
            self.n_clusters = 7
            self.n_nodes = 2708
            self.input_dim = 1433
            self.hidden_dim = 256   # 稀疏特征需要较大维度保留信息
            self.k_hops = 20        # 回归 APPNP 经典设置
            self.alpha = 0.1
            self.tau = 0.4
            self.lambda1 = 0.05
            self.lambda2 = 0.2

        elif self.dataset == 'Citeseer':
            self.n_clusters = 6
            self.n_nodes = 3327
            self.input_dim = 3703
            self.hidden_dim = 256 
            self.k_hops = 32        # 稀疏图且孤立点多，需要深层传播
            self.alpha = 0.1
            self.tau = 0.9          # 结构脆弱，给予更高宽容度
            self.lambda1 = 0.05
            self.lambda2 = 0.2

        # 2. Coauthor Datasets (稠密图)
        elif self.dataset == 'CS':
            self.n_clusters = 15
            self.n_nodes = 18333
            self.input_dim = 6805
            self.hidden_dim = 256
            self.k_hops = 10        # 稠密图传播快，10跳足够
            self.tau = 0.3          # 类别多且细，温度稍低以拉开间距
            self.lambda1 = 0.1
            self.lambda2 = 0.3

        elif self.dataset == 'Physics':
            self.n_clusters = 5
            self.n_nodes = 34493
            self.input_dim = 8415
            self.hidden_dim = 256
            self.k_hops = 10
            self.tau = 0.3
            self.lambda1 = 0.1
            self.lambda2 = 0.3

        # 3. Amazon Datasets (稠密图)
        elif self.dataset == 'Computers':
            self.n_clusters = 10
            self.n_nodes = 13752
            self.input_dim = 767
            self.hidden_dim = 256
            self.k_hops = 10
            self.tau = 0.4
            self.lambda1 = 0.1
            self.lambda2 = 0.4

        elif self.dataset == 'Photo':
            self.n_clusters = 8
            self.n_nodes = 7650
            self.input_dim = 745
            self.hidden_dim = 256
            self.k_hops = 10
            self.tau = 0.3
            self.lambda1 = 0.1
            self.lambda2 = 0.4

        else:
            raise ValueError(f"Unknown dataset: {self.dataset}. Supported: Cora, Citeseer, PubMed, CS, Physics, Computers, Photo")

    def __repr__(self):
        """打印配置信息"""
        return str(self.__dict__)

# 测试代码
if __name__ == '__main__':
    # 测试不同数据集配置
    for name in ['Cora', 'CS', 'Photo']:
        try:
            conf = Config(name)
            print(f"[{name}] Hops: {conf.k_hops}, Alpha: {conf.alpha}, Hidden: {conf.hidden_dim}")
        except ValueError as e:
            print(e)