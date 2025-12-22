import os
import torch
import numpy as np
from torch_geometric.datasets import Planetoid, Coauthor, Amazon
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.transforms import NormalizeFeatures
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from torch_geometric.nn import knn_graph


# --- [环境补丁] 修复 NumPy 1.24+ 移除了 np.float 导致的旧版 PyG 报错 ---
if not hasattr(np, 'float'):
    np.float = float

def load_dataset(dataset_name, data_root='./data'):
    """
    加载数据集，支持以下名称 (不区分大小写):
    - Planetoid: 'Cora', 'Citeseer', 'PubMed'
    - Coauthor:  'CS', 'Physics'
    - Amazon:    'Computers', 'Photo'
    
    所有数据集均会自动应用行归一化 (NormalizeFeatures)
    """
    # 统一转为小写以进行匹配
    name_key = dataset_name.lower()
    
    # 1. Planetoid Datasets
    if name_key in ['cora', 'citeseer', 'pubmed']:
        # 映射回 PyG 要求的标准名称格式
        name_map = {'cora': 'Cora', 'citeseer': 'Citeseer', 'pubmed': 'PubMed'}
        dataset = Planetoid(
            root=data_root, 
            name=name_map[name_key], 
            transform=NormalizeFeatures()
        )
        
    # 2. Coauthor Datasets
    elif name_key in ['cs', 'physics']:
        # CS 需要全大写
        name_map = {'cs': 'CS', 'physics': 'Physics'}
        dataset = Coauthor(
            root=data_root, 
            name=name_map[name_key], 
            transform=NormalizeFeatures()
        )
        
    # 3. Amazon Datasets
    elif name_key in ['computers', 'photo']:
        name_map = {'computers': 'Computers', 'photo': 'Photo'}
        dataset = Amazon(
            root=data_root, 
            name=name_map[name_key], 
            transform=NormalizeFeatures()
        )
        
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported. "
                         f"Available: Cora, Citeseer, PubMed, CS, Physics, Computers, Photo")
    
    # 返回 dataset 对象和类别数
    return dataset, dataset.num_classes

def get_normalized_adj(edge_index, num_nodes):
    """
    计算 GCNII 所需的归一化自环邻接矩阵 P_tilde
    公式: P_tilde = (D+I)^(-1/2) * (A+I) * (D+I)^(-1/2)
    """
    # 1. 添加自环: A -> A + I
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    
    # 2. 计算度矩阵 D_tilde
    row, col = edge_index
    deg = degree(col, num_nodes, dtype=torch.float)
    
    # 3. 计算归一化系数: D^(-1/2)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    
    # 4. 计算边权重: D^(-1/2) * A_tilde * D^(-1/2)
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    
    # 5. 构建稀疏矩阵 (COO格式，兼容 PyTorch 1.9)
    adj = torch.sparse_coo_tensor(
        edge_index, 
        norm, 
        (num_nodes, num_nodes)
    )
    
    return adj

def clustering_metrics(y_true, y_pred):
    """
    计算聚类指标: NMI 和 ARI
    """
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
        
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    
    return nmi, ari


def construct_graph(x, k=10, metric='cosine'):
    """
    基于特征相似度重构图结构 (A')
    """
    # 这里的 x 是 (N, D) 的特征矩阵
    # knn_graph 会返回 [2, E] 的 edge_index
    # loop=False 表示不包含自环 (GATConv 通常自己会处理或不敏感)
    edge_index = knn_graph(x, k=k, loop=False, cosine=(metric=='cosine'))
    return edge_index

# --- 测试代码 ---
if __name__ == "__main__":
    try:
        # 测试加载不同的数据集
        for name in ['Cora', 'CS', 'Photo']:
            print(f"Loading {name}...")
            dataset, n_cls = load_dataset(name)
            data = dataset[0]
            print(f"  - Nodes: {data.num_nodes}")
            print(f"  - Features: {data.num_features}")
            print(f"  - Classes: {n_cls}")
            
            # 测试邻接矩阵计算
            adj = get_normalized_adj(data.edge_index, data.num_nodes)
            print(f"  - Adj shape: {adj.shape}")
            print("-" * 30)
            
    except Exception as e:
        print(f"Test failed: {e}")

        