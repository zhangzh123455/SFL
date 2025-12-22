import sys
import torch
import numpy as np
import random
from config import Config
from utils import load_dataset
from participant import Client
# from coordinator import Server
from coordinator_plus import Server

def set_seed(seed):
    """
    固定随机种子，保证实验可复现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    # ------------------------------------------------------------------
    # 0. 环境初始化
    # ------------------------------------------------------------------
    # 加载配置 (默认 PubMed)
    args = Config('Photo')
    

    
    set_seed(args.seed)
    
    print(f"==================================================")
    print(f"   Distributed Graph Contrastive Learning (SimGCL)")
    print(f"   Dataset: {args.dataset}")
    print(f"   Client: PCA + {args.k_hops}-hop APPNP Propagation")
    print(f"   Server: Contrastive Encoder + InfoNCE Loss")
    print(f"   Device: {args.device}")
    print(f"==================================================\n")

    # 加载原始数据 (仅用于分发给Client和评估)
    dataset, n_cls = load_dataset(args.dataset, args.data_root)
    data = dataset[0]
    y_true = data.y  # 仅用于计算 NMI/ARI

    # ------------------------------------------------------------------
    # Phase 1: 客户端预计算 (Client-Side Pre-computation)
    # ------------------------------------------------------------------
    print(f"--- [Phase 1] Client: Local Computation ---")
    
    # 初始化客户端
    client = Client(data, args)
    
    # 执行预计算: PCA + 16跳线性传播
    # 返回的 x_prop 是包含丰富结构信息的特征矩阵
    x_prop = client.prepare_data()
    
    print(f"[Network] Transmitting data to Server... Shape: {x_prop.shape}\n")
    
    # 模拟数据传输
    x_prop_server = x_prop.to(args.device)

   # Phase 2: 服务器初始化
    # ------------------------------------------------------------------
    print(f"--- [Phase 2] Server: Setup ---")
    server = Server(args)
    
    # 初始评估
    print("Evaluating initial performance...")
    # [修改] 接收 f1
    nmi, ari, acc, f1 = server.evaluate(x_prop_server, y_true)
    print(f"[Init] NMI: {nmi:.4f} | ARI: {ari:.4f} | ACC: {acc:.4f} | F1: {f1:.4f}")
    print("")

    # ------------------------------------------------------------------
    # Phase 3: 服务器训练
    # ------------------------------------------------------------------
    print(f"--- [Phase 3] Server: Contrastive Training ---")
    
    # === [轨道 1] 聚类任务名人堂 (以 NMI 为王) ===
    best_clu_nmi = 0
    best_clu_ari = 0
    best_clu_epoch = 0
    
    # === [轨道 2] 分类任务名人堂 (以 ACC 为王) ===
    best_cls_acc = 0
    best_cls_f1 = 0
    best_cls_epoch = 0

    for epoch in range(1, args.epochs + 1):
        loss = server.train_epoch(x_prop_server, args.tau, args.lambda1, args.lambda2)
        
        if epoch % 10 == 0 or epoch == 1:
            # 获取所有指标
            nmi, ari, acc, f1 = server.evaluate(x_prop_server, y_true)
            
            # --- 更新聚类最佳记录 (只看 NMI) ---
            if nmi > best_clu_nmi:
                best_clu_nmi = nmi
                best_clu_ari = ari
                best_clu_epoch = epoch
                # 你甚至可以在这里保存一个模型 checkopoint_best_cluster.pth
            
            # --- 更新分类最佳记录 (只看 ACC) ---
            if acc > best_cls_acc:
                best_cls_acc = acc
                best_cls_f1 = f1
                best_cls_epoch = epoch
                # 在这里保存另一个模型 checkopoint_best_class.pth
            
            print(f"Epoch {epoch:03d} | Loss: {loss:.5f} | NMI: {nmi:.4f} | ARI: {ari:.4f} | ACC: {acc:.4f} | F1: {f1:.4f}")

    # ------------------------------------------------------------------
    # 4. 最终审判 (Final Verdict)
    # ------------------------------------------------------------------
    print(f"\n==================================================")
    print(f"   >>> Task 1: Clustering Performance (Best NMI)")
    print(f"   Best Epoch: {best_clu_epoch}")
    print(f"   NMI: {best_clu_nmi:.4f}")
    print(f"   ARI: {best_clu_ari:.4f}")
    print(f"--------------------------------------------------")
    print(f"   >>> Task 2: Classification Performance (Best ACC)")
    print(f"   Best Epoch: {best_cls_epoch}")
    print(f"   ACC: {best_cls_acc:.4f}")
    print(f"   F1:  {best_cls_f1:.4f}")
    print(f"==================================================")


if __name__ == "__main__":
    main()