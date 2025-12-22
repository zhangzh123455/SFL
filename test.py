import time
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

# ==========================================
# 1. 配置与超参数
# ==========================================
DATASET_NAME = 'PubMed'
EPOCHS = 10
HIDDEN_DIM = 256
EMBEDDING_DIM = 128  # 最终输出的嵌入维度

# 5G 带宽配置 (单位转换: Mbps -> bits per second)
# 1 Mbps = 1,000,000 bits/s
UPLOAD_BANDWIDTH = 20 * 1_000_000   # 100 Mbps
DOWNLOAD_BANDWIDTH = 200 * 1_000_000 # 500 Mbps

# 数据类型大小 (Float32 占 4 bytes = 32 bits)
BIT_PER_ELEMENT = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on: {device}")

# ==========================================
# 2. 模型定义
# ==========================================

class ClientEncoder(torch.nn.Module):
    """客户端：3层 GCN 编码器"""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.conv3(x, edge_index)  # 输出嵌入 (Embeddings)

# 服务器端使用的是 GAE 的标准 Decoder (InnerProductDecoder)
# 在 PyTorch Geometric 中，GAE 类自带了解码器和损失计算功能

# ==========================================
# 3. 辅助函数：模拟传输
# ==========================================

def calculate_transfer_time(tensor_data, bandwidth_bps):
    """
    计算传输时间并模拟延迟
    :param tensor_data: 要传输的张量
    :param bandwidth_bps: 带宽 (bits per second)
    :return: 传输耗时 (秒), 数据大小 (MB)
    """
    num_elements = tensor_data.numel() # 元素总数
    total_bits = num_elements * BIT_PER_ELEMENT
    
    # 理论传输时间
    transfer_time = total_bits / bandwidth_bps
    
    # 模拟网络延迟 (Sleep)
    time.sleep(transfer_time)
    
    return transfer_time, total_bits / (8 * 1024 * 1024) # 返回秒和MB

# ==========================================
# 4. 主程序
# ==========================================

def run_simulation():
    # --- 加载数据 ---
    dataset = Planetoid(root='./data', name=DATASET_NAME, transform=T.NormalizeFeatures())
    data = dataset[0].to(device)
    
    # --- 初始化模型 ---
    # 客户端模型
    client_model = ClientEncoder(dataset.num_features, HIDDEN_DIM, EMBEDDING_DIM).to(device)
    # 优化器只管理客户端的参数（因为本例中服务器端解码器无参数，如果有参数也需要优化）
    optimizer = torch.optim.Adam(client_model.parameters(), lr=0.01)
    
    # 服务器模型 (GAE 封装，这里只用它的解码和 Loss 功能)
    # 注意：GAE 通常接收一个 encoder，但我们在 split learning 中手动分步执行
    server_model = GAE(encoder=None).to(device) 

    print(f"\n{'='*20} 实验开始 {'='*20}")
    print(f"模型结构: 3-Layer GCN (Client) -> Split -> GAE Decoder (Server)")
    print(f"网络环境: 5G (Up: {UPLOAD_BANDWIDTH/1e6} Mbps, Down: {DOWNLOAD_BANDWIDTH/1e6} Mbps)")
    print(f"节点数量: {data.num_nodes}, 嵌入维度: {EMBEDDING_DIM}")
    print(f"{'-'*60}")
    
    ratio_history = []

    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        
        # --- 计时器开始 (单个 Epoch) ---
        epoch_start_time = time.time()
        
        # 1. [Client] 本地计算 (Forward)
        t0 = time.time()
        z_client = client_model(data.x, data.edge_index)
        client_comp_time_fwd = time.time() - t0
        
        # --- 关键点：切分计算图 (Detaching) ---
        # 为了模拟发送数据到服务器，我们需要切断梯度回传的自动链路
        # client_out 是我们要发送的 payload
        client_out = z_client.detach().requires_grad_(True)
        
        # 2. [Network] 上行传输 (Client -> Server)
        # 传输内容：节点嵌入 (Node Embeddings)
        upload_time, data_size_mb = calculate_transfer_time(z_client, UPLOAD_BANDWIDTH)
        
        # 3. [Server] 服务器计算 (Forward & Loss)
        t1 = time.time()
        # GAE 使用 inner product decoder 重构邻接矩阵并计算 Loss
        # 注意：这里输入的是从客户端"接收"到的 client_out
        loss = server_model.recon_loss(client_out, data.edge_index)
        server_comp_time = time.time() - t1
        
        # 4. [Server] 服务器反向传播 (Backward)
        t2 = time.time()
        loss.backward() # 计算 loss 对 client_out 的梯度
        server_grad_calc_time = time.time() - t2
        
        # 获取要传回客户端的梯度
        server_grads = client_out.grad
        
        # 5. [Network] 下行传输 (Server -> Client)
        # 传输内容：嵌入的梯度 (Gradients of Embeddings)
        # 梯度的大小通常与嵌入的大小完全一致
        download_time, _ = calculate_transfer_time(server_grads, DOWNLOAD_BANDWIDTH)
        
        # 6. [Client] 客户端反向传播 (Backward)
        t3 = time.time()
        # 将接收到的梯度传给客户端计算图的叶子节点
        z_client.backward(server_grads) 
        optimizer.step()
        client_comp_time_bwd = time.time() - t3
        
        # --- 计时器结束 ---
        epoch_total_time = time.time() - epoch_start_time
        
        # 汇总时间
        total_comm_time = upload_time + download_time
        total_comp_time = client_comp_time_fwd + server_comp_time + server_grad_calc_time + client_comp_time_bwd
        
        # 计算占比
        comm_ratio = (total_comm_time / epoch_total_time) * 100
        ratio_history.append(comm_ratio)
        
        print(f"Epoch {epoch:02d} | "
              f"总耗时: {epoch_total_time:.4f}s | "
              f"通信耗时: {total_comm_time:.4f}s | "
              f"计算耗时: {total_comp_time:.4f}s | "
              f"通信占比: {comm_ratio:.2f}% | "
              f"传输数据: {data_size_mb:.2f} MB")

    # ==========================================
    # 5. 结果统计
    # ==========================================
    avg_ratio = sum(ratio_history) / len(ratio_history)
    print(f"{'='*20} 实验结束 {'='*20}")
    print(f"平均通信时间占比 (10 Epochs): {avg_ratio:.2f}%")

if __name__ == "__main__":
    run_simulation()