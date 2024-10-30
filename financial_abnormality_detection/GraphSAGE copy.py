import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import torch_geometric.transforms as T
from utils import DGraphFin
from utils.evaluator import Evaluator
import numpy as np
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ==================== 数据加载和预处理 ====================
# 数据路径设置
path = './datasets/632d74d4e2843a53167ee9a1-momodel/'  # 数据保存路径
save_dir = './results/'  # 模型保存路径
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
dataset_name = 'DGraph'  # 数据集名称

# 加载数据集
dataset = DGraphFin(root=path, name=dataset_name, transform=T.ToSparseTensor())
nlabels = 2  # 仅需预测类别 0 和类别 1
data = dataset[0]
data.adj_t = data.adj_t.to_symmetric()  # 将有向图转换为无向图

# 数据预处理
x = data.x
x = (x - x.mean(0)) / x.std(0)  # 标准化节点特征
data.x = x
if data.y.dim() == 2:
    data.y = data.y.squeeze(1)  # 如果标签维度为 2，则压缩为 1 维

# 划分训练集、验证集和测试集
split_idx = {
    'train': data.train_mask,
    'valid': data.valid_mask,
    'test': data.test_mask
}
train_idx = split_idx['train']

# 将数据移动到设备上（GPU 或 CPU）
data = data.to(device)

# 将稀疏邻接矩阵 adj_t 转换为 edge_index（适用于 SAGEConv）
row, col, _ = data.adj_t.coo()  # 获取 COO 格式的行、列索引
data.edge_index = torch.stack([row, col], dim=0)  # 构建 edge_index 矩阵，形状为 [2, num_edges]

# ==================== 定义模型 ====================
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        
    def forward(self, x, adj_t):
        x = F.relu(self.conv1(x, adj_t))
        x = self.conv2(x, adj_t)
        return F.log_softmax(x, dim=-1)

# 实例化模型并移动到设备上
in_channels = data.x.size(-1)  # 输入特征维度
hidden_channels = 64            # 隐藏层维度
out_channels = nlabels         # 输出类别数
model = GraphSAGE(
    in_channels=in_channels,
    hidden_channels=hidden_channels,
    out_channels=out_channels
).to(device)

# ==================== 训练和评估函数 ====================
# 训练超参数设置
epochs = 2000           # 训练轮数
lr = 0.005              # 学习率
weight_decay = 2e-4     # 权重衰减（L2 正则化系数）

# 优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# 评估器
eval_metric = 'auc'  # 使用 AUC 作为评估指标
evaluator = Evaluator(eval_metric)

# 定义训练函数
def train(model, data, train_idx, optimizer):
    model.train()  # 设置模型为训练模式
    optimizer.zero_grad()  # 清空梯度
    out = model(data.x, data.edge_index)  # 前向传播
    loss = F.nll_loss(out[train_idx], data.y[train_idx])  # 计算损失（负对数似然损失）
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数
    return loss.item()  # 返回损失值

# 定义测试函数
def test(model, data, split_idx, evaluator):
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        out = model(data.x, data.edge_index)  # 前向传播
        y_pred = out.exp()  # 将 Log Softmax 输出转换为概率
        eval_results = {}
        losses = {}
        for key in ['train', 'valid']:
            node_id = split_idx[key]
            losses[key] = F.nll_loss(out[node_id], data.y[node_id]).item()  # 计算损失
            # 计算评估指标（AUC）
            eval_results[key] = evaluator.eval(data.y[node_id], y_pred[node_id])[eval_metric]
    return eval_results, losses  # 返回评估结果和损失

# ==================== 训练、保存并加载最佳模型 ====================
best_valid_auc = 0  # 初始化最佳验证集 AUC
best_model_state = None  # 用于保存最佳模型状态

for epoch in range(1, epochs + 1):
    loss = train(model, data, train_idx, optimizer)  # 训练一步
    eval_results, losses = test(model, data, split_idx, evaluator)  # 在训练集和验证集上测试
    train_auc = eval_results['train']
    valid_auc = eval_results['valid']
    
    if valid_auc > best_valid_auc:
        best_valid_auc = valid_auc
        best_model_state = model.state_dict()  # 保存当前最佳模型状态
        # 保存最佳模型
        model_filename = f'best_sage_model_conv2_hidden{hidden_channels}_lr{lr}_wd{weight_decay}.pt'
        torch.save(best_model_state, os.path.join(save_dir, model_filename))
    
    if epoch % 10 == 0:
        print(f'第 {epoch:04d} 轮，损失值：{loss:.4f}，训练集 AUC：{train_auc * 100:.2f}% ，验证集 AUC：{valid_auc * 100:.2f}%')

print("训练完成。")
print(f"最佳验证集 AUC：{best_valid_auc * 100:.2f}%")
print(f"最佳模型已保存至 {os.path.join(save_dir, model_filename)}")

model.load_state_dict(torch.load(os.path.join(save_dir, model_filename), map_location=device))

# ==================== 测试并保存预测结果函数 ====================
def test_and_save_predictions(model, data, save_path):
    """
    运行模型的前向传播，并保存所有节点的预测结果
    :param model: 训练好的模型
    :param data: 包含节点特征和边的图数据
    :param save_path: 保存预测结果的文件路径
    """
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        # 对所有节点进行前向传播
        out = model(data.x, data.edge_index)
        y_pred = out.exp()  # 将 Log Softmax 输出转换为概率

    # 保存预测结果
    torch.save(y_pred.cpu(), save_path)
    print(f"预测结果已保存至 {save_path}")

# 运行模型并保存预测结果
predictions_save_path = os.path.join(
    save_dir,
    f'best_sage_model_conv2_hidden{hidden_channels}_lr{lr}_wd{weight_decay}_predictions.pt'
)
test_and_save_predictions(model, data, predictions_save_path)

'''
# ==================== 测试-预测函数 ====================
def predict(data, node_id):
    """
    加载模型并在 MoAI 平台进行预测
    :param data: 数据对象，包含 x 和 edge_index 等属性
    :param node_id: int，需要进行预测的节点索引
    :return: tensor，类别 0 和类别 1 的概率
    """
    out = model
    y_pred = out[node_id].exp() # 获取指定节点的预测概率，并增加一个维度
    
    return y_pred  # 返回预测概率

model = torch.load('./results/best_sage_model_conv2_hidden128_lr0.002_wd0.0002_predictions.pt')
'''