import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.transforms as T
from utils import DGraphFin
from utils.evaluator import Evaluator
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ==================== 数据加载和预处理 ====================
# 路径和参数设置
path = './datasets/632d74d4e2843a53167ee9a1-momodel/'  # 数据保存路径
save_dir = './results/'  # 模型保存路径
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
dataset_name = 'DGraph'  # 数据集名称

dataset = DGraphFin(root=path, name=dataset_name, transform=T.ToSparseTensor())
data = dataset[0]

# 数据预处理
x = data.x
x = (x - x.mean(0)) / x.std(0)  # 标准化节点特征
data.x = x
    
# 划分训练集、验证集和测试集
split_idx = {
    'train': data.train_mask,
    'valid': data.valid_mask,
    'test': data.test_mask
}
train_idx = split_idx['train']

# 将数据移动到设备上（GPU 或 CPU）
data = data.to(device)

# ==================== 定义模型 ====================
class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, batchnorm=True):
        super(MLP, self).__init__()
        # 定义多层感知机结构
        self.lins = nn.ModuleList([nn.Linear(in_channels, hidden_channels)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_channels)]) if batchnorm else None
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
            if batchnorm:
                self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        #重置模型参数
        for lin in self.lins:
            lin.reset_parameters()
        if self.bns:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x):
        #模型前向传播
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            if self.bns:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return F.log_softmax(x, dim=-1)

# ==================== 训练和评估函数 ====================
# 训练超参数设置
num_layers = 5
hidden_channels = 128

mlp_parameters = {'num_layers': num_layers, 'hidden_channels':hidden_channels, 'dropout': 0.5, 'batchnorm': True}
in_channels, out_channels = data.x.size(-1), 2
model = MLP(in_channels, **mlp_parameters, out_channels=out_channels).to(device)

# 优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

# 评估器
evaluator = Evaluator('auc')

def train(model, data, train_idx, optimizer):
    """
    训练模型
    :param model: 模型对象
    :param data: 数据对象
    :param train_idx: 训练集索引
    :param optimizer: 优化器
    :return: 损失值
    """
    model.train()
    optimizer.zero_grad()
    out = model(data.x[train_idx])
    loss = F.nll_loss(out, data.y[train_idx].squeeze().long())
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, data, split_idx, evaluator):
    """
    测试模型性能
    :param model: 模型对象
    :param data: 数据对象
    :param split_idx: 数据集划分字典
    :param evaluator: 评估器
    :return: 评估结果、损失和预测值
    """
    model.eval()
    with torch.no_grad():
        losses, eval_results = {}, {}
        for key in ['train', 'valid']:
            node_id = split_idx[key]
            out = model(data.x[node_id])
            y_pred = out.exp()
            losses[key] = F.nll_loss(out, data.y[node_id].squeeze().long()).item()
            eval_results[key] = evaluator.eval(data.y[node_id].squeeze().long(), y_pred)['auc']
    return eval_results, losses, y_pred

# ==================== 训练模型 ====================
def train_model(model, data, split_idx, optimizer, evaluator, save_dir, epochs=1000, log_steps=10):
    """
    执行模型训练并保存最佳模型。
    :param model: 模型对象
    :param data: 数据对象
    :param split_idx: 数据集划分字典
    :param optimizer: 优化器
    :param evaluator: 评估器
    :param save_dir: 模型保存路径
    :param epochs: 训练轮数
    :param log_steps: 日志记录频率
    """
    best_valid_auc, min_valid_loss = 0, float('inf')
    train_idx = split_idx['train']

    for epoch in range(1, epochs + 1):
        loss = train(model, data, train_idx, optimizer)
        eval_results, losses, _ = test(model, data, split_idx, evaluator)
        train_auc, valid_auc = eval_results['train'], eval_results['valid']
        train_loss, valid_loss = losses['train'], losses['valid']

        # 保存最优模型
        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(save_dir, f'best_mlp_model_layers{num_layers}_hidden{hidden_channels}.pt'))

        if epoch % 10 == 0:
            print(f'第 {epoch:04d} 轮，损失值：{loss:.4f}，训练集 AUC：{train_auc * 100:.2f}% ，验证集 AUC：{valid_auc * 100:.2f}%')
            
train_model(model, data, split_idx, optimizer, evaluator, save_dir)

# ==================== 保存并加载最佳模型 ====================
def load_best_model(model, save_dir):
    """
    加载最佳模型权重。
    :param model: 模型对象
    :param save_dir: 模型保存路径
    :return: 加载权重后的模型
    """
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_mlp_model.pt')))
    return model

# ==================== 测试函数 ====================
def predict(model, data, node_id):
    """
    预测指定节点的标签概率。
    :param model: 训练好的模型
    :param data: 数据对象
    :param node_id: 节点索引
    :return: 节点的预测概率
    """
    model.eval()
    with torch.no_grad():
        out = model(data.x[node_id].unsqueeze(0))
        y_pred = out.exp()
    return y_pred
