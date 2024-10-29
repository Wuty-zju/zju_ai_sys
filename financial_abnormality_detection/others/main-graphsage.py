import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv
from utils import DGraphFin
from utils.evaluator import Evaluator
from utils.utils import prepare_folder
import os

# 设置GPU设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 定义GraphSAGE模型，增加解释性功能并修复残差连接的维度问题
class OptimizedGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, use_residual=True):
        super(OptimizedGraphSAGE, self).__init__()
        self.use_residual = use_residual
        
        # 定义卷积层
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))  # 输入层
        
        # 定义中间层的SAGEConv和BatchNorm
        self.norms = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.norms.append(nn.BatchNorm1d(hidden_channels))  # 使用BatchNorm
        
        # 输出层
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

        # 如果 in_channels 和 hidden_channels 不同，添加一个线性变换
        if in_channels != hidden_channels:
            self.residual_transform = nn.Linear(in_channels, hidden_channels)
        else:
            self.residual_transform = None

        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()
        if self.residual_transform is not None:
            self.residual_transform.reset_parameters()

    def forward(self, x, edge_index):
        activations = []
        residual = x  # 用于残差连接
        
        # 如果 in_channels != hidden_channels，调整 residual 维度
        if self.residual_transform is not None:
            residual = self.residual_transform(residual)

        # 第一层卷积
        x = self.convs[0](x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        activations.append(x)  # 保存每一层的激活

        # 中间层卷积
        for i, (conv, norm) in enumerate(zip(self.convs[1:-1], self.norms)):
            x = conv(x, edge_index)
            x = norm(x)  # BatchNorm
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_residual:
                x += residual  # 残差连接
            activations.append(x)

        # 输出层卷积
        x = self.convs[-1](x, edge_index)
        activations.append(x)
        
        return F.log_softmax(x, dim=-1), activations

# 训练函数
def train(model, data, train_idx, optimizer):
    model.train()
    optimizer.zero_grad()
    out, _ = model(data.x, data.edge_index)  # 传入节点特征和边索引
    loss = F.nll_loss(out[train_idx], data.y[train_idx].squeeze().long())  # 确保 data.y 是长整型 (int64)
    loss.backward()
    optimizer.step()
    return loss.item()

# 测试函数
def test(model, data, split_idx, evaluator):
    model.eval()
    with torch.no_grad():
        losses, eval_results = {}, {}
        for key in ['train', 'valid']:
            node_id = split_idx[key]
            out, _ = model(data.x, data.edge_index)  # 传入节点特征和边索引
            y_pred = out[node_id].exp()  # 概率输出
            losses[key] = F.nll_loss(out[node_id], data.y[node_id].squeeze().long()).item()
            eval_results[key] = evaluator.eval(data.y[node_id].squeeze().long(), y_pred)['auc']
    return eval_results, losses, y_pred

# 预测函数
def predict(model, data, node_id):
    model.eval()
    with torch.no_grad():
        out, _ = model(data.x, data.edge_index)  # 传入节点特征和边索引
        y_pred = out[node_id].exp()
    return y_pred

# 动态学习率调整函数
def adjust_learning_rate(optimizer, epoch, lr, decay_factor=0.5, step_size=100):
    """每隔 step_size 个 epoch，调整学习率"""
    if epoch % step_size == 0 and epoch != 0:
        new_lr = lr * decay_factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        print(f"Epoch {epoch}: Learning rate adjusted to {new_lr}")

# 解释模型的重要性，返回激活图
def grad_cam_explanation(activations, importance_idx):
    importance = activations[-1][importance_idx].abs()  # 使用最后一层的激活作为重要性
    return importance

# 主函数
def main():
    # 数据集路径和保存路径
    path = 'financial_abnormality_detection/datasets/632d74d4e2843a53167ee9a1-momodel/'
    save_dir = 'financial_abnormality_detection/results/'
    dataset_name = 'DGraph'

    # 创建保存结果的文件夹
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 加载数据集
    dataset = DGraphFin(root=path, name=dataset_name)
    data = dataset[0]

    # 使用数据集中的 edge_index
    if not hasattr(data, 'edge_index'):
        raise AttributeError("数据集没有 'edge_index' 属性，请检查数据加载过程")

    # 数据标准化
    x = data.x
    x = (x - x.mean(0)) / x.std(0)
    data.x = x

    # 划分数据集
    split_idx = {'train': data.train_mask, 'valid': data.valid_mask, 'test': data.test_mask}
    train_idx = split_idx['train']

    # GraphSAGE 模型参数
    sage_parameters = {
        'num_layers': 4,
        'hidden_channels': 100,
        'dropout': 0.6,  # 使用Dropout防止过拟合
    }

    # 训练超参数
    lr = 0.002  # 初始学习率
    weight_decay = 2e-4  # 正则化系数

    # 初始化GraphSAGE模型
    in_channels = data.x.size(-1)
    out_channels = 2  # 类别0和类别1
    model = OptimizedGraphSAGE(in_channels=in_channels, out_channels=out_channels, **sage_parameters).to(device)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 评估器
    evaluator = Evaluator('auc')

    # 模型训练
    epochs = 10000
    log_steps = 100
    best_valid_auc = 0
    min_valid_loss = float('inf')

    # 将数据移动到设备
    data = data.to(device)


    # 训练过程
    for epoch in range(1, epochs + 1):
        loss = train(model, data, train_idx, optimizer)
        eval_results, losses, _ = test(model, data, split_idx, evaluator)
        train_auc, valid_auc = eval_results['train'], eval_results['valid']
        train_loss, valid_loss = losses['train'], losses['valid']

        # 动态调整学习率
        adjust_learning_rate(optimizer, epoch, lr)

        # 保存最优模型
        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_sage_model_4*100_drop_0.6.pt'))

        if epoch % log_steps == 0:
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, '
                  f'Train AUC: {100 * train_auc:.2f}, Valid AUC: {100 * valid_auc:.2f}')

    # 预测部分
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_sage_model_4*100_drop_0.6.pt')))
    node_idx = 0
    y_pred = predict(model, data, node_idx)
    print(f'节点 {node_idx} 预测的标签为: {torch.argmax(y_pred).item()}')

if __name__ == "__main__":
    main()

