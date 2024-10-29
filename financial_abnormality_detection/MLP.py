# main.py

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.transforms as T
from utils import DGraphFin
from utils.evaluator import Evaluator
from utils.utils import prepare_folder
import os

# 设置GPU设备
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# 定义MLP模型
class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, batchnorm=True):
        super(MLP, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            if self.batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            if self.batchnorm:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return F.log_softmax(x, dim=-1)

# 训练函数
def train(model, data, train_idx, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x[train_idx])  # 确保 out 是 log_softmax 输出
    loss = F.nll_loss(out, data.y[train_idx].squeeze().long())  # 确保 data.y 是长整型 (int64) 且维度正确
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
            out = model(data.x[node_id])
            y_pred = out.exp()  # 概率输出
            losses[key] = F.nll_loss(out, data.y[node_id].squeeze().long()).item()  # 确保 data.y 是长整型且维度正确
            eval_results[key] = evaluator.eval(data.y[node_id].squeeze().long(), y_pred)['auc']  # 同样处理 evaluator 输入
    return eval_results, losses, y_pred

# 预测函数
def predict(model, data, node_id):
    model.eval()
    with torch.no_grad():
        out = model(data.x[node_id].unsqueeze(0))  # 使用 unsqueeze(0) 将 1D 张量转为 2D 张量
        y_pred = out.exp()
    return y_pred

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
    dataset = DGraphFin(root=path, name=dataset_name, transform=T.ToSparseTensor())
    data = dataset[0]

    # 数据标准化
    x = data.x
    x = (x - x.mean(0)) / x.std(0)
    data.x = x

    # 划分数据集
    split_idx = {'train': data.train_mask, 'valid': data.valid_mask, 'test': data.test_mask}
    train_idx = split_idx['train']

    # MLP模型参数
    mlp_parameters = {
        'num_layers': 4,
        'hidden_channels': 512,
        'dropout': 0.5,  # 使用Dropout防止过拟合
        'batchnorm': True,  # 使用批归一化
    }
    
    # 训练超参数
    lr = 0.001  # 学习率
    weight_decay = 5e-4  # 正则化系数

    # 初始化模型
    in_channels = data.x.size(-1)
    out_channels = 2  # 类别0和类别1
    model = MLP(in_channels=in_channels, out_channels=out_channels, **mlp_parameters).to(device)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 评估器
    evaluator = Evaluator('auc')

    # 模型训练
    epochs = 1000
    log_steps = 10
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

        # 保存最优模型
        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_mlp_model.pt'))

        if epoch % log_steps == 0:
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, '
                  f'Train AUC: {100 * train_auc:.2f}, Valid AUC: {100 * valid_auc:.2f}')

    # 预测部分
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_mlp_model.pt')))
    node_idx = 0
    y_pred = predict(model, data, node_idx)
    print(f'节点 {node_idx} 预测的标签为: {torch.argmax(y_pred).item()}')

if __name__ == "__main__":
    main()


"""
Epoch: 010, Train Loss: 0.2327, Valid Loss: 0.2328, Train AUC: 53.78, Valid AUC: 53.24
Epoch: 020, Train Loss: 0.0824, Valid Loss: 0.0823, Train AUC: 68.91, Valid AUC: 68.72
Epoch: 030, Train Loss: 0.0918, Valid Loss: 0.0919, Train AUC: 71.16, Valid AUC: 70.51
Epoch: 040, Train Loss: 0.0769, Valid Loss: 0.0772, Train AUC: 71.89, Valid AUC: 70.95
Epoch: 050, Train Loss: 0.0711, Valid Loss: 0.0714, Train AUC: 71.82, Valid AUC: 70.86
Epoch: 060, Train Loss: 0.0679, Valid Loss: 0.0682, Train AUC: 72.01, Valid AUC: 70.98
Epoch: 070, Train Loss: 0.0666, Valid Loss: 0.0669, Train AUC: 72.06, Valid AUC: 71.01
Epoch: 080, Train Loss: 0.0667, Valid Loss: 0.0671, Train AUC: 72.08, Valid AUC: 71.02
Epoch: 090, Train Loss: 0.0662, Valid Loss: 0.0666, Train AUC: 72.17, Valid AUC: 71.12
Epoch: 100, Train Loss: 0.0654, Valid Loss: 0.0658, Train AUC: 72.23, Valid AUC: 71.15
Epoch: 110, Train Loss: 0.0651, Valid Loss: 0.0655, Train AUC: 72.32, Valid AUC: 71.24
Epoch: 120, Train Loss: 0.0645, Valid Loss: 0.0649, Train AUC: 72.31, Valid AUC: 71.16
Epoch: 130, Train Loss: 0.0647, Valid Loss: 0.0652, Train AUC: 72.38, Valid AUC: 71.30
Epoch: 140, Train Loss: 0.0641, Valid Loss: 0.0645, Train AUC: 72.37, Valid AUC: 71.15
Epoch: 150, Train Loss: 0.0640, Valid Loss: 0.0645, Train AUC: 72.48, Valid AUC: 71.25
Epoch: 160, Train Loss: 0.0639, Valid Loss: 0.0643, Train AUC: 72.54, Valid AUC: 71.29
Epoch: 170, Train Loss: 0.0636, Valid Loss: 0.0641, Train AUC: 72.68, Valid AUC: 71.39
Epoch: 180, Train Loss: 0.0636, Valid Loss: 0.0640, Train AUC: 72.72, Valid AUC: 71.49
Epoch: 190, Train Loss: 0.0635, Valid Loss: 0.0640, Train AUC: 72.72, Valid AUC: 71.44
Epoch: 200, Train Loss: 0.0637, Valid Loss: 0.0641, Train AUC: 72.74, Valid AUC: 71.53
Epoch: 210, Train Loss: 0.0636, Valid Loss: 0.0640, Train AUC: 72.81, Valid AUC: 71.62
Epoch: 220, Train Loss: 0.0635, Valid Loss: 0.0639, Train AUC: 72.81, Valid AUC: 71.53
Epoch: 230, Train Loss: 0.0634, Valid Loss: 0.0638, Train AUC: 72.95, Valid AUC: 71.60
Epoch: 240, Train Loss: 0.0634, Valid Loss: 0.0638, Train AUC: 73.00, Valid AUC: 71.70
Epoch: 250, Train Loss: 0.0633, Valid Loss: 0.0637, Train AUC: 72.98, Valid AUC: 71.70
Epoch: 260, Train Loss: 0.0633, Valid Loss: 0.0638, Train AUC: 73.02, Valid AUC: 71.68
Epoch: 270, Train Loss: 0.0633, Valid Loss: 0.0637, Train AUC: 73.18, Valid AUC: 71.78
Epoch: 280, Train Loss: 0.0632, Valid Loss: 0.0636, Train AUC: 73.21, Valid AUC: 71.85
Epoch: 290, Train Loss: 0.0633, Valid Loss: 0.0637, Train AUC: 73.12, Valid AUC: 71.70
Epoch: 300, Train Loss: 0.0633, Valid Loss: 0.0638, Train AUC: 73.07, Valid AUC: 71.73
Epoch: 310, Train Loss: 0.0633, Valid Loss: 0.0637, Train AUC: 73.30, Valid AUC: 71.88
Epoch: 320, Train Loss: 0.0632, Valid Loss: 0.0637, Train AUC: 73.27, Valid AUC: 71.73
Epoch: 330, Train Loss: 0.0632, Valid Loss: 0.0636, Train AUC: 73.36, Valid AUC: 71.84
Epoch: 340, Train Loss: 0.0633, Valid Loss: 0.0638, Train AUC: 73.19, Valid AUC: 71.67
Epoch: 350, Train Loss: 0.0635, Valid Loss: 0.0638, Train AUC: 72.52, Valid AUC: 71.51
Epoch: 360, Train Loss: 0.0632, Valid Loss: 0.0636, Train AUC: 73.15, Valid AUC: 71.86
Epoch: 370, Train Loss: 0.0637, Valid Loss: 0.0641, Train AUC: 73.12, Valid AUC: 71.70
Epoch: 380, Train Loss: 0.0633, Valid Loss: 0.0638, Train AUC: 73.15, Valid AUC: 71.66
Epoch: 390, Train Loss: 0.0633, Valid Loss: 0.0638, Train AUC: 73.31, Valid AUC: 71.66
Epoch: 400, Train Loss: 0.0632, Valid Loss: 0.0637, Train AUC: 73.38, Valid AUC: 71.73
Epoch: 410, Train Loss: 0.0632, Valid Loss: 0.0636, Train AUC: 73.11, Valid AUC: 71.67
Epoch: 420, Train Loss: 0.0632, Valid Loss: 0.0637, Train AUC: 73.42, Valid AUC: 71.83
Epoch: 430, Train Loss: 0.0632, Valid Loss: 0.0637, Train AUC: 73.41, Valid AUC: 71.83
Epoch: 440, Train Loss: 0.0633, Valid Loss: 0.0639, Train AUC: 73.24, Valid AUC: 71.61
Epoch: 450, Train Loss: 0.0683, Valid Loss: 0.0686, Train AUC: 58.87, Valid AUC: 58.09
Epoch: 460, Train Loss: 0.0640, Valid Loss: 0.0644, Train AUC: 72.20, Valid AUC: 70.82
Epoch: 470, Train Loss: 0.0638, Valid Loss: 0.0644, Train AUC: 73.09, Valid AUC: 71.65
Epoch: 480, Train Loss: 0.0638, Valid Loss: 0.0643, Train AUC: 73.06, Valid AUC: 71.62
Epoch: 490, Train Loss: 0.0639, Valid Loss: 0.0644, Train AUC: 72.93, Valid AUC: 71.48
Epoch: 500, Train Loss: 0.0636, Valid Loss: 0.0642, Train AUC: 73.02, Valid AUC: 71.59
Epoch: 510, Train Loss: 0.0633, Valid Loss: 0.0639, Train AUC: 73.25, Valid AUC: 71.78
Epoch: 520, Train Loss: 0.0632, Valid Loss: 0.0638, Train AUC: 73.22, Valid AUC: 71.60
Epoch: 530, Train Loss: 0.0636, Valid Loss: 0.0641, Train AUC: 72.89, Valid AUC: 71.67
Epoch: 540, Train Loss: 0.0634, Valid Loss: 0.0639, Train AUC: 73.03, Valid AUC: 71.82
Epoch: 550, Train Loss: 0.0636, Valid Loss: 0.0642, Train AUC: 73.29, Valid AUC: 71.66
Epoch: 560, Train Loss: 0.0635, Valid Loss: 0.0640, Train AUC: 73.31, Valid AUC: 71.57
Epoch: 570, Train Loss: 0.0634, Valid Loss: 0.0639, Train AUC: 73.28, Valid AUC: 71.57
Epoch: 580, Train Loss: 0.0632, Valid Loss: 0.0637, Train AUC: 73.36, Valid AUC: 71.68
Epoch: 590, Train Loss: 0.0633, Valid Loss: 0.0638, Train AUC: 73.52, Valid AUC: 71.79
Epoch: 600, Train Loss: 0.0632, Valid Loss: 0.0637, Train AUC: 73.34, Valid AUC: 71.83
Epoch: 610, Train Loss: 0.0631, Valid Loss: 0.0637, Train AUC: 73.50, Valid AUC: 71.85
Epoch: 620, Train Loss: 0.0633, Valid Loss: 0.0639, Train AUC: 73.55, Valid AUC: 71.67
Epoch: 630, Train Loss: 0.0631, Valid Loss: 0.0638, Train AUC: 73.38, Valid AUC: 71.63
Epoch: 640, Train Loss: 0.0633, Valid Loss: 0.0639, Train AUC: 73.45, Valid AUC: 71.81
Epoch: 650, Train Loss: 0.0633, Valid Loss: 0.0639, Train AUC: 73.02, Valid AUC: 71.43
Epoch: 660, Train Loss: 0.0631, Valid Loss: 0.0637, Train AUC: 73.57, Valid AUC: 71.78
Epoch: 670, Train Loss: 0.0632, Valid Loss: 0.0638, Train AUC: 73.52, Valid AUC: 71.64
Epoch: 680, Train Loss: 0.0631, Valid Loss: 0.0638, Train AUC: 73.54, Valid AUC: 71.71
Epoch: 690, Train Loss: 0.0633, Valid Loss: 0.0640, Train AUC: 73.19, Valid AUC: 71.61
Epoch: 700, Train Loss: 0.0630, Valid Loss: 0.0637, Train AUC: 73.61, Valid AUC: 71.86
Epoch: 710, Train Loss: 0.0631, Valid Loss: 0.0638, Train AUC: 73.60, Valid AUC: 71.74
Epoch: 720, Train Loss: 0.0632, Valid Loss: 0.0638, Train AUC: 73.65, Valid AUC: 71.69
Epoch: 730, Train Loss: 0.0631, Valid Loss: 0.0638, Train AUC: 73.41, Valid AUC: 71.64
Epoch: 740, Train Loss: 0.0632, Valid Loss: 0.0638, Train AUC: 73.32, Valid AUC: 71.71
Epoch: 750, Train Loss: 0.0630, Valid Loss: 0.0637, Train AUC: 73.57, Valid AUC: 71.81
Epoch: 760, Train Loss: 0.0632, Valid Loss: 0.0639, Train AUC: 73.55, Valid AUC: 71.64
Epoch: 770, Train Loss: 0.0630, Valid Loss: 0.0636, Train AUC: 73.61, Valid AUC: 71.89
Epoch: 780, Train Loss: 0.0630, Valid Loss: 0.0637, Train AUC: 73.60, Valid AUC: 71.79
Epoch: 790, Train Loss: 0.0632, Valid Loss: 0.0638, Train AUC: 73.50, Valid AUC: 71.65
Epoch: 800, Train Loss: 0.0631, Valid Loss: 0.0638, Train AUC: 73.59, Valid AUC: 71.69
Epoch: 810, Train Loss: 0.0630, Valid Loss: 0.0637, Train AUC: 73.66, Valid AUC: 71.82
Epoch: 820, Train Loss: 0.0631, Valid Loss: 0.0637, Train AUC: 73.69, Valid AUC: 71.81
Epoch: 830, Train Loss: 0.0632, Valid Loss: 0.0637, Train AUC: 73.58, Valid AUC: 71.84
Epoch: 840, Train Loss: 0.0632, Valid Loss: 0.0639, Train AUC: 73.25, Valid AUC: 71.66
Epoch: 850, Train Loss: 0.0632, Valid Loss: 0.0638, Train AUC: 73.53, Valid AUC: 71.72
Epoch: 860, Train Loss: 0.0631, Valid Loss: 0.0637, Train AUC: 73.58, Valid AUC: 71.80
Epoch: 870, Train Loss: 0.0633, Valid Loss: 0.0638, Train AUC: 73.26, Valid AUC: 71.62
Epoch: 880, Train Loss: 0.0631, Valid Loss: 0.0637, Train AUC: 73.41, Valid AUC: 71.79
Epoch: 890, Train Loss: 0.0631, Valid Loss: 0.0638, Train AUC: 73.58, Valid AUC: 71.70
Epoch: 900, Train Loss: 0.0631, Valid Loss: 0.0637, Train AUC: 73.62, Valid AUC: 71.77
Epoch: 910, Train Loss: 0.0633, Valid Loss: 0.0640, Train AUC: 73.41, Valid AUC: 71.62
Epoch: 920, Train Loss: 0.0634, Valid Loss: 0.0641, Train AUC: 73.20, Valid AUC: 71.57
Epoch: 930, Train Loss: 0.0631, Valid Loss: 0.0637, Train AUC: 73.43, Valid AUC: 71.71
Epoch: 940, Train Loss: 0.0631, Valid Loss: 0.0637, Train AUC: 73.68, Valid AUC: 71.77
Epoch: 950, Train Loss: 0.0632, Valid Loss: 0.0638, Train AUC: 73.59, Valid AUC: 71.69
Epoch: 960, Train Loss: 0.0632, Valid Loss: 0.0638, Train AUC: 73.30, Valid AUC: 71.76
Epoch: 970, Train Loss: 0.0631, Valid Loss: 0.0637, Train AUC: 73.63, Valid AUC: 71.80
Epoch: 980, Train Loss: 0.0631, Valid Loss: 0.0637, Train AUC: 73.74, Valid AUC: 71.78
Epoch: 990, Train Loss: 0.0631, Valid Loss: 0.0638, Train AUC: 73.50, Valid AUC: 71.74
Epoch: 1000, Train Loss: 0.0631, Valid Loss: 0.0637, Train AUC: 73.57, Valid AUC: 71.81
"""

"""
Epoch: 100, Train Loss: 0.0654, Valid Loss: 0.0658, Train AUC: 72.23, Valid AUC: 71.15
Epoch: 200, Train Loss: 0.0637, Valid Loss: 0.0641, Train AUC: 72.74, Valid AUC: 71.53
Epoch: 300, Train Loss: 0.0633, Valid Loss: 0.0637, Train AUC: 73.02, Valid AUC: 71.70
Epoch: 400, Train Loss: 0.0632, Valid Loss: 0.0637, Train AUC: 73.34, Valid AUC: 71.83
Epoch: 500, Train Loss: 0.0633, Valid Loss: 0.0638, Train AUC: 73.57, Valid AUC: 71.81
Epoch: 600, Train Loss: 0.0631, Valid Loss: 0.0637, Train AUC: 73.38, Valid AUC: 71.82
Epoch: 700, Train Loss: 0.0632, Valid Loss: 0.0638, Train AUC: 73.66, Valid AUC: 71.69
Epoch: 800, Train Loss: 0.0632, Valid Loss: 0.0639, Train AUC: 73.58, Valid AUC: 71.66
Epoch: 900, Train Loss: 0.0631, Valid Loss: 0.0637, Train AUC: 73.43, Valid AUC: 71.71
Epoch: 1000, Train Loss: 0.0631, Valid Loss: 0.0637, Train AUC: 73.57, Vali
"""