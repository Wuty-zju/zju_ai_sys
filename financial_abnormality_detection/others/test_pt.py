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
        residual = x  # 用于残差连接
        
        # 如果 in_channels != hidden_channels，调整 residual 维度
        if self.residual_transform is not None:
            residual = self.residual_transform(residual)

        # 第一层卷积
        x = self.convs[0](x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 中间层卷积
        for conv, norm in zip(self.convs[1:-1], self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_residual:
                x += residual  # 残差连接

        # 输出层卷积
        x = self.convs[-1](x, edge_index)
        
        return F.log_softmax(x, dim=-1)

def test(model, data, split_idx):
    model.eval()
    with torch.no_grad():
        node_id = split_idx['test']  # 只在测试集上进行评估
        out = model(data.x, data.edge_index).exp()  # 概率输出

        # 只预测测试集中有标签为 -100 的节点
        y_pred = out[node_id]

        return y_pred  # 返回预测结果

# 主函数
def main():
    # 数据集路径和保存路径
    path = 'financial_abnormality_detection/datasets/632d74d4e2843a53167ee9a1-momodel/'
    save_dir = 'financial_abnormality_detection/results/'
    dataset_name = 'DGraph'

    # 加载数据集
    dataset = DGraphFin(root=path, name=dataset_name)
    data = dataset[0]

    data = data.to(device)

    # 数据标准化
    x = data.x
    x = (x - x.mean(0)) / x.std(0)
    data.x = x

    # 划分数据集
    split_idx = {'train': data.train_mask, 'valid': data.valid_mask, 'test': data.test_mask}

    # GraphSAGE 模型参数
    sage_parameters = {
        'num_layers': 4,
        'hidden_channels': 100,
        'dropout': 0.6,  # 使用Dropout防止过拟合
    }

    # 初始化GraphSAGE模型
    in_channels = data.x.size(-1)
    out_channels = 2  # 类别0和类别1
    model = OptimizedGraphSAGE(in_channels=in_channels, out_channels=out_channels, **sage_parameters).to(device)

    # 加载已保存的最优模型
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_sage_model_4*100_drop_0.6.pt')))

    # 预测测试集节点
    y_pred = test(model, data, split_idx)

    # 输出测试集预测结果
    print(f'Predictions for test set: {y_pred}')

if __name__ == "__main__":
    main()