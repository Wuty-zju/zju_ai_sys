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
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        
    def reset_parameters(self):
        for conv in [self.conv1, self.conv2]:
            conv.reset_parameters()
            
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return F.log_softmax(x, dim=-1)

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


# 主函数
def main():
    # 数据集路径和保存路径
    path = 'datasets/632d74d4e2843a53167ee9a1-momodel/'
    save_dir = 'results/'
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
    model = GraphSAGE(in_channels=in_channels, out_channels=out_channels, **sage_parameters).to(device)

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

        # 保存最优模型
        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(save_dir, '1.pt'))

        if epoch % log_steps == 0:
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, '
                  f'Train AUC: {100 * train_auc:.2f}, Valid AUC: {100 * valid_auc:.2f}')

    # 预测部分
    model.load_state_dict(torch.load(os.path.join(save_dir, '1.pt')))
    node_idx = 0
    y_pred = predict(model, data, node_idx)
    print(f'节点 {node_idx} 预测的标签为: {torch.argmax(y_pred).item()}')

if __name__ == "__main__":
    main()

