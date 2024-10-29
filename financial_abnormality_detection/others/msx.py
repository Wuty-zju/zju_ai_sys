###############################################################################
# 重要: 请务必把任务(jobs)中需要保存的文件存放在 results 文件夹内
# Important : Please make sure your files are saved to the 'results' folder
# in your jobs
###############################################################################
from utils import DGraphFin
from utils.utils import prepare_folder
from utils.evaluator import Evaluator

import torch
import torch.nn.functional as F
import torch.nn as nn

import torch_geometric.transforms as T

from torch_geometric.nn import SAGEConv

import numpy as np
from torch_geometric.data import Data
import os

#设置gpu设备
device = 0
device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
path='./datasets/632d74d4e2843a53167ee9a1-momodel/' #数据保存路径
save_dir='./results/' #模型保存路径
dataset_name='DGraph'
dataset = DGraphFin(root=path, name=dataset_name, transform=T.ToSparseTensor())

nlabels = dataset.num_classes
if dataset_name in ['DGraph']:
    nlabels = 2    #本实验中仅需预测类0和类1

data = dataset[0]
data.adj_t = data.adj_t.to_symmetric() #将有向图转化为无向图
data = data.to(device)

if dataset_name in ['DGraph']:
    x = data.x
    x = (x - x.mean(0)) / x.std(0)
    data.x = x
if data.y.dim() == 2:
    data.y = data.y.squeeze(1)

split_idx = {'train': data.train_mask, 'valid': data.valid_mask, 'test': data.test_mask}  #划分训练集，验证集

train_idx = split_idx['train']
result_dir = prepare_folder(dataset_name,'mlp')
print(data)
print(data.x.shape)  #feature
print(data.y.shape)  #label
print(dataset.edge_index.shape)

row, col, _ = data.adj_t.t().coo()
data.edge_index = torch.stack([row, col], axis=0)
data.edge_index = data.edge_index.long()
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
model = GraphSAGE(in_channels=data.x.size(-1), hidden_channels=64, out_channels=nlabels).to(device)
print(f'Model GraphSAGE initialized')

lr = 0.002
weight_decay = 5e-4
epochs = 3000
log_steps = 10 # log记录周期

eval_metric = 'auc'  #使用AUC衡量指标
evaluator = Evaluator(eval_metric)

def train(model, data, train_idx, optimizer):
     # data.y is labels of shape (N, )
    model.train()

    optimizer.zero_grad()

    out = model(data.x, data.edge_index)[train_idx]

    loss = F.nll_loss(out, data.y[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()

def test(model, data, split_idx, evaluator):
    # data.y is labels of shape (N, )
    with torch.no_grad():
        model.eval()

        losses, eval_results = dict(), dict()
        for key in ['train', 'valid']:
            node_id = split_idx[key]

            out = model(data.x, data.edge_index)[node_id]
            y_pred = out.exp()  # (N,num_classes)

            losses[key] = F.nll_loss(out, data.y[node_id]).item()
            eval_results[key] = evaluator.eval(data.y[node_id], y_pred)[eval_metric]

    return eval_results, losses, y_pred

def test_and_save_predictions(model, data, save_path):
    """
    运行模型的前向传播，并保存所有节点的预测结果。
    :param model: 训练好的模型
    :param data: 包含节点特征和边的图数据
    :param save_path: 保存预测结果的文件路径
    """
    model.eval()
    with torch.no_grad():
        # 对所有节点进行前向传播
        out = model(data.x, data.edge_index)
        y_pred = out.exp()  # 将 LogSoftmax 的输出转换为概率

    # 保存预测结果
    torch.save(y_pred, save_path)
    print(f"预测结果已保存到 {save_path}")

model.reset_parameters()
# model.reset_parameters()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
best_valid = 0
min_valid_loss = 1e8

for epoch in range(1,epochs + 1):
    loss = train(model, data, train_idx, optimizer)
    eval_results, losses, out = test(model, data, split_idx, evaluator)
    train_eval, valid_eval = eval_results['train'], eval_results['valid']
    train_loss, valid_loss = losses['train'], losses['valid']

    if valid_loss < min_valid_loss:
        min_valid_loss = valid_loss
        torch.save(model.state_dict(), save_dir+'/model.pt') #将表现最好的模型保存

    if epoch % log_steps == 0:
        print(f'Epoch: {epoch:02d}, '
              f'Loss: {loss:.4f}, '
              f'Train: {100 * train_eval:.3f}, ' # 我们将AUC值乘上100，使其在0-100的区间内
              f'Valid: {100 * valid_eval:.3f} ')
model.load_state_dict(torch.load(save_dir+'/model2.pt')) #载入验证集上表现最好的模型
# 运行模型并保存预测结果
predictions_save_path = './results/sage_predictions2.pt'
test_and_save_predictions(model, data, predictions_save_path)
def predict(data,node_id):
    """
    加载模型和模型预测
    :param node_id: int, 需要进行预测节点的下标
    :return: tensor, 类0以及类1的概率, torch.size[1,2]
    """
    # -------------------------- 实现模型预测部分的代码 ---------------------------
    with torch.no_grad():
        model.eval()
        out = model(data.x, data.edge_index)[node_id]
        y_pred = out.exp()  # (N,num_classes)

    return y_pred
dic={0:"正常用户",1:"欺诈用户"}
node_idx = 0
y_pred = predict(data, node_idx)
print(y_pred)
print(f'节点 {node_idx} 预测对应的标签为:{torch.argmax(y_pred)}, 为{dic[torch.argmax(y_pred).item()]}。')

node_idx = 1
y_pred = predict(data, node_idx)
print(y_pred)
print(f'节点 {node_idx} 预测对应的标签为:{torch.argmax(y_pred)}, 为{dic[torch.argmax(y_pred).item()]}。')
