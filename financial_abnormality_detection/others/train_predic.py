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

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Data loading and preprocessing
path = './datasets/632d74d4e2843a53167ee9a1-momodel/'  # Data save path
save_dir = './results/'  # Model save path
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
dataset_name = 'DGraph'

# Load dataset
dataset = DGraphFin(root=path, name=dataset_name, transform=T.ToSparseTensor())

nlabels = 2  # Only need to predict class 0 and class 1

data = dataset[0]
data.adj_t = data.adj_t.to_symmetric()  # Convert directed graph to undirected graph

# Data preprocessing
x = data.x
x = (x - x.mean(0)) / x.std(0)  # Standardize features
data.x = x
if data.y.dim() == 2:
    data.y = data.y.squeeze(1)

split_idx = {'train': data.train_mask, 'valid': data.valid_mask, 'test': data.test_mask}

train_idx = split_idx['train']

# Move data to device
data = data.to(device)

# Convert adj_t to edge_index for SAGEConv
row, col, _ = data.adj_t.coo()
data.edge_index = torch.stack([row, col], dim=0)

# Model definition
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        # Define three SAGEConv layers
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        
        # Linear layers for residual connections
        self.res1 = nn.Linear(in_channels, hidden_channels) if in_channels != hidden_channels else None
        self.res2 = nn.Linear(hidden_channels, hidden_channels)
        
    def reset_parameters(self):
        # Reset parameters
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        if self.res1:
            self.res1.reset_parameters()
        self.res2.reset_parameters()
        
    def forward(self, x, edge_index):
        # First convolution + residual connection
        identity = x
        x = F.relu(self.conv1(x, edge_index))
        if self.res1:
            identity = self.res1(identity)  # Map to hidden_channels dimensions
        x1 = x + identity  # Residual connection
        
        # Second convolution + residual connection
        identity = x1
        x = F.relu(self.conv2(x1, edge_index))
        x2 = x + self.res2(identity)  # Residual connection
        
        # Third convolution (output layer)
        x3 = self.conv3(x2, edge_index)
        
        # Apply log_softmax to get probabilities
        return F.log_softmax(x3, dim=-1)

# Model initialization and hyperparameters
in_channels = data.x.size(-1)  # Input feature dimension
hidden_channels = 8           # Hidden layer dimension
out_channels = nlabels         # Number of classes

# Instantiate model
model = GraphSAGE(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels).to(device)

# Training hyperparameters
epochs = 1000
lr = 0.005
weight_decay = 2e-4

# Optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# Evaluator
eval_metric = 'auc'
evaluator = Evaluator(eval_metric)

# Training and evaluation functions
def train(model, data, train_idx, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[train_idx], data.y[train_idx])
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, data, split_idx, evaluator):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        y_pred = out.exp()
        eval_results = {}
        losses = {}
        for key in ['train', 'valid']:
            node_id = split_idx[key]
            losses[key] = F.nll_loss(out[node_id], data.y[node_id]).item()
            eval_results[key] = evaluator.eval(data.y[node_id], y_pred[node_id])[eval_metric]
    return eval_results, losses

# Training loop
best_valid_auc = 0
best_model_state = None
for epoch in range(1, epochs+1):
    loss = train(model, data, train_idx, optimizer)
    eval_results, losses = test(model, data, split_idx, evaluator)
    train_auc = eval_results['train']
    valid_auc = eval_results['valid']
    if valid_auc > best_valid_auc:
        best_valid_auc = valid_auc
        best_model_state = model.state_dict()
        # Save the best model
        model_filename = f'best_sage_model_conv3_hidden{hidden_channels}_lr{lr}_wd{weight_decay}.pt'
        torch.save(best_model_state, os.path.join(save_dir, model_filename))
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:04d}, Loss: {loss:.4f}, Train AUC: {train_auc*100:.2f}%, Valid AUC: {valid_auc*100:.2f}%')

print("Training completed.")
print(f"Best validation AUC: {best_valid_auc*100:.2f}%")
print(f"Best model saved to {os.path.join(save_dir, model_filename)}")

# Load best model
model.load_state_dict(torch.load(os.path.join(save_dir, model_filename), map_location=device))

# Define predict function
def predict(data, node_id):
    """
    加载模型和模型预测
    :param data: 数据对象，包含x和edge_index等属性
    :param node_id: int，需要进行预测节点的下标
    :return: tensor，类0以及类1的概率，torch.size[1,2]
    """
    # Ensure the model is in eval mode
    model.eval()
    # Move data to device if not already
    if data.x.device != device:
        data = data.to(device)
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        y_pred = out[node_id].exp().unsqueeze(0)  # (1, 2)
    return y_pred

# Example usage
node_id = 0  # Change this to the node you want to predict
y_pred = predict(data, node_id)
print(f"Prediction for node {node_id}: {y_pred}")

# Add the test_and_save_predictions function
def test_and_save_predictions(model, data, save_path):
    """
    运行模型的前向传播，并保存所有节点的预测结果。
    :param model: 训练好的模型
    :param data: 包含节点特征和边的图数据
    :param save_path: 保存预测结果的文件路径
    """
    model.eval()
    with torch.no_grad():
        # Run forward pass on all nodes
        out = model(data.x, data.edge_index)
        y_pred = out.exp()  # Convert LogSoftmax output to probabilities

    # Save the predictions
    torch.save(y_pred.cpu(), save_path)
    print(f"Predictions saved to {save_path}")

# Run the model and save predictions
predictions_save_path = os.path.join(save_dir, f'best_sage_model_conv3_hidden{hidden_channels}_lr{lr}_wd{weight_decay}_predictions.pt')
test_and_save_predictions(model, data, predictions_save_path)