import os
import random
import numpy as np
from Maze import Maze
from Runner import Runner
from QRobot import QRobot
from ReplayDataSet import ReplayDataSet
from torch_py.MinDQNRobot import MinDQNRobot as TorchRobot # PyTorch版本
from keras_py.MinDQNRobot import MinDQNRobot as KerasRobot # Keras版本
import matplotlib.pyplot as plt

import torch
import time

# ==================== 机器人类定义 ====================
class Robot(TorchRobot):
    def __init__(self, maze):
        """
        初始化 Robot 类，用于在迷宫中执行训练和测试。

        参数:
            maze (Maze): 迷宫对象，用于机器人移动和奖励机制。
        """
        # 调用父类的初始化方法
        super(Robot, self).__init__(maze)
        
        # 设置迷宫的奖励值，确保在较大迷宫中，奖励值足够大以引导机器人到达终点
        maze.set_reward(reward={
            "hit_wall": maze.maze_size * 2.,              # 撞墙时的惩罚，随迷宫大小而增大
            "destination": -maze.maze_size ** 2 * 5.,     # 到达终点的奖励，基于迷宫大小调整
            "default": maze.maze_size * 0.5,              # 每步的默认奖励，微小正值，随迷宫大小而增大
        })
        
        # 记录迷宫对象
        self.maze = maze
        
        # 设置初始探索率 epsilon 为 0，表示完全利用策略
        self.epsilon = 0
        
        # 设置计算设备为 CUDA 或 CPU，加速计算
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 开启全图视野（“金手指”），构建训练的经验回放池
        self.memory.build_full_view(maze=maze)
        
        # 将模型迁移到所选设备上
        self.eval_model = self.eval_model.to(self.device)
        self.target_model = self.target_model.to(self.device)  # 确保目标网络也迁移到同一设备上
        
        # 开始训练并记录训练过程中产生的损失值
        self.loss_list = self.train()

    # ==================== 训练函数 ====================
    def train(self):
        """
        训练模型直到机器人能够成功找到迷宫出口。

        返回:
            list: 训练过程中产生的损失值列表。
        """
        # 初始化损失值列表
        loss_list = []
        
        # 设置批次大小为经验回放池的大小
        batch_size = len(self.memory)
        
        # 记录训练开始时间
        start = time.time()
        
        # 不断训练直至机器人成功走出迷宫
        while True:
            # 从回放池中采样训练，并记录当前轮次的损失值
            loss = self._learn(batch=batch_size)
            loss_list.append(loss)
            
            # 每轮训练后重置机器人位置，测试是否成功到达终点
            self.reset()
            
            # 限制最大移动步数为迷宫面积
            for _ in range(self.maze.maze_size ** 2):
                action, reward = self.test_update()
                
                # 如果奖励值等于到达终点的奖励，则成功找到出口
                if reward == self.maze.reward["destination"]:
                    print('Training time: {:.2f} s'.format(time.time() - start))  # 打印总训练时间
                    return loss_list  # 返回损失值列表，结束训练

    # ==================== 训练更新方法 ====================
    def train_update(self):
        """
        训练过程中调用的更新方法，用于选择动作并获得对应的奖励。

        返回:
            tuple: (action, reward)，选择的动作及执行后的奖励。
        """
        # 获取当前状态（机器人的位置）
        state = self.sense_state()
        
        # 根据当前状态选择动作，遵循 epsilon-greedy 策略（TorchRobot 中实现）
        action = self._choose_action(state)
        
        # 执行选择的动作，并获取对应的奖励
        reward = self.maze.move_robot(action)

        # 更新 epsilon 值以逐步减少探索（注释掉，因为 epsilon 固定为 0）
        # self.epsilon = max(0.01, self.epsilon * 0.995)

        # 返回动作和对应的奖励
        return action, reward

    # ==================== 测试更新方法 ====================
    def test_update(self):
        """
        测试过程中调用的更新方法，使用当前策略选择动作并执行。

        返回:
            tuple: (action, reward)，选择的动作及执行后的奖励。
        """
        # 获取当前状态，并转换为 PyTorch 张量，确保与模型输入格式一致
        state = np.array(self.sense_state(), dtype=np.int16)
        state = torch.from_numpy(state).float().to(self.device)  # 确保数据在同一设备上

        # 设置模型为评估模式，停止参数更新
        self.eval_model.eval()
        
        # 停止计算图，以加快推理速度
        with torch.no_grad():
            # 通过模型获取当前状态对应的 Q 值
            q_value = self.eval_model(state).cpu().data.numpy()

        # 选择 Q 值最小的动作（TorchRobot 的规则），返回动作
        action = self.valid_action[np.argmin(q_value).item()]
        
        # 执行选择的动作，并获取对应的奖励
        reward = self.maze.move_robot(action)
        
        # 返回动作和奖励
        return action, reward


# ==================== 迷宫和机器人初始化和路径测试 ====================
maze_size = 7  # 设置迷宫大小
maze = Maze(maze_size=maze_size)
robot = Robot(maze=maze)

# 打印当前迷宫的奖励机制，观察不同动作对应的奖励值
print("迷宫奖励机制:", robot.maze.reward)

# 测试机器人是否能根据当前策略找到终点
robot.reset()  # 重置机器人的位置
for _ in range(maze.maze_size ** 2):  # 限制最大移动步数，避免死循环
    action, reward = robot.test_update()  # 执行动作并获取奖励
    print("动作:", action, "奖励:", reward)
    
    # 如果获得终点奖励，表示成功找到出口
    if reward == maze.reward["destination"]:
        print("成功到达终点！")
        break

# ==================== DQN算法训练并绘制训练结果 ====================
epoch = 20  # 训练轮数
training_per_epoch = maze_size * maze_size * 5  # 每轮训练的步数
runner = Runner(robot)  # 创建 Runner 实例进行训练管理
runner.run_training(epoch, training_per_epoch)  # 开始训练

plt.plot(robot.loss_list)  # 绘制损失列表
plt.xlabel("Steps")        # x轴标签为“步数”
plt.ylabel("Loss")         # y轴标签为“损失”
plt.title("Training Loss Over Time")  # 图标题为“训练损失变化曲线”
plt.show()  # 显示绘图

runner.plot_results()  # 显示完整训练结果，包括路径规划和策略效果等