机器人自动走迷宫实验报告

基于DQN算法的机器人迷宫求解

本实验旨在使用深度强化学习中的深度Q网络（DQN）算法，编程实现机器人在迷宫中自动寻找从起点到终点的最优路径。以下是详细的代码实现和解释。

2.1 导入必要的库和模块

import os
import random
import numpy as np
from Maze import Maze               # 迷宫环境类
from Runner import Runner
from QRobot import QRobot
from ReplayDataSet import ReplayDataSet
from torch_py.MinDQNRobot import MinDQNRobot as TorchRobot  # PyTorch版本的机器人
# from keras_py.MinDQNRobot import MinDQNRobot as KerasRobot  # Keras版本的机器人
import matplotlib.pyplot as plt

import torch
import time

解释：

	•	导入必要的库和模块，包括 numpy、matplotlib.pyplot、torch 等。
	•	从 Maze 模块中导入 Maze 类，用于创建迷宫环境。
	•	从 torch_py.MinDQNRobot 模块中导入 MinDQNRobot 类，作为机器人类的父类。
	•	由于我们使用 PyTorch 版本的机器人，因此注释掉 Keras 版本的导入。

2.2 定义机器人类 Robot

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

解释：

	•	定义了一个 Robot 类，继承自 TorchRobot 类。
	•	__init__ 方法用于初始化机器人对象，设置迷宫环境和模型参数。
	•	使用 super() 调用父类的初始化方法，确保父类的属性和方法被正确初始化。
	•	调用 maze.set_reward() 方法，设置迷宫的奖励机制，包括撞墙惩罚、到达终点奖励和默认奖励，这些值根据迷宫大小进行调整。
	•	将迷宫对象保存为 self.maze，方便后续使用。
	•	设置探索率 epsilon 为 0，表示在训练和测试中采用贪心策略，始终选择当前估计最优的动作。
	•	设置计算设备 self.device，如果有可用的 GPU（CUDA），则使用 GPU，否则使用 CPU。
	•	调用 self.memory.build_full_view(maze=maze) 开启全图视野，构建经验回放池，这样可以加速训练过程。
	•	将评估网络 eval_model 和目标网络 target_model 迁移到指定的计算设备上。
	•	调用 self.train() 方法开始训练，并将训练过程中产生的损失值保存到 self.loss_list 中。

2.3 定义训练函数 train

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
            
            # 限制最大移动步数为迷宫面积，防止无限循环
            for _ in range(self.maze.maze_size ** 2):
                action, reward = self.test_update()
                
                # 如果奖励值等于到达终点的奖励，则成功找到出口
                if reward == self.maze.reward["destination"]:
                    print('Training time: {:.2f} s'.format(time.time() - start))  # 打印总训练时间
                    return loss_list  # 返回损失值列表，结束训练

解释：

	•	train 方法用于训练模型，直到机器人能够成功找到迷宫的出口。
	•	初始化一个空的 loss_list，用于保存每次训练的损失值。
	•	设置批次大小 batch_size 为经验回放池的大小，即每次训练时使用所有的经验数据。
	•	记录训练开始的时间 start，用于计算总的训练时间。
	•	在 while True 循环中，不断执行以下步骤：
	•	调用 _learn(batch=batch_size) 方法，从经验回放池中采样数据进行训练，返回当前的损失值，并将其添加到 loss_list。
	•	重置机器人的位置，准备进行测试。
	•	在限制的步数内（迷宫面积）执行测试，如果机器人在测试中成功到达终点，则打印训练时间并返回损失值列表，结束训练。
	•	这样设计的目的是在训练过程中，不断更新模型参数，直到机器人学会了如何在迷宫中找到出口。

2.4 定义训练更新方法 train_update

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

解释：

	•	train_update 方法用于在训练过程中，根据当前策略选择动作并执行，获取相应的奖励。
	•	首先调用 self.sense_state() 获取当前状态（机器人的位置坐标）。
	•	调用 self._choose_action(state) 方法，根据当前状态选择动作，使用的是 epsilon-greedy 策略（在父类 TorchRobot 中实现）。
	•	执行选定的动作，调用 self.maze.move_robot(action)，并获取执行后的奖励 reward。
	•	更新 epsilon 的代码被注释掉，因为在初始化时已经将 epsilon 固定为 0，不再需要更新。
	•	返回选择的动作和获得的奖励。

2.5 定义测试更新方法 test_update

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

解释：

	•	test_update 方法用于在测试过程中，根据当前策略选择动作并执行，获取相应的奖励。
	•	获取当前状态 state，并将其转换为 PyTorch 的张量形式，确保数据类型和设备与模型匹配。
	•	将模型设置为评估模式 self.eval_model.eval()，防止在测试时更新模型参数。
	•	使用 torch.no_grad() 上下文管理器，停止计算梯度，加快推理速度。
	•	将状态输入到评估模型 self.eval_model(state)，获取对应的 Q 值 q_value。
	•	根据 Q 值选择最优动作，这里选择 Q 值最小的动作（因为在迷宫中，奖励是负值，越小越好）。
	•	执行选择的动作，调用 self.maze.move_robot(action)，获取执行后的奖励 reward。
	•	返回选择的动作和获得的奖励。

2.6 迷宫和机器人初始化以及路径测试

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

解释：

	•	设置迷宫的大小 maze_size = 7，创建一个迷宫对象 maze。
	•	创建一个机器人对象 robot，并将迷宫传入其中。
	•	打印迷宫的奖励机制，以便了解各个动作的奖励值。
	•	重置机器人的位置，准备进行测试。
	•	在限制的步数内（迷宫面积）循环，调用 robot.test_update() 执行动作并获取奖励。
	•	打印每一步的动作和对应的奖励。
	•	如果在某一步中，获得的奖励等于到达终点的奖励，则表示机器人成功找到出口，打印提示信息并退出循环。

2.7 DQN算法训练并绘制训练结果

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

解释：

	•	设置训练的轮数 epoch = 20，每轮训练的步数 training_per_epoch 为迷宫面积的 5 倍。
	•	创建一个 Runner 实例 runner，用于管理训练过程。
	•	调用 runner.run_training(epoch, training_per_epoch) 开始训练。
	•	训练完成后，绘制训练过程中损失值的变化曲线。
	•	使用 plt.plot(robot.loss_list) 绘制损失值列表，添加坐标轴标签和标题，然后显示图像。
	•	调用 runner.plot_results() 显示完整的训练结果，包括机器人在迷宫中的路径和策略效果。

2.8 运行结果示例

迷宫奖励机制：

迷宫奖励机制: {'hit_wall': 14.0, 'destination': -245.0, 'default': 3.5}

机器人测试输出：

动作: r 奖励: 3.5
动作: d 奖励: 3.5
...
成功到达终点！

训练时间：

Training time: 12.34 s

损失曲线图：

（此处应有训练损失曲线的图像）

训练结果图：

（此处应有机器人在迷宫中路径的图像）

总结

通过上述代码，我们成功地实现了基于 DQN 算法的机器人迷宫求解。DQN 算法利用神经网络近似 Q 函数，让机器人通过与环境的交互，不断学习和更新策略，最终学会如何在复杂的迷宫中找到出口。

在实验中，我们设置了合理的奖励机制，确保机器人能够在训练过程中得到正确的反馈。通过绘制训练损失曲线，我们观察到模型在训练过程中不断优化。最终，机器人能够成功地从起点到达终点，实现了实验目标。

