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

# ==================== 移动方向字典 ====================
move_map = {
    'u': (-1, 0),  # 向上移动
    'r': (0, 1),   # 向右移动
    'd': (1, 0),   # 向下移动
    'l': (0, -1)   # 向左移动
}

# ==================== 搜索树节点类 ====================
class SearchTree:
    def __init__(self, loc=(), action='', parent=None):
        """
        初始化搜索树节点对象。
        
        参数:
            loc (tuple): 当前节点的位置坐标 (x, y)。
            action (str): 到达该节点的动作（'u', 'r', 'd', 'l'）。
            parent (SearchTree): 该节点的父节点。
        """
        self.loc = loc                 # 当前节点的位置
        self.to_this_action = action   # 到达该节点的动作
        self.parent = parent           # 父节点
        self.children = []             # 子节点列表
        self.g = 0                     # 到达该节点的实际代价（步数）
        self.f = 0                     # 启发式估值 f = g + h

    def add_child(self, child):
        """
        添加子节点。
        
        参数:
            child (SearchTree): 待添加的子节点。
        """
        self.children.append(child)

    def is_leaf(self):
        """
        判断当前节点是否是叶子节点。
        
        返回:
            bool: 如果没有子节点，则返回 True。
        """
        return len(self.children) == 0

# ==================== 启发式函数 ====================
def heuristic_manhattan(curr, goal):
    """
    计算当前位置到目标点的曼哈顿距离。
    
    参数:
        curr (tuple): 当前坐标 (x, y)。
        goal (tuple): 目标坐标 (x, y)。
    
    返回:
        int: 曼哈顿距离。
    """
    return abs(curr[0] - goal[0]) + abs(curr[1] - goal[1])

# ==================== 节点扩展函数 ====================
def expand_a_star(maze, closed_set, node, goal, open_list):
    """
    扩展当前节点的所有可行子节点，并将新节点加入 open_list。
    
    参数:
        maze (Maze): 迷宫对象。
        closed_set (set): 已访问节点的集合。
        node (SearchTree): 当前待扩展节点。
        goal (tuple): 目标点坐标。
        open_list (list): 存放待扩展节点的开放列表。
    """
    # 获取从当前节点可以采取的有效动作
    valid_actions = maze.can_move_actions(node.loc)
    for a in valid_actions:
        # 计算新节点的位置
        new_loc = tuple(node.loc[i] + move_map[a][i] for i in range(2))
        
        # 如果新位置没有被访问过，将其添加到 open_list
        if new_loc not in closed_set:
            g_new = node.g + 1  # 更新 g 值（当前步数 + 1）
            h_new = heuristic_manhattan(new_loc, goal)  # 启发式估值
            f_new = g_new + h_new  # f = g + h

            # 创建新节点并设置 g 和 f 值
            child = SearchTree(loc=new_loc, action=a, parent=node)
            child.g = g_new
            child.f = f_new

            # 将新节点加入开放列表
            open_list.append(child)

# ==================== 路径回溯函数 ====================
def back_propagation_a_star(node):
    """
    回溯路径，从目标节点回溯到起始节点，生成路径列表。
    
    参数:
        node (SearchTree): 目标节点。
    
    返回:
        list: 从起点到目标点的路径（动作序列）。
    """
    path = []
    while node.parent is not None:
        path.insert(0, node.to_this_action)  # 在路径开头插入动作
        node = node.parent  # 回溯到父节点
    return path

# ==================== A*搜索算法 ====================
def my_search(maze):
    """
    使用 A* 算法在迷宫中搜索从起点到终点的最短路径。
    
    参数:
        maze (Maze): 待搜索的迷宫对象。
    
    返回:
        list: 最优路径上的动作序列（如果未找到路径，则返回空列表）。
    """
    # 初始化起点和目标点
    start = maze.sense_robot()
    goal = maze.destination
    
    # 初始化起点节点并设置初始 g 值和 f 值
    root = SearchTree(loc=start)
    root.g = 0  # 起点的 g 值为 0
    root.f = heuristic_manhattan(start, goal)  # 计算起点的 f 值

    # 创建开放列表和已访问节点集合
    open_list = [root]
    closed_set = set()

    # 主循环：每次从 open_list 中选取 f 值最小的节点进行扩展
    while open_list:
        # 找到 f 值最小的节点并将其从 open_list 中移除
        current_node = min(open_list, key=lambda x: x.f)
        open_list.remove(current_node)

        # 如果当前节点是目标点，回溯路径
        if current_node.loc == goal:
            return back_propagation_a_star(current_node)

        # 将当前节点的位置添加到 closed_set 中，表示该节点已访问
        closed_set.add(current_node.loc)

        # 扩展当前节点的所有子节点，并将有效子节点加入 open_list
        expand_a_star(maze, closed_set, current_node, goal, open_list)

    # 如果开放列表为空且未找到路径，返回空列表
    return []

# ==================== 测试搜索算法 ====================
# 初始化迷宫，设置迷宫大小为10x10
maze = Maze(maze_size=10)  # 从文件或预设生成迷宫对象

# 使用 A* 搜索算法在迷宫中寻找从起点到终点的最短路径
path_2 = my_search(maze)
print("搜索出的路径：", path_2)  # 打印搜索到的路径（动作序列）

# 根据 A* 算法返回的路径依次移动机器人
for action in path_2:
    maze.move_robot(action)  # 执行动作，将机器人在迷宫中移动

# 检查机器人是否到达目标位置
if maze.sense_robot() == maze.destination:
    print("恭喜你，到达了目标点")  # 打印成功信息

# 输出当前迷宫的状态，包括机器人的位置和路径
print(maze)
