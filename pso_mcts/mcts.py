# mcts.py

import math
import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Union
from geometry_utils import Point, Rectangle, Room, Layout
from input_handler import Constraint

class MCTSNode:
    """蒙特卡洛树搜索算法中的节点，表示一种房间布局状态。"""
    def __init__(self, 
                 parent: Optional['MCTSNode'] = None, 
                 room_type: Optional[str] = None, 
                 room_rect: Optional[Rectangle] = None):
        """
        初始化节点。
        
        Args:
            parent: 父节点
            room_type: 该节点放置的房间类型
            room_rect: 该节点放置的房间矩形区域
        """
        self.parent = parent        # 父节点
        self.children = []          # 子节点列表
        self.visits = 0             # 访问次数
        self.value = 0.0            # 节点价值
        self.room_type = room_type  # 房间类型
        self.room_rect = room_rect  # 房间位置和尺寸
        self.untried_positions = [] # 未尝试的位置列表
    
    def add_child(self, room_type: str, room_rect: Rectangle) -> 'MCTSNode':
        """添加子节点，表示放置新房间后的布局状态。"""
        child = MCTSNode(parent=self, room_type=room_type, room_rect=room_rect)
        self.children.append(child)
        return child
    
    def update(self, result: float) -> None:
        """更新节点统计信息。"""
        self.visits += 1
        self.value += result
    
    def get_uct_value(self, exploration_weight: float = 1.0) -> float:
        """
        计算UCT值，用于选择子节点。
        
        UCT = exploitation + exploration
             = value/visits + exploration_weight * sqrt(2 * ln(parent_visits) / visits)
        """
        if self.visits == 0:
            return float('inf')  # 未访问过的节点优先探索
        
        # 利用项：节点平均价值
        exploitation = self.value / self.visits
        
        # 探索项：UCB公式
        exploration = exploration_weight * math.sqrt(2 * math.log(self.parent.visits) / self.visits)
        
        return exploitation + exploration
    
    def is_fully_expanded(self) -> bool:
        """判断节点是否已完全扩展（所有可能的子节点都已生成）。"""
        return len(self.untried_positions) == 0
    
    def best_child(self, exploration_weight: float = 1.0) -> 'MCTSNode':
        """选择UCT值最高的子节点。"""
        return max(self.children, key=lambda c: c.get_uct_value(exploration_weight))
    
    def is_terminal(self, total_rooms: int) -> bool:
        """判断是否为终端节点（所有房间都已布置）。"""
        # 计算从根节点到当前节点的路径长度
        depth = 0
        node = self
        while node.parent is not None:
            depth += 1
            node = node.parent
        
        # 若路径长度等于总房间数，则为终端节点
        return depth == total_rooms


class MonteCarloTreeSearch:
    """蒙特卡洛树搜索算法，用于确定房间布局位置。"""
    def __init__(self,
                 boundary: Rectangle,
                 room_sizes: List[Tuple[str, Rectangle]],
                 constraints: Dict[str, Constraint],
                 window_facades: List[Tuple[Point, Point]],
                 entrance: Point,
                 evaluator,
                 size_modulus: float = 300.0,
                 max_simulations: int = 1000,
                 exploration_weight: float = 1.0):
        """
        初始化MCTS算法。
        
        Args:
            boundary: 户型边界
            room_sizes: 房间类型和尺寸列表，由PSO确定
            constraints: 房间约束条件
            window_facades: 可开窗的外墙边界
            entrance: 入口位置
            evaluator: 布局评价器
            size_modulus: 尺寸模数，默认为300mm
            max_simulations: 最大模拟次数
            exploration_weight: UCT中的探索权重
        """
        self.boundary = boundary
        self.room_sizes = room_sizes
        self.constraints = constraints
        self.window_facades = window_facades
        self.entrance = entrance
        self.evaluator = evaluator
        self.size_modulus = size_modulus
        self.max_simulations = max_simulations
        self.exploration_weight = exploration_weight
        
        # 创建根节点
        self.root = MCTSNode()
        
        # 初始化：放置第一个房间（通常是入口邻近的活动区域如客厅）
        self._init_first_room()
    
    def _init_first_room(self) -> None:
        """初始化第一个房间，通常选择客厅或走廊作为起始房间并放置在入口附近。"""
        # 确定起始房间类型（优先级：客厅 > 走廊 > 任意房间）
        entrance_room_type = None
        entrance_room_idx = -1
        
        # 尝试找客厅或走廊类房间
        for i, (room_type, _) in enumerate(self.room_sizes):
            if room_type.lower() in ['living_room', 'living room', 'hall']:
                entrance_room_type = room_type
                entrance_room_idx = i
                break
        
        # 如果没找到，以第一个房间作为入口房间
        if entrance_room_type is None:
            entrance_room_type, _ = self.room_sizes[0]
            entrance_room_idx = 0
        
        # 获取该房间的尺寸
        _, entrance_room_rect = self.room_sizes[entrance_room_idx]
        
        # 复制一个新的矩形对象，避免修改原始数据
        entrance_room_rect = Rectangle(
            0, 0, entrance_room_rect.width, entrance_room_rect.height
        )
        
        # 确定入口房间的位置，使其靠近入口点
        entrance_room_rect.x = max(0, min(
            self.entrance.x - entrance_room_rect.width / 2,
            self.boundary.x + self.boundary.width - entrance_room_rect.width
        ))
        
        entrance_room_rect.y = max(0, min(
            self.entrance.y - entrance_room_rect.height / 2,
            self.boundary.y + self.boundary.height - entrance_room_rect.height
        ))
        
        # 创建第一个子节点
        self.root.add_child(entrance_room_type, entrance_room_rect)
    
    def search(self) -> Layout:
        """
        执行MCTS搜索，寻找最佳房间布局。
        
        Returns:
            最佳布局
        """
        total_rooms = len(self.room_sizes)
        
        # 执行指定次数的模拟
        for i in range(self.max_simulations):
            # 1. 选择阶段：选择最有价值的节点进行扩展
            node = self._select(self.root, total_rooms)
            
            # 2. 扩展阶段：为选中的节点添加一个新的子节点
            if not node.is_terminal(total_rooms):
                node = self._expand(node)
            
            # 3. 模拟阶段：从新节点开始，随机完成剩余布局，并评估
            result = self._simulate(node, total_rooms)
            
            # 4. 回溯阶段：更新节点统计信息
            self._backpropagate(node, result)
            
            # 定期打印进度
            if (i + 1) % 100 == 0 or i == 0:
                print(f"MCTS进度: {i+1}/{self.max_simulations} 模拟次数")
        
        # 返回找到的最佳布局
        best_layout = self._get_best_layout()
        return best_layout
    
    def _select(self, node: MCTSNode, total_rooms: int) -> MCTSNode:
        """
        选择阶段：从根节点开始，使用UCT值选择子节点，直到找到未完全扩展的节点或终端节点。
        
        Args:
            node: 当前节点
            total_rooms: 总房间数
            
        Returns:
            选中的节点
        """
        while not node.is_terminal(total_rooms) and node.children and node.is_fully_expanded():
            node = node.best_child(self.exploration_weight)
        
        return node
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        扩展阶段：为选中的节点添加一个新的子节点（放置下一个房间）。
        
        Args:
            node: 待扩展的节点
            
        Returns:
            新添加的子节点
        """
        # 获取当前已放置的房间
        placed_rooms = self._get_placed_rooms(node)
        placed_room_types = [room.room_type for room in placed_rooms]
        
        # 找出下一个要放置的房间类型和尺寸
        next_room_idx = -1
        next_room_type = None
        next_room_rect = None
        
        for i, (room_type, rect) in enumerate(self.room_sizes):
            if room_type not in placed_room_types:
                next_room_idx = i
                next_room_type = room_type
                # 复制尺寸信息，位置稍后确定
                next_room_rect = Rectangle(0, 0, rect.width, rect.height)
                break
        
        if next_room_type is None:
            # 所有房间都已放置，应该不会发生
            return node
        
        # 如果未尝试的位置列表为空，则生成可能的位置
        if not node.untried_positions:
            node.untried_positions = self._generate_valid_positions(
                placed_rooms, next_room_type, next_room_rect
            )
        
        if not node.untried_positions:
            # 如果没有有效位置，则返回当前节点
            return node
        
        # 随机选择一个位置
        x, y = random.choice(node.untried_positions)
        node.untried_positions.remove((x, y))
        
        # 设置房间位置
        next_room_rect.x = x
        next_room_rect.y = y
        
        # 添加子节点
        new_node = node.add_child(next_room_type, next_room_rect)
        return new_node
    
    def _simulate(self, node: MCTSNode, total_rooms: int) -> float:
        """
        模拟阶段：从当前节点开始，随机放置剩余房间，并评估布局。
        
        Args:
            node: 模拟起点节点
            total_rooms: 总房间数
            
        Returns:
            布局评分，范围[0, 100]
        """
        # 获取当前已放置的房间
        placed_rooms = self._get_placed_rooms(node)
        placed_room_types = [room.room_type for room in placed_rooms]
        
        # 获取剩余待放置的房间
        remaining_rooms = []
        for room_type, rect in self.room_sizes:
            if room_type not in placed_room_types:
                # 复制尺寸信息，位置稍后确定
                remaining_rooms.append((room_type, Rectangle(0, 0, rect.width, rect.height)))
        
        # 随机放置剩余房间
        simulation_rooms = placed_rooms.copy()
        for room_type, rect in remaining_rooms:
            # 生成有效位置
            valid_positions = self._generate_valid_positions(simulation_rooms, room_type, rect)
            
            if not valid_positions:
                # 无法放置所有房间，布局无效
                return 0.0
            
            # 随机选择一个位置
            x, y = random.choice(valid_positions)
            rect.x = x
            rect.y = y
            
            # 添加到模拟房间列表
            simulation_rooms.append(Room(room_type, rect))
        
        # 创建布局并评估
        layout = Layout(simulation_rooms, self.boundary)
        score = self.evaluator.evaluate(layout)
        
        return score
    
    def _backpropagate(self, node: MCTSNode, result: float) -> None:
        """
        回溯阶段：从当前节点开始，更新所有祖先节点的统计信息。
        
        Args:
            node: 当前节点
            result: 模拟结果评分
        """
        while node is not None:
            node.update(result)
            node = node.parent
    
    def _get_placed_rooms(self, node: MCTSNode) -> List[Room]:
        """
        获取从根节点到当前节点路径上的所有房间，即当前布局中已放置的房间。
        
        Args:
            node: 当前节点
            
        Returns:
            已放置的房间列表
        """
        rooms = []
        current = node
        
        # 从当前节点向上追溯到根节点
        while current.parent is not None:
            rooms.append(Room(current.room_type, current.room_rect))
            current = current.parent
        
        # 反转列表，使其从第一个放置的房间开始
        return rooms[::-1]
    
    def _generate_valid_positions(self, 
                                placed_rooms: List[Room], 
                                next_room_type: str, 
                                next_room_rect: Rectangle) -> List[Tuple[float, float]]:
        """
        生成下一个房间的有效位置列表，使用节点剪枝方法减少搜索空间。
        
        Args:
            placed_rooms: 已放置的房间列表
            next_room_type: 下一个要放置的房间类型
            next_room_rect: 下一个要放置的房间尺寸
            
        Returns:
            有效位置坐标列表[(x1,y1), (x2,y2), ...]
        """
        # 获取房间约束
        constraint = self.constraints.get(next_room_type)
        if not constraint:
            # 如果没有该房间类型的约束，使用默认值
            constraint = Constraint(room_type=next_room_type)
        
        # 创建可能位置的网格点（按尺寸模数对齐）
        grid_positions = []
        for x in range(int(self.boundary.x), 
                     int(self.boundary.x + self.boundary.width - next_room_rect.width) + 1, 
                     int(self.size_modulus)):
            for y in range(int(self.boundary.y), 
                         int(self.boundary.y + self.boundary.height - next_room_rect.height) + 1, 
                         int(self.size_modulus)):
                grid_positions.append((x, y))
        
        # 应用剪枝规则，筛选有效位置
        valid_positions = []
        for x, y in grid_positions:
            temp_rect = Rectangle(x, y, next_room_rect.width, next_room_rect.height)
            temp_room = Room(next_room_type, temp_rect)
            
            # 应用剪枝规则检查位置是否有效
            if self._is_valid_position(temp_room, placed_rooms, constraint):
                valid_positions.append((x, y))
        
        return valid_positions
    
    def _is_valid_position(self, room: Room, placed_rooms: List[Room], constraint: Constraint) -> bool:
        """
        检查位置是否有效，实现节点剪枝规则。
        
        Args:
            room: 待检查的房间
            placed_rooms: 已放置的房间列表
            constraint: 房间约束条件
            
        Returns:
            位置是否有效
        """
        # 规则1：房间必须在边界内
        if not self._is_inside_boundary(room.rectangle):
            return False
        
        # 规则2：房间不能与已放置的房间重叠
        if any(self._has_overlap(room.rectangle, placed_room.rectangle) for placed_room in placed_rooms):
            return False
        
        # 规则3：检查相邻关系要求
        if constraint.connection and not self._check_adjacency(room, placed_rooms, constraint.connection):
            return False
        
        # 规则4：检查朝向要求
        if constraint.orientation and not self._check_orientation(room, constraint.orientation):
            return False
        
        # 规则5：检查开窗要求
        if constraint.window_access and not self._check_window_access(room):
            return False
        
        # 规则6：检查长宽比例要求（该规则在PSO优化尺寸时已应用）
        
        # 通过所有检查
        return True
    
    def _is_inside_boundary(self, rect: Rectangle) -> bool:
        """检查矩形是否在边界内。"""
        return (self.boundary.x <= rect.x and
                self.boundary.y <= rect.y and
                rect.x + rect.width <= self.boundary.x + self.boundary.width and
                rect.y + rect.height <= self.boundary.y + self.boundary.height)
    
    def _has_overlap(self, rect1: Rectangle, rect2: Rectangle) -> bool:
        """检查两个矩形是否重叠。"""
        return rect1.intersects(rect2)
    
    def _check_adjacency(self, room: Room, placed_rooms: List[Room], required_connections: List[str]) -> bool:
        """
        检查相邻关系约束，至少要与其中一个需要相邻的房间相邻。
        
        如果需要相邻的房间类型还未放置，则不进行检查。
        """
        # 检查已放置的房间类型列表
        placed_types = [room.room_type for room in placed_rooms]
        
        # 需要相邻的房间类型中，已放置的类型
        required_placed_types = [t for t in required_connections if t in placed_types]
        
        # 如果没有需要相邻的已放置房间，则跳过检查
        if not required_placed_types:
            return True
        
        # 检查是否与至少一个需要相邻的房间相邻
        for placed_room in placed_rooms:
            if placed_room.room_type in required_connections:
                if room.is_adjacent_to(placed_room):
                    return True
        
        return False
    
    def _check_orientation(self, room: Room, orientation: str) -> bool:
        """检查房间是否满足朝向要求。"""
        # 判断房间是否靠近对应朝向的边界
        rect = room.rectangle
        boundary = self.boundary
        tolerance = self.size_modulus  # 允许误差
        
        if orientation == 'north':
            return abs(rect.y + rect.height - (boundary.y + boundary.height)) < tolerance
        elif orientation == 'east':
            return abs(rect.x + rect.width - (boundary.x + boundary.width)) < tolerance
        elif orientation == 'south':
            return abs(rect.y - boundary.y) < tolerance
        elif orientation == 'west':
            return abs(rect.x - boundary.x) < tolerance
        
        # 未知朝向
        return True
    
    def _check_window_access(self, room: Room) -> bool:
        """检查房间是否满足开窗要求，即至少有一边靠近可开窗的外墙。"""
        # 获取房间的四条边
        rect = room.rectangle
        room_edges = [
            (Point(rect.x, rect.y), Point(rect.x + rect.width, rect.y)),  # 下边
            (Point(rect.x + rect.width, rect.y), Point(rect.x + rect.width, rect.y + rect.height)),  # 右边
            (Point(rect.x + rect.width, rect.y + rect.height), Point(rect.x, rect.y + rect.height)),  # 上边
            (Point(rect.x, rect.y + rect.height), Point(rect.x, rect.y))  # 左边
        ]
        
        # 检查是否有边靠近窗户外墙
        for room_edge in room_edges:
            for facade in self.window_facades:
                if self._edges_are_close(room_edge, facade):
                    return True
        
        return False
    
    def _edges_are_close(self, edge1: Tuple[Point, Point], edge2: Tuple[Point, Point]) -> bool:
        """
        检查两条边是否靠近，用于判断房间是否满足开窗条件。
        
        Args:
            edge1: 第一条边，表示为(Point, Point)
            edge2: 第二条边，表示为(Point, Point)
            
        Returns:
            两边是否靠近
        """
        # 计算边的中点
        mid1_x = (edge1[0].x + edge1[1].x) / 2
        mid1_y = (edge1[0].y + edge1[1].y) / 2
        mid2_x = (edge2[0].x + edge2[1].x) / 2
        mid2_y = (edge2[0].y + edge2[1].y) / 2
        
        # 计算中点距离
        dist = math.sqrt((mid1_x - mid2_x) ** 2 + (mid1_y - mid2_y) ** 2)
        
        # 距离小于阈值则认为靠近
        return dist < self.size_modulus
    
    def _get_best_layout(self) -> Layout:
        """获取搜索得到的最佳布局，选择访问次数最多的路径。"""
        rooms = []
        current = self.root
        
        # 从根节点开始，每次选择访问次数最多的子节点
        while current.children:
            current = max(current.children, key=lambda c: c.visits)
            rooms.append(Room(current.room_type, current.room_rect))
        
        # 创建布局
        return Layout(rooms, self.boundary)
