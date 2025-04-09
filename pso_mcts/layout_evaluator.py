# layout_evaluator.py

from typing import List, Dict, Tuple, Optional, Union, Set
from geometry_utils import Rectangle, Polygon, Point, Room, Layout, calculate_distance
from input_handler import Constraint

class LayoutEvaluator:
    """评价布局的类，实现论文中的评价函数。"""
    def __init__(self, constraints: Dict[str, Constraint], window_facades: List[Tuple[Point, Point]], entrance: Point):
        """
        初始化评价器。
        
        Args:
            constraints: 房间约束条件
            window_facades: 可开窗的外墙边界
            entrance: 入口位置
        """
        self.constraints = constraints
        self.window_facades = window_facades
        self.entrance = entrance
        
        # 评价权重，参考项目文档中的软约束权重
        self.weights = {
            'hard_constraints': 1.0,  # 硬约束权重
            'connection': 0.3,        # 联通关系权重
            'area': 0.2,              # 面积范围权重
            'orientation': 0.2,       # 朝向要求权重
            'window': 0.2,            # 开窗要求权重
            'aspect_ratio': 0.1       # 长宽比例权重
        }
    
    def evaluate(self, layout: Layout) -> float:
        """
        评价布局，返回总评分。
        
        参考论文中的评价函数 Score = (aS_n + bS_o + cS_z + dS_P) / 4
        这里重新组织为基础分(硬约束满足)和软约束分
        
        Returns:
            布局评分, 取值范围 [0, 100]
        """
        # 检查硬约束
        hard_constraint_score = self._evaluate_hard_constraints(layout)
        if hard_constraint_score < 0.5:  # 不满足硬约束
            return 0  
        
        # 计算软约束得分
        connection_score = self._evaluate_connection(layout)
        area_score = self._evaluate_area(layout)
        orientation_score = self._evaluate_orientation(layout)
        window_score = self._evaluate_window(layout)
        aspect_ratio_score = self._evaluate_aspect_ratio(layout)
        path_score = self._evaluate_path(layout)
        
        # 计算总分
        soft_constraints_score = (
            connection_score * self.weights['connection'] +
            area_score * self.weights['area'] +
            orientation_score * self.weights['orientation'] +
            window_score * self.weights['window'] +
            aspect_ratio_score * self.weights['aspect_ratio']
        )
        
        # 总分 = 基础分100(满足硬约束) + 软约束加权得分
        total_score = hard_constraint_score * 100 + soft_constraints_score * 100
        
        return total_score
    
    def _evaluate_hard_constraints(self, layout: Layout) -> float:
        """
        评价硬约束，满足返回1.0，不满足返回0.0。
        
        1. 房间无重叠 (S_o评分)
        2. 不超边界
        3. 所有需要的房间都已分配
        """
        # 1. 检查是否有房间重叠 (S_o)
        for i, room1 in enumerate(layout.rooms[:-1]):
            for room2 in layout.rooms[i+1:]:
                if room1.rectangle.intersects(room2.rectangle):
                    return 0.0
        
        # 2. 检查是否所有房间都在边界内
        for room in layout.rooms:
            if not self._is_room_inside_boundary(room, layout):
                return 0.0
        
        # 3. 检查是否包含所有需要的房间类型 (S_n)
        required_room_types = set(self.constraints.keys())
        actual_room_types = set(room.room_type for room in layout.rooms)
        if not required_room_types.issubset(actual_room_types):
            return 0.0
        
        return 1.0  # 所有硬约束都满足
    
    def _is_room_inside_boundary(self, room: Room, layout: Layout) -> bool:
        """检查房间是否在边界内。"""
        if isinstance(layout.boundary, Rectangle):
            rect = room.rectangle
            boundary = layout.boundary
            return (boundary.x <= rect.x and
                    boundary.y <= rect.y and
                    rect.x + rect.width <= boundary.x + boundary.width and
                    rect.y + rect.height <= boundary.y + boundary.height)
        else:  # Polygon
            return layout.boundary.contains_rectangle(room.rectangle)
    
    def _evaluate_connection(self, layout: Layout) -> float:
        """
        评价房间联通关系，即论文中的功能分区 S_z。
        满足所有连接关系返回1.0，否则返回满足比例。
        """
        total_connections = 0
        satisfied_connections = 0
        
        for room in layout.rooms:
            if room.room_type not in self.constraints:
                continue
                
            required_connections = self.constraints[room.room_type].connection
            if not required_connections:
                continue
                
            total_connections += len(required_connections)
            adjacent_rooms = self._get_adjacent_rooms(room, layout)
            adjacent_room_types = [r.room_type for r in adjacent_rooms]
            
            for required_type in required_connections:
                if required_type in adjacent_room_types:
                    satisfied_connections += 1
        
        if total_connections == 0:
            return 1.0  # 没有联通关系要求
        
        return satisfied_connections / total_connections
    
    def _get_adjacent_rooms(self, room: Room, layout: Layout) -> List[Room]:
        """获取与指定房间相邻的所有房间。"""
        return [other for other in layout.rooms if other != room and room.is_adjacent_to(other)]
    
    def _evaluate_area(self, layout: Layout) -> float:
        """评价房间面积。"""
        total_rooms = 0
        satisfied_rooms = 0
        
        for room in layout.rooms:
            if room.room_type not in self.constraints:
                continue
                
            area_constraint = self.constraints[room.room_type].area
            if not area_constraint:
                continue
                
            total_rooms += 1
            min_area = area_constraint.get('min', 0)
            max_area = area_constraint.get('max', float('inf'))
            
            # 转换为平方米
            room_area = room.area / 1000000  # mm² -> m²
            
            if min_area <= room_area <= max_area:
                satisfied_rooms += 1
        
        if total_rooms == 0:
            return 1.0  # 没有面积要求
        
        return satisfied_rooms / total_rooms
    
    def _evaluate_orientation(self, layout: Layout) -> float:
        """评价房间朝向。"""
        total_rooms = 0
        satisfied_rooms = 0
        
        for room in layout.rooms:
            if room.room_type not in self.constraints:
                continue
                
            orientation = self.constraints[room.room_type].orientation
            if not orientation:
                continue
                
            total_rooms += 1
            
            # 检查房间是否满足朝向要求
            if self._check_room_orientation(room, orientation, layout):
                satisfied_rooms += 1
        
        if total_rooms == 0:
            return 1.0  # 没有朝向要求
        
        return satisfied_rooms / total_rooms
    
    def _check_room_orientation(self, room: Room, orientation: str, layout: Layout) -> bool:
        """检查房间是否满足指定朝向。"""
        # 根据边界类型分别处理
        if isinstance(layout.boundary, Rectangle):
            rect = room.rectangle
            boundary = layout.boundary
            tolerance = 300  # 允许误差300mm
            
            if orientation == 'north':
                return abs(rect.y + rect.height - (boundary.y + boundary.height)) < tolerance
            elif orientation == 'east':
                return abs(rect.x + rect.width - (boundary.x + boundary.width)) < tolerance
            elif orientation == 'south':
                return abs(rect.y - boundary.y) < tolerance
            elif orientation == 'west':
                return abs(rect.x - boundary.x) < tolerance
        elif isinstance(layout.boundary, Polygon):
            # 对于多边形边界，采用坐标位置百分比判断朝向
            rect = room.rectangle
            boundary = layout.boundary
            
            # 计算房间中心点
            center_x = rect.x + rect.width / 2
            center_y = rect.y + rect.height / 2
            
            # 计算相对位置（百分比）
            rel_x = (center_x - boundary.x) / boundary.width
            rel_y = (center_y - boundary.y) / boundary.height
            
            if orientation == 'north':
                return rel_y > 0.7  # 在边界上70%的位置
            elif orientation == 'east':
                return rel_x > 0.7  # 在边界右70%的位置
            elif orientation == 'south':
                return rel_y < 0.3  # 在边界下30%的位置
            elif orientation == 'west':
                return rel_x < 0.3  # 在边界左30%的位置
            
        return False  # 不支持的朝向或边界类型
    
    def _evaluate_window(self, layout: Layout) -> float:
        """评价开窗要求。"""
        total_rooms = 0
        satisfied_rooms = 0
        
        for room in layout.rooms:
            if room.room_type not in self.constraints:
                continue
                
            window_access = self.constraints[room.room_type].window_access
            if not window_access:
                continue
                
            total_rooms += 1
            
            # 检查房间是否有边靠近可开窗的外墙
            if self._check_window_access(room, layout):
                satisfied_rooms += 1
        
        if total_rooms == 0:
            return 1.0  # 没有开窗要求
        
        return satisfied_rooms / total_rooms
    
    def _check_window_access(self, room: Room, layout: Layout) -> bool:
        """检查房间是否有边靠近可开窗的外墙。"""
        # 获取房间的四个边
        corners = room.rectangle.corners
        room_edges = [
            (corners[0], corners[1]),  # 左边
            (corners[1], corners[2]),  # 上边
            (corners[2], corners[3]),  # 右边
            (corners[3], corners[0])   # 下边
        ]
        
        # 检查房间的边是否靠近窗户外墙
        for room_edge in room_edges:
            for facade in self.window_facades:
                if self._edges_are_close(room_edge, facade):
                    return True
        
        return False
    
    def _edges_are_close(self, edge1: Tuple[Point, Point], edge2: Tuple[Point, Point]) -> bool:
        """检查两条边是否靠近，用于判断房间是否能开窗。"""
        # 计算两条边的中点距离
        mid1_x = (edge1[0].x + edge1[1].x) / 2
        mid1_y = (edge1[0].y + edge1[1].y) / 2
        mid2_x = (edge2[0].x + edge2[1].x) / 2
        mid2_y = (edge2[0].y + edge2[1].y) / 2
        
        distance = ((mid1_x - mid2_x) ** 2 + (mid1_y - mid2_y) ** 2) ** 0.5
        return distance < 300  # 假设300mm为靠近的阈值
    
    def _evaluate_aspect_ratio(self, layout: Layout) -> float:
        """评价长宽比例。"""
        total_rooms = 0
        satisfied_rooms = 0
        
        for room in layout.rooms:
            if room.room_type not in self.constraints:
                continue
                
            aspect_ratio_constraint = self.constraints[room.room_type].aspect_ratio
            if not aspect_ratio_constraint:
                continue
                
            total_rooms += 1
            min_ratio = aspect_ratio_constraint.get('min', 0)
            max_ratio = aspect_ratio_constraint.get('max', float('inf'))
            
            aspect_ratio = room.aspect_ratio
            
            if min_ratio <= aspect_ratio <= max_ratio:
                satisfied_rooms += 1
        
        if total_rooms == 0:
            return 1.0  # 没有长宽比例要求
        
        return satisfied_rooms / total_rooms
    
    def _evaluate_path(self, layout: Layout) -> float:
        """
        评价通路效果，参考论文中的S_p。
        检查是否有通路房间(path/corridor)，以及通路是否连接至主要房间。
        """
        # 找出所有通路房间
        path_rooms = [room for room in layout.rooms if room.room_type in ['path', 'corridor']]
        
        if not path_rooms:
            # 如果没有指定的通路房间，则检查是否能通过其他房间到达所有房间
            return self._evaluate_connectivity(layout)
        
        # 检查通路是否连接了所有需要连接的房间
        connected_rooms = set()
        for path_room in path_rooms:
            adjacent_rooms = self._get_adjacent_rooms(path_room, layout)
            for room in adjacent_rooms:
                connected_rooms.add(room)
        
        # 计算连通率
        total_functional_rooms = [room for room in layout.rooms if room.room_type not in ['path', 'corridor']]
        if not total_functional_rooms:
            return 1.0
        
        return min(1.0, len(connected_rooms) / len(total_functional_rooms))
    
    def _evaluate_connectivity(self, layout: Layout) -> float:
        """评价房间连通性，如果没有专门的通路，检查房间间是否通过相邻关系可达。"""
        # 构建房间连通关系图
        graph = {}
        for i, room in enumerate(layout.rooms):
            graph[i] = []
            for j, other in enumerate(layout.rooms):
                if i != j and room.is_adjacent_to(other):
                    graph[i].append(j)
        
        # 从入口附近的房间开始广度优先搜索
        entrance_room_idx = self._find_entrance_room(layout)
        if entrance_room_idx is None:
            return 0.5  # 找不到入口房间
        
        visited = {entrance_room_idx}
        queue = [entrance_room_idx]
        
        while queue:
            room_idx = queue.pop(0)
            for neighbor in graph[room_idx]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        # 计算连通率
        return len(visited) / len(layout.rooms)
    
    def _find_entrance_room(self, layout: Layout) -> Optional[int]:
        """找到距离入口最近的房间。"""
        min_distance = float('inf')
        entrance_room_idx = None
        
        for i, room in enumerate(layout.rooms):
            # 计算房间中心到入口的距离
            center_x = room.rectangle.x + room.rectangle.width / 2
            center_y = room.rectangle.y + room.rectangle.height / 2
            room_center = Point(center_x, center_y)
            
            distance = calculate_distance(room_center, self.entrance)
            
            if distance < min_distance:
                min_distance = distance
                entrance_room_idx = i
        
        return entrance_room_idx