# input_handler.py

import json
from typing import Dict, List, Optional, Tuple, Union
from geometry_utils import Point, Rectangle, Polygon, Room, Layout

class Constraint:
    """表示一个房间的约束条件。"""
    def __init__(self, 
                 room_type: str,
                 connection: List[str] = None,
                 area: Dict[str, float] = None,
                 orientation: str = None,
                 window_access: bool = False,
                 aspect_ratio: Dict[str, float] = None):
        self.room_type = room_type
        self.connection = connection or []
        self.area = area or {}
        self.orientation = orientation
        self.window_access = window_access
        self.aspect_ratio = aspect_ratio or {}
    
    def __repr__(self):
        return f"Constraint(type={self.room_type}, connection={self.connection}, area={self.area}, orientation={self.orientation}, window_access={self.window_access}, aspect_ratio={self.aspect_ratio})"

class InputHandler:
    """处理输入数据的类。"""
    def __init__(self, constraint_file: str, boundary_info: Dict = None):
        """
        初始化输入处理器。
        
        Args:
            constraint_file: 约束条件文件路径
            boundary_info: 户型边界信息，如果为None则使用默认值
        """
        self.constraints = self._load_constraints(constraint_file)
        self.boundary = self._process_boundary(boundary_info)
        self.window_facades = self._process_window_facades(boundary_info)
        self.entrance = self._process_entrance(boundary_info)
    
    def _load_constraints(self, file_path: str) -> Dict[str, Constraint]:
        """从文件加载约束条件。"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            constraints = {}
            for room_type, room_data in data.get('rooms', {}).items():
                constraints[room_type] = Constraint(
                    room_type=room_type,
                    connection=room_data.get('connection', []),
                    area=room_data.get('area', {}),
                    orientation=room_data.get('orientation', ''),
                    window_access=room_data.get('window_access', False),
                    aspect_ratio=room_data.get('aspect_ratio', {})
                )
            
            return constraints
        except Exception as e:
            print(f"Error loading constraints: {e}")
            return {}
    
    def _process_boundary(self, boundary_info: Dict) -> Union[Rectangle, Polygon]:
        """处理户型边界信息。"""
        if not boundary_info:
            # 如果没有提供边界信息，使用默认的矩形边界
            return Rectangle(0, 0, 12000, 8000)  # 12m x 8m
        
        boundary_type = boundary_info.get('type', 'rectangle')
        if boundary_type == 'rectangle':
            return Rectangle(
                boundary_info.get('x', 0),
                boundary_info.get('y', 0),
                boundary_info.get('width', 12000),
                boundary_info.get('height', 8000)
            )
        elif boundary_type == 'polygon':
            vertices_data = boundary_info.get('vertices', [])
            vertices = [Point(x, y) for x, y in vertices_data]
            return Polygon(vertices)
        else:
            # 默认使用矩形边界
            return Rectangle(0, 0, 12000, 8000)
    
    def _process_window_facades(self, boundary_info: Dict) -> List[Tuple[Point, Point]]:
        """处理可开窗的外墙边界信息。"""
        if not boundary_info or 'window_facades' not in boundary_info:
            # 默认假设所有外墙都可以开窗
            if isinstance(self.boundary, Rectangle):
                corners = self.boundary.corners
                return [
                    (corners[0], corners[1]),  # 左侧
                    (corners[1], corners[2]),  # 上侧
                    (corners[2], corners[3]),  # 右侧
                    (corners[3], corners[0])   # 下侧
                ]
            else:
                # 如果是多边形，假设所有边都可以开窗
                vertices = self.boundary.vertices
                return [(vertices[i], vertices[(i+1) % len(vertices)]) for i in range(len(vertices))]
        
        # 使用提供的可开窗外墙信息
        window_facades = []
        for facade in boundary_info.get('window_facades', []):
            p1 = Point(facade.get('x1', 0), facade.get('y1', 0))
            p2 = Point(facade.get('x2', 0), facade.get('y2', 0))
            window_facades.append((p1, p2))
        
        return window_facades
    
    def _process_entrance(self, boundary_info: Dict) -> Point:
        """处理入口门的位置信息。"""
        if not boundary_info or 'entrance' not in boundary_info:
            # 默认入口位置在边界的左侧中点
            if isinstance(self.boundary, Rectangle):
                return Point(self.boundary.x, self.boundary.y + self.boundary.height / 2)
            else:
                # 如果是多边形，使用第一个顶点作为入口位置
                return self.boundary.vertices[0]
        
        entrance_info = boundary_info.get('entrance', {})
        return Point(entrance_info.get('x', 0), entrance_info.get('y', 0))
    
    def get_constraints(self) -> Dict[str, Constraint]:
        """获取所有约束条件。"""
        return self.constraints
    
    def get_boundary(self) -> Union[Rectangle, Polygon]:
        """获取户型边界。"""
        return self.boundary
    
    def get_window_facades(self) -> List[Tuple[Point, Point]]:
        """获取可开窗的外墙边界。"""
        return self.window_facades
    
    def get_entrance(self) -> Point:
        """获取入口门的位置。"""
        return self.entrance
    
    def get_room_types(self) -> List[str]:
        """获取所有房间类型。"""
        return list(self.constraints.keys())