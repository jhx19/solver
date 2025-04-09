# geometry_utils.py

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Set

class Point:
    """表示二维空间中的一个点。"""
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f"Point({self.x}, {self.y})"

class Rectangle:
    """表示一个矩形，用左下角坐标和宽高表示。"""
    def __init__(self, x: float, y: float, width: float, height: float):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    @property
    def area(self) -> float:
        """计算矩形的面积。"""
        return self.width * self.height
    
    @property
    def aspect_ratio(self) -> float:
        """计算矩形的长宽比。"""
        return self.width / self.height if self.height != 0 else float('inf')
    
    @property
    def corners(self) -> List[Point]:
        """返回矩形的四个顶点，顺序为左下、左上、右上、右下。"""
        return [
            Point(self.x, self.y),  # 左下
            Point(self.x, self.y + self.height),  # 左上
            Point(self.x + self.width, self.y + self.height),  # 右上
            Point(self.x + self.width, self.y)  # 右下
        ]
    
    def contains_point(self, point: Point) -> bool:
        """判断点是否在矩形内部。"""
        return (self.x <= point.x <= self.x + self.width and
                self.y <= point.y <= self.y + self.height)
    
    def intersects(self, other: 'Rectangle') -> bool:
        """判断两个矩形是否相交。"""
        return not (self.x + self.width <= other.x or
                    other.x + other.width <= self.x or
                    self.y + self.height <= other.y or
                    other.y + other.height <= self.y)
    
    def intersection_area(self, other: 'Rectangle') -> float:
        """计算两个矩形的相交面积。"""
        if not self.intersects(other):
            return 0.0
        
        x_overlap = min(self.x + self.width, other.x + other.width) - max(self.x, other.x)
        y_overlap = min(self.y + self.height, other.y + other.height) - max(self.y, other.y)
        
        return x_overlap * y_overlap
    
    def __repr__(self):
        return f"Rectangle(x={self.x}, y={self.y}, width={self.width}, height={self.height})"

class Polygon:
    """表示一个多边形，用顶点列表表示。"""
    def __init__(self, vertices: List[Point]):
        self.vertices = vertices
    
    @property
    def area(self) -> float:
        """计算多边形的面积（使用叉乘法）。"""
        n = len(self.vertices)
        area = 0.0
        
        for i in range(n):
            j = (i + 1) % n
            area += self.vertices[i].x * self.vertices[j].y
            area -= self.vertices[j].x * self.vertices[i].y
        
        return abs(area) / 2.0
    
    def contains_point(self, point: Point) -> bool:
        """判断点是否在多边形内部（使用射线法）。"""
        n = len(self.vertices)
        inside = False
        
        p1x, p1y = self.vertices[0].x, self.vertices[0].y
        for i in range(n + 1):
            p2x, p2y = self.vertices[i % n].x, self.vertices[i % n].y
            
            if point.y > min(p1y, p2y) and point.y <= max(p1y, p2y) and point.x <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (point.y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                
                if p1x == p2x or point.x <= xinters:
                    inside = not inside
            
            p1x, p1y = p2x, p2y
        
        return inside
    
    def contains_rectangle(self, rect: Rectangle) -> bool:
        """判断矩形是否完全在多边形内部。"""
        return all(self.contains_point(corner) for corner in rect.corners)
    
    def __repr__(self):
        return f"Polygon(vertices={self.vertices})"

class Room:
    """表示一个房间。"""
    def __init__(self, room_type: str, rectangle: Rectangle):
        self.room_type = room_type
        self.rectangle = rectangle
    
    @property
    def area(self) -> float:
        """获取房间的面积。"""
        return self.rectangle.area
    
    @property
    def aspect_ratio(self) -> float:
        """获取房间的长宽比。"""
        return self.rectangle.aspect_ratio
    
    def is_adjacent_to(self, other: 'Room') -> bool:
        """判断该房间是否与另一个房间相邻。"""
        return get_adjacent_rectangles(self.rectangle, other.rectangle)
    
    def __repr__(self):
        return f"Room(type={self.room_type}, rect={self.rectangle})"

class Layout:
    """表示一个完整的户型布局，包含多个房间。"""
    def __init__(self, rooms: List[Room], boundary: Union[Rectangle, Polygon]):
        self.rooms = rooms
        self.boundary = boundary
    
    def add_room(self, room: Room) -> bool:
        """添加一个房间到布局中，如果与已有房间重叠或超出边界则返回False。"""
        # 检查是否在边界内
        if not is_inside_boundary(room.rectangle, self.boundary):
            return False
        
        # 检查是否与其他房间重叠
        for existing_room in self.rooms:
            if check_overlapping(room.rectangle, existing_room.rectangle):
                return False
        
        self.rooms.append(room)
        return True
    
    def get_room_by_type(self, room_type: str) -> List[Room]:
        """获取指定类型的所有房间。"""
        return [room for room in self.rooms if room.room_type == room_type]
    
    def get_adjacent_rooms(self, room: Room) -> List[Room]:
        """获取与指定房间相邻的所有房间。"""
        return [other for other in self.rooms if room.is_adjacent_to(other)]
    
    def total_area(self) -> float:
        """计算所有房间的总面积。"""
        return sum(room.area for room in self.rooms)
    
    def __repr__(self):
        return f"Layout(rooms={len(self.rooms)})"

def check_overlapping(rect1: Rectangle, rect2: Rectangle) -> bool:
    """检查两个矩形是否重叠。"""
    return rect1.intersects(rect2)

def calculate_overlapping_area(rect1: Rectangle, rect2: Rectangle) -> float:
    """计算两个矩形的重叠面积。"""
    return rect1.intersection_area(rect2)

def is_inside_boundary(rect: Rectangle, boundary: Union[Rectangle, Polygon]) -> bool:
    """检查矩形是否在边界内。"""
    if isinstance(boundary, Rectangle):
        return (boundary.x <= rect.x and
                boundary.y <= rect.y and
                rect.x + rect.width <= boundary.x + boundary.width and
                rect.y + rect.height <= boundary.y + boundary.height)
    elif isinstance(boundary, Polygon):
        return boundary.contains_rectangle(rect)
    else:
        raise TypeError("Boundary must be either Rectangle or Polygon")

def get_adjacent_rectangles(rect1: Rectangle, rect2: Rectangle) -> bool:
    """检查两个矩形是否相邻。"""
    # 两个矩形相邻，意味着它们有共同的边
    # 我们可以通过检查一个矩形的边是否与另一个矩形的边重叠来判断
    
    # 检查水平方向
    horizontal_adjacent = (
        (abs(rect1.x - (rect2.x + rect2.width)) < 1e-6 or abs(rect2.x - (rect1.x + rect1.width)) < 1e-6) and
        not (rect1.y + rect1.height <= rect2.y or rect2.y + rect2.height <= rect1.y)
    )
    
    # 检查垂直方向
    vertical_adjacent = (
        (abs(rect1.y - (rect2.y + rect2.height)) < 1e-6 or abs(rect2.y - (rect1.y + rect1.height)) < 1e-6) and
        not (rect1.x + rect1.width <= rect2.x or rect2.x + rect2.width <= rect1.x)
    )
    
    return horizontal_adjacent or vertical_adjacent

def calculate_distance(p1: Point, p2: Point) -> float:
    """计算两点之间的欧几里得距离。"""
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)
