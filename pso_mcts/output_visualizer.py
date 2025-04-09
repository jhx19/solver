# output_visualizer.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from typing import List, Dict, Tuple, Optional, Union
from geometry_utils import Point, Rectangle, Room, Layout
from datetime import datetime
import os

class OutputVisualizer:
    """可视化布局结果的类。"""
    
    def __init__(self, 
                 output_dir: str = "outputs",
                 dpi: int = 100, 
                 figsize: Tuple[int, int] = (10, 8)):
        """
        初始化可视化器。
        
        Args:
            output_dir: 输出目录
            dpi: 图像分辨率
            figsize: 图像尺寸
        """
        self.output_dir = output_dir
        self.dpi = dpi
        self.figsize = figsize

        try:
            import matplotlib.font_manager as fm
            # 尝试使用系统字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        except:
            # 如果没有合适的中文字体，使用英文标题
            self.use_english = True
            print("注意: 未找到支持中文的字体，将使用英文标题")

        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 房间颜色映射
        self.room_colors = {
            'living_room': '#FFFFCC',  # 淡黄色
            'bedroom': '#CCFFCC',      # 淡绿色
            'master_bedroom': '#99FF99',  # 绿色
            'kitchen': '#FFCCCC',      # 淡红色
            'bathroom': '#CCCCFF',     # 淡蓝色
            'dining_room': '#FFCC99',  # 淡橙色
            'path': '#DDDDDD',         # 淡灰色
            'corridor': '#DDDDDD',     # 淡灰色
            'storage': '#CCCCCC',      # 灰色
            'balcony': '#99CCFF',      # 天蓝色
        }
        
        # 房间名称显示
        self.room_display_names = {
            'living_room': 'Living Room',
            'bedroom': 'Bedroom',
            'master_bedroom': 'Master Bedroom',
            'kitchen': 'Kitchen',
            'bathroom': 'Bathroom',
            'dining_room': 'Dining Room',
            'path': 'Path',
            'corridor': 'Corridor',
            'storage': 'Storage',
            'balcony': 'Balcony',
        }
    
    def visualize_layout(self, 
                        layout: Layout, 
                        window_facades: List[Tuple[Point, Point]] = None,
                        entrance: Point = None,
                        title: str = "Floor Layout",
                        show_dimensions: bool = True,
                        show_room_names: bool = True,
                        show_room_areas: bool = True,
                        save: bool = True,
                        show: bool = True) -> Optional[Figure]:
        """
        可视化布局，显示房间、边界、窗户和入口等。
        
        Args:
            layout: 要可视化的布局
            window_facades: 可开窗的外墙边界
            entrance: 入口位置
            title: 图像标题
            show_dimensions: 是否显示尺寸
            show_room_names: 是否显示房间名称
            show_room_areas: 是否显示房间面积
            save: 是否保存图像
            show: 是否显示图像
            
        Returns:
            Figure对象，如果不显示则为None
        """
        # 创建画布
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # 绘制边界
        boundary = layout.boundary
        if isinstance(boundary, Rectangle):
            boundary_rect = patches.Rectangle(
                (boundary.x, boundary.y), boundary.width, boundary.height,
                linewidth=2, edgecolor='black', facecolor='none', alpha=0.8
            )
            ax.add_patch(boundary_rect)
        
        # 绘制房间
        for room in layout.rooms:
            rect = room.rectangle
            room_type = room.room_type
            
            # 获取房间颜色，如果没有预定义则使用默认颜色
            color = self.room_colors.get(room_type, '#FFFFFF')
            
            # 创建房间矩形
            room_rect = patches.Rectangle(
                (rect.x, rect.y), rect.width, rect.height,
                linewidth=1.5, edgecolor='black', facecolor=color, alpha=0.7
            )
            ax.add_patch(room_rect)
            
            # 显示房间名称和面积
            if show_room_names or show_room_areas:
                text_parts = []
                
                if show_room_names:
                    display_name = self.room_display_names.get(room_type, room_type)
                    text_parts.append(display_name)
                
                if show_room_areas:
                    # 转换为平方米并保留两位小数
                    area_m2 = room.area / 1000000  # mm² -> m²
                    text_parts.append(f"{area_m2:.2f} m²")
                
                # 合并文本
                text = "\n".join(text_parts)
                
                # 放置在房间中心
                ax.text(
                    rect.x + rect.width / 2, rect.y + rect.height / 2, text,
                    ha='center', va='center', fontsize=8, color='black'
                )
            
            # 显示尺寸标注
            if show_dimensions:
                # 宽度标注（显示在矩形下方）
                width_m = rect.width / 1000  # mm -> m
                ax.text(
                    rect.x + rect.width / 2, rect.y - 150, f"{width_m:.2f}m",
                    ha='center', va='center', fontsize=6, color='blue'
                )
                
                # 高度标注（显示在矩形左侧）
                height_m = rect.height / 1000  # mm -> m
                ax.text(
                    rect.x - 150, rect.y + rect.height / 2, f"{height_m:.2f}m",
                    ha='center', va='center', fontsize=6, color='blue',
                    rotation=90
                )
        
        # 绘制窗户外墙
        if window_facades:
            for facade in window_facades:
                ax.plot(
                    [facade[0].x, facade[1].x], [facade[0].y, facade[1].y],
                    color='blue', linewidth=3, alpha=0.7
                )
        
        # 绘制入口
        if entrance:
            # 绘制入口标记
            ax.scatter(
                entrance.x, entrance.y,
                color='red', marker='s', s=100, alpha=0.8
            )
            ax.text(
                entrance.x, entrance.y - 200, "Entrance",
                ha='center', va='center', fontsize=8, color='red'
            )
        
        # 设置坐标轴比例相等
        ax.set_aspect('equal')
        
        # 设置坐标轴范围，增加一点边距
        margin = 500  # 500mm边距
        x_min = boundary.x - margin if isinstance(boundary, Rectangle) else min(p.x for p in boundary.vertices) - margin
        y_min = boundary.y - margin if isinstance(boundary, Rectangle) else min(p.y for p in boundary.vertices) - margin
        x_max = (boundary.x + boundary.width + margin) if isinstance(boundary, Rectangle) else max(p.x for p in boundary.vertices) + margin
        y_max = (boundary.y + boundary.height + margin) if isinstance(boundary, Rectangle) else max(p.y for p in boundary.vertices) + margin
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # 设置标题
        plt.title(title)
        
        # 添加坐标轴标签
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        
        # 设置网格
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # 保存图像
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_dir}/layout_{timestamp}.png"
            plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
            print(f"Layout visualization saved to {filename}")
        
        # 显示图像
        if show:
            plt.tight_layout()
            plt.show()
            return fig
        else:
            plt.close(fig)
            return None
    
    def visualize_pso_convergence(self, history: Dict[str, List[float]], save: bool = True, show: bool = True) -> None:
        """
        可视化PSO优化过程中的收敛趋势。
        
        Args:
            history: 包含'global_best', 'iteration_best', 'iteration_avg'的字典
            save: 是否保存图像
            show: 是否显示图像
        """
        plt.figure(figsize=(10, 6), dpi=self.dpi)
        
        # 确保所有数据长度一致
        min_length = min(len(history.get('global_best', [])), 
                        len(history.get('iteration_best', [])), 
                        len(history.get('iteration_avg', [])))
        
        if min_length == 0:
            print("警告: PSO历史数据为空，无法绘制收敛图")
            return
        
        iterations = range(min_length)
        
        if 'global_best' in history and len(history['global_best']) >= min_length:
            plt.plot(iterations, history['global_best'][:min_length], 'r-', label='Global Best')
        
        if 'iteration_best' in history and len(history['iteration_best']) >= min_length:
            plt.plot(iterations, history['iteration_best'][:min_length], 'g--', label='Iteration Best')
        
        if 'iteration_avg' in history and len(history['iteration_avg']) >= min_length:
            plt.plot(iterations, history['iteration_avg'][:min_length], 'b-.', label='Iteration Average')
        
        plt.title('PSO Convergence History')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness Score')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend()
        
        # 保存图像
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_dir}/pso_convergence_{timestamp}.png"
            plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
            print(f"PSO convergence plot saved to {filename}")
        
        # 显示图像
        if show:
            plt.tight_layout()
            plt.show()
        else:
            plt.close()