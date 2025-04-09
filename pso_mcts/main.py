# main.py

import argparse
import json
import time
import os
from typing import Dict, List, Tuple, Optional

from geometry_utils import Point, Rectangle, Polygon, Room, Layout, generate_grid_points_in_boundary
from input_handler import InputHandler, Constraint
from layout_evaluator import LayoutEvaluator
from pso import ParticleSwarmOptimization
from mcts import MonteCarloTreeSearch
from output_visualizer import OutputVisualizer

def load_boundary_info(file_path: str) -> Dict:
    """加载户型边界信息。"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading boundary info: {e}")
        # 返回默认边界
        return {
            'type': 'rectangle',
            'width': 12000,  # 12m
            'height': 8000,  # 8m
            'window_facades': [
                # 默认南侧和东侧可开窗
                {'x1': 0, 'y1': 0, 'x2': 12000, 'y2': 0},  # 南边
                {'x1': 12000, 'y1': 0, 'x2': 12000, 'y2': 8000}  # 东边
            ],
            'entrance': {'x': 6000, 'y': 0}  # 默认入口在南侧中间
        }

def run_floor_layout_generation(
    constraint_file: str,
    boundary_file: Optional[str] = None,
    output_dir: str = "outputs",
    num_particles: int = 20,
    pso_iterations: int = 50,
    mcts_simulations: int = 1000,
    size_modulus: float = 300.0,
    exploration_weight: float = 1.5,
    verbose: bool = True,
    show_visualizations: bool = True
) -> Tuple[Layout, float]:
    """
    运行住宅户型自动生成算法。
    
    Args:
        constraint_file: 约束条件文件路径
        boundary_file: 户型边界信息文件路径，如果为None则使用默认边界
        output_dir: 输出目录
        num_particles: PSO粒子数量
        pso_iterations: PSO最大迭代次数
        mcts_simulations: MCTS最大模拟次数
        size_modulus: 尺寸模数，默认为300mm
        exploration_weight: MCTS探索权重
        verbose: 是否打印详细信息
        show_visualizations: 是否显示可视化结果
        
    Returns:
        生成的最佳布局和评分
    """
    start_time = time.time()
    
    if verbose:
        print("=" * 50)
        print("住宅户型自动生成 - PSO-MCTS算法")
        print("=" * 50)
    
    # 1. 加载输入数据
    if verbose:
        print("\n1. 加载输入数据...")
    
    # 加载户型边界信息
    boundary_info = None
    if boundary_file:
        boundary_info = load_boundary_info(boundary_file)
    
    # 初始化输入处理器
    input_handler = InputHandler(constraint_file, boundary_info)
    constraints = input_handler.get_constraints()
    boundary = input_handler.get_boundary()
    window_facades = input_handler.get_window_facades()
    entrance = input_handler.get_entrance()
    room_types = input_handler.get_room_types()
    
    if verbose:
        if isinstance(boundary, Rectangle):
            print(f"边界尺寸: {boundary.width/1000}m x {boundary.height/1000}m")
        else:  # Polygon
            print(f"多边形边界: {len(boundary.vertices)}个顶点")
            print(f"包围盒尺寸: {boundary.width/1000}m x {boundary.height/1000}m")
        print(f"房间数量: {len(room_types)}")
        print(f"房间类型: {', '.join(room_types)}")
    
    # 2. 初始化评价器
    if verbose:
        print("\n2. 初始化布局评价器...")
    
    evaluator = LayoutEvaluator(constraints, window_facades, entrance)
    
    # 3. 运行PSO优化房间尺寸
    if verbose:
        print("\n3. 运行PSO优化房间尺寸...")
    
    # 定义适应度函数
    def fitness_function(room_sizes):
        """PSO适应度函数：使用简化的MCTS评估布局。"""
        # 创建临时房间列表
        rooms = []
        
        # 尝试基本布局：从边界左下角开始依次摆放
        if isinstance(boundary, Rectangle):
            x, y = boundary.x, boundary.y
        else:  # Polygon
            x, y = boundary.x, boundary.y
        
        for room_type, rect in room_sizes:
            # 如果当前行放不下，换到下一行
            if x + rect.width > boundary.x + boundary.width:
                x = boundary.x
                y += max([r.rectangle.height for r in rooms if r.rectangle.y == y]) if rooms else rect.height
            
            # 如果超出边界高度，重置到初始位置
            if y + rect.height > boundary.y + boundary.height:
                return 0.0  # 无法放下所有房间，返回零分
            
            # 创建房间并添加到列表
            temp_rect = Rectangle(x, y, rect.width, rect.height)
            
            # 检查是否在多边形边界内
            if isinstance(boundary, Polygon) and not boundary.contains_rectangle(temp_rect):
                return 0.0  # 房间超出多边形边界，返回零分
            
            rooms.append(Room(room_type, temp_rect))
            
            # 更新x坐标
            x += rect.width
        
        # 创建布局并评估
        layout = Layout(rooms, boundary)
        return evaluator.evaluate(layout)
    
    # 运行PSO算法
    pso = ParticleSwarmOptimization(
        room_types=room_types,
        constraints=constraints,
        fitness_func=fitness_function,
        num_particles=num_particles,
        max_iterations=pso_iterations,
        size_modulus=size_modulus,
        verbose=verbose
    )
    
    best_room_sizes, pso_score = pso.optimize()
    
    if verbose:
        print(f"PSO优化完成，最佳适应度: {pso_score:.2f}")
        print("优化后的房间尺寸:")
        for room_type, rect in best_room_sizes:
            print(f"  {room_type}: {rect.width/1000}m x {rect.height/1000}m = {rect.area/1000000:.2f}m²")
    
    # 4. 运行MCTS确定房间位置
    if verbose:
        print("\n4. 运行MCTS确定房间位置...")
    
    mcts = MonteCarloTreeSearch(
        boundary=boundary,
        room_sizes=best_room_sizes,
        constraints=constraints,
        window_facades=window_facades,
        entrance=entrance,
        evaluator=evaluator,
        size_modulus=size_modulus,
        max_simulations=mcts_simulations,
        exploration_weight=exploration_weight
    )
    
    best_layout = mcts.search()
    
    # 评估最终布局
    final_score = evaluator.evaluate(best_layout)
    
    if verbose:
        print(f"MCTS搜索完成，最佳布局评分: {final_score:.2f}")
        print(f"总房间数: {len(best_layout.rooms)}")
        for room in best_layout.rooms:
            print(f"  {room.room_type}: ({room.rectangle.x/1000:.2f}m, {room.rectangle.y/1000:.2f}m), "
                  f"{room.rectangle.width/1000:.2f}m x {room.rectangle.height/1000:.2f}m = {room.area/1000000:.2f}m²")
    
    # 5. 可视化结果
    if verbose:
        print("\n5. 可视化布局结果...")
    
    visualizer = OutputVisualizer(output_dir=output_dir)
    
    # 可视化最终布局
    visualizer.visualize_layout(
        layout=best_layout,
        window_facades=window_facades,
        entrance=entrance,
        title="最佳住宅户型布局",
        show=show_visualizations
    )
    
    # 可视化PSO收敛历史
    visualizer.visualize_pso_convergence(
        history=pso.history,
        show=show_visualizations
    )
    
    # 计算总运行时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    if verbose:
        print("\n" + "=" * 50)
        print(f"总运行时间: {elapsed_time:.2f}秒")
        print(f"最终布局评分: {final_score:.2f}")
        print("=" * 50)
    
    return best_layout, final_score

def parse_arguments():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="住宅户型自动生成 - PSO-MCTS算法")
    
    parser.add_argument(
        "--constraint_file", 
        type=str, 
        default="input_examples/constraints_example.json",
        help="约束条件文件路径"
    )
    
    parser.add_argument(
        "--boundary_file", 
        type=str, 
        default="input_examples/polygon_boundary.json",
        help="户型边界信息文件路径"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="outputs",
        help="输出目录"
    )
    
    parser.add_argument(
        "--num_particles", 
        type=int, 
        default=20,
        help="PSO粒子数量"
    )
    
    parser.add_argument(
        "--pso_iterations", 
        type=int, 
        default=50,
        help="PSO最大迭代次数"
    )
    
    parser.add_argument(
        "--mcts_simulations", 
        type=int, 
        default=1000,
        help="MCTS最大模拟次数"
    )
    
    parser.add_argument(
        "--size_modulus", 
        type=float, 
        default=300.0,
        help="尺寸模数(mm)"
    )
    
    parser.add_argument(
        "--exploration_weight", 
        type=float, 
        default=1.5,
        help="MCTS探索权重"
    )
    
    parser.add_argument(
        "--no_verbose", 
        action="store_true",
        help="是否不打印详细信息"
    )
    
    parser.add_argument(
        "--no_show", 
        action="store_true",
        help="是否不显示可视化结果"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_arguments()
    
    # 运行算法
    run_floor_layout_generation(
        constraint_file=args.constraint_file,
        boundary_file=args.boundary_file,
        output_dir=args.output_dir,
        num_particles=args.num_particles,
        pso_iterations=args.pso_iterations,
        mcts_simulations=args.mcts_simulations,
        size_modulus=args.size_modulus,
        exploration_weight=args.exploration_weight,
        verbose=not args.no_verbose,
        show_visualizations=not args.no_show
    )