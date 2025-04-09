# pso.py

import numpy as np
import random
from typing import List, Dict, Tuple, Callable, Union
from geometry_utils import Rectangle, Room, Layout
from input_handler import Constraint

class Particle:
    """PSO算法中的粒子，表示一种房间尺寸方案。"""
    def __init__(self, 
                 room_types: List[str], 
                 constraints: Dict[str, Constraint],
                 size_modulus: float = 300.0):
        """
        初始化粒子。
        
        Args:
            room_types: 房间类型列表
            constraints: 房间约束条件
            size_modulus: 尺寸模数，默认为300mm
        """
        self.room_types = room_types
        self.constraints = constraints
        self.size_modulus = size_modulus
        
        # 初始化位置和速度
        self.position = self._init_position()  # 房间的尺寸
        self.velocity = self._init_velocity()  # 速度
        
        self.best_position = self.position.copy()  # 个体最佳位置
        self.best_fitness = -float('inf')  # 个体最佳适应度
    
    def _init_position(self) -> np.ndarray:
        """
        初始化粒子位置（房间尺寸）。
        每个房间有两个维度：长和宽，都是尺寸模数的整数倍。
        """
        position = []
        
        for room_type in self.room_types:
            constraint = self.constraints.get(room_type)
            if not constraint:
                # 如果没有该房间类型的约束，使用默认尺寸
                width = random.randint(10, 20) # 3000mm - 6000mm
                height = random.randint(10, 20) # 3000mm - 6000mm
            else:
                # 使用约束条件生成合理的尺寸
                
                # 1. 处理面积约束
                min_area = constraint.area.get('min', 6) * 1000000 if constraint.area else 6000000  # 默认最小6平方米
                max_area = constraint.area.get('max', 30) * 1000000 if constraint.area else 30000000 # 默认最大30平方米
                
                # 2. 处理长宽比约束
                min_ratio = constraint.aspect_ratio.get('min', 0.5) if constraint.aspect_ratio else 0.5
                max_ratio = constraint.aspect_ratio.get('max', 2.0) if constraint.aspect_ratio else 2.0
                
                # 尝试生成符合约束的尺寸
                valid_dimensions = []
                # 以模数尺寸为单位尝试各种宽度
                for w_units in range(5, 30):  # 1.5m - 9m
                    width = w_units * self.size_modulus
                    
                    # 根据长宽比计算对应高度范围
                    min_height = width / max_ratio
                    max_height = width / min_ratio
                    
                    # 转换为模数单位并取整
                    min_h_units = max(5, int(np.ceil(min_height / self.size_modulus)))
                    max_h_units = min(30, int(np.floor(max_height / self.size_modulus)))
                    
                    # 检查每种可能的高度
                    for h_units in range(min_h_units, max_h_units + 1):
                        height = h_units * self.size_modulus
                        area = width * height
                        
                        # 检查面积是否在约束范围内
                        if min_area <= area <= max_area:
                            valid_dimensions.append((w_units, h_units))
                
                if valid_dimensions:
                    width, height = random.choice(valid_dimensions)
                else:
                    # 如果找不到符合所有约束的尺寸，则根据面积约束生成一个近似值
                    target_area = (min_area + max_area) / 2
                    target_ratio = (min_ratio + max_ratio) / 2
                    
                    # 计算近似的宽度和高度
                    width = int(np.sqrt(target_area * target_ratio) / self.size_modulus)
                    height = int(np.sqrt(target_area / target_ratio) / self.size_modulus)
                    
                    # 确保宽度和高度在合理范围内
                    width = max(5, min(30, width))  # 1.5m - 9m
                    height = max(5, min(30, height))  # 1.5m - 9m
            
            # 将宽度和高度添加到位置向量中
            position.extend([width, height])
        
        return np.array(position, dtype=float)
    
    def _init_velocity(self) -> np.ndarray:
        """初始化粒子速度。"""
        # 速度范围控制在较小范围，防止过大波动
        return np.random.uniform(-2, 2, size=len(self.position))
    
    def update_velocity(self, global_best_position: np.ndarray, w: float = 0.7, c1: float = 1.5, c2: float = 1.5) -> None:
        """
        更新粒子速度。
        
        Args:
            global_best_position: 全局最佳位置
            w: 惯性权重
            c1: 个体认知权重
            c2: 社会认知权重
        """
        r1 = np.random.random(size=len(self.position))
        r2 = np.random.random(size=len(self.position))
        
        cognitive_component = c1 * r1 * (self.best_position - self.position)
        social_component = c2 * r2 * (global_best_position - self.position)
        
        self.velocity = w * self.velocity + cognitive_component + social_component
        
        # 限制速度大小，防止过大波动
        max_velocity = 3.0
        self.velocity = np.clip(self.velocity, -max_velocity, max_velocity)
    
    def update_position(self) -> None:
        """更新粒子位置，并确保位置合法（符合模数要求，并满足最小尺寸）。"""
        self.position += self.velocity
        
        # 向最近的模数取整
        self.position = np.round(self.position)
        
        # 确保所有尺寸至少为5个模数（1.5m）
        min_size = 5  # 最小5个模数单位
        self.position = np.maximum(self.position, min_size)
        
        # 控制最大尺寸不超过30个模数（9m）
        max_size = 30  # 最大30个模数单位
        self.position = np.minimum(self.position, max_size)
    
    def get_room_sizes(self) -> List[Tuple[str, Rectangle]]:
        """根据当前粒子的位置，生成房间尺寸列表。"""
        room_sizes = []
        for i, room_type in enumerate(self.room_types):
            width = int(self.position[i * 2]) * self.size_modulus
            height = int(self.position[i * 2 + 1]) * self.size_modulus
            # 注意：这里只设置尺寸，坐标为0，0，由MCTS决定位置
            room_sizes.append((room_type, Rectangle(0, 0, width, height)))
        
        return room_sizes
    
    def evaluate(self, fitness_func: Callable) -> float:
        """
        计算粒子的适应度，并更新个体最佳位置。
        
        Args:
            fitness_func: 适应度函数，接受房间尺寸，返回评分
        
        Returns:
            适应度分数
        """
        room_sizes = self.get_room_sizes()
        fitness = fitness_func(room_sizes)
        
        # 更新个体最佳位置
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_position = self.position.copy()
        
        return fitness

class ParticleSwarmOptimization:
    """粒子群优化算法，用于优化房间尺寸。"""
    
    def __init__(self, 
                 room_types: List[str],
                 constraints: Dict[str, Constraint],
                 fitness_func: Callable,
                 num_particles: int = 20,
                 max_iterations: int = 50,
                 size_modulus: float = 300.0,
                 verbose: bool = True):
        """
        初始化PSO算法。
        
        Args:
            room_types: 房间类型列表
            constraints: 房间约束条件
            fitness_func: 适应度函数，接受房间尺寸列表，返回评分
            num_particles: 粒子数量
            max_iterations: 最大迭代次数
            size_modulus: 尺寸模数，默认为300mm
            verbose: 是否打印优化过程信息
        """
        self.room_types = room_types
        self.constraints = constraints
        self.fitness_func = fitness_func
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.size_modulus = size_modulus
        self.verbose = verbose
        
        # 初始化粒子群
        self.particles = [
            Particle(room_types, constraints, size_modulus) 
            for _ in range(num_particles)
        ]
        
        # 全局最佳位置和适应度
        self.global_best_position = None
        self.global_best_fitness = -float('inf')
        self.global_best_room_sizes = None
        
        # 优化参数
        self.w = 0.7  # 惯性权重
        self.c1 = 1.5  # 个体学习因子
        self.c2 = 1.5  # 社会学习因子
        
        # 收敛历史记录
        self.history = {
            'global_best': [],
            'iteration_best': [],
            'iteration_avg': []
        }
    
    def optimize(self) -> Tuple[List[Tuple[str, Rectangle]], float]:
        """
        运行PSO优化算法。
        
        Returns:
            Tuple[List[Tuple[str, Rectangle]], float]: 最佳房间尺寸列表和对应的适应度
        """
        if self.verbose:
            print("开始PSO优化房间尺寸...")
        
        # 初始化：评估所有粒子，找到全局最佳
        for particle in self.particles:
            fitness = particle.evaluate(self.fitness_func)
            
            if fitness > self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = particle.position.copy()
                self.global_best_room_sizes = particle.get_room_sizes()
        
        self.history['global_best'].append(self.global_best_fitness)
        
        # PSO迭代优化
        for iteration in range(self.max_iterations):
            iteration_best_fitness = -float('inf')
            iteration_total_fitness = 0.0
            
            # 更新所有粒子
            for particle in self.particles:
                # 更新粒子速度和位置
                particle.update_velocity(self.global_best_position, self.w, self.c1, self.c2)
                particle.update_position()
                
                # 评估适应度
                fitness = particle.evaluate(self.fitness_func)
                iteration_total_fitness += fitness
                
                # 更新迭代最佳
                if fitness > iteration_best_fitness:
                    iteration_best_fitness = fitness
                
                # 更新全局最佳
                if fitness > self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle.position.copy()
                    self.global_best_room_sizes = particle.get_room_sizes()
            
            # 记录收敛历史
            iteration_avg_fitness = iteration_total_fitness / self.num_particles
            self.history['iteration_best'].append(iteration_best_fitness)
            self.history['iteration_avg'].append(iteration_avg_fitness)
            self.history['global_best'].append(self.global_best_fitness)
            
            # 打印优化进度
            if self.verbose and (iteration + 1) % 5 == 0:
                print(f"迭代 {iteration + 1}/{self.max_iterations}: "
                      f"全局最佳={self.global_best_fitness:.2f}, "
                      f"迭代最佳={iteration_best_fitness:.2f}, "
                      f"迭代平均={iteration_avg_fitness:.2f}")
            
            # 检查早停（如果连续多次没有改进，提前结束）
            early_stop = False
            if len(self.history['global_best']) > 10:
                recent_best = self.history['global_best'][-10:]
                if len(set(recent_best)) == 1:  # 连续10次迭代没有改进
                    early_stop = True
            
            if early_stop:
                if self.verbose:
                    print(f"优化在第 {iteration+1} 次迭代提前收敛")
                break
        
        if self.verbose:
            print(f"PSO优化完成，最佳适应度={self.global_best_fitness:.2f}")
        
        return self.global_best_room_sizes, self.global_best_fitness
    
    def get_best_room_sizes(self) -> List[Tuple[str, Rectangle]]:
        """获取优化后的最佳房间尺寸。"""
        return self.global_best_room_sizes
