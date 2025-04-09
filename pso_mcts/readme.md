# 住宅户型自动生成系统 (Residential Layout Generator)

基于PSO-MCTS算法（粒子群优化+蒙特卡洛树搜索）的住宅户型自动生成系统，实现了根据各种约束条件自动设计住宅平面布局的功能。

## 项目概述

本项目实现了论文《Computational design of residential units' floor layout: A heuristic algorithm》中提出的PSO-MCTS混合算法，用于自动生成满足各种约束条件的住宅户型布局。系统的主要特点：

- **多约束处理**：支持面积、朝向、开窗、相邻关系等多种约束条件
- **基于模数**：所有尺寸基于300mm的模数系统，符合建筑设计规范
- **高效搜索**：通过PSO-MCTS混合算法实现大空间高效搜索
- **自动评分**：根据多维度指标自动评价布局质量
- **可视化结果**：生成直观的户型布局图和优化过程图表

## 算法原理

该系统采用混合算法：

1. **粒子群优化(PSO)**：
   - 负责确定房间的尺寸（连续变量）
   - 每个粒子代表一种房间尺寸方案
   - 通过迭代更新粒子位置，寻找最佳尺寸组合

2. **蒙特卡洛树搜索(MCTS)**：
   - 负责确定房间的位置（离散变量）
   - 使用节点剪枝技术减少搜索空间
   - 通过选择、扩展、模拟、回溯四步法搜索最佳布局

3. **评价函数**：
   - 硬约束：房间无重叠、不超边界、所有房间分配等
   - 软约束：联通关系、面积范围、朝向要求、开窗要求、长宽比例

## 项目结构

```
room_layout_generator/
├── main.py                   # 主程序入口
├── geometry_utils.py         # 几何计算工具类
├── input_handler.py          # 输入处理模块
├── layout_evaluator.py       # 布局评价函数
├── pso.py                    # 粒子群优化算法
├── mcts.py                   # 蒙特卡洛树搜索算法
├── output_visualizer.py      # 输出可视化模块
├── constraints_example.json  # 示例约束条件
├── boundary_example.json     # 示例边界信息
└── requirements.txt          # 项目依赖
```

## 安装与使用

### 环境要求

- Python 3.7+
- 依赖包：numpy, matplotlib

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行程序

基本用法：
```bash
python main.py --constraint_file constraints_example.json --boundary_file boundary_example.json
```

高级选项：
```bash
python main.py --constraint_file constraints_example.json \
               --boundary_file boundary_example.json \
               --output_dir outputs \
               --num_particles 30 \
               --pso_iterations 80 \
               --mcts_simulations 1500 \
               --size_modulus 300.0 \
               --exploration_weight 1.5
```

## 输入文件格式

### 约束条件文件 (JSON)

```json
{
  "rooms": {
    "living_room": {
      "connection": ["dining_room", "bedroom"],
      "area": {"min": 20, "max": 30},
      "orientation": "south",
      "window_access": true,
      "aspect_ratio": {"min": 0.5, "max": 2.0}
    },
    "bedroom": {
      "connection": ["living_room"],
      "area": {"min": 12, "max": 18},
      "orientation": "east",
      "window_access": true,
      "aspect_ratio": {"min": 0.6, "max": 1.8}
    }
    // ... 更多房间
  }
}
```

### 边界信息文件 (JSON)

**矩形边界**：
```json
{
  "type": "rectangle",
  "x": 0,
  "y": 0,
  "width": 12000,
  "height": 8000,
  "window_facades": [
    {"x1": 0, "y1": 0, "x2": 12000, "y2": 0},
    {"x1": 12000, "y1": 0, "x2": 12000, "y2": 8000}
  ],
  "entrance": {"x": 6000, "y": 0}
}
```

**多边形边界**：
```json
{
  "type": "polygon",
  "vertices": [
    [0, 0],
    [12000, 0],
    [12000, 8000],
    [8000, 8000],
    [8000, 4000],
    [0, 4000]
  ],
  "window_facades": [
    {"x1": 0, "y1": 0, "x2": 12000, "y2": 0},
    {"x1": 12000, "y1": 0, "x2": 12000, "y2": 8000}
  ],
  "entrance": {"x": 6000, "y": 0}
}
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--constraint_file` | 约束条件文件路径 | `constraints_example.json` |
| `--boundary_file` | 户型边界信息文件路径 | `None`（使用默认边界） |
| `--output_dir` | 输出目录 | `outputs` |
| `--num_particles` | PSO粒子数量 | `20` |
| `--pso_iterations` | PSO最大迭代次数 | `50` |
| `--mcts_simulations` | MCTS最大模拟次数 | `1000` |
| `--size_modulus` | 尺寸模数(mm) | `300.0` |
| `--exploration_weight` | MCTS探索权重 | `1.5` |
| `--no_verbose` | 不打印详细信息 | `False` |
| `--no_show` | 不显示可视化结果 | `False` |

## 输出结果

系统会生成两类输出：

1. **户型布局图**：展示最终的房间布置、门窗位置、尺寸标注等
2. **PSO收敛曲线**：展示优化过程中的适应度变化

所有输出文件保存在指定的输出目录中（默认为`outputs`）。

## 扩展与优化方向

1. 支持更复杂的户型边界（L型、凹凸型等特殊形状）
2. 增加更多的约束条件（如空间隐私性、动线便捷性等）
3. 优化算法效率，减少计算时间
4. 添加用户交互界面，便于调整参数和实时预览
5. 支持多层楼房和特殊功能空间

## 许可证

MIT

## 参考文献

- Yan, S., & Liu, N. (2024). Computational design of residential units' floor layout: A heuristic algorithm. Journal of Building Engineering, 96, 110546.
