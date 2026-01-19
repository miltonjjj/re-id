"""
从Voronoi路径生成动作序列
输入：paths.py生成的paths_results.npz
输出：动作序列和可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os


# ============= 1. 加载路径数据 =============

def load_paths_results(episode_path):
    """
    从paths.py生成的results文件加载数据
    
    参数:
        episode_path: str, episode文件夹路径
    
    返回:
        results: dict 包含所有路径信息
    """
    npz_path = os.path.join(episode_path, 'paths_results.npz')
    
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"找不到路径结果: {npz_path}\n"
                               f"请先运行 paths.py 生成路径数据")
    
    data = np.load(npz_path, allow_pickle=True)
    
    results = {
        'robot_position': tuple(data['robot_position']),
        'start_node': tuple(data['start_node']),
        'leaf_nodes': data['leaf_nodes'].tolist(),
        'all_paths': data['all_paths'].tolist(),
        'skeleton': data['skeleton'],
        'joint_nodes': data['joint_nodes'].tolist(),
        'voronoi_graph': data['voronoi_graph'].item()
    }
    
    print(f"加载路径数据成功:")
    print(f"  路径数量: {len(results['all_paths'])}")
    print(f"  起点节点: {results['start_node']}")
    print(f"  叶节点数: {len(results['leaf_nodes'])}")
    
    return results


# ============= 2. 路径转动作序列 =============

def path_to_actions(path, initial_orientation=90.0, turn_angle=30.0, 
                    map_resolution=5.0, forward_step_meters=0.25):
    """
    将路径坐标序列转换为动作序列
    
    参数:
        path: list of (row, col) 路径坐标（地图像素）
        initial_orientation: float, 初始朝向（度），90°=向上（北）
        turn_angle: float, 每次转向角度（度），默认30
        map_resolution: float, 地图分辨率（cm/像素），默认5
        forward_step_meters: float, 每次前进距离（米），默认0.25
    
    返回:
        actions: list of int, 动作序列 [0=Stop, 1=Forward, 2=Left, 3=Right]
        action_details: list of dict, 每个动作的详细信息
    """
    if len(path) < 2:
        return [0], [{'action': 0, 'action_name': 'Stop', 'position': path[0]}]
    
    actions = []
    action_details = []
    current_orientation = initial_orientation
    
    # 计算每步前进的像素数
    forward_step_cm = forward_step_meters * 100  # 0.25m = 25cm
    pixels_per_forward = int(forward_step_cm / map_resolution)  # 25/5 = 5像素
    
    i = 0
    current_pos = path[0]
    
    while i < len(path) - 1:
        # 找到距离当前位置约pixels_per_forward的下一个目标点
        target_idx = i + 1
        cumulative_dist = 0
        
        for j in range(i + 1, len(path)):
            dist = np.sqrt((path[j][0] - path[j-1][0])**2 + 
                          (path[j][1] - path[j-1][1])**2)
            cumulative_dist += dist
            
            if cumulative_dist >= pixels_per_forward * 0.8:
                target_idx = j
                break
            elif j == len(path) - 1:
                target_idx = j
                break
        
        target_pos = path[target_idx]
        
        # 计算目标点相对于当前位置的角度
        d_row = target_pos[0] - current_pos[0]
        d_col = target_pos[1] - current_pos[1]
        
        # 计算目标方向（90°=向上）
        target_angle = np.degrees(np.arctan2(d_col, d_row))
        
        # 规范化角度到[-180, 180]
        current_orientation = ((current_orientation + 180) % 360) - 180
        target_angle = ((target_angle + 180) % 360) - 180
        
        # 计算相对角度
        relative_angle = ((current_orientation - target_angle + 180) % 360) - 180
        
        # 生成转向动作
        turn_count = 0
        while abs(relative_angle) > turn_angle / 2.0 and turn_count < 12:
            if relative_angle > 0:
                actions.append(3)  # Right
                action_details.append({
                    'action': 3,
                    'action_name': 'Right',
                    'position': current_pos,
                    'orientation': current_orientation
                })
                current_orientation -= turn_angle
            else:
                actions.append(2)  # Left
                action_details.append({
                    'action': 2,
                    'action_name': 'Left',
                    'position': current_pos,
                    'orientation': current_orientation
                })
                current_orientation += turn_angle
            
            current_orientation = ((current_orientation + 180) % 360) - 180
            relative_angle = ((current_orientation - target_angle + 180) % 360) - 180
            turn_count += 1
        
        # 朝向正确后，前进
        actions.append(1)  # Forward
        action_details.append({
            'action': 1,
            'action_name': 'Forward',
            'position': current_pos,
            'orientation': current_orientation
        })
        
        current_pos = target_pos
        i = target_idx
    
    # 添加停止动作
    actions.append(0)
    action_details.append({
        'action': 0,
        'action_name': 'Stop',
        'position': current_pos,
        'orientation': current_orientation
    })
    
    return actions, action_details


# ============= 3. 可视化 =============

def visualize_action_sequence_all_paths(skeleton, all_paths_data, start_node, 
                                        robot_position, leaf_nodes, save_path=None):
    """
    可视化所有路径的动作序列
    三个子图：骨架图、路径图、动作序列图
    
    参数:
        skeleton: (H, W) Voronoi骨架
        all_paths_data: list of dicts，包含路径和动作信息
        start_node: (x, y) 起点节点
        robot_position: (x, y) 机器人位置
        leaf_nodes: list of (x, y) 叶节点
        save_path: 保存路径（可选）
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 生成颜色映射
    n_paths = len(all_paths_data)
    colors = plt.cm.hsv(np.linspace(0, 0.9, n_paths))
    
    # ========== 子图1：骨架 + 节点 ==========
    axes[0].imshow(skeleton, cmap='gray', origin='lower')
    
    # 绘制叶节点
    if len(leaf_nodes) > 0:
        leaves_array = np.array(leaf_nodes)
        axes[0].scatter(leaves_array[:, 1], leaves_array[:, 0],
                       c='red', s=100, marker='*',
                       edgecolors='white', linewidths=1,
                       label='Leaf Nodes', zorder=8)
    
    # 绘制起点节点
    axes[0].scatter(start_node[1], start_node[0],
                   c='cyan', s=200, marker='o',
                   edgecolors='white', linewidths=2,
                   label='Start Node', zorder=9)
    
    # 绘制机器人位置
    axes[0].scatter(robot_position[1], robot_position[0],
                   c='lime', s=300, marker='s',
                   edgecolors='white', linewidths=2,
                   label='Robot', zorder=10)
    
    axes[0].set_title(f'Skeleton & Nodes')
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].axis('off')
    
    # ========== 子图2：彩色路径 ==========
    black_bg = np.zeros_like(skeleton, dtype=np.uint8)
    axes[1].imshow(black_bg, cmap='gray', vmin=0, vmax=255, origin='lower')
    
    # 绘制所有路径
    for i, path_data in enumerate(all_paths_data):
        path_coords = path_data['path_coords']
        if len(path_coords) > 0:
            path_array = np.array(path_coords)
            axes[1].plot(path_array[:, 1], path_array[:, 0],
                        color=colors[i], linewidth=3, alpha=0.9,
                        label=f'Path {i+1}')
    
    # 绘制节点
    if len(leaf_nodes) > 0:
        leaves_array = np.array(leaf_nodes)
        axes[1].scatter(leaves_array[:, 1], leaves_array[:, 0],
                       c='red', s=150, marker='*',
                       edgecolors='white', linewidths=2, zorder=8)
    
    axes[1].scatter(start_node[1], start_node[0],
                   c='cyan', s=200, marker='o',
                   edgecolors='white', linewidths=2, zorder=9)
    
    axes[1].scatter(robot_position[1], robot_position[0],
                   c='lime', s=300, marker='s',
                   edgecolors='white', linewidths=3, zorder=10)
    
    axes[1].set_title(f'Paths ({n_paths} paths)')
    axes[1].legend(loc='upper right', fontsize=8, ncol=2)
    axes[1].axis('off')
    
    # ========== 子图3：动作序列可视化 ==========
    axes[2].imshow(black_bg, cmap='gray', vmin=0, vmax=255, origin='lower')
    
    # 为每条路径绘制动作标记
    for i, path_data in enumerate(all_paths_data):
        path_coords = path_data['path_coords']
        actions = path_data['actions']
        action_details = path_data['action_details']
        
        # 绘制路径（浅色）
        if len(path_coords) > 0:
            path_array = np.array(path_coords)
            axes[2].plot(path_array[:, 1], path_array[:, 0],
                        color=colors[i], linewidth=2, alpha=0.3)
        
        # 收集不同动作的位置
        forward_pos = []
        left_pos = []
        right_pos = []
        
        for detail in action_details:
            if 'position' in detail:
                pos = detail['position']
                if detail['action'] == 1:  # Forward
                    forward_pos.append(pos)
                elif detail['action'] == 2:  # Left
                    left_pos.append(pos)
                elif detail['action'] == 3:  # Right
                    right_pos.append(pos)
        
        # 绘制前进动作：绿色三角形 ▲
        if len(forward_pos) > 0:
            forward_array = np.array(forward_pos)
            axes[2].scatter(forward_array[:, 1], forward_array[:, 0],
                           c='lime', s=60, marker='^',
                           edgecolors='darkgreen', linewidths=1,
                           alpha=0.9, zorder=5)
        
        # 绘制左转动作：蓝色左箭头 ◄
        if len(left_pos) > 0:
            left_array = np.array(left_pos)
            axes[2].scatter(left_array[:, 1], left_array[:, 0],
                           c='dodgerblue', s=60, marker='<',
                           edgecolors='darkblue', linewidths=1,
                           alpha=0.9, zorder=5)
        
        # 绘制右转动作：橙色右箭头 ►
        if len(right_pos) > 0:
            right_array = np.array(right_pos)
            axes[2].scatter(right_array[:, 1], right_array[:, 0],
                           c='orange', s=60, marker='>',
                           edgecolors='darkorange', linewidths=1,
                           alpha=0.9, zorder=5)
    
    # 绘制节点
    if len(leaf_nodes) > 0:
        leaves_array = np.array(leaf_nodes)
        axes[2].scatter(leaves_array[:, 1], leaves_array[:, 0],
                       c='red', s=150, marker='*',
                       edgecolors='white', linewidths=2, zorder=8)
    
    axes[2].scatter(start_node[1], start_node[0],
                   c='cyan', s=200, marker='o',
                   edgecolors='white', linewidths=2, zorder=9)
    
    axes[2].scatter(robot_position[1], robot_position[0],
                   c='lime', s=300, marker='s',
                   edgecolors='white', linewidths=3, zorder=10)
    
    # 创建图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='lime', 
               markersize=10, label='Forward', markeredgecolor='darkgreen'),
        Line2D([0], [0], marker='<', color='w', markerfacecolor='dodgerblue', 
               markersize=10, label='Left', markeredgecolor='darkblue'),
        Line2D([0], [0], marker='>', color='w', markerfacecolor='orange', 
               markersize=10, label='Right', markeredgecolor='darkorange')
    ]
    axes[2].legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    axes[2].set_title('Action Sequence')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n可视化已保存到: {save_path}")
    
    plt.show()


# ============= 4. 保存结果 =============

def save_action_results(all_paths_with_actions, episode_path):
    """
    保存动作序列结果
    
    参数:
        all_paths_with_actions: list of dicts，每条路径包含动作信息
        episode_path: str, episode文件夹路径
    """
    # 准备保存的数据
    results = {}
    
    for i, path_data in enumerate(all_paths_with_actions):
        path_key = f'path_{i+1}'
        
        # 计算总距离
        forward_count = path_data['actions'].count(1)
        total_distance = forward_count * 0.25  # 每次前进0.25米
        
        results[path_key] = {
            'leaf_node': path_data['leaf_node'],
            'actions': path_data['actions'],
            'action_count': {
                'forward': int(path_data['actions'].count(1)),
                'left': int(path_data['actions'].count(2)),
                'right': int(path_data['actions'].count(3)),
                'stop': int(path_data['actions'].count(0))
            },
            'total_distance_meters': float(total_distance),
            'path_length': path_data['path_length']
        }
    
    # 保存为JSON
    json_path = os.path.join(episode_path, 'actions_sequence.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  JSON格式已保存到: {json_path}")
    
    # 保存为NPZ
    npz_path = os.path.join(episode_path, 'actions_sequence.npz')
    np.savez(npz_path, **results, allow_pickle=True)
    print(f"  NPZ格式已保存到: {npz_path}")
    
    return results


# ============= 5. 主函数 =============

def main(episode_path=None):
    """
    主函数：从路径生成动作序列
    
    参数:
        episode_path: str, episode文件夹路径（可选）
    """
    
    print("="*70)
    print("动作序列生成系统（基于Voronoi路径）")
    print("="*70)
    
    # 设置默认路径
    if episode_path is None:
        episode_path = "data/voronoi_processed/Adrian_ep8"
    
    print(f"\nEpisode路径: {episode_path}")
    
    # 1. 加载路径数据
    print("\n步骤1: 加载路径数据...")
    try:
        results = load_paths_results(episode_path)
    except Exception as e:
        print(f"错误: {e}")
        return None
    
    skeleton = results['skeleton']
    all_paths = results['all_paths']
    start_node = results['start_node']
    robot_position = results['robot_position']
    leaf_nodes = results['leaf_nodes']
    
    # 2. 参数设置
    print("\n步骤2: 参数设置...")
    map_resolution = 5  # cm/像素
    forward_step = 0.25  # 米
    turn_angle = 30  # 度
    initial_orientation = 90.0  # 度（向上/北）
    
    print(f"  地图分辨率: {map_resolution} cm/像素")
    print(f"  前进步长: {forward_step}m")
    print(f"  转向角度: {turn_angle}°")
    print(f"  初始朝向: {initial_orientation}°")
    
    # 3. 为每条路径生成动作序列
    print("\n步骤3: 生成动作序列...")
    
    all_paths_with_actions = []
    
    for i, path_info in enumerate(all_paths):
        leaf_node = path_info['leaf_node']
        complete_path = path_info['complete_path']
        
        print(f"\n  路径 {i+1}/{len(all_paths)}: → 叶节点 {leaf_node}")
        print(f"    路径点数: {len(complete_path)}")
        
        # 生成动作序列
        actions, action_details = path_to_actions(
            complete_path,
            initial_orientation=initial_orientation,
            turn_angle=turn_angle,
            map_resolution=map_resolution,
            forward_step_meters=forward_step
        )
        
        print(f"    动作数: {len(actions)} (F:{actions.count(1)}, L:{actions.count(2)}, R:{actions.count(3)})")
        
        # 合并路径和动作信息
        path_with_actions = {
            'leaf_node': leaf_node,
            'path_coords': complete_path,
            'path_length': len(complete_path),
            'actions': actions,
            'action_details': action_details
        }
        
        all_paths_with_actions.append(path_with_actions)
    
    # 4. 打印摘要
    print("\n步骤4: 动作序列摘要...")
    print("="*70)
    for i, path_data in enumerate(all_paths_with_actions):
        actions = path_data['actions']
        print(f"\n路径 {i+1} → 叶节点 {path_data['leaf_node']}:")
        print(f"  总动作: {len(actions)} | "
              f"Forward: {actions.count(1)} | "
              f"Left: {actions.count(2)} | "
              f"Right: {actions.count(3)}")
        print(f"  前进距离: {actions.count(1) * 0.25:.2f}m")
    print("="*70)
    
    # 5. 保存结果
    print("\n步骤5: 保存结果...")
    save_action_results(all_paths_with_actions, episode_path)
    
    # 6. 可视化
    print("\n步骤6: 生成可视化...")
    viz_path = os.path.join(episode_path, 'actions_visualization.png')
    
    visualize_action_sequence_all_paths(
        skeleton,
        all_paths_with_actions,
        start_node,
        robot_position,
        leaf_nodes,
        save_path=viz_path
    )
    
    print("\n" + "="*70)
    print("✓ 完成！")
    print("="*70)
    print(f"\n输出文件:")
    print(f"  - actions_sequence.json")
    print(f"  - actions_sequence.npz")
    print(f"  - actions_visualization.png")
    
    return all_paths_with_actions


if __name__ == "__main__":
    import sys
    
    # 可以从命令行参数指定episode路径
    if len(sys.argv) > 1:
        episode_path = sys.argv[1]
    else:
        episode_path = "data/voronoi_processed/Adrian_ep8"
    
    results = main(episode_path)